"""Hardware benchmark entry point intended for ``torchrun``."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import statistics
from typing import Any

from .cost_model import Strategy
from .planner import Planner
from .runtime import initialize_nccl
from .sizes import parse_size
from .topology import ClusterTopology


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty list")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile / 100 * len(ordered)) - 1)
    return ordered[index]


def _rank_timing_matrix(
    flat: list[float], world: int, iterations: int
) -> list[list[float]]:
    """Rebuild rank-major samples returned by the distributed all-gather."""
    if world < 1 or iterations < 1 or len(flat) != world * iterations:
        raise ValueError("rank timing matrix has an invalid shape")
    if any(not math.isfinite(value) or value < 0 for value in flat):
        raise ValueError("rank timing samples must be finite and nonnegative")
    return [flat[rank * iterations : (rank + 1) * iterations] for rank in range(world)]


def _completion_timings(rank_timings: list[list[float]]) -> list[float]:
    """Approximate each collective's completion by its slowest rank duration."""
    if not rank_timings or not rank_timings[0]:
        raise ValueError("rank timings must be nonempty")
    if any(len(row) != len(rank_timings[0]) for row in rank_timings):
        raise ValueError("all ranks must report the same iteration count")
    return [max(samples) for samples in zip(*rank_timings)]


def _expected_sum(world: int, iteration: int) -> float:
    if world < 1 or iteration < 0:
        raise ValueError("invalid world size or iteration")
    return world * (world + 1) / 2 + world * (iteration % 7)


def _prediction_for(
    topology: ClusterTopology | None,
    strategy: Strategy,
    message_bytes: int,
) -> float | None:
    if topology is None:
        return None
    plan = Planner(topology).plan(message_bytes, runtime_only=True)
    match = next(item for item in plan.candidates if item.strategy == strategy)
    return match.latency_us


def _run_one(
    torch: Any,
    dist: Any,
    runtime: Any,
    *,
    strategy: Strategy,
    message_bytes: int,
    warmup: int,
    iterations: int,
) -> tuple[list[float], float]:
    element_size = torch.empty((), dtype=torch.float32).element_size()
    element_count = math.ceil(message_bytes / element_size)
    tensor = torch.empty(element_count, device="cuda", dtype=torch.float32)

    for _ in range(warmup):
        tensor.fill_(runtime.layout.rank + 1)
        runtime.all_reduce(tensor, strategy)
    torch.cuda.synchronize()

    timings: list[float] = []
    local_error = torch.zeros((), device="cuda", dtype=torch.float64)
    sample_stride = max(1, element_count // 4096)
    for iteration in range(iterations):
        dist.barrier()
        tensor.fill_(runtime.layout.rank + 1 + iteration % 7)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        runtime.all_reduce(tensor, strategy)
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end) * 1000))
        expected = _expected_sum(runtime.layout.world_size, iteration)
        # Sample every timed iteration; a full scan after the loop checks the
        # final result without repeatedly reading a 1 GiB tensor.
        local_error = torch.maximum(
            local_error,
            torch.max(torch.abs(tensor[::sample_stride] - expected)).to(torch.float64),
        )

    final_expected = _expected_sum(runtime.layout.world_size, iterations - 1)
    local_error = torch.maximum(
        local_error, torch.max(torch.abs(tensor - final_expected)).to(torch.float64)
    )
    dist.all_reduce(local_error, op=dist.ReduceOp.MAX)
    return timings, float(local_error.item())


def _gather_timings(torch: Any, dist: Any, values: list[float], world: int) -> list[float]:
    local = torch.tensor(values, device="cuda", dtype=torch.float64)
    gathered = torch.empty(world * len(values), device="cuda", dtype=torch.float64)
    gather = getattr(dist, "all_gather_single", None)
    if gather is None:  # PyTorch < 2.13 compatibility.
        gather = dist.all_gather_into_tensor
    gather(gathered, local)
    return [float(value) for value in gathered.cpu().tolist()]


def _gather_hardware(torch: Any, dist: Any, world: int) -> list[dict[str, object]]:
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    local: dict[str, object] = {
        "hostname": socket.gethostname(),
        # Some container platforms intentionally reuse one hostname on every
        # clustered node. Prefer the provider-supplied stable node identity.
        "node_id": os.environ.get("FABRICFLOW_NODE_ID", socket.gethostname()),
        "node_rank": os.environ.get("FABRICFLOW_NODE_RANK"),
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "gpu_memory_bytes": int(properties.total_memory),
    }
    gathered: list[dict[str, object] | None] = [None] * world
    dist.all_gather_object(gathered, local)
    return [item for item in gathered if item is not None]


def _validate_node_distribution(
    hardware: list[dict[str, object]],
    *,
    expected_nodes: int,
    expected_gpus_per_node: int,
) -> tuple[list[str], dict[str, int]]:
    node_ids = [str(item["node_id"]) for item in hardware]
    counts = dict(sorted(Counter(node_ids).items()))
    if len(counts) != expected_nodes:
        raise RuntimeError(
            f"expected {expected_nodes} distinct nodes, observed {counts}"
        )
    unexpected = {
        node_id: count
        for node_id, count in counts.items()
        if count != expected_gpus_per_node
    }
    if unexpected:
        raise RuntimeError(
            "unexpected GPU-rank distribution across nodes: "
            f"expected {expected_gpus_per_node} per node, observed {counts}"
        )
    return sorted(counts), counts


def _version_string(value: object) -> str:
    if isinstance(value, (tuple, list)):
        return ".".join(str(item) for item in value)
    return str(value)


def _append_jsonl(path: str | Path, row: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FabricFlow NCCL benchmark")
    parser.add_argument(
        "--strategy",
        choices=[
            Strategy.NCCL_ALLREDUCE.value,
            Strategy.HIERARCHICAL_LEADER.value,
            Strategy.HIERARCHICAL_RAIL.value,
        ],
        required=True,
    )
    parser.add_argument("--sizes", required=True, help="comma-separated, e.g. 1MiB,64MiB")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--topology", help="optional model config for predicted_us")
    parser.add_argument("--output", required=True, help="rank 0 JSONL output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup < 1 or args.iterations < 2:
        raise SystemExit("warmup must be >=1 and iterations must be >=2")
    strategy = Strategy(args.strategy)
    sizes = [parse_size(item) for item in args.sizes.split(",")]
    topology = ClusterTopology.load(args.topology) if args.topology else None
    torch, dist, layout, runtime = initialize_nccl()
    try:
        if topology and topology.world_size != layout.world_size:
            raise RuntimeError(
                f"topology world_size={topology.world_size} but torchrun has "
                f"WORLD_SIZE={layout.world_size}"
            )
        hardware = _gather_hardware(torch, dist, layout.world_size)
        gpu_names = sorted({str(item["gpu_name"]) for item in hardware})
        hostnames = sorted({str(item["hostname"]) for item in hardware})
        if (
            os.environ.get("FABRICFLOW_REQUIRE_HOMOGENEOUS", "0") == "1"
            and len(gpu_names) != 1
        ):
            raise RuntimeError(f"heterogeneous GPU models are not allowed: {gpu_names}")
        expected_nodes = int(
            os.environ.get("FABRICFLOW_EXPECTED_NODES", str(layout.nodes))
        )
        node_ids, node_rank_counts = _validate_node_distribution(
            hardware,
            expected_nodes=expected_nodes,
            expected_gpus_per_node=layout.local_size,
        )
        for message_bytes in sizes:
            timings, max_error = _run_one(
                torch,
                dist,
                runtime,
                strategy=strategy,
                message_bytes=message_bytes,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            all_timings = _gather_timings(
                torch, dist, timings, layout.world_size
            )
            if layout.rank == 0:
                rank_timings = _rank_timing_matrix(
                    all_timings, layout.world_size, args.iterations
                )
                completion_timings = _completion_timings(rank_timings)
                rank_medians = [statistics.median(values) for values in rank_timings]
                measured_p50_us = _percentile(completion_timings, 50)
                measured_p95_us = _percentile(completion_timings, 95)
                algorithmic_bandwidth_gb_per_s = (
                    message_bytes / (measured_p50_us * 1000)
                )
                normalized_bus_bandwidth_gb_per_s = (
                    algorithmic_bandwidth_gb_per_s
                    * 2
                    * (layout.world_size - 1)
                    / layout.world_size
                )
                row: dict[str, object] = {
                    "schema_version": 2,
                    "timing_semantics": "per_iteration_max_rank_cuda_duration_us",
                    "rank_timings_us": rank_timings,
                    "completion_timings_us": completion_timings,
                    "correctness_semantics": "sampled_each_iteration_full_final_variable_rank_pattern",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "provenance": "hardware_measurement",
                    "provider": os.environ.get("FABRICFLOW_PROVIDER"),
                    "experiment_scope": os.environ.get(
                        "FABRICFLOW_EXPERIMENT_SCOPE"
                    ),
                    "network_transport_requested": os.environ.get(
                        "FABRICFLOW_NETWORK_TRANSPORT"
                    ),
                    "nccl_net": os.environ.get("NCCL_NET"),
                    "nccl_ib_disable": os.environ.get("NCCL_IB_DISABLE"),
                    "nccl_socket_family": os.environ.get("NCCL_SOCKET_FAMILY"),
                    "nccl_socket_ifname": os.environ.get("NCCL_SOCKET_IFNAME"),
                    "run_id": os.environ.get("FABRICFLOW_RUN_ID"),
                    "repeat_id": os.environ.get("FABRICFLOW_REPEAT_ID"),
                    "cluster_id": os.environ.get("FABRICFLOW_CLUSTER_ID"),
                    "strategy": strategy.value,
                    "message_bytes": message_bytes,
                    "world_size": layout.world_size,
                    "nodes": layout.nodes,
                    "gpus_per_node": layout.local_size,
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                    "measured_p50_us": measured_p50_us,
                    "measured_p95_us": measured_p95_us,
                    "measured_max_us": max(completion_timings),
                    "algorithmic_bandwidth_gb_per_s": (
                        algorithmic_bandwidth_gb_per_s
                    ),
                    "normalized_bus_bandwidth_gb_per_s": (
                        normalized_bus_bandwidth_gb_per_s
                    ),
                    "rank_median_spread_us": max(rank_medians) - min(rank_medians),
                    "predicted_us": _prediction_for(
                        topology, strategy, message_bytes
                    ),
                    "max_abs_error": max_error,
                    "correct": max_error <= 1e-5,
                    "torch_version": torch.__version__,
                    "cuda_version": torch.version.cuda,
                    "nccl_version": _version_string(torch.cuda.nccl.version()),
                    "gpu_names": gpu_names,
                    "homogeneous_gpu_names": len(gpu_names) == 1,
                    "hostnames": hostnames,
                    "node_ids": node_ids,
                    "node_rank_counts": node_rank_counts,
                    "gpu_compute_capabilities": sorted(
                        {
                            ".".join(str(value) for value in item["gpu_capability"])
                            for item in hardware
                        }
                    ),
                    "gpu_memory_bytes": sorted(
                        {int(item["gpu_memory_bytes"]) for item in hardware}
                    ),
                    "rank0_host": socket.gethostname(),
                    "job_id": os.environ.get("SLURM_JOB_ID"),
                }
                _append_jsonl(args.output, row)
                print(json.dumps(row, indent=2), flush=True)
            dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
