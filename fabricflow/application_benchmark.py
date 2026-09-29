"""Held-out distributed-training gate for FabricFlow communication policies.

This benchmark intentionally uses the same gradient packing and bucket schedule
for every candidate.  Only the collective selected for each bucket changes, so
the end-to-end step-time comparison is auditable.
"""

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
import time
from typing import Any, Iterable

from .benchmark import (
    _completion_timings,
    _gather_hardware,
    _gather_timings,
    _percentile,
    _rank_timing_matrix,
    _validate_node_distribution,
    _version_string,
)
from .cost_model import Strategy
from .runtime import initialize_nccl
from .sizes import parse_size


POLICY_NAME = "payload_policy"


def select_bucket_strategy(
    requested: str,
    bucket_bytes: int,
    *,
    policy_min_bytes: int,
    policy_max_bytes: int,
) -> Strategy:
    """Resolve the measured crossover policy for one gradient bucket."""

    if requested != POLICY_NAME:
        return Strategy(requested)
    if policy_min_bytes <= bucket_bytes <= policy_max_bytes:
        return Strategy.HIERARCHICAL_RAIL
    return Strategy.NCCL_ALLREDUCE


def bucket_ranges(
    element_count: int,
    element_size: int,
    bucket_bytes: int,
) -> list[tuple[int, int]]:
    if element_count < 1 or element_size < 1 or bucket_bytes < element_size:
        raise ValueError("invalid gradient bucket dimensions")
    elements_per_bucket = max(1, bucket_bytes // element_size)
    return [
        (start, min(element_count, start + elements_per_bucket))
        for start in range(0, element_count, elements_per_bucket)
    ]


def _append_jsonl(path: str | Path, row: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _make_batches(
    torch: Any,
    *,
    rank: int,
    count: int,
    batch_size: int,
    width: int,
) -> list[tuple[Any, Any]]:
    generator = torch.Generator(device="cuda")
    generator.manual_seed(71_000 + rank)
    return [
        (
            torch.randn(
                batch_size,
                width,
                device="cuda",
                dtype=torch.float32,
                generator=generator,
            ),
            torch.randn(
                batch_size,
                width,
                device="cuda",
                dtype=torch.float32,
                generator=generator,
            ),
        )
        for _ in range(count)
    ]


def _reduce_gradients(
    torch: Any,
    runtime: Any,
    parameters: Iterable[Any],
    *,
    requested_strategy: str,
    bucket_bytes: int,
    policy_min_bytes: int,
    policy_max_bytes: int,
) -> Counter[str]:
    gradients = [parameter.grad for parameter in parameters]
    if not gradients or any(gradient is None for gradient in gradients):
        raise RuntimeError("every model parameter must produce a gradient")
    flat = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
    ranges = bucket_ranges(flat.numel(), flat.element_size(), bucket_bytes)
    selections: Counter[str] = Counter()
    for start, end in ranges:
        bucket = flat[start:end]
        selected = select_bucket_strategy(
            requested_strategy,
            bucket.numel() * bucket.element_size(),
            policy_min_bytes=policy_min_bytes,
            policy_max_bytes=policy_max_bytes,
        )
        runtime.all_reduce(bucket, selected)
        bucket.div_(runtime.layout.world_size)
        selections[selected.value] += 1
    offset = 0
    for gradient in gradients:
        count = gradient.numel()
        gradient.copy_(flat[offset : offset + count].view_as(gradient))
        offset += count
    return selections


def _parameter_spread(torch: Any, dist: Any, model: Any) -> float:
    flat = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
    reference = flat.clone()
    dist.broadcast(reference, src=0)
    local = torch.max(torch.abs(flat - reference)).to(torch.float64)
    dist.all_reduce(local, op=dist.ReduceOp.MAX)
    return float(local.item())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FabricFlow application promotion gate")
    parser.add_argument(
        "--strategy",
        choices=[
            Strategy.NCCL_ALLREDUCE.value,
            Strategy.HIERARCHICAL_LEADER.value,
            Strategy.HIERARCHICAL_RAIL.value,
            POLICY_NAME,
        ],
        required=True,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--bucket-bytes", default="1MiB")
    parser.add_argument("--policy-min-bytes", default="256KiB")
    parser.add_argument("--policy-max-bytes", default="4MiB")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.strategy == POLICY_NAME
        and os.environ.get("FABRICFLOW_EXPERIMENT_SCOPE") == "multi_node_rdma"
    ):
        raise SystemExit(
            "the historical payload policy is not validated on H200/RDMA; "
            "use an explicit strategy"
        )
    if args.warmup < 1 or args.steps < 3:
        raise SystemExit("warmup must be >=1 and steps must be >=3")
    if args.batch_size < 1 or args.width < 2 or args.depth < 1:
        raise SystemExit("invalid model dimensions")
    bucket_bytes = parse_size(args.bucket_bytes)
    policy_min_bytes = parse_size(args.policy_min_bytes)
    policy_max_bytes = parse_size(args.policy_max_bytes)
    if not policy_min_bytes <= policy_max_bytes:
        raise SystemExit("policy minimum must not exceed policy maximum")

    torch, dist, layout, runtime = initialize_nccl()
    try:
        torch.manual_seed(19_991)
        layers: list[Any] = []
        for _ in range(args.depth):
            layers.extend([torch.nn.Linear(args.width, args.width), torch.nn.GELU()])
        model = torch.nn.Sequential(*layers).cuda()
        ddp = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[layout.local_rank],
            output_device=layout.local_rank,
            broadcast_buffers=False,
        )
        optimizer = torch.optim.SGD(ddp.parameters(), lr=1e-4)
        loss_function = torch.nn.MSELoss()
        batches = _make_batches(
            torch,
            rank=layout.rank,
            count=args.warmup + args.steps,
            batch_size=args.batch_size,
            width=args.width,
        )

        hardware = _gather_hardware(torch, dist, layout.world_size)
        gpu_names = sorted({str(item["gpu_name"]) for item in hardware})
        if (
            os.environ.get("FABRICFLOW_REQUIRE_HOMOGENEOUS", "0") == "1"
            and len(gpu_names) != 1
        ):
            raise RuntimeError(f"heterogeneous GPU models are not allowed: {gpu_names}")
        node_ids, node_rank_counts = _validate_node_distribution(
            hardware,
            expected_nodes=int(
                os.environ.get("FABRICFLOW_EXPECTED_NODES", str(layout.nodes))
            ),
            expected_gpus_per_node=layout.local_size,
        )

        timings_us: list[float] = []
        selection_totals: Counter[str] = Counter()
        initial_loss: float | None = None
        final_loss = math.nan
        total_steps = args.warmup + args.steps
        for step, (inputs, targets) in enumerate(batches):
            dist.barrier()
            torch.cuda.synchronize()
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with ddp.no_sync():
                prediction = ddp(inputs)
                loss = loss_function(prediction, targets)
                loss.backward()
            selections = _reduce_gradients(
                torch,
                runtime,
                ddp.parameters(),
                requested_strategy=args.strategy,
                bucket_bytes=bucket_bytes,
                policy_min_bytes=policy_min_bytes,
                policy_max_bytes=policy_max_bytes,
            )
            optimizer.step()
            torch.cuda.synchronize()
            elapsed_us = (time.perf_counter() - started) * 1_000_000
            loss_value = float(loss.detach().item())
            if step == 0:
                initial_loss = loss_value
            final_loss = loss_value
            if step >= args.warmup:
                timings_us.append(elapsed_us)
                selection_totals.update(selections)
        if len(timings_us) != args.steps or total_steps != len(batches):
            raise RuntimeError("application timing accounting failed")

        averaged_loss = torch.tensor(final_loss, device="cuda", dtype=torch.float64)
        dist.all_reduce(averaged_loss, op=dist.ReduceOp.SUM)
        averaged_loss.div_(layout.world_size)
        max_parameter_spread = _parameter_spread(torch, dist, ddp.module)
        all_timings = _gather_timings(
            torch, dist, timings_us, layout.world_size
        )
        if layout.rank == 0:
            rank_timings = _rank_timing_matrix(
                all_timings, layout.world_size, args.steps
            )
            completion_timings = _completion_timings(rank_timings)
            rank_medians = [statistics.median(values) for values in rank_timings]
            p50 = _percentile(completion_timings, 50)
            p95 = _percentile(completion_timings, 95)
            correct = (
                math.isfinite(float(averaged_loss.item()))
                and max_parameter_spread <= 1e-5
            )
            row: dict[str, object] = {
                "schema_version": 2,
                "timing_semantics": "per_step_max_rank_wall_duration_us",
                "rank_step_timings_us": rank_timings,
                "completion_step_timings_us": completion_timings,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "provenance": "hardware_application_measurement",
                "provider": os.environ.get("FABRICFLOW_PROVIDER"),
                "experiment_scope": os.environ.get("FABRICFLOW_EXPERIMENT_SCOPE"),
                "network_transport_requested": os.environ.get(
                    "FABRICFLOW_NETWORK_TRANSPORT"
                ),
                "run_id": os.environ.get("FABRICFLOW_RUN_ID"),
                "repeat_id": os.environ.get("FABRICFLOW_REPEAT_ID"),
                "cluster_id": os.environ.get("FABRICFLOW_CLUSTER_ID"),
                "strategy": args.strategy,
                "world_size": layout.world_size,
                "nodes": layout.nodes,
                "gpus_per_node": layout.local_size,
                "warmup_steps": args.warmup,
                "measured_steps": args.steps,
                "global_batch_size": args.batch_size * layout.world_size,
                "model_width": args.width,
                "model_depth": args.depth,
                "parameter_count": sum(p.numel() for p in ddp.parameters()),
                "gradient_bytes": sum(
                    p.numel() * p.element_size() for p in ddp.parameters()
                ),
                "bucket_bytes": bucket_bytes,
                "policy_min_bytes": policy_min_bytes,
                "policy_max_bytes": policy_max_bytes,
                "collective_bucket_counts": dict(sorted(selection_totals.items())),
                "step_p50_us": p50,
                "step_p95_us": p95,
                "step_max_us": max(completion_timings),
                "rank_median_spread_us": max(rank_medians) - min(rank_medians),
                "samples_per_second_p50": (
                    args.batch_size * layout.world_size / (p50 / 1_000_000)
                ),
                "initial_local_loss": initial_loss,
                "final_global_loss": float(averaged_loss.item()),
                "max_parameter_spread": max_parameter_spread,
                "correct": correct,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "nccl_version": _version_string(torch.cuda.nccl.version()),
                "gpu_names": gpu_names,
                "node_ids": node_ids,
                "node_rank_counts": node_rank_counts,
                "hostname": socket.gethostname(),
            }
            _append_jsonl(args.output, row)
            print(json.dumps(row, indent=2), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
