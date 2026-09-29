"""Adversarial multi-node SUM validation, separate from timed benchmarks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from .benchmark import _gather_hardware, _validate_node_distribution
from .cost_model import Strategy
from .runtime import initialize_nccl


DEFAULT_COUNTS = (1, 7, 9, 1025, 262147)
DTYPES = ("float32", "float16", "bfloat16")
STRATEGIES = (
    Strategy.NCCL_ALLREDUCE,
    Strategy.HIERARCHICAL_LEADER,
    Strategy.HIERARCHICAL_RAIL,
)


def input_seed(case_index: int, rank: int) -> int:
    if case_index < 0 or rank < 0:
        raise ValueError("case index and rank must be nonnegative")
    return 91827 + case_index * 101 + rank


def parse_counts(raw: str) -> tuple[int, ...]:
    counts = tuple(int(item.strip()) for item in raw.split(","))
    if not counts or any(count < 1 or count > 1_000_000 for count in counts):
        raise ValueError("element counts must be in [1, 1000000]")
    if len(set(counts)) != len(counts):
        raise ValueError("element counts must be unique")
    return counts


def _case(
    torch: Any,
    dist: Any,
    runtime: Any,
    *,
    strategy: Strategy,
    dtype_name: str,
    elements: int,
    repetition: int,
    case_index: int,
) -> dict[str, object]:
    dtype = getattr(torch, dtype_name)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(input_seed(case_index, runtime.layout.rank))
    # Small signed integers sum exactly in FP32, FP16, and BF16 at 16 ranks.
    source = torch.randint(
        -4, 5, (elements,), device="cuda", dtype=torch.int32, generator=generator
    ).to(dtype)
    reference = source.clone()
    dist.all_reduce(reference, op=dist.ReduceOp.SUM)
    actual = source.clone()
    runtime.all_reduce(actual, strategy)
    local_mismatch = torch.tensor(
        int(not torch.equal(actual, reference)), device="cuda", dtype=torch.int32
    )
    dist.all_reduce(local_mismatch, op=dist.ReduceOp.MAX)
    local_error = torch.max(torch.abs(actual.float() - reference.float()))
    dist.all_reduce(local_error, op=dist.ReduceOp.MAX)
    return {
        "strategy": strategy.value,
        "dtype": dtype_name,
        "elements": elements,
        "repetition": repetition,
        "correct": not bool(local_mismatch.item()),
        "max_abs_error": float(local_error.item()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--counts", default=",".join(map(str, DEFAULT_COUNTS)))
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args(argv)
    counts = parse_counts(args.counts)
    if not 1 <= args.repetitions <= 10:
        raise SystemExit("repetitions must be in [1, 10]")

    torch, dist, layout, runtime = initialize_nccl()
    try:
        hardware = _gather_hardware(torch, dist, layout.world_size)
        gpu_names = sorted({str(item["gpu_name"]) for item in hardware})
        if os.environ.get("FABRICFLOW_REQUIRE_HOMOGENEOUS") == "1" and len(gpu_names) != 1:
            raise RuntimeError(f"heterogeneous GPU models: {gpu_names}")
        node_ids, node_rank_counts = _validate_node_distribution(
            hardware,
            expected_nodes=int(os.environ.get("FABRICFLOW_EXPECTED_NODES", layout.nodes)),
            expected_gpus_per_node=layout.local_size,
        )
        cases: list[dict[str, object]] = []
        case_index = 0
        for repetition in range(1, args.repetitions + 1):
            for dtype_name in DTYPES:
                for elements in counts:
                    for strategy in STRATEGIES:
                        cases.append(
                            _case(
                                torch, dist, runtime,
                                strategy=strategy,
                                dtype_name=dtype_name,
                                elements=elements,
                                repetition=repetition,
                                case_index=case_index,
                            )
                        )
                        case_index += 1
        result: dict[str, object] = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": "hardware_correctness_validation",
            "run_id": os.environ.get("FABRICFLOW_RUN_ID"),
            "cluster_id": os.environ.get("FABRICFLOW_CLUSTER_ID"),
            "nodes": layout.nodes,
            "gpus_per_node": layout.local_size,
            "world_size": layout.world_size,
            "gpu_names": gpu_names,
            "node_ids": node_ids,
            "node_rank_counts": node_rank_counts,
            "cases": cases,
            "correct": all(bool(case["correct"]) for case in cases),
        }
        if layout.rank == 0:
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({
                "output": str(path), "cases": len(cases), "correct": result["correct"]
            }), flush=True)
        dist.barrier()
        if not result["correct"]:
            raise RuntimeError("multi-node correctness validation failed")
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
