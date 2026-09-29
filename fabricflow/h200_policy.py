"""Conservative, topology-locked empirical decision from H200 sweep rows.

Leave-one-repeat-out checks are a within-session diagnostic, not an independent
application or cluster validation. Unknown payloads and contexts use NCCL.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Mapping

from .calibration import load_jsonl
from .cost_model import Strategy


STRATEGIES = (
    Strategy.NCCL_ALLREDUCE.value,
    Strategy.HIERARCHICAL_LEADER.value,
    Strategy.HIERARCHICAL_RAIL.value,
)
REPEATS = {"1", "2", "3"}


def _valid_row(row: Mapping[str, object]) -> bool:
    return (
        row.get("provenance") == "hardware_measurement"
        and row.get("nodes") == 2
        and row.get("gpus_per_node") == 8
        and row.get("world_size") == 16
        and row.get("gpu_names") == ["NVIDIA H200"]
        and row.get("correct") is True
        and math.isfinite(float(row.get("measured_p50_us", math.nan)))
        and float(row["measured_p50_us"]) > 0
    )


def build_policy(
    rows: list[Mapping[str, object]], *, minimum_improvement: float = 0.05
) -> dict[str, object]:
    if not 0 <= minimum_improvement < 1:
        raise ValueError("minimum improvement must be in [0, 1)")
    if not rows or not all(_valid_row(row) for row in rows):
        raise ValueError("policy requires correct 2x8 H200 hardware rows only")
    run_ids = {str(row.get("run_id")) for row in rows}
    schemas = {int(row.get("schema_version", 0)) for row in rows}
    semantics = {str(row.get("timing_semantics", "legacy_pooled_rank_samples")) for row in rows}
    if len(run_ids) != 1 or len(schemas) != 1 or len(semantics) != 1:
        raise ValueError("do not combine runs or timing schemas in one policy")

    by_size: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        size = int(row["message_bytes"])
        strategy = str(row["strategy"])
        repeat = str(row["repeat_id"])
        if size < 1 or strategy not in STRATEGIES or repeat not in REPEATS:
            raise ValueError("unexpected payload, strategy, or repeat")
        if repeat in by_size[size][strategy]:
            raise ValueError(f"duplicate row for {size}/{strategy}/{repeat}")
        by_size[size][strategy][repeat] = float(row["measured_p50_us"])

    decisions: dict[str, dict[str, object]] = {}
    for size in sorted(by_size):
        measurements = by_size[size]
        if set(measurements) != set(STRATEGIES) or any(set(measurements[s]) != REPEATS for s in STRATEGIES):
            raise ValueError(f"incomplete strategy/repeat matrix at {size} bytes")
        folds: list[dict[str, object]] = []
        for held_out in sorted(REPEATS):
            train_median = {
                strategy: statistics.median(
                    value for repeat, value in measurements[strategy].items()
                    if repeat != held_out
                )
                for strategy in STRATEGIES
            }
            fastest = min(STRATEGIES, key=lambda strategy: train_median[strategy])
            if fastest != Strategy.NCCL_ALLREDUCE.value and train_median[fastest] > (
                1 - minimum_improvement
            ) * train_median[Strategy.NCCL_ALLREDUCE.value]:
                fastest = Strategy.NCCL_ALLREDUCE.value
            held_out_valid = (
                fastest == Strategy.NCCL_ALLREDUCE.value
                or measurements[fastest][held_out]
                <= (1 - minimum_improvement)
                * measurements[Strategy.NCCL_ALLREDUCE.value][held_out]
            )
            folds.append({
                "held_out_repeat": held_out,
                "train_selected": fastest,
                "held_out_pass": held_out_valid,
            })
        selections = {str(fold["train_selected"]) for fold in folds}
        stable = len(selections) == 1 and all(bool(fold["held_out_pass"]) for fold in folds)
        selected = next(iter(selections)) if stable else Strategy.NCCL_ALLREDUCE.value
        decisions[str(size)] = {
            "selected": selected,
            "stable_within_session": stable,
            "median_p50_us": {
                strategy: round(statistics.median(measurements[strategy].values()), 2)
                for strategy in STRATEGIES
            },
            "folds": folds,
        }
    return {
        "schema_version": 1,
        "source_run_id": next(iter(run_ids)),
        "source_row_schema_version": next(iter(schemas)),
        "source_timing_semantics": next(iter(semantics)),
        "context": "2 nodes x 8 NVIDIA H200; measured payloads only",
        "validation_scope": "leave_one_repeat_out_within_one_cluster_session",
        "minimum_improvement": minimum_improvement,
        "fallback": Strategy.NCCL_ALLREDUCE.value,
        "decisions_by_message_bytes": decisions,
    }


def select(policy: Mapping[str, object], message_bytes: int) -> str:
    if message_bytes < 1:
        raise ValueError("message size must be positive")
    decisions = policy["decisions_by_message_bytes"]
    if not isinstance(decisions, Mapping):
        raise ValueError("invalid policy decisions")
    decision = decisions.get(str(message_bytes))
    if decision is None:
        return Strategy.NCCL_ALLREDUCE.value
    if not isinstance(decision, Mapping):
        raise ValueError("invalid policy entry")
    return str(decision["selected"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--minimum-improvement", type=float, default=0.05)
    args = parser.parse_args(argv)
    result = build_policy(
        load_jsonl(args.input), minimum_improvement=args.minimum_improvement
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
