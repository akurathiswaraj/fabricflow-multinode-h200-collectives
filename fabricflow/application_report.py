"""Build the predeclared held-out application promotion decision."""

from __future__ import annotations

import argparse
from collections import defaultdict
import math
from pathlib import Path
import statistics
from typing import Mapping

from .calibration import load_jsonl
from .measurement_v2 import validate_application_row


def build_application_markdown(
    rows: list[Mapping[str, object]],
    *,
    minimum_p95_improvement: float = 0.05,
    loss_relative_tolerance: float = 1e-3,
) -> str:
    hardware = [
        row
        for row in rows
        if row.get("provenance") == "hardware_application_measurement"
    ]
    if not hardware:
        raise ValueError("application report requires hardware application rows")
    for row in hardware:
        if row.get("schema_version") == 2:
            validate_application_row(row)
    context_fields = (
        "run_id", "cluster_id", "nodes", "gpus_per_node", "world_size",
        "gpu_names", "network_transport_requested", "gradient_bytes",
        "bucket_bytes", "model_width", "model_depth", "measured_steps",
        "timing_semantics",
    )
    first = hardware[0]
    for row in hardware:
        if any(row.get(field) != first.get(field) for field in context_fields):
            raise ValueError("application report cannot combine incompatible runs")
        if not all(
            math.isfinite(float(row[field]))
            for field in ("step_p50_us", "step_p95_us", "final_global_loss")
        ):
            raise ValueError("application report requires finite measurements")
        if float(row["step_p50_us"]) <= 0 or float(row["step_p95_us"]) <= 0:
            raise ValueError("application step timings must be positive")
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in hardware:
        grouped[str(row["strategy"])].append(row)
    baseline_rows = grouped.get("nccl_allreduce", [])
    if {str(row.get("repeat_id")) for row in baseline_rows} != {"1", "2", "3"} or len(baseline_rows) != 3 or not all(
        bool(row.get("correct")) for row in baseline_rows
    ):
        raise ValueError("promotion requires three correct NCCL baseline repeats")
    baseline_p50 = statistics.median(
        float(row["step_p50_us"]) for row in baseline_rows
    )
    baseline_p95 = statistics.median(
        float(row["step_p95_us"]) for row in baseline_rows
    )
    baseline_loss = statistics.median(
        float(row["final_global_loss"]) for row in baseline_rows
    )
    lines = [
        "# FabricFlow held-out application gate",
        "",
        "> Promotion requires three correct repeats, equivalent final loss, and "
        f"at least {minimum_p95_improvement * 100:.1f}% median p95 step-time improvement.",
        "",
        "## Run context",
        "",
        f"- Nodes: {first['nodes']}",
        f"- GPUs per node: {first['gpus_per_node']}",
        f"- World size: {first['world_size']}",
        f"- GPU: {', '.join(str(value) for value in first.get('gpu_names', []))}",
        f"- Transport: {first.get('network_transport_requested', 'unspecified')}",
        f"- Gradient bytes: {first['gradient_bytes']}",
        f"- Bucket bytes: {first['bucket_bytes']}",
        f"- Timing semantics: {first.get('timing_semantics', 'legacy pooled rank samples')}",
        "",
        "## Promotion decision",
        "",
        "| Strategy | Repeats | Median p50 (ms) | Median p95 (ms) | p95 speedup | Max relative loss delta | Correct | Promoted |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    promoted: list[str] = []
    for strategy in sorted(grouped):
        strategy_rows = grouped[strategy]
        p50 = statistics.median(
            float(row["step_p50_us"]) for row in strategy_rows
        )
        p95 = statistics.median(
            float(row["step_p95_us"]) for row in strategy_rows
        )
        correct = len(strategy_rows) == 3 and {str(row.get("repeat_id")) for row in strategy_rows} == {"1", "2", "3"} and all(
            bool(row.get("correct")) for row in strategy_rows
        )
        denominator = max(abs(baseline_loss), 1e-12)
        loss_delta = max(
            abs(float(row["final_global_loss"]) - baseline_loss) / denominator
            for row in strategy_rows
        )
        p95_improvement = 1 - p95 / baseline_p95
        is_baseline = strategy == "nccl_allreduce"
        passes = (
            not is_baseline
            and correct
            and loss_delta <= loss_relative_tolerance
            and p95_improvement >= minimum_p95_improvement
        )
        if passes:
            promoted.append(strategy)
        lines.append(
            f"| {strategy} | {len(strategy_rows)} | {p50 / 1000:.3f} | "
            f"{p95 / 1000:.3f} | {baseline_p95 / p95:.3f}x | "
            f"{loss_delta:.3e} | {'yes' if correct else 'no'} | "
            f"{'yes' if passes else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Outcome",
            "",
            f"- NCCL median p50: {baseline_p50 / 1000:.3f} ms.",
            f"- NCCL median p95: {baseline_p95 / 1000:.3f} ms.",
            "- Promoted strategies: " + (", ".join(promoted) if promoted else "none"),
            "- This decision applies only to the recorded workload, topology, and transport.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build application promotion report")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-p95-improvement", type=float, default=0.05)
    args = parser.parse_args(argv)
    report = build_application_markdown(
        load_jsonl(args.input),
        minimum_p95_improvement=args.minimum_p95_improvement,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
