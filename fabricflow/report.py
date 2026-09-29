"""Create a concise decision report from correctness-checked hardware rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import statistics
from typing import Mapping

from .calibration import load_jsonl
from .measurement_v2 import validate_benchmark_row
from .sizes import format_size


def build_markdown(rows: list[Mapping[str, object]]) -> str:
    hardware = [
        row for row in rows if row.get("provenance") == "hardware_measurement"
    ]
    if not hardware:
        raise ValueError("report requires rows with provenance=hardware_measurement")
    for row in hardware:
        if row.get("schema_version") == 2:
            validate_benchmark_row(row)
    timing_semantics = {
        str(row.get("timing_semantics", "legacy_pooled_rank_samples"))
        for row in hardware
    }
    if len(timing_semantics) != 1:
        raise ValueError("cannot aggregate different timing semantics")
    grouped: dict[
        tuple[str, int, int, int, int], list[Mapping[str, object]]
    ] = defaultdict(list)
    for row in hardware:
        context = (
            str(row.get("network_transport_requested") or "unspecified"),
            int(row["nodes"]),
            int(row["gpus_per_node"]),
            int(row["world_size"]),
            int(row["message_bytes"]),
        )
        grouped[context].append(row)

    first = hardware[0]
    scopes = sorted(
        {
            str(row["experiment_scope"])
            for row in hardware
            if row.get("experiment_scope")
        }
    )
    node_counts = sorted({int(row["nodes"]) for row in hardware})
    local_sizes = sorted({int(row["gpus_per_node"]) for row in hardware})
    world_sizes = sorted({int(row["world_size"]) for row in hardware})
    transports = sorted(
        {
            str(row.get("network_transport_requested") or "unspecified")
            for row in hardware
        }
    )
    lines = [
        "# FabricFlow hardware decision report",
        "",
        "> Generated only from rows labeled `hardware_measurement`. A winner must "
        "also pass the application-level promotion gate.",
        "",
        "## Run context",
        "",
        f"- Experiment scope: {', '.join(scopes) if scopes else 'unspecified'}",
        f"- Requested network transport: {', '.join(transports)}",
        f"- World sizes: {', '.join(str(value) for value in world_sizes)}",
        f"- Nodes: {', '.join(str(value) for value in node_counts)}",
        f"- GPUs per node: {', '.join(str(value) for value in local_sizes)}",
        f"- PyTorch: {first.get('torch_version', 'unknown')}",
        f"- CUDA: {first.get('cuda_version', 'unknown')}",
        f"- NCCL: {first.get('nccl_version', 'unknown')}",
        f"- Timing semantics: {next(iter(timing_semantics))}",
        "",
        "## Correctness and latency",
        "",
        "| Transport | Nodes | GPUs/node | World | Payload | Repeats | NCCL median p50 (us) | Best fully-correct strategy | Median p50 (us) | Worst run p95 (us) | p50 speedup vs NCCL |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for context in sorted(grouped):
        transport, nodes, gpus_per_node, world_size, size = context
        candidates = grouped[context]
        by_strategy: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in candidates:
            by_strategy[str(row.get("strategy"))].append(row)
        baseline_rows = by_strategy.get("nccl_allreduce", [])
        if not baseline_rows:
            raise ValueError(f"missing nccl_allreduce baseline for {size} bytes")
        if not all(bool(row.get("correct")) for row in baseline_rows):
            raise ValueError(f"NCCL baseline has a correctness failure for {size} bytes")
        baseline = statistics.median(
            float(row["measured_p50_us"]) for row in baseline_rows
        )
        aggregates: list[tuple[str, float, float, int]] = []
        for strategy, strategy_rows in by_strategy.items():
            if not strategy_rows or not all(
                bool(row.get("correct")) for row in strategy_rows
            ):
                continue
            aggregates.append(
                (
                    strategy,
                    statistics.median(
                        float(row["measured_p50_us"]) for row in strategy_rows
                    ),
                    max(float(row["measured_p95_us"]) for row in strategy_rows),
                    len(strategy_rows),
                )
            )
        if not aggregates:
            lines.append(
                f"| {transport} | {nodes} | {gpus_per_node} | {world_size} | "
                f"{format_size(size)} | - | {baseline:.2f} | none | - | - | - |"
            )
            continue
        strategy, best, worst_p95, repeats = min(
            aggregates, key=lambda item: item[1]
        )
        lines.append(
            f"| {transport} | {nodes} | {gpus_per_node} | {world_size} | {format_size(size)} | "
            f"{repeats} | {baseline:.2f} | {strategy} | {best:.2f} | "
            f"{worst_p95:.2f} | {baseline / best:.3f}x |"
        )
    failed = [row for row in hardware if not bool(row.get("correct"))]
    lines.extend(
        [
            "",
            "## Decision notes",
            "",
            f"- Correctness failures: {len(failed)} of {len(hardware)} rows.",
            "- Treat isolated microbenchmark wins as hypotheses until a held-out "
            "training or inference workload improves at p95.",
            "- Re-run randomized strategy order under isolated and contended fabric "
            "conditions before promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a FabricFlow hardware report")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build_markdown(load_jsonl(args.input))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
