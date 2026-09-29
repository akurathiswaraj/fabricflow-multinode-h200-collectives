"""Fail-closed audit of the published two-node H200/RDMA campaign."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


STRATEGIES = {"nccl_allreduce", "hierarchical_leader", "hierarchical_rail"}
FINAL_ID = "h200-rdma-final-001"
SMOKE_ID = "h200-rdma-smoke-001"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _rows(path: Path) -> list[dict[str, object]]:
    _require(path.is_file(), f"missing rows: {path.name}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    checksums = root / "SHA256SUMS"
    _require(checksums.is_file(), "missing SHA256SUMS")
    checked: set[str] = set()
    for line in checksums.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", maxsplit=1)
        path = (root / relative).resolve()
        _require(path.is_relative_to(root), f"checksum path escapes evidence: {relative}")
        _require(path.is_file(), f"missing checksummed file: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(actual == expected, f"checksum mismatch: {relative}")
        _require(relative not in checked, f"duplicate checksum entry: {relative}")
        checked.add(relative)
    _require(len(checked) == 198, f"expected 198 source files, got {len(checked)}")
    actual_files = {str(p.relative_to(root)).replace("\\", "/") for p in root.rglob("*") if p.is_file() and p != checksums}
    _require(actual_files == checked, "unlisted or missing evidence files")

    final = _rows(root / f"{FINAL_ID}.jsonl")
    smoke = _rows(root / f"{SMOKE_ID}.jsonl")
    _require(len(final) == 99, f"expected 99 final rows, got {len(final)}")
    _require(len(smoke) == 6, f"expected 6 smoke rows, got {len(smoke)}")
    _require(all(row.get("correct") is True and float(row.get("max_abs_error", -1)) == 0.0 for row in final + smoke), "incorrect hardware row")
    _require(all(row.get("provenance") == "hardware_measurement" for row in final + smoke), "non-hardware provenance")
    _require(all(row.get("experiment_scope") == "multi_node_rdma" for row in final + smoke), "wrong experiment scope")
    _require(all(row.get("nodes") == 2 and row.get("gpus_per_node") == 8 and row.get("world_size") == 16 for row in final + smoke), "wrong cluster shape")
    _require(all(row.get("gpu_names") == ["NVIDIA H200"] for row in final + smoke), "wrong GPU model")
    _require(all(len(row.get("node_ids", [])) == 2 and sorted(row.get("node_rank_counts", {}).values()) == [8, 8] for row in final + smoke), "unverified node/rank identity")
    _require(all(row.get("run_id") == FINAL_ID for row in final), "mixed final run IDs")
    _require(all(row.get("run_id") == SMOKE_ID for row in smoke), "mixed smoke run IDs")
    _require(Counter(str(row["strategy"]) for row in final) == {name: 33 for name in STRATEGIES}, "incomplete strategy sweep")
    _require({str(row["repeat_id"]) for row in final} == {"1", "2", "3"}, "missing repeats")
    sizes = sorted({int(row["message_bytes"]) for row in final})
    _require(sizes == [1024 * 4**i for i in range(11)], "unexpected payload grid")

    latency: dict[int, dict[str, float]] = defaultdict(dict)
    for size in sizes:
        for strategy in STRATEGIES:
            selected = [row for row in final if int(row["message_bytes"]) == size and row["strategy"] == strategy]
            _require(len(selected) == 3 and {str(row["repeat_id"]) for row in selected} == {"1", "2", "3"}, f"missing repeats for {strategy}/{size}")
            latency[size][strategy] = round(statistics.median(float(row["measured_p50_us"]) for row in selected), 2)
    winners = {size: min(values, key=values.get) for size, values in latency.items()}
    _require(set(winners.values()) == {"nccl_allreduce"}, "published NCCL-wins-all result changed")

    transport: dict[str, dict[str, int]] = {}
    for run_id, launches in ((SMOKE_ID, 3), (FINAL_ID, 9)):
        logs = sorted(root.glob(f"{run_id}.node-*.nccl.*.log"))
        _require(len(logs) == launches * 16, f"missing rank logs for {run_id}")
        counts = {"using_ib": 0, "ib_channels": 0, "gdrdma_channels": 0, "socket_channels": 0}
        for log in logs:
            data = log.read_text(encoding="utf-8", errors="replace")
            counts["using_ib"] += data.count("Using network IB")
            counts["ib_channels"] += data.count("via NET/IB")
            counts["gdrdma_channels"] += data.count("GDRDMA")
            counts["socket_channels"] += data.count("via NET/Socket")
        _require(counts["using_ib"] > 0 and counts["ib_channels"] > 0 and counts["gdrdma_channels"] > 0 and counts["socket_channels"] == 0, f"transport not proven for {run_id}")
        for node in (0, 1):
            inventory = json.loads((root / f"{run_id}.inventory.node-{node}.json").read_text(encoding="utf-8"))
            _require(inventory.get("container_rank") == node and inventory.get("gpu_count") == 8 and inventory.get("gpu_names") == ["NVIDIA H200"] * 8, f"bad inventory for {run_id}/node-{node}")
        transport[run_id] = {"log_files": len(logs), **counts}

    return {
        "checksummed_source_files": len(checked),
        "smoke_rows": len(smoke),
        "final_rows": len(final),
        "all_correct": True,
        "nodes": 2,
        "gpus_per_node": 8,
        "payloads": len(sizes),
        "winner_at_all_payloads": "nccl_allreduce",
        "median_p50_us_by_payload": latency,
        "transport": transport,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.evidence), indent=2))


if __name__ == "__main__":
    main()
