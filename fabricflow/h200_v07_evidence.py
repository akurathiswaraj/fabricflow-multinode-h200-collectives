"""Audit the private, checksummed H200 v0.7 hardware-evidence ZIP.

The public release may summarize this evidence, but should not contain raw
provider network addresses or unredacted NCCL logs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics
import zipfile

from .application_report import build_application_markdown
from .measurement_v2 import validate_application_row, validate_benchmark_row


RUNS = {
    "h200-v07-correctness-001": ("correctness", 1),
    "h200-v07-smoke-001": ("smoke", 3),
    "h200-v07-final-001": ("final", 9),
    "h200-v07-application-001": ("application", 9),
}
STRATEGIES = {"nccl_allreduce", "hierarchical_leader", "hierarchical_rail"}
SIZES = [1024 * 4**index for index in range(11)]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(archive: zipfile.ZipFile, name: str) -> dict[str, object]:
    value = json.loads(archive.read(name))
    _require(isinstance(value, dict), f"expected JSON object: {name}")
    return value


def _jsonl(archive: zipfile.ZipFile, name: str) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in archive.read(name).splitlines() if line.strip()]
    _require(all(isinstance(row, dict) for row in rows), f"invalid JSONL: {name}")
    return rows


def _layout(value: dict[str, object], run_id: str) -> None:
    _require(value.get("run_id") == run_id, f"mixed run ID: {run_id}")
    _require(value.get("cluster_id"), f"missing cluster ID: {run_id}")
    _require(
        value.get("nodes") == 2
        and value.get("gpus_per_node") == 8
        and value.get("world_size") == 16,
        f"wrong cluster layout: {run_id}",
    )
    _require(value.get("gpu_names") == ["NVIDIA H200"], f"wrong GPU: {run_id}")
    counts = value.get("node_rank_counts")
    _require(
        isinstance(counts, dict)
        and len(counts) == 2
        and sorted(counts.values()) == [8, 8],
        f"node distribution is not proven: {run_id}",
    )


def audit_bundle(bundle: Path, project_root: Path) -> dict[str, object]:
    project_root = project_root.resolve()
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        _require(len(names) == len(set(names)), "duplicate ZIP member")
        _require(archive.testzip() is None, "ZIP CRC failure")
        for name in names:
            parts = PurePosixPath(name).parts
            _require(
                name and not name.startswith("/") and ".." not in parts,
                f"unsafe ZIP member: {name}",
            )
        _require("SHA256SUMS.txt" in names, "missing checksums")
        expected: dict[str, str] = {}
        for line in archive.read("SHA256SUMS.txt").decode().splitlines():
            digest, name = line.split("  ", maxsplit=1)
            _require(name not in expected, f"duplicate checksum: {name}")
            expected[name] = digest
        _require(set(expected) == set(names) - {"SHA256SUMS.txt"}, "incomplete checksums")
        _require(len(expected) == 369, "unexpected source file count")
        for name, digest in expected.items():
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            _require(actual == digest, f"checksum mismatch: {name}")

        manifests = {}
        clusters = {}
        transport: dict[str, dict[str, int]] = {}
        for run_id, (mode, launches) in RUNS.items():
            manifest = _json(archive, f"{run_id}.manifest.json")
            _require(
                manifest.get("run_id") == run_id
                and manifest.get("mode") == mode
                and manifest.get("nodes") == 2
                and manifest.get("gpus_per_node") == 8
                and manifest.get("gpu_request") == "H200:8"
                and manifest.get("rdma_requested") is True,
                f"incorrect manifest: {run_id}",
            )
            manifests[run_id] = manifest
            inventories = [
                _json(archive, f"{run_id}.inventory.node-{node}.json")
                for node in (0, 1)
            ]
            _require(
                [item.get("container_rank") for item in inventories] == [0, 1]
                and all(item.get("gpu_count") == 8 for item in inventories)
                and all(item.get("gpu_names") == ["NVIDIA H200"] * 8 for item in inventories),
                f"incorrect node inventory: {run_id}",
            )
            _require(
                inventories[0].get("container_ip") != inventories[1].get("container_ip"),
                f"nodes are not distinct: {run_id}",
            )
            clusters[run_id] = inventories[0].get("cluster_id")
            _require(
                clusters[run_id] == inventories[1].get("cluster_id"),
                f"inconsistent cluster ID: {run_id}",
            )
            source_hashes = manifest.get("source_sha256")
            _require(isinstance(source_hashes, dict) and source_hashes, "missing source hashes")
            for inventory in inventories:
                _require(
                    inventory.get("source_sha256") == source_hashes,
                    f"source hashes differ between nodes: {run_id}",
                )
            for relative, digest in source_hashes.items():
                path = (project_root / relative).resolve()
                _require(path.is_relative_to(project_root) and path.is_file(), f"missing source: {relative}")
                _require(
                    hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                    f"local source differs from executed source: {relative}",
                )

            totals = Counter()
            for node in (0, 1):
                for launch in range(1, launches + 1):
                    prefix = f"{run_id}.node-{node}.launch-{launch}.nccl."
                    logs = [name for name in expected if name.startswith(prefix) and name.endswith(".log")]
                    _require(len(logs) == 8, f"missing NCCL rank logs: {prefix}")
                    markers = Counter()
                    for log in logs:
                        data = archive.read(log)
                        markers["using_ib"] += data.count(b"Using network IB")
                        markers["ib_channels"] += data.count(b"via NET/IB")
                        markers["gdrdma"] += data.count(b"GDRDMA")
                        markers["socket_channels"] += data.count(b"via NET/Socket")
                    _require(
                        markers["using_ib"] > 0
                        and markers["ib_channels"] > 0
                        and markers["gdrdma"] > 0
                        and markers["socket_channels"] == 0,
                        f"RDMA data path not proven: {prefix}",
                    )
                    totals.update(markers)
                    totals["log_files"] += len(logs)
            transport[run_id] = dict(totals)

        correctness_id = "h200-v07-correctness-001"
        correctness = _json(archive, f"{correctness_id}.correctness.json")
        _layout(correctness, correctness_id)
        _require(correctness.get("cluster_id") == clusters[correctness_id], "wrong correctness cluster")
        cases = correctness.get("cases")
        _require(correctness.get("correct") is True and isinstance(cases, list) and len(cases) == 90, "correctness suite failed")
        case_keys = Counter(
            (case["strategy"], case["dtype"], case["elements"], case["repetition"])
            for case in cases
        )
        _require(len(case_keys) == 90 and set(case_keys.values()) == {1}, "duplicate correctness case")
        _require(
            {case["strategy"] for case in cases} == STRATEGIES
            and {case["dtype"] for case in cases} == {"float32", "float16", "bfloat16"}
            and {case["elements"] for case in cases} == {1, 7, 9, 1025, 262147}
            and {case["repetition"] for case in cases} == {1, 2}
            and all(case["correct"] is True and case["max_abs_error"] == 0 for case in cases),
            "incomplete or incorrect correctness matrix",
        )

        benchmark_summary = {}
        for run_id, size_grid, repeat_grid in (
            ("h200-v07-smoke-001", [1048576, 67108864], {"1"}),
            ("h200-v07-final-001", SIZES, {"1", "2", "3"}),
        ):
            rows = _jsonl(archive, f"{run_id}.jsonl")
            _require(len(rows) == 3 * len(size_grid) * len(repeat_grid), f"incomplete rows: {run_id}")
            keys = Counter((row["strategy"], row["message_bytes"], str(row["repeat_id"])) for row in rows)
            _require(len(keys) == len(rows) and set(keys.values()) == {1}, f"duplicate row: {run_id}")
            _require(
                {row["strategy"] for row in rows} == STRATEGIES
                and {row["message_bytes"] for row in rows} == set(size_grid)
                and {str(row["repeat_id"]) for row in rows} == repeat_grid,
                f"incomplete sweep: {run_id}",
            )
            for row in rows:
                validate_benchmark_row(row)
                _layout(row, run_id)
                _require(row["cluster_id"] == clusters[run_id], f"wrong benchmark cluster: {run_id}")
                _require(row["correct"] is True and row["max_abs_error"] == 0, f"incorrect benchmark: {run_id}")
            benchmark_summary[run_id] = rows

        application_id = "h200-v07-application-001"
        application = _jsonl(archive, f"{application_id}.application.jsonl")
        _require(len(application) == 9, "incomplete application rows")
        app_keys = Counter((row["strategy"], str(row["repeat_id"])) for row in application)
        _require(len(app_keys) == 9 and set(app_keys.values()) == {1}, "duplicate application row")
        _require(
            {row["strategy"] for row in application} == STRATEGIES
            and {str(row["repeat_id"]) for row in application} == {"1", "2", "3"},
            "incomplete application sweep",
        )
        for row in application:
            validate_application_row(row)
            _layout(row, application_id)
            _require(row["cluster_id"] == clusters[application_id], "wrong application cluster")
            _require(row["correct"] is True and row["max_parameter_spread"] == 0, "incorrect application row")
        report = archive.read(f"{application_id}.application.report.md").decode()
        _require(report == build_application_markdown(application), "application report does not reproduce")
        _require("- Promoted strategies: none" in report, "unexpected application promotion")

        final = benchmark_summary["h200-v07-final-001"]
        medians: dict[int, dict[str, float]] = defaultdict(dict)
        for size in SIZES:
            for strategy in STRATEGIES:
                samples = [float(row["measured_p50_us"]) for row in final if row["message_bytes"] == size and row["strategy"] == strategy]
                medians[size][strategy] = statistics.median(samples)
        winners = {str(size): min(values, key=values.get) for size, values in medians.items()}

        return {
            "checksummed_files": len(expected),
            "source_hashes_match_executed_code": True,
            "correctness_cases": len(cases),
            "smoke_rows": len(benchmark_summary["h200-v07-smoke-001"]),
            "final_rows": len(final),
            "application_rows": len(application),
            "application_promoted_strategies": [],
            "final_winners_by_bytes": winners,
            "transport": transport,
        }


def audit_public_directory(root: Path, project_root: Path) -> dict[str, object]:
    """Verify the published, redacted timing records without the private logs."""

    root = root.resolve()
    project_root = project_root.resolve()
    checksum_file = root / "SHA256SUMS"
    _require(checksum_file.is_file(), "missing public SHA256SUMS")
    expected: dict[str, str] = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        _require(name not in expected, f"duplicate public checksum: {name}")
        expected[name] = digest
    actual_names = {path.name for path in root.iterdir() if path.is_file() and path != checksum_file}
    _require(set(expected) == actual_names, "public file list does not match checksums")
    _require(len(expected) == 8, "unexpected public evidence file count")
    for name, digest in expected.items():
        _require(PurePosixPath(name).name == name, f"unsafe public name: {name}")
        _require(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, f"public checksum mismatch: {name}")

    metadata = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    _require(metadata["nodes"] == 2 and metadata["gpus_per_node"] == 8, "wrong public topology")
    _require(metadata["gpu_model"] == "NVIDIA H200", "wrong public GPU")
    _require(metadata["raw_rank_logs_published"] is False, "unexpected raw-log statement")
    _require(len(metadata["private_evidence_archive_sha256"]) == 64, "missing private bundle digest")
    for relative, digest in metadata["source_sha256"].items():
        path = (project_root / relative).resolve()
        _require(path.is_relative_to(project_root) and path.is_file(), f"missing executed source: {relative}")
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f"source hash mismatch: {relative}")

    correctness = json.loads((root / "correctness.json").read_text(encoding="utf-8"))
    _require(correctness["correct"] is True and len(correctness["cases"]) == 90, "public correctness failed")
    _require(all(case["correct"] is True for case in correctness["cases"]), "incorrect public case")
    final = [json.loads(line) for line in (root / "final.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    smoke = [json.loads(line) for line in (root / "smoke.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    application = [json.loads(line) for line in (root / "application.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    _require((len(smoke), len(final), len(application)) == (6, 99, 9), "public row counts changed")
    for row in smoke + final:
        validate_benchmark_row(row)
        _require(row["correct"] is True, "incorrect public benchmark")
    for row in application:
        validate_application_row(row)
        _require(row["correct"] is True, "incorrect public application")
    forbidden = {"cluster_id", "node_ids", "node_rank_counts", "hostname", "hostnames", "rank0_host", "job_id"}
    _require(
        all(not forbidden.intersection(row) for row in [correctness, *smoke, *final, *application]),
        "unredacted provider identity in public rows",
    )
    report = (root / "application_report.md").read_text(encoding="utf-8")
    _require(report == build_application_markdown(application), "public application report does not reproduce")
    _require("- Promoted strategies: none" in report, "unexpected public promotion")
    transport = json.loads((root / "transport_audit.json").read_text(encoding="utf-8"))
    _require(set(transport) == set(RUNS), "missing public transport audit")
    for run_id, (_, launches) in RUNS.items():
        counts = transport[run_id]
        _require(
            counts["log_files"] == launches * 16
            and counts["using_ib"] > 0
            and counts["ib_channels"] > 0
            and counts["gdrdma"] > 0
            and counts["socket_channels"] == 0,
            f"invalid public transport summary: {run_id}",
        )
    return {
        "checksummed_public_files": len(expected),
        "correctness_cases": len(correctness["cases"]),
        "smoke_rows": len(smoke),
        "final_rows": len(final),
        "application_rows": len(application),
        "application_promoted_strategies": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--public-directory", action="store_true")
    args = parser.parse_args()
    result = (
        audit_public_directory(args.bundle, args.project_root)
        if args.public_directory
        else audit_bundle(args.bundle, args.project_root)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
