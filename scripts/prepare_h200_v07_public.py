"""Create a redacted, checksummed H200 evidence directory for publication.

Run the private-bundle audit first. Never copy provider inventories or raw NCCL
logs to the public directory: they may expose internal addresses and GUIDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from fabricflow.h200_v07_evidence import RUNS, audit_bundle


PRIVATE_FIELDS = {
    "cluster_id", "node_ids", "node_rank_counts", "hostname", "hostnames",
    "rank0_host", "job_id",
}


def _redact(value: dict[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items() if key not in PRIVATE_FIELDS}


def _jsonl_redacted(raw: bytes) -> bytes:
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return ("\n".join(json.dumps(_redact(row), sort_keys=True) for row in rows) + "\n").encode()


def prepare(bundle: Path, output: Path, project_root: Path) -> dict[str, object]:
    summary = audit_bundle(bundle, project_root)
    if output.exists():
        raise FileExistsError(f"public output already exists: {output}")
    output.mkdir(parents=True)
    with zipfile.ZipFile(bundle) as source:
        correctness = _redact(json.loads(source.read("h200-v07-correctness-001.correctness.json")))
        (output / "correctness.json").write_text(
            json.dumps(correctness, indent=2) + "\n", encoding="utf-8"
        )
        for public_name, private_name in (
            ("smoke.jsonl", "h200-v07-smoke-001.jsonl"),
            ("final.jsonl", "h200-v07-final-001.jsonl"),
            ("application.jsonl", "h200-v07-application-001.application.jsonl"),
        ):
            (output / public_name).write_bytes(_jsonl_redacted(source.read(private_name)))
        (output / "application_report.md").write_bytes(
            source.read("h200-v07-application-001.application.report.md")
        )
        manifest = json.loads(source.read("h200-v07-final-001.manifest.json"))
        inventory = json.loads(source.read("h200-v07-final-001.inventory.node-0.json"))
        metadata = {
            "schema_version": 1,
            "description": "Redacted copies of independently audited H200 hardware measurements",
            "private_evidence_archive_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
            "private_evidence_files_audited": summary["checksummed_files"],
            "source_sha256": manifest["source_sha256"],
            "nodes": 2,
            "gpus_per_node": 8,
            "gpu_model": "NVIDIA H200",
            "pytorch": inventory["pytorch"],
            "cuda_runtime": inventory["cuda_runtime"],
            "nccl": inventory["nccl"],
            "transport": "RoCE / GPUDirect RDMA, verified by NCCL NET/IB/GDRDMA logs",
            "redacted_fields": sorted(PRIVATE_FIELDS),
            "raw_rank_logs_published": False,
            "independent_audit_command": "python -m fabricflow.h200_v07_evidence PRIVATE_BUNDLE.zip --project-root .",
        }
        (output / "provenance.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        (output / "transport_audit.json").write_text(
            json.dumps(summary["transport"], indent=2) + "\n", encoding="utf-8"
        )
    (output / "README.md").write_text(
        "# H200 v0.7 public evidence\n\n"
        "These are redacted copies of four verified runs: 90 correctness cases, "
        "six smoke rows, 99 final microbenchmark rows, and nine synthetic "
        "training-step rows. The timing arrays are preserved so the reported "
        "p50/p95 values can be recomputed. Provider node addresses, cluster IDs, "
        "GUIDs, and raw NCCL logs are intentionally not published.\n\n"
        "`transport_audit.json` contains per-run counts computed from all 352 "
        "private NCCL rank logs. `provenance.json` records the private bundle's "
        "SHA-256 and the SHA-256 hashes of the executed source files. The "
        "private bundle can be audited with `fabricflow.h200_v07_evidence` if "
        "made available under an appropriate sharing arrangement.\n\n"
        "The physical link is Ethernet/RoCE; NCCL's `IB` label does not mean "
        "physical InfiniBand. Logical process-group lanes are not proof of "
        "per-lane NIC affinity. See `docs/results_h200_v07.md` for the "
        "decision and limitations.\n",
        encoding="utf-8",
    )
    files = sorted(path for path in output.iterdir() if path.is_file())
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
        for path in files
    ]
    (output / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"public_files": len(files), "output": str(output), "private_audit": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(prepare(args.bundle, args.output, args.project_root), indent=2))


if __name__ == "__main__":
    main()
