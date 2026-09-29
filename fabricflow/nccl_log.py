"""Small, auditable summaries of NCCL transport-selection logs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable


TRANSPORT_MARKERS = {
    "socket": ("NET/Socket", "Using network Socket"),
    "rdma": ("NET/IB", "Using network IB"),
}


def summarize_nccl_logs(paths: Iterable[str | Path]) -> dict[str, object]:
    files = sorted(Path(path) for path in paths)
    counts: Counter[str] = Counter()
    evidence: list[str] = []
    for path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as error:
            evidence.append(f"{path.name}: unreadable: {error}")
            continue
        for line in lines:
            matched = False
            for transport, markers in TRANSPORT_MARKERS.items():
                if any(marker in line for marker in markers):
                    counts[transport] += 1
                    matched = True
            if matched and len(evidence) < 200:
                evidence.append(f"{path.name}: {line.strip()}")

    observed = [name for name in TRANSPORT_MARKERS if counts[name] > 0]
    if len(observed) == 1:
        selected = observed[0]
    elif len(observed) > 1:
        selected = "mixed"
    else:
        selected = "unknown"
    return {
        "schema_version": 1,
        "files_scanned": len(files),
        "observed_transport": selected,
        "marker_counts": dict(sorted(counts.items())),
        "evidence": evidence,
    }
