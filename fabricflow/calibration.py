"""Empirical correction factors learned from benchmark JSONL."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Mapping

from .cost_model import Prediction, Strategy


@dataclass(frozen=True)
class CalibrationProfile:
    factors: Mapping[str, float] = field(default_factory=dict)
    sample_counts: Mapping[str, int] = field(default_factory=dict)
    source: str = "uncalibrated"

    def factor_for(self, strategy: Strategy) -> float:
        factor = float(self.factors.get(strategy.value, 1.0))
        return factor if math.isfinite(factor) and factor > 0 else 1.0

    def apply(self, prediction: Prediction) -> Prediction:
        return prediction.calibrated(self.factor_for(prediction.strategy))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source": self.source,
            "factors": dict(self.factors),
            "sample_counts": dict(self.sample_counts),
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationProfile":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            factors={str(k): float(v) for k, v in raw.get("factors", {}).items()},
            sample_counts={
                str(k): int(v) for k, v in raw.get("sample_counts", {}).items()
            },
            source=str(raw.get("source", path)),
        )


def fit_calibration(
    rows: Iterable[Mapping[str, object]],
    *,
    source: str = "benchmark-jsonl",
) -> CalibrationProfile:
    ratios: dict[str, list[float]] = {}
    for row in rows:
        strategy = str(row.get("strategy", ""))
        measured = float(row.get("measured_p50_us", 0) or 0)
        predicted = float(row.get("predicted_us", 0) or 0)
        if strategy and measured > 0 and predicted > 0:
            ratios.setdefault(strategy, []).append(measured / predicted)
    factors = {
        strategy: min(4.0, max(0.25, statistics.median(values)))
        for strategy, values in ratios.items()
    }
    return CalibrationProfile(
        factors=factors,
        sample_counts={key: len(value) for key, value in ratios.items()},
        source=source,
    )


def load_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON on line {line_number}: {error}") from error
    return rows

