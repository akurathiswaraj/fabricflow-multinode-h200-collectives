"""Recompute version-2 distributed timing claims from saved rank samples."""

from __future__ import annotations

import math
from typing import Mapping

from .benchmark import _completion_timings, _percentile


def _samples(row: Mapping[str, object], key: str, count: int) -> list[list[float]]:
    raw = row.get(key)
    world = int(row["world_size"])
    if not isinstance(raw, list) or len(raw) != world:
        raise ValueError(f"{key} must have one timing series per rank")
    result: list[list[float]] = []
    for rank_values in raw:
        if not isinstance(rank_values, list) or len(rank_values) != count:
            raise ValueError(f"{key} has an invalid iteration count")
        values = [float(value) for value in rank_values]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"{key} contains invalid timings")
        result.append(values)
    return result


def _match(actual: float, expected: float, label: str) -> None:
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-6):
        raise ValueError(f"{label} does not match saved rank samples")


def validate_benchmark_row(row: Mapping[str, object]) -> None:
    if int(row.get("schema_version", 0)) != 2:
        raise ValueError("benchmark timing audit requires schema version 2")
    if row.get("timing_semantics") != "per_iteration_max_rank_cuda_duration_us":
        raise ValueError("unexpected benchmark timing semantics")
    rank_values = _samples(row, "rank_timings_us", int(row["iterations"]))
    completion = _completion_timings(rank_values)
    stored = row.get("completion_timings_us")
    if not isinstance(stored, list) or len(stored) != len(completion):
        raise ValueError("missing benchmark completion timings")
    for actual, expected in zip(stored, completion):
        _match(float(actual), expected, "completion timing")
    _match(float(row["measured_p50_us"]), _percentile(completion, 50), "benchmark p50")
    _match(float(row["measured_p95_us"]), _percentile(completion, 95), "benchmark p95")
    _match(float(row["measured_max_us"]), max(completion), "benchmark max")


def validate_application_row(row: Mapping[str, object]) -> None:
    if int(row.get("schema_version", 0)) != 2:
        raise ValueError("application timing audit requires schema version 2")
    if row.get("timing_semantics") != "per_step_max_rank_wall_duration_us":
        raise ValueError("unexpected application timing semantics")
    rank_values = _samples(row, "rank_step_timings_us", int(row["measured_steps"]))
    completion = _completion_timings(rank_values)
    stored = row.get("completion_step_timings_us")
    if not isinstance(stored, list) or len(stored) != len(completion):
        raise ValueError("missing application completion timings")
    for actual, expected in zip(stored, completion):
        _match(float(actual), expected, "completion step")
    _match(float(row["step_p50_us"]), _percentile(completion, 50), "application p50")
    _match(float(row["step_p95_us"]), _percentile(completion, 95), "application p95")
    _match(float(row["step_max_us"]), max(completion), "application max")
