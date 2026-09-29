"""Auditable strategy selection over analytical and calibrated predictions."""

from __future__ import annotations

from dataclasses import dataclass

from .calibration import CalibrationProfile
from .cost_model import Prediction, predict_all
from .topology import ClusterTopology


@dataclass(frozen=True)
class Plan:
    selected: Prediction
    candidates: tuple[Prediction, ...]
    message_bytes: int
    reason: str


class Planner:
    def __init__(
        self,
        topology: ClusterTopology,
        *,
        calibration: CalibrationProfile | None = None,
        pipeline_chunks: int = 8,
    ) -> None:
        self.topology = topology
        self.calibration = calibration or CalibrationProfile()
        self.pipeline_chunks = pipeline_chunks

    def plan(self, message_bytes: int, *, runtime_only: bool = True) -> Plan:
        raw = predict_all(
            self.topology,
            message_bytes,
            pipeline_chunks=self.pipeline_chunks,
        )
        candidates = tuple(self.calibration.apply(item) for item in raw)
        eligible = tuple(
            item for item in candidates if item.runtime_supported or not runtime_only
        )
        if not eligible:
            raise RuntimeError("no strategy satisfies the planner constraints")
        selected = min(eligible, key=lambda item: item.latency_us)
        scope = "runtime-capable" if runtime_only else "modeled"
        reason = (
            f"lowest calibrated critical-path estimate among {len(eligible)} "
            f"{scope} candidates; factor={selected.calibration_factor:.3f}"
        )
        return Plan(
            selected=selected,
            candidates=candidates,
            message_bytes=message_bytes,
            reason=reason,
        )

