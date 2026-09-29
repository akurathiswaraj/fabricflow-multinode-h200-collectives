"""Alpha-beta cost models for multi-node all-reduce strategies.

These estimates are decision-support, not a replacement for nccl-tests or an
application benchmark. The planner keeps each phase visible so a reviewer can
challenge assumptions instead of accepting a black-box score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import math
from typing import Iterable

from .topology import ClusterTopology, LinkTier


class Strategy(str, Enum):
    NCCL_ALLREDUCE = "nccl_allreduce"
    HIERARCHICAL_LEADER = "hierarchical_leader"
    HIERARCHICAL_RAIL = "hierarchical_rail"
    PIPELINED_RAIL = "pipelined_rail"


@dataclass(frozen=True)
class Phase:
    name: str
    link: str
    duration_us: float
    algorithmic_bytes_per_rank: float
    steps: int


@dataclass(frozen=True)
class Prediction:
    strategy: Strategy
    payload_bytes: int
    latency_us: float
    algorithmic_bytes_per_rank: float
    phases: tuple[Phase, ...]
    runtime_supported: bool
    assumption: str
    calibration_factor: float = 1.0

    @property
    def payload_goodput_gbps(self) -> float:
        if self.latency_us <= 0:
            return math.inf
        return self.payload_bytes * 8 / (self.latency_us * 1e-6) / 1e9

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["strategy"] = self.strategy.value
        return result

    def calibrated(self, factor: float) -> "Prediction":
        if factor <= 0:
            raise ValueError("calibration factor must be positive")
        return replace(
            self,
            latency_us=self.latency_us * factor,
            phases=tuple(
                replace(phase, duration_us=phase.duration_us * factor)
                for phase in self.phases
            ),
            calibration_factor=self.calibration_factor * factor,
        )


def _phase(
    name: str,
    link: LinkTier,
    *,
    steps: int,
    bytes_per_step: float,
) -> Phase:
    return Phase(
        name=name,
        link=link.name,
        duration_us=link.transfer_us(bytes_per_step, steps=steps),
        algorithmic_bytes_per_rank=steps * bytes_per_step,
        steps=steps,
    )


def _ring_reduce_scatter(name: str, link: LinkTier, participants: int, size: int) -> Phase:
    if participants <= 1:
        return _phase(name, link, steps=0, bytes_per_step=0)
    return _phase(
        name,
        link,
        steps=participants - 1,
        bytes_per_step=size / participants,
    )


def _ring_allgather(name: str, link: LinkTier, participants: int, size: int) -> Phase:
    return _ring_reduce_scatter(name, link, participants, size)


def _ring_allreduce(name: str, link: LinkTier, participants: int, size: int) -> Phase:
    if participants <= 1:
        return _phase(name, link, steps=0, bytes_per_step=0)
    return _phase(
        name,
        link,
        steps=2 * (participants - 1),
        bytes_per_step=size / participants,
    )


def _tree_phase(name: str, link: LinkTier, participants: int, size: int) -> Phase:
    if participants <= 1:
        return _phase(name, link, steps=0, bytes_per_step=0)
    return _phase(
        name,
        link,
        steps=math.ceil(math.log2(participants)),
        bytes_per_step=size,
    )


def _prediction(
    strategy: Strategy,
    phases: Iterable[Phase],
    *,
    payload_bytes: int,
    runtime_supported: bool,
    assumption: str,
) -> Prediction:
    materialized = tuple(phases)
    return Prediction(
        strategy=strategy,
        payload_bytes=payload_bytes,
        latency_us=sum(phase.duration_us for phase in materialized),
        algorithmic_bytes_per_rank=sum(
            phase.algorithmic_bytes_per_rank for phase in materialized
        ),
        phases=materialized,
        runtime_supported=runtime_supported,
        assumption=assumption,
    )


def nccl_allreduce_proxy(topology: ClusterTopology, size: int) -> Prediction:
    """Ring proxy for the opaque NCCL auto-selection baseline."""

    link = topology.inter_node if topology.nodes > 1 else topology.intra_node
    phase = _ring_allreduce(
        "world all-reduce", link, topology.world_size, size
    )
    return _prediction(
        Strategy.NCCL_ALLREDUCE,
        [phase],
        payload_bytes=size,
        runtime_supported=True,
        assumption=(
            "ring critical-path proxy; the runtime delegates actual algorithm and "
            "protocol selection to NCCL"
        ),
    )


def hierarchical_leader(topology: ClusterTopology, size: int) -> Prediction:
    phases = [
        _tree_phase(
            "intra-node reduce",
            topology.intra_node,
            topology.gpus_per_node,
            size,
        ),
        _ring_allreduce(
            "leader inter-node all-reduce",
            topology.inter_node,
            topology.nodes,
            size,
        ),
        _tree_phase(
            "intra-node broadcast",
            topology.intra_node,
            topology.gpus_per_node,
            size,
        ),
    ]
    return _prediction(
        Strategy.HIERARCHICAL_LEADER,
        phases,
        payload_bytes=size,
        runtime_supported=True,
        assumption="one network-facing leader per node; favors latency over bandwidth",
    )


def _lane_link(topology: ClusterTopology) -> LinkTier:
    # Each same-local-rank lane gets an equal share of the usable network rails.
    lane_share = topology.rail_parallelism / topology.gpus_per_node
    return replace(
        topology.inter_node,
        name=f"{topology.inter_node.name}:rail-share",
        bandwidth_gbps=topology.inter_node.bandwidth_gbps * lane_share,
    )


def hierarchical_rail(topology: ClusterTopology, size: int) -> Prediction:
    shard_size = size / topology.gpus_per_node
    phases = [
        _ring_reduce_scatter(
            "intra-node reduce-scatter",
            topology.intra_node,
            topology.gpus_per_node,
            size,
        ),
        _ring_allreduce(
            "same-local-rank inter-node all-reduce",
            _lane_link(topology),
            topology.nodes,
            shard_size,
        ),
        _ring_allgather(
            "intra-node all-gather",
            topology.intra_node,
            topology.gpus_per_node,
            size,
        ),
    ]
    return _prediction(
        Strategy.HIERARCHICAL_RAIL,
        phases,
        payload_bytes=size,
        runtime_supported=True,
        assumption=(
            "same-local-rank lanes run concurrently and share measured rail bandwidth"
        ),
    )


def pipelined_rail(
    topology: ClusterTopology,
    size: int,
    *,
    chunks: int = 8,
) -> Prediction:
    if chunks < 2:
        raise ValueError("pipelined strategy requires at least 2 chunks")
    chunk_size = size / chunks
    shard_size = chunk_size / topology.gpus_per_node
    rs = _ring_reduce_scatter(
        "chunk reduce-scatter",
        topology.intra_node,
        topology.gpus_per_node,
        chunk_size,
    )
    inter = _ring_allreduce(
        "chunk inter-node all-reduce",
        _lane_link(topology),
        topology.nodes,
        shard_size,
    )
    ag = _ring_allgather(
        "chunk all-gather",
        topology.intra_node,
        topology.gpus_per_node,
        chunk_size,
    )
    first_chunk_us = rs.duration_us + inter.duration_us + ag.duration_us
    steady_stage_us = max(rs.duration_us, inter.duration_us, ag.duration_us)
    pipeline = Phase(
        name=f"{chunks}-chunk three-stage pipeline",
        link="overlapped intra/inter streams",
        duration_us=first_chunk_us + (chunks - 1) * steady_stage_us,
        algorithmic_bytes_per_rank=chunks
        * (
            rs.algorithmic_bytes_per_rank
            + inter.algorithmic_bytes_per_rank
            + ag.algorithmic_bytes_per_rank
        ),
        steps=chunks,
    )
    return _prediction(
        Strategy.PIPELINED_RAIL,
        [pipeline],
        payload_bytes=size,
        runtime_supported=False,
        assumption=(
            "ideal steady-state overlap on independent streams; design target, not "
            "enabled in the reference runtime"
        ),
    )


def predict_all(
    topology: ClusterTopology,
    size: int,
    *,
    pipeline_chunks: int = 8,
) -> tuple[Prediction, ...]:
    if size <= 0:
        raise ValueError("message size must be positive")
    return (
        nccl_allreduce_proxy(topology, size),
        hierarchical_leader(topology, size),
        hierarchical_rail(topology, size),
        pipelined_rail(topology, size, chunks=pipeline_chunks),
    )
