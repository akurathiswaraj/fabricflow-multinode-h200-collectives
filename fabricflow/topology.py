"""Validated cluster topology inputs for the analytical planner.

The model deliberately consumes measured or operator-supplied link properties.
It does not encode vendor peak claims as if they were achieved bandwidth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class LinkTier:
    """A link tier on the critical path.

    ``bandwidth_gbps`` is decimal gigabits/s. ``efficiency`` captures protocol,
    software, and topology loss. ``contention`` is a multiplier where 1.0 means
    no oversubscription and 2.0 halves the available bandwidth.
    """

    name: str
    bandwidth_gbps: float
    latency_us: float
    efficiency: float = 0.80
    contention: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("link tier name must be non-empty")
        if self.bandwidth_gbps <= 0:
            raise ValueError("bandwidth_gbps must be positive")
        if self.latency_us < 0:
            raise ValueError("latency_us cannot be negative")
        if not 0 < self.efficiency <= 1:
            raise ValueError("efficiency must be in (0, 1]")
        if self.contention < 1:
            raise ValueError("contention must be >= 1")

    @property
    def effective_bytes_per_second(self) -> float:
        return self.bandwidth_gbps * 1e9 / 8 * self.efficiency / self.contention

    def transfer_us(self, byte_count: float, *, steps: int = 1) -> float:
        if byte_count < 0:
            raise ValueError("byte_count cannot be negative")
        if steps < 0:
            raise ValueError("steps cannot be negative")
        return steps * (
            self.latency_us + byte_count / self.effective_bytes_per_second * 1e6
        )


@dataclass(frozen=True)
class ClusterTopology:
    name: str
    nodes: int
    gpus_per_node: int
    rails_per_node: int
    intra_node: LinkTier
    inter_node: LinkTier
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("topology name must be non-empty")
        if self.nodes < 1:
            raise ValueError("nodes must be >= 1")
        if self.gpus_per_node < 1:
            raise ValueError("gpus_per_node must be >= 1")
        if self.rails_per_node < 1:
            raise ValueError("rails_per_node must be >= 1")
        if self.rails_per_node > self.gpus_per_node:
            raise ValueError("rails_per_node cannot exceed gpus_per_node")

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node

    @property
    def rail_parallelism(self) -> int:
        return min(self.rails_per_node, self.gpus_per_node)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ClusterTopology":
        required = {
            "name",
            "nodes",
            "gpus_per_node",
            "rails_per_node",
            "intra_node",
            "inter_node",
        }
        missing = required - raw.keys()
        if missing:
            raise ValueError(f"missing topology fields: {sorted(missing)}")
        return cls(
            name=str(raw["name"]),
            nodes=int(raw["nodes"]),
            gpus_per_node=int(raw["gpus_per_node"]),
            rails_per_node=int(raw["rails_per_node"]),
            intra_node=LinkTier(**dict(raw["intra_node"])),
            inter_node=LinkTier(**dict(raw["inter_node"])),
            metadata=dict(raw.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ClusterTopology":
        source = Path(path)
        with source.open("r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def topology_summary(topology: ClusterTopology) -> str:
    return (
        f"{topology.name}: {topology.nodes} nodes x "
        f"{topology.gpus_per_node} GPUs ({topology.world_size} ranks), "
        f"{topology.rails_per_node} network rails/node"
    )

