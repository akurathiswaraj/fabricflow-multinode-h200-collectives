"""FabricFlow: topology-aware multi-node collective planning."""

from .cost_model import Prediction, Strategy
from .planner import Plan, Planner
from .topology import ClusterTopology, LinkTier

__all__ = [
    "ClusterTopology",
    "LinkTier",
    "Plan",
    "Planner",
    "Prediction",
    "Strategy",
]

__version__ = "0.7.0.dev0"
