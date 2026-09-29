"""Reference PyTorch/NCCL implementations of the runtime-capable strategies.

The group creation order is intentionally global and deterministic. PyTorch
requires all ranks to create overlapping process groups in a consistent order;
violating that rule is a common source of multi-node hangs.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from .cost_model import Strategy


@dataclass(frozen=True)
class ProcessGroupLayout:
    rank: int
    world_size: int
    local_size: int

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if self.local_size < 1:
            raise ValueError("local_size must be positive")
        if self.world_size % self.local_size:
            raise ValueError("world_size must be divisible by local_size")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank is outside world_size")

    @property
    def nodes(self) -> int:
        return self.world_size // self.local_size

    @property
    def node_id(self) -> int:
        return self.rank // self.local_size

    @property
    def local_rank(self) -> int:
        return self.rank % self.local_size

    @property
    def node_leader(self) -> int:
        return self.node_id * self.local_size

    def local_ranks(self, node_id: int | None = None) -> list[int]:
        node = self.node_id if node_id is None else node_id
        if not 0 <= node < self.nodes:
            raise ValueError("node_id is outside cluster")
        start = node * self.local_size
        return list(range(start, start + self.local_size))

    def lane_ranks(self, local_rank: int | None = None) -> list[int]:
        lane = self.local_rank if local_rank is None else local_rank
        if not 0 <= lane < self.local_size:
            raise ValueError("local_rank is outside node")
        return [node * self.local_size + lane for node in range(self.nodes)]

    def leader_ranks(self) -> list[int]:
        return [node * self.local_size for node in range(self.nodes)]


def layout_from_env() -> ProcessGroupLayout:
    required = ("RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            f"missing torchrun environment variables: {', '.join(missing)}"
        )
    return ProcessGroupLayout(
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        local_size=int(os.environ["LOCAL_WORLD_SIZE"]),
    )


class CollectiveRuntime:
    """Owns deterministic subgroups and executes in-place SUM all-reduces."""

    def __init__(self, dist: Any, torch: Any, layout: ProcessGroupLayout) -> None:
        self.dist = dist
        self.torch = torch
        self.layout = layout
        self.local_group: Any | None = None
        self.lane_group: Any | None = None
        self.leader_group: Any | None = None
        self._create_groups()

    def _create_groups(self) -> None:
        # Every rank executes every new_group call in this exact order.
        for node_id in range(self.layout.nodes):
            ranks = self.layout.local_ranks(node_id)
            group = self.dist.new_group(ranks=ranks)
            if node_id == self.layout.node_id:
                self.local_group = group
        for local_rank in range(self.layout.local_size):
            ranks = self.layout.lane_ranks(local_rank)
            group = self.dist.new_group(ranks=ranks)
            if local_rank == self.layout.local_rank:
                self.lane_group = group
        self.leader_group = self.dist.new_group(ranks=self.layout.leader_ranks())

    def all_reduce(self, tensor: Any, strategy: Strategy | str) -> None:
        selected = Strategy(strategy)
        if selected == Strategy.NCCL_ALLREDUCE:
            self.dist.all_reduce(tensor, op=self.dist.ReduceOp.SUM)
        elif selected == Strategy.HIERARCHICAL_LEADER:
            self._leader_all_reduce(tensor)
        elif selected == Strategy.HIERARCHICAL_RAIL:
            self._rail_all_reduce(tensor)
        else:
            raise ValueError(f"strategy is not runtime-capable: {selected.value}")

    def _leader_all_reduce(self, tensor: Any) -> None:
        assert self.local_group is not None
        self.dist.reduce(
            tensor,
            dst=self.layout.node_leader,
            op=self.dist.ReduceOp.SUM,
            group=self.local_group,
        )
        if self.layout.local_rank == 0:
            self.dist.all_reduce(
                tensor,
                op=self.dist.ReduceOp.SUM,
                group=self.leader_group,
            )
        self.dist.broadcast(
            tensor,
            src=self.layout.node_leader,
            group=self.local_group,
        )

    def _rail_all_reduce(self, tensor: Any) -> None:
        assert self.local_group is not None
        assert self.lane_group is not None
        original = tensor.contiguous().view(-1)
        original_count = original.numel()
        remainder = original_count % self.layout.local_size
        if remainder:
            pad = self.layout.local_size - remainder
            working = self.torch.cat(
                [
                    original,
                    self.torch.zeros(pad, dtype=original.dtype, device=original.device),
                ]
            )
        else:
            working = original
        shard = self.torch.empty(
            working.numel() // self.layout.local_size,
            dtype=working.dtype,
            device=working.device,
        )
        reduce_scatter = getattr(self.dist, "reduce_scatter_single", None)
        if reduce_scatter is None:  # PyTorch < 2.13 compatibility.
            reduce_scatter = self.dist.reduce_scatter_tensor
        reduce_scatter(
            shard,
            working,
            op=self.dist.ReduceOp.SUM,
            group=self.local_group,
        )
        self.dist.all_reduce(
            shard,
            op=self.dist.ReduceOp.SUM,
            group=self.lane_group,
        )
        gathered = self.torch.empty_like(working)
        gather = getattr(self.dist, "all_gather_single", None)
        if gather is None:  # PyTorch < 2.13 compatibility.
            gather = self.dist.all_gather_into_tensor
        gather(
            gathered,
            shard,
            group=self.local_group,
        )
        tensor.copy_(gathered[:original_count].view_as(tensor))


def initialize_nccl() -> tuple[Any, Any, ProcessGroupLayout, CollectiveRuntime]:
    try:
        import torch
        import torch.distributed as dist
    except ImportError as error:
        raise RuntimeError(
            "PyTorch is required for the NCCL runtime; install the 'runtime' extra"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    layout = layout_from_env()
    local_rank = int(os.environ.get("LOCAL_RANK", layout.local_rank))
    if local_rank != layout.local_rank:
        raise RuntimeError(
            "LOCAL_RANK does not match rank modulo LOCAL_WORLD_SIZE; "
            "use contiguous rank placement"
        )
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device("cuda", local_rank),
    )
    runtime = CollectiveRuntime(dist, torch, layout)
    return torch, dist, layout, runtime
