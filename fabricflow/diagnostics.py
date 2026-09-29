"""Read-only preflight checks for common multi-node launch failures."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any


def collect_diagnostics() -> dict[str, Any]:
    environment = {
        name: os.environ.get(name)
        for name in (
            "RANK",
            "WORLD_SIZE",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "NCCL_SOCKET_IFNAME",
            "NCCL_IB_HCA",
            "NCCL_NET",
            "NCCL_DEBUG",
            "NCCL_DEBUG_SUBSYS",
            "NCCL_RAS_ENABLE",
            "NCCL_RAS_ADDR",
        )
    }
    checks: list[dict[str, object]] = []
    try:
        import torch

        checks.extend(
            [
                {
                    "name": "torch_import",
                    "ok": True,
                    "detail": torch.__version__,
                },
                {
                    "name": "cuda_available",
                    "ok": bool(torch.cuda.is_available()),
                    "detail": torch.version.cuda,
                },
                {
                    "name": "nccl_available",
                    "ok": bool(torch.distributed.is_nccl_available()),
                    "detail": (
                        torch.cuda.nccl.version() if torch.cuda.is_available() else None
                    ),
                },
            ]
        )
    except ImportError as error:
        checks.append({"name": "torch_import", "ok": False, "detail": str(error)})

    launch_keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    launch_values = [environment[key] for key in launch_keys]
    checks.append(
        {
            "name": "torchrun_rank_environment",
            "ok": all(value is not None for value in launch_values),
            "detail": "present" if all(launch_values) else "run under torchrun",
        }
    )
    if all(value is not None for value in launch_values):
        rank = int(environment["RANK"] or 0)
        world = int(environment["WORLD_SIZE"] or 0)
        local_rank = int(environment["LOCAL_RANK"] or 0)
        local_world = int(environment["LOCAL_WORLD_SIZE"] or 0)
        mapping_ok = (
            world > 0
            and local_world > 0
            and world % local_world == 0
            and rank % local_world == local_rank
        )
        checks.append(
            {
                "name": "contiguous_rank_mapping",
                "ok": mapping_ok,
                "detail": "required by hierarchical subgroup construction",
            }
        )
    return {"checks": checks, "environment": environment}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FabricFlow cluster preflight")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    result = collect_diagnostics()
    print(json.dumps(result, indent=2, default=str))
    failed = [item for item in result["checks"] if not item["ok"]]
    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

