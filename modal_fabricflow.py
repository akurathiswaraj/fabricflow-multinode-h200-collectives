"""Modal launcher for FabricFlow multi-node NCCL experiments.

This file intentionally uses Modal's clustered Function API rather than a
Notebook GPU kernel. All modes share the same H200/RDMA hardware shape.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import modal
import modal.experimental


APP_NAME = "fabricflow-multinode"
NODES = 2
GPUS_PER_NODE = 8
REMOTE_PROJECT = "/opt/fabricflow"
RESULTS_ROOT = "/results"

project_root = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("pciutils", "ibverbs-utils")
    .pip_install("torch==2.13.0", "numpy==2.3.3")
    .add_local_dir(
        str(project_root / "fabricflow"),
        remote_path=f"{REMOTE_PROJECT}/fabricflow",
        copy=True,
    )
    .add_local_dir(
        str(project_root / "configs"),
        remote_path=f"{REMOTE_PROJECT}/configs",
        copy=True,
    )
    .add_local_file(
        str(project_root / "pyproject.toml"),
        remote_path=f"{REMOTE_PROJECT}/pyproject.toml",
        copy=True,
    )
)

app = modal.App(APP_NAME, image=image)
results_volume = modal.Volume.from_name(
    "fabricflow-results",
    create_if_missing=True,
)


def _source_hashes() -> dict[str, str]:
    names = (
        "fabricflow/benchmark.py",
        "fabricflow/runtime.py",
        "fabricflow/correctness.py",
        "fabricflow/application_benchmark.py",
        "fabricflow/application_report.py",
        "fabricflow/h200_policy.py",
        "pyproject.toml",
    )
    return {
        name: hashlib.sha256((Path(REMOTE_PROJECT) / name).read_bytes()).hexdigest()
        for name in names
    }


def _capture(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def _inventory(cluster_info: object, torch: object) -> dict[str, object]:
    nccl_version = torch.cuda.nccl.version()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provider": "modal",
        "cluster_id": cluster_info.cluster_id,
        "container_rank": cluster_info.rank,
        "container_ip": cluster_info.container_ips[cluster_info.rank],
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nccl": (
            ".".join(str(value) for value in nccl_version)
            if isinstance(nccl_version, (tuple, list))
            else str(nccl_version)
        ),
        "gpu_count": torch.cuda.device_count(),
        "gpu_names": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "nvidia_smi": _capture(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "local_topology": _capture(["nvidia-smi", "topo", "-m"]),
        "rdma_devices": _capture(["ibv_devices"]),
        "rdma_device_info": _capture(["ibv_devinfo"]),
        "source_sha256": _source_hashes(),
    }


def _write_manifest(run_id: str, mode: str, output_path: str) -> str:
    path = Path(RESULTS_ROOT) / f"{run_id}.manifest.json"
    data = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "mode": mode,
        "provider": "modal",
        "app_name": APP_NAME,
        "nodes": NODES,
        "gpus_per_node": GPUS_PER_NODE,
        "gpu_request": "H200:8",
        "rdma_requested": True,
        "results_volume": "fabricflow-results",
        "primary_output": output_path,
        "source_sha256": _source_hashes(),
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return str(path)


def _benchmark_plan(mode: str) -> tuple[str, int, int, list[list[str]]]:
    strategies = [
        "nccl_allreduce",
        "hierarchical_leader",
        "hierarchical_rail",
    ]
    if mode == "smoke":
        return "1MiB,64MiB", 5, 10, [strategies]
    if mode == "final":
        sizes = "1KiB,4KiB,16KiB,64KiB,256KiB,1MiB,4MiB,16MiB,64MiB,256MiB,1GiB"
        return sizes, 20, 100, [
            strategies,
            ["hierarchical_rail", "nccl_allreduce", "hierarchical_leader"],
            ["hierarchical_leader", "hierarchical_rail", "nccl_allreduce"],
        ]
    raise ValueError("mode must be 'smoke' or 'final'")


def _worker_environment(cluster_info: object, run_id: str, repeat_id: int, launch_number: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": REMOTE_PROJECT,
            "PYTHONUNBUFFERED": "1",
            "FABRICFLOW_PROVIDER": "modal",
            "FABRICFLOW_EXPERIMENT_SCOPE": "multi_node_rdma",
            "FABRICFLOW_NETWORK_TRANSPORT": "rdma_requested",
            "FABRICFLOW_RUN_ID": run_id,
            "FABRICFLOW_REPEAT_ID": str(repeat_id),
            "FABRICFLOW_CLUSTER_ID": cluster_info.cluster_id,
            "FABRICFLOW_NODE_ID": cluster_info.container_ips[cluster_info.rank],
            "FABRICFLOW_NODE_RANK": str(cluster_info.rank),
            "FABRICFLOW_REQUIRE_HOMOGENEOUS": "1",
            "FABRICFLOW_EXPECTED_NODES": str(NODES),
            "NCCL_RAS_ENABLE": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,NET,RAS",
            "NCCL_DEBUG_FILE": (
                f"{RESULTS_ROOT}/{run_id}.node-{cluster_info.rank}."
                f"launch-{launch_number}.nccl.%h.%p.log"
            ),
        }
    )
    return environment


def _torchrun_prefix(cluster_info: object, launch_number: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={NODES}",
        f"--node-rank={cluster_info.rank}",
        f"--master-addr={cluster_info.container_ips[0]}",
        f"--master-port={29500 + launch_number}",
        f"--nproc-per-node={GPUS_PER_NODE}",
        "--module",
    ]


@app.function(
    gpu="H200:8",
    timeout=60 * 60,
    volumes={RESULTS_ROOT: results_volume},
)
@modal.experimental.clustered(size=NODES, rdma=True)
def run_cluster(mode: str, run_id: str) -> dict[str, object]:
    import torch

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("run_id may contain only letters, numbers, dot, dash, underscore")
    cluster_info = modal.experimental.get_cluster_info()
    if torch.cuda.device_count() != GPUS_PER_NODE:
        raise RuntimeError(
            f"expected {GPUS_PER_NODE} GPUs per node, got {torch.cuda.device_count()}"
        )

    inventory_path = Path(RESULTS_ROOT) / (
        f"{run_id}.inventory.node-{cluster_info.rank}.json"
    )
    inventory_path.write_text(
        json.dumps(_inventory(cluster_info, torch), indent=2) + "\n",
        encoding="utf-8",
    )
    results_volume.commit()

    if mode == "correctness":
        environment = _worker_environment(cluster_info, run_id, 1, 1)
        output_path = f"{RESULTS_ROOT}/{run_id}.correctness.json"
        command = _torchrun_prefix(cluster_info, 1) + [
            "fabricflow.correctness", "--output", output_path
        ]
        subprocess.run(command, check=True, cwd=REMOTE_PROJECT, env=environment)
        if cluster_info.rank == 0:
            _write_manifest(run_id, mode, output_path)
        results_volume.commit()
        return {"run_id": run_id, "mode": mode, "output": output_path}

    if mode == "application":
        output_path = f"{RESULTS_ROOT}/{run_id}.application.jsonl"
        orders = [
            ["nccl_allreduce", "hierarchical_leader", "hierarchical_rail"],
            ["hierarchical_rail", "nccl_allreduce", "hierarchical_leader"],
            ["hierarchical_leader", "hierarchical_rail", "nccl_allreduce"],
        ]
        launch_number = 0
        for repeat_id, order in enumerate(orders, start=1):
            for strategy in order:
                launch_number += 1
                environment = _worker_environment(
                    cluster_info, run_id, repeat_id, launch_number
                )
                command = _torchrun_prefix(cluster_info, launch_number) + [
                    "fabricflow.application_benchmark",
                    "--strategy", strategy,
                    "--warmup", "5",
                    "--steps", "30",
                    "--bucket-bytes", "1MiB",
                    "--output", output_path,
                ]
                subprocess.run(command, check=True, cwd=REMOTE_PROJECT, env=environment)
        report_path = f"{RESULTS_ROOT}/{run_id}.application.report.md"
        if cluster_info.rank == 0:
            subprocess.run(
                [sys.executable, "-m", "fabricflow.application_report",
                 "--input", output_path, "--output", report_path],
                check=True, cwd=REMOTE_PROJECT,
                env=_worker_environment(cluster_info, run_id, 3, launch_number),
            )
            _write_manifest(run_id, mode, output_path)
        results_volume.commit()
        return {"run_id": run_id, "mode": mode, "output": output_path,
                "report": report_path}

    sizes, warmup, iterations, orders = _benchmark_plan(mode)
    output_path = f"{RESULTS_ROOT}/{run_id}.jsonl"
    launch_number = 0
    for repeat_id, order in enumerate(orders, start=1):
        for strategy in order:
            launch_number += 1
            environment = _worker_environment(
                cluster_info, run_id, repeat_id, launch_number
            )
            command = _torchrun_prefix(cluster_info, launch_number) + [
                "fabricflow.benchmark",
                "--strategy",
                strategy,
                "--sizes",
                sizes,
                "--warmup",
                str(warmup),
                "--iterations",
                str(iterations),
                "--output",
                output_path,
            ]
            subprocess.run(
                command,
                check=True,
                cwd=REMOTE_PROJECT,
                env=environment,
            )

    if cluster_info.rank == 0:
        _write_manifest(run_id, mode, output_path)
    results_volume.commit()
    return {
        "run_id": run_id,
        "mode": mode,
        "cluster_id": cluster_info.cluster_id,
        "nodes": NODES,
        "gpus_per_node": GPUS_PER_NODE,
        "output": output_path,
        "inventory": str(inventory_path),
    }


@app.local_entrypoint()
def main(mode: str = "smoke", run_id: str = "") -> None:
    selected_run_id = run_id or datetime.now(timezone.utc).strftime(
        "fabricflow-%Y%m%dT%H%M%SZ"
    )
    result = run_cluster.remote(mode, selected_run_id)
    print(json.dumps(result, indent=2))
