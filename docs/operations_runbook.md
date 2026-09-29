# Multi-node operations runbook

## Before launch

- Confirm every node has the same driver, CUDA, PyTorch, and NCCL versions.
- Confirm one process per GPU and contiguous global rank placement per node.
- Check GPU-to-NIC affinity and whether each intended rail is active.
- Verify `MASTER_ADDR` resolves from every node and `MASTER_PORT` is reachable.
- Run the strict diagnostic command under the same launcher environment.
- Start with NCCL defaults. Add tuning variables one at a time and record them.

Useful inventory commands vary by cluster, but commonly include:

```bash
nvidia-smi topo -m
nvidia-smi -q
ip -j link
ip -j addr
ibstat
ibv_devinfo
```

For the H200 experiment, preserve NCCL INFO logs that identify `Using network
IB` and cross-node `NET/IB/.../GDRDMA` channels. Confirm the devices are RoCE
from inventory. Do not infer the data path from RDMA device presence alone,
and do not describe NCCL's `IB` plugin label as physical InfiniBand.

## Recommended experiment logging

For a temporary diagnostic run:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH,NET,COLL,RAS
export NCCL_DEBUG_FILE="results/nccl.%h.%p.log"
```

Use a unique `%h.%p` log path so ranks do not overwrite each other. Remove debug
settings after the experiment; verbose logs and forced algorithm settings should
not silently become production defaults.

Current NCCL includes a RAS subsystem that can report communicator state and
unresponsive peers. When the installed NCCL version supports it, keep RAS
enabled and query the local RAS endpoint during a suspected hang. Restrict any
externally reachable endpoint because it exposes cluster diagnostic state.

## Hang triage

1. Confirm every rank entered the same collective with identical element count,
   dtype, operation, and group. Mismatches can hang or corrupt results.
2. Compare last progress timestamps across rank logs; identify the earliest rank
   that diverged, not merely the rank that timed out first.
3. Query RAS status for unresponsive processes and communicator state.
4. Inspect group-creation order. All ranks must create local, lane, and leader
   groups in the same global order.
5. Verify rank mapping and node boundaries match `LOCAL_WORLD_SIZE`.
6. Inspect NIC selection and logs for a silent fallback from RDMA to sockets.
7. Reproduce with `nccl_allreduce`; if it also fails, isolate fabric or launcher
   health before investigating the custom hierarchy.
8. Reduce to two nodes and one payload, then restore dimensions one at a time.

## Slow-run triage

- Compare p50 with maximum rank latency to detect stragglers.
- Check whether one rail carries most traffic while others are idle.
- Compare isolated and background-load runs for oversubscription.
- Inspect CUDA timelines for serialization between local and inter-node phases.
- Check whether communication kernels consume SM resources needed by compute.
- Refit measured efficiency and contention rather than forcing the model to fit
  through arbitrary algorithm constants.

## Safe fallback

The application-level integration should preserve `nccl_allreduce` as the
default and use a feature flag per communication bucket. On any unsupported
layout, subgroup initialization failure, failed correctness canary, or stale
calibration profile, disable the custom path and return to NCCL.
