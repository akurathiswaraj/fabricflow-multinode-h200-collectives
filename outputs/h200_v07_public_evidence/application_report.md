# FabricFlow held-out application gate

> Promotion requires three correct repeats, equivalent final loss, and at least 5.0% median p95 step-time improvement.

## Run context

- Nodes: 2
- GPUs per node: 8
- World size: 16
- GPU: NVIDIA H200
- Transport: rdma_requested
- Gradient bytes: 33587200
- Bucket bytes: 1048576
- Timing semantics: per_step_max_rank_wall_duration_us

## Promotion decision

| Strategy | Repeats | Median p50 (ms) | Median p95 (ms) | p95 speedup | Max relative loss delta | Correct | Promoted |
|---|---:|---:|---:|---:|---:|---|---|
| hierarchical_leader | 3 | 4.934 | 9.718 | 0.946x | 0.000e+00 | yes | no |
| hierarchical_rail | 3 | 6.002 | 11.380 | 0.808x | 0.000e+00 | yes | no |
| nccl_allreduce | 3 | 4.034 | 9.194 | 1.000x | 0.000e+00 | yes | no |

## Outcome

- NCCL median p50: 4.034 ms.
- NCCL median p95: 9.194 ms.
- Promoted strategies: none
- This decision applies only to the recorded workload, topology, and transport.
