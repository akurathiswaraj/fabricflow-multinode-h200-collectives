# FabricFlow: multi-node H200 collective communication

FabricFlow is a measured performance-engineering study of three all-reduce
schedules on **two full NVIDIA H200 nodes (16 GPUs)** over observed
**RoCE / GPUDirect RDMA**. It implements two explicit hierarchical schedules,
compares them with native NCCL, verifies numerical correctness, audits the
network transport, and applies a held-out synthetic training-step gate.

**Decision: retain NCCL.** Across 99 correct final microbenchmark rows (11
payloads, three strategies, three rotated repeats), NCCL had the lowest
median p50 at all 11 sizes. In nine correct application runs, neither custom
schedule met the predeclared 5% p95 improvement gate. The contribution is an
auditable design-and-measurement process, not a fabricated speedup.

| Payload | NCCL p50 | Leader p50 | Rail p50 |
|---:|---:|---:|---:|
| 1 KiB | 68.22 µs | 107.20 µs | 161.22 µs |
| 1 MiB | 106.11 µs | 142.75 µs | 156.38 µs |
| 64 MiB | 479.30 µs | 2,515.14 µs | 745.82 µs |
| 1 GiB | 4,392.45 µs | 29,731.04 µs | 9,195.39 µs |

The table uses the median of three repeat-level p50s. Each repeat-level
sample is the **slowest rank's CUDA-event duration in that iteration**. The
complete table, p95 caveats, and application outcome are in the
[H200 v0.7 result report](docs/results_h200_v07.md).

## What is implemented

- Native NCCL AllReduce, local-reduce / node-leader AllReduce /
  local-broadcast, and local reduce-scatter / cross-node lane AllReduce /
  local all-gather. The named lanes are logical process groups; their
  existence does not establish physical NIC affinity.
- A 90-case independent correctness suite covering FP32, FP16, BF16,
  rank-varying inputs, and uneven tensor lengths; 90/90 passed on 16 H200s.
- A 99-row hardware sweep with saved per-rank/per-iteration timing arrays,
  slowest-rank p50/p95/max, three rotated repeats, and sampled correctness on
  every timed iteration plus a full final check.
- A synthetic DDP-style application gate with matched gradient bucketing,
  three repeats per strategy, loss equivalence, and parameter-spread checks.
  NCCL had 4.034 ms median p50 and 9.194 ms median p95 step time; no custom
  strategy was promoted.
- A fail-closed private-evidence audit that checks 369 file hashes, matches
  source hashes to the code executed on both nodes, recomputes timings, and
  examines all 352 NCCL rank logs. Every launch showed `NET/IB/.../GDRDMA`
  and no Socket data channels. The mlx5 inventory identifies Ethernet/RoCE,
  **not physical InfiniBand**.

## Inspect without GPUs

The checked-in [public evidence](outputs/h200_v07_public_evidence/README.md)
preserves the measurement arrays and outcomes but removes provider addresses,
cluster IDs, NIC GUIDs, and raw logs. It records the SHA-256 of the private
bundle and an audit summary of its transport markers. Run:

```bash
python -m unittest discover -s tests -v
python -m fabricflow.h200_v07_evidence outputs/h200_v07_public_evidence --public-directory
```

If you have the private archive, run the stronger audit (do not commit that
archive or publish the unredacted logs):

```bash
python -m fabricflow.h200_v07_evidence PRIVATE_BUNDLE.zip --project-root .
```

The executed project modules were hashed by each Modal node and compared with
the source in this release. `pyproject.toml` retains the `0.7.0.dev0` version
string that was actually executed; the `v0.7` release label marks completion
of hardware validation, not a code change to that hashed file.

## Reproduce on a two-node H200 cluster

The launcher requests `2 × H200:8`, `rdma=True`, a persistent results volume,
and a one-hour function timeout. These commands allocate **16 paid H200 GPUs**;
use unique run IDs and inspect each result before starting the next:

```bash
modal run modal_fabricflow.py --mode correctness --run-id YOUR_CORRECTNESS_ID
modal run modal_fabricflow.py --mode smoke --run-id YOUR_SMOKE_ID
modal run modal_fabricflow.py --mode final --run-id YOUR_FINAL_ID
modal run modal_fabricflow.py --mode application --run-id YOUR_APPLICATION_ID
```

See the [run plan](docs/h200_v07_validation_plan.md),
[methodology](docs/benchmark_methodology.md),
[architecture](docs/architecture.md), and
[operations runbook](docs/operations_runbook.md).

## Scope and limitations

The result is one H200 topology, four separate cluster sessions, and a
synthetic application workload. Three rotated repeats per mode were within
one session, not independent cluster provisions. It does not show a custom
H200 speedup, production LLM throughput gain, physical multirail utilization,
more-node scaling, fault tolerance, or a profiler-proven cause of NCCL's
advantage. The analytical alpha-beta model is not calibrated on this H200
fabric; the separate empirical policy is topology-locked and falls back to
NCCL outside the observed payload grid.

An earlier pooled-rank H200 campaign is kept separately as historical
evidence. Its timing statistic is not comparable to the v0.7 slowest-rank
statistic and is intentionally excluded from this release's result table.
