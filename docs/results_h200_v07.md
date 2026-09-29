# H200 v0.7: two-node RoCE/GDRDMA collective study

## Decision

**Retain native NCCL AllReduce on this measured topology.** FabricFlow's two
hierarchical alternatives were numerically correct, but NCCL had the lowest
median p50 at all 11 payload sizes and neither alternative passed the
predeclared synthetic-training p95 promotion gate. The project demonstrates
the design, execution, and rejection of explicit multi-node schedules—not a
custom-collective speedup.

## Hardware and evidence

- Modal cluster: two distinct nodes, eight NVIDIA H200 GPUs per node, 16 ranks.
- Software: PyTorch `2.13.0+cu130`, CUDA `13.0`, NCCL `2.29.7`.
- Observed data path: NCCL `Using network IB` and `NET/IB/.../GDRDMA` over
  mlx5 devices whose inventory identifies Ethernet link-layer RoCE. The
  independent private-bundle audit examined **352 rank logs** across the
  correctness, smoke, final, and application runs. Every launch on both nodes
  showed IB/GDRDMA channels and no Socket data channels.
- `IB` is NCCL's transport label; this is **not** a claim of physical
  InfiniBand. Eight exposed NICs per node do not prove that FabricFlow's eight
  logical lanes each used a separate NIC.
- Four run IDs: `h200-v07-correctness-001`, `h200-v07-smoke-001`,
  `h200-v07-final-001`, and `h200-v07-application-001`. Each has a separate
  source-hash manifest and two node inventories. The private archive's 369
  files pass SHA-256 checks, and both nodes' executed-source hashes match the
  source published with this release.

The [public evidence](../outputs/h200_v07_public_evidence/README.md) retains
the raw *timing arrays and outcomes* but redacts provider node addresses and
cluster identifiers. It includes an audit summary of the private NCCL logs,
not the logs themselves. Its `provenance.json` records the SHA-256 of the
private evidence archive. Use the private auditor when the archive is
available; the public auditor checks the redacted measurements and checksums:

```bash
python -m fabricflow.h200_v07_evidence outputs/h200_v07_public_evidence --public-directory
python -m fabricflow.h200_v07_evidence PRIVATE_BUNDLE.zip --project-root .
```

## Correctness and microbenchmark

The independent correctness suite passed **90/90 cases**: three strategies,
FP32/FP16/BF16, five tensor lengths including uneven lengths, and two
rank-varying deterministic random repetitions. The smoke run passed 6/6 rows;
the final sweep passed **99/99 rows** with maximum absolute error 0.0.

The final sweep measured 11 binary payload sizes, 20 warmups and 100 timed
iterations per row, with three repeats in rotated strategy order on one
cluster session. For each iteration, the reported completion time is the
**maximum CUDA-event duration among all 16 ranks**. Per-row p50/p95/max are
recomputed from these 100 maxima; the table is the median of the three
per-repeat p50 values. The legacy v0.6 pooled-rank statistic must not be
merged with this one.

| Payload | NCCL p50 (µs) | Leader p50 (µs) | Rail p50 (µs) |
|---:|---:|---:|---:|
| 1 KiB | 68.22 | 107.20 | 161.22 |
| 4 KiB | 67.68 | 108.32 | 161.31 |
| 16 KiB | 73.63 | 113.47 | 150.24 |
| 64 KiB | 85.89 | 122.18 | 148.22 |
| 256 KiB | 94.46 | 137.31 | 162.53 |
| 1 MiB | 106.11 | 142.75 | 156.38 |
| 4 MiB | 134.85 | 378.37 | 245.63 |
| 16 MiB | 214.34 | 1,011.46 | 279.65 |
| 64 MiB | 479.30 | 2,515.14 | 745.82 |
| 256 MiB | 1,299.97 | 7,941.18 | 2,478.62 |
| 1 GiB | 4,392.45 | 29,731.04 | 9,195.39 |

These are within-session microbenchmark results, **not** end-to-end training
speedups. Some p95 measurements have substantial outliers; this table is
deliberately p50-only. The full per-iteration arrays and p95 values are in
`final.jsonl`.

## Held-out synthetic training-step gate

The application gate used a deterministic, synthetic DDP-style workload:
eight width-1024 linear layers with GELU activations (8,396,800 parameters,
33,587,200 FP32 gradient bytes), 1 MiB gradient buckets, global batch 2,048,
five warmup steps, and 30 measured steps per repeat. All three strategies had
three rotated repeats. The same forward, backward, bucketing, and optimizer
path was used; only the collective for each bucket changed. Each step's time
is the maximum wall duration over 16 ranks. All nine runs were correct, had
zero measured parameter spread, and matched final loss.

| Strategy | Median p50 step (ms) | Median p95 step (ms) | p95 speedup vs NCCL | Promoted? |
|---|---:|---:|---:|---|
| NCCL AllReduce | 4.034 | 9.194 | 1.000× | baseline |
| Hierarchical leader | 4.934 | 9.718 | 0.946× | no |
| Hierarchical rail | 6.002 | 11.380 | 0.808× | no |

Promotion required three correct repeats, equivalent final loss, and at least
5% improvement in the median of repeat-level p95 step times. **No custom
strategy was promoted.** This is a synthetic training-like gate, not a
frontier-model or production training benchmark. The measured step samples
include compute and optimizer work; the reported medians should not be
generalized beyond this workload.

## What this does and does not establish

The evidence establishes working multi-node collective implementations,
numerical correctness across dtypes and uneven sizes, transport-confirmed
RoCE/GDRDMA execution, audited slowest-rank latency measurements, and a
negative application-level decision on one H200 cluster configuration.

It does **not** establish NIC affinity per lane, sustained multirail
utilization, a causal profiler explanation for NCCL's advantage, performance
under contention, more-node scaling, fault recovery, or a production LLM
training speedup. Nsight traces and NIC counters would be needed for a
stronger causal claim. Three repeats occurred in one cluster session per mode;
they are not independent cluster provisions. The analytical alpha-beta model
was not calibrated for this H200 fabric; the separate empirical H200 policy is
topology-locked and falls back to NCCL outside observed sizes.
