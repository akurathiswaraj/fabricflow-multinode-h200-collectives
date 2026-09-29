# H200 v0.7 benchmark methodology

The measured question is whether two explicit hierarchical all-reduce
schedules outperform native NCCL on a two-node, 16-GPU H200 RoCE/GDRDMA
cluster. The three executable paths compute the same global FP32 sum. A
pipelined design remains model-only and is not reported as a hardware result.

## Sequence and workload control

1. Run a separate 90-case correctness matrix across three strategies,
   FP32/FP16/BF16, five tensor lengths including uneven sizes, and two
   rank-varying deterministic repetitions.
2. Smoke-test 1 MiB and 64 MiB with five warmups and ten timed iterations.
3. Sweep 11 payload sizes (1 KiB to 1 GiB), 20 warmups, 100 timed iterations,
   and three rotated strategy orders on one cluster session. Validate sampled
   output every timed iteration and the complete final tensor.
4. Run a synthetic DDP-style gate using identical model, input generation,
   1 MiB gradient buckets, optimizer, five warmups, and 30 timed steps for
   each strategy and repeat. Repeat three times in rotated order. Only the
   bucket collective changes.

The final sweep and application mode used separate cluster sessions, so a
microbenchmark win alone would not constitute a workload win. The v0.7
measurements are in [the result report](results_h200_v07.md).

## Timing definition

For every collective iteration, each rank records a CUDA-event duration.
The row's completion sample is the **maximum of the 16 rank durations for
that iteration**. The stored p50/p95/max are calculated from those 100
completion samples, not from pooled rank samples. The public JSONL retains
the full 16 × 100 timing matrix and 100 completion samples per row; the
auditor recomputes the summaries. The headline value for each payload and
strategy is the median of three repeat-level p50s.

The application uses synchronized per-rank wall durations around each
compute, backward, collective, and optimizer step. Its completion sample is
the maximum rank duration for the corresponding step. The gate compares the
median of three repeat-level p95 step times. This is not a production LLM
benchmark and does not isolate communication time inside each step.

The historical v0.6 sweep used pooled rank timing samples. It is preserved
separately but **must not be numerically combined** with v0.7's slowest-rank
samples.

## Validity and promotion

An accepted row must be hardware-labeled, have two distinct provider node
identities with eight H200 ranks each, pass numerical correctness, and expose
raw timing samples that reproduce its summary. The private audit checks the
source hashes recorded by both nodes, all 369 evidence-file checksums, and
352 rank logs. Every launch on both nodes must show `Using network IB`,
`NET/IB/.../GDRDMA`, and no `NET/Socket` data channels. The device inventory
identifies Ethernet link-layer RoCE. NCCL's `IB` label does **not** imply
physical InfiniBand.

A custom path is promoted only after three correct application repeats,
equivalent final loss, and at least 5% improvement in median repeat-level
p95 step time against NCCL. No custom strategy met that gate. This negative
result applies to the recorded workload and topology, not all deployments.

## Limits

The three repeats of each mode are within one cluster session rather than
independent node provisions. Verbose NCCL logs can perturb timing. There is
no Nsight-backed causal attribution, per-NIC lane traffic measurement,
cross-rack placement comparison, fabric contention sweep, failure injection,
or production-model workload. The alpha-beta analytical model was not
calibrated with H200 link measurements. Logical communication lanes do not
prove independent physical rails.
