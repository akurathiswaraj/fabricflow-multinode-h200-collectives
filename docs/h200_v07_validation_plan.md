# H200 v0.7 validation and reproduction plan

The original `h200-rdma-smoke-001` and `h200-rdma-final-001` evidence is
immutable. Its 99 rows use the older pooled-rank timing statistic. The updated
v0.7 correctness suite, benchmark, and synthetic training-step gate were
validated on H200 on 2026-09-16. Do not combine old and new timing rows into
one report. The verified outcome is documented in [results_h200_v07.md](results_h200_v07.md).

## Preflight without paid GPUs

In the extracted candidate source directory:

```bash
python -m unittest discover -s tests -v
python -m py_compile modal_fabricflow.py fabricflow/benchmark.py fabricflow/correctness.py fabricflow/application_benchmark.py
modal run modal_fabricflow.py --help
```

Verify the launcher still requests `2 × H200:8`, `rdma=True`, the persistent
`fabricflow-results` volume, and a one-hour timeout. Confirm the account's
limits and expected cost before executing any mode.

## Run sequence — stop after any failure

Use unique run IDs. Run one mode at a time and inspect its output before
starting the next. The commands below are for a Modal notebook shell; inside
a Python notebook code cell, prefix a command with `!`.

```bash
modal run modal_fabricflow.py --mode correctness --run-id h200-v07-correctness-001
modal run modal_fabricflow.py --mode smoke --run-id h200-v07-smoke-001
modal run modal_fabricflow.py --mode final --run-id h200-v07-final-001
modal run modal_fabricflow.py --mode application --run-id h200-v07-application-001
```

The correctness mode compares all three schedules against NCCL on 16 H200
ranks for FP32/FP16/BF16, uneven tensor lengths, and two deterministic random
repetitions. It must pass all 90 cases. The smoke mode checks the versioned
slowest-rank latency statistic, sampled correctness each timed iteration, and
a full final-iteration check at two sizes.
The final mode repeats the 11-size sweep three times in rotated order. The
application mode performs nine equally bucketed synthetic-training runs:
three repeats each for NCCL, leader, and rail. It reports p50/p95 of the
slowest-rank step duration, loss equivalence, and parameter spread. This is an
application-level *synthetic* gate, not production-model performance.

## Acceptance criteria

- Correctness file: 90/90 cases pass, including uneven sizes and all three
  dtypes; maximum absolute error zero for the bounded integer inputs.
- New microbenchmark rows: `schema_version=2`, 99/99 correct, two distinct
  provider-issued nodes with eight H200 ranks each; 100 rank timing samples
  per row, with recomputable per-iteration maxima and p50/p95.
- Logs: NCCL rank logs show `NET/IB/.../GDRDMA` on both nodes, no Socket data
  channels; inventories show H200 and mlx5 RoCE devices.
- Each mode's manifest and both node inventories include SHA-256 hashes of the
  executed project modules; the two nodes' hashes must agree.
- Application rows: nine `schema_version=2` rows; all correct, finite loss,
  zero or tolerably small cross-rank parameter spread, and matched model,
  bucket, batch, and step configurations. A custom strategy is promoted only
  if the held-out report explicitly passes its predeclared p95/loss gate.
- The four v0.7 runs were downloaded, independently audited, and included as
  redacted measurements in a checksummed public release. The unredacted
  evidence bundle remains private because the logs and inventories contain
  provider network identifiers.

## Optional diagnostic depth

If profiling tools are available in the H200 container, capture a short
Nsight Systems trace for the 1 MiB and 64 MiB controls. Inspect local
reduce-scatter/all-gather time, cross-node communication, launch gaps, NCCL
algorithm selection, and NIC traffic. Do not label this causal analysis until
traces or counters support it. An independent multi-node `nccl-tests`
`all_reduce_perf` sweep can sanity-check the baseline. Verbose NCCL logging
may perturb performance; a short repeat with debug logging disabled is useful
if the new timing result differs substantially from the original campaign.
