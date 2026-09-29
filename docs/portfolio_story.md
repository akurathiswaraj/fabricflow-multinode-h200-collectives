# FabricFlow interview and résumé guide

## Thirty-second explanation

I built two explicit hierarchical all-reduce schedules and compared them with
native NCCL on two H200 nodes, 16 GPUs total. I verified three data types and
uneven tensor sizes, measured the slowest GPU per collective iteration, and
checked NCCL logs to confirm RoCE/GPUDirect RDMA rather than a Socket fallback.
NCCL won at all 11 payload sizes, and neither custom path improved p95 in a
held-out synthetic training-step test. My decision was to keep NCCL for this
topology. The engineering value is the validated design and disciplined
rejection of a slower alternative.

## Résumé bullets

Use the project title **FabricFlow — Multi-Node H200 Collective Performance**.
Choose one or two bullets, subject to space:

> Designed and implemented two hierarchical PyTorch/NCCL all-reduce schedules
> on a 2-node, 16×H200 RoCE/GDRDMA cluster; validated 90 correctness cases and
> 99 repeated hardware measurements across 11 payload sizes, using
> per-iteration slowest-rank latency and audited transport logs.

> Built a synthetic DDP-style application gate with matched gradient buckets,
> loss/parameter checks, and three rotated repeats per strategy; retained
> native NCCL after both custom schedules lost the predeclared p95 step-time
> promotion test.

Do **not** claim a custom speedup, production LLM throughput improvement,
physical InfiniBand, guaranteed independent NIC rails, or a causal profiler
explanation. The workload and topology scope matter in an interview.

## Technical walkthrough

1. State the invariant: every rank receives the same global elementwise sum.
   The leader schedule reduces locally, all-reduces between leaders, then
   broadcasts locally. The rail schedule reduce-scatters locally, all-reduces
   corresponding shards across nodes, and all-gathers locally.
2. Explain how provider-issued node identities validate two nodes with eight
   ranks each. Show why logical lane groups are **not** proof of per-NIC
   binding.
3. Describe the measurement fix: record every rank's CUDA duration and use
   the maximum rank duration for each iteration before computing p50/p95.
   Do not mix the earlier pooled-rank run with this statistic.
4. Show the 90-case FP32/FP16/BF16 and uneven-length matrix, 99-row final
   sweep, three rotated orders, and source-hash/checksum audit. NCCL logs show
   `NET/IB/.../GDRDMA`; node inventory identifies Ethernet/RoCE.
5. End with the held-out synthetic training-step gate: nine correct runs,
   equivalent loss, no custom p95 promotion. Then discuss the next questions:
   Nsight phase tracing, NIC counters, contention, and independent cluster
   provisions.

The [result report](results_h200_v07.md) contains all numbers and limits.
