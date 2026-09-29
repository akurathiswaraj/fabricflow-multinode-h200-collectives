# Architecture and design record

## Scope

FabricFlow focuses on SUM all-reduce because it is both performance-critical in
data-parallel training and rich enough to demonstrate topology, scheduling,
correctness, and operational tradeoffs. The design is intentionally extensible
to reduce-scatter, all-gather, and all-to-all, but it does not claim support for
those standalone collectives in version 0.1.

Goals:

- choose among structurally different paths rather than only tune one algorithm;
- use measured topology inputs and keep every assumption inspectable;
- execute two hierarchical strategies through PyTorch/NCCL;
- produce correctness-checked, rank-aggregated, machine-readable results;
- learn conservative empirical corrections without hiding the base model;
- fail early on rank-layout and world-size mistakes.

Non-goals:

- replace NCCL's internal topology graph or tuner;
- claim performance on hardware that was not measured;
- implement fault recovery inside an in-flight collective;
- promise that an analytical winner will beat NCCL on every cluster.

## Cost model

For a link tier with startup latency `alpha`, effective bandwidth `B`, and a
transfer of `m` bytes per step, the critical-path estimate is:

```text
T(steps, m) = steps * (alpha + m / B)
B = line_rate * efficiency / contention
```

The ring phases use these standard critical-path volumes for `p` participants
and a tensor of `n` bytes:

```text
reduce-scatter: (p - 1) steps, n / p bytes per step
all-gather:     (p - 1) steps, n / p bytes per step
all-reduce:   2*(p - 1) steps, n / p bytes per step
```

The node-leader path models local reduce and broadcast as tree phases with
`ceil(log2(g))` steps over `g` local GPUs. The leader inter-node phase is a ring
across `q` nodes carrying the full tensor. It reduces message startups for tiny
payloads but constrains inter-node data to one leader lane.

The rail path has three phases:

1. ring reduce-scatter of `n` bytes across `g` local GPUs;
2. ring all-reduce of a `n/g` shard across `q` nodes for each local-rank lane;
3. ring all-gather of `n` bytes across `g` local GPUs.

If a node has `r` usable network rails, each of the `g` concurrent lanes receives
an analytical share `min(r, g) / g` of per-rail bandwidth. This conservative
constraint prevents the model from manufacturing bandwidth by creating more
communicators.

The pipelined design splits the tensor into `c` chunks. Its idealized makespan is:

```text
T_pipeline = T_rs(chunk) + T_inter(chunk) + T_ag(chunk)
             + (c - 1) * max(T_rs(chunk), T_inter(chunk), T_ag(chunk))
```

Because CUDA stream concurrency, communicator scheduling, and SM pressure can
invalidate ideal overlap, this strategy is model-only and excluded from default
selection.

## Deterministic subgroup construction

Assume contiguous ranks within each node:

```text
node_id    = rank // local_world_size
local_rank = rank %  local_world_size
```

For 4 nodes x 4 GPUs, rank 6 belongs to:

```text
local group: [4, 5, 6, 7]
lane group:  [2, 6, 10, 14]
leader group:[0, 4, 8, 12]
```

Every rank calls `new_group` for every local group in node order, every lane
group in local-rank order, and the leader group last. Only after this common
sequence does a rank retain the groups it belongs to. This ordering avoids a
class of group-creation deadlocks.

## Rail strategy correctness

Let rank `(i, j)` mean node `i`, local GPU `j`, and let its input tensor be
`x[i,j]`. Local reduce-scatter yields shard `j`:

```text
s[i,j] = shard_j(sum_k x[i,k])
```

The lane all-reduce produces:

```text
g[j] = sum_i s[i,j]
     = shard_j(sum_i sum_k x[i,k])
```

The local all-gather concatenates `g[0] ... g[g-1]`, so every rank receives the
global elementwise sum. If element count is not divisible by local GPU count,
the runtime zero-pads the tail before reduce-scatter and removes it after the
all-gather.

## Measurement and calibration

The historical H200 benchmark initialized rank `r` with the scalar `r + 1` and
pooled rank-level timing samples. Its immutable rows are schema version 1. The
new, hardware-validated schema version 2 varies the input by iteration,
records every rank's CUDA duration, and reports p50/p95/max over the maximum
rank duration for each iteration. It samples correctness each iteration,
checks the full final tensor, and provides a separate random-input,
uneven-length FP32/FP16/BF16 correctness matrix. Old and new timing statistics
must never be combined in one report.

For each strategy, calibration computes the median of
`measured_p50_us / predicted_us` over usable rows and clips the factor to
`[0.25, 4.0]`. The raw equations remain visible, and the sample count and source
path travel with the calibration profile.

The original H200 rows contain no `predicted_us` values and did not calibrate
the analytical link-tier model. The H200 empirical policy is a separate,
exact-context lookup with leave-one-repeat-out checks and NCCL fallback. It
does not extrapolate to another cluster or prove a custom path is faster.

## Production hardening path

Before enabling a non-NCCL path in an application:

1. prove correctness across dtypes, uneven sizes, world sizes, and repeated calls;
2. compare against NCCL over the full payload and node-count matrix;
3. profile phase overlap, launch serialization, and SM utilization;
4. run fault injection for process exit, stalled rank, NIC loss, and timeout;
5. canary by model bucket with automatic fallback to NCCL;
6. promote only if p95 step time improves without worsening tail or recovery time.

## Application integration boundary

The completed H200 held-out gate keeps gradient formation and communication selection separate.
DistributedDataParallel synchronizes initial parameters, then `no_sync` permits
FabricFlow to reduce a deterministic flattened gradient schedule. Both the NCCL
baseline and two explicit hierarchical candidates pay the same flatten,
bucket, divide, and copy-back costs. This isolates the strategy decision while
retaining forward, backward, optimizer, and synchronization costs in measured
wall-clock step latency. It is a synthetic communication-heavy workload without
normal DDP gradient-computation overlap, not a frontier-model benchmark. A
cross-rank maximum parameter spread check and equivalent final loss are
required before a candidate can pass the promotion report. All nine H200
application runs were correct, but neither custom schedule passed the
predeclared 5% p95 improvement threshold; NCCL remains the selected path.
