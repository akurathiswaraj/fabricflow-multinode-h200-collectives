# H200 v0.7 public evidence

These are redacted copies of four verified runs: 90 correctness cases, six smoke rows, 99 final microbenchmark rows, and nine synthetic training-step rows. The timing arrays are preserved so the reported p50/p95 values can be recomputed. Provider node addresses, cluster IDs, GUIDs, and raw NCCL logs are intentionally not published.

`transport_audit.json` contains per-run counts computed from all 352 private NCCL rank logs. `provenance.json` records the private bundle's SHA-256 and the SHA-256 hashes of the executed source files. The private bundle can be audited with `fabricflow.h200_v07_evidence` if made available under an appropriate sharing arrangement.

The physical link is Ethernet/RoCE; NCCL's `IB` label does not mean physical InfiniBand. Logical process-group lanes are not proof of per-lane NIC affinity. See `docs/results_h200_v07.md` for the decision and limitations.
