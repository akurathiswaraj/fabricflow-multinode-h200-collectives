import json
from pathlib import Path
import unittest

from fabricflow.h200_policy import build_policy, select


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "outputs" / "h200_v07_public_evidence" / "final.jsonl"


def synthetic_rows(rail: tuple[float, float, float]) -> list[dict[str, object]]:
    rows = []
    for strategy, values in (
        ("nccl_allreduce", (10.0, 10.0, 10.0)),
        ("hierarchical_leader", (20.0, 20.0, 20.0)),
        ("hierarchical_rail", rail),
    ):
        for repeat, value in enumerate(values, start=1):
            rows.append({
                "schema_version": 2,
                "timing_semantics": "per_iteration_max_rank_cuda_duration_us",
                "provenance": "hardware_measurement",
                "nodes": 2,
                "gpus_per_node": 8,
                "world_size": 16,
                "gpu_names": ["NVIDIA H200"],
                "correct": True,
                "run_id": "synthetic-test",
                "message_bytes": 1024,
                "strategy": strategy,
                "repeat_id": str(repeat),
                "measured_p50_us": value,
            })
    return rows


class H200PolicyTests(unittest.TestCase):
    def test_real_evidence_selects_nccl_at_all_recorded_sizes(self) -> None:
        rows = [json.loads(line) for line in EVIDENCE.read_text(encoding="utf-8").splitlines()]
        policy = build_policy(rows)
        self.assertEqual("per_iteration_max_rank_cuda_duration_us", policy["source_timing_semantics"])
        self.assertEqual(11, len(policy["decisions_by_message_bytes"]))
        self.assertTrue(all(entry["selected"] == "nccl_allreduce" for entry in policy["decisions_by_message_bytes"].values()))
        self.assertEqual("nccl_allreduce", select(policy, 12345))

    def test_custom_strategy_requires_stable_held_out_win(self) -> None:
        policy = build_policy(synthetic_rows((8.0, 8.0, 8.0)))
        self.assertEqual("hierarchical_rail", select(policy, 1024))
        unstable = build_policy(synthetic_rows((8.0, 8.0, 13.0)))
        self.assertEqual("nccl_allreduce", select(unstable, 1024))

    def test_incomplete_or_mixed_evidence_fails(self) -> None:
        rows = synthetic_rows((8.0, 8.0, 8.0))
        with self.assertRaises(ValueError):
            build_policy(rows[:-1])
        rows[0]["gpu_names"] = ["NVIDIA A10G"]
        with self.assertRaises(ValueError):
            build_policy(rows)


if __name__ == "__main__":
    unittest.main()
