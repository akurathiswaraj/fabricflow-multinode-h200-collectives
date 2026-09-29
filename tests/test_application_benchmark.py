import unittest
from unittest.mock import patch

from fabricflow.application_benchmark import (
    POLICY_NAME,
    bucket_ranges,
    main,
    select_bucket_strategy,
)
from fabricflow.cost_model import Strategy


class ApplicationBenchmarkTests(unittest.TestCase):
    def test_historical_policy_is_disabled_for_h200_rdma(self) -> None:
        with patch.dict("os.environ", {"FABRICFLOW_EXPERIMENT_SCOPE": "multi_node_rdma"}):
            with self.assertRaisesRegex(SystemExit, "not validated on H200"):
                main(["--strategy", POLICY_NAME, "--output", "unused.jsonl"])

    def test_payload_policy_uses_measured_crossover_window(self) -> None:
        options = {"policy_min_bytes": 256 << 10, "policy_max_bytes": 4 << 20}
        self.assertEqual(
            select_bucket_strategy(POLICY_NAME, 1 << 20, **options),
            Strategy.HIERARCHICAL_RAIL,
        )
        self.assertEqual(
            select_bucket_strategy(POLICY_NAME, 64 << 10, **options),
            Strategy.NCCL_ALLREDUCE,
        )
        self.assertEqual(
            select_bucket_strategy(POLICY_NAME, 16 << 20, **options),
            Strategy.NCCL_ALLREDUCE,
        )

    def test_explicit_strategy_bypasses_policy(self) -> None:
        self.assertEqual(
            select_bucket_strategy(
                "hierarchical_leader",
                1 << 20,
                policy_min_bytes=256 << 10,
                policy_max_bytes=4 << 20,
            ),
            Strategy.HIERARCHICAL_LEADER,
        )

    def test_bucket_ranges_cover_each_element_once(self) -> None:
        ranges = bucket_ranges(11, 4, 16)
        self.assertEqual(ranges, [(0, 4), (4, 8), (8, 11)])
        covered = [index for start, end in ranges for index in range(start, end)]
        self.assertEqual(covered, list(range(11)))

    def test_invalid_bucket_dimensions_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            bucket_ranges(10, 4, 2)


if __name__ == "__main__":
    unittest.main()
