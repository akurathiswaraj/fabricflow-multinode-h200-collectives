import unittest

from fabricflow.benchmark import (
    _completion_timings,
    _expected_sum,
    _percentile,
    _rank_timing_matrix,
    _validate_node_distribution,
    _version_string,
)


class BenchmarkUtilityTests(unittest.TestCase):
    def test_completion_uses_slowest_rank_per_iteration(self) -> None:
        ranks = _rank_timing_matrix([10.0, 20.0, 30.0, 50.0, 5.0, 25.0], 2, 3)
        self.assertEqual(ranks, [[10.0, 20.0, 30.0], [50.0, 5.0, 25.0]])
        self.assertEqual(_completion_timings(ranks), [50.0, 20.0, 30.0])
        self.assertEqual(_percentile(_completion_timings(ranks), 50), 30.0)

    def test_timing_matrix_rejects_incomplete_or_invalid_samples(self) -> None:
        with self.assertRaises(ValueError):
            _rank_timing_matrix([1.0, 2.0, 3.0], 2, 2)
        with self.assertRaises(ValueError):
            _rank_timing_matrix([1.0, float("nan")], 1, 2)
        with self.assertRaises(ValueError):
            _completion_timings([[1.0], [2.0, 3.0]])

    def test_expected_sum_changes_each_timed_iteration(self) -> None:
        self.assertEqual([_expected_sum(2, i) for i in (0, 1, 6, 7)], [3.0, 5.0, 15.0, 3.0])

    def test_nearest_rank_percentile(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(_percentile(values, 50), 2.0)
        self.assertEqual(_percentile(values, 95), 4.0)

    def test_version_formats_tuple_or_scalar(self) -> None:
        self.assertEqual(_version_string((2, 31, 2)), "2.31.2")
        self.assertEqual(_version_string(23102), "23102")

    def test_provider_node_ids_allow_reused_container_hostnames(self) -> None:
        hardware = [
            {"hostname": "modal", "node_id": "fd00::1"},
            {"hostname": "modal", "node_id": "fd00::1"},
            {"hostname": "modal", "node_id": "fd00::2"},
            {"hostname": "modal", "node_id": "fd00::2"},
        ]
        node_ids, counts = _validate_node_distribution(
            hardware,
            expected_nodes=2,
            expected_gpus_per_node=2,
        )
        self.assertEqual(node_ids, ["fd00::1", "fd00::2"])
        self.assertEqual(counts, {"fd00::1": 2, "fd00::2": 2})

    def test_invalid_node_rank_distribution_is_rejected(self) -> None:
        hardware = [
            {"node_id": "fd00::1"},
            {"node_id": "fd00::1"},
            {"node_id": "fd00::1"},
            {"node_id": "fd00::2"},
        ]
        with self.assertRaisesRegex(RuntimeError, "rank distribution"):
            _validate_node_distribution(
                hardware,
                expected_nodes=2,
                expected_gpus_per_node=2,
            )


if __name__ == "__main__":
    unittest.main()
