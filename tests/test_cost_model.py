import unittest

from fabricflow.cost_model import (
    Strategy,
    hierarchical_leader,
    hierarchical_rail,
    nccl_allreduce_proxy,
    pipelined_rail,
    predict_all,
)
from fabricflow.topology import ClusterTopology


class CostModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.topology = ClusterTopology.load("configs/illustrative_8x8_dual_rail.json")

    def test_all_candidates_are_positive_and_auditable(self) -> None:
        candidates = predict_all(self.topology, 64 << 20)
        self.assertEqual(len(candidates), 4)
        for candidate in candidates:
            self.assertGreater(candidate.latency_us, 0)
            self.assertGreater(candidate.algorithmic_bytes_per_rank, 0)
            self.assertTrue(candidate.phases)

    def test_leader_strategy_is_latency_friendly_for_tiny_payload(self) -> None:
        size = 1024
        leader = hierarchical_leader(self.topology, size)
        rail = hierarchical_rail(self.topology, size)
        self.assertLess(leader.latency_us, rail.latency_us)

    def test_rail_strategy_beats_single_path_proxy_for_large_payload(self) -> None:
        size = 1 << 30
        rail = hierarchical_rail(self.topology, size)
        baseline = nccl_allreduce_proxy(self.topology, size)
        self.assertLess(rail.latency_us, baseline.latency_us)

    def test_pipeline_is_labeled_as_design_target(self) -> None:
        prediction = pipelined_rail(self.topology, 1 << 30, chunks=8)
        self.assertEqual(prediction.strategy, Strategy.PIPELINED_RAIL)
        self.assertFalse(prediction.runtime_supported)
        self.assertEqual(prediction.phases[0].steps, 8)

    def test_calibration_scales_phases_and_total(self) -> None:
        original = hierarchical_rail(self.topology, 32 << 20)
        calibrated = original.calibrated(1.5)
        self.assertAlmostEqual(calibrated.latency_us, original.latency_us * 1.5)
        self.assertAlmostEqual(calibrated.calibration_factor, 1.5)


if __name__ == "__main__":
    unittest.main()

