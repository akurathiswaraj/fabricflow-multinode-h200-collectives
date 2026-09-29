from pathlib import Path
import tempfile
import unittest

from fabricflow.calibration import CalibrationProfile, fit_calibration
from fabricflow.cost_model import Strategy, hierarchical_rail
from fabricflow.topology import ClusterTopology


class CalibrationTests(unittest.TestCase):
    def test_fit_uses_robust_median_ratio(self) -> None:
        rows = [
            {"strategy": "hierarchical_rail", "predicted_us": 100, "measured_p50_us": 200},
            {"strategy": "hierarchical_rail", "predicted_us": 100, "measured_p50_us": 180},
            {"strategy": "hierarchical_rail", "predicted_us": 100, "measured_p50_us": 220},
        ]
        profile = fit_calibration(rows)
        self.assertEqual(profile.factor_for(Strategy.HIERARCHICAL_RAIL), 2.0)
        self.assertEqual(profile.sample_counts["hierarchical_rail"], 3)

    def test_profile_round_trip_and_application(self) -> None:
        profile = CalibrationProfile(
            factors={"hierarchical_rail": 1.25},
            sample_counts={"hierarchical_rail": 4},
            source="test",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            profile.save(path)
            loaded = CalibrationProfile.load(path)
        topology = ClusterTopology.load("configs/illustrative_8x8_dual_rail.json")
        prediction = hierarchical_rail(topology, 1 << 20)
        self.assertAlmostEqual(loaded.apply(prediction).latency_us, prediction.latency_us * 1.25)


if __name__ == "__main__":
    unittest.main()

