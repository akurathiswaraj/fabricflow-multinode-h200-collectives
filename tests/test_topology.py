import json
from pathlib import Path
import tempfile
import unittest

from fabricflow.topology import ClusterTopology, LinkTier


class LinkTierTests(unittest.TestCase):
    def test_effective_bandwidth_and_transfer(self) -> None:
        link = LinkTier("test", 100, 5, efficiency=0.8, contention=2)
        self.assertAlmostEqual(link.effective_bytes_per_second, 5e9)
        self.assertAlmostEqual(link.transfer_us(5_000), 6.0)

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LinkTier("bad", 0, 1)
        with self.assertRaises(ValueError):
            LinkTier("bad", 1, 1, efficiency=1.1)


class TopologyTests(unittest.TestCase):
    def test_load_and_world_size(self) -> None:
        topology = ClusterTopology.load("configs/illustrative_8x8_dual_rail.json")
        self.assertEqual(topology.world_size, 64)
        self.assertEqual(topology.rail_parallelism, 4)
        self.assertFalse(topology.metadata["hardware_claim"])

    def test_missing_fields_are_actionable(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing topology fields"):
            ClusterTopology.from_dict({"name": "incomplete"})

    def test_round_trip(self) -> None:
        original = ClusterTopology.load("configs/illustrative_2x8_single_rail.json")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topology.json"
            path.write_text(json.dumps(original.to_dict()), encoding="utf-8")
            self.assertEqual(ClusterTopology.load(path), original)


if __name__ == "__main__":
    unittest.main()

