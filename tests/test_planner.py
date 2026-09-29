import unittest

from fabricflow.cost_model import Strategy
from fabricflow.planner import Planner
from fabricflow.topology import ClusterTopology


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.planner = Planner(
            ClusterTopology.load("configs/illustrative_8x8_dual_rail.json")
        )

    def test_small_message_selects_leader(self) -> None:
        plan = self.planner.plan(1024)
        self.assertEqual(plan.selected.strategy, Strategy.HIERARCHICAL_LEADER)

    def test_large_message_selects_rail_runtime(self) -> None:
        plan = self.planner.plan(1 << 30)
        self.assertEqual(plan.selected.strategy, Strategy.HIERARCHICAL_RAIL)

    def test_design_target_is_excluded_by_default(self) -> None:
        plan = self.planner.plan(1 << 30)
        self.assertTrue(plan.selected.runtime_supported)
        experimental = self.planner.plan(1 << 30, runtime_only=False)
        self.assertIn(experimental.selected, experimental.candidates)


if __name__ == "__main__":
    unittest.main()

