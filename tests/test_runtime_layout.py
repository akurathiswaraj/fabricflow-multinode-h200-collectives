import unittest

from fabricflow.runtime import ProcessGroupLayout


class ProcessGroupLayoutTests(unittest.TestCase):
    def test_contiguous_mapping(self) -> None:
        layout = ProcessGroupLayout(rank=19, world_size=32, local_size=8)
        self.assertEqual(layout.nodes, 4)
        self.assertEqual(layout.node_id, 2)
        self.assertEqual(layout.local_rank, 3)
        self.assertEqual(layout.node_leader, 16)
        self.assertEqual(layout.local_ranks(), list(range(16, 24)))
        self.assertEqual(layout.lane_ranks(), [3, 11, 19, 27])
        self.assertEqual(layout.leader_ranks(), [0, 8, 16, 24])

    def test_invalid_layouts_fail_before_group_creation(self) -> None:
        with self.assertRaises(ValueError):
            ProcessGroupLayout(rank=0, world_size=10, local_size=8)
        with self.assertRaises(ValueError):
            ProcessGroupLayout(rank=10, world_size=8, local_size=8)


if __name__ == "__main__":
    unittest.main()

