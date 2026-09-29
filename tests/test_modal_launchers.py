from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ModalLauncherTests(unittest.TestCase):
    def test_h200_launcher_matches_published_shape_and_evidence_capture(self) -> None:
        source = (ROOT / "modal_fabricflow.py").read_text(encoding="utf-8")
        self.assertIn('gpu="H200:8"', source)
        self.assertIn("@modal.experimental.clustered(size=NODES, rdma=True)", source)
        self.assertIn('"FABRICFLOW_NODE_RANK": str(cluster_info.rank)', source)
        self.assertIn('"FABRICFLOW_NODE_ID": cluster_info.container_ips[cluster_info.rank]', source)
        self.assertIn('"NCCL_DEBUG": "INFO"', source)
        self.assertIn('f"launch-{launch_number}.nccl.%h.%p.log"', source)

    def test_h200_launcher_installs_object_collective_dependency(self) -> None:
        source = (ROOT / "modal_fabricflow.py").read_text(encoding="utf-8")
        self.assertIn('"numpy==2.3.3"', source)


if __name__ == "__main__":
    unittest.main()
