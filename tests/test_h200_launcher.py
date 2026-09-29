from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class H200LauncherTests(unittest.TestCase):
    def test_cluster_shape_and_logging(self) -> None:
        source = (ROOT / "modal_fabricflow.py").read_text(encoding="utf-8")
        for marker in (
            "NODES = 2",
            "GPUS_PER_NODE = 8",
            'gpu="H200:8"',
            "@modal.experimental.clustered(size=NODES, rdma=True)",
            '"FABRICFLOW_NODE_RANK": str(cluster_info.rank)',
            '"FABRICFLOW_NODE_ID": cluster_info.container_ips[cluster_info.rank]',
            '"NCCL_DEBUG": "INFO"',
            'f"launch-{launch_number}.nccl.%h.%p.log"',
            "timeout=60 * 60",
            'if mode == "correctness":',
            '"fabricflow.correctness"',
            '"fabricflow.application_benchmark"',
            '"source_sha256": _source_hashes()',
            '_write_manifest(run_id, mode, output_path)',
        ):
            self.assertIn(marker, source)


if __name__ == "__main__":
    unittest.main()
