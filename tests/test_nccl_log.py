import tempfile
from pathlib import Path
import unittest

from fabricflow.nccl_log import summarize_nccl_logs


class NcclLogTests(unittest.TestCase):
    def test_detects_socket_transport_and_keeps_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "rank0.log"
            log.write_text(
                "NCCL INFO NET/Socket : Using [0]eth0:10.0.0.1\n",
                encoding="utf-8",
            )
            summary = summarize_nccl_logs([log])
        self.assertEqual(summary["observed_transport"], "socket")
        self.assertEqual(summary["marker_counts"], {"socket": 1})
        self.assertTrue(summary["evidence"])

    def test_mixed_or_missing_markers_are_not_mislabeled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            socket_log = root / "socket.log"
            ib_log = root / "ib.log"
            empty_log = root / "empty.log"
            socket_log.write_text("NCCL INFO NET/Socket\n", encoding="utf-8")
            ib_log.write_text("NCCL INFO NET/IB\n", encoding="utf-8")
            empty_log.write_text("NCCL INFO bootstrap\n", encoding="utf-8")
            mixed = summarize_nccl_logs([socket_log, ib_log])
            unknown = summarize_nccl_logs([empty_log])
        self.assertEqual(mixed["observed_transport"], "mixed")
        self.assertEqual(unknown["observed_transport"], "unknown")


if __name__ == "__main__":
    unittest.main()
