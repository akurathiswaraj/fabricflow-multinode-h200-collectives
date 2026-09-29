from pathlib import Path
import unittest

from fabricflow.h200_evidence import audit
from fabricflow.h200_v07_evidence import audit_public_directory


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "outputs" / "hardware_evidence" / "h200-rdma"
PUBLIC_V07 = ROOT / "outputs" / "h200_v07_public_evidence"


class CheckedInEvidenceTests(unittest.TestCase):
    def test_h200_public_source_integrity_and_published_decision(self) -> None:
        public = audit_public_directory(PUBLIC_V07, ROOT)
        self.assertEqual(8, public["checksummed_public_files"])
        self.assertEqual(90, public["correctness_cases"])
        self.assertEqual(99, public["final_rows"])
        self.assertEqual(9, public["application_rows"])
        self.assertEqual([], public["application_promoted_strategies"])
        # The private working tree may also retain the immutable legacy data;
        # public releases deliberately omit unredacted provider logs.
        if EVIDENCE.exists():
            legacy = audit(EVIDENCE)
            self.assertEqual(198, legacy["checksummed_source_files"])
            self.assertEqual("nccl_allreduce", legacy["winner_at_all_payloads"])


if __name__ == "__main__":
    unittest.main()
