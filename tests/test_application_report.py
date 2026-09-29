import unittest

from fabricflow.application_report import build_application_markdown


def rows_for(strategy: str, p95_values: list[float], loss: float = 1.0):
    return [
        {
            "provenance": "hardware_application_measurement",
            "strategy": strategy,
            "repeat_id": str(index),
            "step_p50_us": p95 * 0.8,
            "step_p95_us": p95,
            "final_global_loss": loss,
            "correct": True,
            "nodes": 2,
            "gpus_per_node": 4,
            "world_size": 8,
            "gpu_names": ["test-gpu"],
            "network_transport_requested": "socket",
            "gradient_bytes": 32 << 20,
            "bucket_bytes": 1 << 20,
        }
        for index, p95 in enumerate(p95_values, start=1)
    ]


class ApplicationReportTests(unittest.TestCase):
    def test_duplicate_repeat_cannot_pass_promotion(self) -> None:
        rows = rows_for("nccl_allreduce", [100, 102, 98])
        rows += rows_for("hierarchical_rail", [70, 71, 72])
        rows[-1]["repeat_id"] = "2"
        report = build_application_markdown(rows)
        self.assertIn("Promoted strategies: none", report)

    def test_incompatible_hardware_context_is_rejected(self) -> None:
        rows = rows_for("nccl_allreduce", [100, 102, 98])
        rows += rows_for("hierarchical_rail", [89, 90, 91])
        rows[-1]["gpus_per_node"] = 8
        with self.assertRaisesRegex(ValueError, "incompatible runs"):
            build_application_markdown(rows)

    def test_policy_is_promoted_only_after_p95_and_loss_gates(self) -> None:
        rows = rows_for("nccl_allreduce", [100, 102, 98])
        rows += rows_for("payload_policy", [89, 90, 91], loss=1.0001)
        report = build_application_markdown(rows)
        self.assertIn("| payload_policy | 3", report)
        self.assertIn("| yes | yes |", report)
        self.assertIn("Promoted strategies: payload_policy", report)

    def test_fast_policy_with_wrong_loss_is_not_promoted(self) -> None:
        rows = rows_for("nccl_allreduce", [100, 102, 98])
        rows += rows_for("payload_policy", [80, 81, 79], loss=1.1)
        report = build_application_markdown(rows)
        self.assertIn("Promoted strategies: none", report)

    def test_requires_three_correct_baseline_repeats(self) -> None:
        with self.assertRaisesRegex(ValueError, "three correct NCCL"):
            build_application_markdown(rows_for("nccl_allreduce", [100, 101]))

    def test_microbenchmark_rows_cannot_enter_application_report(self) -> None:
        with self.assertRaisesRegex(ValueError, "application rows"):
            build_application_markdown(
                [{"provenance": "hardware_measurement"}]
            )


if __name__ == "__main__":
    unittest.main()
