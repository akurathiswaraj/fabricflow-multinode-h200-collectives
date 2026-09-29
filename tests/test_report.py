import unittest

from fabricflow.report import build_markdown


class ReportTests(unittest.TestCase):
    def test_old_and_new_timing_semantics_cannot_be_aggregated(self) -> None:
        base = {
            "provenance": "hardware_measurement",
            "message_bytes": 1024,
            "world_size": 16,
            "nodes": 2,
            "gpus_per_node": 8,
            "strategy": "nccl_allreduce",
            "measured_p50_us": 100.0,
            "measured_p95_us": 120.0,
            "correct": True,
        }
        with self.assertRaisesRegex(ValueError, "timing semantics"):
            build_markdown([
                base,
                {**base, "timing_semantics": "per_iteration_max_rank_cuda_duration_us"},
            ])

    def test_report_selects_fastest_correct_strategy(self) -> None:
        common = {
            "provenance": "hardware_measurement",
            "message_bytes": 1 << 20,
            "world_size": 16,
            "nodes": 2,
            "gpus_per_node": 8,
            "torch_version": "test",
            "cuda_version": "test",
            "nccl_version": "test",
        }
        rows = [
            {
                **common,
                "strategy": "nccl_allreduce",
                "measured_p50_us": 100,
                "measured_p95_us": 110,
                "correct": True,
            },
            {
                **common,
                "strategy": "hierarchical_rail",
                "measured_p50_us": 80,
                "measured_p95_us": 90,
                "correct": True,
            },
        ]
        report = build_markdown(rows)
        self.assertIn("hierarchical_rail", report)
        self.assertIn("1.250x", report)

    def test_simulation_cannot_be_rebranded_as_hardware(self) -> None:
        with self.assertRaisesRegex(ValueError, "hardware_measurement"):
            build_markdown([{"provenance": "analytical_simulation_not_hardware_measurement"}])

    def test_repeated_runs_use_median_not_best_case(self) -> None:
        common = {
            "provenance": "hardware_measurement",
            "message_bytes": 1 << 20,
            "world_size": 16,
            "nodes": 2,
            "gpus_per_node": 8,
            "torch_version": "test",
            "cuda_version": "test",
            "nccl_version": "test",
            "correct": True,
        }
        rows = []
        for strategy, p50_values in {
            "nccl_allreduce": [100, 110, 1000],
            "hierarchical_rail": [90, 95, 96],
        }.items():
            rows.extend(
                {
                    **common,
                    "strategy": strategy,
                    "measured_p50_us": p50,
                    "measured_p95_us": p50 + 10,
                }
                for p50 in p50_values
            )
        report = build_markdown(rows)
        self.assertIn(
            "| unspecified | 2 | 8 | 16 | 1MiB | 3 | 110.00 | hierarchical_rail | 95.00",
            report,
        )
        self.assertIn("1.158x", report)

    def test_world_sizes_are_reported_separately(self) -> None:
        common = {
            "provenance": "hardware_measurement",
            "message_bytes": 1 << 20,
            "nodes": 1,
            "torch_version": "test",
            "cuda_version": "test",
            "nccl_version": "test",
            "correct": True,
        }
        rows = []
        for world_size, latency in ((2, 20), (8, 80)):
            for strategy in ("nccl_allreduce", "hierarchical_rail"):
                rows.append(
                    {
                        **common,
                        "world_size": world_size,
                        "gpus_per_node": world_size,
                        "strategy": strategy,
                        "measured_p50_us": latency,
                        "measured_p95_us": latency + 5,
                    }
                )
        report = build_markdown(rows)
        self.assertIn("| unspecified | 1 | 2 | 2 | 1MiB", report)
        self.assertIn("| unspecified | 1 | 8 | 8 | 1MiB", report)
        self.assertIn("World sizes: 2, 8", report)

    def test_network_transports_are_not_aggregated_together(self) -> None:
        common = {
            "provenance": "hardware_measurement",
            "message_bytes": 1 << 20,
            "world_size": 16,
            "nodes": 2,
            "gpus_per_node": 8,
            "strategy": "nccl_allreduce",
            "measured_p95_us": 120,
            "correct": True,
        }
        report = build_markdown(
            [
                {
                    **common,
                    "network_transport_requested": "socket",
                    "measured_p50_us": 100,
                },
                {
                    **common,
                    "network_transport_requested": "rdma",
                    "measured_p50_us": 20,
                },
            ]
        )
        self.assertIn("| socket | 2 | 8 | 16 | 1MiB", report)
        self.assertIn("| rdma | 2 | 8 | 16 | 1MiB", report)


if __name__ == "__main__":
    unittest.main()
