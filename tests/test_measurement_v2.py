import unittest

from fabricflow.measurement_v2 import validate_application_row, validate_benchmark_row


class MeasurementV2Tests(unittest.TestCase):
    def test_benchmark_row_recomputes_slowest_rank_distribution(self) -> None:
        row = {
            "schema_version": 2,
            "timing_semantics": "per_iteration_max_rank_cuda_duration_us",
            "world_size": 2,
            "iterations": 3,
            "rank_timings_us": [[1.0, 4.0, 5.0], [3.0, 2.0, 1.0]],
            "completion_timings_us": [3.0, 4.0, 5.0],
            "measured_p50_us": 4.0,
            "measured_p95_us": 5.0,
            "measured_max_us": 5.0,
        }
        validate_benchmark_row(row)
        with self.assertRaisesRegex(ValueError, "benchmark p95"):
            validate_benchmark_row({**row, "measured_p95_us": 4.0})
        with self.assertRaisesRegex(ValueError, "completion timing"):
            validate_benchmark_row({**row, "completion_timings_us": [3.0, 4.0, 4.0]})

    def test_application_row_recomputes_slowest_rank_step(self) -> None:
        row = {
            "schema_version": 2,
            "timing_semantics": "per_step_max_rank_wall_duration_us",
            "world_size": 2,
            "measured_steps": 3,
            "rank_step_timings_us": [[1.0, 4.0, 5.0], [3.0, 2.0, 1.0]],
            "completion_step_timings_us": [3.0, 4.0, 5.0],
            "step_p50_us": 4.0,
            "step_p95_us": 5.0,
            "step_max_us": 5.0,
        }
        validate_application_row(row)
        with self.assertRaisesRegex(ValueError, "invalid iteration count"):
            validate_application_row({**row, "rank_step_timings_us": [[1.0], [2.0]]})


if __name__ == "__main__":
    unittest.main()
