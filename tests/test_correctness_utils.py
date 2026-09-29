import unittest

from fabricflow.correctness import DEFAULT_COUNTS, DTYPES, STRATEGIES, input_seed, parse_counts


class CorrectnessUtilityTests(unittest.TestCase):
    def test_default_matrix_covers_uneven_lengths_and_dtypes(self) -> None:
        self.assertIn(1025, DEFAULT_COUNTS)
        self.assertIn(262147, DEFAULT_COUNTS)
        self.assertEqual({"float32", "float16", "bfloat16"}, set(DTYPES))
        self.assertEqual(3, len(STRATEGIES))

    def test_counts_and_seeds_are_unambiguous(self) -> None:
        self.assertEqual((1, 9, 1025), parse_counts("1,9,1025"))
        self.assertNotEqual(input_seed(0, 0), input_seed(0, 1))
        self.assertNotEqual(input_seed(0, 0), input_seed(1, 0))
        for raw in ("", "0", "1,1", "1000001"):
            with self.assertRaises(ValueError):
                parse_counts(raw)


if __name__ == "__main__":
    unittest.main()
