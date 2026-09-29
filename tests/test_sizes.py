import unittest

from fabricflow.sizes import format_size, parse_size


class SizeTests(unittest.TestCase):
    def test_binary_and_decimal_units(self) -> None:
        self.assertEqual(parse_size("1MiB"), 1 << 20)
        self.assertEqual(parse_size("1.5 MB"), 1_500_000)
        self.assertEqual(format_size(64 << 20), "64MiB")

    def test_invalid_values(self) -> None:
        for value in ("", "-1MiB", "12widgets", "0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_size(value)


if __name__ == "__main__":
    unittest.main()

