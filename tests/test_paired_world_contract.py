"""Population guards for the original-WHAM versus phone world comparison."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/evaluation"))


class PairedWorldContractTests(unittest.TestCase):
    def test_accepts_identical_complete_track_populations(self) -> None:
        from paired_world_contract import verify_paired_population

        expected = {"track_a": 3, "track_b": 2}
        self.assertEqual(
            verify_paired_population(expected, expected.copy(), expected.copy()),
            5,
        )

    def test_rejects_dropped_or_truncated_track_in_either_arm(self) -> None:
        from paired_world_contract import verify_paired_population

        expected = {"track_a": 3, "track_b": 2}
        with self.assertRaisesRegex(ValueError, "original"):
            verify_paired_population(expected, {"track_a": 3}, expected.copy())
        with self.assertRaisesRegex(ValueError, "phone"):
            verify_paired_population(expected, expected.copy(), {"track_a": 3, "track_b": 1})


if __name__ == "__main__":
    unittest.main()
