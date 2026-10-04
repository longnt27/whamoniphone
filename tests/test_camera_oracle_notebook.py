"""Portable-notebook checks that do not require Kaggle or licensed 3DPW data."""

from __future__ import annotations

import ast
import base64
import csv
import gzip
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tools/kaggle/hmr2s_camera_oracle_full11_3dpw_v2_kaggle.ipynb"


class CameraOracleNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = ["".join(cell["source"]) for cell in cls.notebook["cells"]]

    def test_notebook_has_unselected_test_only_protocol(self) -> None:
        self.assertIn("oracle-input experiment", self.cells[0])
        self.assertIn("--test-parsed", self.cells[6])
        self.assertNotIn("--val-parsed", self.cells[6])
        self.assertIn("ground-truth camera rotations", self.cells[0])
        self.assertIn("not a phone gyroscope test", self.cells[0])
        self.assertIn("11,349 recurrent frames", self.cells[0])
        self.assertIn("fails instead of silently evaluating a shortened subset", self.cells[0])

    def test_all_python_cells_parse_and_embed_matching_sources(self) -> None:
        for index, code in enumerate(self.cells[1:], start=1):
            if index == 2:  # Kaggle IPython %pip magic.
                continue
            with self.subTest(cell=index):
                ast.parse(code)
        constants = ast.parse(self.cells[3]).body
        assignment = next(
            node for node in constants
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "embedded" for target in node.targets)
        )
        embedded = ast.literal_eval(assignment.value)
        for name, (digest, encoded) in embedded.items():
            with self.subTest(source=name):
                contents = gzip.decompress(base64.b64decode(encoded))
                self.assertEqual(hashlib.sha256(contents).hexdigest(), digest)
                self.assertEqual(contents, (ROOT / "tools/evaluation" / name).read_bytes())

    def test_kaggle_outputs_are_report_only(self) -> None:
        self.assertIn("hmr2s_camera_oracle_full11_3dpw_reports.zip", self.cells[7])
        self.assertIn("hmr2s_camera_oracle_full11_3dpw.json", self.cells[7])
        self.assertIn("hmr2s_camera_oracle_full11_3dpw.csv", self.cells[7])
        self.assertEqual(len(self.notebook["cells"]), 8)

    def test_locked_full_track_population_matches_previous_body_pose_csv(self) -> None:
        evaluator_source = (ROOT / "tools/evaluation/evaluate_hmr2s_camera_oracle.py").read_text()
        evaluator = ast.parse(evaluator_source)
        self.assertNotIn("Noncontiguous frame IDs in locked 30-fps track", evaluator_source)
        self.assertIn('"noncontiguous_frame_transitions"', evaluator_source)
        assignment = next(
            node for node in evaluator.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "EXPECTED_TEST_TRACK_FRAMES"
                for target in node.targets
            )
        )
        expected = ast.literal_eval(assignment.value)
        with (ROOT / "evaluation/results/selected/hmr2s_temporal_smoothing_3dpw.csv").open() as stream:
            previous = {
                row["sequence"]: int(row["frames"])
                for row in csv.DictReader(stream)
                if row["split"] == "test" and row["filter"] == "light"
            }
        self.assertEqual(expected, previous)
        self.assertEqual(len(expected), 11)
        self.assertEqual(sum(expected.values()), 11349)


if __name__ == "__main__":
    unittest.main()
