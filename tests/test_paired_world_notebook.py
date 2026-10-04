"""Checks that the Kaggle notebook carries the exact paired evaluator sources."""

from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tools/kaggle/hmr2s_paired_world_full11_3dpw_kaggle.ipynb"


class PairedWorldNotebookTests(unittest.TestCase):
    def test_every_embedded_source_matches_the_local_file(self) -> None:
        self.assertTrue(NOTEBOOK.is_file(), "paired world notebook is missing")
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cells = ["".join(cell["source"]) for cell in notebook["cells"]]
        configuration = ast.parse(cells[3]).body
        assignment = next(
            node for node in configuration
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "embedded" for target in node.targets)
        )
        embedded = ast.literal_eval(assignment.value)
        for name, (digest, encoded) in embedded.items():
            with self.subTest(name=name):
                contents = gzip.decompress(base64.b64decode(encoded))
                self.assertEqual(hashlib.sha256(contents).hexdigest(), digest)
                expected = (
                    ROOT / "tools/export" / name
                    if name == "export_wham_world_step.py"
                    else ROOT / "tools/evaluation" / name
                )
                self.assertEqual(contents, expected.read_bytes())

    def test_notebook_outputs_only_a_small_paired_report(self) -> None:
        self.assertTrue(NOTEBOOK.is_file(), "paired world notebook is missing")
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cells = ["".join(cell["source"]) for cell in notebook["cells"]]
        self.assertIn("hmr2s_paired_world_full11_3dpw_reports.zip", cells[7])
        self.assertIn("evaluate_hmr2s_paired_world.py", cells[6])
        self.assertIn("3dpw_test_vit.pth", cells[1])
        self.assertIn("J_regressor_feet.npy", cells[4])
        self.assertIn("--feet-regressor", cells[6])
        self.assertEqual(len(cells), 8)

    def test_original_initializer_never_uses_a_test_label_root(self) -> None:
        evaluator = (ROOT / "tools/evaluation/evaluate_hmr2s_paired_world.py").read_text()
        reference = (ROOT / "tools/evaluation/evaluate_full_pipeline_tradeoff.py").read_text()
        self.assertEqual(evaluator.count("use_target_first_root=False"), 2)
        self.assertIn("init_root = init_pose[:, :, 0].reshape(1, 1, 6)", reference)
        self.assertIn('combined["contact"] = normal_output["contact"]', evaluator)


if __name__ == "__main__":
    unittest.main()
