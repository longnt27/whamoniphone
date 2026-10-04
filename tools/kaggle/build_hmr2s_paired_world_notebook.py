"""Build the locked original-WHAM versus iPhone world-accuracy notebook."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE / "hmr2s_camera_oracle_full11_3dpw_v2_kaggle.ipynb"
OUTPUT = HERE / "hmr2s_paired_world_full11_3dpw_kaggle.ipynb"
EVALUATION = ROOT / "tools/evaluation"
EXPORT = ROOT / "tools/export"
SOURCES = (
    "camera_motion_oracle.py",
    "paired_world_contract.py",
    "evaluate_hmr2s_paired_world.py",
    "evaluate_hmr2s_camera_oracle.py",
    "evaluate_hmr2s_temporal_smoothing.py",
    "hmr2s_frozen.py",
    "evaluate_frozen_hmr2s_wham.py",
    "evaluate_full_pipeline_tradeoff.py",
    "evaluate_mobile_pipeline_3dpw.py",
    "evaluate_wham_feature_substitution.py",
    "export_wham_world_step.py",
)


def lines(value: str) -> list[str]:
    return value.splitlines(True)


def main() -> None:
    notebook = json.loads(BASE.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if len(cells) != 8:
        raise RuntimeError(f"Expected eight reviewed base cells, found {len(cells)}")
    cells[0]["source"] = lines(
        "# Released WHAM versus iPhone pipeline: paired full-track 3DPW accuracy\n\n"
        "Run **both pipelines on exactly the same 11 one-person 3DPW test tracks and "
        "all 11,349 recurrent frames**. Both receive the same ground-truth 3DPW camera rotation "
        "at WHAM's camera-angular input, with identical zero fallbacks at nonadjacent/invalid "
        "transitions. The released WHAM branch uses official cached ViTPose/HMR2 features and "
        "flip-averaged pose/shape, and normal-branch contact/trajectory. The iPhone branch "
        "uses YOLO26m-pose, released HMR2.0-S, the "
        "selected token adapter, released WHAM, and its locked light output smoothing. "
        "Both branches execute the **same causal SMPL/contact-aware WHAM world-refiner step "
        "exported for the iPhone app**. Neither receives ground-truth body pose or root orientation "
        "as an inference initializer. The evaluator refuses incomplete tracks or failure to "
        "reproduce the previously locked camera-relative body-pose results. No training, model "
        "selection, or 100-frame metric substitution occurs.\n\n"
        "Scores use identical 3DPW SMPL world labels and the same 14-joint, one-first-frame "
        "rigid alignment across each **entire track**. Camera-relative PA-MPJPE, MPJPE, PVE, "
        "and acceleration are reported alongside root orientation, root displacement, and "
        "full-track world-joint error. This is a controlled **oracle-camera** comparison, "
        "not measured iPhone gyro/ARKit accuracy, not a compiled Core ML numerical parity test, "
        "and not WHAM's EMDB-2 W-MPJPE100 score.\n\n"
        "## Kaggle inputs\n\n"
        "Attach the same three inputs as the previous full-11 notebook: (1) `3dpw-model` "
        "with `imageFiles/imageFiles`, `sequenceFiles/sequenceFiles/test`, `3dpw_test_vit.pth`, "
        "and licensed SMPL neutral/male/female models; (2) saved YOLO26 grid output with "
        "`m/hmr2s_to_hmr2a_adapter_best.pth`; (3) the official HMR2.0-S "
        "`hmr_vit-small_d3-a4x16-m128.zip` or matching `last.ckpt`. Enable GPU and Internet. "
        "No extra dataset is required. Download "
        "`hmr2s_paired_world_full11_3dpw_reports.zip` at the end and send that small zip back."
    )
    cells[1]["source"] = lines(
        "".join(cells[1]["source"]).replace(
            "hmr2s_camera_oracle_full11", "hmr2s_paired_world_full11"
        )
    )
    resource_code = "".join(cells[4]["source"])
    if resource_code.count("    'J_regressor_wham.npy':") != 1:
        raise RuntimeError("Base notebook's WHAM regressor download entry changed")
    resource_code = resource_code.replace(
        "    'J_regressor_wham.npy':",
        "    'J_regressor_feet.npy': ('https://huggingface.co/camenduru/WHAM/resolve/main/J_regressor_feet.npy?download=true', '7ef9e6d64796f2f342983a9fde6a6d9f8e3544f1239e7f86aa4f6b7aa82f4cf6'),\n"
        "    'J_regressor_wham.npy':",
    )
    resource_code += "FEET_REGRESSOR = downloaded['J_regressor_feet.npy']\n"
    cells[4]["source"] = lines(resource_code)
    embedded_lines = []
    for name in SOURCES:
        path = (EXPORT if name == "export_wham_world_step.py" else EVALUATION) / name
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        compressed = base64.b64encode(gzip.compress(payload, compresslevel=9)).decode("ascii")
        embedded_lines.append(f"    {name!r}: ({digest!r}, {compressed!r}),")
    cells[3]["source"] = lines(
        "# Materialize checksum-verified evaluation and app-world-step sources.\n"
        "import base64, gzip\n\n"
        "SCRATCH_DIR.mkdir(parents=True, exist_ok=True)\n"
        "embedded = {\n" + "\n".join(embedded_lines) + "\n}\n"
        "for name, (expected, payload) in embedded.items():\n"
        "    contents = gzip.decompress(base64.b64decode(payload))\n"
        "    if hashlib.sha256(contents).hexdigest() != expected:\n"
        "        raise RuntimeError(f'Embedded source checksum mismatch: {name}')\n"
        "    (SCRATCH_DIR / name).write_bytes(contents)\n"
        "print({name: digest for name, (digest, _) in embedded.items()}, flush=True)\n"
    )
    cells[6]["source"] = lines(
        "# Verify inputs and run the complete paired 11-track comparison.\n"
        "import sys\n\n"
        "EVALUATOR = SCRATCH_DIR / 'evaluate_hmr2s_paired_world.py'\n"
        "environment = os.environ.copy()\n"
        "environment['PYTHONPATH'] = str(SCRATCH_DIR) + os.pathsep + environment.get('PYTHONPATH', '')\n"
        "subprocess.run([sys.executable, '-u', str(EVALUATOR), '--self-test'], check=True, env=environment)\n"
        "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
        "arguments = [\n"
        "    '--test-parsed', str(TEST_PARSED),\n"
        "    '--three-dpw-root', str(THREEDPW_ROOT),\n"
        "    '--wham-repo', str(WHAM_REPO),\n"
        "    '--wham-checkpoint', str(WHAM_CHECKPOINT),\n"
        "    '--hmr2s-repo', str(HMR2S_REPO),\n"
        "    '--hmr2s-checkpoint', str(HMR2S_CHECKPOINT),\n"
        "    '--yolo26m-weights', str(YOLO26M_WEIGHTS),\n"
        "    '--expected-yolo-sha256', YOLO26M_SHA256,\n"
        "    '--adapter-checkpoint', str(ADAPTER_CHECKPOINT),\n"
        "    '--expected-adapter-sha256', ADAPTER_SHA256,\n"
        "    '--smpl-model-directory', str(SMPL_MODEL_DIR),\n"
        "    '--h36m-joint-regressor', str(H36M_REGRESSOR),\n"
        "    '--wham-joint-regressor', str(WHAM_REGRESSOR),\n"
        "    '--feet-regressor', str(FEET_REGRESSOR),\n"
        "    '--output-json', str(OUTPUT_DIR / 'hmr2s_paired_world_full11_3dpw.json'),\n"
        "    '--output-csv', str(OUTPUT_DIR / 'hmr2s_paired_world_full11_3dpw.csv'),\n"
        "    '--pose-batch-size', str(POSE_BATCH_SIZE),\n"
        "    '--hmr-batch-size', str(HMR2S_BATCH_SIZE),\n"
        "    '--smpl-batch-size', str(SMPL_BATCH_SIZE),\n"
        "]\n"
        "subprocess.run([sys.executable, '-u', str(EVALUATOR), *arguments], check=True, env=environment)\n"
    )
    cells[7]["source"] = lines(
        "# Print headline numbers and package only the small reports.\n"
        "import json, zipfile\n"
        "report_path = OUTPUT_DIR / 'hmr2s_paired_world_full11_3dpw.json'\n"
        "csv_path = OUTPUT_DIR / 'hmr2s_paired_world_full11_3dpw.csv'\n"
        "report = json.loads(report_path.read_text())\n"
        "print(json.dumps({'population': report['population'], 'iphone_minus_original_mean': report['iphone_minus_original_mean']}, indent=2))\n"
        "manifest = {path.name: {'bytes': path.stat().st_size, 'sha256': sha256_file(path)} for path in (report_path, csv_path)}\n"
        "manifest_path = OUTPUT_DIR / 'artifact_manifest.json'\n"
        "manifest_path.write_text(json.dumps(manifest, indent=2) + '\\n')\n"
        "bundle = Path('/kaggle/working/hmr2s_paired_world_full11_3dpw_reports.zip')\n"
        "with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:\n"
        "    for path in (report_path, csv_path, manifest_path):\n"
        "        archive.write(path, arcname=path.name)\n"
        "print('Download this:', bundle)\n"
    )
    for cell in cells:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
