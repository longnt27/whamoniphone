"""Build the paired zero-camera versus 3DPW-oracle Kaggle notebook."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALUATION = HERE.parent / "evaluation"
BASE = HERE / "hmr2s_temporal_smoothing_kaggle.ipynb"
OUTPUT = HERE / "hmr2s_camera_oracle_full11_3dpw_v2_kaggle.ipynb"
ADAPTER_SHA256 = "4fd581b2b7f2d0cac8bda7597692f7e77ca082435c5e352c69964d64da10f526"
SOURCES = (
    "camera_motion_oracle.py",
    "evaluate_hmr2s_camera_oracle.py",
    "evaluate_hmr2s_temporal_smoothing.py",
    "hmr2s_frozen.py",
    "evaluate_frozen_hmr2s_wham.py",
    "evaluate_full_pipeline_tradeoff.py",
    "evaluate_mobile_pipeline_3dpw.py",
    "evaluate_wham_feature_substitution.py",
)


def source(text: str) -> list[str]:
    return text.splitlines(True)


def main() -> None:
    notebook = json.loads(BASE.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    if len(cells) != 8:
        raise RuntimeError(f"Expected eight reviewed base cells, got {len(cells)}")
    cells[0]["source"] = source(
        "# Full 11-track 3DPW test: ground-truth camera-motion oracle\n\n"
        "This notebook runs the **same locked tiny pipeline** twice on the **same 3DPW test frames**: "
        "YOLO26m-pose → released HMR2.0-S → selected token adapter → released WHAM → locked light output smoothing. "
        "The only changed WHAM input is camera angular motion: all zeros versus the official WHAM "
        "6D encoding computed from 3DPW ground-truth camera rotations. Nothing is trained, tuned, or selected. "
        "The notebook asserts the **exact 11 single-person tracks and all 11,349 recurrent frames** from the "
        "previous body-pose report. It fails instead of silently evaluating a shortened subset.\n\n"
        "This is an **oracle-input experiment, not a phone gyroscope test**. 3DPW camera poses are privileged "
        "offline labels. They contain no actual iPhone 11 Pro Max sensor stream, timing offset, drift, noise, or dropout. "
        "Only camera **rotation**, not translation, enters WHAM. We report paired first-frame-aligned "
        "world-root orientation/displacement and world 14-joint error; these are **not** the paper's "
        "official 100-frame W-MPJPE/W-PA-MPJPE. "
        "Camera-relative pose/shape metrics are expected to be unchanged because camera motion enters a separate "
        "WHAM trajectory branch. All body-pose frames are processed. A transition with an invalid camera pose "
        "or skipped raw frame ID uses zero motion only for that transition; the report counts frame gaps. "
        "World-space metrics score every frame with finite world-body labels, "
        "with exact coverage reported. This GPU notebook measures no iPhone latency.\n\n"
        "## Kaggle inputs to attach\n\n"
        "1. `3dpw-model` with `imageFiles/imageFiles`, `sequenceFiles/sequenceFiles/test`, "
        "`3dpw_test_vit.pth`, and the licensed SMPL neutral/male/female model files.\n"
        "2. Saved output of the completed YOLO26 grid notebook containing "
        "`m/hmr2s_to_hmr2a_adapter_best.pth` (verified SHA-256).\n"
        "3. Your private official HMR2.0-S `hmr_vit-small_d3-a4x16-m128.zip` or verified `last.ckpt`.\n\n"
        "Enable a Kaggle **GPU** and **Internet**. No COCO, BEDLAM, `3dpw-vit` validation set, or HF token is required. "
        "At the end, download `hmr2s_camera_oracle_full11_3dpw_reports.zip` and send it back. "
        "Import this v2 notebook as a new Kaggle notebook; an already-saved Kaggle notebook "
        "does not update when this local file changes. Do **not** rerun the earlier version."
    )
    cells[1]["source"] = source(
        "# Configuration and bounded input discovery.\n"
        "from pathlib import Path\n"
        "import hashlib, os\n\n"
        "KAGGLE_INPUT = Path('/kaggle/input')\n"
        "SCRATCH_DIR = Path('/tmp/hmr2s_camera_oracle_full11')\n"
        "OUTPUT_DIR = Path('/kaggle/working/hmr2s_camera_oracle_full11')\n"
        f"ADAPTER_SHA256 = {ADAPTER_SHA256!r}\n"
        "POSE_BATCH_SIZE = 24\n"
        "HMR2S_BATCH_SIZE = 24\n"
        "SMPL_BATCH_SIZE = 192\n\n"
        "def sha256_file(path):\n"
        "    digest = hashlib.sha256()\n"
        "    with path.open('rb') as stream:\n"
        "        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):\n"
        "            digest.update(block)\n"
        "    return digest.hexdigest()\n\n"
        "def first_directory(candidates, description):\n"
        "    result = next((path for path in candidates if path.is_dir()), None)\n"
        "    if result is None:\n"
        "        raise FileNotFoundError(f'Missing {description}; checked {candidates}')\n"
        "    return result\n\n"
        "THREEDPW_ROOT = first_directory([\n"
        "    Path('/kaggle/input/datasets/nguyntrunglong/3dpw-model'),\n"
        "    Path('/kaggle/input/3dpw-model'),\n"
        "], '3dpw-model')\n"
        "TEST_PARSED = THREEDPW_ROOT / '3dpw_test_vit.pth'\n"
        "if not TEST_PARSED.is_file():\n"
        "    matches = list(THREEDPW_ROOT.glob('*/3dpw_test_vit.pth'))\n"
        "    if len(matches) != 1:\n"
        "        raise FileNotFoundError(f'Missing 3dpw_test_vit.pth below {THREEDPW_ROOT}')\n"
        "    TEST_PARSED = matches[0]\n\n"
        "def bounded_files(names):\n"
        "    found = []\n"
        "    for parent, directories, files in os.walk(KAGGLE_INPUT):\n"
        "        directories[:] = [name for name in directories if name not in {'imageFiles', 'sequenceFiles', 'coco2017'}]\n"
        "        for filename in files:\n"
        "            if filename in names:\n"
        "                found.append(Path(parent) / filename)\n"
        "    return found\n\n"
        "adapter_candidates = bounded_files({'hmr2s_to_hmr2a_adapter_best.pth'})\n"
        "adapter_matches = [path for path in adapter_candidates if sha256_file(path) == ADAPTER_SHA256]\n"
        "if len(adapter_matches) != 1:\n"
        "    raise FileNotFoundError(f'Attach the saved YOLO26 grid output with the selected m adapter; verified matches={adapter_matches}')\n"
        "ADAPTER_CHECKPOINT = adapter_matches[0]\n"
        "print({'3dpw_test': str(TEST_PARSED), 'adapter': str(ADAPTER_CHECKPOINT)}, flush=True)\n"
    )
    embedded_lines = []
    for name in SOURCES:
        payload = (EVALUATION / name).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        compressed = base64.b64encode(gzip.compress(payload, compresslevel=9)).decode("ascii")
        embedded_lines.append(f"    {name!r}: ({digest!r}, {compressed!r}),")
    cells[3]["source"] = source(
        "# Materialize the checksum-verified evaluation sources.\n"
        "import base64, gzip\n\n"
        "SCRATCH_DIR.mkdir(parents=True, exist_ok=True)\n"
        "embedded = {\n"
        + "\n".join(embedded_lines)
        + "\n}\n"
        "for name, (expected, payload) in embedded.items():\n"
        "    contents = gzip.decompress(base64.b64decode(payload))\n"
        "    if hashlib.sha256(contents).hexdigest() != expected:\n"
        "        raise RuntimeError(f'Embedded source checksum mismatch: {name}')\n"
        "    (SCRATCH_DIR / name).write_bytes(contents)\n"
        "print({name: digest for name, (digest, _) in embedded.items()}, flush=True)\n"
    )
    cells[6]["source"] = source(
        "# Self-test, then run the locked paired test experiment.\n"
        "import sys\n\n"
        "EVALUATOR = SCRATCH_DIR / 'evaluate_hmr2s_camera_oracle.py'\n"
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
        "    '--output-json', str(OUTPUT_DIR / 'hmr2s_camera_oracle_full11_3dpw.json'),\n"
        "    '--output-csv', str(OUTPUT_DIR / 'hmr2s_camera_oracle_full11_3dpw.csv'),\n"
        "    '--pose-batch-size', str(POSE_BATCH_SIZE),\n"
        "    '--hmr-batch-size', str(HMR2S_BATCH_SIZE),\n"
        "    '--smpl-batch-size', str(SMPL_BATCH_SIZE),\n"
        "]\n"
        "subprocess.run([sys.executable, '-u', str(EVALUATOR), *arguments], check=True, env=environment)\n"
    )
    cells[7]["source"] = source(
        "# Show the paired result and package the small reports.\n"
        "import json, zipfile\n"
        "report_path = OUTPUT_DIR / 'hmr2s_camera_oracle_full11_3dpw.json'\n"
        "csv_path = OUTPUT_DIR / 'hmr2s_camera_oracle_full11_3dpw.csv'\n"
        "report = json.loads(report_path.read_text())\n"
        "print(json.dumps({'population': report['population'], 'oracle_minus_zero_mean': report['oracle_minus_zero_mean']}, indent=2))\n"
        "manifest = {path.name: {'bytes': path.stat().st_size, 'sha256': sha256_file(path)} for path in (report_path, csv_path)}\n"
        "manifest_path = OUTPUT_DIR / 'artifact_manifest.json'\n"
        "manifest_path.write_text(json.dumps(manifest, indent=2) + '\\n')\n"
        "bundle = Path('/kaggle/working/hmr2s_camera_oracle_full11_3dpw_reports.zip')\n"
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
