#!/usr/bin/env python3
"""Build the BEDLAM adaptation plus locked-test Kaggle notebook."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "bedlam_retrain_and_test_tiny_pipeline_kaggle.ipynb"
WHAM_COMMIT = "2b54f7797391c94876848b905ed875b154c4a295"
SOURCE_DEPLOYMENT_SHA256 = (
    "d47ac0c855f44f84b67b20b9abb2cd3a66af702de479692350723cd79815c2e4"
)
SOURCES = (
    ROOT / "train_bedlam_tiny_pipeline.py",
    ROOT / "train_deployment_tiny_pipeline.py",
    ROOT / "evaluate_deployment_tiny_pipeline_3dpw.py",
    ROOT / "evaluate_wham_feature_substitution.py",
    ROOT / "evaluate_mobile_pipeline_3dpw.py",
    ROOT / "evaluate_full_pipeline_tradeoff.py",
    ROOT / "finetune_fastvit_wham_downstream.py",
)


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(True)}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(True),
    }


def embedded(path: Path) -> tuple[str, str]:
    contents = path.read_bytes()
    return (
        hashlib.sha256(contents).hexdigest(),
        base64.b64encode(gzip.compress(contents, compresslevel=9)).decode("ascii"),
    )


def main() -> None:
    payloads = {path.name: embedded(path) for path in SOURCES}
    embedded_lines = "\n".join(
        f"    {name!r}: ({digest!r}, {payload!r}),"
        for name, (digest, payload) in payloads.items()
    )
    notebook = {
        "cells": [
            markdown(
                "# BEDLAM retraining + one locked 3DPW test for the tiny iPhone pipeline\n\n"
                "This notebook starts from the latest real deployment checkpoint and performs one "
                "controlled experiment:\n\n"
                "1. verify the uploaded BEDLAM SMPL labels and gated Hugging Face access;\n"
                "2. download a diverse, size-bounded set of BEDLAM MP4 scene archives one at a time;\n"
                "3. train on actual YOLO26-matched person crops using BEDLAM camera-space SMPL pose;\n"
                "4. adapt only FastViT's late layers, the learned initializer, and WHAM's input-facing "
                "modules—WHAM recurrent weights stay frozen;\n"
                "5. recalibrate on 3DPW train, selecting every candidate only on 3DPW validation; and\n"
                "6. run the selected checkpoint once on the same 11-track 3DPW test population.\n\n"
                "The source checkpoint remains an eligible validation baseline. If BEDLAM adaptation "
                "does not improve validation, it cannot replace the existing model. The actual tiny "
                "path remains `YOLO26 → FastViT → learned initializer → split WHAM`; HMR2 is not part "
                "of inference.\n\n"
                "## Attach exactly these five inputs\n\n"
                "- `bedlam`: the private dataset shown in your screenshot, containing "
                "`bedlam-labels/*.npz`;\n"
                "- `3dpw-model`: raw `imageFiles`, `sequenceFiles`, and `3dpw_test_vit.pth`;\n"
                "- `3dpw-vit`: `3dpw_train_vit.pth` and `3dpw_val_vit.pth`;\n"
                "- your private licensed SMPL model dataset with neutral, male, and female PKL files;\n"
                "- the saved output of the latest tiny-pipeline notebook containing "
                "`tiny_pipeline_best.pth` with SHA-256 `d47ac0c8…`.\n\n"
                "Do **not** attach COCO, BEDLAM PNGs, HMR2a, or the old phase-two/phase-three notebook "
                "outputs. Add a Kaggle secret named `HF_TOKEN`, allow this notebook to use it, enable "
                "Internet, and select a GPU. The Hugging Face account behind the token must already "
                "have BEDLAM access.\n\n"
                "Default T4 runtime is expected to be about 3–5 hours. Downloads are capped at 2 GiB "
                "and removed scene-by-scene. Progress is printed during downloads, YOLO processing, "
                "training, validation, and the final test.\n"
            ),
            code(
                "# Configuration and filename/hash-based input discovery.\n"
                "from pathlib import Path\n"
                "import hashlib, os\n\n"
                "KAGGLE_INPUT = Path('/kaggle/input')\n"
                "SCRATCH_DIR = Path('/tmp/bedlam_tiny_pipeline')\n"
                "CACHE_DIR = SCRATCH_DIR / 'cache'\n"
                "OUTPUT_DIR = Path('/kaggle/working/bedlam_tiny_pipeline')\n\n"
                "# Bounded BEDLAM run: six diverse scene archives, no more than 2 GiB total.\n"
                "MAXIMUM_SCENES = 6\n"
                "MAXIMUM_DOWNLOAD_GIB = 2.0\n"
                "VIDEOS_PER_SCENE = 32\n"
                "FRAMES_PER_VIDEO = 30\n"
                "CLIPS_PER_VIDEO = 12\n"
                "BEDLAM_CLIP_LENGTH = 8\n"
                "BEDLAM_CLIP_STRIDE = 4\n"
                "BEDLAM_BATCH_SIZE = 2\n"
                "BEDLAM_FRAME_BATCH_SIZE = 24\n\n"
                "# Final real-data calibration and evaluation.\n"
                "THREEDPW_CLIP_LENGTH = 24\n"
                "THREEDPW_STRIDE = 12\n"
                "THREEDPW_MAX_CLIPS = 1200\n"
                "THREEDPW_BATCH_SIZE = 2\n"
                "THREEDPW_JOINT_EPOCHS = 2\n"
                "THREEDPW_LAST_STAGE_EPOCHS = 1\n"
                "VAL_TRACKS = 8\n"
                "VAL_FRAMES = 300\n"
                "YOLO_BATCH_SIZE = 32\n"
                "FEATURE_BATCH_SIZE = 64\n"
                "WORKERS = 4\n"
                "SMPL_BATCH_SIZE = 256  # reduce only this to 128 if final SMPL decode OOMs\n\n"
                f"EXPECTED_SOURCE_SHA256 = {SOURCE_DEPLOYMENT_SHA256!r}\n\n"
                "def sha256_file(path):\n"
                "    digest = hashlib.sha256()\n"
                "    with path.open('rb') as stream:\n"
                "        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):\n"
                "            digest.update(chunk)\n"
                "    return digest.hexdigest()\n\n"
                "def exactly_one(filename):\n"
                "    matches = sorted(KAGGLE_INPUT.rglob(filename))\n"
                "    if len(matches) != 1:\n"
                "        raise FileNotFoundError(\n"
                "            f'Expected exactly one {filename} in Kaggle inputs; found {matches}'\n"
                "        )\n"
                "    return matches[0]\n\n"
                "bedlam_roots = sorted({\n"
                "    path.parent for path in KAGGLE_INPUT.rglob('*.npz')\n"
                "    if path.name.startswith('202210')\n"
                "})\n"
                "bedlam_roots = [\n"
                "    root for root in bedlam_roots\n"
                "    if len(list(root.glob('202210*.npz'))) >= 20\n"
                "]\n"
                "if len(bedlam_roots) != 1:\n"
                "    raise FileNotFoundError(\n"
                "        'Attach exactly one extracted BEDLAM label dataset containing at least '\n"
                "        f'20 scene NPZ files; candidates={bedlam_roots}'\n"
                "    )\n"
                "BEDLAM_LABEL_ROOT = bedlam_roots[0]\n"
                "TRAIN_PARSED = exactly_one('3dpw_train_vit.pth')\n"
                "VAL_PARSED = exactly_one('3dpw_val_vit.pth')\n"
                "TEST_PARSED = exactly_one('3dpw_test_vit.pth')\n"
                "source_candidates = sorted(KAGGLE_INPUT.rglob('tiny_pipeline_best.pth'))\n"
                "source_matches = [\n"
                "    path for path in source_candidates\n"
                "    if sha256_file(path) == EXPECTED_SOURCE_SHA256\n"
                "]\n"
                "if len(source_matches) != 1:\n"
                "    found = [(str(path), sha256_file(path)) for path in source_candidates]\n"
                "    raise FileNotFoundError(\n"
                "        'Attach the one latest deployment notebook output containing '\n"
                "        f'{EXPECTED_SOURCE_SHA256}; found={found}'\n"
                "    )\n"
                "SOURCE_CHECKPOINT = source_matches[0]\n\n"
                "def find_3dpw_root(test_path):\n"
                "    for candidate in (test_path.parent, *test_path.parents):\n"
                "        has_images = any((candidate / name).is_dir() for name in ('imageFiles', '3DPW'))\n"
                "        has_sequences = (candidate / 'sequenceFiles').is_dir() or any(candidate.rglob('sequenceFiles'))\n"
                "        if has_images and has_sequences:\n"
                "            return candidate\n"
                "        if candidate == KAGGLE_INPUT:\n"
                "            break\n"
                "    raise FileNotFoundError(f'Could not resolve raw 3DPW root from {test_path}')\n\n"
                "THREEDPW_ROOT = find_3dpw_root(TEST_PARSED)\n"
                "print({\n"
                "    'bedlam_labels': str(BEDLAM_LABEL_ROOT),\n"
                "    'source_checkpoint': str(SOURCE_CHECKPOINT),\n"
                "    'source_sha256': sha256_file(SOURCE_CHECKPOINT),\n"
                "    '3dpw_root': str(THREEDPW_ROOT),\n"
                "    '3dpw_train': str(TRAIN_PARSED),\n"
                "    '3dpw_validation': str(VAL_PARSED),\n"
                "    '3dpw_locked_test': str(TEST_PARSED),\n"
                "    'output': str(OUTPUT_DIR),\n"
                "})\n"
            ),
            code(
                "# Install only the runtime dependencies; Kaggle's CUDA PyTorch is retained.\n"
                "%pip install -q timm==1.0.22 ultralytics==8.4.146 einops==0.8.1 "
                "yacs==0.1.8 joblib==1.5.2 loguru==0.7.3 smplx==0.1.28 "
                "chumpy==0.70 opencv-python-headless==4.10.0.84 "
                "scikit-image==0.25.2 tqdm==4.67.1 huggingface_hub==0.36.0 "
                "'hf_xet>=1.1.5,<2'\n"
            ),
            code(
                "# Read the gated-dataset token without printing it.\n"
                "from kaggle_secrets import UserSecretsClient\n\n"
                "try:\n"
                "    HF_TOKEN = UserSecretsClient().get_secret('HF_TOKEN').strip()\n"
                "except Exception as error:\n"
                "    raise RuntimeError(\n"
                "        'Kaggle secret HF_TOKEN is missing or not enabled for this notebook.'\n"
                "    ) from error\n"
                "if not HF_TOKEN:\n"
                "    raise RuntimeError('HF_TOKEN is empty')\n"
                "os.environ['HF_TOKEN'] = HF_TOKEN\n"
                "print('HF_TOKEN is available (value intentionally hidden).')\n"
            ),
            code(
                "# Materialize the reviewed, checksum-verified programs embedded in this notebook.\n"
                "import base64, gzip\n\n"
                "SCRATCH_DIR.mkdir(parents=True, exist_ok=True)\n"
                "embedded = {\n"
                f"{embedded_lines}\n"
                "}\n"
                "for name, (expected, payload) in embedded.items():\n"
                "    contents = gzip.decompress(base64.b64decode(payload))\n"
                "    actual = hashlib.sha256(contents).hexdigest()\n"
                "    if actual != expected:\n"
                "        raise RuntimeError(f'Embedded script checksum mismatch for {name}')\n"
                "    (SCRATCH_DIR / name).write_bytes(contents)\n"
                "print({name: digest for name, (digest, _) in embedded.items()})\n"
            ),
            code(
                "# Fetch pinned public WHAM/YOLO evaluation assets and locate licensed SMPL files.\n"
                "import shutil, subprocess, urllib.request\n\n"
                "WHAM_REPO = SCRATCH_DIR / 'WHAM'\n"
                "if not (WHAM_REPO / 'lib/models/wham.py').is_file():\n"
                "    subprocess.run([\n"
                "        'git', 'clone', '--filter=blob:none', '--no-checkout',\n"
                "        'https://github.com/yohanshin/WHAM.git', str(WHAM_REPO),\n"
                "    ], check=True)\n"
                f"    subprocess.run(['git', 'checkout', '--detach', {WHAM_COMMIT!r}], "
                "cwd=WHAM_REPO, check=True)\n"
                "actual_commit = subprocess.check_output(\n"
                "    ['git', 'rev-parse', 'HEAD'], cwd=WHAM_REPO, text=True\n"
                ").strip()\n"
                f"if actual_commit != {WHAM_COMMIT!r}:\n"
                "    raise RuntimeError(f'Wrong WHAM commit: {actual_commit}')\n\n"
                "downloads = {\n"
                "    'wham_vit_bedlam_w_3dpw.pth.tar': (\n"
                "        'https://huggingface.co/camenduru/WHAM/resolve/main/'\n"
                "        'wham_vit_bedlam_w_3dpw.pth.tar?download=true',\n"
                "        '2ba0cb6a7dd597023a6b2ad6056e7a8b6b33144a35fabea570bfd00842cd4eaf',\n"
                "    ),\n"
                "    'yolo26n-pose.pt': (\n"
                "        'https://github.com/ultralytics/assets/releases/download/v8.4.0/'\n"
                "        'yolo26n-pose.pt',\n"
                "        'eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9',\n"
                "    ),\n"
                "    'J_regressor_h36m.npy': (\n"
                "        'https://huggingface.co/camenduru/WHAM/resolve/main/'\n"
                "        'J_regressor_h36m.npy?download=true',\n"
                "        'c655cd7013d7829eb9acbebf0e43f952a3fa0305a53c35880e39192bfb6444a0',\n"
                "    ),\n"
                "}\n"
                "downloaded = {}\n"
                "for name, (url, expected) in downloads.items():\n"
                "    path = SCRATCH_DIR / name\n"
                "    if not path.is_file():\n"
                "        print(f'Downloading {name}...', flush=True)\n"
                "        urllib.request.urlretrieve(url, path)\n"
                "    actual = sha256_file(path)\n"
                "    if actual != expected:\n"
                "        raise RuntimeError(f'{name} checksum mismatch: {actual}')\n"
                "    downloaded[name] = path\n"
                "WHAM_CHECKPOINT = downloaded['wham_vit_bedlam_w_3dpw.pth.tar']\n"
                "YOLO26_WEIGHTS = downloaded['yolo26n-pose.pt']\n"
                "H36M_REGRESSOR = downloaded['J_regressor_h36m.npy']\n\n"
                "SMPL_MODEL_DIR = SCRATCH_DIR / 'smpl'\n"
                "SMPL_MODEL_DIR.mkdir(parents=True, exist_ok=True)\n"
                "model_aliases = {\n"
                "    'SMPL_NEUTRAL.pkl': [\n"
                "        'SMPL_NEUTRAL.pkl', 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'\n"
                "    ],\n"
                "    'SMPL_MALE.pkl': [\n"
                "        'SMPL_MALE.pkl', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl',\n"
                "        'basicModel_m_lbs_10_207_0_v1.0.0.pkl'\n"
                "    ],\n"
                "    'SMPL_FEMALE.pkl': [\n"
                "        'SMPL_FEMALE.pkl', 'basicModel_f_lbs_10_207_0_v1.0.0.pkl'\n"
                "    ],\n"
                "}\n"
                "def licensed_asset(names):\n"
                "    matches = sorted({path for name in names for path in KAGGLE_INPUT.rglob(name)})\n"
                "    if not matches:\n"
                "        raise FileNotFoundError(\n"
                "            f'Missing licensed SMPL asset {names}; attach your private SMPL dataset.'\n"
                "        )\n"
                "    return matches[0]\n"
                "for output_name, aliases in model_aliases.items():\n"
                "    source = licensed_asset(aliases)\n"
                "    shutil.copy2(source, SMPL_MODEL_DIR / output_name)\n"
                "    print(f'{output_name}: {source}')\n"
                "print({'wham_commit': actual_commit, **{name: sha256_file(path) for name, path in downloaded.items()}})\n"
            ),
            code(
                "# Fast fail before the long run: local math, files, labels, HF access, and archive sizes.\n"
                "import sys\n\n"
                "TRAINER = SCRATCH_DIR / 'train_bedlam_tiny_pipeline.py'\n"
                "common = [\n"
                "    '--bedlam-label-root', str(BEDLAM_LABEL_ROOT),\n"
                "    '--output-dir', str(OUTPUT_DIR),\n"
                "    '--scratch-dir', str(SCRATCH_DIR / 'runtime'),\n"
                "    '--cache-dir', str(CACHE_DIR),\n"
                "    '--three-dpw-root', str(THREEDPW_ROOT),\n"
                "    '--sequence-root', str(THREEDPW_ROOT),\n"
                "    '--train-parsed', str(TRAIN_PARSED),\n"
                "    '--val-parsed', str(VAL_PARSED),\n"
                "    '--source-checkpoint', str(SOURCE_CHECKPOINT),\n"
                "    '--wham-repo', str(WHAM_REPO),\n"
                "    '--wham-checkpoint', str(WHAM_CHECKPOINT),\n"
                "    '--yolo26-weights', str(YOLO26_WEIGHTS),\n"
                "    '--maximum-scenes', str(MAXIMUM_SCENES),\n"
                "    '--maximum-download-gib', str(MAXIMUM_DOWNLOAD_GIB),\n"
                "    '--val-tracks', str(VAL_TRACKS),\n"
                "    '--val-frames', str(VAL_FRAMES),\n"
                "]\n"
                "environment = os.environ.copy()\n"
                "environment['PYTHONPATH'] = str(SCRATCH_DIR) + os.pathsep + environment.get('PYTHONPATH', '')\n"
                "subprocess.run([sys.executable, '-u', str(TRAINER), '--self-test'], check=True, env=environment)\n"
                "subprocess.run([sys.executable, '-u', str(TRAINER), '--inspect-data', *common], check=True, env=environment)\n"
            ),
            code(
                "# Long cell: bounded BEDLAM adaptation followed by real 3DPW calibration.\n"
                "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
                "training_command = [\n"
                "    sys.executable, '-u', str(TRAINER), *common,\n"
                "    '--videos-per-scene', str(VIDEOS_PER_SCENE),\n"
                "    '--frames-per-video', str(FRAMES_PER_VIDEO),\n"
                "    '--clips-per-video', str(CLIPS_PER_VIDEO),\n"
                "    '--bedlam-clip-length', str(BEDLAM_CLIP_LENGTH),\n"
                "    '--bedlam-clip-stride', str(BEDLAM_CLIP_STRIDE),\n"
                "    '--bedlam-batch-size', str(BEDLAM_BATCH_SIZE),\n"
                "    '--bedlam-frame-batch-size', str(BEDLAM_FRAME_BATCH_SIZE),\n"
                "    '--yolo-batch-size', str(YOLO_BATCH_SIZE),\n"
                "    '--three-dpw-clip-length', str(THREEDPW_CLIP_LENGTH),\n"
                "    '--three-dpw-stride', str(THREEDPW_STRIDE),\n"
                "    '--three-dpw-max-clips', str(THREEDPW_MAX_CLIPS),\n"
                "    '--three-dpw-batch-size', str(THREEDPW_BATCH_SIZE),\n"
                "    '--three-dpw-joint-epochs', str(THREEDPW_JOINT_EPOCHS),\n"
                "    '--three-dpw-last-stage-epochs', str(THREEDPW_LAST_STAGE_EPOCHS),\n"
                "    '--workers', str(WORKERS),\n"
                "    '--feature-batch-size', str(FEATURE_BATCH_SIZE),\n"
                "    '--log-every', '25',\n"
                "]\n"
                "print('Starting BEDLAM adaptation and validation-locked calibration...', flush=True)\n"
                "subprocess.run(training_command, check=True, env=environment)\n"
                "DEPLOYMENT_CHECKPOINT = OUTPUT_DIR / 'bedlam_tiny_pipeline_best.pth'\n"
                "TRAINING_REPORT = OUTPUT_DIR / 'bedlam_training_report.json'\n"
                "TRAINING_HISTORY = OUTPUT_DIR / 'bedlam_training_history.csv'\n"
                "DOWNLOAD_MANIFEST = OUTPUT_DIR / 'bedlam_download_manifest.json'\n"
                "for path in (DEPLOYMENT_CHECKPOINT, TRAINING_REPORT, TRAINING_HISTORY, DOWNLOAD_MANIFEST):\n"
                "    if not path.is_file():\n"
                "        raise FileNotFoundError(f'Training did not produce {path}')\n"
            ),
            code(
                "# One confirmatory 3DPW test after all validation-based selection is finished.\n"
                "EVALUATOR = SCRATCH_DIR / 'evaluate_deployment_tiny_pipeline_3dpw.py'\n"
                "TEST_REPORT = OUTPUT_DIR / 'bedlam_tiny_pipeline_final_3dpw.json'\n"
                "TEST_CSV = OUTPUT_DIR / 'bedlam_tiny_pipeline_final_3dpw.csv'\n"
                "test_command = [\n"
                "    sys.executable, '-u', str(EVALUATOR),\n"
                "    '--parsed-3dpw', str(TEST_PARSED),\n"
                "    '--three-dpw-root', str(THREEDPW_ROOT),\n"
                "    '--deployment-checkpoint', str(DEPLOYMENT_CHECKPOINT),\n"
                "    '--wham-repo', str(WHAM_REPO),\n"
                "    '--wham-checkpoint', str(WHAM_CHECKPOINT),\n"
                "    '--yolo26-weights', str(YOLO26_WEIGHTS),\n"
                "    '--smpl-model-directory', str(SMPL_MODEL_DIR),\n"
                "    '--h36m-joint-regressor', str(H36M_REGRESSOR),\n"
                "    '--pose-batch-size', str(YOLO_BATCH_SIZE),\n"
                "    '--student-batch-size', str(FEATURE_BATCH_SIZE),\n"
                "    '--smpl-batch-size', str(SMPL_BATCH_SIZE),\n"
                "    '--output', str(TEST_REPORT),\n"
                "    '--per-sequence-output', str(TEST_CSV),\n"
                "]\n"
                "print('Selection is locked. Starting the single confirmatory 3DPW test...', flush=True)\n"
                "subprocess.run(test_command, check=True, env=environment)\n"
            ),
            code(
                "# Compact result tables and one download bundle.\n"
                "import json, pandas as pd, shutil\n"
                "from IPython.display import display\n\n"
                "training = json.loads(TRAINING_REPORT.read_text())\n"
                "test = json.loads(TEST_REPORT.read_text())\n"
                "baseline = training['baseline_validation']\n"
                "best = training['best_validation']\n"
                "display(pd.DataFrame([\n"
                "    {\n"
                "        'validation model': 'Source tiny checkpoint',\n"
                "        'selection score': baseline['selection_score'],\n"
                "        'pose error (deg)': baseline['pose_error_deg'],\n"
                "        'root error (deg)': baseline['root_error_deg'],\n"
                "        'shape RMSE': baseline['shape_rmse'],\n"
                "    },\n"
                "    {\n"
                "        'validation model': f\"Selected: {training['best_stage']}\",\n"
                "        'selection score': best['selection_score'],\n"
                "        'pose error (deg)': best['pose_error_deg'],\n"
                "        'root error (deg)': best['root_error_deg'],\n"
                "        'shape RMSE': best['shape_rmse'],\n"
                "    },\n"
                "]).round(4))\n\n"
                "comparison = test['comparison']\n"
                "original = comparison['reference_metrics']\n"
                "tiny = comparison['tiny_metrics']\n"
                "display(pd.DataFrame([\n"
                "    {\n"
                "        'test pipeline': 'Original released WHAM',\n"
                "        'PA-MPJPE (mm)': original['pa_mpjpe_mm'],\n"
                "        'MPJPE (mm)': original['mpjpe_mm'],\n"
                "        'PVE (mm)': original['pve_mm'],\n"
                "        'Accel': original['accel_official_30fps'],\n"
                "    },\n"
                "    {\n"
                "        'test pipeline': 'Tiny after BEDLAM + calibration',\n"
                "        'PA-MPJPE (mm)': tiny['pa_mpjpe_mm']['mean'],\n"
                "        'MPJPE (mm)': tiny['mpjpe_mm']['mean'],\n"
                "        'PVE (mm)': tiny['pve_mm']['mean'],\n"
                "        'Accel': tiny['accel_official_30fps']['mean'],\n"
                "    },\n"
                "]).round(3))\n"
                "print('Selected checkpoint:', json.dumps({\n"
                "    'stage': training['best_stage'],\n"
                "    'epoch': training['best_epoch'],\n"
                "    'sha256': training['best_checkpoint_sha256'],\n"
                "    'validation_delta': training['validation_delta'],\n"
                "}, indent=2))\n"
                "print('Tiny minus original:', json.dumps(comparison['tiny_minus_original'], indent=2))\n"
                "print('Relative change:', json.dumps({\n"
                "    key: f'{100 * value:+.1f}%'\n"
                "    for key, value in comparison['tiny_relative_change_vs_original'].items()\n"
                "}, indent=2))\n\n"
                "bundle = Path(shutil.make_archive(\n"
                "    '/kaggle/working/bedlam_tiny_pipeline_results',\n"
                "    'zip', root_dir=OUTPUT_DIR,\n"
                "))\n"
                "print(f'Download this one file: {bundle}')\n"
                "print('Bundle contents:', sorted(path.name for path in OUTPUT_DIR.iterdir()))\n"
                "shutil.rmtree(SCRATCH_DIR, ignore_errors=True)\n"
            ),
            markdown(
                "## Return only these files\n\n"
                "Download `bedlam_tiny_pipeline_results.zip`. If that is inconvenient, the minimum "
                "diagnostic pair is `bedlam_training_report.json` and "
                "`bedlam_tiny_pipeline_final_3dpw.json`. Keep the selected PTH in the saved Kaggle "
                "output; it is needed later for Core ML export and the iPhone latency run.\n"
            ),
        ],
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"isGpuEnabled": True, "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print({name: digest for name, (digest, _) in payloads.items()})


if __name__ == "__main__":
    main()
