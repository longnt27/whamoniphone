#!/usr/bin/env python3
"""Build the one-run deployment-aware training plus locked-test Kaggle notebook."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "train_and_test_deployment_tiny_pipeline_kaggle.ipynb"
WHAM_COMMIT = "2b54f7797391c94876848b905ed875b154c4a295"
SOURCES = (
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
                "# Train and test the real tiny iPhone pipeline — one Kaggle run\n\n"
                "This notebook performs the complete remaining experiment without a second "
                "Kaggle run:\n\n"
                "1. cache actual YOLO26 observations on 3DPW train and validation;\n"
                "2. train compact FastViT pose/3D-joint initializer heads;\n"
                "3. adapt only WHAM's input-facing layers while keeping its recurrent weights frozen;\n"
                "4. select the checkpoint strictly on 3DPW validation; and\n"
                "5. immediately run that locked checkpoint once on the exact 11-track 3DPW test "
                "population used by the saved original-WHAM result.\n\n"
                "The evaluated tiny path is exactly `YOLO26 → FastViT → learned initializer → "
                "WHAM_I → WHAM_ImageStep`. HMR2 is not loaded or run. SMPL is used only after "
                "prediction to calculate MPJPE/PVE, never in tiny inference. The recurrent Python "
                "test uses the exported iPhone step's per-frame missing-feature behavior.\n\n"
                "Attach these existing inputs before running all cells with a GPU and Internet:\n\n"
                "- `3dpw-model` (raw images, sequence files, and `3dpw_test_vit.pth`);\n"
                "- `3dpw-vit` (`3dpw_train_vit.pth` and `3dpw_val_vit.pth`);\n"
                "- the saved phase-three notebook output containing the checkpoint whose SHA-256 "
                "starts with `f15875f3`; and\n"
                "- your private dataset containing the licensed neutral/male/female SMPL files.\n\n"
                "Expected T4 time is roughly 2–3 hours. Progress is printed during YOLO caching "
                "and every 25 training steps, so a healthy run will not appear frozen.\n"
            ),
            code(
                "# Configuration and exact input resolution. No generic WORK_DIR is used.\n"
                "from pathlib import Path\n"
                "import hashlib\n\n"
                "KAGGLE_INPUT = Path('/kaggle/input')\n"
                "THREEDPW_ROOT = Path('/kaggle/input/datasets/nguyntrunglong/3dpw-model')\n"
                "PARSED_3DPW_ROOT = Path('/kaggle/input/datasets/nguyntrunglong/3dpw-vit')\n"
                "SCRATCH_DIR = Path('/tmp/deployment_tiny_pipeline')\n"
                "CACHE_DIR = SCRATCH_DIR / 'cache'\n"
                "OUTPUT_DIR = Path('/kaggle/working/deployment_tiny_pipeline')\n\n"
                "CLIP_LENGTH = 24\n"
                "STRIDE = 12\n"
                "MAX_CLIPS = 1200\n"
                "TRAIN_BATCH_SIZE = 2\n"
                "WORKERS = 4\n"
                "YOLO_BATCH_SIZE = 32\n"
                "FEATURE_BATCH_SIZE = 64\n"
                "INITIALIZER_EPOCHS = 2\n"
                "JOINT_EPOCHS = 4\n"
                "LAST_STAGE_EPOCHS = 1\n"
                "VAL_TRACKS = 8\n"
                "VAL_FRAMES = 300\n"
                "SMPL_BATCH_SIZE = 256  # change only this to 128 if final metric decode OOMs\n\n"
                "EXPECTED_SOURCE_SHA256 = (\n"
                "    'f15875f3fed12538312f59956b6c93e9cca2ab41a9e8d87edf593dd85f311ab1'\n"
                ")\n\n"
                "def sha256_file(path):\n"
                "    digest = hashlib.sha256()\n"
                "    with path.open('rb') as stream:\n"
                "        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):\n"
                "            digest.update(chunk)\n"
                "    return digest.hexdigest()\n\n"
                "def exactly_one(root, filename):\n"
                "    matches = sorted(root.rglob(filename))\n"
                "    if len(matches) != 1:\n"
                "        raise FileNotFoundError(\n"
                "            f'Expected exactly one {filename} below {root}; found {matches}'\n"
                "        )\n"
                "    return matches[0]\n\n"
                "if not THREEDPW_ROOT.is_dir():\n"
                "    raise FileNotFoundError(f'Missing 3dpw-model input: {THREEDPW_ROOT}')\n"
                "if not PARSED_3DPW_ROOT.is_dir():\n"
                "    raise FileNotFoundError(f'Missing 3dpw-vit input: {PARSED_3DPW_ROOT}')\n"
                "TRAIN_PARSED = exactly_one(PARSED_3DPW_ROOT, '3dpw_train_vit.pth')\n"
                "VAL_PARSED = exactly_one(PARSED_3DPW_ROOT, '3dpw_val_vit.pth')\n"
                "TEST_PARSED = exactly_one(THREEDPW_ROOT, '3dpw_test_vit.pth')\n"
                "source_candidates = sorted(KAGGLE_INPUT.rglob('fastvit_hmr2_best.pth'))\n"
                "source_matches = [\n"
                "    path for path in source_candidates\n"
                "    if sha256_file(path) == EXPECTED_SOURCE_SHA256\n"
                "]\n"
                "if len(source_matches) != 1:\n"
                "    found = [(str(path), sha256_file(path)) for path in source_candidates]\n"
                "    raise FileNotFoundError(\n"
                "        'Attach exactly one saved phase-three checkpoint with SHA-256 '\n"
                "        f'{EXPECTED_SOURCE_SHA256}; found {found}'\n"
                "    )\n"
                "SOURCE_CHECKPOINT = source_matches[0]\n"
                "print({\n"
                "    'source_checkpoint': str(SOURCE_CHECKPOINT),\n"
                "    'train_parsed': str(TRAIN_PARSED),\n"
                "    'validation_parsed': str(VAL_PARSED),\n"
                "    'locked_test_parsed': str(TEST_PARSED),\n"
                "    'output': str(OUTPUT_DIR),\n"
                "})\n"
            ),
            code(
                "# Kaggle supplies CUDA PyTorch; keep it. Resolver warnings about unrelated packages are harmless.\n"
                "%pip install -q timm==1.0.22 ultralytics==8.4.146 einops==0.8.1 "
                "yacs==0.1.8 joblib==1.5.2 loguru==0.7.3 smplx==0.1.28 "
                "chumpy==0.70 opencv-python-headless==4.10.0.84 "
                "scikit-image==0.25.2 tqdm==4.67.1\n"
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
                "# Fetch pinned WHAM/YOLO assets and normalize the licensed SMPL filenames.\n"
                "import shutil, subprocess, urllib.request\n\n"
                "WHAM_REPO = SCRATCH_DIR / 'WHAM'\n"
                "if not (WHAM_REPO / 'lib/models/wham.py').is_file():\n"
                "    subprocess.run([\n"
                "        'git', 'clone', '--filter=blob:none', '--no-checkout',\n"
                "        'https://github.com/yohanshin/WHAM.git', str(WHAM_REPO),\n"
                "    ], check=True)\n"
                f"    subprocess.run(['git', 'checkout', '--detach', '{WHAM_COMMIT}'], "
                "cwd=WHAM_REPO, check=True)\n"
                "actual_commit = subprocess.check_output(\n"
                "    ['git', 'rev-parse', 'HEAD'], cwd=WHAM_REPO, text=True\n"
                ").strip()\n"
                f"if actual_commit != '{WHAM_COMMIT}':\n"
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
                "            f'Missing licensed SMPL asset {names}; attach your private asset dataset.'\n"
                "        )\n"
                "    return matches[0]\n"
                "for output_name, aliases in model_aliases.items():\n"
                "    source = licensed_asset(aliases)\n"
                "    shutil.copy2(source, SMPL_MODEL_DIR / output_name)\n"
                "    print(f'{output_name}: {source}')\n"
                "print({'wham_commit': actual_commit, **{name: sha256_file(path) for name, path in downloaded.items()}})\n"
            ),
            code(
                "# Fast preflight: code math, exact files, parsed fields, and raw-image mapping.\n"
                "import os, sys\n\n"
                "TRAINER = SCRATCH_DIR / 'train_deployment_tiny_pipeline.py'\n"
                "common = [\n"
                "    '--output-dir', str(OUTPUT_DIR),\n"
                "    '--cache-dir', str(CACHE_DIR),\n"
                "    '--three-dpw-root', str(THREEDPW_ROOT),\n"
                "    '--sequence-root', str(THREEDPW_ROOT),\n"
                "    '--train-parsed', str(TRAIN_PARSED),\n"
                "    '--val-parsed', str(VAL_PARSED),\n"
                "    '--source-checkpoint', str(SOURCE_CHECKPOINT),\n"
                "    '--wham-repo', str(WHAM_REPO),\n"
                "    '--wham-checkpoint', str(WHAM_CHECKPOINT),\n"
                "    '--yolo26-weights', str(YOLO26_WEIGHTS),\n"
                "    '--val-tracks', str(VAL_TRACKS),\n"
                "    '--val-frames', str(VAL_FRAMES),\n"
                "]\n"
                "environment = os.environ.copy()\n"
                "environment['PYTHONPATH'] = str(SCRATCH_DIR) + os.pathsep + environment.get('PYTHONPATH', '')\n"
                "subprocess.run([sys.executable, '-u', str(TRAINER), '--self-test', *common], check=True, env=environment)\n"
                "subprocess.run([sys.executable, '-u', str(TRAINER), '--inspect-data', *common], check=True, env=environment)\n"
            ),
            code(
                "# Deployment-aware training and validation selection. This is the long cell.\n"
                "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
                "training_command = [\n"
                "    sys.executable, '-u', str(TRAINER), *common,\n"
                "    '--clip-length', str(CLIP_LENGTH),\n"
                "    '--stride', str(STRIDE),\n"
                "    '--max-clips', str(MAX_CLIPS),\n"
                "    '--batch-size', str(TRAIN_BATCH_SIZE),\n"
                "    '--workers', str(WORKERS),\n"
                "    '--yolo-batch-size', str(YOLO_BATCH_SIZE),\n"
                "    '--feature-batch-size', str(FEATURE_BATCH_SIZE),\n"
                "    '--initializer-epochs', str(INITIALIZER_EPOCHS),\n"
                "    '--joint-epochs', str(JOINT_EPOCHS),\n"
                "    '--last-stage-epochs', str(LAST_STAGE_EPOCHS),\n"
                "    '--log-every', '25',\n"
                "]\n"
                "print('Starting one deployment-aware training run...', flush=True)\n"
                "subprocess.run(training_command, check=True, env=environment)\n"
                "DEPLOYMENT_CHECKPOINT = OUTPUT_DIR / 'tiny_pipeline_best.pth'\n"
                "TRAINING_REPORT = OUTPUT_DIR / 'deployment_training_report.json'\n"
                "TRAINING_HISTORY = OUTPUT_DIR / 'deployment_training_history.csv'\n"
                "for path in (DEPLOYMENT_CHECKPOINT, TRAINING_REPORT, TRAINING_HISTORY):\n"
                "    if not path.is_file():\n"
                "        raise FileNotFoundError(f'Training did not produce {path}')\n"
            ),
            code(
                "# Locked test: run the selected checkpoint immediately, once, on the saved population.\n"
                "EVALUATOR = SCRATCH_DIR / 'evaluate_deployment_tiny_pipeline_3dpw.py'\n"
                "TEST_REPORT = OUTPUT_DIR / 'tiny_pipeline_final_3dpw.json'\n"
                "TEST_CSV = OUTPUT_DIR / 'tiny_pipeline_final_3dpw.csv'\n"
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
                "print('Training is finished. Starting the one locked 3DPW test now...', flush=True)\n"
                "subprocess.run(test_command, check=True, env=environment)\n"
            ),
            code(
                "# Final original-vs-tiny table, validation provenance, and useful artifact bundle.\n"
                "import json, pandas as pd, shutil\n"
                "from IPython.display import display\n\n"
                "training = json.loads(TRAINING_REPORT.read_text())\n"
                "test = json.loads(TEST_REPORT.read_text())\n"
                "comparison = test['comparison']\n"
                "original = comparison['reference_metrics']\n"
                "tiny = comparison['tiny_metrics']\n"
                "display(pd.DataFrame([\n"
                "    {\n"
                "        'pipeline': 'Original released WHAM (saved)',\n"
                "        'PA-MPJPE (mm)': original['pa_mpjpe_mm'],\n"
                "        'MPJPE (mm)': original['mpjpe_mm'],\n"
                "        'PVE (mm)': original['pve_mm'],\n"
                "        'Accel': original['accel_official_30fps'],\n"
                "    },\n"
                "    {\n"
                "        'pipeline': 'Tiny accurate: YOLO26 + FastViT + split WHAM',\n"
                "        'PA-MPJPE (mm)': tiny['pa_mpjpe_mm']['mean'],\n"
                "        'MPJPE (mm)': tiny['mpjpe_mm']['mean'],\n"
                "        'PVE (mm)': tiny['pve_mm']['mean'],\n"
                "        'Accel': tiny['accel_official_30fps']['mean'],\n"
                "    },\n"
                "]).round(3))\n"
                "print('Validation-selected checkpoint:', json.dumps({\n"
                "    'epoch': training['best_epoch'],\n"
                "    'stage': training['best_stage'],\n"
                "    'checkpoint_sha256': training['best_checkpoint_sha256'],\n"
                "    'validation': {\n"
                "        key: value for key, value in training['best_validation'].items()\n"
                "        if key != 'per_track'\n"
                "    },\n"
                "}, indent=2))\n"
                "print('Tiny minus original:', json.dumps(comparison['tiny_minus_original'], indent=2))\n"
                "print('Relative change:', json.dumps({\n"
                "    key: f'{100 * value:+.1f}%'\n"
                "    for key, value in comparison['tiny_relative_change_vs_original'].items()\n"
                "}, indent=2))\n"
                "print('Detection:', json.dumps(test['detection'], indent=2))\n\n"
                "bundle = Path(shutil.make_archive(\n"
                "    '/kaggle/working/deployment_tiny_pipeline_results',\n"
                "    'zip', root_dir=OUTPUT_DIR,\n"
                "))\n"
                "print(f'Download this bundle: {bundle}')\n"
                "print('Inside:', sorted(path.name for path in OUTPUT_DIR.iterdir()))\n"
                "shutil.rmtree(SCRATCH_DIR, ignore_errors=True)\n"
            ),
            markdown(
                "## Outputs\n\n"
                "The ZIP contains the validation-selected deployable checkpoint, compact training "
                "history/report, and the final per-sequence/aggregate 3DPW test metrics. The test "
                "is intentionally run only after checkpoint selection. Do not change training "
                "settings in response to the test result; that would turn the test set into a "
                "validation set.\n"
            ),
        ],
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
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
