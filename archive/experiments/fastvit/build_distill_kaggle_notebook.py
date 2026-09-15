#!/usr/bin/env python3
"""Regenerate the self-contained Kaggle notebook from the training script."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "distill_fastvit_hmr2.py"
OUTPUT = ROOT / "distill_fastvit_hmr2_kaggle.ipynb"


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    source = SCRIPT.read_bytes()
    compressed = base64.b64encode(gzip.compress(source, compresslevel=9)).decode(
        "ascii"
    )
    digest = hashlib.sha256(source).hexdigest()
    notebook = {
        "cells": [
            markdown(
                "# FastViT → WHAM/HMR2 distillation (corrected)\n\n"
                "This phase-two notebook resumes the spatial FastViT student trained on **person "
                "crops**, applies a stronger cosine/root-pose objective, evaluates on COCO val, "
                "and marks the checkpoint "
                "accepted only when every feature and pose-readout gate passes.\n\n"
                "Before running: enable a Kaggle GPU (T4 or better), enable Internet, attach the "
                "COCO 2017 dataset, and attach the saved output of the first training notebook. "
                "The prior WHAM checkout, HMR2 checkpoint, and complete teacher cache are consumed "
                "directly from that read-only input; phase two writes only compact new artifacts "
                "under `/kaggle/working/fastvit_hmr2_phase2`.\n"
            ),
            code(
                "# Configuration\n"
                "from pathlib import Path\n\n"
                "SAVED_NOTEBOOK_ROOT = Path(\n"
                "    '/kaggle/input/notebooks/nguyntrunglong/'\n"
                "    'distill-fastvit-hmr2-kagglef9b9f724ae'\n"
                ")\n"
                "PHASE2_DIR = Path('/kaggle/working/fastvit_hmr2_phase2')\n"
                "# Direct inner root for awsaf49/coco-2017-dataset: no recursive filesystem scan.\n"
                "COCO_ROOT = Path('/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017')\n"
                "DOWNLOAD_COCO = False\n\n"
                "TRAIN_LIMIT = 48_000\n"
                "VAL_LIMIT = 3_000\n"
                "TEACHER_BATCH_SIZE = 8   # lower to 4 if HMR2 runs out of memory\n"
                "TRAIN_BATCH_SIZE = 48    # lower to 24 if student training runs out of memory\n"
                "WORKERS = 4\n"
                "HEAD_EPOCHS = 0       # ignored when resuming\n"
                "FINETUNE_EPOCHS = 6\n"
                "TOKEN_LOSS_WEIGHT = 1.0\n"
                "COSINE_LOSS_WEIGHT = 1.0      # raised after the validated run plateaued\n"
                "POSE_LOSS_WEIGHT = 0.10\n"
                "ROOT_POSE_LOSS_WEIGHT = 0.25  # root was the dominant pose error\n"
                "FAIL_ON_REJECT = True\n\n"
                "# A saved Kaggle notebook is mounted read-only. Locate its original run\n"
                "# without copying the 2.7 GB teacher or token cache.\n"
                "source_candidates = (\n"
                "    SAVED_NOTEBOOK_ROOT / 'wham_fastvit_distill',\n"
                "    SAVED_NOTEBOOK_ROOT,\n"
                ")\n"
                "SOURCE_RUN_DIR = next(\n"
                "    (path for path in source_candidates\n"
                "     if (path / 'fastvit_hmr2_best.pth').is_file()),\n"
                "    None,\n"
                ")\n"
                "if SOURCE_RUN_DIR is None:\n"
                "    visible = (sorted(path.name for path in SAVED_NOTEBOOK_ROOT.iterdir())\n"
                "               if SAVED_NOTEBOOK_ROOT.is_dir() else [])\n"
                "    raise FileNotFoundError(\n"
                "        f'Could not find fastvit_hmr2_best.pth below {SAVED_NOTEBOOK_ROOT}. '\n"
                "        f'Top-level entries: {visible}'\n"
                "    )\n"
                "RESUME_CHECKPOINT = SOURCE_RUN_DIR / 'fastvit_hmr2_best.pth'\n"
                "SOURCE_HMR2_CHECKPOINT = SOURCE_RUN_DIR / 'checkpoints/hmr2a.ckpt'\n"
                "SOURCE_TEACHER_CACHE = SOURCE_RUN_DIR / 'cache'\n"
                "required_inputs = (\n"
                "    RESUME_CHECKPOINT,\n"
                "    SOURCE_HMR2_CHECKPOINT,\n"
                "    SOURCE_TEACHER_CACHE / 'train_hmr2_tokens.npy',\n"
                "    SOURCE_TEACHER_CACHE / 'train_hmr2_tokens.json',\n"
                "    SOURCE_TEACHER_CACHE / 'val_hmr2_tokens.npy',\n"
                "    SOURCE_TEACHER_CACHE / 'val_hmr2_tokens.json',\n"
                ")\n"
                "missing_inputs = [str(path) for path in required_inputs if not path.exists()]\n"
                "if missing_inputs:\n"
                "    raise FileNotFoundError('Saved run is incomplete:\\n' + '\\n'.join(missing_inputs))\n"
                "print(f'Reading prior run from: {SOURCE_RUN_DIR}')\n"
                "print(f'Writing phase two to: {PHASE2_DIR}')\n"
            ),
            code(
                "# Pinned dependencies; Kaggle already supplies CUDA-enabled PyTorch.\n"
                "%pip install -q timm==1.0.22 einops==0.8.1 yacs==0.1.8 gdown==5.2.0 "
                "pillow==11.3.0 tqdm==4.67.1\n"
            ),
            code(
                f"# Materialize the reviewed training program embedded in this notebook.\n"
                "import base64, gzip, hashlib\n\n"
                f"EXPECTED_SCRIPT_SHA256 = '{digest}'\n"
                f"SCRIPT_GZIP_BASE64 = '{compressed}'\n"
                "script_bytes = gzip.decompress(base64.b64decode(SCRIPT_GZIP_BASE64))\n"
                "assert hashlib.sha256(script_bytes).hexdigest() == EXPECTED_SCRIPT_SHA256\n"
                "SCRIPT_PATH = Path('/kaggle/working/distill_fastvit_hmr2.py')\n"
                "SCRIPT_PATH.write_bytes(script_bytes)\n"
                "print(f'Wrote {SCRIPT_PATH} ({len(script_bytes):,} bytes), SHA-256={EXPECTED_SCRIPT_SHA256}')\n"
            ),
            code(
                "# Cheap preflight: verifies crop padding, spatial output shape, target scaling, and metric math.\n"
                "import subprocess, sys\n"
                "subprocess.run([sys.executable, str(SCRIPT_PATH), '--self-test'], check=True)\n"
            ),
            code(
                "# Dataset preflight: resolves one unambiguous set of official train/val keypoint\n"
                "# JSONs and image directories. This exits before parsing large JSONs or loading HMR2.\n"
                "data_check = [\n"
                "    sys.executable, str(SCRIPT_PATH),\n"
                "    '--coco-root', str(COCO_ROOT),\n"
                "    '--train-limit', '8', '--val-limit', '8',\n"
                "    '--inspect-data',\n"
                "]\n"
                "subprocess.run(data_check, check=True)\n"
            ),
            code(
                "# Phase two: all large immutable inputs stay in the saved notebook mount.\n"
                "command = [\n"
                "    sys.executable, str(SCRIPT_PATH),\n"
                "    '--work-dir', str(PHASE2_DIR),\n"
                "    '--coco-root', str(COCO_ROOT),\n"
                "    '--hmr2-checkpoint', str(SOURCE_HMR2_CHECKPOINT),\n"
                "    '--teacher-cache-dir', str(SOURCE_TEACHER_CACHE),\n"
                "    '--resume', str(RESUME_CHECKPOINT),\n"
                "    '--train-limit', str(TRAIN_LIMIT),\n"
                "    '--val-limit', str(VAL_LIMIT),\n"
                "    '--teacher-batch-size', str(TEACHER_BATCH_SIZE),\n"
                "    '--train-batch-size', str(TRAIN_BATCH_SIZE),\n"
                "    '--workers', str(WORKERS),\n"
                "    '--head-epochs', str(HEAD_EPOCHS),\n"
                "    '--finetune-epochs', str(FINETUNE_EPOCHS),\n"
                "    '--token-loss-weight', str(TOKEN_LOSS_WEIGHT),\n"
                "    '--cosine-loss-weight', str(COSINE_LOSS_WEIGHT),\n"
                "    '--pose-loss-weight', str(POSE_LOSS_WEIGHT),\n"
                "    '--root-pose-loss-weight', str(ROOT_POSE_LOSS_WEIGHT),\n"
                "]\n"
                "if DOWNLOAD_COCO:\n"
                "    command.append('--download-coco')\n"
                "if FAIL_ON_REJECT:\n"
                "    command.append('--fail-on-reject')\n"
                "print(' '.join(command))\n"
                "training_result = subprocess.run(command, check=False)\n"
                "if training_result.returncode:\n"
                "    print('REJECTED OR FAILED: inspect the report below; do not deploy this checkpoint.')\n"
            ),
            code(
                "# Review the decision and expose only the three files we need.\n"
                "import json\n"
                "from IPython.display import FileLink, display\n\n"
                "report_path = PHASE2_DIR / 'fastvit_hmr2_training_report.json'\n"
                "if report_path.exists():\n"
                "    report = json.loads(report_path.read_text())\n"
                "    print(json.dumps(report, indent=2))\n"
                "artifacts = (\n"
                "    PHASE2_DIR / 'fastvit_hmr2_best.pth',\n"
                "    PHASE2_DIR / 'fastvit_hmr2_training_report.json',\n"
                "    PHASE2_DIR / 'fastvit_hmr2_history.csv',\n"
                ")\n"
                "for artifact in artifacts:\n"
                "    if artifact.exists():\n"
                "        display(FileLink(str(artifact)))\n"
                "    else:\n"
                "        print(f'Missing: {artifact}')\n"
            ),
            markdown(
                "## After Kaggle\n\n"
                "Download the checkpoint, report, and history linked above. Use "
                "`fastvit_hmr2_best.pth` only when `accepted` is `true` in the report. On the Mac, "
                "run `utils/export_fastvit_normalized.py`, then rerun the direct HMR2 comparison "
                "and the 3DPW/Core ML diagnostic before enabling the student in the iPhone app.\n"
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
    print(f"Wrote {OUTPUT}; embedded script SHA-256={digest}")


if __name__ == "__main__":
    main()
