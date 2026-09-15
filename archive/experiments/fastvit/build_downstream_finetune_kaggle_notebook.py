#!/usr/bin/env python3
"""Build the self-contained Kaggle notebook for frozen-WHAM fine-tuning."""

from __future__ import annotations

import base64
import copy
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "finetune_fastvit_wham_downstream.py"
EVALUATOR = ROOT / "evaluate_wham_feature_substitution.py"
OUTPUT = ROOT / "finetune_fastvit_wham_downstream_kaggle.ipynb"
CONTINUATION_OUTPUT = ROOT / "continue_fastvit_wham_downstream_kaggle.ipynb"
POLISH_OUTPUT = ROOT / "polish_fastvit_wham_downstream_kaggle.ipynb"
WHAM_COMMIT = "2b54f7797391c94876848b905ed875b154c4a295"


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


def compressed_source(path: Path) -> tuple[str, str]:
    source = path.read_bytes()
    payload = base64.b64encode(gzip.compress(source, compresslevel=9)).decode("ascii")
    return payload, hashlib.sha256(source).hexdigest()


def main() -> None:
    trainer_payload, trainer_digest = compressed_source(TRAINER)
    evaluator_payload, evaluator_digest = compressed_source(EVALUATOR)
    notebook = {
        "cells": [
            markdown(
                "# Downstream-aware FastViT fine-tuning through frozen WHAM\n\n"
                "This notebook starts from the saved phase-two FastViT checkpoint. It trains "
                "on the official **3DPW train** split and selects a checkpoint on **3DPW "
                "validation**. The WHAM recurrent core is frozen; gradients pass through it only "
                "to teach FastViT which feature errors actually alter WHAM pose output.\n\n"
                "The **3DPW test split is not opened here**. After saving this notebook version, "
                "run the separate substitution notebook once on the selected checkpoint. That "
                "untouched test result is the deployment decision.\n\n"
                "Required Kaggle inputs:\n\n"
                "- Saved notebook `distill-fastvit-hmr2-kaggle7f1f81131a`, containing the new "
                "phase-two checkpoint.\n"
                "- Your existing registered private `3dpw-model` dataset containing the nested "
                "`imageFiles/imageFiles/` and `sequenceFiles/sequenceFiles/` trees.\n"
                "- Your existing `3dpw-vit` dataset containing `3dpw_train_vit.pth` and "
                "`3dpw_val_vit.pth`. No Google Drive download is attempted inside Kaggle.\n\n"
                "Only three useful artifacts are retained in `/kaggle/working`: the selected "
                "checkpoint, compact history CSV, and training report JSON.\n"
            ),
            code(
                "# Configuration. Change a mount if Kaggle assigned a different dataset slug.\n"
                "from pathlib import Path\n\n"
                "CONTINUE_FROM_PHASE3 = False\n"
                "POLISH_PHASE3 = False\n"
                "PHASE2_NOTEBOOK_ROOT = Path(\n"
                "    '/kaggle/input/notebooks/nguyntrunglong/'\n"
                "    'distill-fastvit-hmr2-kaggle7f1f81131a'\n"
                ")\n"
                "THREEDPW_ROOT = Path(\n"
                "    '/kaggle/input/datasets/nguyntrunglong/3dpw-model'\n"
                ")\n"
                "PARSED_3DPW_ROOT = Path(\n"
                "    '/kaggle/input/datasets/nguyntrunglong/3dpw-vit'\n"
                ")\n"
                "OUTPUT_DIR = Path('/kaggle/working/fastvit_hmr2_phase3')\n"
                "SCRATCH_DIR = Path('/tmp/fastvit_hmr2_phase3')\n"
                "if CONTINUE_FROM_PHASE3:\n"
                "    phase3_matches = sorted(\n"
                "        Path('/kaggle/input/notebooks/nguyntrunglong').glob(\n"
                "            '*/fastvit_hmr2_phase3/fastvit_hmr2_best.pth'\n"
                "        )\n"
                "    )\n"
                "    if len(phase3_matches) != 1:\n"
                "        raise FileNotFoundError(\n"
                "            'Attach exactly one saved phase-three notebook output; '\n"
                "            f'found {phase3_matches}'\n"
                "        )\n"
                "    SOURCE_CHECKPOINT = phase3_matches[0]\n"
                "else:\n"
                "    SOURCE_CHECKPOINT = (\n"
                "        PHASE2_NOTEBOOK_ROOT / 'fastvit_hmr2_phase2/fastvit_hmr2_best.pth'\n"
                "    )\n\n"
                "CLIP_LENGTH = 32\n"
                "STRIDE = 16\n"
                "BATCH_SIZE = 2\n"
                "WORKERS = 4\n"
                "HEAD_EPOCHS = 0 if CONTINUE_FROM_PHASE3 else 2\n"
                "LAST_STAGE_EPOCHS = 10 if POLISH_PHASE3 else (6 if CONTINUE_FROM_PHASE3 else 4)\n"
                "FINETUNE_HEAD_LR = 5e-6 if POLISH_PHASE3 else (1e-5 if CONTINUE_FROM_PHASE3 else 2e-5)\n"
                "FINETUNE_BACKBONE_LR = 5e-7 if POLISH_PHASE3 else (1e-6 if CONTINUE_FROM_PHASE3 else 2e-6)\n"
                "ROTATION_LOSS = 'geodesic' if CONTINUE_FROM_PHASE3 else 'cosine'\n"
                "WHAM_POSE_WEIGHT = 4.0 if CONTINUE_FROM_PHASE3 else 25.0\n"
                "WHAM_ROOT_WEIGHT = 1.5 if CONTINUE_FROM_PHASE3 else 10.0\n"
                "WHAM_GT_POSE_WEIGHT = 1.0 if CONTINUE_FROM_PHASE3 else 0.0\n"
                "WHAM_GT_ROOT_WEIGHT = 0.0\n"
                "MAX_CLIPS = 0  # 0 means every available training clip.\n\n"
                "if not SOURCE_CHECKPOINT.is_file():\n"
                "    raise FileNotFoundError(\n"
                "        f'The configured source checkpoint is missing: {SOURCE_CHECKPOINT}. '\n"
                "        'Attach the corresponding saved notebook version.'\n"
                "    )\n"
                "if not THREEDPW_ROOT.is_dir():\n"
                "    raise FileNotFoundError(\n"
                "        f'Change THREEDPW_ROOT to your registered private 3DPW input: {THREEDPW_ROOT}'\n"
                "    )\n"
                "if not PARSED_3DPW_ROOT.is_dir():\n"
                "    raise FileNotFoundError(\n"
                "        'Attach your existing 3dpw-vit dataset and set '\n"
                "        f'PARSED_3DPW_ROOT to its mount: {PARSED_3DPW_ROOT}'\n"
                "    )\n"
                "sequence_candidates = (\n"
                "    THREEDPW_ROOT / 'sequenceFiles/train',\n"
                "    THREEDPW_ROOT / 'sequenceFiles/sequenceFiles/train',\n"
                "    THREEDPW_ROOT / '3DPW/sequenceFiles/train',\n"
                ")\n"
                "SEQUENCE_TRAIN_ROOT = next(\n"
                "    (path for path in sequence_candidates if path.is_dir() and next(path.glob('*.pkl'), None)),\n"
                "    None,\n"
                ")\n"
                "if SEQUENCE_TRAIN_ROOT is None:\n"
                "    raise FileNotFoundError(\n"
                "        'The attached 3dpw-model input has no sequenceFiles/.../train/*.pkl. '\n"
                "        'Expand the nested sequenceFiles tree and verify that train exists. '\n"
                "        'It is required to map WHAM numeric training '\n"
                "        'track ids to the correct image sequences.'\n"
                "    )\n"
                "print(f'Phase-two source: {SOURCE_CHECKPOINT}')\n"
                "print(f'3DPW root: {THREEDPW_ROOT}')\n"
                "print(f'Parsed 3DPW tensors: {PARSED_3DPW_ROOT}')\n"
                "print(f'3DPW train annotations: {SEQUENCE_TRAIN_ROOT}')\n"
                "print(f'Final artifacts only: {OUTPUT_DIR}')\n"
            ),
            code(
                "# Kaggle supplies CUDA PyTorch; do not replace it.\n"
                "%pip install -q timm==1.0.22 einops==0.8.1 yacs==0.1.8 "
                "joblib==1.5.2 loguru==0.7.3 smplx==0.1.28 "
                "opencv-python-headless==4.12.0.88 scikit-image==0.25.2 "
                "tqdm==4.67.1\n"
            ),
            code(
                "# Materialize the reviewed scripts in temporary storage and verify their bytes.\n"
                "import base64, gzip, hashlib\n\n"
                "SCRATCH_DIR.mkdir(parents=True, exist_ok=True)\n"
                "embedded = {\n"
                f"    'evaluate_wham_feature_substitution.py': ('{evaluator_digest}', '{evaluator_payload}'),\n"
                f"    'finetune_fastvit_wham_downstream.py': ('{trainer_digest}', '{trainer_payload}'),\n"
                "}\n"
                "for name, (expected_sha256, payload) in embedded.items():\n"
                "    contents = gzip.decompress(base64.b64decode(payload))\n"
                "    actual_sha256 = hashlib.sha256(contents).hexdigest()\n"
                "    if actual_sha256 != expected_sha256:\n"
                "        raise RuntimeError(f'Embedded script checksum mismatch for {name}')\n"
                "    (SCRATCH_DIR / name).write_bytes(contents)\n"
                "print({name: digest for name, (digest, _) in embedded.items()})\n"
            ),
            code(
                "# Verify the manual parsed labels, then acquire the pinned WHAM release assets.\n"
                "import hashlib, subprocess, urllib.request\n\n"
                "def require_parsed_file(name):\n"
                "    matches = sorted(PARSED_3DPW_ROOT.rglob(name))\n"
                "    if len(matches) != 1:\n"
                "        raise FileNotFoundError(\n"
                "            f'Expected exactly one {name} below {PARSED_3DPW_ROOT}; found {matches}'\n"
                "        )\n"
                "    return matches[0]\n\n"
                "TRAIN_PARSED = require_parsed_file('3dpw_train_vit.pth')\n"
                "VAL_PARSED = require_parsed_file('3dpw_val_vit.pth')\n\n"
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
                "    raise RuntimeError(f'Unexpected WHAM commit: {actual_commit}')\n\n"
                "WHAM_CHECKPOINT = SCRATCH_DIR / 'wham_vit_bedlam_w_3dpw.pth.tar'\n"
                "WHAM_CHECKPOINT_URL = (\n"
                "    'https://huggingface.co/camenduru/WHAM/resolve/main/'\n"
                "    'wham_vit_bedlam_w_3dpw.pth.tar?download=true'\n"
                ")\n"
                "WHAM_CHECKPOINT_SHA256 = (\n"
                "    '2ba0cb6a7dd597023a6b2ad6056e7a8b6b33144a35fabea570bfd00842cd4eaf'\n"
                ")\n"
                "def sha256_file(path):\n"
                "    digest = hashlib.sha256()\n"
                "    with path.open('rb') as stream:\n"
                "        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):\n"
                "            digest.update(chunk)\n"
                "    return digest.hexdigest()\n"
                "if not WHAM_CHECKPOINT.is_file():\n"
                "    print('Downloading checksum-pinned WHAM release checkpoint...')\n"
                "    urllib.request.urlretrieve(WHAM_CHECKPOINT_URL, WHAM_CHECKPOINT)\n"
                "checkpoint_sha256 = sha256_file(WHAM_CHECKPOINT)\n"
                "if checkpoint_sha256 != WHAM_CHECKPOINT_SHA256:\n"
                "    raise RuntimeError(f'WHAM checkpoint SHA-256 mismatch: {checkpoint_sha256}')\n"
                "print({\n"
                "    'train_parsed': str(TRAIN_PARSED),\n"
                "    'validation_parsed': str(VAL_PARSED),\n"
                "    'wham_commit': actual_commit,\n"
                "    'wham_checkpoint_sha256': checkpoint_sha256,\n"
                "})\n"
            ),
            code(
                "# Cheap code and data preflight. Training does not start unless both pass.\n"
                "import subprocess, sys\n\n"
                "TRAINER_PATH = SCRATCH_DIR / 'finetune_fastvit_wham_downstream.py'\n"
                "common = [\n"
                "    '--work-dir', str(OUTPUT_DIR),\n"
                "    '--three-dpw-root', str(THREEDPW_ROOT),\n"
                "    '--sequence-root', str(THREEDPW_ROOT),\n"
                "    '--train-parsed', str(TRAIN_PARSED),\n"
                "    '--val-parsed', str(VAL_PARSED),\n"
                "    '--source-checkpoint', str(SOURCE_CHECKPOINT),\n"
                "    '--wham-repo', str(WHAM_REPO),\n"
                "    '--wham-checkpoint', str(WHAM_CHECKPOINT),\n"
                "    '--clip-length', str(CLIP_LENGTH),\n"
                "    '--stride', str(STRIDE),\n"
                "    '--max-clips', str(MAX_CLIPS),\n"
                "]\n"
                "subprocess.run([sys.executable, str(TRAINER_PATH), '--self-test', *common], check=True)\n"
                "subprocess.run([sys.executable, str(TRAINER_PATH), '--inspect-data', *common], check=True)\n"
            ),
            code(
                "# Full downstream-aware run. Expect this cell to take substantial GPU time.\n"
                "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
                "command = [\n"
                "    sys.executable, str(TRAINER_PATH), *common,\n"
                "    '--batch-size', str(BATCH_SIZE),\n"
                "    '--workers', str(WORKERS),\n"
                "    '--head-epochs', str(HEAD_EPOCHS),\n"
                "    '--last-stage-epochs', str(LAST_STAGE_EPOCHS),\n"
                "    '--finetune-head-lr', str(FINETUNE_HEAD_LR),\n"
                "    '--finetune-backbone-lr', str(FINETUNE_BACKBONE_LR),\n"
                "    '--wham-pose-weight', str(WHAM_POSE_WEIGHT),\n"
                "    '--wham-root-weight', str(WHAM_ROOT_WEIGHT),\n"
                "    '--wham-gt-pose-weight', str(WHAM_GT_POSE_WEIGHT),\n"
                "    '--wham-gt-root-weight', str(WHAM_GT_ROOT_WEIGHT),\n"
                "    '--rotation-loss', ROTATION_LOSS,\n"
                "    '--log-every', '50',\n"
                "]\n"
                "if CONTINUE_FROM_PHASE3:\n"
                "    command.append('--stop-when-accepted')\n"
                "print('Starting frozen-WHAM training; 3DPW test is not an input.')\n"
                "training_result = subprocess.run(command, check=False)\n"
                "if training_result.returncode != 0:\n"
                "    raise RuntimeError(f'Downstream training failed with exit code {training_result.returncode}')\n"
            ),
            code(
                "# Compact result and download links; remove every heavy temporary asset.\n"
                "import json, shutil\n"
                "from IPython.display import FileLink, display\n\n"
                "REPORT_PATH = OUTPUT_DIR / 'fastvit_hmr2_training_report.json'\n"
                "CHECKPOINT_PATH = OUTPUT_DIR / 'fastvit_hmr2_best.pth'\n"
                "HISTORY_PATH = OUTPUT_DIR / 'fastvit_hmr2_history.csv'\n"
                "for path in (REPORT_PATH, CHECKPOINT_PATH, HISTORY_PATH):\n"
                "    if not path.is_file():\n"
                "        raise FileNotFoundError(f'Expected output was not produced: {path}')\n"
                "report = json.loads(REPORT_PATH.read_text())\n"
                "best = report['best_validation']\n"
                "compact = {\n"
                "    'accepted_on_validation': report['accepted_on_validation'],\n"
                "    'deployment_accepted': report['deployment_accepted'],\n"
                "    'acceptance_state': report['acceptance_state'],\n"
                "    'tracks': best['tracks'],\n"
                "    'frames': best['frames'],\n"
                "    'gates': best['gates'],\n"
                "    'feature_cosine_mean': best['feature_cosine_mean'],\n"
                "    'teacher_pose_error_deg': best['teacher_pose_error_deg'],\n"
                "    'student_pose_error_deg': best['student_pose_error_deg'],\n"
                "    'pose_degradation_deg': best['pose_degradation_deg'],\n"
                "    'relative_pose_degradation': best['relative_pose_degradation'],\n"
                "    'student_teacher_pose_drift_deg': best['student_teacher_pose_drift_deg'],\n"
                "    'checkpoint_sha256': report['best_checkpoint_sha256'],\n"
                "}\n"
                "print(json.dumps(compact, indent=2))\n"
                "for path in (CHECKPOINT_PATH, HISTORY_PATH, REPORT_PATH):\n"
                "    display(FileLink(str(path)))\n"
                "shutil.rmtree(SCRATCH_DIR, ignore_errors=True)\n"
                "if report['accepted_on_validation']:\n"
                "    print('Save a new version, then run the separate untouched-test notebook once.')\n"
                "else:\n"
                "    print('Validation rejected this checkpoint. Do not run the untouched test; inspect the report and history.')\n"
            ),
            markdown(
                "## What a successful run means\n\n"
                "`accepted_on_validation: true` means the checkpoint passed the predeclared "
                "rotation-space substitution gates on 3DPW validation. It intentionally still "
                "says `deployment_accepted: false` and `awaiting_untouched_3dpw_test`. Save this "
                "notebook version, attach it to the separate test notebook, and run that test "
                "exactly once. Do not tune again from its result.\n"
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
    continuation = copy.deepcopy(notebook)
    continuation["cells"][0] = markdown(
        "# Continue downstream-aware FastViT fine-tuning\n\n"
        "This notebook resumes from the selected checkpoint produced by the first "
        "continuation run. It performs at most six lower-learning-rate last-stage "
        "epochs and stops at the first accepted checkpoint. The revised objective "
        "matches WHAM pose in angle space and adds low-weight ground-truth pose "
        "supervision from the official 3DPW train split. WHAM stays frozen. It still "
        "selects only on validation and never opens the 3DPW test split.\n\n"
        "Before running, save the epoch-24 continuation notebook as a Kaggle version "
        "and attach only that notebook output as the phase-three checkpoint input. "
        "Also attach the same `3dpw-model` and `3dpw-vit` datasets. Only the selected "
        "checkpoint, history, and report remain in `/kaggle/working`.\n"
    )
    for cell in continuation["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "CONTINUE_FROM_PHASE3 = False" in source:
            cell["source"] = source.replace(
                "CONTINUE_FROM_PHASE3 = False", "CONTINUE_FROM_PHASE3 = True"
            ).splitlines(True)
            break
    else:
        raise RuntimeError("Could not configure the continuation notebook")
    CONTINUATION_OUTPUT.write_text(
        json.dumps(continuation, indent=1) + "\n", encoding="utf-8"
    )
    polish = copy.deepcopy(continuation)
    polish["cells"][0] = markdown(
        "# Final low-rate FastViT validation polish\n\n"
        "This notebook resumes from the selected epoch-30 checkpoint. The prior "
        "continuation already passed the relative-degradation and WHAM-drift gates; "
        "only absolute pose degradation remained 0.139 degrees above its threshold. "
        "This run keeps the proven loss unchanged, halves both learning rates, and "
        "allows at most ten last-stage epochs. It stops on the first validation pass.\n\n"
        "WHAM remains frozen. Training uses only 3DPW train, selection uses only "
        "3DPW validation, and the 3DPW test split is never opened. Attach only the "
        "saved epoch-30 notebook output as the phase-three checkpoint input, plus "
        "the same `3dpw-model` and `3dpw-vit` datasets.\n"
    )
    for cell in polish["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "POLISH_PHASE3 = False" in source:
            cell["source"] = source.replace(
                "POLISH_PHASE3 = False", "POLISH_PHASE3 = True"
            ).splitlines(True)
            break
    else:
        raise RuntimeError("Could not configure the polish notebook")
    POLISH_OUTPUT.write_text(json.dumps(polish, indent=1) + "\n", encoding="utf-8")
    print(
        f"Wrote {OUTPUT}, {CONTINUATION_OUTPUT}, and {POLISH_OUTPUT}; "
        f"trainer={trainer_digest}; evaluator={evaluator_digest}"
    )


if __name__ == "__main__":
    main()
