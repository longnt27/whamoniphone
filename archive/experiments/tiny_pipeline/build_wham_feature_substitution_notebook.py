#!/usr/bin/env python3
"""Build a self-contained Kaggle notebook for the 3DPW WHAM feature A/B test."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "evaluate_wham_feature_substitution.py"
MOBILE_EVALUATOR = ROOT / "evaluate_mobile_pipeline_3dpw.py"
EXPORTER = ROOT / "export_fastvit_normalized.py"
OUTPUT = ROOT / "wham_feature_substitution_3dpw_kaggle.ipynb"
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


def main() -> None:
    source = SCRIPT.read_bytes()
    compressed = base64.b64encode(gzip.compress(source, compresslevel=9)).decode(
        "ascii"
    )
    digest = hashlib.sha256(source).hexdigest()
    mobile_source = MOBILE_EVALUATOR.read_bytes()
    mobile_compressed = base64.b64encode(
        gzip.compress(mobile_source, compresslevel=9)
    ).decode("ascii")
    mobile_digest = hashlib.sha256(mobile_source).hexdigest()
    exporter_source = EXPORTER.read_bytes()
    exporter_compressed = base64.b64encode(
        gzip.compress(exporter_source, compresslevel=9)
    ).decode("ascii")
    exporter_digest = hashlib.sha256(exporter_source).hexdigest()
    notebook = {
        "cells": [
            markdown(
                "# Full FastViT feature substitution test inside WHAM on 3DPW\n\n"
                "This is the downstream A/B test. For each identical 3DPW frame, it runs the "
                "official stored HMR2 feature and the distilled FastViT feature through the same "
                "frozen WHAM recurrent core. It compares both outputs with 3DPW ground-truth SMPL "
                "rotations and measures student-versus-teacher drift.\n\n"
                "Required Kaggle inputs:\n\n"
                "1. The saved phase-three polishing notebook version containing the exact "
                "epoch-36 `fastvit_hmr2_best.pth` selected on validation.\n"
                "2. Your **registered, private 3DPW dataset** containing `imageFiles/`, plus the "
                "official public `3dpw_test_vit.pth` from WHAM's dataset folder. Do not use an "
                "unlicensed raw-data mirror. Set `THREEDPW_ROOT` below to its exact mount.\n\n"
                "The WHAM checkpoint is fetched into temporary storage and verified by SHA-256. "
                "If the locked product gates pass, the notebook also exports and zips the "
                "phase-three Core ML package. Only those useful artifacts remain in "
                "`/kaggle/working`. This rotation diagnostic does not claim the paper's "
                "PA-MPJPE/MPJPE/PVE numbers because licensed SMPL model files are not used.\n"
            ),
            code(
                "# Configuration — change THREEDPW_ROOT only if Kaggle changed its mount.\n"
                "from pathlib import Path\n"
                "import hashlib\n\n"
                "THREEDPW_ROOT = Path(\n"
                "    '/kaggle/input/datasets/nguyntrunglong/3dpw-model'\n"
                ")\n"
                "SAVED_NOTEBOOKS_ROOT = Path('/kaggle/input/notebooks/nguyntrunglong')\n"
                "SCRATCH_DIR = Path('/tmp/wham_feature_substitution')\n"
                "OUTPUT_DIR = Path('/kaggle/working/wham_feature_substitution')\n"
                "SEQUENCES = 0  # 0 means every matching test track.\n"
                "FRAMES_PER_SEQUENCE = 0  # 0 means every available frame.\n"
                "STUDENT_BATCH_SIZE = 64\n\n"
                "EXPECTED_STUDENT_SHA256 = (\n"
                "    'f15875f3fed12538312f59956b6c93e9cca2ab41a9e8d87edf593dd85f311ab1'\n"
                ")\n"
                "def checkpoint_sha256(path):\n"
                "    digest = hashlib.sha256()\n"
                "    with path.open('rb') as stream:\n"
                "        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):\n"
                "            digest.update(chunk)\n"
                "    return digest.hexdigest()\n"
                "checkpoint_candidates = sorted(\n"
                "    SAVED_NOTEBOOKS_ROOT.rglob('fastvit_hmr2_best.pth')\n"
                ")\n"
                "phase3_matches = [\n"
                "    path for path in checkpoint_candidates\n"
                "    if checkpoint_sha256(path) == EXPECTED_STUDENT_SHA256\n"
                "]\n"
                "if len(phase3_matches) != 1:\n"
                "    raise FileNotFoundError(\n"
                "        'Attach the saved epoch-36 phase-three notebook output. '\n"
                "        f'Expected SHA-256 {EXPECTED_STUDENT_SHA256}; found candidates '\n"
                "        f'{[(str(path), checkpoint_sha256(path)) for path in checkpoint_candidates]}.'\n"
                "    )\n"
                "STUDENT_CHECKPOINT = phase3_matches[0]\n"
                "if not THREEDPW_ROOT.is_dir():\n"
                "    raise FileNotFoundError(\n"
                "        f'Change THREEDPW_ROOT to your registered 3DPW mount: {THREEDPW_ROOT}'\n"
                "    )\n"
                "parsed_candidates = (\n"
                "    THREEDPW_ROOT / '3dpw_test_vit.pth',\n"
                "    THREEDPW_ROOT / 'parsed_data/3dpw_test_vit.pth',\n"
                "    THREEDPW_ROOT.parent / '3dpw_test_vit.pth',\n"
                ")\n"
                "PARSED_3DPW = next((path for path in parsed_candidates if path.is_file()), None)\n"
                "if PARSED_3DPW is None:\n"
                "    raise FileNotFoundError(\n"
                "        'Add WHAM 3dpw_test_vit.pth to the private 3DPW Kaggle input, either '\n"
                "        'beside the 3DPW directory or inside it.'\n"
                "    )\n"
                "print(f'Student checkpoint: {STUDENT_CHECKPOINT}')\n"
                "print(f'3DPW input: {THREEDPW_ROOT}')\n"
                "print(f'WHAM parsed 3DPW: {PARSED_3DPW}')\n"
                "print(f'Results only: {OUTPUT_DIR}')\n"
            ),
            code(
                "# Kaggle supplies CUDA-enabled PyTorch. These do not replace torch.\n"
                "%pip install -q timm==1.0.22 ultralytics==8.4.146 coremltools==9.0 'numpy<=2.3.5' einops==0.8.1 yacs==0.1.8 "
                "joblib==1.5.2 loguru==0.7.3 smplx==0.1.28 "
                "opencv-python-headless==4.12.0.88 scikit-image==0.25.2 tqdm==4.67.1\n"
            ),
            code(
                "# Materialize the reviewed evaluator and exporter embedded in this notebook.\n"
                "import base64, gzip, hashlib\n\n"
                "SCRATCH_DIR.mkdir(parents=True, exist_ok=True)\n"
                "embedded = {\n"
                f"    'evaluate_wham_feature_substitution.py': ('{digest}', '{compressed}'),\n"
                f"    'evaluate_mobile_pipeline_3dpw.py': ('{mobile_digest}', '{mobile_compressed}'),\n"
                f"    'export_fastvit_normalized.py': ('{exporter_digest}', '{exporter_compressed}'),\n"
                "}\n"
                "for name, (expected_sha256, payload) in embedded.items():\n"
                "    contents = gzip.decompress(base64.b64decode(payload))\n"
                "    actual_sha256 = hashlib.sha256(contents).hexdigest()\n"
                "    if actual_sha256 != expected_sha256:\n"
                "        raise RuntimeError(f'Embedded script checksum mismatch for {name}')\n"
                "    (SCRATCH_DIR / name).write_bytes(contents)\n"
                "SCRIPT_PATH = SCRATCH_DIR / 'evaluate_wham_feature_substitution.py'\n"
                "MOBILE_EVALUATOR_PATH = SCRATCH_DIR / 'evaluate_mobile_pipeline_3dpw.py'\n"
                "EXPORTER_PATH = SCRATCH_DIR / 'export_fastvit_normalized.py'\n"
                "print({name: digest for name, (digest, _) in embedded.items()})\n"
            ),
            code(
                "# Fetch the release WHAM checkpoint into ephemeral storage.\n"
                "import hashlib, subprocess, urllib.request\n\n"
                "SCRATCH_DIR.mkdir(parents=True, exist_ok=True)\n"
                "WHAM_REPO = SCRATCH_DIR / 'WHAM'\n"
                "if not (WHAM_REPO / 'lib/models/wham.py').is_file():\n"
                "    subprocess.run([\n"
                "        'git', 'clone', '--filter=blob:none', '--no-checkout',\n"
                "        'https://github.com/yohanshin/WHAM.git', str(WHAM_REPO),\n"
                "    ], check=True)\n"
                f"    subprocess.run(['git', 'checkout', '--detach', '{WHAM_COMMIT}'], "
                "cwd=WHAM_REPO, check=True)\n"
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
                "    print('Downloading the checksum-pinned WHAM release checkpoint mirror...')\n"
                "    urllib.request.urlretrieve(WHAM_CHECKPOINT_URL, WHAM_CHECKPOINT)\n"
                "checkpoint_sha256 = sha256_file(WHAM_CHECKPOINT)\n"
                "if checkpoint_sha256 != WHAM_CHECKPOINT_SHA256:\n"
                "    raise RuntimeError(\n"
                "        f'WHAM checkpoint SHA-256 mismatch: {checkpoint_sha256}. '\n"
                "        'Delete the temporary file and retry.'\n"
                "    )\n"
                "YOLO_WEIGHTS = {\n"
                "    'yolov8n-pose.pt': (\n"
                "        'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-pose.pt',\n"
                "        'c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212',\n"
                "    ),\n"
                "    'yolo26n-pose.pt': (\n"
                "        'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt',\n"
                "        'eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9',\n"
                "    ),\n"
                "}\n"
                "for name, (url, expected_sha256) in YOLO_WEIGHTS.items():\n"
                "    path = SCRATCH_DIR / name\n"
                "    if not path.is_file():\n"
                "        urllib.request.urlretrieve(url, path)\n"
                "    actual_sha256 = sha256_file(path)\n"
                "    if actual_sha256 != expected_sha256:\n"
                "        raise RuntimeError(f'{name} SHA-256 mismatch: {actual_sha256}')\n"
                "YOLOV8_WEIGHTS = SCRATCH_DIR / 'yolov8n-pose.pt'\n"
                "YOLO26_WEIGHTS = SCRATCH_DIR / 'yolo26n-pose.pt'\n"
                "assert PARSED_3DPW.stat().st_size > 300_000_000\n"
                "assert WHAM_CHECKPOINT.stat().st_size == 190_975_610\n"
                "print('Parsed 3DPW labels and verified WHAM checkpoint are ready')\n"
            ),
            code(
                "# Run the controlled downstream substitution test.\n"
                "import subprocess, sys\n\n"
                "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
                "REPORT_PATH = OUTPUT_DIR / 'wham_feature_substitution_3dpw.json'\n"
                "CSV_PATH = OUTPUT_DIR / 'wham_feature_substitution_3dpw.csv'\n"
                "REPORT_PATH.unlink(missing_ok=True)\n"
                "CSV_PATH.unlink(missing_ok=True)\n"
                "command = [\n"
                "    sys.executable, str(SCRIPT_PATH),\n"
                "    '--parsed-3dpw', str(PARSED_3DPW),\n"
                "    '--three-dpw-root', str(THREEDPW_ROOT),\n"
                "    '--student-checkpoint', str(STUDENT_CHECKPOINT),\n"
                "    '--wham-repo', str(WHAM_REPO),\n"
                "    '--wham-checkpoint', str(WHAM_CHECKPOINT),\n"
                "    '--sequences', str(SEQUENCES),\n"
                "    '--frames', str(FRAMES_PER_SEQUENCE),\n"
                "    '--student-batch-size', str(STUDENT_BATCH_SIZE),\n"
                "    '--max-pose-degradation-deg', '1.15',\n"
                "    '--max-relative-pose-degradation', '0.10',\n"
                "    '--max-teacher-drift-deg', '5.0',\n"
                "    '--acceptance-policy-label',\n"
                "    'post_hoc_product_policy_locked_before_untouched_test',\n"
                "    '--output', str(REPORT_PATH),\n"
                "    '--per-sequence-output', str(CSV_PATH),\n"
                "]\n"
                "print(' '.join(command))\n"
                "evaluation_result = subprocess.run(command, check=False)\n"
                "print(f'Evaluator exit code: {evaluation_result.returncode}')\n"
            ),
            code(
                "# Full app-like 3DPW A/B: YOLO8/26 -> crop -> FastViT -> neutral WHAM_I -> WHAM.\n"
                "MOBILE_REPORT_PATH = OUTPUT_DIR / 'mobile_pipeline_3dpw.json'\n"
                "MOBILE_CSV_PATH = OUTPUT_DIR / 'mobile_pipeline_3dpw.csv'\n"
                "MOBILE_REPORT_PATH.unlink(missing_ok=True)\n"
                "MOBILE_CSV_PATH.unlink(missing_ok=True)\n"
                "mobile_command = [\n"
                "    sys.executable, str(MOBILE_EVALUATOR_PATH),\n"
                "    '--parsed-3dpw', str(PARSED_3DPW),\n"
                "    '--three-dpw-root', str(THREEDPW_ROOT),\n"
                "    '--student-checkpoint', str(STUDENT_CHECKPOINT),\n"
                "    '--wham-repo', str(WHAM_REPO),\n"
                "    '--wham-checkpoint', str(WHAM_CHECKPOINT),\n"
                "    '--yolov8-weights', str(YOLOV8_WEIGHTS),\n"
                "    '--yolo26-weights', str(YOLO26_WEIGHTS),\n"
                "    '--sequences', str(SEQUENCES),\n"
                "    '--frames', str(FRAMES_PER_SEQUENCE),\n"
                "    '--pose-batch-size', '32',\n"
                "    '--student-batch-size', str(STUDENT_BATCH_SIZE),\n"
                "    '--output', str(MOBILE_REPORT_PATH),\n"
                "    '--per-sequence-output', str(MOBILE_CSV_PATH),\n"
                "]\n"
                "print(' '.join(mobile_command))\n"
                "mobile_result = subprocess.run(mobile_command, check=False)\n"
                "print(f'Mobile-pipeline evaluator exit code: {mobile_result.returncode}')\n"
            ),
            code(
                "# Compact result, conditional Core ML export, and download links.\n"
                "import json, shutil, subprocess, sys\n"
                "from IPython.display import FileLink, display\n\n"
                "if REPORT_PATH.is_file():\n"
                "    report = json.loads(REPORT_PATH.read_text())\n"
                "    compact = {\n"
                "        'accepted_for_device_diagnostic': report['accepted_for_device_diagnostic'],\n"
                "        'gates': report['gates'],\n"
                "        'thresholds': report['thresholds'],\n"
                "        'scope': report['scope'],\n"
                "        'feature_cosine': report['feature']['cosine'],\n"
                "        'teacher_pose_error_deg': report['teacher_wham_vs_ground_truth']['all_joints_deg'],\n"
                "        'student_pose_error_deg': report['student_wham_vs_ground_truth']['all_joints_deg'],\n"
                "        'student_minus_teacher': report['student_minus_teacher'],\n"
                "    }\n"
                "    print(json.dumps(compact, indent=2))\n"
                "    display(FileLink(str(REPORT_PATH)))\n"
                "    display(FileLink(str(CSV_PATH)))\n"
                "    if MOBILE_REPORT_PATH.is_file():\n"
                "        mobile_report = json.loads(MOBILE_REPORT_PATH.read_text())\n"
                "        mobile_compact = {\n"
                "            'scope': mobile_report['scope'],\n"
                "            'yolov8n_pose': mobile_report['variants']['yolov8n_pose'],\n"
                "            'yolo26n_pose': mobile_report['variants']['yolo26n_pose'],\n"
                "            'yolo26_minus_yolov8_pose_error_deg': (\n"
                "                mobile_report['yolo26_minus_yolov8_pose_error_deg']\n"
                "            ),\n"
                "        }\n"
                "        print(json.dumps({'mobile_pipeline': mobile_compact}, indent=2))\n"
                "        display(FileLink(str(MOBILE_REPORT_PATH)))\n"
                "        display(FileLink(str(MOBILE_CSV_PATH)))\n"
                "    else:\n"
                "        print('No mobile-pipeline report was produced; inspect the previous cell.')\n"
                "    if report['accepted_for_device_diagnostic']:\n"
                "        COREML_PATH = OUTPUT_DIR / 'FastViTNormalized.mlpackage'\n"
                "        subprocess.run([\n"
                "            sys.executable, str(EXPORTER_PATH),\n"
                "            '--weights', str(STUDENT_CHECKPOINT),\n"
                "            '--accept-product-validation',\n"
                "            '--output', str(COREML_PATH),\n"
                "        ], check=True)\n"
                "        archive_base = OUTPUT_DIR / 'FastViTNormalized_phase3_mlpackage'\n"
                "        archive_path = Path(str(archive_base) + '.zip')\n"
                "        archive_path.unlink(missing_ok=True)\n"
                "        shutil.make_archive(\n"
                "            str(archive_base), 'zip',\n"
                "            root_dir=OUTPUT_DIR, base_dir=COREML_PATH.name,\n"
                "        )\n"
                "        print({\n"
                "            'coreml_archive': str(archive_path),\n"
                "            'archive_bytes': archive_path.stat().st_size,\n"
                "            'runtime_parity': 'must be rerun on macOS before iPhone deployment',\n"
                "        })\n"
                "        display(FileLink(str(archive_path)))\n"
                "    else:\n"
                "        print('Product test rejected the checkpoint; Core ML export skipped.')\n"
                "    shutil.rmtree(SCRATCH_DIR, ignore_errors=True)\n"
                "else:\n"
                "    print('No report was produced. Keep the scratch directory and inspect the prior cell.')\n"
            ),
            markdown(
                "## Decision rule\n\n"
                "This run uses every available 3DPW test track and frame. The product policy was "
                "adopted after validation but is locked before this untouched test: no more than "
                "1.15° absolute pose-error degradation, no more than 10% relative degradation, "
                "and no more than 5° mean pose drift from the official HMR2-feature WHAM output. "
                "Passing does not establish the paper's full "
                "world-grounded accuracy; it only shows that replacing HMR2 features with FastViT "
                "does not materially damage this held-out recurrent pose diagnostic.\n"
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
