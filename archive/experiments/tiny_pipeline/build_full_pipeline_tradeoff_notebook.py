#!/usr/bin/env python3
"""Build the Kaggle notebook for full WHAM-versus-iPhone 3DPW evaluation."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FULL_EVALUATOR = ROOT / "evaluate_full_pipeline_tradeoff.py"
FEATURE_EVALUATOR = ROOT / "evaluate_wham_feature_substitution.py"
MOBILE_EVALUATOR = ROOT / "evaluate_mobile_pipeline_3dpw.py"
EXPORTER = ROOT / "export_fastvit_normalized.py"
OUTPUT = ROOT / "full_pipeline_tradeoff_3dpw_kaggle.ipynb"
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


def embedded(path: Path) -> tuple[str, str]:
    source = path.read_bytes()
    return (
        hashlib.sha256(source).hexdigest(),
        base64.b64encode(gzip.compress(source, compresslevel=9)).decode("ascii"),
    )


def main() -> None:
    payloads = {
        path.name: embedded(path)
        for path in (
            FULL_EVALUATOR,
            FEATURE_EVALUATOR,
            MOBILE_EVALUATOR,
            EXPORTER,
        )
    }
    embedded_lines = "\n".join(
        f"    {name!r}: ({digest!r}, {payload!r}),"
        for name, (digest, payload) in payloads.items()
    )
    notebook = {
        "cells": [
            markdown(
                "# Released WHAM vs the proposed iPhone subset on 3DPW\n\n"
                "This is the final accuracy tradeoff run. It evaluates the official and "
                "controlled rows on every 3DPW test person track, then evaluates the "
                "app-like row against a WHAM reference on every single-person test video:\n\n"
                "1. released WHAM with the official flip average;\n"
                "2. released WHAM without flip (the fair controlled reference);\n"
                "3. phase-three FastViT with official keypoints and initialization; and\n"
                "4. a same-population WHAM reference for the single-person videos; and\n"
                "5. the actual proposed iPhone subset on those videos: YOLOv8n-pose + "
                "phase-three FastViT + neutral/identity initialization.\n\n"
                "The current app selects one highest-confidence person and has no "
                "multi-person identity association. Restricting only the app comparison "
                "to single-person videos prevents one detection from being incorrectly "
                "scored against two different people.\n\n"
                "The notebook uses licensed SMPL files only to decode predictions for the "
                "paper's PA-MPJPE, MPJPE, PVE, and acceleration metrics. SMPL is **not** "
                "silently added to the phone workload. WHAM's official 3DPW evaluator sets "
                "camera angular velocity to zero, so this is camera-coordinate body "
                "reconstruction, not a world-grounded trajectory test.\n\n"
                "Required Kaggle inputs:\n\n"
                "- your existing private `3dpw-model` dataset with `imageFiles/`, "
                "`sequenceFiles/`, and `3dpw_test_vit.pth`;\n"
                "- the saved phase-three notebook output containing the exact epoch-36 "
                "`fastvit_hmr2_best.pth`; and\n"
                "- one private SMPL-assets dataset containing the three licensed SMPL model "
                "files. The next cell searches all Kaggle inputs, so the dataset slug does "
                "not matter. WHAM's public `J_regressor_h36m.npy` is fetched separately with "
                "a pinned checksum.\n"
            ),
            code(
                "# Configuration. Usually no edit is needed.\n"
                "from pathlib import Path\n"
                "import hashlib\n\n"
                "KAGGLE_INPUT = Path('/kaggle/input')\n"
                "THREEDPW_ROOT = Path('/kaggle/input/datasets/nguyntrunglong/3dpw-model')\n"
                "SAVED_NOTEBOOKS_ROOT = Path('/kaggle/input/notebooks/nguyntrunglong')\n"
                "SCRATCH_DIR = Path('/tmp/wham_full_pipeline_tradeoff')\n"
                "OUTPUT_DIR = Path('/kaggle/working/wham_full_pipeline_tradeoff')\n"
                "SEQUENCES = 0  # 0 = all matching test tracks.\n"
                "FRAMES_PER_SEQUENCE = 0  # 0 = every frame.\n"
                "POSE_BATCH_SIZE = 32\n"
                "STUDENT_BATCH_SIZE = 64\n"
                "SMPL_BATCH_SIZE = 256  # lower to 128 if the GPU runs out of memory.\n\n"
                "EXPECTED_STUDENT_SHA256 = (\n"
                "    'f15875f3fed12538312f59956b6c93e9cca2ab41a9e8d87edf593dd85f311ab1'\n"
                ")\n\n"
                "def sha256_file(path):\n"
                "    digest = hashlib.sha256()\n"
                "    with path.open('rb') as stream:\n"
                "        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):\n"
                "            digest.update(chunk)\n"
                "    return digest.hexdigest()\n\n"
                "checkpoint_candidates = sorted(\n"
                "    SAVED_NOTEBOOKS_ROOT.rglob('fastvit_hmr2_best.pth')\n"
                ")\n"
                "phase3_matches = [\n"
                "    path for path in checkpoint_candidates\n"
                "    if sha256_file(path) == EXPECTED_STUDENT_SHA256\n"
                "]\n"
                "if len(phase3_matches) != 1:\n"
                "    raise FileNotFoundError(\n"
                "        'Attach the saved epoch-36 phase-three notebook output. '\n"
                "        f'Expected SHA-256 {EXPECTED_STUDENT_SHA256}; found '\n"
                "        f'{[(str(path), sha256_file(path)) for path in checkpoint_candidates]}.'\n"
                "    )\n"
                "STUDENT_CHECKPOINT = phase3_matches[0]\n"
                "if not THREEDPW_ROOT.is_dir():\n"
                "    raise FileNotFoundError(\n"
                "        f'Change THREEDPW_ROOT to the existing private 3DPW mount: {THREEDPW_ROOT}'\n"
                "    )\n"
                "parsed_candidates = list(THREEDPW_ROOT.rglob('3dpw_test_vit.pth'))\n"
                "if not parsed_candidates:\n"
                "    parsed_candidates = list(KAGGLE_INPUT.rglob('3dpw_test_vit.pth'))\n"
                "if len(parsed_candidates) != 1:\n"
                "    raise FileNotFoundError(\n"
                "        f'Expected exactly one 3dpw_test_vit.pth, found {parsed_candidates}'\n"
                "    )\n"
                "PARSED_3DPW = parsed_candidates[0]\n"
                "print(f'Phase-three checkpoint: {STUDENT_CHECKPOINT}')\n"
                "print(f'3DPW root: {THREEDPW_ROOT}')\n"
                "print(f'Parsed labels: {PARSED_3DPW}')\n"
                "print(f'Useful outputs only: {OUTPUT_DIR}')\n"
            ),
            code(
                "# Runtime dependencies. Resolver warnings about unrelated preinstalled Kaggle packages are harmless.\n"
                "%pip install -q timm==1.0.22 ultralytics==8.4.146 coremltools==9.0 "
                "einops==0.8.1 yacs==0.1.8 joblib==1.5.2 loguru==0.7.3 "
                "smplx==0.1.28 chumpy==0.70 opencv-python-headless==4.10.0.84 "
                "scikit-image==0.25.2 tqdm==4.67.1\n"
            ),
            code(
                "# Locate and normalize the three manually licensed SMPL model files.\n"
                "import shutil, urllib.request\n\n"
                "SCRATCH_DIR.mkdir(parents=True, exist_ok=True)\n"
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
                "}\n\n"
                "def find_asset(names):\n"
                "    matches = []\n"
                "    for name in names:\n"
                "        matches.extend(KAGGLE_INPUT.rglob(name))\n"
                "    matches = sorted(set(matches))\n"
                "    if not matches:\n"
                "        raise FileNotFoundError(\n"
                "            f'Missing licensed asset {names}. Upload it in a private Kaggle dataset.'\n"
                "        )\n"
                "    if len(matches) > 1:\n"
                "        print(f'Multiple matches for {names}; using {matches[0]}')\n"
                "    return matches[0]\n\n"
                "for output_name, aliases in model_aliases.items():\n"
                "    source = find_asset(aliases)\n"
                "    shutil.copy2(source, SMPL_MODEL_DIR / output_name)\n"
                "    print(f'{output_name}: {source}')\n"
                "H36M_REGRESSOR = SCRATCH_DIR / 'J_regressor_h36m.npy'\n"
                "H36M_REGRESSOR_URL = (\n"
                "    'https://huggingface.co/camenduru/WHAM/resolve/main/'\n"
                "    'J_regressor_h36m.npy?download=true'\n"
                ")\n"
                "H36M_REGRESSOR_SHA256 = (\n"
                "    'c655cd7013d7829eb9acbebf0e43f952a3fa0305a53c35880e39192bfb6444a0'\n"
                ")\n"
                "if not H36M_REGRESSOR.is_file():\n"
                "    print('Downloading checksum-pinned public H36M joint regressor...')\n"
                "    urllib.request.urlretrieve(H36M_REGRESSOR_URL, H36M_REGRESSOR)\n"
                "actual_regressor_sha256 = sha256_file(H36M_REGRESSOR)\n"
                "if actual_regressor_sha256 != H36M_REGRESSOR_SHA256:\n"
                "    raise RuntimeError(\n"
                "        f'J_regressor_h36m.npy SHA-256 mismatch: {actual_regressor_sha256}'\n"
                "    )\n"
                "print(f'J_regressor_h36m.npy: {actual_regressor_sha256}')\n"
            ),
            code(
                "# Materialize the reviewed scripts embedded in this notebook.\n"
                "import base64, gzip, hashlib\n\n"
                "embedded = {\n"
                f"{embedded_lines}\n"
                "}\n"
                "for name, (expected_sha256, payload) in embedded.items():\n"
                "    contents = gzip.decompress(base64.b64decode(payload))\n"
                "    actual_sha256 = hashlib.sha256(contents).hexdigest()\n"
                "    if actual_sha256 != expected_sha256:\n"
                "        raise RuntimeError(f'Embedded script checksum mismatch for {name}')\n"
                "    (SCRATCH_DIR / name).write_bytes(contents)\n"
                "FULL_EVALUATOR = SCRATCH_DIR / 'evaluate_full_pipeline_tradeoff.py'\n"
                "EXPORTER = SCRATCH_DIR / 'export_fastvit_normalized.py'\n"
                "print({name: digest for name, (digest, _) in embedded.items()})\n"
            ),
            code(
                "# Fetch only the pinned WHAM source and weights needed by this evaluation.\n"
                "import subprocess, urllib.request\n\n"
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
                "    'yolov8n-pose.pt': (\n"
                "        'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-pose.pt',\n"
                "        'c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212',\n"
                "    ),\n"
                "}\n"
                "downloaded = {}\n"
                "for name, (url, expected_sha256) in downloads.items():\n"
                "    path = SCRATCH_DIR / name\n"
                "    if not path.is_file():\n"
                "        print(f'Downloading {name}...', flush=True)\n"
                "        urllib.request.urlretrieve(url, path)\n"
                "    actual_sha256 = sha256_file(path)\n"
                "    if actual_sha256 != expected_sha256:\n"
                "        raise RuntimeError(f'{name} SHA-256 mismatch: {actual_sha256}')\n"
                "    downloaded[name] = path\n"
                "WHAM_CHECKPOINT = downloaded['wham_vit_bedlam_w_3dpw.pth.tar']\n"
                "YOLOV8_WEIGHTS = downloaded['yolov8n-pose.pt']\n"
                "print(f'WHAM source: {actual_commit}')\n"
                "print({name: sha256_file(path) for name, path in downloaded.items()})\n"
            ),
            code(
                "# Full 3DPW comparison. Based on the earlier two-detector run, budget about 45-70 minutes.\n"
                "import os, sys\n\n"
                "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
                "REPORT = OUTPUT_DIR / 'full_pipeline_tradeoff_3dpw.json'\n"
                "PER_SEQUENCE = OUTPUT_DIR / 'full_pipeline_tradeoff_3dpw.csv'\n"
                "command = [\n"
                "    sys.executable, '-u', str(FULL_EVALUATOR),\n"
                "    '--parsed-3dpw', str(PARSED_3DPW),\n"
                "    '--three-dpw-root', str(THREEDPW_ROOT),\n"
                "    '--student-checkpoint', str(STUDENT_CHECKPOINT),\n"
                "    '--wham-repo', str(WHAM_REPO),\n"
                "    '--wham-checkpoint', str(WHAM_CHECKPOINT),\n"
                "    '--yolov8-weights', str(YOLOV8_WEIGHTS),\n"
                "    '--smpl-model-directory', str(SMPL_MODEL_DIR),\n"
                "    '--h36m-joint-regressor', str(H36M_REGRESSOR),\n"
                "    '--sequences', str(SEQUENCES),\n"
                "    '--frames', str(FRAMES_PER_SEQUENCE),\n"
                "    '--pose-batch-size', str(POSE_BATCH_SIZE),\n"
                "    '--student-batch-size', str(STUDENT_BATCH_SIZE),\n"
                "    '--smpl-batch-size', str(SMPL_BATCH_SIZE),\n"
                "    '--output', str(REPORT),\n"
                "    '--per-sequence-output', str(PER_SEQUENCE),\n"
                "]\n"
                "environment = os.environ.copy()\n"
                "environment['PYTHONPATH'] = str(SCRATCH_DIR) + os.pathsep + environment.get('PYTHONPATH', '')\n"
                "print(' '.join(command), flush=True)\n"
                "subprocess.run(command, check=True, env=environment)\n"
            ),
            code(
                "# Compact result table. These are the values needed for the report.\n"
                "import json\n"
                "from IPython.display import display\n"
                "import pandas as pd\n\n"
                "report = json.loads(REPORT.read_text())\n"
                "rows = []\n"
                "for name, variant in report['variants'].items():\n"
                "    metrics = variant['metrics']\n"
                "    rows.append({\n"
                "        'variant': name,\n"
                "        'recurrent frames': metrics['mpjpe_mm']['samples'],\n"
                "        'PA-MPJPE (mm)': metrics['pa_mpjpe_mm']['mean'],\n"
                "        'MPJPE (mm)': metrics['mpjpe_mm']['mean'],\n"
                "        'PVE (mm)': metrics['pve_mm']['mean'],\n"
                "        'Accel (official code)': metrics['accel_official_30fps']['mean'],\n"
                "    })\n"
                "mobile_reference = report['iphone_reference_paper_wham_single_person_subset']\n"
                "reference_metrics = mobile_reference['metrics']\n"
                "rows.append({\n"
                "    'variant': 'paper_wham_bedlam_flip_single_person_reference',\n"
                "    'recurrent frames': reference_metrics['mpjpe_mm']['samples'],\n"
                "    'PA-MPJPE (mm)': reference_metrics['pa_mpjpe_mm']['mean'],\n"
                "    'MPJPE (mm)': reference_metrics['mpjpe_mm']['mean'],\n"
                "    'PVE (mm)': reference_metrics['pve_mm']['mean'],\n"
                "    'Accel (official code)': reference_metrics['accel_official_30fps']['mean'],\n"
                "})\n"
                "display(pd.DataFrame(rows).round(3))\n"
                "print('iPhone minus paper WHAM on the same single-person subset:', json.dumps(\n"
                "    report['iphone_minus_paper_wham_same_single_person_subset'], indent=2\n"
                "))\n"
                "print('Baseline reproduction check:', json.dumps(\n"
                "    report['baseline_reproduction'], indent=2\n"
                "))\n"
                "print('Controlled FastViT drift:', json.dumps(\n"
                "    report['controlled_fastvit']['student_teacher_pose_drift']['all_joints_deg'],\n"
                "    indent=2,\n"
                "))\n"
                "print('Mobile detection:', json.dumps(report['iphone_detection'], indent=2))\n"
            ),
            code(
                "# Export the exact phase-three FastViT as a clearly labelled diagnostic package.\n"
                "# It is for the one-off phone latency run, not an acceptance claim.\n"
                "COREML_PACKAGE = SCRATCH_DIR / 'FastViTNormalized.mlpackage'\n"
                "export_command = [\n"
                "    sys.executable, '-u', str(EXPORTER),\n"
                "    '--weights', str(STUDENT_CHECKPOINT),\n"
                "    '--output', str(COREML_PACKAGE),\n"
                "    '--accept-product-validation',\n"
                "    '--full-evaluation-report', str(REPORT),\n"
                "]\n"
                "subprocess.run(export_command, check=True, env=environment)\n"
                "coreml_zip_base = OUTPUT_DIR / 'FastViTNormalized_phase3_diagnostic'\n"
                "coreml_zip = Path(shutil.make_archive(\n"
                "    str(coreml_zip_base), 'zip',\n"
                "    root_dir=COREML_PACKAGE.parent, base_dir=COREML_PACKAGE.name,\n"
                "))\n"
                "manifest = {\n"
                "    'student_checkpoint_sha256': EXPECTED_STUDENT_SHA256,\n"
                "    'coreml_zip': coreml_zip.name,\n"
                "    'deployment_accepted': False,\n"
                "    'purpose': 'single physical-iPhone latency diagnostic',\n"
                "    'full_evaluation_report': REPORT.name,\n"
                "}\n"
                "MANIFEST = OUTPUT_DIR / 'artifact_manifest.json'\n"
                "MANIFEST.write_text(json.dumps(manifest, indent=2) + '\\n')\n"
                "print(json.dumps(manifest, indent=2))\n"
            ),
            code(
                "# One small download bundle; raw logs and temporary repositories stay out of it.\n"
                "bundle_base = Path('/kaggle/working/wham_full_pipeline_tradeoff_results')\n"
                "bundle = Path(shutil.make_archive(\n"
                "    str(bundle_base), 'zip', root_dir=OUTPUT_DIR\n"
                "))\n"
                "print(f'Download: {bundle}')\n"
                "print('Inside:', sorted(path.name for path in OUTPUT_DIR.iterdir()))\n"
            ),
        ],
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
