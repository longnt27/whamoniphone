#!/usr/bin/env python3
"""Build the one-pass Kaggle accuracy notebook for the complete tiny pipeline."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TINY_EVALUATOR = ROOT / "evaluate_tiny_pipeline_3dpw.py"
FULL_HELPERS = ROOT / "evaluate_full_pipeline_tradeoff.py"
FEATURE_HELPERS = ROOT / "evaluate_wham_feature_substitution.py"
MOBILE_HELPERS = ROOT / "evaluate_mobile_pipeline_3dpw.py"
OUTPUT = ROOT / "tiny_pipeline_accuracy_3dpw_kaggle.ipynb"
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
        for path in (TINY_EVALUATOR, FULL_HELPERS, FEATURE_HELPERS, MOBILE_HELPERS)
    }
    embedded_lines = "\n".join(
        f"    {name!r}: ({digest!r}, {payload!r}),"
        for name, (digest, payload) in payloads.items()
    )
    notebook = {
        "cells": [
            markdown(
                "# Final tiny-pipeline accuracy on 3DPW\n\n"
                "This notebook runs **only** the complete tiny pipeline:\n\n"
                "`YOLO26n-pose → phase-three FastViT → complete FastViT first-frame "
                "initialization → WHAM_I → WHAM_ImageStep`\n\n"
                "It does not rerun released WHAM. The final table compares against the "
                "already measured released-WHAM result on the exact same 11 single-person "
                "tracks (11,349 recurrent frames). Track names and frame counts are pinned, "
                "and the evaluator aborts if the population differs.\n\n"
                "For initialization, the first FastViT token goes through HMR2's small frozen "
                "pose and shape linear heads, followed by one licensed SMPL body decode to "
                "obtain the 17 initial 3D joints. The 2.7 GB HMR2 backbone is never created or "
                "run. This removes the previous neutral-pose/zero-3D fallback.\n\n"
                "Attach these existing Kaggle inputs before running:\n\n"
                "- `3dpw-model` (images, parsed `3dpw_test_vit.pth`, and sequence files);\n"
                "- the saved phase-three FastViT notebook output containing the epoch-36 "
                "`fastvit_hmr2_best.pth`;\n"
                "- the original saved distillation notebook output "
                "`distill-fastvit-hmr2-kagglef9b9f724ae`, which contains `hmr2a.ckpt`; and\n"
                "- the private dataset containing your three licensed SMPL model files.\n\n"
                "Enable a Kaggle GPU and Internet. Run all cells once.\n"
            ),
            code(
                "# Resolve and checksum the existing inputs. No WORK_DIR is used.\n"
                "from pathlib import Path\n"
                "import hashlib\n\n"
                "KAGGLE_INPUT = Path('/kaggle/input')\n"
                "THREEDPW_ROOT = Path('/kaggle/input/datasets/nguyntrunglong/3dpw-model')\n"
                "SCRATCH_DIR = Path('/tmp/tiny_pipeline_accuracy_3dpw')\n"
                "OUTPUT_DIR = Path('/kaggle/working/tiny_pipeline_accuracy_3dpw')\n"
                "POSE_BATCH_SIZE = 32\n"
                "STUDENT_BATCH_SIZE = 64\n"
                "SMPL_BATCH_SIZE = 256  # use 128 only if the GPU runs out of memory\n\n"
                "EXPECTED_STUDENT_SHA256 = (\n"
                "    'f15875f3fed12538312f59956b6c93e9cca2ab41a9e8d87edf593dd85f311ab1'\n"
                ")\n"
                "EXPECTED_HMR2_SHA256 = (\n"
                "    '2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e'\n"
                ")\n\n"
                "def sha256_file(path):\n"
                "    digest = hashlib.sha256()\n"
                "    with path.open('rb') as stream:\n"
                "        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):\n"
                "            digest.update(chunk)\n"
                "    return digest.hexdigest()\n\n"
                "def one_checksum_match(filename, expected):\n"
                "    candidates = sorted(KAGGLE_INPUT.rglob(filename))\n"
                "    matches = [path for path in candidates if sha256_file(path) == expected]\n"
                "    if not matches:\n"
                "        found = [(str(path), sha256_file(path)) for path in candidates]\n"
                "        raise FileNotFoundError(\n"
                "            f'Could not find {filename} with SHA-256 {expected}; found {found}'\n"
                "        )\n"
                "    if len(matches) > 1:\n"
                "        print(f'Multiple identical {filename} inputs found; using {matches[0]}')\n"
                "    return matches[0]\n\n"
                "if not THREEDPW_ROOT.is_dir():\n"
                "    raise FileNotFoundError(f'3DPW dataset is not mounted at {THREEDPW_ROOT}')\n"
                "parsed_candidates = sorted(THREEDPW_ROOT.rglob('3dpw_test_vit.pth'))\n"
                "if len(parsed_candidates) != 1:\n"
                "    raise FileNotFoundError(\n"
                "        f'Expected one 3dpw_test_vit.pth below {THREEDPW_ROOT}; found {parsed_candidates}'\n"
                "    )\n"
                "PARSED_3DPW = parsed_candidates[0]\n"
                "STUDENT_CHECKPOINT = one_checksum_match(\n"
                "    'fastvit_hmr2_best.pth', EXPECTED_STUDENT_SHA256\n"
                ")\n"
                "HMR2_CHECKPOINT = one_checksum_match('hmr2a.ckpt', EXPECTED_HMR2_SHA256)\n"
                "print(f'3DPW: {THREEDPW_ROOT}')\n"
                "print(f'Parsed labels: {PARSED_3DPW}')\n"
                "print(f'FastViT: {STUDENT_CHECKPOINT}')\n"
                "print(f'HMR2 readout source only: {HMR2_CHECKPOINT}')\n"
                "print(f'Useful output directory: {OUTPUT_DIR}')\n"
            ),
            code(
                "# Runtime dependencies. Warnings about unrelated preinstalled packages are harmless.\n"
                "%pip install -q timm==1.0.22 ultralytics==8.4.146 einops==0.8.1 "
                "yacs==0.1.8 joblib==1.5.2 loguru==0.7.3 smplx==0.1.28 "
                "chumpy==0.70 opencv-python-headless==4.10.0.84 "
                "scikit-image==0.25.2 tqdm==4.67.1\n"
            ),
            code(
                "# Normalize the licensed SMPL filenames and fetch the two public joint regressors.\n"
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
                "            f'Missing licensed SMPL asset {names}; attach your private asset dataset.'\n"
                "        )\n"
                "    if len(matches) > 1:\n"
                "        print(f'Multiple matches for {names}; using {matches[0]}')\n"
                "    return matches[0]\n\n"
                "for output_name, aliases in model_aliases.items():\n"
                "    source = find_asset(aliases)\n"
                "    shutil.copy2(source, SMPL_MODEL_DIR / output_name)\n"
                "    print(f'{output_name}: {source}')\n\n"
                "public_assets = {\n"
                "    'J_regressor_h36m.npy': (\n"
                "        'https://huggingface.co/camenduru/WHAM/resolve/main/J_regressor_h36m.npy?download=true',\n"
                "        'c655cd7013d7829eb9acbebf0e43f952a3fa0305a53c35880e39192bfb6444a0',\n"
                "    ),\n"
                "    'J_regressor_wham.npy': (\n"
                "        'https://huggingface.co/camenduru/WHAM/resolve/main/J_regressor_wham.npy?download=true',\n"
                "        'f938dcfd5cd88d0b19ee34e442d49f1dc370d3d8c4f5aef57a93d0cf2e267c4c',\n"
                "    ),\n"
                "}\n"
                "downloaded_public = {}\n"
                "for name, (url, expected) in public_assets.items():\n"
                "    path = SCRATCH_DIR / name\n"
                "    if not path.is_file():\n"
                "        print(f'Downloading {name}...', flush=True)\n"
                "        urllib.request.urlretrieve(url, path)\n"
                "    actual = sha256_file(path)\n"
                "    if actual != expected:\n"
                "        raise RuntimeError(f'{name} checksum mismatch: {actual}')\n"
                "    downloaded_public[name] = path\n"
                "H36M_REGRESSOR = downloaded_public['J_regressor_h36m.npy']\n"
                "WHAM_REGRESSOR = downloaded_public['J_regressor_wham.npy']\n"
                "print({name: sha256_file(path) for name, path in downloaded_public.items()})\n"
            ),
            code(
                "# Materialize the reviewed evaluator embedded in this notebook.\n"
                "import base64, gzip\n\n"
                "embedded = {\n"
                f"{embedded_lines}\n"
                "}\n"
                "for name, (expected_sha256, payload) in embedded.items():\n"
                "    contents = gzip.decompress(base64.b64decode(payload))\n"
                "    actual_sha256 = hashlib.sha256(contents).hexdigest()\n"
                "    if actual_sha256 != expected_sha256:\n"
                "        raise RuntimeError(f'Embedded script checksum mismatch for {name}')\n"
                "    (SCRATCH_DIR / name).write_bytes(contents)\n"
                "EVALUATOR = SCRATCH_DIR / 'evaluate_tiny_pipeline_3dpw.py'\n"
                "print({name: digest for name, (digest, _) in embedded.items()})\n"
            ),
            code(
                "# Fetch the pinned WHAM source/checkpoint and YOLO26n-pose weights.\n"
                "import subprocess\n\n"
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
                "print(f'WHAM source: {actual_commit}')\n"
                "print({name: sha256_file(path) for name, path in downloaded.items()})\n"
            ),
            code(
                "# Run only the complete tiny pipeline on the exact saved-reference population.\n"
                "import os, sys\n\n"
                "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
                "REPORT = OUTPUT_DIR / 'tiny_pipeline_accuracy_3dpw.json'\n"
                "PER_SEQUENCE = OUTPUT_DIR / 'tiny_pipeline_accuracy_3dpw.csv'\n"
                "command = [\n"
                "    sys.executable, '-u', str(EVALUATOR),\n"
                "    '--parsed-3dpw', str(PARSED_3DPW),\n"
                "    '--three-dpw-root', str(THREEDPW_ROOT),\n"
                "    '--student-checkpoint', str(STUDENT_CHECKPOINT),\n"
                "    '--hmr2-checkpoint', str(HMR2_CHECKPOINT),\n"
                "    '--wham-repo', str(WHAM_REPO),\n"
                "    '--wham-checkpoint', str(WHAM_CHECKPOINT),\n"
                "    '--yolo26-weights', str(YOLO26_WEIGHTS),\n"
                "    '--smpl-model-directory', str(SMPL_MODEL_DIR),\n"
                "    '--h36m-joint-regressor', str(H36M_REGRESSOR),\n"
                "    '--wham-joint-regressor', str(WHAM_REGRESSOR),\n"
                "    '--pose-batch-size', str(POSE_BATCH_SIZE),\n"
                "    '--student-batch-size', str(STUDENT_BATCH_SIZE),\n"
                "    '--smpl-batch-size', str(SMPL_BATCH_SIZE),\n"
                "    '--output', str(REPORT),\n"
                "    '--per-sequence-output', str(PER_SEQUENCE),\n"
                "]\n"
                "environment = os.environ.copy()\n"
                "environment['PYTHONPATH'] = (\n"
                "    str(SCRATCH_DIR) + os.pathsep + environment.get('PYTHONPATH', '')\n"
                ")\n"
                "print(' '.join(command), flush=True)\n"
                "subprocess.run(command, check=True, env=environment)\n"
            ),
            code(
                "# Final two-row accuracy table and direct delta.\n"
                "import json\n"
                "import pandas as pd\n"
                "from IPython.display import display\n\n"
                "report = json.loads(REPORT.read_text())\n"
                "comparison = report['comparison']\n"
                "original = comparison['reference_metrics']\n"
                "tiny = comparison['tiny_metrics']\n"
                "rows = [\n"
                "    {\n"
                "        'pipeline': 'Original released WHAM (saved)',\n"
                "        'PA-MPJPE (mm)': original['pa_mpjpe_mm'],\n"
                "        'MPJPE (mm)': original['mpjpe_mm'],\n"
                "        'PVE (mm)': original['pve_mm'],\n"
                "        'Accel': original['accel_official_30fps'],\n"
                "    },\n"
                "    {\n"
                "        'pipeline': 'Tiny: YOLO26 + FastViT + split WHAM',\n"
                "        'PA-MPJPE (mm)': tiny['pa_mpjpe_mm']['mean'],\n"
                "        'MPJPE (mm)': tiny['mpjpe_mm']['mean'],\n"
                "        'PVE (mm)': tiny['pve_mm']['mean'],\n"
                "        'Accel': tiny['accel_official_30fps']['mean'],\n"
                "    },\n"
                "]\n"
                "display(pd.DataFrame(rows).round(3))\n"
                "print('Tiny minus original:', json.dumps(\n"
                "    comparison['tiny_minus_original'], indent=2\n"
                "))\n"
                "print('Relative change:', json.dumps({\n"
                "    key: f'{100 * value:+.1f}%'\n"
                "    for key, value in comparison['tiny_relative_change_vs_original'].items()\n"
                "}, indent=2))\n"
                "print('Detection:', json.dumps(report['detection'], indent=2))\n"
            ),
            code(
                "# Download only the useful JSON and CSV.\n"
                "bundle_base = Path('/kaggle/working/tiny_pipeline_accuracy_3dpw_results')\n"
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
