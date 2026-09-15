#!/usr/bin/env python3
"""Build the no-training Kaggle notebook for the frozen phone candidate."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "frozen_hmr2s_wham_accuracy_3dpw_kaggle.ipynb"
WHAM_COMMIT = "2b54f7797391c94876848b905ed875b154c4a295"
HMR2S_COMMIT = "d69218f411e003621f29df1940b23b076067fad1"
HMR2S_CHECKPOINT_SHA256 = (
    "823728e846c901c75edb12d469fa240e07606a24cfd44c208244a94bb26fc423"
)
SOURCES = (
    ROOT / "evaluate_frozen_hmr2s_wham.py",
    ROOT / "hmr2s_frozen.py",
    ROOT / "evaluate_full_pipeline_tradeoff.py",
    ROOT / "evaluate_mobile_pipeline_3dpw.py",
    ROOT / "evaluate_wham_feature_substitution.py",
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
                "# Frozen original WHAM vs frozen iPhone candidate — 3DPW accuracy\n\n"
                "This notebook does **no training**. It runs one locked comparison on the "
                "same 3DPW test population:\n\n"
                "1. **Released WHAM:** stored official ViTPose/HMR2a observations, the "
                "official HMR2a initialization, flip averaging, and released WHAM weights.\n"
                "2. **Frozen phone candidate:** YOLO26n-pose, released HMR2.0-S's 1024-D "
                "token plus its real first-frame SMPL initialization, no flip, and the same "
                "released WHAM weights.\n\n"
                "The comparison uses all matching single-person tracks. This matches the "
                "current app, which follows one highest-confidence person and does not yet "
                "associate identities in multi-person scenes. Both rows are decoded with "
                "licensed SMPL assets only to calculate PA-MPJPE, MPJPE, PVE, and acceleration. "
                "No training, validation selection, or threshold tuning occurs.\n\n"
                "Attach inputs:\n\n"
                "- `3dpw-model`, containing raw `imageFiles`, `3dpw_test_vit.pth`, and the "
                "registered 3DPW data you used successfully before;\n"
                "- the licensed `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl`, and "
                "`SMPL_FEMALE.pkl` files, either inside `3dpw-model` or in one additional "
                "private Kaggle dataset;\n"
                "- one small private `hmr2s` Kaggle dataset containing the official "
                "`hmr_vit-small_d3-a4x16-m128.zip` you downloaded manually from the authors. "
                "The notebook also accepts the extracted `last.ckpt`.\n\n"
                "Do not attach BEDLAM, COCO, `3dpw-vit`, any FastViT checkpoint, or an old "
                "notebook output. Enable Internet and select a GPU. No Hugging Face token or "
                "`gdown` is needed. Temporary repositories and multi-gigabyte weights stay under `/tmp`; "
                "only the compact JSON/CSV report is saved to `/kaggle/working`."
            ),
            code(
                "# Configuration and bounded input discovery. No WORK_DIR is used.\n"
                "from pathlib import Path\n"
                "import hashlib, os\n\n"
                "KAGGLE_INPUT = Path('/kaggle/input')\n"
                "SCRATCH_DIR = Path('/tmp/frozen_hmr2s_wham_accuracy')\n"
                "OUTPUT_DIR = Path('/kaggle/working/frozen_hmr2s_wham_accuracy')\n"
                "SEQUENCES = 0  # 0 = every matching single-person 3DPW test track.\n"
                "FRAMES_PER_SEQUENCE = 0  # 0 = every frame.\n"
                "POSE_BATCH_SIZE = 32\n"
                "HMR2S_BATCH_SIZE = 32  # reduce to 16 only if CUDA runs out of memory.\n"
                "SMPL_BATCH_SIZE = 256  # reduce to 128 only if metric decoding OOMs.\n\n"
                "def sha256_file(path):\n"
                "    digest = hashlib.sha256()\n"
                "    with path.open('rb') as stream:\n"
                "        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):\n"
                "            digest.update(block)\n"
                "    return digest.hexdigest()\n\n"
                "root_candidates = [\n"
                "    Path('/kaggle/input/datasets/nguyntrunglong/3dpw-model'),\n"
                "    Path('/kaggle/input/3dpw-model'),\n"
                "]\n"
                "THREEDPW_ROOT = next((p for p in root_candidates if p.is_dir()), None)\n"
                "if THREEDPW_ROOT is None:\n"
                "    # Search directory names only; never crawl the 53,000 image files.\n"
                "    candidates = []\n"
                "    for parent, directories, _ in os.walk(KAGGLE_INPUT):\n"
                "        base = Path(parent)\n"
                "        if base.name == '3dpw-model':\n"
                "            candidates.append(base)\n"
                "            directories[:] = []\n"
                "        elif len(base.relative_to(KAGGLE_INPUT).parts) >= 4:\n"
                "            directories[:] = []\n"
                "    if len(candidates) != 1:\n"
                "        raise FileNotFoundError(f'Attach exactly one 3dpw-model input; found {candidates}')\n"
                "    THREEDPW_ROOT = candidates[0]\n"
                "parsed_candidates = [\n"
                "    THREEDPW_ROOT / '3dpw_test_vit.pth',\n"
                "    THREEDPW_ROOT / 'imageFiles' / '3dpw_test_vit.pth',\n"
                "]\n"
                "PARSED_3DPW = next((p for p in parsed_candidates if p.is_file()), None)\n"
                "if PARSED_3DPW is None:\n"
                "    bounded = list(THREEDPW_ROOT.glob('*/3dpw_test_vit.pth'))\n"
                "    if len(bounded) != 1:\n"
                "        raise FileNotFoundError(\n"
                "            f'3dpw_test_vit.pth is missing directly below {THREEDPW_ROOT}'\n"
                "        )\n"
                "    PARSED_3DPW = bounded[0]\n"
                "print({'3dpw_root': str(THREEDPW_ROOT), 'parsed_test': str(PARSED_3DPW), "
                "'output': str(OUTPUT_DIR), 'training': False})\n"
            ),
            code(
                "# Runtime-only dependencies. Kaggle's existing CUDA PyTorch is retained.\n"
                "%pip install -q timm==0.6.13 ultralytics==8.4.146 einops==0.8.1 "
                "yacs==0.1.8 joblib==1.5.2 loguru==0.7.3 smplx==0.1.28 "
                "opencv-python-headless==4.10.0.84 "
                "scikit-image==0.25.2 tqdm==4.67.1 progress==1.6\n"
                "%pip install -q --no-build-isolation chumpy==0.70\n"
            ),
            code(
                "# Materialize the reviewed, checksum-verified evaluation sources.\n"
                "import base64, gzip\n\n"
                "SCRATCH_DIR.mkdir(parents=True, exist_ok=True)\n"
                "embedded = {\n"
                f"{embedded_lines}\n"
                "}\n"
                "for name, (expected, payload) in embedded.items():\n"
                "    contents = gzip.decompress(base64.b64decode(payload))\n"
                "    actual = hashlib.sha256(contents).hexdigest()\n"
                "    if actual != expected:\n"
                "        raise RuntimeError(f'Embedded source checksum mismatch: {name}')\n"
                "    (SCRATCH_DIR / name).write_bytes(contents)\n"
                "print({name: digest for name, (digest, _) in embedded.items()})\n"
            ),
            code(
                "# Fetch exact source commits and frozen public weights, with visible progress.\n"
                "import shutil, subprocess, time, urllib.request, zipfile\n\n"
                f"WHAM_COMMIT = {WHAM_COMMIT!r}\n"
                f"HMR2S_COMMIT = {HMR2S_COMMIT!r}\n"
                f"HMR2S_CHECKPOINT_SHA256 = {HMR2S_CHECKPOINT_SHA256!r}\n\n"
                "def checkout_exact(url, commit, destination):\n"
                "    if destination.exists():\n"
                "        shutil.rmtree(destination)\n"
                "    destination.mkdir(parents=True)\n"
                "    subprocess.run(['git', 'init', '-q'], cwd=destination, check=True)\n"
                "    subprocess.run(['git', 'remote', 'add', 'origin', url], cwd=destination, check=True)\n"
                "    print(f'Fetching {url} at {commit}...', flush=True)\n"
                "    subprocess.run(['git', 'fetch', '--depth=1', 'origin', commit], cwd=destination, check=True)\n"
                "    subprocess.run(['git', 'checkout', '-q', '--detach', 'FETCH_HEAD'], cwd=destination, check=True)\n"
                "    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=destination, text=True).strip()\n"
                "    if actual != commit:\n"
                "        raise RuntimeError(f'Expected commit {commit}, got {actual}')\n"
                "    return actual\n\n"
                "def download_with_progress(url, destination):\n"
                "    last_print = [0.0]\n"
                "    def hook(blocks, block_size, total):\n"
                "        now = time.monotonic()\n"
                "        if now - last_print[0] >= 15 or (total > 0 and blocks * block_size >= total):\n"
                "            received = blocks * block_size\n"
                "            total_text = 'unknown' if total <= 0 else f'{total / 2**30:.2f} GiB'\n"
                "            print(f'  {destination.name}: {received / 2**30:.2f} GiB / {total_text}', flush=True)\n"
                "            last_print[0] = now\n"
                "    urllib.request.urlretrieve(url, destination, reporthook=hook)\n\n"
                "WHAM_REPO = SCRATCH_DIR / 'WHAM'\n"
                "HMR2S_REPO = SCRATCH_DIR / 'TruncHierVFM'\n"
                "checkout_exact('https://github.com/yohanshin/WHAM.git', WHAM_COMMIT, WHAM_REPO)\n"
                "checkout_exact('https://github.com/nttcom/TruncHierVFM.git', HMR2S_COMMIT, HMR2S_REPO)\n\n"
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
                "    'J_regressor_wham.npy': (\n"
                "        'https://huggingface.co/camenduru/WHAM/resolve/main/'\n"
                "        'J_regressor_wham.npy?download=true',\n"
                "        'f938dcfd5cd88d0b19ee34e442d49f1dc370d3d8c4f5aef57a93d0cf2e267c4c',\n"
                "    ),\n"
                "}\n"
                "downloaded = {}\n"
                "for name, (url, expected) in downloads.items():\n"
                "    path = SCRATCH_DIR / name\n"
                "    if not path.is_file():\n"
                "        print(f'Downloading {name}...', flush=True)\n"
                "        download_with_progress(url, path)\n"
                "    actual = sha256_file(path)\n"
                "    if actual != expected:\n"
                "        raise RuntimeError(f'{name} checksum mismatch: {actual}')\n"
                "    downloaded[name] = path\n"
                "    print(f'Verified {name}: {actual}', flush=True)\n\n"
                "hmr_files = []\n"
                "for parent, directories, files in os.walk(KAGGLE_INPUT):\n"
                "    directories[:] = [d for d in directories if d not in {'imageFiles', 'sequenceFiles'}]\n"
                "    for filename in files:\n"
                "        if filename == 'last.ckpt' or filename == 'hmr_vit-small_d3-a4x16-m128.zip':\n"
                "            hmr_files.append(Path(parent) / filename)\n"
                "checkpoint_matches = [path for path in hmr_files if path.name == 'last.ckpt' and sha256_file(path) == HMR2S_CHECKPOINT_SHA256]\n"
                "if checkpoint_matches:\n"
                "    HMR2S_CHECKPOINT = checkpoint_matches[0]\n"
                "else:\n"
                "    archives = [path for path in hmr_files if path.name.endswith('.zip')]\n"
                "    if len(archives) != 1:\n"
                "        raise FileNotFoundError(\n"
                "            'Attach one Kaggle input containing the authors’ official '\n"
                "            f'hmr_vit-small_d3-a4x16-m128.zip; found {hmr_files}'\n"
                "        )\n"
                "    hmr_extract = SCRATCH_DIR / 'hmr2s_weights'\n"
                "    print(f'Extracting the attached HMR2.0-S archive: {archives[0]}', flush=True)\n"
                "    with zipfile.ZipFile(archives[0]) as archive:\n"
                "        archive.extractall(hmr_extract)\n"
                "    candidates = list(hmr_extract.rglob('last.ckpt'))\n"
                "    checkpoint_matches = [path for path in candidates if sha256_file(path) == HMR2S_CHECKPOINT_SHA256]\n"
                "    if len(checkpoint_matches) != 1:\n"
                "        raise RuntimeError(f'Attached archive is not the pinned official HMR2.0-S release: {candidates}')\n"
                "    HMR2S_CHECKPOINT = checkpoint_matches[0]\n"
                "actual_hmr = sha256_file(HMR2S_CHECKPOINT)\n"
                "if actual_hmr != HMR2S_CHECKPOINT_SHA256:\n"
                "    raise RuntimeError(f'HMR2.0-S checkpoint checksum mismatch: {actual_hmr}')\n"
                "WHAM_CHECKPOINT = downloaded['wham_vit_bedlam_w_3dpw.pth.tar']\n"
                "YOLO26_WEIGHTS = downloaded['yolo26n-pose.pt']\n"
                "H36M_REGRESSOR = downloaded['J_regressor_h36m.npy']\n"
                "WHAM_REGRESSOR = downloaded['J_regressor_wham.npy']\n"
                "print('All frozen public artifacts verified.', flush=True)\n"
            ),
            code(
                "# Locate the three private licensed SMPL files without scanning imageFiles.\n"
                "SMPL_MODEL_DIR = SCRATCH_DIR / 'licensed_smpl'\n"
                "SMPL_MODEL_DIR.mkdir(parents=True, exist_ok=True)\n"
                "aliases = {\n"
                "    'SMPL_NEUTRAL.pkl': {'SMPL_NEUTRAL.pkl', 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'},\n"
                "    'SMPL_MALE.pkl': {'SMPL_MALE.pkl', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl', 'basicModel_m_lbs_10_207_0_v1.0.0.pkl'},\n"
                "    'SMPL_FEMALE.pkl': {'SMPL_FEMALE.pkl', 'basicModel_f_lbs_10_207_0_v1.0.0.pkl'},\n"
                "}\n"
                "wanted = set().union(*aliases.values())\n"
                "found = {}\n"
                "for parent, directories, files in os.walk(KAGGLE_INPUT):\n"
                "    directories[:] = [d for d in directories if d not in {'imageFiles', 'sequenceFiles'}]\n"
                "    for filename in files:\n"
                "        if filename in wanted:\n"
                "            found.setdefault(filename, []).append(Path(parent) / filename)\n"
                "for destination, names in aliases.items():\n"
                "    matches = sorted({path for name in names for path in found.get(name, [])})\n"
                "    if not matches:\n"
                "        raise FileNotFoundError(\n"
                "            f'Missing licensed {destination}. Attach your private SMPL model dataset.'\n"
                "        )\n"
                "    shutil.copy2(matches[0], SMPL_MODEL_DIR / destination)\n"
                "    print(f'{destination}: {matches[0]}')\n"
            ),
            code(
                "# Fast-fail smoke test before the full evaluation.\n"
                "import json, subprocess, sys\n\n"
                "environment = os.environ.copy()\n"
                "environment['PYTHONPATH'] = str(SCRATCH_DIR) + os.pathsep + environment.get('PYTHONPATH', '')\n"
                "subprocess.run([sys.executable, '-m', 'py_compile', *[str(SCRATCH_DIR / name) for name in embedded]], check=True, env=environment)\n"
                "smoke = f'''\n"
                "from pathlib import Path\n"
                "import torch\n"
                "from hmr2s_frozen import FrozenHMR2S\n"
                "model = FrozenHMR2S(Path({str(HMR2S_REPO)!r}), Path({str(HMR2S_CHECKPOINT)!r})).eval()\n"
                "with torch.inference_mode():\n"
                "    outputs = model(torch.zeros(1, 3, 256, 256))\n"
                "expected = [(1, 1024), (1, 24, 6), (1, 10), (1, 3)]\n"
                "actual = [tuple(value.shape) for value in outputs]\n"
                "assert actual == expected, (actual, expected)\n"
                "assert all(torch.isfinite(value).all() for value in outputs)\n"
                "print({{'hmr2s_shapes': actual, 'parameters': sum(p.numel() for p in model.parameters()), 'training': False}})\n"
                "'''\n"
                "subprocess.run([sys.executable, '-u', '-c', smoke], check=True, env=environment)\n"
                "print('Smoke test passed; starting the locked comparison next.', flush=True)\n"
            ),
            code(
                "# Full two-row comparison. Progress and one compact line per track are printed.\n"
                "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
                "REPORT = OUTPUT_DIR / 'frozen_hmr2s_wham_accuracy_3dpw.json'\n"
                "PER_SEQUENCE = OUTPUT_DIR / 'frozen_hmr2s_wham_accuracy_3dpw.csv'\n"
                "EVALUATOR = SCRATCH_DIR / 'evaluate_frozen_hmr2s_wham.py'\n"
                "command = [\n"
                "    sys.executable, '-u', str(EVALUATOR),\n"
                "    '--parsed-3dpw', str(PARSED_3DPW),\n"
                "    '--three-dpw-root', str(THREEDPW_ROOT),\n"
                "    '--wham-repo', str(WHAM_REPO),\n"
                "    '--wham-checkpoint', str(WHAM_CHECKPOINT),\n"
                "    '--hmr2s-repo', str(HMR2S_REPO),\n"
                "    '--hmr2s-checkpoint', str(HMR2S_CHECKPOINT),\n"
                "    '--yolo26-weights', str(YOLO26_WEIGHTS),\n"
                "    '--smpl-model-directory', str(SMPL_MODEL_DIR),\n"
                "    '--h36m-joint-regressor', str(H36M_REGRESSOR),\n"
                "    '--wham-joint-regressor', str(WHAM_REGRESSOR),\n"
                "    '--sequences', str(SEQUENCES),\n"
                "    '--frames', str(FRAMES_PER_SEQUENCE),\n"
                "    '--pose-batch-size', str(POSE_BATCH_SIZE),\n"
                "    '--hmr-batch-size', str(HMR2S_BATCH_SIZE),\n"
                "    '--smpl-batch-size', str(SMPL_BATCH_SIZE),\n"
                "    '--output', str(REPORT),\n"
                "    '--per-sequence-output', str(PER_SEQUENCE),\n"
                "]\n"
                "print('Launching frozen evaluation (training=False)...', flush=True)\n"
                "subprocess.run(command, check=True, env=environment)\n"
            ),
            code(
                "# Show the final numbers and create one small download bundle.\n"
                "import pandas as pd\n"
                "from IPython.display import display\n\n"
                "report = json.loads(REPORT.read_text())\n"
                "table = []\n"
                "for variant, result in report['variants'].items():\n"
                "    metrics = result['metrics']\n"
                "    table.append({\n"
                "        'pipeline': variant,\n"
                "        'frames': metrics['mpjpe_mm']['samples'],\n"
                "        'PA-MPJPE (mm)': metrics['pa_mpjpe_mm']['mean'],\n"
                "        'MPJPE (mm)': metrics['mpjpe_mm']['mean'],\n"
                "        'PVE (mm)': metrics['pve_mm']['mean'],\n"
                "        'Accel (m/s²)': metrics['accel_official_30fps']['mean'],\n"
                "    })\n"
                "display(pd.DataFrame(table).round(3))\n"
                "print('Phone minus released WHAM:', json.dumps(report['phone_minus_released_same_population'], indent=2))\n"
                "print('Relative change:', json.dumps(report['phone_relative_change_vs_released_same_population'], indent=2))\n"
                "print('Detection:', json.dumps(report['detection'], indent=2))\n"
                "MANIFEST = OUTPUT_DIR / 'artifact_manifest.json'\n"
                "manifest = {\n"
                "    'training_performed': False,\n"
                "    'report': REPORT.name,\n"
                "    'per_sequence': PER_SEQUENCE.name,\n"
                "    'send_back': [REPORT.name, PER_SEQUENCE.name],\n"
                "}\n"
                "MANIFEST.write_text(json.dumps(manifest, indent=2) + '\\n')\n"
                "bundle_base = Path('/kaggle/working/frozen_hmr2s_wham_accuracy_results')\n"
                "bundle = Path(shutil.make_archive(str(bundle_base), 'zip', root_dir=OUTPUT_DIR))\n"
                "print(f'Download {bundle} ({bundle.stat().st_size / 1024:.1f} KiB).')\n"
                "print('Send back the JSON and CSV, or just this ZIP. No logs/checkpoints are needed.')\n"
            ),
        ],
        "metadata": {
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
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
