#!/usr/bin/env python3
"""Build the Kaggle notebook for HMR2-S token adaptation experiments."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVALUATION_ROOT = ROOT.parent / "evaluation"
OUTPUT = ROOT / "hmr2s_wham_adapter_and_finetune_kaggle.ipynb"
WHAM_COMMIT = "2b54f7797391c94876848b905ed875b154c4a295"
HMR2S_COMMIT = "d69218f411e003621f29df1940b23b076067fad1"
HMR2S_CHECKPOINT_SHA256 = (
    "823728e846c901c75edb12d469fa240e07606a24cfd44c208244a94bb26fc423"
)
HMR2A_CHECKPOINT_SHA256 = (
    "2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e"
)
SOURCES = (
    EVALUATION_ROOT / "train_hmr2s_wham_adaptation.py",
    EVALUATION_ROOT / "hmr2s_frozen.py",
    EVALUATION_ROOT / "evaluate_frozen_hmr2s_wham.py",
    EVALUATION_ROOT / "train_bedlam_tiny_pipeline.py",
    EVALUATION_ROOT / "train_deployment_tiny_pipeline.py",
    EVALUATION_ROOT / "distill_fastvit_hmr2.py",
    EVALUATION_ROOT / "evaluate_full_pipeline_tradeoff.py",
    EVALUATION_ROOT / "evaluate_mobile_pipeline_3dpw.py",
    EVALUATION_ROOT / "evaluate_wham_feature_substitution.py",
    EVALUATION_ROOT / "evaluate_deployment_tiny_pipeline_3dpw.py",
    EVALUATION_ROOT / "finetune_fastvit_wham_downstream.py",
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
    raw = path.read_bytes()
    return (
        hashlib.sha256(raw).hexdigest(),
        base64.b64encode(gzip.compress(raw, compresslevel=9)).decode("ascii"),
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
                "# Repair the HMR2-S → WHAM interface: adapter vs native WHAM tuning\n\n"
                "This notebook runs **two independent candidates** from released weights.\n\n"
                "- **Candidate 2 — token adapter:** freeze released HMR2-S and released WHAM; "
                "learn a small residual MLP that maps the native HMR2-S 1024-D token into the "
                "HMR2a token space WHAM was trained to read.\n"
                "- **Candidate 3 — native-token WHAM:** create an independent copy of released WHAM; "
                "keep HMR2-S frozen and tune WHAM's image integrator/decoder to read native "
                "HMR2-S tokens. It never consumes Candidate 2.\n\n"
                "COCO 2017, 3DPW train, and a bounded licensed BEDLAM subset provide token "
                "pairs for Candidate 2. Only BEDLAM and 3DPW train have the temporal 3D labels "
                "needed for Candidate 3. Each epoch is selected on real end-to-end 3DPW "
                "validation SMPL metrics. Only after both candidates are locked does the "
                "notebook open 3DPW test once and report four identical-population rows. "
                "BEDLAM supervision is admitted only when YOLO confidence is at least 0.5 "
                "and its box matches the labeled person at IoU at least 0.45. The rows are: "
                "released WHAM, the naïve direct HMR2-S plug, Candidate 2, and Candidate 3.\n\n"
                "## Attach these inputs\n\n"
                "1. `3dpw-model`: raw `imageFiles`, `sequenceFiles`, and `3dpw_test_vit.pth`.\n"
                "2. `3dpw-vit`: `3dpw_train_vit.pth` and `3dpw_val_vit.pth`.\n"
                "3. `bedlam`: extracted `bedlam-labels/*.npz`.\n"
                "4. COCO 2017: `awsaf49/coco-2017-dataset`.\n"
                "5. `distill-fastvit-hmr2-kagglef9b9f724ae`: the saved notebook output "
                "containing the verified official `hmr2a.ckpt`; no FastViT checkpoint is used.\n"
                "6. A small private dataset containing the official "
                "`hmr_vit-small_d3-a4x16-m128.zip` (or its verified `last.ckpt`).\n"
                "7. Your private licensed `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl`, and "
                "`SMPL_FEMALE.pkl` files (they may already be inside `3dpw-model`).\n\n"
                "Enable Internet and a GPU, and expose the existing `HF_TOKEN` Kaggle secret. "
                "Do not attach any old trained tiny-pipeline checkpoint. Temporary caches and "
                "BEDLAM downloads stay under `/tmp`; only reports and the two selected candidate "
                "checkpoints are saved under `/kaggle/working`."
            ),
            code(
                "# Configuration and bounded input discovery.\n"
                "from pathlib import Path\n"
                "import hashlib, os\n\n"
                "KAGGLE_INPUT = Path('/kaggle/input')\n"
                "SCRATCH_DIR = Path('/tmp/hmr2s_wham_adaptation')\n"
                "OUTPUT_DIR = Path('/kaggle/working/hmr2s_wham_adaptation')\n"
                "COCO_TRAIN_PEOPLE = 20000\n"
                "COCO_VAL_PEOPLE = 2000\n"
                "MAXIMUM_BEDLAM_SCENES = 10\n"
                "MAXIMUM_BEDLAM_DOWNLOAD_GIB = 4.0\n"
                "BEDLAM_VIDEOS_PER_SCENE = 24\n"
                "BEDLAM_FRAMES_PER_VIDEO = 200\n"
                "VALIDATION_TRACKS = 12\n"
                "VALIDATION_FRAMES = 300\n"
                "ADAPTER_EPOCHS = 6\n"
                "INTEGRATION_EPOCHS = 3\n"
                "DECODER_EPOCHS = 3\n"
                "POSE_BATCH_SIZE = 24\n"
                "TEACHER_BATCH_SIZE = 8  # proven safe for the large HMR2a teacher on T4.\n"
                "HMR2S_BATCH_SIZE = 24\n"
                "SMPL_BATCH_SIZE = 192\n"
                "WORKERS = 2\n\n"
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
                "THREEDPW_VIT_ROOT = first_directory([\n"
                "    Path('/kaggle/input/datasets/nguyntrunglong/3dpw-vit'),\n"
                "    Path('/kaggle/input/3dpw-vit'),\n"
                "], '3dpw-vit')\n"
                "BEDLAM_DATASET_ROOT = first_directory([\n"
                "    Path('/kaggle/input/datasets/nguyntrunglong/bedlam'),\n"
                "    Path('/kaggle/input/bedlam'),\n"
                "], 'bedlam')\n"
                "bedlam_label_candidates = [\n"
                "    BEDLAM_DATASET_ROOT / 'bedlam-labels',\n"
                "    BEDLAM_DATASET_ROOT / 'bedlam-labels' / 'bedlam-labels',\n"
                "    BEDLAM_DATASET_ROOT,\n"
                "]\n"
                "BEDLAM_LABEL_ROOT = next((path for path in bedlam_label_candidates if path.is_dir() and len(list(path.glob('*.npz'))) >= 20), None)\n"
                "if BEDLAM_LABEL_ROOT is None:\n"
                "    raise FileNotFoundError(f'No directory with the extracted BEDLAM npz files: {bedlam_label_candidates}')\n"
                "COCO_ROOT = first_directory([\n"
                "    Path('/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017'),\n"
                "    Path('/kaggle/input/awsaf49/coco-2017-dataset/coco2017'),\n"
                "    Path('/kaggle/input/coco-2017-dataset/coco2017'),\n"
                "], 'COCO 2017')\n"
                "TRAIN_PARSED = THREEDPW_VIT_ROOT / '3dpw_train_vit.pth'\n"
                "VAL_PARSED = THREEDPW_VIT_ROOT / '3dpw_val_vit.pth'\n"
                "TEST_PARSED = THREEDPW_ROOT / '3dpw_test_vit.pth'\n"
                "if not TEST_PARSED.is_file():\n"
                "    matches = list(THREEDPW_ROOT.glob('*/3dpw_test_vit.pth'))\n"
                "    if len(matches) != 1:\n"
                "        raise FileNotFoundError(f'Missing 3dpw_test_vit.pth below {THREEDPW_ROOT}')\n"
                "    TEST_PARSED = matches[0]\n"
                "for path in (TRAIN_PARSED, VAL_PARSED, TEST_PARSED):\n"
                "    if not path.is_file():\n"
                "        raise FileNotFoundError(path)\n\n"
                "from kaggle_secrets import UserSecretsClient\n"
                "HF_TOKEN = UserSecretsClient().get_secret('HF_TOKEN')\n"
                "if not HF_TOKEN:\n"
                "    raise RuntimeError('Expose the HF_TOKEN Kaggle secret to this notebook')\n"
                "os.environ['HF_TOKEN'] = HF_TOKEN\n"
                "print({\n"
                "    '3dpw_root': str(THREEDPW_ROOT), '3dpw_train': str(TRAIN_PARSED),\n"
                "    '3dpw_val': str(VAL_PARSED), '3dpw_test': str(TEST_PARSED),\n"
                "    'bedlam_labels': str(BEDLAM_LABEL_ROOT), 'coco2017': str(COCO_ROOT),\n"
                "    'output': str(OUTPUT_DIR),\n"
                "})\n"
            ),
            code(
                "# Runtime dependencies. Kaggle's existing CUDA PyTorch is retained.\n"
                "%pip install -q timm==1.0.22 ultralytics==8.4.146 einops==0.8.1 "
                "yacs==0.1.8 joblib==1.5.2 loguru==0.7.3 smplx==0.1.28 "
                "opencv-python-headless==4.10.0.84 scikit-image==0.25.2 "
                "huggingface_hub==0.36.0 'hf_xet>=1.1.5,<2' tqdm==4.67.1 progress==1.6\n"
                "%pip install -q --no-build-isolation chumpy==0.70\n"
            ),
            code(
                "# Materialize the reviewed, checksum-verified experiment sources.\n"
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
                "# Fetch exact source commits and public frozen artifacts.\n"
                "import shutil, subprocess, time, urllib.request, zipfile\n\n"
                f"WHAM_COMMIT = {WHAM_COMMIT!r}\n"
                f"HMR2S_COMMIT = {HMR2S_COMMIT!r}\n"
                f"HMR2S_CHECKPOINT_SHA256 = {HMR2S_CHECKPOINT_SHA256!r}\n"
                f"HMR2A_CHECKPOINT_SHA256 = {HMR2A_CHECKPOINT_SHA256!r}\n\n"
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
                "        raise RuntimeError(f'Expected {commit}, got {actual}')\n\n"
                "def download_with_progress(url, destination):\n"
                "    last = [0.0]\n"
                "    def hook(blocks, block_size, total):\n"
                "        now = time.monotonic()\n"
                "        if now - last[0] >= 15 or (total > 0 and blocks * block_size >= total):\n"
                "            got = blocks * block_size\n"
                "            total_text = 'unknown' if total <= 0 else f'{total / 2**30:.2f} GiB'\n"
                "            print(f'  {destination.name}: {got / 2**30:.2f} GiB / {total_text}', flush=True)\n"
                "            last[0] = now\n"
                "    urllib.request.urlretrieve(url, destination, reporthook=hook)\n\n"
                "WHAM_REPO = SCRATCH_DIR / 'WHAM'\n"
                "HMR2S_REPO = SCRATCH_DIR / 'TruncHierVFM'\n"
                "checkout_exact('https://github.com/yohanshin/WHAM.git', WHAM_COMMIT, WHAM_REPO)\n"
                "checkout_exact('https://github.com/nttcom/TruncHierVFM.git', HMR2S_COMMIT, HMR2S_REPO)\n"
                "downloads = {\n"
                "    'wham_vit_bedlam_w_3dpw.pth.tar': (\n"
                "        'https://huggingface.co/camenduru/WHAM/resolve/main/wham_vit_bedlam_w_3dpw.pth.tar?download=true',\n"
                "        '2ba0cb6a7dd597023a6b2ad6056e7a8b6b33144a35fabea570bfd00842cd4eaf'),\n"
                "    'yolo26n-pose.pt': (\n"
                "        'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt',\n"
                "        'eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9'),\n"
                "    'J_regressor_h36m.npy': (\n"
                "        'https://huggingface.co/camenduru/WHAM/resolve/main/J_regressor_h36m.npy?download=true',\n"
                "        'c655cd7013d7829eb9acbebf0e43f952a3fa0305a53c35880e39192bfb6444a0'),\n"
                "    'J_regressor_wham.npy': (\n"
                "        'https://huggingface.co/camenduru/WHAM/resolve/main/J_regressor_wham.npy?download=true',\n"
                "        'f938dcfd5cd88d0b19ee34e442d49f1dc370d3d8c4f5aef57a93d0cf2e267c4c'),\n"
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
                "    print(f'Verified {name}', flush=True)\n\n"
                "def bounded_files(names):\n"
                "    found = []\n"
                "    for parent, directories, files in os.walk(KAGGLE_INPUT):\n"
                "        directories[:] = [name for name in directories if name not in {'imageFiles', 'sequenceFiles', 'coco2017'}]\n"
                "        for filename in files:\n"
                "            if filename in names:\n"
                "                found.append(Path(parent) / filename)\n"
                "    return found\n\n"
                "hmr2a_matches = [path for path in bounded_files({'hmr2a.ckpt'}) if sha256_file(path) == HMR2A_CHECKPOINT_SHA256]\n"
                "if len(hmr2a_matches) != 1:\n"
                "    raise FileNotFoundError(f'Attach the saved official hmr2a.ckpt input; verified matches={hmr2a_matches}')\n"
                "HMR2A_CHECKPOINT = hmr2a_matches[0]\n"
                "hmr_files = bounded_files({'last.ckpt', 'hmr_vit-small_d3-a4x16-m128.zip'})\n"
                "checkpoint_matches = [path for path in hmr_files if path.name == 'last.ckpt' and sha256_file(path) == HMR2S_CHECKPOINT_SHA256]\n"
                "if checkpoint_matches:\n"
                "    HMR2S_CHECKPOINT = checkpoint_matches[0]\n"
                "else:\n"
                "    archives = [path for path in hmr_files if path.name.endswith('.zip')]\n"
                "    if len(archives) != 1:\n"
                "        raise FileNotFoundError(f'Attach one official HMR2-S zip; found {hmr_files}')\n"
                "    extraction = SCRATCH_DIR / 'hmr2s_weights'\n"
                "    with zipfile.ZipFile(archives[0]) as archive:\n"
                "        archive.extractall(extraction)\n"
                "    matches = [path for path in extraction.rglob('last.ckpt') if sha256_file(path) == HMR2S_CHECKPOINT_SHA256]\n"
                "    if len(matches) != 1:\n"
                "        raise RuntimeError('The attached HMR2-S archive is not the pinned official release')\n"
                "    HMR2S_CHECKPOINT = matches[0]\n"
                "WHAM_CHECKPOINT = downloaded['wham_vit_bedlam_w_3dpw.pth.tar']\n"
                "YOLO26_WEIGHTS = downloaded['yolo26n-pose.pt']\n"
                "H36M_REGRESSOR = downloaded['J_regressor_h36m.npy']\n"
                "WHAM_REGRESSOR = downloaded['J_regressor_wham.npy']\n"
                "print({'hmr2a': str(HMR2A_CHECKPOINT), 'hmr2s': str(HMR2S_CHECKPOINT)})\n"
            ),
            code(
                "# Locate and normalize the three licensed SMPL model filenames.\n"
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
                "    directories[:] = [name for name in directories if name not in {'imageFiles', 'sequenceFiles', 'coco2017'}]\n"
                "    for filename in files:\n"
                "        if filename in wanted:\n"
                "            found.setdefault(filename, []).append(Path(parent) / filename)\n"
                "for destination, names in aliases.items():\n"
                "    matches = sorted({path for name in names for path in found.get(name, [])})\n"
                "    if not matches:\n"
                "        raise FileNotFoundError(f'Missing licensed {destination}')\n"
                "    shutil.copy2(matches[0], SMPL_MODEL_DIR / destination)\n"
                "print(sorted(path.name for path in SMPL_MODEL_DIR.iterdir()))\n"
            ),
            code(
                "# Fast fail before the long run: code self-test and complete data/HF preflight.\n"
                "import sys\n\n"
                "TRAINER = SCRATCH_DIR / 'train_hmr2s_wham_adaptation.py'\n"
                "common = [\n"
                "    '--coco-root', str(COCO_ROOT),\n"
                "    '--bedlam-label-root', str(BEDLAM_LABEL_ROOT),\n"
                "    '--three-dpw-root', str(THREEDPW_ROOT),\n"
                "    '--sequence-root', str(THREEDPW_ROOT),\n"
                "    '--train-parsed', str(TRAIN_PARSED),\n"
                "    '--val-parsed', str(VAL_PARSED),\n"
                "    '--test-parsed', str(TEST_PARSED),\n"
                "    '--wham-repo', str(WHAM_REPO),\n"
                "    '--wham-checkpoint', str(WHAM_CHECKPOINT),\n"
                "    '--hmr2a-checkpoint', str(HMR2A_CHECKPOINT),\n"
                "    '--hmr2s-repo', str(HMR2S_REPO),\n"
                "    '--hmr2s-checkpoint', str(HMR2S_CHECKPOINT),\n"
                "    '--yolo26-weights', str(YOLO26_WEIGHTS),\n"
                "    '--smpl-model-directory', str(SMPL_MODEL_DIR),\n"
                "    '--h36m-joint-regressor', str(H36M_REGRESSOR),\n"
                "    '--wham-joint-regressor', str(WHAM_REGRESSOR),\n"
                "    '--output-dir', str(OUTPUT_DIR),\n"
                "    '--scratch-dir', str(SCRATCH_DIR / 'runtime'),\n"
                "    '--coco-train-people', str(COCO_TRAIN_PEOPLE),\n"
                "    '--coco-val-people', str(COCO_VAL_PEOPLE),\n"
                "    '--maximum-bedlam-scenes', str(MAXIMUM_BEDLAM_SCENES),\n"
                "    '--maximum-bedlam-download-gib', str(MAXIMUM_BEDLAM_DOWNLOAD_GIB),\n"
                "    '--bedlam-videos-per-scene', str(BEDLAM_VIDEOS_PER_SCENE),\n"
                "    '--bedlam-frames-per-video', str(BEDLAM_FRAMES_PER_VIDEO),\n"
                "    '--validation-tracks', str(VALIDATION_TRACKS),\n"
                "    '--validation-frames', str(VALIDATION_FRAMES),\n"
                "    '--adapter-epochs', str(ADAPTER_EPOCHS),\n"
                "    '--integration-epochs', str(INTEGRATION_EPOCHS),\n"
                "    '--decoder-epochs', str(DECODER_EPOCHS),\n"
                "    '--pose-batch-size', str(POSE_BATCH_SIZE),\n"
                "    '--teacher-batch-size', str(TEACHER_BATCH_SIZE),\n"
                "    '--hmr-batch-size', str(HMR2S_BATCH_SIZE),\n"
                "    '--smpl-batch-size', str(SMPL_BATCH_SIZE),\n"
                "    '--workers', str(WORKERS),\n"
                "]\n"
                "environment = os.environ.copy()\n"
                "environment['PYTHONPATH'] = str(SCRATCH_DIR) + os.pathsep + environment.get('PYTHONPATH', '')\n"
                "subprocess.run([sys.executable, '-u', str(TRAINER), '--self-test'], check=True, env=environment)\n"
                "subprocess.run([sys.executable, '-u', str(TRAINER), '--inspect-data', *common], check=True, env=environment)\n"
            ),
            code(
                "# Long cell: cache all three training domains, train both branches, lock on validation, then test once.\n"
                "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
                "print('Starting the two independent adaptation experiments...', flush=True)\n"
                "subprocess.run([sys.executable, '-u', str(TRAINER), *common], check=True, env=environment)\n"
            ),
            code(
                "# Show the report-ready result and create a small reports-only download.\n"
                "import json, zipfile\n"
                "training_report = json.loads((OUTPUT_DIR / 'hmr2s_wham_training_report.json').read_text())\n"
                "final_report = json.loads((OUTPUT_DIR / 'hmr2s_wham_final_3dpw.json').read_text())\n"
                "print('Validation-selected candidate:', training_report['protocol']['validation_selected_candidate'])\n"
                "print(json.dumps(training_report['final_test_summary'], indent=2))\n"
                "manifest = {}\n"
                "for path in sorted(OUTPUT_DIR.iterdir()):\n"
                "    if path.is_file():\n"
                "        manifest[path.name] = {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}\n"
                "(OUTPUT_DIR / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2) + '\\n')\n"
                "reports = [\n"
                "    OUTPUT_DIR / 'hmr2s_wham_training_report.json',\n"
                "    OUTPUT_DIR / 'hmr2s_wham_training_history.csv',\n"
                "    OUTPUT_DIR / 'hmr2s_wham_final_3dpw.json',\n"
                "    OUTPUT_DIR / 'hmr2s_wham_final_3dpw.csv',\n"
                "    OUTPUT_DIR / 'artifact_manifest.json',\n"
                "]\n"
                "bundle = Path('/kaggle/working/hmr2s_wham_adaptation_reports.zip')\n"
                "with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:\n"
                "    for path in reports:\n"
                "        archive.write(path, arcname=path.name)\n"
                "print('Download this for review:', bundle)\n"
                "print('Keep the saved notebook output: the two selected checkpoints remain in', OUTPUT_DIR)\n"
            ),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
            "kaggle": {"accelerator": "gpu", "internet": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
