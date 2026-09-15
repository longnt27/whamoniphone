"""Build the Kaggle notebook for causal smoothing selection and evaluation."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVALUATION_ROOT = ROOT.parent / "evaluation"
OUTPUT = ROOT / "hmr2s_temporal_smoothing_kaggle.ipynb"
WHAM_COMMIT = "2b54f7797391c94876848b905ed875b154c4a295"
HMR2S_COMMIT = "d69218f411e003621f29df1940b23b076067fad1"
HMR2S_SHA256 = "823728e846c901c75edb12d469fa240e07606a24cfd44c208244a94bb26fc423"
YOLO26M_SHA256 = "2fbf16367022256a226035695c5c389384c6706e8bb8ab8fcd0e7976f05443c4"
ADAPTER_SHA256 = "4fd581b2b7f2d0cac8bda7597692f7e77ca082435c5e352c69964d64da10f526"
SOURCES = (
    EVALUATION_ROOT / "evaluate_hmr2s_temporal_smoothing.py",
    EVALUATION_ROOT / "hmr2s_frozen.py",
    EVALUATION_ROOT / "evaluate_frozen_hmr2s_wham.py",
    EVALUATION_ROOT / "evaluate_full_pipeline_tradeoff.py",
    EVALUATION_ROOT / "evaluate_mobile_pipeline_3dpw.py",
    EVALUATION_ROOT / "evaluate_wham_feature_substitution.py",
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
                "# Is temporal smoothing worth it?\n\n"
                "This is an evaluation-only ablation of the locked deployment pipeline: "
                "YOLO26m-pose → released HMR2.0-S → validation-selected token adapter → "
                "released WHAM. Nothing is trained.\n\n"
                "The notebook compares raw output with three causal filters on the same 3DPW "
                "validation tracks. Rotations use geodesic exponential smoothing and body shape "
                "uses a causal exponential moving average. A filter is eligible only if PA-MPJPE, "
                "MPJPE, and PVE each remain within 5% of raw. The lowest predeclared normalized "
                "validation score is locked; only afterward is 3DPW test opened for raw-versus-locked "
                "reporting. Filtering happens after WHAM, so YOLO, HMR2-S, the adapter, and WHAM inputs "
                "are unchanged.\n\n"
                "## Attach these inputs\n\n"
                "1. `3dpw-model`: raw `imageFiles`, licensed SMPL files, and `3dpw_test_vit.pth`.\n"
                "2. `3dpw-vit`: `3dpw_val_vit.pth`.\n"
                "3. The saved output of the completed YOLO26 grid notebook; it must contain "
                "`m/hmr2s_to_hmr2a_adapter_best.pth`.\n"
                "4. The small private input containing official `hmr_vit-small_d3-a4x16-m128.zip` "
                "or its verified `last.ckpt`.\n\n"
                "Enable Internet and a GPU. COCO, BEDLAM, HMR2a, and `HF_TOKEN` are not needed."
            ),
            code(
                "# Configuration and bounded input discovery.\n"
                "from pathlib import Path\n"
                "import hashlib, os\n\n"
                "KAGGLE_INPUT = Path('/kaggle/input')\n"
                "SCRATCH_DIR = Path('/tmp/hmr2s_temporal_smoothing')\n"
                "OUTPUT_DIR = Path('/kaggle/working/hmr2s_temporal_smoothing')\n"
                f"ADAPTER_SHA256 = {ADAPTER_SHA256!r}\n"
                "VALIDATION_TRACKS = 12\n"
                "VALIDATION_FRAMES = 300\n"
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
                "THREEDPW_VIT_ROOT = first_directory([\n"
                "    Path('/kaggle/input/datasets/nguyntrunglong/3dpw-vit'),\n"
                "    Path('/kaggle/input/3dpw-vit'),\n"
                "], '3dpw-vit')\n"
                "VAL_PARSED = THREEDPW_VIT_ROOT / '3dpw_val_vit.pth'\n"
                "TEST_PARSED = THREEDPW_ROOT / '3dpw_test_vit.pth'\n"
                "if not TEST_PARSED.is_file():\n"
                "    matches = list(THREEDPW_ROOT.glob('*/3dpw_test_vit.pth'))\n"
                "    if len(matches) != 1:\n"
                "        raise FileNotFoundError(f'Missing 3dpw_test_vit.pth below {THREEDPW_ROOT}')\n"
                "    TEST_PARSED = matches[0]\n"
                "for path in (VAL_PARSED, TEST_PARSED):\n"
                "    if not path.is_file():\n"
                "        raise FileNotFoundError(path)\n\n"
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
                "    raise FileNotFoundError(f'Attach the saved grid output containing the selected m adapter; verified matches={adapter_matches}')\n"
                "ADAPTER_CHECKPOINT = adapter_matches[0]\n"
                "print({'3dpw_val': str(VAL_PARSED), '3dpw_test': str(TEST_PARSED), 'adapter': str(ADAPTER_CHECKPOINT)})\n"
            ),
            code(
                "# Runtime dependencies. Kaggle's CUDA PyTorch is retained.\n"
                "%pip install -q timm==1.0.22 ultralytics==8.4.146 einops==0.8.1 "
                "yacs==0.1.8 joblib==1.5.2 loguru==0.7.3 smplx==0.1.28 "
                "opencv-python-headless==4.10.0.84 scikit-image==0.25.2 "
                "scipy==1.16.1 tqdm==4.67.1 progress==1.6\n"
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
                "# Fetch exact repositories and public frozen artifacts.\n"
                "import shutil, subprocess, time, urllib.request, zipfile\n\n"
                f"WHAM_COMMIT = {WHAM_COMMIT!r}\n"
                f"HMR2S_COMMIT = {HMR2S_COMMIT!r}\n"
                f"HMR2S_SHA256 = {HMR2S_SHA256!r}\n"
                f"YOLO26M_SHA256 = {YOLO26M_SHA256!r}\n\n"
                "def checkout_exact(url, commit, destination):\n"
                "    if destination.exists():\n"
                "        shutil.rmtree(destination)\n"
                "    destination.mkdir(parents=True)\n"
                "    subprocess.run(['git', 'init', '-q'], cwd=destination, check=True)\n"
                "    subprocess.run(['git', 'remote', 'add', 'origin', url], cwd=destination, check=True)\n"
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
                "            print(f'{destination.name}: {blocks * block_size / 2**20:.1f} MiB', flush=True)\n"
                "            last[0] = now\n"
                "    urllib.request.urlretrieve(url, destination, reporthook=hook)\n\n"
                "WHAM_REPO = SCRATCH_DIR / 'WHAM'\n"
                "HMR2S_REPO = SCRATCH_DIR / 'TruncHierVFM'\n"
                "checkout_exact('https://github.com/yohanshin/WHAM.git', WHAM_COMMIT, WHAM_REPO)\n"
                "checkout_exact('https://github.com/nttcom/TruncHierVFM.git', HMR2S_COMMIT, HMR2S_REPO)\n"
                "downloads = {\n"
                "    'wham_vit_bedlam_w_3dpw.pth.tar': ('https://huggingface.co/camenduru/WHAM/resolve/main/wham_vit_bedlam_w_3dpw.pth.tar?download=true', '2ba0cb6a7dd597023a6b2ad6056e7a8b6b33144a35fabea570bfd00842cd4eaf'),\n"
                "    'yolo26m-pose.pt': ('https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-pose.pt', YOLO26M_SHA256),\n"
                "    'J_regressor_h36m.npy': ('https://huggingface.co/camenduru/WHAM/resolve/main/J_regressor_h36m.npy?download=true', 'c655cd7013d7829eb9acbebf0e43f952a3fa0305a53c35880e39192bfb6444a0'),\n"
                "    'J_regressor_wham.npy': ('https://huggingface.co/camenduru/WHAM/resolve/main/J_regressor_wham.npy?download=true', 'f938dcfd5cd88d0b19ee34e442d49f1dc370d3d8c4f5aef57a93d0cf2e267c4c'),\n"
                "}\n"
                "downloaded = {}\n"
                "for name, (url, expected) in downloads.items():\n"
                "    path = SCRATCH_DIR / name\n"
                "    if not path.is_file():\n"
                "        download_with_progress(url, path)\n"
                "    actual = sha256_file(path)\n"
                "    if actual != expected:\n"
                "        raise RuntimeError(f'{name} checksum mismatch: {actual}')\n"
                "    downloaded[name] = path\n\n"
                "hmr_files = bounded_files({'last.ckpt', 'hmr_vit-small_d3-a4x16-m128.zip'})\n"
                "checkpoint_matches = [path for path in hmr_files if path.name == 'last.ckpt' and sha256_file(path) == HMR2S_SHA256]\n"
                "if checkpoint_matches:\n"
                "    HMR2S_CHECKPOINT = checkpoint_matches[0]\n"
                "else:\n"
                "    archives = [path for path in hmr_files if path.name.endswith('.zip')]\n"
                "    if len(archives) != 1:\n"
                "        raise FileNotFoundError(f'Attach one official HMR2-S zip; found {hmr_files}')\n"
                "    extraction = SCRATCH_DIR / 'hmr2s_weights'\n"
                "    with zipfile.ZipFile(archives[0]) as archive:\n"
                "        archive.extractall(extraction)\n"
                "    matches = [path for path in extraction.rglob('last.ckpt') if sha256_file(path) == HMR2S_SHA256]\n"
                "    if len(matches) != 1:\n"
                "        raise RuntimeError('Attached HMR2-S archive is not the pinned official release')\n"
                "    HMR2S_CHECKPOINT = matches[0]\n"
                "WHAM_CHECKPOINT = downloaded['wham_vit_bedlam_w_3dpw.pth.tar']\n"
                "YOLO26M_WEIGHTS = downloaded['yolo26m-pose.pt']\n"
                "H36M_REGRESSOR = downloaded['J_regressor_h36m.npy']\n"
                "WHAM_REGRESSOR = downloaded['J_regressor_wham.npy']\n"
            ),
            code(
                "# Locate and normalize the three licensed SMPL filenames.\n"
                "SMPL_MODEL_DIR = SCRATCH_DIR / 'licensed_smpl'\n"
                "SMPL_MODEL_DIR.mkdir(parents=True, exist_ok=True)\n"
                "aliases = {\n"
                "    'SMPL_NEUTRAL.pkl': {'SMPL_NEUTRAL.pkl', 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'},\n"
                "    'SMPL_MALE.pkl': {'SMPL_MALE.pkl', 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl', 'basicModel_m_lbs_10_207_0_v1.0.0.pkl'},\n"
                "    'SMPL_FEMALE.pkl': {'SMPL_FEMALE.pkl', 'basicModel_f_lbs_10_207_0_v1.0.0.pkl'},\n"
                "}\n"
                "wanted = set().union(*aliases.values())\n"
                "found = {}\n"
                "for path in bounded_files(wanted):\n"
                "    found.setdefault(path.name, []).append(path)\n"
                "for destination, names in aliases.items():\n"
                "    matches = sorted({path for name in names for path in found.get(name, [])})\n"
                "    if not matches:\n"
                "        raise FileNotFoundError(f'Missing licensed {destination}')\n"
                "    shutil.copy2(matches[0], SMPL_MODEL_DIR / destination)\n"
                "print(sorted(path.name for path in SMPL_MODEL_DIR.iterdir()))\n"
            ),
            code(
                "# Self-test, then run the validation-lock/test-once smoothing experiment.\n"
                "import sys\n\n"
                "EVALUATOR = SCRATCH_DIR / 'evaluate_hmr2s_temporal_smoothing.py'\n"
                "environment = os.environ.copy()\n"
                "environment['PYTHONPATH'] = str(SCRATCH_DIR) + os.pathsep + environment.get('PYTHONPATH', '')\n"
                "subprocess.run([sys.executable, '-u', str(EVALUATOR), '--self-test'], check=True, env=environment)\n"
                "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
                "arguments = [\n"
                "    '--val-parsed', str(VAL_PARSED),\n"
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
                "    '--output-json', str(OUTPUT_DIR / 'hmr2s_temporal_smoothing_3dpw.json'),\n"
                "    '--output-csv', str(OUTPUT_DIR / 'hmr2s_temporal_smoothing_3dpw.csv'),\n"
                "    '--validation-tracks', str(VALIDATION_TRACKS),\n"
                "    '--validation-frames', str(VALIDATION_FRAMES),\n"
                "    '--pose-batch-size', str(POSE_BATCH_SIZE),\n"
                "    '--hmr-batch-size', str(HMR2S_BATCH_SIZE),\n"
                "    '--smpl-batch-size', str(SMPL_BATCH_SIZE),\n"
                "]\n"
                "subprocess.run([sys.executable, '-u', str(EVALUATOR), *arguments], check=True, env=environment)\n"
            ),
            code(
                "# Print the answer and package the two small result files.\n"
                "import json, zipfile\n"
                "report_path = OUTPUT_DIR / 'hmr2s_temporal_smoothing_3dpw.json'\n"
                "csv_path = OUTPUT_DIR / 'hmr2s_temporal_smoothing_3dpw.csv'\n"
                "report = json.loads(report_path.read_text())\n"
                "print('LOCKED FILTER:', report['locked_filter'])\n"
                "print(json.dumps(report['decision'], indent=2))\n"
                "manifest = {path.name: {'bytes': path.stat().st_size, 'sha256': sha256_file(path)} for path in (report_path, csv_path)}\n"
                "manifest_path = OUTPUT_DIR / 'artifact_manifest.json'\n"
                "manifest_path.write_text(json.dumps(manifest, indent=2) + '\\n')\n"
                "bundle = Path('/kaggle/working/hmr2s_temporal_smoothing_reports.zip')\n"
                "with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:\n"
                "    for path in (report_path, csv_path, manifest_path):\n"
                "        archive.write(path, arcname=path.name)\n"
                "print('Download this:', bundle)\n"
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
