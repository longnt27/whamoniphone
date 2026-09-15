#!/usr/bin/env python3
"""Build the global pooled BEDLAM replay Kaggle notebook."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

import build_bedlam_tiny_pipeline_notebook as base_builder

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "bedlam_global_replay_and_test_kaggle.ipynb"
SOURCES = (
    ROOT / "train_bedlam_hmr2_replay.py",
    ROOT / "train_bedlam_tiny_pipeline.py",
    ROOT / "distill_fastvit_hmr2.py",
    ROOT / "train_deployment_tiny_pipeline.py",
    ROOT / "evaluate_deployment_tiny_pipeline_3dpw.py",
    ROOT / "evaluate_wham_feature_substitution.py",
    ROOT / "evaluate_mobile_pipeline_3dpw.py",
    ROOT / "evaluate_full_pipeline_tradeoff.py",
    ROOT / "finetune_fastvit_wham_downstream.py",
)


def lines(source: str) -> list[str]:
    return source.splitlines(True)


def embedded(path: Path) -> tuple[str, str]:
    raw = path.read_bytes()
    return (
        hashlib.sha256(raw).hexdigest(),
        base64.b64encode(gzip.compress(raw, compresslevel=9)).decode("ascii"),
    )


def main() -> None:
    # Reuse the already tested input discovery, dependency setup, pinned assets,
    # final evaluator, and SMPL normalization cells from the prior notebook.
    base_builder.main()
    notebook = json.loads(base_builder.OUTPUT.read_text(encoding="utf-8"))

    notebook["cells"][0]["source"] = lines(
        "# Global BEDLAM geometry retraining with HMR2 and real-data replay\n\n"
        "This fixes the previous notebook's per-scene rollback error. It performs one "
        "global pooled retraining run:\n\n"
        "1. stream a size-bounded set of licensed BEDLAM MP4 archives;\n"
        "2. keep only YOLO26 detections that match the labeled person at IoU >= 0.45;\n"
        "3. pool every accepted crop and temporal clip before any optimization;\n"
        "4. use direct BEDLAM pose/shape supervision as the primary objective and HMR2's "
        "1024-D token as a regularizer;\n"
        "5. pair every BEDLAM optimization step with a supervised 3DPW-train replay step;\n"
        "6. fine-tune FastViT's last backbone stage, spatial head, initializer, and WHAM's "
        "input-facing adapters;\n"
        "7. accumulate training across global epochs and validate each epoch with actual SMPL "
        "PA-MPJPE, MPJPE, PVE, and acceleration on every available validation track (up to 16); "
        "and\n"
        "8. run the selected tiny pipeline on the established 11-track 3DPW comparison.\n\n"
        "HMR2 remains a training teacher only. The deployed path is still "
        "`YOLO26 -> FastViT -> learned initializer -> split WHAM`.\n\n"
        "## Attach exactly these six inputs\n\n"
        "- `bedlam`: the extracted `bedlam-labels/*.npz` dataset;\n"
        "- `3dpw-model`: raw imageFiles, sequenceFiles, and `3dpw_test_vit.pth`;\n"
        "- `3dpw-vit`: `3dpw_train_vit.pth` and `3dpw_val_vit.pth`;\n"
        "- your private licensed SMPL model dataset;\n"
        "- `train-and-test-deployment-tiny-pipeline-kaggle`, containing the previous winning "
        "`tiny_pipeline_best.pth` (SHA-256 starts `d47ac0c8`); and\n"
        "- `distill-fastvit-hmr2-kagglef9b9f724ae`, containing `hmr2a.ckpt` "
        "(SHA-256 starts `2dcf7963`).\n\n"
        "Do not attach the rejected BEDLAM checkpoint. Enable Internet, a GPU, and the existing "
        "`HF_TOKEN` Kaggle secret. Expected T4 runtime is roughly 3-5 hours; the notebook "
        "prints progress during every slow section.\n"
    )

    config = "".join(notebook["cells"][1]["source"])
    config = config.replace(
        "# Bounded BEDLAM run: six diverse scene archives, no more than 2 GiB total.",
        "# Bounded BEDLAM run: twelve diverse scene archives, no more than 4.5 GiB total.",
    )
    config = config.replace(
        "SCRATCH_DIR = Path('/tmp/bedlam_tiny_pipeline')",
        "SCRATCH_DIR = Path('/tmp/bedlam_global_replay')",
    )
    config = config.replace(
        "OUTPUT_DIR = Path('/kaggle/working/bedlam_tiny_pipeline')",
        "OUTPUT_DIR = Path('/kaggle/working/bedlam_global_replay')",
    )
    config = config.replace("MAXIMUM_SCENES = 6", "MAXIMUM_SCENES = 12")
    config = config.replace("MAXIMUM_DOWNLOAD_GIB = 2.0", "MAXIMUM_DOWNLOAD_GIB = 4.5")
    config = config.replace("VIDEOS_PER_SCENE = 32", "VIDEOS_PER_SCENE = 24")
    config = config.replace("THREEDPW_MAX_CLIPS = 1200", "THREEDPW_MAX_CLIPS = 1600")
    config = config.replace("VAL_TRACKS = 8", "VAL_TRACKS = 16")
    config = config.replace(
        "BEDLAM_FRAME_BATCH_SIZE = 24\n",
        "BEDLAM_FRAME_BATCH_SIZE = 20\n"
        "HMR2_BATCH_SIZE = 6\n"
        "TOKEN_EPOCHS = 2\n"
        "MIXED_EPOCHS = 8\n"
        "TOKEN_STEPS_PER_EPOCH = 256\n"
        "MIXED_STEPS_PER_EPOCH = 256\n"
        "BEDLAM_REPLAY_WEIGHT = 0.30\n"
        "MINIMUM_BEDLAM_SAMPLES = 200\n"
        "MINIMUM_GLOBAL_SAMPLES = 3000\n"
        "MINIMUM_GLOBAL_CLIPS = 150\n"
        "DIVERGENCE_SCORE = 1.03\n",
    )
    hmr2_discovery = (
        "EXPECTED_HMR2_SHA256 = "
        "'2dcf79638109781d1ae5f5c44fee5f55bc83291c210653feead9b7f04fa6f20e'\n"
        "hmr2_candidates = sorted(KAGGLE_INPUT.rglob('hmr2a.ckpt'))\n"
        "hmr2_matches = [path for path in hmr2_candidates "
        "if sha256_file(path) == EXPECTED_HMR2_SHA256]\n"
        "if not hmr2_matches:\n"
        "    found = [(str(path), sha256_file(path)) for path in hmr2_candidates]\n"
        "    raise FileNotFoundError(\n"
        "        'Attach distill-fastvit-hmr2-kagglef9b9f724ae containing the pinned '"
        "        f'hmr2a.ckpt; found={found}'\n"
        "    )\n"
        "HMR2_CHECKPOINT = hmr2_matches[0]\n"
        "print({'hmr2_checkpoint': str(HMR2_CHECKPOINT), "
        "'hmr2_sha256': sha256_file(HMR2_CHECKPOINT)})\n\n"
    )
    marker = "def find_3dpw_root(test_path):\n"
    if marker not in config:
        raise RuntimeError("Base configuration cell changed")
    config = config.replace(marker, hmr2_discovery + marker)
    notebook["cells"][1]["source"] = lines(config)

    payloads = {path.name: embedded(path) for path in SOURCES}
    embedded_lines = "\n".join(
        f"    {name!r}: ({digest!r}, {payload!r}),"
        for name, (digest, payload) in payloads.items()
    )
    notebook["cells"][4]["source"] = lines(
        "# Materialize checksum-verified programs embedded in this notebook.\n"
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
    )

    notebook["cells"][6]["source"] = lines(
        "# Fast fail before downloads/training: hashes, labels, HF layout, and licensed assets.\n"
        "import sys\n\n"
        "TRAINER = SCRATCH_DIR / 'train_bedlam_hmr2_replay.py'\n"
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
        "    '--hmr2-checkpoint', str(HMR2_CHECKPOINT),\n"
        "    '--smpl-model-directory', str(SMPL_MODEL_DIR),\n"
        "    '--h36m-joint-regressor', str(H36M_REGRESSOR),\n"
        "    '--maximum-scenes', str(MAXIMUM_SCENES),\n"
        "    '--maximum-download-gib', str(MAXIMUM_DOWNLOAD_GIB),\n"
        "    '--val-tracks', str(VAL_TRACKS),\n"
        "    '--val-frames', str(VAL_FRAMES),\n"
        "]\n"
        "environment = os.environ.copy()\n"
        "environment['PYTHONPATH'] = str(SCRATCH_DIR) + os.pathsep + environment.get('PYTHONPATH', '')\n"
        "subprocess.run([sys.executable, '-u', str(TRAINER), '--self-test'], check=True, env=environment)\n"
        "subprocess.run([sys.executable, '-u', str(TRAINER), '--inspect-data', *common], check=True, env=environment)\n"
    )

    notebook["cells"][7]["source"] = lines(
        "# Long cell: build the global BEDLAM pool, train across global epochs, then select.\n"
        "OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
        "training_command = [\n"
        "    sys.executable, '-u', str(TRAINER), *common,\n"
        "    '--videos-per-scene', str(VIDEOS_PER_SCENE),\n"
        "    '--frames-per-video', str(FRAMES_PER_VIDEO),\n"
        "    '--clips-per-video', str(CLIPS_PER_VIDEO),\n"
        "    '--bedlam-clip-length', str(BEDLAM_CLIP_LENGTH),\n"
        "    '--bedlam-clip-stride', str(BEDLAM_CLIP_STRIDE),\n"
        "    '--bedlam-frame-batch-size', str(BEDLAM_FRAME_BATCH_SIZE),\n"
        "    '--bedlam-clip-batch-size', str(BEDLAM_BATCH_SIZE),\n"
        "    '--hmr2-batch-size', str(HMR2_BATCH_SIZE),\n"
        "    '--token-epochs', str(TOKEN_EPOCHS),\n"
        "    '--mixed-epochs', str(MIXED_EPOCHS),\n"
        "    '--token-steps-per-epoch', str(TOKEN_STEPS_PER_EPOCH),\n"
        "    '--mixed-steps-per-epoch', str(MIXED_STEPS_PER_EPOCH),\n"
        "    '--bedlam-weight', str(BEDLAM_REPLAY_WEIGHT),\n"
        "    '--minimum-bedlam-samples', str(MINIMUM_BEDLAM_SAMPLES),\n"
        "    '--minimum-global-samples', str(MINIMUM_GLOBAL_SAMPLES),\n"
        "    '--minimum-global-clips', str(MINIMUM_GLOBAL_CLIPS),\n"
        "    '--divergence-score', str(DIVERGENCE_SCORE),\n"
        "    '--yolo-batch-size', str(YOLO_BATCH_SIZE),\n"
        "    '--three-dpw-clip-length', str(THREEDPW_CLIP_LENGTH),\n"
        "    '--three-dpw-stride', str(THREEDPW_STRIDE),\n"
        "    '--three-dpw-max-clips', str(THREEDPW_MAX_CLIPS),\n"
        "    '--three-dpw-batch-size', str(THREEDPW_BATCH_SIZE),\n"
        "    '--workers', str(WORKERS),\n"
        "    '--feature-batch-size', str(FEATURE_BATCH_SIZE),\n"
        "    '--smpl-batch-size', str(SMPL_BATCH_SIZE),\n"
        "]\n"
        "print('Starting global pooled BEDLAM geometry training...', flush=True)\n"
        "subprocess.run(training_command, check=True, env=environment)\n"
        "DEPLOYMENT_CHECKPOINT = OUTPUT_DIR / 'bedlam_global_replay_best.pth'\n"
        "TRAINING_REPORT = OUTPUT_DIR / 'bedlam_global_replay_training_report.json'\n"
        "TRAINING_HISTORY = OUTPUT_DIR / 'bedlam_global_replay_history.csv'\n"
        "DOWNLOAD_MANIFEST = OUTPUT_DIR / 'bedlam_global_replay_manifest.json'\n"
        "for path in (DEPLOYMENT_CHECKPOINT, TRAINING_REPORT, TRAINING_HISTORY, DOWNLOAD_MANIFEST):\n"
        "    if not path.is_file():\n"
        "        raise FileNotFoundError(f'Training did not produce {path}')\n"
    )

    test_cell = "".join(notebook["cells"][8]["source"])
    test_cell = test_cell.replace(
        "# One confirmatory 3DPW test after all validation-based selection is finished.",
        "# Iterative 3DPW comparison after validation-based selection is finished.",
    ).replace(
        "Selection is locked. Starting the single confirmatory 3DPW test...",
        "Starting the post-training 3DPW comparison...",
    )
    test_cell = test_cell.replace(
        "bedlam_tiny_pipeline_final_3dpw.json",
        "bedlam_global_replay_final_3dpw.json",
    ).replace(
        "bedlam_tiny_pipeline_final_3dpw.csv",
        "bedlam_global_replay_final_3dpw.csv",
    )
    test_cell += (
        "\n# This is an iterative engineering comparison: earlier test results informed "
        "the retraining design.\n"
        "import json\n"
        "test_payload = json.loads(TEST_REPORT.read_text())\n"
        "test_payload['scope']['test_run_policy'] = (\n"
        "    'iterative comparison after prior test feedback; not a pristine untouched benchmark'\n"
        ")\n"
        "test_payload['scope']['prior_test_result_informed_training_design'] = True\n"
        "TEST_REPORT.write_text(json.dumps(test_payload, indent=2) + '\\n')\n"
    )
    notebook["cells"][8]["source"] = lines(test_cell)

    notebook["cells"][9]["source"] = lines(
        "# Compact result tables and one download bundle.\n"
        "import json, pandas as pd, shutil\n"
        "from IPython.display import display\n\n"
        "training = json.loads(TRAINING_REPORT.read_text())\n"
        "test = json.loads(TEST_REPORT.read_text())\n"
        "baseline = training['baseline_validation']\n"
        "best = training['best_validation']\n"
        "display(pd.DataFrame([\n"
        "    {'validation model': 'Previous tiny checkpoint', **{key: baseline[key] for key in "
        "('pa_mpjpe_mm','mpjpe_mm','pve_mm','accel_official_30fps')}},\n"
        "    {'validation model': f\"Selected: {training['best_stage']}\", **{key: best[key] for key in "
        "('pa_mpjpe_mm','mpjpe_mm','pve_mm','accel_official_30fps')}},\n"
        "]).round(3))\n"
        "original = test['comparison']['reference_metrics']\n"
        "tiny = test['comparison']['tiny_metrics']\n"
        "previous = training['previous_tiny_test']\n"
        "display(pd.DataFrame([\n"
        "    {'test model': 'Released WHAM', **original},\n"
        "    {'test model': 'Previous tiny', **previous},\n"
        "    {'test model': 'Global BEDLAM replay tiny', **{key: tiny[key]['mean'] for key in previous}},\n"
        "]).round(3))\n"
        "print(json.dumps({\n"
        "    'best_stage': training['best_stage'],\n"
        "    'best_epoch': training['best_epoch'],\n"
        "    'checkpoint_sha256': training['best_checkpoint_sha256'],\n"
        "    'validation_composite_change_percent': 100 * (training['best_composite'] - 1.0),\n"
        "    'test_change_vs_previous_tiny_percent': {\n"
        "        key: 100 * (tiny[key]['mean'] / previous[key] - 1.0) for key in previous\n"
        "    },\n"
        "}, indent=2))\n"
        "report_bundle_dir = Path('/tmp/bedlam_global_replay_report_bundle')\n"
        "shutil.rmtree(report_bundle_dir, ignore_errors=True)\n"
        "report_bundle_dir.mkdir(parents=True)\n"
        "for path in (TRAINING_REPORT, TRAINING_HISTORY, TEST_REPORT):\n"
        "    shutil.copy2(path, report_bundle_dir / path.name)\n"
        "bundle = Path(shutil.make_archive(\n"
        "    '/kaggle/working/bedlam_global_replay_results', 'zip', root_dir=report_bundle_dir\n"
        "))\n"
        "print(f'Download this one file: {bundle}')\n"
        "print('Small report bundle contents:', sorted(path.name for path in report_bundle_dir.iterdir()))\n"
        "shutil.rmtree(SCRATCH_DIR, ignore_errors=True)\n"
    )
    notebook["cells"][10]["source"] = lines(
        "## Return these results\n\n"
        "Download `bedlam_global_replay_results.zip`. The minimum review files are "
        "`bedlam_global_replay_training_report.json`, `bedlam_global_replay_history.csv`, and "
        "`bedlam_global_replay_final_3dpw.json`. Keep the selected PTH in the saved Kaggle output "
        "for Core ML export if it beats the previous tiny checkpoint.\n"
    )

    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print({name: digest for name, (digest, _) in payloads.items()})


if __name__ == "__main__":
    main()
