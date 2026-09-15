"""Build one Kaggle notebook for the YOLO26 n/s/m × HMR2-S/WHAM grid."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

import build_hmr2s_wham_adaptation_notebook as base

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "hmr2s_yolo26_grid_kaggle.ipynb"
SOURCES = (
    *base.SOURCES,
    base.EVALUATION_ROOT / "evaluate_selected_hmr2s_yolo_grid.py",
)


def set_source(cell: dict[str, object], source: str) -> None:
    cell["source"] = source.splitlines(True)


def embedded_source() -> str:
    lines: list[str] = []
    for path in SOURCES:
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        payload = base64.b64encode(gzip.compress(raw, compresslevel=9)).decode("ascii")
        lines.append(f"    {path.name!r}: ({digest!r}, {payload!r}),")
    return (
        "# Materialize the reviewed, checksum-verified experiment sources.\n"
        "import base64, gzip\n\n"
        "SCRATCH_DIR.mkdir(parents=True, exist_ok=True)\n"
        "embedded = {\n" + "\n".join(lines) + "\n}\n"
        "for name, (expected, payload) in embedded.items():\n"
        "    contents = gzip.decompress(base64.b64decode(payload))\n"
        "    actual = hashlib.sha256(contents).hexdigest()\n"
        "    if actual != expected:\n"
        "        raise RuntimeError(f'Embedded source checksum mismatch: {name}')\n"
        "    (SCRATCH_DIR / name).write_bytes(contents)\n"
        "print({name: digest for name, (digest, _) in embedded.items()})\n"
    )


def main() -> None:
    base.main()
    notebook = json.loads(base.OUTPUT.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    set_source(
        cells[0],
        """# YOLO26 n/s/m × HMR2-S/WHAM adaptation grid

This is one controlled model-selection experiment with **six candidates**:

- YOLO26n-pose, YOLO26s-pose, and YOLO26m-pose;
- for each detector, independently train the 1024-D token adapter and the native-token WHAM fine-tune from the same released weights.

All six candidates use the same COCO, 3DPW-train, BEDLAM budget, split, seed,
losses, epochs, and selection metric. The notebook chooses the global winner using
3DPW validation only. A separate process then opens 3DPW test for the first time
and evaluates the locked YOLO/candidate combination. This prevents test-set model
selection. The released-WHAM reference and all phone variants in the final report
use the same eligible 3DPW-test population.

## Attach these inputs

1. `3dpw-model`: raw `imageFiles`, `sequenceFiles`, and `3dpw_test_vit.pth`.
2. `3dpw-vit`: `3dpw_train_vit.pth` and `3dpw_val_vit.pth`.
3. `bedlam`: extracted `bedlam-labels/*.npz`.
4. COCO 2017: `awsaf49/coco-2017-dataset`.
5. `distill-fastvit-hmr2-kagglef9b9f724ae`: saved output containing the verified official `hmr2a.ckpt`.
6. A small private dataset containing official `hmr_vit-small_d3-a4x16-m128.zip` or its verified `last.ckpt`.
7. Licensed `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl`, and `SMPL_FEMALE.pkl` (they may already be inside `3dpw-model`).

Enable Internet and a GPU, and expose `HF_TOKEN` as a Kaggle secret. BEDLAM
archives are cached once under `/tmp` and reused by n/s/m. Only reports and six
validation-selected checkpoints are saved under `/kaggle/working`.
""",
    )
    config = "".join(cells[1]["source"])
    config = config.replace(
        "Path('/tmp/hmr2s_wham_adaptation')", "Path('/tmp/hmr2s_yolo26_grid')"
    ).replace(
        "Path('/kaggle/working/hmr2s_wham_adaptation')",
        "Path('/kaggle/working/hmr2s_yolo26_grid')",
    )
    set_source(cells[1], config)
    set_source(cells[3], embedded_source())

    downloads = "".join(cells[4]["source"])
    needle = (
        "    'yolo26n-pose.pt': (\n"
        "        'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt',\n"
        "        'eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9'),\n"
    )
    addition = needle + (
        "    'yolo26s-pose.pt': (\n"
        "        'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt',\n"
        "        'a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646'),\n"
        "    'yolo26m-pose.pt': (\n"
        "        'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-pose.pt',\n"
        "        '2fbf16367022256a226035695c5c389384c6706e8bb8ab8fcd0e7976f05443c4'),\n"
    )
    if downloads.count(needle) != 1:
        raise RuntimeError("Could not locate the YOLO26n download stanza")
    downloads = downloads.replace(needle, addition)
    needle = "YOLO26_WEIGHTS = downloaded['yolo26n-pose.pt']\n"
    addition = (
        "YOLO_GRID = {\n"
        "    'n': {'variant': 'yolo26n-pose', 'weights': downloaded['yolo26n-pose.pt'], 'sha256': 'eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9'},\n"
        "    's': {'variant': 'yolo26s-pose', 'weights': downloaded['yolo26s-pose.pt'], 'sha256': 'a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646'},\n"
        "    'm': {'variant': 'yolo26m-pose', 'weights': downloaded['yolo26m-pose.pt'], 'sha256': '2fbf16367022256a226035695c5c389384c6706e8bb8ab8fcd0e7976f05443c4'},\n"
        "}\n"
        "YOLO26_WEIGHTS = YOLO_GRID['n']['weights']\n"
    )
    if downloads.count(needle) != 1:
        raise RuntimeError("Could not locate the YOLO assignment")
    set_source(cells[4], downloads.replace(needle, addition))

    set_source(
        cells[6],
        """# Fast fail before the long run: code self-test and complete data/HF preflight.
import sys

TRAINER = SCRATCH_DIR / 'train_hmr2s_wham_adaptation.py'
FINAL_EVALUATOR = SCRATCH_DIR / 'evaluate_selected_hmr2s_yolo_grid.py'
base_common = [
    '--coco-root', str(COCO_ROOT),
    '--bedlam-label-root', str(BEDLAM_LABEL_ROOT),
    '--three-dpw-root', str(THREEDPW_ROOT),
    '--sequence-root', str(THREEDPW_ROOT),
    '--train-parsed', str(TRAIN_PARSED),
    '--val-parsed', str(VAL_PARSED),
    '--test-parsed', str(TEST_PARSED),
    '--wham-repo', str(WHAM_REPO),
    '--wham-checkpoint', str(WHAM_CHECKPOINT),
    '--hmr2a-checkpoint', str(HMR2A_CHECKPOINT),
    '--hmr2s-repo', str(HMR2S_REPO),
    '--hmr2s-checkpoint', str(HMR2S_CHECKPOINT),
    '--smpl-model-directory', str(SMPL_MODEL_DIR),
    '--h36m-joint-regressor', str(H36M_REGRESSOR),
    '--wham-joint-regressor', str(WHAM_REGRESSOR),
    '--hf-cache-dir', str(SCRATCH_DIR / 'shared_hf_cache'),
    '--coco-train-people', str(COCO_TRAIN_PEOPLE),
    '--coco-val-people', str(COCO_VAL_PEOPLE),
    '--maximum-bedlam-scenes', str(MAXIMUM_BEDLAM_SCENES),
    '--maximum-bedlam-download-gib', str(MAXIMUM_BEDLAM_DOWNLOAD_GIB),
    '--bedlam-videos-per-scene', str(BEDLAM_VIDEOS_PER_SCENE),
    '--bedlam-frames-per-video', str(BEDLAM_FRAMES_PER_VIDEO),
    '--validation-tracks', str(VALIDATION_TRACKS),
    '--validation-frames', str(VALIDATION_FRAMES),
    '--adapter-epochs', str(ADAPTER_EPOCHS),
    '--integration-epochs', str(INTEGRATION_EPOCHS),
    '--decoder-epochs', str(DECODER_EPOCHS),
    '--pose-batch-size', str(POSE_BATCH_SIZE),
    '--teacher-batch-size', str(TEACHER_BATCH_SIZE),
    '--hmr-batch-size', str(HMR2S_BATCH_SIZE),
    '--smpl-batch-size', str(SMPL_BATCH_SIZE),
    '--workers', str(WORKERS),
]

def variant_args(key):
    config = YOLO_GRID[key]
    return [
        '--yolo26-weights', str(config['weights']),
        '--yolo-variant', config['variant'],
        '--expected-yolo-sha256', config['sha256'],
        '--output-dir', str(OUTPUT_DIR / key),
        '--scratch-dir', str(SCRATCH_DIR / f'runtime_{key}'),
    ]

environment = os.environ.copy()
environment['PYTHONPATH'] = str(SCRATCH_DIR) + os.pathsep + environment.get('PYTHONPATH', '')
subprocess.run([sys.executable, '-u', str(TRAINER), '--self-test'], check=True, env=environment)
subprocess.run(
    [sys.executable, '-u', str(TRAINER), '--inspect-data', *base_common, *variant_args('n')],
    check=True,
    env=environment,
)
""",
    )
    set_source(
        cells[7],
        """# Long cell: train six candidates, lock globally on validation, then open test once.
import csv, json

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
variant_reports = {}
for ordinal, key in enumerate(('n', 's', 'm'), start=1):
    print(f'[{ordinal}/3] Training adapter and independent WHAM fine-tune for {YOLO_GRID[key]["variant"]}...', flush=True)
    subprocess.run(
        [sys.executable, '-u', str(TRAINER), '--skip-final-test', *base_common, *variant_args(key)],
        check=True,
        env=environment,
    )
    report_path = OUTPUT_DIR / key / 'hmr2s_wham_training_report.json'
    variant_reports[key] = json.loads(report_path.read_text())

candidate_rows = []
for key in ('n', 's', 'm'):
    report = variant_reports[key]
    for report_key, candidate in (
        ('candidate_2_adapter', 'adapter_hmr2s_released_wham'),
        ('candidate_3_native_wham_tuning', 'native_hmr2s_tuned_wham'),
    ):
        summary = report[report_key]
        candidate_rows.append({
            'yolo_key': key,
            'yolo_variant': YOLO_GRID[key]['variant'],
            'candidate': candidate,
            'within_yolo_score_vs_naive': float(summary['score_vs_naive']),
            'pa_mpjpe_mm': float(summary['selected']['pa_mpjpe_mm']),
            'mpjpe_mm': float(summary['selected']['mpjpe_mm']),
            'pve_mm': float(summary['selected']['pve_mm']),
            'accel_official_30fps': float(summary['selected']['accel_official_30fps']),
            'selected_epoch': int(summary['selected_epoch']),
            'selected_stage': summary.get('selected_stage', 'token_adapter'),
        })

locked = min(candidate_rows, key=lambda row: (
    row['pa_mpjpe_mm'], row['mpjpe_mm'], row['pve_mm'],
    row['accel_official_30fps'], row['yolo_key'], row['candidate'],
))
grid_report = {
    'schema_version': 1,
    'experiment': 'yolo26_pose_size_by_hmr2s_wham_adaptation_grid',
    'protocol': {
        'grid': 'YOLO26n/s/m-pose x token adapter/native-token WHAM fine-tune',
        'candidate_count': 6,
        'selection_split': '3DPW validation only',
        'test_status_at_selection': 'not loaded by any training process',
        'selection_rule': 'lowest absolute PA-MPJPE; deterministic tie-breaks are MPJPE, PVE, acceleration, YOLO key, candidate name',
        'fixed_across_grid': 'data budget, splits, seed, crop policy, losses, epochs, batch sizes, confidence and IoU thresholds',
    },
    'candidates': candidate_rows,
    'locked_winner': locked,
}
GRID_JSON = OUTPUT_DIR / 'yolo26_grid_validation.json'
GRID_CSV = OUTPUT_DIR / 'yolo26_grid_validation.csv'
GRID_JSON.write_text(json.dumps(grid_report, indent=2) + '\\n')
with GRID_CSV.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(candidate_rows[0]))
    writer.writeheader()
    writer.writerows(candidate_rows)
print('LOCKED ON VALIDATION:', json.dumps(locked, indent=2), flush=True)

winner_key = locked['yolo_key']
winner_config = YOLO_GRID[winner_key]
winner_dir = OUTPUT_DIR / winner_key
FINAL_JSON = OUTPUT_DIR / 'yolo26_grid_final_3dpw.json'
FINAL_CSV = OUTPUT_DIR / 'yolo26_grid_final_3dpw.csv'
final_args = [
    '--test-parsed', str(TEST_PARSED),
    '--three-dpw-root', str(THREEDPW_ROOT),
    '--wham-repo', str(WHAM_REPO),
    '--wham-checkpoint', str(WHAM_CHECKPOINT),
    '--hmr2s-repo', str(HMR2S_REPO),
    '--hmr2s-checkpoint', str(HMR2S_CHECKPOINT),
    '--yolo26-weights', str(winner_config['weights']),
    '--yolo-variant', winner_config['variant'],
    '--expected-yolo-sha256', winner_config['sha256'],
    '--adapter-checkpoint', str(winner_dir / 'hmr2s_to_hmr2a_adapter_best.pth'),
    '--tuned-wham-checkpoint', str(winner_dir / 'hmr2s_native_wham_finetuned_best.pth'),
    '--selected-candidate', locked['candidate'],
    '--grid-validation-report', str(GRID_JSON),
    '--smpl-model-directory', str(SMPL_MODEL_DIR),
    '--h36m-joint-regressor', str(H36M_REGRESSOR),
    '--wham-joint-regressor', str(WHAM_REGRESSOR),
    '--output-json', str(FINAL_JSON),
    '--output-csv', str(FINAL_CSV),
    '--pose-batch-size', str(POSE_BATCH_SIZE),
    '--hmr-batch-size', str(HMR2S_BATCH_SIZE),
    '--smpl-batch-size', str(SMPL_BATCH_SIZE),
]
print('Validation lock complete. Opening 3DPW test exactly once...', flush=True)
subprocess.run([sys.executable, '-u', str(FINAL_EVALUATOR), *final_args], check=True, env=environment)
""",
    )
    set_source(
        cells[8],
        """# Show the locked result and create a reports-only download.
import json, zipfile

grid_report = json.loads((OUTPUT_DIR / 'yolo26_grid_validation.json').read_text())
final_report = json.loads((OUTPUT_DIR / 'yolo26_grid_final_3dpw.json').read_text())
locked = grid_report['locked_winner']
print('Validation-locked winner:', json.dumps(locked, indent=2))
print('Final 3DPW-test metrics:')
for variant, result in final_report['variants'].items():
    metrics = {name: value['mean'] for name, value in result['metrics'].items()}
    print(variant, metrics)

winner_dir = OUTPUT_DIR / locked['yolo_key']
checkpoint_name = (
    'hmr2s_to_hmr2a_adapter_best.pth'
    if locked['candidate'] == 'adapter_hmr2s_released_wham'
    else 'hmr2s_native_wham_finetuned_best.pth'
)
selected_checkpoint = winner_dir / checkpoint_name
selection = {
    'yolo_variant': locked['yolo_variant'],
    'candidate': locked['candidate'],
    'checkpoint_relative_to_output': str(selected_checkpoint.relative_to(OUTPUT_DIR)),
    'checkpoint_bytes': selected_checkpoint.stat().st_size,
    'checkpoint_sha256': sha256_file(selected_checkpoint),
    'official_yolo_sha256': YOLO_GRID[locked['yolo_key']]['sha256'],
}
(OUTPUT_DIR / 'selected_deployment_artifact.json').write_text(json.dumps(selection, indent=2) + '\\n')

manifest = {}
for path in sorted(path for path in OUTPUT_DIR.rglob('*') if path.is_file()):
    manifest[str(path.relative_to(OUTPUT_DIR))] = {'bytes': path.stat().st_size, 'sha256': sha256_file(path)}
(OUTPUT_DIR / 'artifact_manifest.json').write_text(json.dumps(manifest, indent=2) + '\\n')
reports = [
    OUTPUT_DIR / 'yolo26_grid_validation.json',
    OUTPUT_DIR / 'yolo26_grid_validation.csv',
    OUTPUT_DIR / 'yolo26_grid_final_3dpw.json',
    OUTPUT_DIR / 'yolo26_grid_final_3dpw.csv',
    OUTPUT_DIR / 'selected_deployment_artifact.json',
    OUTPUT_DIR / 'artifact_manifest.json',
]
for key in ('n', 's', 'm'):
    reports.extend([
        OUTPUT_DIR / key / 'hmr2s_wham_training_report.json',
        OUTPUT_DIR / key / 'hmr2s_wham_training_history.csv',
    ])
bundle = Path('/kaggle/working/hmr2s_yolo26_grid_reports.zip')
with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for path in reports:
        archive.write(path, arcname=str(path.relative_to(OUTPUT_DIR)))
print('Download this reports bundle for review:', bundle)
print('Also save this notebook version: the selected checkpoint is', selected_checkpoint)
""",
    )

    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
