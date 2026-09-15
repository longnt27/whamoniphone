# Full-pipeline tradeoff run

## What the Kaggle run measures

Run utils/full_pipeline_tradeoff_3dpw_kaggle.ipynb once on a GPU. It scores the
official and controlled variants on every registered 3DPW test person track:

1. released WHAM with its official flip average;
2. released WHAM without flip;
3. phase-three FastViT with official keypoints and initialization.

The app-like comparison then scores:

4. released WHAM with flip on every single-person test video; and
5. the proposed iPhone subset using YOLOv8n-pose, phase-three FastViT, and the
   neutral/identity initializer on those same videos.

The first two rows show the flip-evaluation effect. The second and third rows
isolate FastViT over the full official population. The fourth and fifth rows
give the actual same-population end-to-end accuracy tradeoff. The restriction
is necessary because the current app selects one highest-confidence person and
does not associate identities in multi-person video; scoring that detection
against both 3DPW identities would be invalid. Licensed SMPL is used after
inference only to calculate PA-MPJPE,
MPJPE, PVE, and acceleration. This evaluation-only decode is not added to the
iPhone runtime.

WHAM's official 3DPW evaluator sets camera angular velocity to zero. Therefore
this test is camera-coordinate body reconstruction, not a world-grounded
trajectory test. A world-grounded comparison would be a separate EMDB split-2
experiment.

## Required private SMPL-assets dataset

Create one private Kaggle dataset containing these three files (nested
folders are fine):

- SMPL_NEUTRAL.pkl
- SMPL_MALE.pkl
- SMPL_FEMALE.pkl

The three model files come from the user's licensed SMPL/SMPLify downloads.
Do not publish or mirror them. The notebook searches all mounted Kaggle inputs
and accepts the original long SMPL v1 filenames as aliases. WHAM's public
J_regressor_h36m.npy is downloaded automatically and verified by SHA-256.

The existing 3dpw-model dataset remains unchanged and remains the source of
imageFiles, sequenceFiles, and 3dpw_test_vit.pth. The saved phase-three notebook
must also be attached; its fastvit_hmr2_best.pth is selected by the pinned
SHA-256, not by notebook name.

## Outputs to keep

Download wham_full_pipeline_tradeoff_results.zip. It contains:

- full_pipeline_tradeoff_3dpw.json: aggregate metrics and provenance;
- full_pipeline_tradeoff_3dpw.csv: per-track metrics;
- FastViTNormalized_phase3_diagnostic.zip: the exact phase-three Core ML model;
- artifact_manifest.json: checkpoint identity and diagnostic-only status.

## One physical-iPhone run

After the Kaggle run:

1. Unzip FastViTNormalized_phase3_diagnostic.zip.
2. Replace the Xcode project's FastViTNormalized.mlpackage with the extracted
   package, preserving that exact resource name.
3. Select the physical iPhone and the Release build configuration.
4. Open the Benchmark tab and tap Run proposed pipeline once.
5. Let the app perform its untimed warm-up and one timed pass over the eight
   bundled WHAM example frames. Do not rerun it for this requested smoke test.
6. Export the benchmark JSON with the share button.

The JSON records device and iOS version, thermal state, detections, model-load
time, YOLO time, crop/normalization time, FastViT time, WHAM initializer and
recurrent-step time, total eight-frame time, amortized time per source frame,
resident memory, selected model bytes, and the image-feature connection guard.

This single phone pass is a latency smoke test, not a statistically stable
benchmark. The final tradeoff should present its measured value as one device
observation and use the full 3DPW run for accuracy.

## Measured accuracy result

The completed run is preserved in
`evaluation/results/full_pipeline_tradeoff_3dpw.json` and its per-track CSV.
The released checkpoint closely reproduced the paper: 35.698 mm PA-MPJPE,
57.614 mm MPJPE, 68.638 mm PVE, and 6.512 m/s² acceleration. The strict
reproduction check is still recorded as failed because PVE was 1.238 mm above
the table value, exceeding the preregistered 1.0 mm tolerance by 0.238 mm.

On all 37 tracks, replacing HMR2 features with FastViT while keeping official
keypoints and initialization increased PA-MPJPE by 10.31 mm, MPJPE by 13.12 mm,
PVE by 16.79 mm, and acceleration by 1.17 m/s². On the same 11 single-person
tracks, the complete proposed subset increased PA-MPJPE by 39.52 mm, MPJPE by
103.76 mm, PVE by 115.82 mm, and acceleration by 9.50 m/s² relative to released
WHAM. Detection succeeded on 98.72% of source frames, so missed detections alone
do not explain the degradation.

The phase-three FastViT package remains useful for the requested latency
diagnostic, but these accuracy results reject it as a deployment replacement.
