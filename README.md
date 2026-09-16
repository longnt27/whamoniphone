# WHAM on iPhone

This repository runs a validated, image-conditioned WHAM body-motion pipeline
one frame at a time on iPhone. The recurrent and world-refiner states are
explicit, so Core ML does not unroll an entire video and exhaust device memory.

The selected system is:

`YOLO26m-pose → HMR2.0-S → residual token adapter → WHAM init/step → light causal smoothing → SMPL/world refiner`

It is the result of a validation-only six-candidate grid and a separately
locked smoothing ablation—not the earlier FastViT fallback. The complete
experiment narrative, metric definitions, and limitations are in the
[`final report`](docs/FINAL_REPORT.md).

## Measured result

On the untouched 3DPW test population of 11 single-person tracks and 11,349
poses:

| Pipeline | PA-MPJPE ↓ | MPJPE ↓ | PVE ↓ | Acceleration error ↓ |
| --- | ---: | ---: | ---: | ---: |
| Released WHAM, official stored inputs | 32.67 mm | 55.58 mm | 65.64 mm | 6.23 m/s² |
| Selected mobile pipeline, light smoothing | **51.80 mm** | **92.06 mm** | **107.20 mm** | **10.72 m/s²** |

YOLO detected a person in 99.52% of source frames. Light smoothing reduced the
raw mobile acceleration error by 14.43% for only 0.03 mm extra PA-MPJPE.

On an iPhone 11 Pro Max running iOS 18.6.2, the warmed fixed-image workload had
a median amortized latency of **187.13 ms/frame (5.34 fps)**. The seven compiled
models total 243.8 MB and measured peak resident memory was 381.6 MB. One
extreme YOLO stall dominated the mean, so the report uses the median for typical
latency and explicitly leaves tail reliability unresolved.

## Pipeline contract

1. `yolo26m-pose` supplies one person's COCO keypoints and person crop.
2. `HMR2SFrontend` supplies pose, shape, and a 1024-D image token.
3. `HMR2SSMPLInit` supplies real first-frame 3D joints to `WHAM_I`; there is no
   neutral/zero initializer fallback.
4. `HMR2STokenAdapter` maps the HMR2-S token to released WHAM's HMR2a token
   distribution.
5. `WHAM_ImageStep` advances one frame of recurrent state.
6. `TemporalOutputSmoother` applies geodesic pose smoothing (`alpha = 0.75`)
   and shape EMA (`alpha = 0.35`) without feeding smoothed pose back into WHAM.
7. `WHAM_WorldStep` advances contact-aware trajectory state and emits
   world-space joints plus all 6,890 SMPL vertices.

The offline video path uses recorded device gyro as an approximate camera
rotation signal. The 3DPW accuracy and fixed-image device tests use zero camera
angular velocity because neither input contains a synchronized phone gyro.

## Build and run

Requirements:

- Xcode 26 or a compatible Xcode capable of opening the checked-in project
- iOS 18.5 or newer
- a physical iPhone for meaningful Core ML/thermal measurements
- the seven generated Core ML packages listed below
- a licensed HMR/HMR2 checkpoint from which to extract SMPL face topology

Place these directories in `WhamApp/WhamApp`:

- `yolo26m-pose.mlpackage`
- `HMR2SFrontend.mlpackage`
- `HMR2SSMPLInit.mlpackage`
- `HMR2STokenAdapter.mlpackage`
- `WHAM_I.mlpackage`
- `WHAM_ImageStep.mlpackage`
- `WHAM_WorldStep.mlpackage`

Generate the fixed SMPL topology from your licensed checkpoint. The resulting
82,680-byte resource is intentionally ignored by Git:

```bash
python3 tools/export/export_smpl_topology.py \
  --checkpoint /path/to/licensed_hmr_checkpoint.ckpt \
  --output WhamApp/WhamApp/SMPLFaces.bin
```

The packages are intentionally ignored because they are large and recoverable;
the exporter scripts and conversion reports are tracked. Open
`WhamApp/WhamApp.xcodeproj`, choose your signing team and physical device, then
run a Release build. The equivalent command is:

```bash
xcodebuild -project WhamApp/WhamApp.xcodeproj \
  -scheme WhamApp -configuration Release \
  -destination 'id=YOUR_DEVICE_UDID' build
```

In the app:

- the normal offline flow selects a video and its recorded gyro JSON, writes a
  lightweight JSON result plus a Float16 `.whammesh` sidecar, and opens the
  shaded blue body in either **Mesh** or **Skeleton** mode; and
- the **Benchmark** tab runs the deterministic 5 × 8-frame workload and exports
  `selected_mobile_pipeline_device_benchmark.json`.

Both **Image connected** and **Smoothing active** must report `PASS`. Run on a
cool phone with Low Power Mode off and keep the app foregrounded.

## Reproduce the experiments

Rebuild the portable Kaggle notebooks:

```bash
python3 tools/kaggle/build_hmr2s_yolo26_grid_notebook.py
python3 tools/kaggle/build_hmr2s_smoothing_notebook.py
```

The notebook input list, execution order, and licensed-SMPL boundary are
documented in [`tools/kaggle`](tools/kaggle). The grid notebook performs model
selection on 3DPW validation, locks YOLO26m plus the residual adapter, and only
then evaluates 3DPW test. The smoothing notebook repeats that
validation-then-test discipline for the output filter.

Validate the checked-in evidence, hashes, selection identity, and report:

```bash
python3 tools/evaluation/validate_final_evidence.py
```

Core ML conversion entry points are in [`tools/export`](tools/export). They
convert pinned models and record numerical agreement; they do not retrain the
selected network.

## Repository map

- [`WhamApp`](WhamApp) — SwiftUI app, shared mobile pipeline, tests, and fixed
  benchmark frames
- [`tools/export`](tools/export) — selected PyTorch-to-Core-ML exporters
- [`tools/evaluation`](tools/evaluation) — selected evaluation modules and
  evidence validator
- [`tools/kaggle`](tools/kaggle) — final notebook builders and generated
  notebooks
- [`evaluation/results/selected`](evaluation/results/selected) — immutable raw
  reports plus SHA-256 manifest
- [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md) — full scientific and device
  report
- [`archive/experiments`](archive/experiments) — superseded FastViT, BEDLAM,
  and early tiny-pipeline work retained for auditability

## Current limitations and follow-up

- The app follows one highest-confidence person; it has no multi-person identity
  tracker.
- Gyro rotation is not a drop-in accuracy equivalent to DPVO, and global
  trajectory accuracy has not yet been validated against synchronized ground
  truth.
- The five-pass phone workload is enough for a smoke-test median, not a stable
  tail-latency claim.
- The filled SceneKit viewer renders the full 6,890-vertex SMPL body, but the
  existing phone benchmark measures inference rather than rendering cost.
- A `.whammesh` cache costs about 41 KB per frame, or 74 MB per minute at
  30 fps. Legacy results still open, but must be analyzed once more to add the
  camera/crop metadata required by the aligned video overlay.
- The video tab uses HMR2-S's estimated crop camera rather than measured phone
  intrinsics. It aligns the mesh through the same projection used by HMR2, but
  residual HMR/detector errors can still produce visible registration error.

## Attribution

WHAM is by Soyong Shin, Juyong Kim, Eni Halilaj, and Michael J. Black (CVPR
2024). See the [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Shin_WHAM_Reconstructing_World-grounded_Humans_with_Accurate_3D_Motion_CVPR_2024_paper.html)
and [official implementation](https://github.com/yohanshin/WHAM). HMR2.0-S is
from the released
[`TruncHierVFM`](https://github.com/nttcom/TruncHierVFM) implementation. Respect
the original repositories' licenses and the SMPL/3DPW dataset terms.
