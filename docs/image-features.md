# FastViT -> WHAM image-feature integration

## What changes

The old app computed FastViT features without passing them to WHAM. The old
step export also bypassed the trained feature integrator. The new path is:

```text
keypoints -> encoder -> context -----> trajectory decoder
                    |        |
                    |        + features_step -> integrator -> motion decoder
                    |                                          |
                    + pred_kp3d                    pose / shape / camera / contact
```

`features_step` is float32 `[1, 1, 1024]`, from the SAME frame/person crop as the
keypoints. Pose/shape are SMPL parameters. `pred_kp3d` remains the earlier
17-joint encoder output, so the existing skeleton renderer is intentionally
unchanged. This PR does not add SMPL mesh reconstruction or claim better accuracy.
The JSON now retains `shape`, `timestamp_s` (when valid) and `has_image_features`.

The Swift reader stores one paired record per source sample, including missing
detections/crop failures. Fallback tensors are explicitly zero-filled. Crops are
clamped to image bounds and converted from YOLO's top-left coordinates to Core
Image coordinates. A failed crop no longer shifts all later features by a frame.

## Required migration: replace BOTH exported models

Source changes cannot add an input to an existing compiled model. Old packages
are rejected with an explicit error rather than silently running image-free.
The app checks the WHAM input schema and FastViT normalization metadata.

Use a Python environment with mutually compatible PyTorch, coremltools, timm,
NumPy, PyYAML, loguru and smplx versions. Upstream WHAM imports smplx even though
this exporter never instantiates a body model. Do not install the CUDA/DPVO demo
stack just for these exports. Native Core ML prediction needs macOS.

From this repository's root, with the upstream WHAM checkout and your existing
checkpoints available locally:

```bash
python -m unittest discover -s tests -v

python utils/wham_coreml.py \
  --wham-root /absolute/path/to/WHAM \
  --checkpoint /absolute/path/to/wham_vit_bedlam_w_3dpw.pth.tar \
  --out exports --verify-coreml

python utils/extractViT.py \
  --checkpoint /absolute/path/to/fastvit_student_1024.pth \
  --out exports/_FastViT.mlpackage
```

The WHAM exporter uses CPU, loads the ViT configuration, and checks every
non-SMPL checkpoint key strictly. Missing integrator weights are an error, not a
randomly initialized replacement. No SMPL body data is needed for these neural
network exports. No model files are included in this PR.

Checkpoints are loaded with `weights_only=True`. If an official checkpoint
contains extra pickle objects, inspect/trust its provenance before explicitly
adding `--trust-checkpoint`. Never use that option for an untrusted file.

Replace the app's `WHAM_I.mlpackage`, `WHAM_S.mlpackage` and FastViT package with
these outputs. Preserve the model names used by the existing generated Swift
classes (`WHAM_I`, `WHAM_S`, `_FastViT`), add them to the app target, and clean/build
in Xcode. Model packages and the Xcode project are not tracked in this repository.
The `MLDictionaryFeatureProvider` step call avoids relying on a stale generated
`WHAM_SInput` class, but it does NOT make an old model support image features.

`ExtractWHAM.ipynb` is now a thin entry point to the same exporter, rather than a
second, diverging copy of the model graph. Configure its local paths before use.

## FastViT preprocessing contract

The training notebook used RGB, resized to 256 x 256, then:

```python
normalized = (pixels / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
```

The old ImageType export did not perform this transformation. The new export
puts exact per-channel normalization IN the graph and declares identity RGB
image preprocessing at its boundary. Swift supplies the raw 256 x 256 crop;
do not normalize it a second time. No retraining is required merely to make this
transformation match training. Differences in crop/resize implementations still
need a real-image comparison on macOS/iPhone.

## Verification and limitations

The CPU tests use small synthetic LSTMs with upstream-compatible interfaces.
They check feature sensitivity, upstream residual-mask semantics (including
all-zero and partially zero features), all recurrent outputs/state feedback,
traced versus eager execution, strict checkpoint loading, and RGB normalization.
They are not trained-model accuracy tests and do not exercise Apple frameworks.

At export time, the actual loaded checkpoint is checked against an unmodified
upstream integrator on one-frame inputs. Pose AND shape must react to changed
features while encoder joints and trajectory outputs stay unchanged for the
same incoming state. `--verify-coreml` additionally runs 16 synthetic frames on
macOS CPU, feeding each backend its own states, and compares all 13 outputs.
The FP16 smoke tolerance is `rtol=atol=0.02`; this is a numerical alarm threshold,
not a task-accuracy requirement. Inspect `exports/export_report.json` and repeat
on realistic traces and the actual phone before relying on accuracy.

The static-shape export preserves upstream's unusual residual condition:
`(features != 0).all(-1).all(-1)`. Zero features do NOT bypass the integrator;
they still pass through its MLP, without the residual. Upstream's full-clip call
reduces that condition over time, whereas this one-frame export applies it per
step. A clip containing zeros can therefore differ from a full-clip upstream
run. This is explicitly a streaming, one-frame interface, not a claim of exact
full-WHAM sequence parity under missing image features.

Before merging/deploying, complete these local checks:

- Export both actual checkpoints; run `--verify-coreml` on the Mac.
- Compare a known RGB crop through Python FastViT and its Core ML package.
- Build the app in Xcode. Exercise missing detections, invalid/edge crops,
  portrait/landscape orientation, and rejection of an outdated package.
- Run a short iPhone clip; confirm frame alignment and finite pose/shape logs.

Still out of scope: matching upstream 2D bbox/keypoint normalization and masking,
first-frame initialization quality (zero 6D rotations are not valid rotations),
consistent person tracking, gyro/camera integration, SMPL rendering, and
 task-accuracy evaluation. Zero-filled fallbacks make behavior deterministic;
they do not solve these modeling issues.

## Sources

- Upstream graph: https://github.com/yohanshin/WHAM/blob/main/lib/models/wham.py
- Integrator and recurrent modules: https://github.com/yohanshin/WHAM/blob/main/lib/models/layers/modules.py
- Local training preprocessing: `utils/distillvit-2.ipynb`
- Core ML prediction: https://apple.github.io/coremltools/docs-guides/source/model-prediction.html

Respect the WHAM checkpoint and SMPL licenses when distributing models. This
change publishes code only, not pretrained checkpoints or body-model assets.
