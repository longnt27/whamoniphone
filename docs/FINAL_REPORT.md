# WHAM on iPhone: final evaluation report

## Result in one paragraph

The final app is a real, image-conditioned, frame-at-a-time WHAM pipeline:
**YOLO26m-pose → released HMR2.0-S → a validation-selected residual token
adapter → released WHAM init/recurrent/world steps → light output-only causal
smoothing**. On the untouched 3DPW test population it reaches 51.80 mm
PA-MPJPE, 92.06 mm MPJPE, 107.20 mm PVE, and 10.72 m/s² acceleration error.
That is materially worse than released WHAM with official stored inputs, but it
is no longer a disconnected fallback. On an iPhone 11 Pro Max, the typical
measured workload is 187.13 ms per source frame, or about 5.34 fps, after
warm-up. The complete compiled model set is 243.8 MB and the measured peak
resident memory is 381.6 MB. This is viable for offline or low-rate mobile
processing; it is not a 30 fps result.

All final numbers below trace to the immutable files in
[`evaluation/results/selected`](../evaluation/results/selected). Their hashes
and cross-report identities are checked by
[`validate_final_evidence.py`](../tools/evaluation/validate_final_evidence.py).

## What was actually built

The original [WHAM architecture](https://github.com/yohanshin/WHAM) is more
than one recurrent network. Its custom-video path uses a person detector,
ViTPose keypoints, HMR2 image features and first-frame SMPL initialization,
DPVO camera motion, the temporal WHAM network, SMPL forward kinematics, and a
contact-aware world trajectory refiner. A phone implementation therefore needs
credible replacements at every input boundary, not merely a small recurrent
core.

The selected mobile pipeline is:

1. **YOLO26m-pose** finds the highest-confidence person and supplies the 17
   COCO keypoints. Its 640-pixel aspect-preserving letterbox and inverse mapping
   are shared by the evaluator and Swift implementation.
2. **HMR2.0-S** processes a 256 × 256 square person crop. On frame zero,
   `HMR2SSMPLInit` converts its pose and shape into real first-frame 3D joints
   for `WHAM_I`. This replaces the old neutral-pose/zero-joint fallback.
3. **HMR2STokenAdapter** maps the released HMR2-S 1024-D token into the HMR2a
   token space that released WHAM learned to consume. HMR2-S and WHAM remain
   frozen in the selected candidate.
4. **WHAM_I** runs once. **WHAM_ImageStep** then consumes one frame and explicit
   recurrent state at a time, avoiding a Core ML graph that unrolls an entire
   clip and exhausts phone memory.
5. **Light smoothing** applies geodesic exponential smoothing to pose
   (`alpha = 0.75`) and an EMA to shape (`alpha = 0.35`). It changes only the
   decoded output; raw WHAM pose remains the next recurrent input.
6. **WHAM_WorldStep** performs SMPL forward and contact-aware trajectory
   refinement with explicit one-frame state. It emits world-space joints and
   all 6,890 SMPL vertices.

For recorded phone video, CoreMotion angular velocity is converted into WHAM's
six-value relative-rotation input. This is an engineering substitute for DPVO,
not a claim that a gyroscope reconstructs visual-odometry translation.

## How the project reached this architecture

### 1. The apparent image pipeline was a fallback

The first audit found that FastViT features were computed but did not reach the
WHAM motion decoder. The exported recurrent model lacked the corresponding
image-token and validity-mask inputs; `WHAM_I` received zero 3D joints and a
fallback pose; keypoints and crops did not consistently use WHAM geometry.
Visually plausible output therefore could not establish that the proposed
network was running.

The first useful correction was structural: expose a genuine image feature in
the one-frame recurrent model, make hidden state explicit, use confidence masks
and crop-relative keypoints, and add an A/B guard proving that token ablation
changes pose output.

### 2. FastViT showed why feature imitation was not enough

The original FastViT choice was an arbitrary mobile-backbone substitution. The
distillation experiments exposed two separate problems. First, a global pooled
student did not preserve the spatial/body information expected by HMR2 and
WHAM. Second, changing an encoder, adapting downstream decoding to its error,
and then moving the encoder back toward the original teacher changes the
interface twice. A decoder trained for yesterday's latent errors is not
automatically correct for today's latent distribution.

COCO, 3DPW, and bounded BEDLAM experiments improved individual proxy metrics
without recovering acceptable end-to-end geometry. Those runs remain in
[`archive/experiments`](../archive/experiments) because they explain the design
decision, but none is presented as the deployable result.

### 3. The reset: a released small HMR plus one controlled interface repair

Rather than continue training a hand-built vision replacement, the project
adopted released HMR2.0-S as the complete small HMR frontend. The remaining
scientific question became narrow: HMR2-S produces a useful 1024-D token, but
released WHAM was trained on HMR2a's token distribution. Two independent
repairs were compared:

- freeze both released models and learn a residual token adapter; or
- keep HMR2-S native and tune an independent copy of WHAM to read its token.

Each repair was trained independently for YOLO26n-, YOLO26s-, and YOLO26m-pose,
making six candidates with fixed data budget, splits, seed, crop policy,
losses, epochs, batch sizes, and thresholds. The selection report locked the
lowest validation PA-MPJPE, with MPJPE, PVE, acceleration, detector key, and
candidate name as deterministic tie-breakers.

The winner was **YOLO26m-pose plus the residual adapter and released WHAM**:
45.89 mm validation PA-MPJPE, 67.91 mm MPJPE, 81.29 mm PVE, and 11.52 m/s²
acceleration error. The 3DPW test file was not loaded by training and was opened
only after that choice was locked. See the
[`validation grid`](../evaluation/results/selected/yolo26_grid_validation.json)
and [`selected artifact`](../evaluation/results/selected/selected_deployment_artifact.json).

## Accuracy experiment

### Protocol

The locked comparison uses all 11 matching single-person 3DPW test tracks with
a valid initializer detection: 11,360 source frames and 11,349 recurrent poses.
YOLO found a person in 11,306 frames, a 99.52% detection rate. Missing detections
remain in the temporal sequence through the keypoint and image-feature validity
masks rather than silently deleting time.

The released-WHAM reference uses its official stored ViTPose/HMR2a inputs and
flip averaging. The mobile row uses YOLO26m-pose, released HMR2-S pose/shape
initialization and image token, the selected adapter, and the same released WHAM
checkpoint. Both rows use the same eligible population. Because 3DPW does not
provide the phone gyroscope stream used by the app, camera angular velocity is
zero; these are **camera-relative body accuracy** results, not a validated
world-trajectory score.

Kaggle evaluated the pinned PyTorch checkpoints with licensed SMPL assets. The
Core ML exporters separately checked numerical agreement for HMR2-S, the
adapter, and the WHAM world step, while the physical-phone test executed the
compiled packages. This separates dataset accuracy, conversion fidelity, and
device workload instead of pretending one small benchmark measures all three.

### What the metrics mean

- **PA-MPJPE** is average 3D joint error after rigidly aligning prediction and
  ground truth. It emphasizes pose articulation while discounting global
  placement and scale.
- **MPJPE** is average 3D joint error after the evaluator's root convention but
  without the full Procrustes alignment. A 92 mm result literally means about
  9.2 cm average joint-position error under that protocol; it is not “every
  joint is always exactly 9.2 cm wrong.”
- **PVE** is average error over the reconstructed SMPL surface vertices. It
  measures body surface/shape as well as the sparse skeleton.
- **Acceleration error** compares second-order joint motion at 30 fps, in m/s².
  Lower values generally mean less temporal jitter, although smoothing can
  trade responsiveness for stability.

### Locked 3DPW test results

| Pipeline | PA-MPJPE ↓ | MPJPE ↓ | PVE ↓ | Acceleration error ↓ |
| --- | ---: | ---: | ---: | ---: |
| Released WHAM, official stored inputs | 32.67 mm | 55.58 mm | 65.64 mm | 6.23 m/s² |
| Selected mobile frontend, raw WHAM output | 51.77 mm | 91.91 mm | 107.03 mm | 12.53 m/s² |
| Selected mobile frontend, light smoothing | **51.80 mm** | **92.06 mm** | **107.20 mm** | **10.72 m/s²** |

Relative to released WHAM, the final smoothed mobile path adds 19.13 mm
PA-MPJPE (+58.6%), 36.49 mm MPJPE (+65.7%), 41.57 mm PVE (+63.3%), and
4.50 m/s² acceleration error (+72.2%). That is a meaningful accuracy cost, not
“almost the same,” but it is also far better than the earlier disconnected or
naïvely substituted tiny pipelines. Full aggregate distributions and per-track
rows are in the
[`final 3DPW JSON`](../evaluation/results/selected/yolo26_grid_final_3dpw.json)
and [`CSV`](../evaluation/results/selected/yolo26_grid_final_3dpw.csv).

### Was smoothing worth it?

Raw, light, medium, and strong causal filters were selected on eight validation
tracks with a predeclared gate: no spatial validation metric could regress more
than 5%. Light smoothing won the normalized validation score. On locked test it
reduced acceleration error from 12.53 to 10.72 m/s², a **14.43% improvement**,
while adding only 0.03 mm PA-MPJPE, 0.16 mm MPJPE, and 0.17 mm PVE. The result
supports this modest filter, not arbitrary extra smoothing. The complete
ablation is in the
[`smoothing report`](../evaluation/results/selected/hmr2s_temporal_smoothing_3dpw.json).

## Core ML conversion

All seven app models are explicit in `MobileWhamModelCatalog`. The selected
checkpoint identities are pinned in code and reports. Conversion was not
treated as training: each exporter recorded source/checkpoint provenance and a
PyTorch-versus-Core-ML numerical check.

The HMR2-S report's largest recorded relative output error is 1.77% for shape
betas; image-token error is 0.45%, pose-6D error is 0.39%, and initial-joint
error is 0.15%. The selected adapter's relative maximum error is 0.23%. The
world step's largest recorded relative maximum error is 0.46% in recurrent
refiner state and 0.33% in vertices. These bounded differences do not explain
the much larger end-to-end gap to released WHAM; that gap already exists in the
mobile frontend/interface experiment before Core ML deployment.

See the [HMR2-S](../evaluation/results/selected/hmr2s_coreml_export_report.json),
[adapter](../evaluation/results/selected/hmr2s_token_adapter_export_report.json),
and [world-step](../evaluation/results/selected/wham_world_step_export_report.json)
conversion reports.

## Physical iPhone workload and latency

### Protocol

The Release app ran on an **iPhone 11 Pro Max** (`iPhone12,5`) with iOS 18.6.2.
It loaded the seven selected Core ML packages, ran one unmeasured full-track
warm-up, then processed the same eight real images five times: **5 × 8 = 40
source frames**. All 40 had detections. Both runtime guards passed:

- ablating the adapted image token changed WHAM pose by up to 0.1074; and
- output smoothing changed pose by up to 0.2441.

The fixed images provide a repeatable workload, not another accuracy dataset.
They contain no gyro stream, so their camera input is zero. Raw evidence is the
[`device benchmark JSON`](../evaluation/results/selected/selected_mobile_pipeline_device_benchmark.json).

### Typical latency

| Stage | Median | p95 | Samples |
| --- | ---: | ---: | ---: |
| YOLO26m pose + parse | 70.00 ms | 265.68 ms | 40 |
| Crop + keypoint normalization | 1.88 ms | 4.79 ms | 40 |
| HMR2-S frontend | 68.19 ms | 95.50 ms | 40 |
| Token adapter | 0.67 ms | 2.41 ms | 35 |
| First-frame HMR2-S SMPL init | 11.35 ms | 14.93 ms | 5 |
| WHAM init | 6.01 ms | 32.03 ms | 5 |
| WHAM recurrent step | 14.10 ms | 77.62 ms | 35 |
| Output smoothing | 0.14 ms | 0.82 ms | 35 |
| SMPL + world/refiner step | 7.15 ms | 24.89 ms | 35 |
| Eight-frame track | 1,497.07 ms | — | 5 |
| Amortized source frame | **187.13 ms** | — | 5 |

The median amortized rate is about **5.34 fps**. Median is the defensible
“normal run” summary for this particular capture, but the sample is too small
to claim a stable production tail distribution.

### The YOLO stall must not be hidden

One measured pass contained an extreme YOLO stall. Consequently, YOLO's
40-frame mean is 3,570.05 ms despite a 70.00 ms median, the five-pass mean
amortized frame latency is 3,683.50 ms, and its interpolated five-pass p95 is
14,181.91 ms. The stage timings identify the detector path as the source of the
outlier, but the report does not contain enough raw per-frame samples to assign
a cause. Thermal state moved from **nominal** before the test to **fair** after
it. The stall may involve thermal/resource scheduling or a one-off Core ML
event; it cannot be erased as “noise,” and it also cannot be generalized from
one event.

The appropriate conclusion is therefore two-part: typical warmed latency was
187.13 ms/frame, while tail reliability is unresolved and needs a longer,
cool-start/steady-state capture that retains individual frame samples.

### Loading, size, and memory

Cold model creation totaled **11.42 s** in this run. YOLO alone took 6.61 s,
HMR2-S frontend 2.18 s, and the split WHAM models about 2.31 s combined. Loading
is a startup cost and should be amortized by retaining models rather than
recreating them per frame.

| Compiled model | Bytes | Decimal MB |
| --- | ---: | ---: |
| YOLO26m-pose | 43,446,531 | 43.45 |
| HMR2-S frontend | 47,472,865 | 47.47 |
| HMR2-S first-frame SMPL init | 10,220,320 | 10.22 |
| Token adapter | 4,209,132 | 4.21 |
| WHAM init | 24,601,901 | 24.60 |
| WHAM image step | 94,803,254 | 94.80 |
| WHAM world step | 19,049,777 | 19.05 |
| **Total** | **243,803,780** | **243.80** |

Resident memory started at 163.3 MB, peaked at **381.6 MB** (363.9 MiB), and
ended at 373.9 MB. The benchmark intentionally retains every model at once so
it can time the complete fixed track. The normal offline analyzer uses a
two-phase strategy—extract observations, release vision models, then load the
recurrent/world models—so this single benchmark should not be misread as the
only possible memory schedule.

## Limitations

- The accuracy comparison is restricted to 11 single-person 3DPW tracks because
  the app selects the highest-confidence person and has no identity tracker.
- Released WHAM's row uses official stored inputs, while the mobile row measures
  a different live frontend. The gap is the complete practical substitution
  cost, not an isolated measure of YOLO or HMR2-S alone.
- 3DPW provides no synchronized phone gyro for this app, so no global trajectory
  accuracy claim is made. A synchronized iPhone/ground-truth capture or an
  appropriate licensed benchmark is required to validate the gyro path.
- The device workload has only five measured clips. Its median is useful; its
  tail percentiles are not yet statistically stable, and one YOLO stall
  dominates the mean.
- Model packages are generated and ignored rather than committed. Reproduction
  requires the pinned public checkpoints plus licensed SMPL assets under their
  original terms.
- `WHAM_WorldStep` already computes the 6,890-vertex SMPL body, but the current
  app viewer still renders the 17-joint skeleton. A filled mesh renderer is a
  follow-up presentation feature, not missing model inference.

## Conclusion and next experiment

The project achieved the narrow goal that the original prototype had not: the
iPhone executes the selected image-conditioned HMR/WHAM/SMPL pipeline with real
initialization, explicit causal state, measurable image-token influence, and a
locked accuracy-versus-latency report. The cost is roughly 5.34 fps typical
throughput and a 59–66% increase in the three spatial errors versus released
WHAM's official-input reference.

The next engineering task is the SMPL mesh viewer. It should reuse the vertices
already emitted by `WHAM_WorldStep`, bundle the fixed SMPL face topology, cache
frame results in a compact binary format rather than large JSON arrays, update
a SceneKit or Metal vertex buffer, and benchmark rendering separately from
model inference. The next scientific task is longer on-device latency sampling
and synchronized camera/gyro trajectory evaluation—not another round of
uncontrolled encoder retraining.
