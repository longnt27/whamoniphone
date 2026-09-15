# FastViT/HMR2 retraining handoff

Use the self-contained Kaggle notebook:

[`utils/distill_fastvit_hmr2_kaggle.ipynb`](../utils/distill_fastvit_hmr2_kaggle.ipynb)

The notebook embeds the reviewed training program and verifies its SHA-256
before executing it. The readable source is
[`utils/distill_fastvit_hmr2.py`](../utils/distill_fastvit_hmr2.py); regenerate
the notebook after editing that source with:

```bash
python utils/build_distill_kaggle_notebook.py
```

## Kaggle setup

1. Create a notebook and import `distill_fastvit_hmr2_kaggle.ipynb`.
2. In **Notebook options**, select a T4 or better GPU and enable Internet.
3. Attach a COCO 2017 dataset that contains `train2017`, `val2017`,
   `person_keypoints_train2017.json`, and `person_keypoints_val2017.json`.
   The notebook is configured for
   `/kaggle/input/datasets/awsaf49/coco-2017-dataset/coco2017`. If Kaggle mounts
   it at a different location, use the dataset panel's **Copy file path** action
   on the `coco2017` directory and update `COCO_ROOT`.
4. Do not point `COCO_ROOT` at all of `/kaggle/input` when both the full and a
   person-only COCO dataset are attached. The preflight cell deliberately
   refuses ambiguous annotation/image matches and prints the exact resolved
   train and validation inputs before any expensive teacher work begins.
5. Run all cells. If no COCO dataset is attached, `DOWNLOAD_COCO=True` uses the
   official archives, but needs roughly 40 GB of peak working storage. Attaching
   an already-extracted Kaggle dataset is strongly preferred.

The full COCO 2017 archive is intentional, but the training records are not
generic object examples: the script reads only the official `person_keypoints`
annotations and makes one WHAM-style person crop per usable annotation. A
"COCO person only" dataset is also sufficient only if it preserves both
official keypoint JSON files, the matching `train2017` and `val2017` images,
and their original filenames. An images-only person folder is not sufficient.

The default run uses 48,000 training people and 3,000 held-out validation
people. HMR2 token caching is the expensive first phase. Its `.npy` files and
progress metadata are resumable within the Kaggle working directory. If HMR2
runs out of GPU memory, reduce `TEACHER_BATCH_SIZE` from 8 to 4. If student
training runs out of memory, reduce `TRAIN_BATCH_SIZE` from 48 to 24.

## What is different from the rejected notebook

- COCO annotations produce one square person crop per example. Visible
  keypoints use the same 1.2 expansion as WHAM; the COCO bounding box with a
  1.05 expansion is the fallback.
- Crops outside the source frame are black-padded rather than clamped and
  distorted.
- Both teacher and student use HMR2's center `256x192` field of view.
- The student retains a `4x3` spatial grid before its 1024-D projection.
- FastViT remains frozen while the new head learns. Only its last stage is then
  fine-tuned, at a much smaller learning rate.
- Teacher targets are standardized per dimension during optimization. Raw
  cosine and the frozen HMR2 pose readout are auxiliary losses.
- COCO validation is completely separate from training. A constant mean-token
  baseline is reported so a deceptively high raw cosine cannot pass by itself.

## Acceptance contract

The output is accepted only when all default gates pass:

| Gate | Required |
| --- | ---: |
| Raw feature cosine mean | >= 0.95 |
| Centered/standardized cosine mean | >= 0.50 |
| Standardized feature RMSE | <= 0.85 |
| HMR2 pose-readout rotation error | <= 20 degrees |
| Pose error relative to mean-token baseline | < 75% |

These are engineering gates, not paper metrics. If any gate fails, the report
sets `accepted=false`, the training cell prints a rejection, and the Core ML
exporter refuses the checkpoint. The final cell still exposes the report and
diagnostic bundle. Do not weaken a gate merely to get an export;
inspect the validation curves and adjust the model or training instead.

## Downstream WHAM arbitration on 3DPW

The feature-space gate is deliberately conservative. Before spending more GPU
time on another student, run the isolated downstream A/B notebook:

[`utils/wham_feature_substitution_3dpw_kaggle.ipynb`](../utils/wham_feature_substitution_3dpw_kaggle.ipynb)

It feeds the official stored HMR2 feature and the FastViT feature for each
identical frame through the same frozen WHAM recurrent core. The checkpoint's
failed training marker is ignored only for this diagnostic; it is not silently
changed to accepted.

The notebook requires two attached Kaggle inputs:

1. The saved phase-two notebook version
   `distill-fastvit-hmr2-kaggle7f1f81131a`, containing
   `fastvit_hmr2_phase2/fastvit_hmr2_best.pth`. Do not attach the earlier
   `distill-fastvit-hmr2-kagglef9b9f724ae` run as the student source.
2. A private dataset made from a registered 3DPW download. It needs the raw
   `imageFiles` tree and WHAM's official public `3dpw_test_vit.pth`, available
   from the folder linked by the
   [WHAM dataset instructions](https://github.com/yohanshin/WHAM/blob/main/docs/DATASET.md).

A minimal useful layout is:

```text
3DPW/
├── imageFiles/
│   ├── downtown_arguing_00/
│   │   └── image_00000.jpg
│   └── ...
└── 3dpw_test_vit.pth
```

Set `THREEDPW_ROOT` in the configuration cell to that `3DPW` directory, enable
Internet and a Kaggle GPU, and run all cells. Large assets live only under
`/tmp`; `/kaggle/working/wham_feature_substitution` retains just the JSON report
and per-track CSV. The test is a rotation-space substitution diagnostic because
licensed SMPL assets are intentionally not bundled. It keeps official HMR2
initialization fixed, so it isolates the feature replacement rather than
claiming an end-to-end mobile result.

The phase-two checkpoint did not pass this test: over 10 person tracks and
3,000 frames it added 1.607 degrees (15.94%) of pose error and drifted 5.132
degrees from the teacher-feature WHAM result. The three limits are 1 degree,
10%, and 5 degrees respectively. It must not be exported as the production
image branch.

## Frozen-WHAM downstream fine-tuning

Use the separate training notebook:

[`utils/finetune_fastvit_wham_downstream_kaggle.ipynb`](../utils/finetune_fastvit_wham_downstream_kaggle.ipynb)

This run starts from the corrected phase-two checkpoint. It trains on 3DPW
train and selects on 3DPW validation while the entire WHAM recurrent network is
frozen. The objective retains token reconstruction losses but also backpropagates
teacher-versus-student WHAM pose and root-rotation losses into FastViT. The test
split is never loaded by this notebook.

The existing Kaggle inputs already provide both parts: `3dpw-model` contains
the nested `imageFiles/imageFiles` and `sequenceFiles/sequenceFiles` trees,
while `3dpw-vit` contains WHAM's `3dpw_train_vit.pth` and
`3dpw_val_vit.pth`. The notebook resolves those two mounts separately and does
not attempt a Google Drive download.

A minimal layout for this training run is:

```text
3dpw-vit/
├── 3dpw_train_vit.pth
└── 3dpw_val_vit.pth

3dpw-model/
├── imageFiles/
│   └── imageFiles/
│       └── ...
└── sequenceFiles/
    └── sequenceFiles/
        └── train/
            └── ...
```

The selected checkpoint deliberately remains marked
`awaiting_untouched_3dpw_test`, even if validation passes. Save the training
notebook as a new version, attach that saved output to the isolated substitution
notebook, and run the test once. The current substitution notebook automatically
finds exactly one attached `fastvit_hmr2_phase3/fastvit_hmr2_best.pth`; it refuses
an ambiguous or phase-two-only input. Only that result can clear the deployment
gate.

## Files to download

Download these three files from the final cell:

- `fastvit_hmr2_best.pth` — versioned student checkpoint with its validation
  decision and target normalization buffers;
- `fastvit_hmr2_history.csv` — epoch metrics;
- `fastvit_hmr2_training_report.json` — thresholds, gates, hashes, and data
  provenance.

The default output excludes COCO, the 2.5 GB HMR2 teacher, cached tokens, the
redundant last-epoch checkpoint, and a duplicate zip. Pass
`--keep-last-checkpoint` or `--make-bundle` only when one of those optional
artifacts is specifically needed.

## Mac validation and Core ML export

Only export a checkpoint whose report says `accepted=true`:

```bash
python utils/export_fastvit_normalized.py \
  --weights /path/to/fastvit_hmr2_best.pth \
  --output WhamApp/WhamApp/FastViTNormalized.mlpackage
```

The exporter checks the v2 architecture and embedded acceptance marker before
conversion. It also performs PyTorch/Core ML parity and writes the acceptance
metrics into the Core ML package; the iPhone Benchmark tab reads that metadata
to report feature fidelity separately from simple image-branch connectivity.

Then rerun the direct teacher check on the bundled real frames:

```bash
python utils/export_fastvit_initializer.py \
  --hmr2-checkpoint /path/to/hmr2a.ckpt \
  --output /tmp/FastViTInitialPose.mlpackage

python utils/evaluate_fastvit_distillation.py \
  --wham-repo /path/to/WHAM \
  --hmr2-checkpoint /path/to/hmr2a.ckpt \
  --pose-head /tmp/FastViTInitialPose.mlpackage \
  --output evaluation/results/fastvit_vs_hmr2_mini.json
```

Finally rerun `utils/benchmark_coreml_pipeline.py`. Replace the model in the
iPhone build only after the held-out teacher comparison, Core ML parity, image
connection guard, and workload benchmark all pass.
