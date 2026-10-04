# Kaggle notebooks

The final experiment is reproduced in this order:

1. `hmr2s_yolo26_grid_kaggle.ipynb` trains the two interface candidates for
   YOLO26n/s/m, selects on 3DPW validation, then evaluates the locked winner on
   3DPW test.
2. `hmr2s_temporal_smoothing_kaggle.ipynb` selects an output-only causal filter
   on validation and evaluates it on the same locked test population.
3. `hmr2s_camera_oracle_full11_3dpw_v2_kaggle.ipynb` is a follow-up, paired
   oracle-input ablation of that locked pipeline. It compares zero camera
   motion with motion calculated from 3DPW ground-truth camera rotations on
   **all 11,349 recurrent frames of the same 11 single-person test tracks**.
   It asserts track IDs and frame counts, and does not retrain or select a model.

The third notebook's ground-truth camera rotations are privileged offline
labels, **not** measurements of the iPhone's gyroscope or ARKit. It supplies
rotation only, not camera translation. Its first-frame-aligned world-root
and world-joint diagnostics are not the paper's official 100-frame world-MPJPE,
and its Kaggle GPU run
does not measure iPhone latency. All body-pose frames are processed. Transitions
without valid camera poses or adjacent raw frame IDs use zero motion in the
oracle arm; the report counts the skipped-frame transitions. World-root error
is scored on every frame with finite world-body labels, with explicit coverage
in the report. Camera-pose invalidity alone does not remove a world-body frame.
The earlier `hmr2s_camera_oracle_3dpw_kaggle.ipynb` selected only the longest
valid run per track; its result must not be used as the full 11-track result.

`hmr2s_wham_adapter_and_finetune_kaggle.ipynb` is the two-candidate base
experiment embedded by the grid builder. Rebuild notebooks with:

```bash
python3 tools/kaggle/build_hmr2s_yolo26_grid_notebook.py
python3 tools/kaggle/build_hmr2s_smoothing_notebook.py
python3 tools/kaggle/build_hmr2s_camera_oracle_notebook.py
```

The notebooks embed checksum-verified Python sources, so they remain portable
when uploaded to Kaggle.
