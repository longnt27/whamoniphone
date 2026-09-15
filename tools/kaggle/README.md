# Kaggle notebooks

The final experiment is reproduced in this order:

1. `hmr2s_yolo26_grid_kaggle.ipynb` trains the two interface candidates for
   YOLO26n/s/m, selects on 3DPW validation, then evaluates the locked winner on
   3DPW test.
2. `hmr2s_temporal_smoothing_kaggle.ipynb` selects an output-only causal filter
   on validation and evaluates it on the same locked test population.

`hmr2s_wham_adapter_and_finetune_kaggle.ipynb` is the two-candidate base
experiment embedded by the grid builder. Rebuild notebooks with:

```bash
python3 tools/kaggle/build_hmr2s_yolo26_grid_notebook.py
python3 tools/kaggle/build_hmr2s_smoothing_notebook.py
```

The notebooks embed checksum-verified Python sources, so they remain portable
when uploaded to Kaggle.
