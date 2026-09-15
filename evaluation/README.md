# Evaluation evidence

The canonical result set is [`results/selected`](results/selected). Its
[`selected/manifest.json`](results/selected/manifest.json) records the source
and SHA-256 of every retained report.

The evaluation follows two locks:

1. Six YOLO26/interface candidates are ranked on 3DPW validation. YOLO26m-pose
   plus the residual HMR2-S→HMR2a token adapter wins. No training process loads
   3DPW test before this choice is fixed.
2. Raw/light/medium/strong output filters are ranked on validation under a 5%
   spatial-regression gate. Light smoothing is locked before its 3DPW test
   comparison.

The selected directory contains:

- `yolo26_grid_validation.*` — candidate selection;
- `yolo26_grid_final_3dpw.*` — released-WHAM and locked mobile test results;
- `selected_deployment_artifact.json` — exact YOLO/adapter identity;
- `hmr2s_temporal_smoothing_3dpw.*` — smoothing selection and test ablation;
- `selected_mobile_pipeline_device_benchmark.json` — physical iPhone workload;
- `*_export_report.json` — Core ML numerical conversion checks; and
- `manifest.json` — artifact origins and hashes.

Validate hashes, selection consistency, runtime guards, and required report
coverage from the repository root:

```bash
python3 tools/evaluation/validate_final_evidence.py
```

Earlier FastViT, BEDLAM, frozen-HMR2-S, and partial-pipeline reports are under
[`archive/experiments`](../archive/experiments). They are historical diagnostic
evidence and must not be quoted as the selected mobile result. The narrative and
interpretation are in [`docs/FINAL_REPORT.md`](../docs/FINAL_REPORT.md).
