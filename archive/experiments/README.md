# Historical experiments

This directory preserves experiments that informed the final mobile pipeline but
are **not** the deployable implementation and must not be used as current result
evidence.

- `fastvit/` contains the original FastViT distillation path. It established
  that matching one teacher feature does not guarantee end-to-end WHAM accuracy.
- `bedlam/` contains the synthetic-data replay/retraining attempts. These exposed
  the cost of changing an encoder and decoder out of sequence.
- `tiny_pipeline/` contains early partial and frozen-component evaluations.
- `early_results/` contains reports superseded by the locked YOLO26m/HMR2-S
  adapter experiment.
- `legacy/` contains the first extraction/conversion notebooks and sample data.

The final architecture, locked evidence, and reproduction entry points live in
`README.md`, `evaluation/results/selected`, and `tools` respectively. Historical
source is retained for auditability; no archived metric is quoted as the final
mobile result.
