# Evaluation modules

The large Kaggle notebook builders embed these modules by checksum. Some support
modules retain historical names such as `distill_fastvit_hmr2.py`; they provide
shared data-loading and metric functions but do not make FastViT part of the
selected pipeline.

To verify that the checked-in reports consistently identify one locked pipeline:

```bash
python3 tools/evaluation/validate_final_evidence.py
```
