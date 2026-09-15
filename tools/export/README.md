# Core ML export

These scripts perform conversion only; they do not train models. The generated
`.mlpackage` directories are intentionally ignored because they are large and
recoverable from pinned checkpoints.

- `export_hmr2s_coreml.py`: image token/SMPL frontend and first-frame SMPL init.
- `export_hmr2s_token_adapter.py`: validation-selected residual 1024-D adapter.
- `export_wham_image_step.py`: explicit one-frame recurrent WHAM core.
- `export_wham_world_step.py`: SMPL mesh and contact-aware world refiner step.

The numerical conversion reports retained for the final experiment are under
`evaluation/results/selected`.
