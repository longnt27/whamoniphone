# Custom 3DPW world-space results

These are the unmodified Kaggle report archives supplied after the locked
camera-relative 3DPW evaluation. They use a separate, explicitly documented
project protocol, so they are stored apart from the selected camera-relative
and device evidence manifest. Their numerical results are reported, not
discarded or represented as WHAM-paper-equivalent benchmarks.

| Archive | SHA-256 | Purpose |
| --- | --- | --- |
| `camera_oracle_full11_3dpw_reports.zip` | `7f1107e26355ab11528034acb4fb364902cadd416fefb0a783b13709fb6da9fc` | Zero camera rotation versus 3DPW ground-truth rotation, mobile branch only |
| `paired_world_full11_3dpw_reports.zip` | `31c73d3ce1a37a17c836cf76428a4dce3f3e31d49e2fe21356fd9bd18a977ebb` | Released-WHAM inputs versus mobile inputs, both with oracle rotation and the same exported-world-step implementation |

Both archives contain their original JSON, per-track CSV, and internal artifact
manifest. Each run used 11 single-person tracks and 11,349 recurrent frames.
Their parsed frame IDs include **143 nonadjacent transitions**. The scripts
advanced recurrent state across each gap while replacing the camera-motion
input with zero. World-position integration therefore omits elapsed motion.
The paired run's first-frame-aligned full-track world-joint errors were
**4.863 m versus 4.163 m**: the mobile branch scored lower under this exact
protocol, but the size and direction of the skipped-time bias are unknown.
These are not the WHAM paper's EMDB-2 100-frame world metric or
whole-trajectory RTE.

The ground-truth rotation is an offline oracle, not an iPhone gyro recording.
Neither archive validates gyro error or on-device global-trajectory accuracy.
See the [final report](../../../docs/FINAL_REPORT.md#custom-3dpw-world-space-protocol)
for the bounded conclusion.
