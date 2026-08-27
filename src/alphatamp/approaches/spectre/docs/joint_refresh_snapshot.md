# Joint-training refresh snapshot (2026-08-26) — pre-residual baseline

Frozen record of the **jointly-trained** (step-join, no residual) refresh, captured before
the residual-adaptive re-do overwrote the SPECTRE cache. The joint checkpoints
(`checkpoints_spectre_atoms_v3final`, `_abl_records`, `_abl_scalars`) and the cache backups
(`compare_cache.joint_backup/`) remain on disk; restore a backup over the live cache to
re-render the joint notebook. Key finding: **records-only arm inert** (Δ +1.33 ns), which
motivated the residual-adaptive fix.


### dd2d  (env_variant=dd2d_v4)  §1 mean FP
| method           | seeds | ALL          | s0          | s1           | s2           | s3           |
|------------------|-------|--------------|-------------|--------------|--------------|--------------|
| astar-dist       | -     | 34.52        | 0.00        | 2.24         | 17.08        | 118.76       |
| PIGINet          | 3     | 17.27 ± 0.19 | 0.05 ± 0.02 | 5.04 ± 1.49  | 18.77 ± 1.58 | 45.20 ± 0.84 |
| SPECTRE-adaptive | 3     | 7.11 ± 0.49  | 0.00 ± 0.00 | 9.03 ± 1.58  | 10.07 ± 1.32 | 9.33 ± 0.78  |
| SPECTRE-static   | 3     | 20.52 ± 3.05 | 0.00 ± 0.00 | 24.35 ± 2.71 | 24.25 ± 2.43 | 33.48 ± 8.96 |
| LAZY-adaptive    | 3     | 23.26 ± 0.50 | 0.36 ± 0.08 | 9.59 ± 0.13  | 24.44 ± 1.41 | 58.65 ± 1.15 |
| VLMPlan-32B      | 1     | 23.55        | 6.76        | 5.04         | 13.16        | 69.24        |
| VLMPlan-GPT5.6   | 1     | 35.23        | 26.90       | 26.70        | 28.00        | 59.30        |

### sb2d_kinder  (env_variant=stickbutton2d_v1_kinder)  §1 mean FP
| method           | seeds | ALL         | s0          | s1          | s2          | s3          |
|------------------|-------|-------------|-------------|-------------|-------------|-------------|
| astar-dist       | -     | 16.29       | 0.08        | 0.56        | 2.96        | 61.56       |
| PIGINet          | 3     | 2.28 ± 0.29 | 0.07 ± 0.02 | 0.35 ± 0.02 | 1.17 ± 0.09 | 7.55 ± 1.06 |
| SPECTRE-adaptive | 3     | 1.88 ± 0.07 | 0.08 ± 0.00 | 0.41 ± 0.10 | 1.31 ± 0.06 | 5.72 ± 0.16 |
| SPECTRE-static   | 3     | 2.17 ± 0.13 | 0.08 ± 0.00 | 0.43 ± 0.08 | 1.40 ± 0.04 | 6.79 ± 0.41 |
| LAZY-adaptive    | 3     | 1.85 ± 0.02 | 0.08 ± 0.00 | 0.36 ± 0.00 | 2.32 ± 0.14 | 4.63 ± 0.15 |
| VLMPlan-32B      | 1     | 13.18       | 0.70        | 1.30        | 6.20        | 44.50       |
| VLMPlan-GPT5.6   | 1     | 6.42        | 0.00        | 2.40        | 0.90        | 22.40       |

### dd2d_gen_shapeonly  (env_variant=dd2d_v4gen_shapeonly_sz07)  §1 mean FP
| method           | seeds | ALL          | s0          | s1           | s2           | s3           |
|------------------|-------|--------------|-------------|--------------|--------------|--------------|
| astar-dist       | -     | 34.73        | 0.00        | 2.50         | 13.70        | 122.70       |
| PIGINet          | 3     | 22.68 ± 0.39 | 0.60 ± 0.20 | 15.80 ± 2.39 | 10.13 ± 2.15 | 64.17 ± 1.17 |
| SPECTRE-adaptive | 3     | 3.95 ± 1.00  | 0.00 ± 0.00 | 6.70 ± 2.07  | 3.03 ± 0.67  | 6.07 ± 2.45  |
| SPECTRE-static   | 3     | 15.57 ± 0.97 | 0.00 ± 0.00 | 23.20 ± 0.40 | 8.80 ± 0.85  | 30.27 ± 4.92 |

### dd2d_holdout_s3  (env_variant=dd2d_v4_holdout_s3)  §1 mean FP
| method           | seeds | ALL          | s0          | s1          | s2           | s3            |
|------------------|-------|--------------|-------------|-------------|--------------|---------------|
| astar-dist       | -     | 34.52        | 0.00        | 2.24        | 17.08        | 118.76        |
| PIGINet          | 3     | 27.88 ± 2.51 | 0.04 ± 0.04 | 6.08 ± 0.37 | 19.51 ± 1.19 | 85.89 ± 9.25  |
| SPECTRE-adaptive | 3     | 5.94 ± 2.68  | 0.00 ± 0.00 | 2.56 ± 1.71 | 9.59 ± 5.15  | 11.61 ± 6.75  |
| SPECTRE-static   | 3     | 20.24 ± 4.04 | 0.00 ± 0.00 | 8.89 ± 7.02 | 34.29 ± 8.85 | 37.79 ± 10.33 |
| VLMPlan-GPT5.6   | 1     | 35.23        | 26.90       | 26.70       | 28.00        | 59.30         |

### sb2d_holdout_b5  (env_variant=stickbutton2d_v1_kinder_holdout_b5)  §1 mean FP
| method           | seeds | ALL         | s0          | s1          | s2          | s3           |
|------------------|-------|-------------|-------------|-------------|-------------|--------------|
| astar-dist       | -     | 16.29       | 0.08        | 0.56        | 2.96        | 61.56        |
| PIGINet          | 3     | 1.68 ± 0.20 | 0.07 ± 0.02 | 0.32 ± 0.04 | 0.99 ± 0.22 | 5.36 ± 0.66  |
| SPECTRE-adaptive | 3     | 2.10 ± 0.78 | 0.08 ± 0.00 | 0.21 ± 0.08 | 1.40 ± 0.31 | 6.69 ± 2.82  |
| SPECTRE-static   | 3     | 3.34 ± 1.96 | 0.08 ± 0.00 | 0.35 ± 0.08 | 1.44 ± 0.46 | 11.51 ± 8.12 |
| VLMPlan-GPT5.6   | 1     | 6.42        | 0.00        | 2.40        | 0.90        | 22.40        |

### dd2d  §4.3 ablation (mean FP; Δ vs static, paired bootstrap)
| arm | ALL | s0 | s1 | s2 | s3 | Δ vs static |
|---|---|---|---|---|---|---|
| static | 18.35 | 0.00 | 21.84 | 21.09 | 30.45 | — |
| +records | 19.68 | 0.00 | 19.72 | 24.79 | 34.20 | +1.33 [-0.27, +2.91] |
| +scalars | 7.98 | 0.00 | 12.60 | 9.91 | 9.41 | -10.37 [-13.46, -7.55] * |
| full(deployed) | 7.11 | 0.00 | 9.03 | 10.07 | 9.33 | -11.24 [-14.88, -7.99] * |

### sb2d_kinder  §4.3 ablation (mean FP; Δ vs static, paired bootstrap)
| arm | ALL | s0 | s1 | s2 | s3 | Δ vs static |
|---|---|---|---|---|---|---|
| static | 2.22 | 0.08 | 0.36 | 1.52 | 6.92 | — |
| +records | 2.36 | 0.08 | 0.29 | 1.84 | 7.23 | +0.14 [-0.06, +0.37] |
| +scalars | 1.95 | 0.08 | 0.33 | 1.23 | 6.16 | -0.27 [-0.43, -0.11] * |
| full(deployed) | 1.88 | 0.08 | 0.41 | 1.31 | 5.72 | -0.34 [-0.54, -0.18] * |
