# Encoder Pipeline

End-to-end instructions for running the encoder pipeline on any supported environment.

## Supported Environments

| Config name | Environment ID                       | Artifact root              |
|-------------|--------------------------------------|----------------------------|
| `o2`        | `kinder/Obstruction2D-o2-v0`         | `artifacts_ob/encoder_o2`  |
| `o3`        | `kinder/Obstruction2D-o3-v0`         | `artifacts_ob/encoder_o3`  |
| `o4`        | `kinder/Obstruction2D-o4-v0`         | `artifacts_ob/encoder_o4`  |
| `sb1`       | `kinder/StickButton2D-b1-v0`         | `artifacts_sb/encoder_sb1` |
| `sb2`       | `kinder/StickButton2D-b2-v0`         | `artifacts_sb/encoder_sb2` |
| `sb3`       | `kinder/StickButton2D-b3-v0`         | `artifacts_sb/encoder_sb3` |
| `sb5`       | `kinder/StickButton2D-b5-v0`         | `artifacts_sb/encoder_sb5` |
| `sb10`      | `kinder/StickButton2D-b10-v0`        | `artifacts_sb/encoder_sb10`|

Artifact roots are resolved automatically by all launch scripts based on the
env name prefix (`o*` → `artifacts_ob/`, `sb*` → `artifacts_sb/`).

To add a new environment, create a config file in
`experiments/conf/encoder_dataset_difficulty/<name>.yaml` following the pattern
of the existing files, and set `output_dir` to the appropriate artifact root.

---

## Pipeline Overview

```
Stage 0: Vocabulary build + filter dataset  (all_filtered mode)
    ↓
Stage 1: Train/validation/test dataset build
    ↓
Stage 2: Vocabulary re-filter using full training dataset  (optional but recommended)
    ↓
    ├─── Encoder MAE pathway ───────────────────────────────┐
    │  Stage 3: Encoder (MAE) training                      │
    │  Stage 4: MAE offline evaluation                      │
    └───────────────────────────────────────────────────────┘
    │
    └─── Belief Encoder pathway ────────────────────────────┐
       Stage 5: HDF5 conversion  (pkl → HDF5)              │
       Stage 6: Belief encoder training                     │
       Stage 7: Comparison evaluation                       │
    └───────────────────────────────────────────────────────┘
```

All stages are submitted as SLURM jobs from the repo root. Run every command
from the repo root directory.

---

## Stage 0: Vocabulary Build and Initial Filter

Builds the full grounded-op vocabulary from seeds [0, 500), evaluates it on a
small filter dataset (seeds [500, 550)), then filters out sequences that are
never successfully refined.

```bash
# Obstruction2D:
bash experiments/launch_encoder_all_filtered_matrix.sh "o2 o3 o4"

# StickButton2D:
bash experiments/launch_encoder_all_filtered_matrix.sh "sb1 sb2 sb3 sb5 sb10"

# Both families at once:
bash experiments/launch_encoder_all_filtered_matrix.sh "o2 o3 o4 sb1 sb2 sb3 sb5 sb10"
```

**Wait for all SLURM jobs to complete before proceeding.**

Outputs per environment (e.g. `sb2` → `artifacts_sb/encoder_sb2/`):
- `encoder_vocab_full_all_filtered.pkl` — unfiltered vocab
- `encoder_filter_dataset.pkl` — 50-seed filter eval matrix
- `encoder_filter_dataset_filtered.pkl` — filtered matrix
- `encoder_vocab_filtered_all_filtered.pkl` — filtered vocab ← used in Stage 1

---

## Stage 1: Build Train / Validation / Test Datasets

Evaluates every vocab skeleton against each seed in the split, producing
applicability / success / steps-completed matrices.

Seed ranges:
- `train`:      [0, 500) — 500 seeds
- `validation`: [1000, 1100) — 100 seeds
- `test`:       [2000, 2100) — 100 seeds

```bash
bash experiments/launch_encoder_dataset_matrix.sh "sb1 sb2 sb3 sb5 sb10"
```

**Wait for all SLURM jobs to complete before proceeding.**

Outputs per environment (e.g. `artifacts_sb/encoder_sb2/`):
- `encoder_train_dataset.pkl`
- `encoder_validation_dataset.pkl`
- `encoder_test_dataset.pkl`

---

## Stage 2: Re-filter Vocabulary Using Training Dataset

Re-runs vocab filtering using the full 500-seed training split (more reliable
than the 50-seed filter dataset from Stage 0). Also propagates the filter to
validation and test splits.

```bash
# Default thresholds (threshold=0.05, min_appl_count=20):
bash experiments/launch_encoder_refilter_train_matrix.sh 0.05 20 "sb1 sb2 sb3 sb5 sb10"
```

**Wait for all SLURM jobs to complete before proceeding.**

Outputs per environment (e.g. `artifacts_sb/encoder_sb2/`):
- `encoder_vocab_filtered_train_filtered.pkl` — final vocab used for training
- `encoder_train_filtered_dataset.pkl`
- `encoder_validation_filtered_dataset.pkl`
- `encoder_test_filtered_dataset.pkl`

> **Alternative (faster but less robust):** Re-filter from the 50-seed filter
> dataset instead:
> ```bash
> bash experiments/launch_encoder_refilter_matrix.sh 0.05 15 "sb1 sb2 sb3 sb5 sb10"
> ```

---

## Stage 3: Encoder (MAE) Training

Trains a masked autoencoder for each environment × architecture combination.
The vocabulary size `M` is read automatically from the filtered vocab artifact.

```bash
# Steps mode (soft-label BCE on steps_completed_fraction, default):
bash experiments/launch_training_sweep.sh "sb1 sb2 sb3 sb5 sb10"

# Binary mode (hard BCE on success):
bash experiments/launch_training_sweep.sh "sb1 sb2 sb3 sb5 sb10" binary
```

**Wait for all GPU SLURM jobs to complete before proceeding.**

Architectures trained (hidden_dims = `[128, B, 128]`):
| `arch_name`      | Bottleneck `B`    |
|------------------|-------------------|
| `no_bottleneck`  | 128               |
| `full_M`         | M                 |
| `half_M`         | ceil(M/2)         |
| `quarter_M`      | ceil(M/4)         |
| `eighth_M`       | ceil(M/8)         |

Outputs per (env, arch), e.g. `artifacts_sb/encoder_sb2/arch_full_M/`:
- `encoder_best.pt`
- `encoder_last.pt`

---

## Stage 4: Offline Evaluation

Compares encoder-guided vs. baseline fixed-order rollout using precomputed
matrices (no simulator calls).

```bash
bash experiments/launch_eval_sweep.sh "sb1 sb2 sb3 sb5 sb10"

# Binary mode:
bash experiments/launch_eval_sweep.sh "sb1 sb2 sb3 sb5 sb10" binary
```

Outputs per (env, arch), e.g. `artifacts_sb/encoder_sb2/arch_full_M/offline_eval/`:
- `summary.json`, `metrics.npz`, `*.png`

---

## Full Pipeline for StickButton2D (all variants)

```bash
# Stage 0
bash experiments/launch_encoder_all_filtered_matrix.sh "sb1 sb2 sb3 sb5 sb10"
# ... wait ...

# Stage 1
bash experiments/launch_encoder_dataset_matrix.sh "sb1 sb2 sb3 sb5 sb10"
# ... wait ...

# Stage 2
bash experiments/launch_encoder_refilter_train_matrix.sh 0.05 20 "sb1 sb2 sb3 sb5 sb10"
# ... wait ...

# Stage 3
bash experiments/launch_training_sweep.sh "sb1 sb2 sb3 sb5 sb10"
# ... wait ...

# Stage 4
bash experiments/launch_eval_sweep.sh "sb1 sb2 sb3 sb5 sb10"
```

---

## Running Both Environment Families Together

All scripts accept any mix of env names:

```bash
bash experiments/launch_encoder_all_filtered_matrix.sh "o2 o3 o4 sb1 sb2 sb3 sb5 sb10"
bash experiments/launch_encoder_dataset_matrix.sh      "o2 o3 o4 sb1 sb2 sb3 sb5 sb10"
bash experiments/launch_encoder_refilter_train_matrix.sh 0.05 20 "o2 o3 o4 sb1 sb2 sb3 sb5 sb10"
bash experiments/launch_training_sweep.sh              "o2 o3 o4 sb1 sb2 sb3 sb5 sb10"
bash experiments/launch_eval_sweep.sh                  "o2 o3 o4 sb1 sb2 sb3 sb5 sb10"
# or equivalently:
bash experiments/launch_eval_sweep.sh all
```

---

## Smoke Test (Local, No SLURM)

To verify a new environment config works before submitting cluster jobs, run a
minimal local test with a tiny seed range:

```bash
# Replace sb2 with the env you want to test.
uv run python experiments/build_encoder_dataset.py \
  encoder_dataset_difficulty=sb2 \
  run.mode=all_filtered \
  vocab.seed_start=0 \
  vocab.seed_stop=5 \
  vocab.filter_seed_start=5 \
  vocab.filter_seed_stop=10 \
  vocab.filter_success_rate_threshold=0.0

# Verify the artifact was created:
ls artifacts_sb/encoder_sb2/
```

---

## Artifact Directory Layout

```
artifacts_ob/                              Obstruction2D outputs
└── encoder_<env>/                         e.g. encoder_o2/
    ├── encoder_vocab_full_all_filtered.pkl         Stage 0: unfiltered vocab
    ├── encoder_filter_dataset.pkl                  Stage 0: 50-seed filter matrix
    ├── encoder_filter_dataset_filtered.pkl         Stage 0: filtered matrix
    ├── encoder_vocab_filtered_all_filtered.pkl     Stage 0: filtered vocab
    ├── encoder_train_dataset.pkl                   Stage 1: raw train matrix
    ├── encoder_validation_dataset.pkl              Stage 1: raw val matrix
    ├── encoder_test_dataset.pkl                    Stage 1: raw test matrix
    ├── encoder_vocab_filtered_train_filtered.pkl   Stage 2: final vocab
    ├── encoder_train_filtered_dataset.pkl          Stage 2: filtered train
    ├── encoder_validation_filtered_dataset.pkl     Stage 2: filtered val
    ├── encoder_test_filtered_dataset.pkl           Stage 2: filtered test
    └── arch_<arch>/                                Stage 3+4 per architecture
        ├── encoder_best.pt
        ├── encoder_last.pt
        └── offline_eval/
            ├── summary.json
            ├── metrics.npz
            └── *.png

artifacts_sb/                              StickButton2D outputs
└── encoder_<env>/                         e.g. encoder_sb2/
    └── <same structure as above>

artifacts_hdf5/                            HDF5 datasets (Stage 5 output)
└── encoder_<env>/                         e.g. encoder_o2/
    ├── train.h5
    ├── validation.h5
    └── test.h5

checkpoints/                               Belief encoder checkpoints (Stage 6 output)
└── belief_encoder/
    └── <env>/                             e.g. o2/
        ├── belief_best.pt
        └── belief_last.pt

results/                                   Comparison results (Stage 7 output)
└── comparison/
    └── <env>/                             e.g. o2/
        ├── comparison_results.md
        └── success_first_ordering.json
```

---

## Belief Encoder Pipeline (Stages 5–7)

Stages 5–7 build on the outputs of Stage 2 to train and evaluate a belief
encoder that uses rollout-consistent prefixes and the index policy
`argmin E[T|Y=1] / P(Y=1)`.

### Stage 5: HDF5 Conversion

Converts filtered pickle artifacts to self-contained HDF5 files. Downstream
stages (training, comparison) read HDF5 directly — no dill bootstrap needed.

```bash
bash experiments/launch_hdf5_convert_matrix.sh "o2 o3 o4"

# StickButton2D:
bash experiments/launch_hdf5_convert_matrix.sh "sb1 sb2 sb3 sb5 sb10"
```

**Wait for all SLURM jobs to complete before proceeding.**

Outputs per environment (e.g. `encoder_o2` → `artifacts_hdf5/encoder_o2/`):
- `train.h5`
- `validation.h5`
- `test.h5`

Sanity check:
```bash
ls artifacts_hdf5/encoder_o2/*.h5 | wc -l   # should be 3
```

---

### Stage 6: Belief Encoder Training

Trains the full pipeline: SkeletonEncoder → TokenBuilder → BeliefEncoder →
prediction heads (Y, F, T, JointY). Uses DAgger-scheduled rollout-consistent
prefixes.

```bash
bash experiments/launch_belief_train_matrix.sh "o2 o3 o4"
```

**Wait for all GPU SLURM jobs to complete before proceeding.**

Outputs per environment (e.g. `checkpoints/belief_encoder/o2/`):
- `belief_best.pt` — best validation NLL checkpoint
- `belief_last.pt` — final epoch checkpoint

Sanity check:
```bash
ls checkpoints/belief_encoder/o2/belief_best.pt   # should exist
```

---

### Stage 7: Comparison Evaluation

Compares IndexPolicy (belief-encoder-guided) against Oracle (lower bound),
SuccessFirst (fitted on training set), and ShortestFirst (by plan length).
Reports markdown table with mean TTFS, 95% bootstrap CI, success@k.

```bash
bash experiments/launch_belief_comparison_matrix.sh "o2 o3 o4"
```

Outputs per environment (e.g. `results/comparison/o2/`):
- `comparison_results.md`
- `success_first_ordering.json`

Sanity check:
```bash
cat results/comparison/o2/comparison_results.md   # should show markdown table
```

---

### Full Belief Pipeline (One Command)

Chains stages 5 → 6 → 7 per difficulty using SLURM job dependencies. Each
difficulty runs independently — o2's training does not block o3's conversion.

```bash
# Submit all three stages for all difficulties:
bash experiments/launch_full_belief_pipeline.sh "o2 o3 o4"

# Dry-run (prints sbatch commands without submitting):
DRY_RUN=1 bash experiments/launch_full_belief_pipeline.sh "o2 o3 o4"
# or equivalently:
bash experiments/launch_full_belief_pipeline.sh --dry-run "o2 o3 o4"
```

Monitor progress:
```bash
squeue -u $USER
```

Tail logs for a specific job:
```bash
tail -f experiments/slurm_outputs/belief_encoder_<jobid>.out
```

---

### Running the Complete Pipeline End-to-End

Starting from Stages 0–2 outputs through the belief encoder evaluation:

```bash
# Stages 0-2 (existing — build and filter datasets)
bash experiments/launch_encoder_all_filtered_matrix.sh "o2 o3 o4"
# ... wait ...
bash experiments/launch_encoder_dataset_matrix.sh "o2 o3 o4"
# ... wait ...
bash experiments/launch_encoder_refilter_train_matrix.sh 0.05 20 "o2 o3 o4"
# ... wait ...

# Stages 5-7 (belief encoder — one command with dependencies)
bash experiments/launch_full_belief_pipeline.sh "o2 o3 o4"
```

---

### Rerunning a Single Stage or Complexity

```bash
# Rerun HDF5 conversion for o3 only:
sbatch experiments/convert_hdf5.slurm o3

# Retrain belief encoder for o2 with more epochs:
sbatch experiments/train_belief_encoder.slurm \
  data.train_path=artifacts_hdf5/encoder_o2/train.h5 \
  data.val_path=artifacts_hdf5/encoder_o2/validation.h5 \
  checkpoint.output_dir=checkpoints/belief_encoder/o2 \
  train.num_epochs=100

# Rerun comparison for o4 on CPU:
sbatch experiments/belief_comparison.slurm o4 --device cpu
```

---

### Troubleshooting

**Missing pickle prerequisites**
```
Missing: artifacts_ob/encoder_o3/encoder_train_filtered_dataset.pkl
```
Stages 0-2 must complete first. Check `squeue -u $USER` for pending jobs,
or re-run the relevant stage.

**OOM during belief encoder training**
If the job fails with CUDA OOM, reduce batch size:
```bash
sbatch experiments/train_belief_encoder.slurm \
  data.train_path=... data.val_path=... checkpoint.output_dir=... \
  train.batch_size=32
```

**Checkpoint loading error in comparison (shape mismatch)**
The model hyperparameters in `scripts/run_comparison.py` must match
`experiments/conf/train_belief_encoder_config.yaml`. Check that
`N_LAYERS_BELIEF` matches `belief_encoder.n_layers` in the config.

**dill bootstrap error for StickButton2D**
```
ModuleNotFoundError: No module named 'stickbutton2d_module'
```
The `convert_artifacts_to_hdf5.py` script must bootstrap the correct
environment before loading pickle files. Ensure the difficulty name (e.g.,
`sb3`) is passed correctly to the conversion script.
