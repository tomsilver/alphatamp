#!/bin/bash
# Submit three SLURM jobs (train / val / test) for SPECTRE dataset collection
# on kinder/StickButton2D-b5-v0. Each job runs ``spectre_collect.slurm`` with
# ``workers=8`` inside the SLURM allocation so one job saturates all 8 cores
# (see experiments/spectre/conf/hydra/launcher/slurm.yaml).
#
# StickButton2D-b5 is natively pre-registered in kinder (see
# ``kinder/__init__.py::_register_kinematic2d``) — no env bootstrap required.
#
# Split seed ranges follow src/alphatamp/approaches/spectre/docs/archive/SPECTRE_METHOD_SPEC.md §5.1:
#   train = [0, 500)    500 problems
#   val   = [500, 600)  100 problems
#   test  = [600, 700)  100 problems
#
# Each problem uses K_max=50 skeleton attempts (inherited from
# experiments/spectre/conf/spectre_collect.yaml, matching the ClutteredStorage2D
# collection convention).
#
# Usage from the repo root:
#
#   ./experiments/spectre/submit_spectre_stickbutton2d_b5.sh
#
# Overrides (via environment variables):
#   TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END  (ints)
#   DATA_ROOT      default: data/spectre
#   EXTRA_ARGS     trailing Hydra overrides forwarded to every job,
#                  e.g.   EXTRA_ARGS="K_max=30 refinement_timeout_s=30.0" ./...
#
# After submission, monitor with:
#   squeue -u $USER
#   tail -f experiments/slurm_outputs/spectre_stickbutton2d_b5_train_<jobid>.out

set -euo pipefail

ENV_VARIANT=stickbutton2d_b5
OUT_DIR=experiments/slurm_outputs
mkdir -p "${OUT_DIR}"

# Per-split seed ranges — override with env vars to resume / extend.
TRAIN_START=${TRAIN_START:-0}
TRAIN_END=${TRAIN_END:-500}
VAL_START=${VAL_START:-500}
VAL_END=${VAL_END:-600}
TEST_START=${TEST_START:-600}
TEST_END=${TEST_END:-700}

DATA_ROOT=${DATA_ROOT:-data/spectre}
EXTRA_ARGS=${EXTRA_ARGS:-}

if ! command -v sbatch >/dev/null 2>&1; then
    echo "error: sbatch not found on PATH; run this on a SLURM login node" >&2
    exit 1
fi

submit_split() {
    local split="$1" start="$2" end="$3"
    local jobname="spectre_${ENV_VARIANT}_${split}"
    local job_id
    # shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
    job_id=$(sbatch --parsable \
        -J "${jobname}" \
        -o "${OUT_DIR}/${jobname}_%j.out" \
        -e "${OUT_DIR}/${jobname}_%j.err" \
        experiments/spectre/spectre_collect.slurm \
            env=${ENV_VARIANT} \
            split=${split} \
            problem_seed_start=${start} \
            problem_seed_end=${end} \
            data_root=${DATA_ROOT} \
            ${EXTRA_ARGS})
    printf "  submitted %-6s job_id=%s  range=[%d, %d)  log=%s\n" \
        "${split}" "${job_id}" "${start}" "${end}" \
        "${OUT_DIR}/${jobname}_${job_id}.out"
}

echo "Submitting SPECTRE collection on ${ENV_VARIANT} (data_root=${DATA_ROOT}):"
submit_split train "${TRAIN_START}" "${TRAIN_END}"
submit_split val   "${VAL_START}"   "${VAL_END}"
submit_split test  "${TEST_START}"  "${TEST_END}"

cat <<EOF

Monitor:  squeue -u \$USER
Cancel:   scancel -u \$USER --name=spectre_${ENV_VARIANT}_train  # or val / test
Output:   ${DATA_ROOT}/raw/${ENV_VARIANT}/{train,val,test}/episodes/ep_NNNNN.pkl.gz

After all three jobs finish, build the vocab from the train split:
  python experiments/spectre/spectre_build_vocab.py env=${ENV_VARIANT} data_root=${DATA_ROOT}
EOF
