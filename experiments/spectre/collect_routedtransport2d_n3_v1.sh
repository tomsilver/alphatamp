#!/bin/bash
# Collect the full SPECTRE dataset for RoutedTransport2D-n3-v1 locally.
#
# Three splits per src/alphatamp/approaches/spectre/docs/archive/ROUTED_TRANSPORT2D_SPEC.md §4.4:
#   train = [0, 500)        500 problems
#   val   = [1000, 1100)    100 problems
#   test  = [2000, 2100)    100 problems
#
# The non-overlapping seed ranges (1000-gap between splits) ensure no shared
# RNG state leaks across splits. Existing files are skipped, so the script is
# safe to re-run after an interruption.
#
# Usage from the repo root:
#
#   ./experiments/spectre/collect_routedtransport2d_n3_v1.sh
#
# Overrides (env vars):
#   DATA_ROOT     default: data/spectre
#   K_MAX         default: 30 (the family-balanced cap from spec §5.1)
#   EXTRA_ARGS    extra Hydra overrides forwarded verbatim to every split,
#                 e.g.  EXTRA_ARGS="workers=4" ./experiments/spectre/collect_...

set -euo pipefail

ENV_VARIANT=routedtransport2d_n3_v1
DATA_ROOT=${DATA_ROOT:-data/spectre}
K_MAX=${K_MAX:-30}
EXTRA_ARGS=${EXTRA_ARGS:-}

# Activate the project venv so this works from a fresh shell.
if [ -z "${VIRTUAL_ENV:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

run_split() {
    local split="$1" start="$2" end="$3"
    local count=$((end - start))
    echo
    echo "=== ${ENV_VARIANT} / ${split}: collecting ${count} problems [seeds ${start}, ${end})"
    # shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
    python experiments/spectre/spectre_collect.py \
        env="${ENV_VARIANT}" \
        split="${split}" \
        problem_seed_start="${start}" \
        problem_seed_end="${end}" \
        num_problems="${count}" \
        data_root="${DATA_ROOT}" \
        K_max="${K_MAX}" \
        abstract_plan_timeout_s=1.0 \
        refinement_timeout_s=0.1 \
        ${EXTRA_ARGS}
}

echo "Collecting RoutedTransport2D-n3-v1 → ${DATA_ROOT}/raw/${ENV_VARIANT}/"

run_split train 0    500
run_split val   1000 1100
run_split test  2000 2100

cat <<EOF

=== Done. Episodes written to:
  ${DATA_ROOT}/raw/${ENV_VARIANT}/{train,val,test}/episodes/ep_NNNNN.pkl.gz

Quick sanity check:
  ls ${DATA_ROOT}/raw/${ENV_VARIANT}/train/episodes/ | wc -l   # expect 500
  ls ${DATA_ROOT}/raw/${ENV_VARIANT}/val/episodes/   | wc -l   # expect 100
  ls ${DATA_ROOT}/raw/${ENV_VARIANT}/test/episodes/  | wc -l   # expect 100

Next step — run EDA on the collected splits (see src/alphatamp/approaches/spectre/docs/archive/SPECTRE_EDA_SPEC.md):
  python experiments/spectre/spectre_build_vocab.py data_root=${DATA_ROOT}
EOF
