#!/bin/bash
#SBATCH --job-name=convert_hdf5
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/slurm_outputs/convert_hdf5_%j.out
#SBATCH --error=experiments/slurm_outputs/convert_hdf5_%j.err

set -euo pipefail

# Inputs:  artifacts_{ob|sb}/encoder_<difficulty>/encoder_{train,validation,test}_filtered_dataset.pkl
# Outputs: artifacts_hdf5/encoder_<difficulty>/{train,validation,test}.h5
# Runtime: ~2 minutes per difficulty (3 splits, each ~30s)

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p experiments/slurm_outputs

if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

# Usage:
# sbatch experiments/convert_hdf5.slurm <difficulty>
#
# Positional args:
#   $1  difficulty : o2|o3|o4|sb1|sb2|sb3|sb5|sb10
#
# Examples:
#   sbatch experiments/convert_hdf5.slurm o3
#   sbatch experiments/convert_hdf5.slurm sb2

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch experiments/convert_hdf5.slurm <difficulty>" >&2
  exit 2
fi

DIFFICULTY="$1"

case "$DIFFICULTY" in
  o2|o3|o4|sb1|sb2|sb3|sb5|sb10)
    ;;
  *)
    echo "Unknown difficulty: $DIFFICULTY (expected o2|o3|o4|sb1|sb2|sb3|sb5|sb10)" >&2
    exit 2
    ;;
esac

# Map difficulty name to its artifact root directory.
artifact_dir() {
  case "$1" in
    o*)  echo "artifacts_ob/encoder_$1" ;;
    sb*) echo "artifacts_sb/encoder_$1" ;;
    *)   echo "artifacts/encoder_$1" ;;
  esac
}

PKL_ROOT="$(artifact_dir "$DIFFICULTY")"

# Verify filtered pkl artifacts exist for all 3 splits.
for split in train validation test; do
  pkl="${PKL_ROOT}/encoder_${split}_filtered_dataset.pkl"
  if [[ ! -f "$pkl" ]]; then
    echo "Missing: $pkl" >&2
    echo "Run Stages 0-2 (vocab build + dataset build + refilter) first." >&2
    exit 2
  fi
done

echo "[convert_hdf5] difficulty=$DIFFICULTY  pkl_root=$PKL_ROOT"

python -u experiments/convert_artifacts_to_hdf5.py "$DIFFICULTY"
