"""Generate a synthetic skeleton reordering dataset for testing.

Produces a dataset_dict compatible with write_skeleton_dataset() without
any dependency on kinder, bilevel_planning, or real TAMP environments.
Operator sequences use duck-typed lightweight objects that expose the same
interface as GroundOperator (op.name, op.parameters, p.name, p.type.name).

All invariants are guaranteed by construction:
  - Y=1  ⟹  F=1.0
  - A=0  ⟹  F=0, Y=0, T=0
  - F = K/L for integer K ∈ [0, L]   (L = skeleton length)

Usage
-----
    uv run python experiments/build_synthetic_dataset.py \\
        --output data/synthetic.h5 --N 500 --M 20 --seed 42

    # Validate the result:
    uv run python experiments/validate_skeleton_dataset.py data/synthetic.h5
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Ensure src is importable when run directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from alphatamp.data.skeleton_dataset import write_skeleton_dataset

# ---------------------------------------------------------------------------
# Duck-typed synthetic operator types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SyntheticType:
    name: str


@dataclass(frozen=True)
class _SyntheticObj:
    name: str
    type: _SyntheticType


@dataclass(frozen=True)
class _SyntheticOp:
    name: str
    parameters: tuple[_SyntheticObj, ...]


# ---------------------------------------------------------------------------
# Default domain configuration
# ---------------------------------------------------------------------------

_BLOCK = _SyntheticType("block")
_LOCATION = _SyntheticType("location")

_DEFAULT_OP_ARITIES: dict[str, int] = {
    "Pick": 1,
    "Place": 2,
    "Stack": 2,
    "Unstack": 2,
    "Move": 1,
}

_DEFAULT_OBJ_POOL: list[_SyntheticObj] = [
    _SyntheticObj("obj0", _BLOCK),
    _SyntheticObj("obj1", _BLOCK),
    _SyntheticObj("obj2", _BLOCK),
    _SyntheticObj("loc0", _LOCATION),
    _SyntheticObj("loc1", _LOCATION),
]


# ---------------------------------------------------------------------------
# Skeleton generation
# ---------------------------------------------------------------------------


def _sample_op_sequence(
    rng: np.random.Generator,
    op_arities: dict[str, int],
    obj_pool: list[_SyntheticObj],
    length: int,
) -> tuple[_SyntheticOp, ...]:
    """Sample one synthetic operator sequence of the given length."""
    op_names = list(op_arities.keys())
    ops: list[_SyntheticOp] = []
    for _ in range(length):
        op_name = op_names[rng.integers(len(op_names))]
        arity = op_arities[op_name]
        # Sample objects with replacement
        chosen = tuple(
            obj_pool[int(i)] for i in rng.integers(len(obj_pool), size=arity)
        )
        ops.append(_SyntheticOp(name=op_name, parameters=chosen))
    return tuple(ops)


def build_synthetic_dataset(
    N: int = 500,
    M: int = 20,
    rng_seed: int = 42,
    op_arities: dict[str, int] | None = None,
    obj_pool: list[_SyntheticObj] | None = None,
    max_skeleton_length: int = 6,
    min_skeleton_length: int = 1,
    applicability_rate: float = 0.6,
    success_given_applicable_rate: float = 0.3,
) -> dict[str, Any]:
    """Generate a synthetic dataset_dict in build_dataset() format.

    Parameters
    ----------
    N:
        Number of problem instances (rows).
    M:
        Vocabulary size — number of distinct skeletons (columns).
    rng_seed:
        NumPy random seed for reproducibility.
    op_arities:
        Mapping from operator name to number of parameters.
        Defaults to Pick/Place/Stack/Unstack/Move with arities 1-2.
    obj_pool:
        List of synthetic objects to draw parameters from.
        Defaults to obj0-obj2 (block) and loc0-loc1 (location).
    max_skeleton_length:
        Maximum number of operators per skeleton.
    min_skeleton_length:
        Minimum number of operators per skeleton (≥ 0).
    applicability_rate:
        Bernoulli probability that A[i,j]=1 for each (instance, skeleton) pair.
    success_given_applicable_rate:
        Bernoulli probability that Y[i,j]=1 given A[i,j]=1.

    Returns
    -------
    dict with keys matching EncoderApproach.build_dataset() output:
        seed_ids, op_sequence_vocab, applicability, success,
        refinement_time, steps_completed_fraction, skeleton_lengths.
    """
    if op_arities is None:
        op_arities = _DEFAULT_OP_ARITIES
    if obj_pool is None:
        obj_pool = _DEFAULT_OBJ_POOL

    rng = np.random.default_rng(rng_seed)

    # --- Sample M distinct skeleton lengths and op sequences ---
    lengths = rng.integers(min_skeleton_length, max_skeleton_length + 1, size=M)
    op_sequence_vocab: list[tuple[_SyntheticOp, ...]] = []
    for j in range(M):
        seq = _sample_op_sequence(rng, op_arities, obj_pool, int(lengths[j]))
        op_sequence_vocab.append(seq)

    skeleton_lengths = np.array([len(seq) for seq in op_sequence_vocab], dtype=np.int16)

    # --- Sample per-(instance, skeleton) outcome matrices ---
    applicability = np.zeros((N, M), dtype=np.float32)
    success = np.zeros((N, M), dtype=np.float32)
    steps_completed_fraction = np.zeros((N, M), dtype=np.float32)
    refinement_time = np.zeros((N, M), dtype=np.float32)

    for j in range(M):
        L_j = int(skeleton_lengths[j])

        # Applicability column
        app_col = (rng.random(N) < applicability_rate).astype(np.float32)
        applicability[:, j] = app_col

        applicable_mask = app_col > 0.5

        if not np.any(applicable_mask):
            continue

        n_applicable = int(np.sum(applicable_mask))

        # Success among applicable
        success_among_applicable = rng.random(n_applicable) < success_given_applicable_rate

        # Refinement time (log-normal, always > 0 for applicable)
        times = rng.lognormal(mean=0.0, sigma=0.5, size=n_applicable) + 0.01

        app_indices = np.where(applicable_mask)[0]

        for idx, (row, suc, t) in enumerate(
            zip(app_indices, success_among_applicable, times)
        ):
            success[row, j] = 1.0 if suc else 0.0
            refinement_time[row, j] = float(t)

            if L_j == 0:
                # Empty skeleton: no steps to complete; cannot succeed
                success[row, j] = 0.0
                steps_completed_fraction[row, j] = 0.0
            elif suc:
                # Success: all steps completed
                steps_completed_fraction[row, j] = 1.0
            else:
                # Partial: K ∈ {0, ..., L_j-1}
                K = int(rng.integers(0, L_j))
                steps_completed_fraction[row, j] = float(K) / float(L_j)

    seed_ids = list(range(N))

    return {
        "seed_ids": seed_ids,
        "op_sequence_vocab": op_sequence_vocab,
        "applicability": applicability,
        "success": success,
        "refinement_time": refinement_time,
        "steps_completed_fraction": steps_completed_fraction,
        "skeleton_lengths": skeleton_lengths,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic skeleton dataset and save to HDF5."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic.h5"),
        help="Output .h5 file path (default: data/synthetic.h5)",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=500,
        help="Number of problem instances (default: 500)",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=20,
        help="Vocabulary size — number of skeletons (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=6,
        help="Maximum skeleton length in operators (default: 6)",
    )
    parser.add_argument(
        "--applicability-rate",
        type=float,
        default=0.6,
        help="Fraction of (instance, skeleton) pairs that are applicable (default: 0.6)",
    )
    parser.add_argument(
        "--success-rate",
        type=float,
        default=0.3,
        help="Success rate given applicable (default: 0.3)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"Generating synthetic dataset: N={args.N}, M={args.M}, seed={args.seed}")
    dataset_dict = build_synthetic_dataset(
        N=args.N,
        M=args.M,
        rng_seed=args.seed,
        max_skeleton_length=args.max_length,
        applicability_rate=args.applicability_rate,
        success_given_applicable_rate=args.success_rate,
    )

    print(f"Writing to {args.output} ...")
    write_skeleton_dataset(
        args.output,
        dataset_dict,
        source_description=f"Synthetic N={args.N} M={args.M} seed={args.seed}",
    )
    print("Done.")

    # Quick summary
    app = dataset_dict["applicability"]
    suc = dataset_dict["success"]
    app_frac = float(app.mean())
    suc_given_app = float(suc[app > 0.5].mean()) if (app > 0.5).any() else float("nan")
    print(f"  Applicable fraction:          {app_frac:.3f}")
    print(f"  Success | applicable:          {suc_given_app:.3f}")
    print(f"  Skeleton lengths (first 10):  {dataset_dict['skeleton_lengths'][:10].tolist()}")


if __name__ == "__main__":
    main()
