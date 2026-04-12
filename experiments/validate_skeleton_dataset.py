"""Validate a skeleton HDF5 dataset against all required invariants.

Usage
-----
    uv run python experiments/validate_skeleton_dataset.py path/to/file.h5
    uv run python experiments/validate_skeleton_dataset.py path/to/file.h5 --no-strict

Invariants checked
------------------
1.  Shape consistency: all (N, M) arrays match attrs N and M.
2.  Binary fields: applicability ∈ {0.0, 1.0}, success ∈ {0.0, 1.0}.
3.  Y ≤ A: success implies applicability.
4.  Inapplicable ⟹ F=0, Y=0, T=0.
5.  Y=1 ⟹ F=1.0.
6.  F ∈ [0, 1].
7.  T ≥ 0 for applicable entries.
8.  F=K/L consistency: for each skeleton j, round(F[i,j] * L_j) must be
    an integer in [0, L_j] within tolerance 1e-4 (float32 rounding).
    Skeletons with L_j=0 must have F[i,j]=0 always.
9.  CSR consistency: op_offsets, param_offsets match array lengths;
    skeleton_lengths == diff(op_offsets).
10. ID bounds: all token IDs within vocab sizes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

# Ensure src is importable when run directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))


def validate_skeleton_dataset(
    hdf5_path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load an HDF5 skeleton dataset and check all invariants.

    Parameters
    ----------
    hdf5_path:
        Path to the .h5 file written by write_skeleton_dataset().
    strict:
        If True, raise AssertionError on the first invariant violation.
        If False, collect all violations and return them in the summary.

    Returns
    -------
    dict with keys:
        N, M, applicable_fraction, success_fraction,
        mean_refinement_time_applicable, violations (list[str]),
        skeleton_length_distribution (dict: length -> count),
        F_K_L_max_error (float)
    """
    hdf5_path = Path(hdf5_path)
    violations: list[str] = []

    def _check(condition: bool, message: str) -> None:
        if not condition:
            if strict:
                raise AssertionError(f"Invariant violated: {message}")
            violations.append(message)

    with h5py.File(hdf5_path, "r") as hf:
        N = int(hf.attrs["N"])
        M = int(hf.attrs["M"])

        # Load instance arrays
        applicability = hf["instances/applicability"][:]
        success = hf["instances/success"][:]
        steps = hf["instances/steps_completed_fraction"][:]
        time = hf["instances/refinement_time"][:]
        seed_ids = hf["instances/seed_ids"][:]

        # Load vocab arrays
        skeleton_lengths = hf["vocab/skeleton_lengths"][:]
        op_offsets = hf["vocab/op_offsets"][:]
        op_type_ids = hf["vocab/op_type_ids"][:]
        param_offsets = hf["vocab/param_offsets"][:]
        obj_ids_flat = hf["vocab/obj_ids_flat"][:]
        type_ids_flat = hf["vocab/type_ids_flat"][:]

        # Load string vocabs
        op_type_vocab_len = len(hf["meta/op_type_vocab"])
        obj_vocab_len = len(hf["meta/obj_vocab"])
        type_vocab_len = len(hf["meta/type_vocab"])

    # -----------------------------------------------------------------------
    # 1. Shape consistency
    # -----------------------------------------------------------------------
    _check(applicability.shape == (N, M), f"applicability shape {applicability.shape} != ({N}, {M})")
    _check(success.shape == (N, M), f"success shape {success.shape} != ({N}, {M})")
    _check(steps.shape == (N, M), f"steps_completed_fraction shape {steps.shape} != ({N}, {M})")
    _check(time.shape == (N, M), f"refinement_time shape {time.shape} != ({N}, {M})")
    _check(seed_ids.shape == (N,), f"seed_ids shape {seed_ids.shape} != ({N},)")
    _check(skeleton_lengths.shape == (M,), f"skeleton_lengths shape {skeleton_lengths.shape} != ({M},)")
    _check(op_offsets.shape == (M + 1,), f"op_offsets shape {op_offsets.shape} != ({M+1},)")

    # -----------------------------------------------------------------------
    # 2. Binary fields
    # -----------------------------------------------------------------------
    app_vals = np.unique(applicability)
    invalid_app = app_vals[~np.isin(app_vals, [0.0, 1.0])]
    _check(len(invalid_app) == 0, f"applicability has non-binary values: {invalid_app[:5]}")

    suc_vals = np.unique(success)
    invalid_suc = suc_vals[~np.isin(suc_vals, [0.0, 1.0])]
    _check(len(invalid_suc) == 0, f"success has non-binary values: {invalid_suc[:5]}")

    # -----------------------------------------------------------------------
    # 3. Y ≤ A
    # -----------------------------------------------------------------------
    y_gt_a = np.sum((success > applicability))
    _check(y_gt_a == 0, f"Y > A in {y_gt_a} entries (success=1 but applicability=0)")

    # -----------------------------------------------------------------------
    # 4. Inapplicable ⟹ F=0, Y=0, T=0
    # -----------------------------------------------------------------------
    inapplicable = applicability < 0.5

    f_nonzero_inapplicable = np.sum(inapplicable & (steps > 1e-6))
    _check(
        f_nonzero_inapplicable == 0,
        f"F > 0 for {f_nonzero_inapplicable} inapplicable entries",
    )

    y_nonzero_inapplicable = np.sum(inapplicable & (success > 0.5))
    _check(
        y_nonzero_inapplicable == 0,
        f"Y=1 for {y_nonzero_inapplicable} inapplicable entries",
    )

    t_nonzero_inapplicable = np.sum(inapplicable & (time > 1e-6))
    _check(
        t_nonzero_inapplicable == 0,
        f"T > 0 for {t_nonzero_inapplicable} inapplicable entries",
    )

    # -----------------------------------------------------------------------
    # 5. Y=1 ⟹ F=1.0
    # -----------------------------------------------------------------------
    success_mask = success > 0.5
    if np.any(success_mask):
        f_at_success = steps[success_mask]
        bad_f_at_success = np.sum(f_at_success < 1.0 - 1e-4)
        _check(
            bad_f_at_success == 0,
            f"F < 1.0 for {bad_f_at_success} entries where Y=1",
        )

    # -----------------------------------------------------------------------
    # 6. F ∈ [0, 1]
    # -----------------------------------------------------------------------
    f_below_zero = np.sum(steps < -1e-6)
    _check(f_below_zero == 0, f"F < 0 in {f_below_zero} entries")

    f_above_one = np.sum(steps > 1.0 + 1e-4)
    _check(f_above_one == 0, f"F > 1 in {f_above_one} entries")

    # -----------------------------------------------------------------------
    # 7. T ≥ 0 for applicable entries
    # -----------------------------------------------------------------------
    applicable_mask = applicability > 0.5
    if np.any(applicable_mask):
        t_negative = np.sum(time[applicable_mask] < -1e-8)
        _check(t_negative == 0, f"T < 0 for {t_negative} applicable entries")

    # -----------------------------------------------------------------------
    # 8. F = K/L consistency
    # -----------------------------------------------------------------------
    max_f_error = 0.0

    for j in range(M):
        L_j = int(skeleton_lengths[j])
        f_col = steps[:, j]

        if L_j == 0:
            # All F must be 0 for empty skeletons
            nonzero = np.sum(np.abs(f_col) > 1e-6)
            _check(
                nonzero == 0,
                f"Skeleton {j} has L=0 but {nonzero} entries with F != 0",
            )
            continue

        # For each applicable entry: K = round(F * L) must be integer in [0, L]
        app_col = applicability[:, j] > 0.5
        if not np.any(app_col):
            continue

        f_app = f_col[app_col]
        k_float = f_app * L_j
        k_rounded = np.round(k_float)
        error = np.abs(k_float - k_rounded)
        col_max_error = float(np.max(error)) if len(error) > 0 else 0.0
        max_f_error = max(max_f_error, col_max_error)

        _check(
            col_max_error < 1e-4,
            f"Skeleton {j}: F=K/L error up to {col_max_error:.2e} (L={L_j})",
        )

        # K must be in [0, L]
        k_int = k_rounded.astype(np.int64)
        out_of_range = np.sum((k_int < 0) | (k_int > L_j))
        _check(
            out_of_range == 0,
            f"Skeleton {j}: {out_of_range} entries have K outside [0, {L_j}]",
        )

    # -----------------------------------------------------------------------
    # 9. CSR consistency
    # -----------------------------------------------------------------------
    T = len(op_type_ids)
    _check(
        int(op_offsets[-1]) == T,
        f"op_offsets[-1]={int(op_offsets[-1])} != len(op_type_ids)={T}",
    )

    expected_lengths = np.diff(op_offsets).astype(np.int16)
    lengths_match = np.all(expected_lengths == skeleton_lengths)
    _check(
        lengths_match,
        "skeleton_lengths != diff(op_offsets)",
    )

    P = len(obj_ids_flat)
    _check(
        int(param_offsets[-1]) == P,
        f"param_offsets[-1]={int(param_offsets[-1])} != len(obj_ids_flat)={P}",
    )
    _check(
        len(type_ids_flat) == P,
        f"len(type_ids_flat)={len(type_ids_flat)} != len(obj_ids_flat)={P}",
    )
    _check(
        len(param_offsets) == T + 1,
        f"len(param_offsets)={len(param_offsets)} != T+1={T+1}",
    )

    # -----------------------------------------------------------------------
    # 10. ID bounds
    # -----------------------------------------------------------------------
    if T > 0:
        max_op_id = int(np.max(op_type_ids))
        _check(
            max_op_id < op_type_vocab_len,
            f"op_type_id {max_op_id} >= op_type_vocab size {op_type_vocab_len}",
        )
    if P > 0:
        max_obj_id = int(np.max(obj_ids_flat))
        _check(
            max_obj_id < obj_vocab_len,
            f"obj_id {max_obj_id} >= obj_vocab size {obj_vocab_len}",
        )
        max_type_id = int(np.max(type_ids_flat))
        _check(
            max_type_id < type_vocab_len,
            f"type_id {max_type_id} >= type_vocab size {type_vocab_len}",
        )

    # -----------------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------------
    app_fraction = float(np.mean(applicability)) if N * M > 0 else float("nan")
    suc_fraction = (
        float(np.mean(success[applicable_mask]))
        if np.any(applicable_mask)
        else float("nan")
    )
    mean_time_applicable = (
        float(np.mean(time[applicable_mask]))
        if np.any(applicable_mask)
        else float("nan")
    )

    length_values, length_counts = np.unique(skeleton_lengths, return_counts=True)
    skeleton_length_distribution = {
        int(v): int(c) for v, c in zip(length_values, length_counts)
    }

    return {
        "N": N,
        "M": M,
        "applicable_fraction": app_fraction,
        "success_fraction": suc_fraction,
        "mean_refinement_time_applicable": mean_time_applicable,
        "violations": violations,
        "skeleton_length_distribution": skeleton_length_distribution,
        "F_K_L_max_error": max_f_error,
    }


def _print_summary(hdf5_path: Path, summary: dict[str, Any]) -> None:
    print()
    print("=== Skeleton Dataset Validation ===")
    print(f"File: {hdf5_path}")
    print(f"N={summary['N']}, M={summary['M']}")
    print(f"Applicable fraction:              {summary['applicable_fraction']:.3f}")
    print(f"Success fraction (of applicable): {summary['success_fraction']:.3f}")
    print(
        f"Mean refinement_time (applicable): {summary['mean_refinement_time_applicable']:.4f} s"
    )
    print(f"Skeleton length distribution:     {summary['skeleton_length_distribution']}")
    print(f"Max F=K/L reconstruction error:   {summary['F_K_L_max_error']:.2e}")

    n_violations = len(summary["violations"])
    if n_violations == 0:
        print("Violations found: 0")
        print("All invariants PASSED.")
    else:
        print(f"Violations found: {n_violations}")
        for v in summary["violations"]:
            print(f"  FAIL: {v}")
        print("SOME INVARIANTS FAILED.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a skeleton HDF5 dataset against all required invariants."
    )
    parser.add_argument("hdf5_path", type=Path, help="Path to the .h5 file to validate")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Collect all violations instead of raising on first failure",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    strict = not args.no_strict

    summary = validate_skeleton_dataset(args.hdf5_path, strict=strict)
    _print_summary(args.hdf5_path, summary)

    if summary["violations"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
