"""Restock3D-v3 REAL (hybrid-prune) collection path in ``collect.collect_episode``.

``refiner_mode="hybrid_prune"``: classify all K_max candidates analytically, then REAL-refine only
the analytic-feasible ones + a deterministic 25% audit sample of the analytic-infeasible ones,
trusting the analytic label for the rest. Each ``OutcomeRecord`` carries
``label_source in {real, analytic}``. Marked ``slow`` -- the real-refined subset needs PyBullet +
IKFast (the analytic-trusted bulk does not).
"""

from __future__ import annotations

import glob
import os
import pathlib

import pytest

pytestmark = pytest.mark.slow


def _blas_shim() -> None:
    b = os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
    os.environ.setdefault("LAPACK_DIR", b)
    os.environ.setdefault("BLAS_DIR", b)
    pathlib.Path(b).mkdir(parents=True, exist_ok=True)
    for a, (sd, pt) in {
        "liblapack.a": ("lapack", "liblapack.so.3*"),
        "libblas.a": ("blas", "libblas.so.3*"),
    }.items():
        lk = pathlib.Path(b) / a
        if not (lk.exists() or lk.is_symlink()):
            cands = sorted(
                glob.glob(f"/usr/lib/x86_64-linux-gnu/{sd}/{pt}")
                + glob.glob(f"/usr/lib/x86_64-linux-gnu/{pt}")
            )
            real = next((c for c in cands if os.path.isfile(c)), None)
            if real:
                lk.symlink_to(real)


def _collect_hybrid(stratum: int = 0, k_max: int = 12, r_cap: float = 12.0):
    pytest.importorskip("kinder")
    _blas_shim()
    from alphatamp.approaches.spectre.collect import collect_episode
    from alphatamp.approaches.spectre.config import CollectionConfig
    from alphatamp.approaches.spectre.envs.restock3d import strata_v3 as S

    pid = S.problem_id("train", stratum, 0)
    cfg = CollectionConfig(
        env_id=S.env_id(stratum),
        env_variant="restock3d_v3_real",
        model_name="restock3d_v3",
        model_kwargs={"stratum": stratum},
        split="train",
        num_problems=1,
        problem_seed_start=pid,
        problem_seed_end=pid + 1,
        K_max=k_max,
        abstract_plan_timeout_s=120.0,
        refinement_timeout_s=r_cap,
        num_sampling_attempts_per_step=6,
        max_trajectory_steps=500,
        plan_generator="closed_form",
        refiner_mode="hybrid_prune",
    )
    return collect_episode(cfg, pid)


def test_hybrid_prune_labels_and_invariants() -> None:
    ep = _collect_hybrid()

    assert ep.summary.num_skeletons == len(ep.outcomes) > 0

    n_real = n_analytic = 0
    for o in ep.outcomes:
        # (1) Every candidate is labelled real or analytic.
        assert o.label_source in ("real", "analytic")
        reason = (o.refiner_metadata or {}).get("prune_reason")
        if o.label_source == "real":
            n_real += 1
            # (3) real-refined either because analytic-feasible, or drawn into the audit.
            assert reason in ("analytic_feasible", "audit_sample")
        else:
            n_analytic += 1
            # (2) analytic-trusted candidates are always infeasible => a "fail" carrying an
            # analytic failure record; never a success (an analytic-feasible is ALWAYS real-refined).
            assert reason == "analytic_trusted"
            assert o.outcome == "fail"
            assert (o.refiner_metadata or {}).get("failures")
        # (4) every SUCCESS is real-verified (analytic-feasible => always real-refined).
        if o.outcome == "success":
            assert o.label_source == "real"

    # (5) hybrid actually pruned AND actually spent real MP: a mix of both label sources. The
    # geometry-first pool puts a feasible skeleton early (=> >=1 real), and ~95% of candidates are
    # analytically infeasible with only 25% audited (=> >=1 trusted-analytic).
    assert n_real >= 1, "expected at least one real-refined candidate"
    assert n_analytic >= 1, "expected at least one analytic-trusted candidate (hybrid pruned)"


def test_hybrid_prune_label_source_is_deterministic() -> None:
    # The real-vs-trust partition depends only on classify_skeleton + a string-seeded audit draw,
    # both deterministic within a process -- so the label_source sequence is reproducible even
    # though a borderline candidate's real outcome could differ with MP timing.
    a = [o.label_source for o in _collect_hybrid().outcomes]
    b = [o.label_source for o in _collect_hybrid().outcomes]
    assert a == b and len(a) > 0
