"""Step-2 gate (docs/piginet_dd2d_plan.md): DD2D collector core `collect_problem`.

The exact per-problem contract (stop-at-first-success, drop-unsolvable, exact-stratum
rejection, determinism) is tested with injected fakes so it is deterministic and free of
the planner/refiner stack. One small real problem is generated once and reused
(injected) so the record-building path exercises real geometry/crops cheaply.

(Coordinator / splits / manifest tests arrive with Step 3.)
"""

from __future__ import annotations

import json
import os

import pytest

from alphatamp.approaches.spectre.envs.dd2d.dd2d import collect
from alphatamp.approaches.spectre.envs.dd2d.dd2d.collect import (
    DD2DCollectConfig,
    collect_problem,
)
from alphatamp.approaches.spectre.envs.dd2d.dd2d.problem import generate_dd2d_problem
from alphatamp.approaches.spectre.envs.dd2d.refine import RefineResult
from alphatamp.approaches.spectre.envs.dd2d.skeleton import Skeleton

CFG = DD2DCollectConfig()


@pytest.fixture(scope="module")
def problem():
    # small + certify=False keeps it fast; reused across tests via the `problem=` seam.
    return generate_dd2d_problem(lam=0.8, seed=0, n_items=9, crowd=0, certify=False)


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakePlanner:
    name = "fake-astar-dist"

    def __init__(self, skeletons):
        self._skeletons = skeletons
        self.calls = 0

    def plan(self, problem, k):
        self.calls += 1
        return list(self._skeletons[:k])


def _sk(*action_tuples) -> Skeleton:
    return Skeleton.from_action_tuples(action_tuples)


def _res(feasible: bool, sk: Skeleton) -> RefineResult:
    return RefineResult(
        status="feasible" if feasible else "infeasible",
        steps_bound=sk.length if feasible else 0,
        plan_length=sk.length,
        n_attempts=1,
        failure_action=None if feasible else str(sk.actions[0]),
    )


# --------------------------------------------------------------------------- #
# 1. stop-at-first-success & the persisted record set
# --------------------------------------------------------------------------- #
def test_full_pool_refines_all_and_keeps_all(problem, tmp_path):
    # full_pool=True (default): refine EVERY plan, keep all pos+neg (no early cutoff, no
    # length confound). sk1 and sk3 feasible -> 4 records labelled [F,T,F,T].
    sks = [
        _sk(("retrieve", "target")),
        _sk(("pick", "o0"), ("place-buffer", "o0"), ("retrieve", "target")),
        _sk(("pick", "o1"), ("place-buffer", "o1"), ("retrieve", "target")),
        _sk(("pick", "o2"), ("place-buffer", "o2"), ("retrieve", "target")),
    ]
    feas = {sks[1].key(), sks[3].key()}

    def refine_fn(sk, scene, seed):
        return _res(sk.key() in feas, sk)

    split_dir = tmp_path / "train"
    res = collect_problem(
        seed=0,
        stratum=problem.min_feasible_subset,
        config=CFG,
        split_dir=str(split_dir),
        planner=FakePlanner(sks),
        refine_fn=refine_fn,
        problem=problem,
    )
    assert res.kept and res.reason == "solved"
    assert res.n_refined == 4  # ALL refined (no cutoff)
    assert [e.label for e in res.examples] == [False, True, False, True]
    assert res.n_pos == 2 and res.n_neg == 2  # multiple positives kept
    assert res.first_feasible_rank == 2
    pdir = split_dir / problem.problem_id
    assert sorted(p.name for p in pdir.glob("*.json")) == [
        f"{i:03d}.json" for i in range(4)
    ]
    assert (pdir / "images").is_dir()


def test_legacy_stop_at_first_success(problem, tmp_path):
    # full_pool=False: legacy stop-at-first-success (1 positive + preceding negatives).
    cfg = DD2DCollectConfig(full_pool=False)
    sk0 = _sk(("retrieve", "target"))
    sk1 = _sk(("pick", "o0"), ("place-buffer", "o0"), ("retrieve", "target"))
    sk2 = _sk(
        ("pick", "o1"), ("place-buffer", "o1"), ("retrieve", "target")
    )  # feasible
    sk3 = _sk(
        ("pick", "o2"), ("place-buffer", "o2"), ("retrieve", "target")
    )  # must NOT refine
    outcomes = {sk0.key(): False, sk1.key(): False, sk2.key(): True}

    def refine_fn(sk, scene, seed):
        return _res(outcomes[sk.key()], sk)  # KeyError if sk3 is ever refined

    res = collect_problem(
        seed=0,
        stratum=problem.min_feasible_subset,
        config=cfg,
        split_dir=str(tmp_path / "train"),
        planner=FakePlanner([sk0, sk1, sk2, sk3]),
        refine_fn=refine_fn,
        problem=problem,
    )
    assert (
        res.kept and res.n_refined == 3 and res.first_feasible_rank == 3
    )  # sk3 untouched
    assert [e.label for e in res.examples] == [False, False, True]
    assert res.n_pos == 1 and res.n_neg == 2


# --------------------------------------------------------------------------- #
# 2. drop-unsolvable: nothing kept, nothing written
# --------------------------------------------------------------------------- #
def test_drop_unsolvable(problem, tmp_path):
    sks = [_sk(("pick", f"o{i}"), ("retrieve", "target")) for i in range(3)]

    def refine_fn(sk, scene, seed):
        return _res(False, sk)

    split_dir = tmp_path / "train"
    res = collect_problem(
        seed=0,
        stratum=problem.min_feasible_subset,
        config=CFG,
        split_dir=str(split_dir),
        planner=FakePlanner(sks),
        refine_fn=refine_fn,
        problem=problem,
    )
    assert not res.kept and res.reason == "unsolved"
    assert res.examples == [] and res.n_refined == 3
    assert not (split_dir / problem.problem_id).exists()  # nothing persisted


# --------------------------------------------------------------------------- #
# 3. exact-stratum rejection: wrong stratum -> drop before planning/refining
# --------------------------------------------------------------------------- #
def test_exact_stratum_rejection(problem, tmp_path):
    planner = FakePlanner([_sk(("retrieve", "target"))])
    refined = {"n": 0}

    def refine_fn(sk, scene, seed):
        refined["n"] += 1
        return _res(True, sk)

    split_dir = tmp_path / "train"
    res = collect_problem(
        seed=0,
        stratum=problem.min_feasible_subset + 1,
        config=CFG,
        split_dir=str(split_dir),
        planner=planner,
        refine_fn=refine_fn,
        problem=problem,
    )
    assert not res.kept and res.reason == "wrong_stratum"
    assert planner.calls == 0 and refined["n"] == 0  # never planned or refined
    assert not (split_dir / problem.problem_id).exists()


# --------------------------------------------------------------------------- #
# 4. determinism: same problem + deterministic fakes -> identical records
# --------------------------------------------------------------------------- #
def test_determinism(problem, tmp_path):
    sk0 = _sk(("pick", "o0"), ("retrieve", "target"))
    sk1 = _sk(("pick", "o1"), ("place-buffer", "o1"), ("retrieve", "target"))
    outcomes = {sk0.key(): False, sk1.key(): True}

    def refine_fn(sk, scene, seed):
        return _res(outcomes[sk.key()], sk)

    # Same split (provenance.split reflects it, correctly); the 2nd run overwrites the 1st's
    # files harmlessly -- we compare the in-memory records, which must be identical.
    def run():
        return collect_problem(
            seed=0,
            stratum=problem.min_feasible_subset,
            config=CFG,
            split_dir=str(tmp_path / "train"),
            planner=FakePlanner([sk0, sk1]),
            refine_fn=refine_fn,
            problem=problem,
        )

    r1, r2 = run(), run()
    assert r1.kept and r2.kept
    assert [e.to_json() for e in r1.examples] == [e.to_json() for e in r2.examples]


def test_stable_seed_helpers_deterministic():
    assert collect._stable_seed(("x",)) == collect._stable_seed(("x",))
    assert all(10 <= collect._sample_n_items(s) <= 13 for s in range(200))


# --------------------------------------------------------------------------- #
# Step 3: coordinator (balanced strata, splits, manifest)
# --------------------------------------------------------------------------- #
def test_stratum_targets_balanced():
    assert collect._stratum_targets(400) == [134, 133, 133]
    assert collect._stratum_targets(100) == [34, 33, 33]
    assert collect._stratum_targets(3) == [1, 1, 1]
    assert collect._stratum_targets(2) == [1, 1, 0]
    for total in (7, 50, 99, 400):
        assert sum(collect._stratum_targets(total)) == total


def test_split_and_stratum_bands_disjoint():
    bands = collect._split_bands(1_000_000)
    intervals = sorted(bands.values())
    for (a0, a1), (b0, b1) in zip(intervals, intervals[1:]):
        assert a1 <= b0  # splits disjoint
    # per-stratum sub-bands are disjoint and cover the split band
    sub = collect._stratum_bands(bands["train"], 3)
    assert sub[0][0] == 0 and sub[-1][1] == 1_000_000
    for (a0, a1), (b0, b1) in zip(sub, sub[1:]):
        assert a1 == b0


def _fake_task_factory():
    """Fake pool task: even seeds solve (write 1 pos + 2 neg dummy json), odd drop."""

    def fake_task(args):
        seed, stratum, config, split_dir = args
        pid = f"dd2d_n11_l80_c5dc_s{seed}"
        if seed % 2 == 0:
            pdir = os.path.join(split_dir, pid)
            os.makedirs(pdir, exist_ok=True)
            for i in range(3):
                collect._atomic_write(os.path.join(pdir, f"{i:03d}.json"), "{}")
            return collect.ProblemResult(
                problem_id=pid,
                seed=seed,
                stratum=stratum,
                n_items=11,
                kept=True,
                reason="solved",
                n_refined=3,
                n_pos=1,
                n_neg=2,
            )
        return collect.ProblemResult(
            problem_id=pid,
            seed=seed,
            stratum=stratum,
            n_items=11,
            kept=False,
            reason="unsolved",
            n_refined=3,
        )

    return fake_task


def test_collect_split_balanced_manifest_and_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "_collect_task", _fake_task_factory())
    # 4 strata (0,1,2,3): band (0,400) -> sub-bands [0,100)/[100,200)/[200,300)/[300,400);
    # even seeds start each -> instant fill.
    summary = collect.collect_split(
        "train", (0, 400), target=4, config=CFG, workers=1, out_root=str(tmp_path)
    )

    assert [summary["strata"][str(s)]["kept"] for s in (0, 1, 2, 3)] == [1, 1, 1, 1]
    assert (
        summary["overall"]["kept"] == 4 and summary["overall"]["neg_pos_ratio"] == 2.0
    )

    train = tmp_path / "train"
    dirs = sorted(d.name for d in train.iterdir() if d.is_dir())
    assert len(dirs) == 4  # manifest kept == #dirs on disk
    for d in dirs:
        assert sorted(p.name for p in (train / d).glob("*.json")) == [
            "000.json",
            "001.json",
            "002.json",
        ]
    assert sorted(summary["seeds_used"]) == [
        0,
        100,
        200,
        300,
    ]  # first seed of each sub-band
    manifest = json.loads((train / "manifest.json").read_text())
    assert manifest["overall"]["kept"] == 4 and manifest["overall"]["n_pos"] == 4


def _make_kept_dir(split_dir, seed):
    """Create a fake kept problem dir (valid problem_id -> parseable seed)."""
    pid = f"dd2d_n11_l80_c5dc_s{seed}"
    os.makedirs(os.path.join(split_dir, pid), exist_ok=True)
    collect._atomic_write(os.path.join(split_dir, pid, "000.json"), "{}")
    return pid


def test_truncate_to_targets(tmp_path):
    # band (0,400) -> sub-bands [0,100)/[100,200)/[200,300)/[300,400); sub-target 2 each.
    split_dir = str(tmp_path / "train")
    os.makedirs(split_dir)
    seeds = {
        0: [0, 1, 2],  # over: keep 0,1  drop 2
        1: [100, 101],  # exact: keep both
        2: [200],  # under: keep the one (no-op)
        3: [303, 300, 301, 302],  # over + unordered: keep 300,301 (lowest 2)
    }
    for s, ss in seeds.items():
        for sd in ss:
            _make_kept_dir(split_dir, sd)

    strata = (0, 1, 2, 3)
    sub_bands = collect._stratum_bands((0, 400), 4)
    survivors = collect._truncate_to_targets(split_dir, sub_bands, [2, 2, 2, 2], strata)

    # exact per-stratum counts, lowest-seed survivors, extras rmtree'd
    assert survivors[0] == ["dd2d_n11_l80_c5dc_s0", "dd2d_n11_l80_c5dc_s1"]
    assert survivors[2] == ["dd2d_n11_l80_c5dc_s200"]
    assert survivors[3] == ["dd2d_n11_l80_c5dc_s300", "dd2d_n11_l80_c5dc_s301"]
    on_disk = {d.name for d in (tmp_path / "train").iterdir() if d.is_dir()}
    assert "dd2d_n11_l80_c5dc_s2" not in on_disk  # dropped
    assert (
        "dd2d_n11_l80_c5dc_s302" not in on_disk
        and "dd2d_n11_l80_c5dc_s303" not in on_disk
    )
    assert len(on_disk) == 2 + 2 + 1 + 2  # exact
    # idempotent
    again = collect._truncate_to_targets(split_dir, sub_bands, [2, 2, 2, 2], strata)
    assert again == survivors


def test_collect_split_truncates_overshoot_to_exact(tmp_path, monkeypatch):
    # A prior (pre-cap) run left 4 s0 dirs; --resume must finalize s0 to EXACTLY its
    # sub-target (1) by truncation, while the other strata collect their 1 each.
    train = tmp_path / "train"
    train.mkdir()
    for sd in (0, 2, 4, 6):  # even seeds in the s0 sub-band [0,100)
        _make_kept_dir(str(train), sd)

    monkeypatch.setattr(collect, "_collect_task", _fake_task_factory())  # even=keep
    summary = collect.collect_split(
        "train",
        (0, 400),
        target=4,  # -> [1,1,1,1]
        config=CFG,
        workers=1,
        out_root=str(tmp_path),
        resume=True,
        progress=False,
    )

    assert [summary["strata"][str(s)]["kept"] for s in (0, 1, 2, 3)] == [1, 1, 1, 1]
    assert summary["overall"]["kept"] == 4
    dirs = {d.name for d in train.iterdir() if d.is_dir()}
    assert len(dirs) == 4  # exact on disk, no overshoot
    assert "dd2d_n11_l80_c5dc_s0" in dirs  # lowest-seed s0 survivor kept
    extras = {f"dd2d_n11_l80_c5dc_s{sd}" for sd in (2, 4, 6)}
    assert not (extras & dirs)  # over-target s0 dirs truncated


def test_collect_split_writes_only_kept_dirs(tmp_path, monkeypatch):
    def fake_task(args):
        seed, stratum, config, split_dir = args
        pid = f"dd2d_n11_l80_c5dc_s{seed}"
        if seed % 2 == 0:
            pdir = os.path.join(split_dir, pid)
            os.makedirs(pdir, exist_ok=True)
            collect._atomic_write(os.path.join(pdir, "000.json"), "{}")
            return collect.ProblemResult(
                pid,
                seed,
                stratum,
                11,
                kept=True,
                reason="solved",
                n_refined=1,
                n_pos=1,
                n_neg=0,
            )
        return collect.ProblemResult(
            pid, seed, stratum, 11, kept=False, reason="unsolved"
        )

    monkeypatch.setattr(collect, "_collect_task", fake_task)
    # band (1,401): every stratum sub-band starts at an odd seed -> seed(odd) drops, next even kept
    summary = collect.collect_split(
        "test", (1, 401), target=4, config=CFG, workers=1, out_root=str(tmp_path)
    )
    test = tmp_path / "test"
    assert len([d for d in test.iterdir() if d.is_dir()]) == 4  # one kept per stratum
    assert summary["strata"]["0"]["attempted"] == 2  # seed1 dropped, seed2 kept
    assert not (test / "dd2d_n11_l80_c5dc_s1").exists()  # dropped odd seed -> no dir


# --------------------------------------------------------------------------- #
# Step 5: resume + progress
# --------------------------------------------------------------------------- #
def test_resume_skips_kept_and_logged(tmp_path, monkeypatch):
    submitted: list[int] = []

    def fake_task(args):
        seed, stratum, config, split_dir = args
        submitted.append(seed)
        pid = f"dd2d_n11_l80_c5dc_s{seed}"
        # every 3rd seed solves; others drop -> exercises both kept-dir + attempted.log skip
        if seed % 3 == 0:
            os.makedirs(os.path.join(split_dir, pid), exist_ok=True)
            collect._atomic_write(os.path.join(split_dir, pid, "000.json"), "{}")
            return collect.ProblemResult(
                pid,
                seed,
                stratum,
                11,
                kept=True,
                reason="solved",
                n_refined=1,
                n_pos=1,
                n_neg=0,
            )
        return collect.ProblemResult(
            pid, seed, stratum, 11, kept=False, reason="unsolved"
        )

    monkeypatch.setattr(collect, "_collect_task", fake_task)
    # first pass: single stratum, small band, fill target 2
    s1 = collect.collect_split(
        "train",
        (0, 300),
        target=2,
        config=CFG,
        workers=1,
        out_root=str(tmp_path),
        strata=(1,),
        progress=False,
    )
    assert s1["strata"]["1"]["kept"] == 2
    first_submitted = list(submitted)
    submitted.clear()

    # resume: previously kept + logged-dropped seeds must NOT be re-submitted
    s2 = collect.collect_split(
        "train",
        (0, 300),
        target=2,
        config=CFG,
        workers=1,
        out_root=str(tmp_path),
        strata=(1,),
        progress=False,
        resume=True,
    )
    assert not set(submitted) & set(first_submitted)  # no seed re-attempted
    # already at target from recovered kept -> nothing new needed
    assert s2["strata"]["1"]["kept"] == 2


def test_progress_line_emits(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(collect, "_collect_task", _fake_task_factory())
    collect.collect_split(
        "train",
        (0, 300),
        target=3,
        config=CFG,
        workers=1,
        out_root=str(tmp_path),
        progress=True,
        progress_every=0.0,
    )
    out = capsys.readouterr().out
    assert "[train]" in out and "kept" in out and "ETA" in out
