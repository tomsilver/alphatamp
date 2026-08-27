"""Restock3D-**v3** budget-calibration sweep: how low can ``num_sampling_attempts_per_step``
(samples/step) and ``r_cap`` (refinement_timeout_s) go before feasible skeletons stop refining?

Motivation: the real-refiner v3 collection is slow because both budgets are high (samples=18;
r_cap 50/70/90/110 s per stratum). Both are **pure runtime knobs** on kinder's
``BacktrackingRefiner`` (samples -> the per-abstract-step sampling loop; r_cap -> the ``timeout``
arg), so we can measure their effect without re-collecting or retraining.

The already-collected v3 dataset is **synthetic** (analytic labels from
``feasibility_v3.classify_skeleton`` -- pure geometry, no motion planning), so a labelled
"success" can be an analytic **false positive** the real refiner can't realise.

**One fused task per problem** (this is deliberate -- an earlier two-phase design re-recovered the
gold skeleton in a *second* process, where the hash-dependent, K_max-truncated pool order dropped
it and a silent fallback measured the *wrong* skeleton). Each task, in a single process:

1. recreates the pid scene, regenerates the deterministic pool, and picks the dataset's
   analytic-success skeleton(s);
2. refines the first such candidate at the **top** of the samples grid (=18). The first that
   reaches the goal is the problem's **gold** skeleton (analytic false positives fail here and are
   skipped). "gold" and "succeeds at samples=18" are thus the *same* measurement -- no cross-phase
   contradiction is possible;
3. re-refines that **same** ``(state_plan, action_plan)`` object across the rest of the grid
   (1 deterministic seed each), timing every refine;
4. optionally (``--with-infeasible``, default on) times one analytic-infeasible skeleton across the
   grid too -- where the r_cap saving actually accrues (feasible-success time is ~flat; the many
   infeasible candidates exhaust / run to r_cap).

Everything is **append-only, single-writer, resumable**: re-running the identical command is the
resume path (every attempted pid is logged to ``goldA_attempts_r{s}.jsonl`` -- written *last* as a
commit marker -- and skipped next time). ``--report-only`` rebuilds the markdown + figures from
disk. Workers are sized by ``min(0.85*CPU, 0.85*freeRAM/per_worker)`` (whichever caps first) with a
free-RAM watchdog; strata run **sequentially** (one sized pool each) so per-worker RAM is
predictable.

Run (repo root, venv active)::

    # smoke on the cheapest stratum:
    python experiments/spectre/restock3d_v3_budget_sweep.py --strata 0 --gold-target 2 --samples 8,18
    # full run in the background (stratum 3 at r_cap 110 s dominates):
    bash experiments/spectre/spectre_run.sh v3_budget_sweep \
        python experiments/spectre/restock3d_v3_budget_sweep.py --strata 0,1,2,3 --gold-target 20
    # rebuild report/figures from an existing sweep_results.jsonl:
    python experiments/spectre/restock3d_v3_budget_sweep.py --report-only
"""

from __future__ import annotations

# --- single-thread BLAS per worker (clean N-way parallelism + honest wall-clock) BEFORE numpy.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# --- IKFast needs static LAPACK/BLAS; shim the shared libs (once, cached afterwards). ----------
import glob
import pathlib

_B = os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
os.environ.setdefault("LAPACK_DIR", _B)
os.environ.setdefault("BLAS_DIR", _B)
pathlib.Path(_B).mkdir(parents=True, exist_ok=True)
for _a, (_sd, _pt) in {
    "liblapack.a": ("lapack", "liblapack.so.3*"),
    "libblas.a": ("blas", "libblas.so.3*"),
}.items():
    _lk = pathlib.Path(_B) / _a
    if not (_lk.exists() or _lk.is_symlink()):
        _cs = sorted(
            glob.glob(f"/usr/lib/x86_64-linux-gnu/{_sd}/{_pt}")
            + glob.glob(f"/usr/lib/x86_64-linux-gnu/{_pt}")
        )
        _r = next((c for c in _cs if os.path.isfile(c)), None)
        if _r:
            _lk.symlink_to(_r)

import argparse
import gzip
import itertools
import json
import multiprocessing as mp
import pickle
import statistics as stt
import subprocess
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import kinder
import psutil
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph

from alphatamp.approaches.spectre import collect as C
from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.env_registry import register_extra_envs
from alphatamp.approaches.spectre.envs.restock3d import feasibility_v3 as F
from alphatamp.approaches.spectre.envs.restock3d import strata_v3 as S

_RAW = Path("data/spectre/raw/restock3d_v3/train/episodes")
_OUT = Path("data/spectre/derived/restock3d_v3/budget_sweep")
_N_BY_STRATUM = {0: 6, 1: 7, 2: 8, 3: 9}
_DEFAULT_SAMPLES = [4, 6, 8, 10, 12, 14, 16, 18]
_HEARTBEAT_S = 30.0

# Conservative per-worker RSS estimate for the SWEEP (GB), calibrated from the first live run's
# wRSSmax (n=8 ~5.6 GB) and restock3d_v3_run_all.sh's worker counts. Over-sizes safely; the
# wRSSmax heartbeat validates them live.
_PER_WORKER_GB = {0: 1.5, 1: 3.0, 2: 5.6, 3: 8.0}


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------
def _steps_of(action_plan) -> tuple:
    """Canonical (op_name, (arg_names,)) tuple for a ground-operator sequence."""
    return tuple((op.name, tuple(p.name for p in op.parameters)) for op in action_plan)


def _steps_json(action_plan) -> list:
    return [[op.name, [p.name for p in op.parameters]] for op in action_plan]


def _load_episode(pid: int):
    path = _RAW / f"ep_{pid:05d}.pkl.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def _stratum_episode_pids(stratum: int) -> list[int]:
    """Train problem ids for a stratum, in seed order, restricted to files present."""
    pids = []
    for path in sorted(_RAW.glob("ep_*.pkl.gz")):
        pid = int(path.name[len("ep_") :].split(".")[0])
        if S.stratum_of(pid) == stratum:
            pids.append(pid)
    return sorted(pids)


def _fresh_bpg(x0, s0) -> BilevelPlanningGraph:
    b = BilevelPlanningGraph()
    b.add_abstract_state_node(s0)
    b.add_state_node(x0)
    b.add_state_abstractor_edge(x0, s0)
    return b


def _cfg(stratum: int, pid: int, samples: int, r_cap: float, k_max: int) -> CollectionConfig:
    """One CollectionConfig for the real v3 refiner path (mirrors restock3d_v3_demos)."""
    return CollectionConfig(
        env_id=S.env_id(stratum),
        env_variant="restock3d_v3",
        model_name="restock3d_v3",
        model_kwargs={"stratum": stratum},
        split="train",
        num_problems=1,
        problem_seed_start=pid,
        problem_seed_end=pid + 1,
        K_max=k_max,
        abstract_plan_timeout_s=120.0,
        refinement_timeout_s=r_cap,
        num_sampling_attempts_per_step=samples,
        max_trajectory_steps=500,
        plan_generator="closed_form",
        refiner_mode="real",
    )


def _open_problem(cfg: CollectionConfig, pid: int):
    """Recreate the pid scene + models (mirrors restock3d_v3_gates.gate_g1). Caller closes
    env + sim. Returns (env, sim, em, obs, x0, s0, goal)."""
    register_extra_envs()
    env = kinder.make(cfg.env_id)
    obs, _ = env.reset(seed=pid)
    em = C._make_env_models(cfg, env.observation_space, env.action_space)
    sim = C._restock_extras["sim"]
    x0 = em.observation_to_state(obs)
    s0 = em.state_abstractor(x0)
    goal = em.goal_deriver(x0)
    return env, sim, em, obs, x0, s0, goal


def _regen_pool(cfg, em, obs, pid, x0, s0, goal, k_max):
    """Materialise the deterministic geometry-guided pool. Returns (pool, {steps_key: (sp, ap)})."""
    gen = C._make_plan_generator(cfg, em, obs, pid, x0)
    bpg = _fresh_bpg(x0, s0)
    pool = list(
        itertools.islice(gen(x0, s0, goal, cfg.abstract_plan_timeout_s, bpg), k_max)
    )
    by_steps = {_steps_of(ap): (sp, ap) for sp, ap in pool}
    return pool, by_steps


def _refine_once(
    stratum, pid, k_max, r_cap, samples, cfg_base, em, obs, x0, s0, sp, ap, seed_idx, goal_atoms
) -> tuple[bool, float]:
    """One real refinement of (sp, ap) at ``samples`` attempts/step, timed. Fresh sampler +
    refiner + bpg so each samples-value is measured independently (same deterministic seed).

    ``goal_atoms`` not None -> also assert the goal actually holds; None -> success is just
    ``plan is not None`` (used for the infeasible probe)."""
    sampler = C._make_trajectory_sampler(cfg_base, em)
    cfg_s = _cfg(stratum, pid, samples, r_cap, k_max)
    seed = C._refinement_seed(cfg_base.refinement_seed_rule, pid, seed_idx)
    refiner = C._make_refiner(cfg_s, obs, sampler, seed)
    if hasattr(sampler, "clear"):
        sampler.clear()
    bpg = _fresh_bpg(x0, s0)
    t0 = time.perf_counter()
    try:
        plan = refiner(x0, sp, ap, r_cap, bpg)
    except BaseException:  # noqa: BLE001 — a refiner crash counts as a failed refine
        plan = None
    dt = time.perf_counter() - t0
    ok = plan is not None
    if ok and goal_atoms is not None:
        ok = set(goal_atoms).issubset(em.state_abstractor(plan.states[-1]).atoms)
    return bool(ok), float(dt)


# --------------------------------------------------------------------------------------------
# Fused worker (top-level, picklable for spawn): verify gold @ top-of-grid, then sweep it.
# --------------------------------------------------------------------------------------------
def _sweep_task(task) -> dict:
    stratum, pid, k_max, r_cap, samples, with_infeasible = task
    ep = _load_episode(pid)
    if ep is None or getattr(ep.summary, "num_success", 0) == 0:
        return {"stratum": stratum, "pid": pid, "gold": False, "reason": "no_success"}
    cfg = _cfg(stratum, pid, max(samples), r_cap, k_max)
    top = max(samples)
    env = sim = None
    try:
        env, sim, em, obs, x0, s0, goal = _open_problem(cfg, pid)
        pool, by_steps = _regen_pool(cfg, em, obs, pid, x0, s0, goal, k_max)
        goal_atoms = set(ep.goal_atoms)

        # candidate feasible skeletons: dataset analytic-successes matched by op-sequence (idx
        # order); else live-feasible in pool order. Recovery is in-process, so it is consistent
        # with the sweep that follows.
        candidates: list[tuple[int, object, object]] = []
        for idx in sorted(o.skeleton_idx for o in ep.outcomes if o.outcome == "success"):
            key = _steps_of(ep.skeleton_pool[idx].operator_seq)
            if key in by_steps:
                sp, ap = by_steps[key]
                candidates.append((idx, sp, ap))
        if not candidates:
            dims, pos = C._restock3d_analytic_inputs(x0)
            for i, (sp, ap) in enumerate(pool):
                if F.classify_skeleton(list(_steps_of(ap)), dims, pos) is None:
                    candidates.append((i, sp, ap))

        # gold = first candidate that reaches the goal at the top budget (this IS the gold gate).
        gold = None
        n_fp = 0
        for idx, sp, ap in candidates:
            ok, dt = _refine_once(
                stratum, pid, k_max, r_cap, top, cfg, em, obs, x0, s0, sp, ap, idx, goal_atoms
            )
            if ok:
                gold = (idx, sp, ap, dt)
                break
            n_fp += 1
        if gold is None:
            return {"stratum": stratum, "pid": pid, "gold": False,
                    "reason": "false_positive", "n_fp": n_fp}

        g_idx, g_sp, g_ap, top_dt = gold
        feasible = [{"samples": top, "success": True, "dt": round(top_dt, 3)}]
        for s in sorted(samples):
            if s == top:
                continue
            ok, dt = _refine_once(
                stratum, pid, k_max, r_cap, s, cfg, em, obs, x0, s0, g_sp, g_ap, g_idx, goal_atoms
            )
            feasible.append({"samples": s, "success": ok, "dt": round(dt, 3)})
        feasible.sort(key=lambda r: r["samples"])

        infeasible = []
        if with_infeasible:
            dims, pos = C._restock3d_analytic_inputs(x0)
            inf = next(
                ((i, sp, ap) for i, (sp, ap) in enumerate(pool)
                 if F.classify_skeleton(list(_steps_of(ap)), dims, pos) is not None),
                None,
            )
            if inf is not None:
                i_idx, i_sp, i_ap = inf
                for s in sorted(samples):
                    ok, dt = _refine_once(
                        stratum, pid, k_max, r_cap, s, cfg, em, obs, x0, s0, i_sp, i_ap, i_idx, None
                    )
                    infeasible.append({"samples": s, "success": ok, "dt": round(dt, 3)})

        return {"stratum": stratum, "pid": pid, "gold": True, "seed_idx": g_idx,
                "steps": _steps_json(g_ap), "top_dt": round(top_dt, 2), "n_fp": n_fp,
                "feasible": feasible, "infeasible": infeasible}
    except BaseException as exc:  # noqa: BLE001
        return {"stratum": stratum, "pid": pid, "gold": False,
                "reason": f"error:{type(exc).__name__}:{exc}"}
    finally:
        if env is not None:
            env.close()
        if sim is not None and hasattr(sim, "close"):
            sim.close()


# --------------------------------------------------------------------------------------------
# Worker sizing / OOM (mirrors restock3d_v2_collect._sized_workers / _worker_rss_gb)
# --------------------------------------------------------------------------------------------
def _sized_workers(stratum: int, cpu: int, avail_gb: float, mem_floor_gb: float) -> int:
    """min(0.85*CPU, 0.85*freeRAM / per_worker), whichever caps first, floor-guarded."""
    pwg = _PER_WORKER_GB[stratum]
    ram_budget = min(0.85 * avail_gb, avail_gb - (mem_floor_gb + 3.0))
    return max(1, min(int(0.85 * cpu), int(ram_budget / pwg)))


def _worker_rss_gb() -> float:
    """Max RSS (GB) across worker children -- the empirical check on _PER_WORKER_GB."""
    try:
        kids = psutil.Process().children(recursive=True)
        rss = [k.memory_info().rss for k in kids if k.is_running()]
    except Exception:  # noqa: BLE001
        return 0.0
    return max(rss) / 1e9 if rss else 0.0


def _fmt_elapsed(sec: float) -> str:
    sec = int(sec)
    m, s = divmod(sec, 60)
    return f"{m}m{s:02d}s"


# --------------------------------------------------------------------------------------------
# Bounded, RAM-watchdogged process pool
# --------------------------------------------------------------------------------------------
def _run_pool(source_next, source_done, on_result, worker_fn, n_workers, log, label,
              mem_floor, t_start, progress) -> None:
    """Drive a spawn ProcessPoolExecutor: pull tasks from ``source_next(inflight_ct)`` (None = hold),
    call ``on_result`` per completion, pause submissions under low RAM, emit a heartbeat/ETA.
    ``source_done()`` True == nothing more will ever be submitted."""
    ctx = mp.get_context("spawn")
    inflight: dict = {}
    paused = [False]
    last_hb = [time.perf_counter()]

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:

        def fill() -> None:
            while len(inflight) < n_workers and not paused[0]:
                task = source_next(len(inflight))
                if task is None:
                    break
                inflight[ex.submit(worker_fn, task)] = task

        def heartbeat() -> None:
            avail = psutil.virtual_memory().available / 1e9
            wmax = _worker_rss_gb()
            done, total = progress()
            el = time.perf_counter() - t_start
            eta = (el / done * (total - done)) if done else 0.0
            log(
                f"  [train] {_fmt_elapsed(el)} | kept {done}/{total}  ({label}) | "
                f"inflight {len(inflight)} freeRAM {avail:.1f}GB wRSSmax {wmax:.1f}GB | "
                f"ETA {eta / 60:.0f}m"
            )

        fill()
        while inflight or not source_done():
            if not inflight:  # pending work but nothing running: paused or transiently idle
                time.sleep(1.0)
                avail = psutil.virtual_memory().available / 1e9
                if paused[0] and avail > mem_floor + 3.0:
                    paused[0] = False
                    log(f"  {label}: RAM recovered ({avail:.1f}GB) -- resuming")
                fill()
                continue
            done_set, _ = wait(list(inflight), timeout=_HEARTBEAT_S, return_when=FIRST_COMPLETED)
            for fut in done_set:
                task = inflight.pop(fut)
                try:
                    res = fut.result()
                except BaseException as exc:  # noqa: BLE001
                    res = {"error": f"{type(exc).__name__}: {exc}",
                           "pid": task[1] if len(task) > 1 else None,
                           "stratum": task[0] if task else None, "gold": False}
                on_result(res)
            avail = psutil.virtual_memory().available / 1e9
            if avail < mem_floor:
                if not paused[0]:
                    log(f"  {label}: low RAM {avail:.1f}GB < {mem_floor}GB -- pausing submissions")
                paused[0] = True
            elif paused[0] and avail > mem_floor + 3.0:
                log(f"  {label}: RAM recovered ({avail:.1f}GB) -- resuming")
                paused[0] = False
            if not paused[0]:
                fill()
            now = time.perf_counter()
            if now - last_hb[0] >= _HEARTBEAT_S or not inflight:
                heartbeat()
                last_hb[0] = now


# --------------------------------------------------------------------------------------------
# Per-stratum driver (single-writer, append-only, resumable)
# --------------------------------------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn final line from a crash -- discard
    return out


def _run_stratum(stratum, target, samples_grid, with_infeasible, log, mem_floor, workers_override) -> None:
    k_max, r_cap = S.budget(stratum)
    att_path = _OUT / f"goldA_attempts_r{stratum}.jsonl"
    gold_path = _OUT / f"gold_r{stratum}.jsonl"
    sweep_path = _OUT / "sweep_results.jsonl"
    attempted = {r["pid"] for r in _load_jsonl(att_path)}
    gold_pids = {r["pid"] for r in _load_jsonl(att_path) if r.get("gold")}
    if len(gold_pids) >= target:
        log(f"  r{stratum}: {len(gold_pids)} gold on disk (>= {target}); skip")
        return
    pids = [p for p in _stratum_episode_pids(stratum) if p not in attempted]
    log(f"  r{stratum}: {len(gold_pids)}/{target} gold; {len(pids)} pids to try "
        f"(k_max={k_max}, r_cap={r_cap}s, infeasible={'on' if with_infeasible else 'off'})")
    it = iter(pids)
    state = {"exhausted": False, "gold": len(gold_pids)}
    att_fh = open(att_path, "a", encoding="utf-8")
    gold_fh = open(gold_path, "a", encoding="utf-8")
    sweep_fh = open(sweep_path, "a", encoding="utf-8")

    def source_next(inflight_ct):
        if state["gold"] >= target or state["gold"] + inflight_ct >= target:
            return None
        try:
            return (stratum, next(it), k_max, r_cap, tuple(samples_grid), with_infeasible)
        except StopIteration:
            state["exhausted"] = True
            return None

    def source_done():
        return state["gold"] >= target or state["exhausted"]

    def on_result(res):
        pid = res.get("pid")
        if res.get("gold"):
            # sweep rows + gold entry FIRST, then the attempts commit-marker LAST (so a crash
            # mid-write leaves an un-committed pid that resume re-runs; report de-dups any overlap).
            gold_fh.write(json.dumps({
                "pid": pid, "seed_idx": res["seed_idx"],
                "steps": res.get("steps"), "top_dt": res.get("top_dt"),
            }) + "\n")
            gold_fh.flush()
            for row in res.get("feasible", []):
                sweep_fh.write(json.dumps({
                    "kind": "feasible", "stratum": stratum, "pid": pid,
                    "seed_idx": res.get("seed_idx"), "samples": row["samples"],
                    "success": row["success"], "dt": row["dt"],
                }) + "\n")
            for row in res.get("infeasible", []):
                sweep_fh.write(json.dumps({
                    "kind": "infeasible", "stratum": stratum, "pid": pid,
                    "samples": row["samples"], "success": row["success"], "dt": row["dt"],
                }) + "\n")
            sweep_fh.flush()
            state["gold"] += 1
        att_fh.write(json.dumps({
            "pid": pid, "gold": res.get("gold", False), "reason": res.get("reason"),
            "seed_idx": res.get("seed_idx"), "n_fp": res.get("n_fp"), "top_dt": res.get("top_dt"),
        }) + "\n")
        att_fh.flush()
        if res.get("gold"):
            succ = [r["samples"] for r in res.get("feasible", []) if r["success"]]
            log(f"  r{stratum} GOLD pid={pid} ({state['gold']}/{target}, {res.get('n_fp')} FP, "
                f"top_dt={res.get('top_dt')}s, succeeds at {succ})")
        else:
            log(f"  r{stratum} pid={pid} not gold: {res.get('reason')}")

    workers = workers_override or _sized_workers(
        stratum, os.cpu_count() or 4, psutil.virtual_memory().available / 1e9, mem_floor)
    log(f"  r{stratum} workers={workers}")
    _run_pool(source_next, source_done, on_result, _sweep_task, workers, log,
              f"r{stratum}", mem_floor, time.perf_counter(),
              lambda: (state["gold"], target))
    att_fh.close()
    gold_fh.close()
    sweep_fh.close()
    if state["gold"] < target:
        log(f"  r{stratum}: only {state['gold']}/{target} gold (ran out of episodes)")


# --------------------------------------------------------------------------------------------
# Report + figures (derived from sweep_results.jsonl; regenerable, --report-only)
# --------------------------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _load_dedup_rows() -> list[dict]:
    """Load sweep rows, de-duplicated by (stratum, pid, kind, samples) keeping the LAST write
    (so a resumed/re-run problem's fresh rows supersede any crash-partial earlier ones)."""
    latest: dict = {}
    for r in _load_jsonl(_OUT / "sweep_results.jsonl"):
        if r.get("kind") not in ("feasible", "infeasible"):
            continue
        latest[(r["stratum"], r["pid"], r["kind"], r["samples"])] = r
    return list(latest.values())


def _build_report(strata, samples_grid, success_threshold, rcap_margin, log) -> None:
    rows = _load_dedup_rows()
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    pidset: dict = defaultdict(set)
    for r in rows:
        data[r["stratum"]][r["kind"]][r["samples"]].append((bool(r["success"]), float(r["dt"])))
        if r["kind"] == "feasible":
            pidset[r["stratum"]].add(r["pid"])

    grid = sorted(samples_grid)
    lines = [
        "# Restock3D-v3 budget-calibration sweep",
        "",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M', time.gmtime())} UTC · git {_git_sha()}",
        f"- samples grid: {grid} · r_cap ceiling: per-stratum "
        f"({', '.join(f'{s}:{S.budget(s)[1]:.0f}s' for s in sorted(strata))})",
        f"- gold problems / stratum: "
        f"{', '.join(f'{s}:{len(pidset[s])}' for s in sorted(strata))}"
        " (1 deterministic refinement seed per point; gold ⟺ succeeds at samples="
        f"{max(grid)})",
        "",
        "**Feasible skeletons** — each gold plan re-refined at every samples budget. Note the top of"
        " the grid is 1.00 by construction (that refine *is* the gold gate). Wall-clock is over the"
        " successful refines; it is measured **under worker contention**, so near-r_cap times are"
        " noisy.",
        "",
    ]
    for stratum in sorted(strata):
        feas = data[stratum]["feasible"]
        if not feas:
            continue
        n = len(pidset[stratum])
        cur_rcap = S.budget(stratum)[1]
        lines.append(f"## Stratum {stratum} (n={_N_BY_STRATUM[stratum]}), {n} gold problems\n")
        lines.append("| samples | success-rate | success dt median / mean / max (s) |")
        lines.append("|---:|---:|---:|")
        best_samples = None
        for sm in grid:
            trials = feas.get(sm, [])
            if not trials:
                lines.append(f"| {sm} | — | — |")
                continue
            succ = [dt for ok, dt in trials if ok]
            rate = len(succ) / len(trials)
            wtxt = (f"{stt.median(succ):.1f} / {stt.fmean(succ):.1f} / {max(succ):.1f}"
                    if succ else "—")
            lines.append(f"| {sm} | {rate:.2f} ({len(succ)}/{len(trials)}) | {wtxt} |")
            if rate >= success_threshold and best_samples is None:
                best_samples = sm
        if best_samples is not None:
            succ_at = [dt for ok, dt in feas.get(best_samples, []) if ok]
            mx = max(succ_at) if succ_at else None
            rcap = (mx * rcap_margin) if mx is not None else None
            if rcap is None:
                rtxt = "—"
            elif rcap >= cur_rcap:
                rtxt = (f"≈ current {cur_rcap:.0f}s — **at its floor** (feasibles need up to "
                        f"{mx:.1f}s; {mx:.1f}×{rcap_margin} = {rcap:.0f}s ≥ current)")
            else:
                rtxt = (f"**{rcap:.0f}s** (= {mx:.1f}s max feasible-success × {rcap_margin}), "
                        f"down from {cur_rcap:.0f}s")
            lines.append(
                f"\n**Recommendation** — samples ≥ **{best_samples}** keeps success-rate "
                f"≥ {success_threshold:.2f}; r_cap {rtxt} "
                f"(current: samples 18, r_cap {cur_rcap:.0f}s).\n"
            )
        else:
            lines.append(
                f"\n**Recommendation** — no samples level reached success-rate "
                f"≥ {success_threshold:.2f}; keep samples 18.\n"
            )
        infe = data[stratum]["infeasible"]
        if infe:
            lines.append("_Infeasible skeleton time-to-`None` (where the r_cap saving accrues):_\n")
            lines.append("| samples | mean exhaust dt (s) | ran to r_cap? |")
            lines.append("|---:|---:|---:|")
            for sm in grid:
                trials = infe.get(sm, [])
                if not trials:
                    lines.append(f"| {sm} | — | — |")
                    continue
                dts = [dt for _ok, dt in trials]
                hit = sum(1 for dt in dts if dt >= 0.95 * cur_rcap)
                lines.append(f"| {sm} | {stt.fmean(dts):.1f} | {hit}/{len(dts)} |")
            lines.append("")

    md = "\n".join(lines) + "\n"
    _OUT.mkdir(parents=True, exist_ok=True)
    (_OUT / "budget_sweep_results.md").write_text(md, encoding="utf-8")
    log(f"wrote {_OUT / 'budget_sweep_results.md'}")
    _make_figures(data, pidset, sorted(strata), grid, log)


def _make_figures(data, pidset, strata, grid, log) -> None:
    import matplotlib
    matplotlib.use("Agg")  # headless; before pyplot
    import matplotlib.pyplot as plt

    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex", "nature"])
    except Exception:  # noqa: BLE001
        pass
    plt.rcParams.update({
        "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.6, "figure.dpi": 150, "savefig.dpi": 300,
    })

    fig1, ax1 = plt.subplots(figsize=(4.2, 3.0))
    any1 = False
    for stratum in strata:
        feas = data[stratum]["feasible"]
        xs, ys = [], []
        for sm in grid:
            trials = feas.get(sm, [])
            if not trials:
                continue
            xs.append(sm)
            ys.append(sum(1 for ok, _ in trials if ok) / len(trials))
        if xs:
            any1 = True
            ax1.plot(xs, ys, marker="o",
                     label=f"s{stratum} (n={_N_BY_STRATUM[stratum]}, {len(pidset[stratum])}p)")
    ax1.set_xlabel("samples_per_step")
    ax1.set_ylabel("feasible success-rate")
    ax1.set_ylim(-0.03, 1.03)
    ax1.set_title("Restock3D-v3: feasible refine success vs sampling budget")
    if any1:
        ax1.legend(fontsize=7, frameon=False)
    fig1.tight_layout()
    fig1.savefig(_OUT / "success_vs_samples.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    for stratum, ax in zip(strata, axes.ravel()):
        feas = data[stratum]["feasible"]
        infe = data[stratum]["infeasible"]
        xs, med, lo, hi = [], [], [], []
        for sm in grid:
            succ = [dt for ok, dt in feas.get(sm, []) if ok]
            if not succ:
                continue
            xs.append(sm)
            med.append(stt.median(succ))
            lo.append(min(succ))
            hi.append(max(succ))
        if xs:
            ax.plot(xs, med, marker="o", color="C0", label="feasible (median)")
            ax.fill_between(xs, lo, hi, color="C0", alpha=0.18, label="feasible min–max")
        if infe:
            ix = [sm for sm in grid if infe.get(sm)]
            iy = [stt.fmean([dt for _o, dt in infe[sm]]) for sm in ix]
            if ix:
                ax.plot(ix, iy, marker="s", ls="--", color="C3", label="infeasible (mean)")
        ax.axhline(S.budget(stratum)[1], color="gray", ls=":", lw=1,
                   label=f"r_cap {S.budget(stratum)[1]:.0f}s")
        ax.set_title(f"s{stratum} (n={_N_BY_STRATUM[stratum]})")
        ax.set_xlabel("samples_per_step")
        ax.set_ylabel("wall-clock (s)")
        ax.legend(fontsize=6, frameon=False)
    fig2.suptitle("Restock3D-v3: refinement wall-clock vs sampling budget")
    fig2.tight_layout()
    fig2.savefig(_OUT / "wallclock_vs_samples.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)
    log(f"wrote {_OUT / 'success_vs_samples.png'} and {_OUT / 'wallclock_vs_samples.png'}")


# --------------------------------------------------------------------------------------------
def _check_meta(samples_grid, force, log) -> None:
    """Config gate: refuse to silently merge rows collected under a different samples grid / sha."""
    meta_path = _OUT / "_meta.json"
    cur = {"samples": sorted(samples_grid), "git_sha": _git_sha()}
    if meta_path.exists():
        prev = json.loads(meta_path.read_text())
        if prev.get("samples") != cur["samples"] and not force:
            raise SystemExit(
                f"[budget-sweep] samples grid changed ({prev.get('samples')} -> {cur['samples']}); "
                f"existing sweep_results.jsonl is incompatible. Pass --force, or move {_OUT} aside."
            )
        if prev.get("git_sha") != cur["git_sha"]:
            log(f"[budget-sweep] WARNING: git_sha changed ({prev.get('git_sha')} -> "
                f"{cur['git_sha']}); resuming across a code change.")
    meta_path.write_text(json.dumps(cur, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata", default="0,1,2,3")
    ap.add_argument("--gold-target", type=int, default=20)
    ap.add_argument("--samples", default=",".join(str(s) for s in _DEFAULT_SAMPLES))
    ap.add_argument("--with-infeasible", dest="with_infeasible", action="store_true", default=True)
    ap.add_argument("--no-infeasible", dest="with_infeasible", action="store_false")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild markdown + figures from sweep_results.jsonl; no refinement")
    ap.add_argument("--success-threshold", type=float, default=1.0,
                    help="min feasible success-rate the recommended samples must clear")
    ap.add_argument("--rcap-margin", type=float, default=1.5,
                    help="recommended r_cap = max feasible-success dt × this")
    ap.add_argument("--mem-floor-gb", type=float, default=6.0)
    ap.add_argument("--workers", type=int, default=None,
                    help="override per-stratum RAM-sized workers")
    ap.add_argument("--force", action="store_true", help="override the samples-grid meta gate")
    args = ap.parse_args()

    strata = [int(s) for s in args.strata.split(",")]
    samples_grid = sorted(int(s) for s in args.samples.split(","))
    _OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    def log(m: str) -> None:
        print(f"[{time.time() - t_start:7.0f}s] {m}", flush=True)

    if args.report_only:
        log("report-only: rebuilding markdown + figures from sweep_results.jsonl")
        _build_report(strata, samples_grid, args.success_threshold, args.rcap_margin, log)
        return

    _check_meta(samples_grid, args.force, log)
    log(f"budget sweep: strata={strata} gold_target={args.gold_target} samples={samples_grid} "
        f"infeasible={'on' if args.with_infeasible else 'off'} -> {_OUT}")

    for stratum in strata:  # sequential: predictable per-stratum RAM, incremental output
        log(f"=== stratum {stratum} (n={_N_BY_STRATUM[stratum]}) — find gold + sweep ===")
        _run_stratum(stratum, args.gold_target, samples_grid, args.with_infeasible,
                     log, args.mem_floor_gb, args.workers)
        _build_report(strata, samples_grid, args.success_threshold, args.rcap_margin, log)
        log(f"=== stratum {stratum} done; report refreshed ===")

    _build_report(strata, samples_grid, args.success_threshold, args.rcap_margin, log)
    log(f"DONE in {(time.time() - t_start) / 60:.1f} min. See {_OUT}/budget_sweep_results.md")


if __name__ == "__main__":
    main()
