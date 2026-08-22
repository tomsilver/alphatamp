"""Phase 7 — Restock3D-v3 gates (all cheap-ish; any failure blocks collection).

* **G1 — analytic <-> real agreement.** Over a few pilot problems, run each candidate skeleton
  through BOTH the analytic ``feasibility_v3.classify_skeleton`` and the REAL refiner (the exact
  collection/eval pipeline). Report the feasible/infeasible confusion matrix overall AND on the
  near-threshold slice (where disagreement hides). Proceed only if disagreement is small + unstructured.
* **G2 — static ceiling.** Label many (scene, split) pairs analytically (fits?), train a small probe
  on the per-block scene features, and check it is accurate on clear cases but materially imperfect
  near-threshold (so the static model has adaptivity headroom, not a saturated ceiling).
* **G3 — difficulty sanity.** On the hard strata, the best greedy hand-rule's first attempt fails on
  >= 50% of problems, and the analytic failure culprits are spread across objects, not concentrated.

Run:
    python experiments/spectre/restock3d_v3_gates.py --gate all
    # G1 uses a LABEL-AWARE budget: --g1-feas-cap-s 120 (feasible ~40 s/candidate) and
    # --g1-infeas-cap-s 15 (infeasible plans fail regardless). A flat short cap starves the
    # feasible side and reports spurious disagreement (the 10 s-cap trap, decisions/07 2026-08-20).
"""

from __future__ import annotations

import argparse
import itertools
import time
from collections import Counter

from alphatamp.approaches.spectre import collect as C
from alphatamp.approaches.spectre.config import CollectionConfig
from alphatamp.approaches.spectre.envs.restock3d import feasibility_v3 as F
from alphatamp.approaches.spectre.envs.restock3d import generator_v3 as G
from alphatamp.approaches.spectre.envs.restock3d import strata_v3 as S


# ------------------------------------------------------------------ shared helpers
def _steps_of(action_plan):
    return [(op.name, [p.name for p in op.parameters]) for op in action_plan]


def _dims_pos(x0):
    dims, pos = {}, {}
    for o in x0:
        if o.name.startswith("obj_goal"):
            dims[o.name] = (
                2 * x0.get(o, "half_extent_x"),
                2 * x0.get(o, "half_extent_z"),
            )
            p = x0.get_object_pose(o.name).position
            pos[o.name] = (float(p[0]), float(p[1]))
    return dims, pos


def _near_threshold(dims) -> bool:
    """A scene is near-threshold if some block's height is within 0.02 of the short
    cutoff (the genuine tall/short decision) — where analytic and real are likeliest to
    disagree."""
    return any(abs(h - F.SHORT_CUTOFF) <= 0.02 for (_w, h) in dims.values())


def _v3_cfg(stratum, pid, k_max, cap_s, samples):
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
        plan_generator="closed_form",
        abstract_plan_timeout_s=30.0,
        refinement_timeout_s=cap_s,
        num_sampling_attempts_per_step=samples,
        max_trajectory_steps=500,
    )


# ------------------------------------------------------------------ G1
def gate_g1(strata, per_stratum, k_max, feas_cap, infeas_cap, samples) -> None:
    import kinder
    from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph

    from alphatamp.approaches.spectre.env_registry import register_extra_envs

    register_extra_envs()
    # confusion: (analytic_feasible, real_success) -> count; kept overall + near-threshold slice
    conf = {"all": Counter(), "near": Counter()}
    fam_disagree = Counter()
    t0 = time.perf_counter()
    n_problems = 0
    for st in strata:
        for idx in range(per_stratum):
            pid = S.problem_id("train", st, idx)
            cfg = _v3_cfg(st, pid, k_max, feas_cap, samples)
            env = kinder.make(cfg.env_id)
            try:
                obs, _ = env.reset(seed=pid)
                em = C._make_env_models(cfg, env.observation_space, env.action_space)
                x0 = em.observation_to_state(obs)
                s0 = em.state_abstractor(x0)
                goal = em.goal_deriver(x0)
                dims, pos = _dims_pos(x0)
                near = _near_threshold(dims)
                bpg = BilevelPlanningGraph()
                bpg.add_abstract_state_node(s0)
                bpg.add_state_node(x0)
                bpg.add_state_abstractor_edge(x0, s0)
                pg = C._make_plan_generator(cfg, em, obs, pid, x0)
                pool = list(
                    itertools.islice(
                        pg(x0, s0, goal, cfg.abstract_plan_timeout_s, bpg), k_max
                    )
                )
                sampler = C._make_trajectory_sampler(cfg, em)
                for cidx, (sp, ap) in enumerate(pool):
                    rec = F.classify_skeleton(_steps_of(ap), dims, pos)
                    analytic_feasible = rec is None
                    # Label-aware budget: an infeasible plan fails regardless of time (a short cap
                    # confirms it cheaply), but a FEASIBLE plan needs the full ~40 s/candidate real MP
                    # to actually seat every block. A single flat short cap starves the feasible side
                    # and reports spurious disagreement (the 10 s trap, decisions/07 2026-08-20).
                    cap = feas_cap if analytic_feasible else infeas_cap
                    seed = C._refinement_seed(cfg.refinement_seed_rule, pid, cidx)
                    refiner = C._make_refiner(cfg, obs, sampler, seed)
                    if hasattr(sampler, "clear"):
                        sampler.clear()
                    try:
                        plan = refiner(x0, sp, ap, cap, bpg)
                        real_success = plan is not None
                    except BaseException:  # noqa: BLE001
                        real_success = False
                    key = (analytic_feasible, real_success)
                    conf["all"][key] += 1
                    if near:
                        conf["near"][key] += 1
                    if analytic_feasible != real_success:
                        fam_disagree[
                            ("feasible" if analytic_feasible else "infeasible")
                        ] += 1
                n_problems += 1
                print(
                    f"  [G1] stratum {st} idx {idx} pid {pid}: pool {len(pool)} "
                    f"(elapsed {time.perf_counter()-t0:.0f}s)",
                    flush=True,
                )
            finally:
                env.close()

    print("\n=== G1: analytic (feasible) vs real (success) ===")
    for slice_name in ("all", "near"):
        c = conf[slice_name]
        tot = sum(c.values()) or 1
        agree = c[(True, True)] + c[(False, False)]
        print(
            f"[{slice_name}] n={sum(c.values())} agreement={100*agree/tot:.1f}%  "
            f"TP(feas&succ)={c[(True,True)]} TN(infeas&fail)={c[(False,False)]} "
            f"FP(feas&fail)={c[(True,False)]} FN(infeas&succ)={c[(False,True)]}"
        )
    print(f"disagreement direction: {dict(fam_disagree)}")
    print(f"G1 done in {time.perf_counter()-t0:.0f}s over {n_problems} problems")


# ------------------------------------------------------------------ G2
def _q(v, q):
    return round(v / q) * q if q else v


def _g2_dataset(stratum, n_scenes, quant):
    """Balanced (scene, split)->fits dataset. Per scene, take one feasible split (if any) and one
    infeasible split, so classes are ~50/50 (a random split at rho~0.006 is almost always
    infeasible -- a degenerate, trivially-predicted set). Features = per-block (width, height,
    is-short), optionally quantized to ``quant`` m to simulate finite perception."""
    import numpy as np

    from alphatamp.approaches.spectre.envs.restock3d.generator import _Rng

    p = S.params(stratum)
    n = p.n
    X, Y, near = [], [], []
    seed = 0
    while len(Y) < 2 * n_scenes:
        rng = _Rng(seed * 97 + stratum + 3)
        seed += 1
        widths = [G._u(rng, F.WIDTH_MIN, F.WIDTH_MAX) for _ in range(n)]
        heights = G._sample_heights(rng, n, p.n_forced, p.n_near)
        blocks = [F.Block(f"o{i}", widths[i], heights[i]) for i in range(n)]
        feas = F.enumerate_feasible_splits(blocks)
        if not feas:
            continue
        # one feasible + one infeasible split
        chosen = [(feas[int(rng.uniform() * len(feas))], 1.0)]
        for _try in range(20):
            rnd = {
                f"o{i}": ("short" if rng.uniform() < 0.5 else "tall") for i in range(n)
            }
            if not F.split_is_feasible(rnd, blocks):
                chosen.append((rnd, 0.0))
                break
        for assign, label in chosen:
            feat = []
            for i in range(n):
                feat += [
                    _q(widths[i], quant),
                    _q(heights[i], quant),
                    1.0 if assign[f"o{i}"] == "short" else 0.0,
                ]
            X.append(feat)
            Y.append(label)
            near.append(any(abs(h - F.SHORT_CUTOFF) <= 0.02 for h in heights))
    return np.array(X, dtype="float32"), np.array(Y, dtype="float32"), np.array(near)


def _train_probe(X, Y, near, n):
    import numpy as np
    import torch

    Xt = torch.tensor(X)
    Yt = torch.tensor(Y)
    ntr = int(0.8 * len(X))
    mlp = torch.nn.Sequential(
        torch.nn.Linear(3 * n, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 1),
    )
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(600):
        opt.zero_grad()
        loss = lossf(mlp(Xt[:ntr]).squeeze(-1), Yt[:ntr])
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (mlp(Xt[ntr:]).squeeze(-1) > 0).float()
    yte, near_te = Yt[ntr:], near[ntr:]
    acc = (pred == yte).float().mean().item()
    cm = ~near_te
    ac = (
        (pred[torch.tensor(cm)] == yte[torch.tensor(cm)]).float().mean().item()
        if cm.any()
        else float("nan")
    )
    an = (
        (pred[torch.tensor(near_te)] == yte[torch.tensor(near_te)])
        .float()
        .mean()
        .item()
        if near_te.any()
        else float("nan")
    )
    return acc, ac, an, float(yte.mean())


def gate_g2(stratum, n_scenes) -> None:
    """Static ceiling on a BALANCED (scene, split) set, EXACT vs perception-DEGRADED
    features.

    Predicting the analytic fits-label from EXACT features is deterministic arithmetic,
    so an exact probe saturates BY DESIGN -- the headroom for a learned method comes
    from finite perception (and the analytic<->real gap G1 measures). The degraded probe
    (features quantized to a perception granularity) should stay accurate on clear cases
    but drop near-threshold: the §0 evidence that the static ceiling is not saturated
    once perception is realistic.
    """
    print(f"\n=== G2 static ceiling (stratum {stratum}, balanced n={2*n_scenes}) ===")
    p = S.params(stratum)
    for label, quant in (("exact", 0.0), ("perception~15mm", 0.015)):
        X, Y, near = _g2_dataset(stratum, n_scenes, quant)
        acc, ac, an, pos = _train_probe(X, Y, near, p.n)
        print(
            f"[{label:14s}] overall {acc:.3f} | clear {ac:.3f} | near-threshold {an:.3f} "
            f"| pos-rate {pos:.2f}"
        )
    print(
        "want: exact ~saturates; degraded near-threshold materially imperfect (~0.60-0.85)"
    )


# ------------------------------------------------------------------ G3
def gate_g3(strata, n) -> None:
    print(
        "\n=== G3 difficulty sanity (hand-rule first-attempt failure + culprit spread) ==="
    )
    for st in strata:
        p = S.params(st)
        n_crack = 0
        culprit_objs = Counter()
        for seed in range(n):
            try:
                spec = G.build_spec_v3(seed, st)
            except RuntimeError:
                continue
            blocks = spec.blocks()
            # best greedy rule: does its split fail?
            both_fail = all(
                not F.split_is_feasible(r(blocks), blocks)
                for r in F.HAND_RULES.values()
            )
            n_crack += both_fail
            # culprit spread: classify the send-shortest-up ordered skeleton, collect the culprit object
            assign = F.greedy_send_shortest_up(blocks)
            plan = _greedy_plan(spec, assign)
            rec = F.classify_skeleton(
                plan,
                {b.name: (b.width, b.height) for b in blocks},
                {b.name: (b.x, b.y) for b in blocks},
            )
            if rec is not None:
                for c in rec["culprits"]:
                    culprit_objs[c] += 1
        frac = n_crack / n
        distinct = len(culprit_objs)
        top = culprit_objs.most_common(1)[0][1] if culprit_objs else 0
        conc = (top / sum(culprit_objs.values())) if culprit_objs else 0.0
        print(
            f"stratum {st} (crack-required={p.require_crack}): both-greedy-fail {100*frac:.0f}% | "
            f"distinct culprit objs {distinct} | top-object concentration {conc:.2f}"
        )


def _greedy_plan(spec, assign):
    """A south-to-north pick + assigned-section place plan for a greedy split (to
    observe culprits)."""
    order = sorted(range(spec.n), key=lambda i: spec.floor[i][1])  # nearest-first
    steps = []
    for i in order:
        name = spec.names[i]
        steps.append(("pick", ["robot", name]))
        steps.append(
            (
                ("place_short" if assign[name] == "short" else "place_tall"),
                ["robot", name],
            )
        )
    return steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", choices=["g1", "g2", "g3", "all"], default="all")
    ap.add_argument("--g1-problems", type=int, default=6)
    ap.add_argument("--g1-strata", type=int, nargs="+", default=[0, 2, 3])
    ap.add_argument("--g1-kmax", type=int, default=12)
    ap.add_argument(
        "--g1-feas-cap-s",
        type=float,
        default=120.0,
        help="real-refiner cap for analytic-FEASIBLE candidates (~40s/candidate needed)",
    )
    ap.add_argument(
        "--g1-infeas-cap-s",
        type=float,
        default=15.0,
        help="short cap for analytic-INFEASIBLE candidates (they fail regardless)",
    )
    ap.add_argument("--g1-samples", type=int, default=18)
    ap.add_argument("--g2-scenes", type=int, default=4000)
    ap.add_argument("--g3-n", type=int, default=150)
    args = ap.parse_args()

    if args.gate in ("g3", "all"):
        gate_g3([2, 3], args.g3_n)
    if args.gate in ("g2", "all"):
        gate_g2(3, args.g2_scenes)
    if args.gate in ("g1", "all"):
        gate_g1(
            args.g1_strata,
            args.g1_problems,
            args.g1_kmax,
            args.g1_feas_cap_s,
            args.g1_infeas_cap_s,
            args.g1_samples,
        )


if __name__ == "__main__":
    main()
