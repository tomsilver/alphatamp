"""Collect the pooled StickButton2D dataset: 400 / 100 / 100 over b1/b2/b3/b5.

Why this exists rather than ``spectre_collect.py``: that entry point collects a fixed
``[seed_start, seed_end)`` range of one env variant, and this collection needs two things
it does not do — pool four kinder env variants into a single ``env_variant`` directory
under the stratum-encoded problem ids of ``envs/stickbutton2d/strata.py``, and **reject
and resample** problems that yield no feasible skeleton at all.

**Why rejection.** ``dataset.py`` drops episodes with ``num_success == 0``, so an
all-negative problem is not a hard example, it is a problem that silently costs its full
refinement budget and then contributes nothing. DD2D already resamples these (its
collector records them as ``reason="unsolved"``); this mirrors that. It does bias the
collection toward solvable scenes — that is the same bias DD2D's numbers carry, and it is
recorded per variant in the census this prints, so the rejection rate is visible rather
than implicit.

Problems are independent, so they run concurrently (spectre ``CLAUDE.md``, "Use the
hardware"); ``spawn`` is required because pyperplan and bilevel_planning keep module-level
caches that do not survive a concurrent ``fork``. Progress, per-variant keep rate and an
ETA are printed on a heartbeat so a run that has gone wrong can be stopped early rather
than discovered at the end.

Usage::

    python experiments/spectre/sb2d_collect.py --workers 30
    python experiments/spectre/sb2d_collect.py --variants 3 --split test --target 25
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from alphatamp.approaches.spectre.config import (  # noqa: E402  pylint: disable=wrong-import-position
    CollectionConfig,
)
from alphatamp.approaches.spectre.envs.stickbutton2d.strata import (  # noqa: E402  pylint: disable=wrong-import-position
    BUTTON_COUNTS,
    ENV_VARIANT,
    SPLIT_SIZES,
    env_id,
    problem_id,
)

_HEARTBEAT_S = 60.0

# Budgets. `K_max` and `refinement_timeout_s` are the stated collection contract (200
# plan attempts, 20 s each). The sampler settings are the ones the feasibility harness
# measured 100% (b3) / 75% (b5) solvability under -- changing them would invalidate that
# measurement, which is the only evidence this collection is worth running.
_K_MAX = 200
_REFINEMENT_TIMEOUT_S = 20.0
_ABSTRACT_PLAN_TIMEOUT_S = 60.0
_SAMPLES_PER_STEP = 5
_MAX_TRAJECTORY_STEPS = 200


def _config(
    num_buttons: int, split: str, env_variant: str = ENV_VARIANT
) -> CollectionConfig:
    return CollectionConfig(
        env_id=env_id(num_buttons),
        env_variant=env_variant,
        model_name="stickbutton2d",
        model_kwargs={"num_buttons": num_buttons},
        split=split,  # type: ignore[arg-type]
        num_problems=SPLIT_SIZES[split],
        problem_seed_start=0,
        problem_seed_end=1,  # unused: this driver passes explicit problem ids
        K_max=_K_MAX,
        abstract_plan_timeout_s=_ABSTRACT_PLAN_TIMEOUT_S,
        refinement_timeout_s=_REFINEMENT_TIMEOUT_S,
        num_sampling_attempts_per_step=_SAMPLES_PER_STEP,
        max_trajectory_steps=_MAX_TRAJECTORY_STEPS,
    )


def _collect_one(args: tuple[int, str, int, str, str]) -> dict:
    """Worker: collect one problem, keep it only if some skeleton refined.

    Returns a verdict dict rather than raising, so one pathological problem cannot take
    a worker down and strand the pool. ``env_variant`` travels in the args tuple because
    workers are spawned (fresh module import), so a module-global override in ``main``
    would not reach them.
    """
    num_buttons, split, index, data_root, env_variant = args
    # pylint: disable=import-outside-toplevel
    from alphatamp.approaches.spectre.collect import collect_episode, episode_path
    from alphatamp.approaches.spectre.io import atomic_write_pickle_gz

    pid = problem_id(split, num_buttons, index)
    path = episode_path(Path(data_root), env_variant, split, pid)
    start = time.perf_counter()
    if path.exists():
        return {"pid": pid, "kept": True, "cached": True, "index": index, "s": 0.0}
    try:
        episode = collect_episode(_config(num_buttons, split, env_variant), pid)
    except BaseException as exc:  # pylint: disable=broad-exception-caught
        return {
            "pid": pid,
            "kept": False,
            "index": index,
            "error": f"{type(exc).__name__}: {exc}",
            "s": time.perf_counter() - start,
        }
    kept = episode.summary.num_success >= 1
    if kept:
        atomic_write_pickle_gz(episode, path)
    return {
        "pid": pid,
        "kept": kept,
        "index": index,
        "pool": len(episode.skeleton_pool),
        "succ": episode.summary.num_success,
        "first": episode.summary.first_success_idx,
        "s": time.perf_counter() - start,
    }


class _VariantState:
    """Book-keeping for one (button count, split): how many kept, what to draw next."""

    def __init__(
        self, num_buttons: int, split: str, target: int, index_start: int = 0
    ) -> None:
        self.num_buttons = num_buttons
        self.split = split
        self.target = target
        self.index_start = index_start
        self.kept = 0
        self.drawn = 0
        self.rejected: list[int] = []
        self.errors: list[str] = []
        self.pool_sizes: list[int] = []
        self.seconds = 0.0

    @property
    def done(self) -> bool:
        """Whether this variant has reached its keeper target."""
        return self.kept >= self.target

    def next_index(self) -> int:
        """The next problem index to draw, and reserve it."""
        index = self.index_start + self.drawn
        self.drawn += 1
        return index

    def observe(self, result: dict) -> None:
        """Fold one worker verdict into the running census."""
        self.seconds += float(result.get("s", 0.0))
        if result.get("error"):
            self.errors.append(str(result["error"]))
            return
        if result["kept"]:
            self.kept += 1
            if "pool" in result:
                self.pool_sizes.append(int(result["pool"]))
        else:
            self.rejected.append(int(result["index"]))

    def census(self) -> dict:
        """Everything worth writing down about this variant's collection."""
        pools = self.pool_sizes
        return {
            "num_buttons": self.num_buttons,
            "split": self.split,
            "kept": self.kept,
            "drawn": self.drawn,
            "rejected": len(self.rejected),
            "rejected_indices": self.rejected,
            "errors": self.errors[:5],
            "n_errors": len(self.errors),
            "mean_pool": round(sum(pools) / len(pools), 1) if pools else 0.0,
            "cpu_seconds": round(self.seconds, 1),
        }


def _heartbeat(states: list[_VariantState], t0: float, inflight: int) -> None:
    kept = sum(s.kept for s in states)
    target = sum(s.target for s in states)
    elapsed = time.time() - t0
    rate = kept / elapsed if elapsed > 0 and kept else 0.0
    eta = (target - kept) / rate if rate > 0 else float("inf")
    per = " ".join(
        f"b{s.num_buttons}/{s.split[:2]}={s.kept}/{s.target}"
        for s in states
        if not s.done
    )
    print(
        f"[{elapsed/60:6.1f}m] kept {kept}/{target}  inflight {inflight}"
        f"  ETA {eta/60:.0f}m  | {per or 'all targets met'}",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the collection to its per-variant keeper targets."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--data-root", default="data/spectre")
    ap.add_argument(
        "--env-variant",
        default=ENV_VARIANT,
        help="collection variant to write into (default the standard stickbutton2d_v1). "
        "Set to a NEW variant (e.g. stickbutton2d_v2) to top up / expand a stratum "
        "without mutating v1's frozen collection.",
    )
    ap.add_argument(
        "--variants",
        default=",".join(str(b) for b in BUTTON_COUNTS),
        help="button counts to collect",
    )
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument(
        "--target",
        type=int,
        default=0,
        help="override keepers per (variant, split); 0 = the standard 100/25/25",
    )
    ap.add_argument(
        "--targets",
        default="",
        help="per-split keeper targets, e.g. 'test=25,val=20,train=40'. Overrides "
        "--target for the splits named. Exists because the expensive variants are "
        "budgeted per split rather than uniformly: the test split sizes the headline "
        "and is held at full size, while the train split is the one that gets cut when "
        "compute runs short",
    )
    ap.add_argument(
        "--max-draws-factor",
        type=float,
        default=4.0,
        help="give up on a variant after target * this many draws",
    )
    ap.add_argument(
        "--index-start",
        type=int,
        default=0,
        help="first problem index to draw. Lets a second process work the same "
        "(variant, split) on a disjoint index range -- e.g. to put cores freed by a "
        "finished variant onto a slower one without either duplicating draws. Keep the "
        "ranges genuinely disjoint: two processes drawing the same index both refine it "
        "and one result is thrown away",
    )
    a = ap.parse_args(argv)

    variants = [int(v) for v in a.variants.split(",") if v.strip()]
    splits = [s for s in a.splits.split(",") if s.strip()]
    env_variant = a.env_variant
    overrides = dict(
        (k.strip(), int(v))
        for k, v in (p.split("=", 1) for p in a.targets.split(",") if p.strip())
    )

    def _target_for(split: str) -> int:
        return overrides.get(split, a.target or SPLIT_SIZES[split])

    states = [
        _VariantState(b, s, _target_for(s), a.index_start)
        for s in splits
        for b in variants
    ]
    root = (
        str(REPO / a.data_root) if not Path(a.data_root).is_absolute() else a.data_root
    )

    total = sum(s.target for s in states)
    print(
        f"collecting {total} episodes into {env_variant}"
        f" (variants {variants}, splits {splits}, {a.workers} workers)",
        flush=True,
    )
    print(
        f"budgets: K_max={_K_MAX} refine={_REFINEMENT_TIMEOUT_S}s"
        f" plan={_ABSTRACT_PLAN_TIMEOUT_S}s samples={_SAMPLES_PER_STEP}",
        flush=True,
    )

    t0 = time.time()
    last_beat = t0
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx) as pool:
        pending: dict[Future, _VariantState] = {}

        def _refill() -> None:
            """Keep the pool fed: one task per free slot, from any unfinished variant.

            Draws are issued against the *outstanding* count, not the kept count, so a
            variant near its target does not over-draw by a whole wave of workers.
            """
            free = a.workers * 2 - len(pending)
            for state in states:
                while free > 0 and not state.done:
                    outstanding = sum(1 for s in pending.values() if s is state)
                    if state.kept + outstanding >= state.target:
                        break
                    if state.drawn >= state.target * a.max_draws_factor:
                        break
                    fut = pool.submit(
                        _collect_one,
                        (
                            state.num_buttons,
                            state.split,
                            state.next_index(),
                            root,
                            env_variant,
                        ),
                    )
                    pending[fut] = state
                    free -= 1

        _refill()
        while pending:
            done = [f for f in list(pending) if f.done()]
            if not done:
                if time.time() - last_beat >= _HEARTBEAT_S:
                    _heartbeat(states, t0, len(pending))
                    last_beat = time.time()
                time.sleep(0.5)
                continue
            for fut in done:
                state = pending.pop(fut)
                try:
                    state.observe(fut.result())
                except BaseException as exc:  # pylint: disable=broad-exception-caught
                    state.errors.append(f"{type(exc).__name__}: {exc}")
            _refill()

    census = [s.census() for s in states]
    # One file per (variant set, split set). A single shared name loses every concurrent
    # job but the last to exit -- which is the normal way this is run, since the variants
    # differ by ~2x in cost and are launched separately.
    tag = "b" + "".join(str(v) for v in variants) + "_" + "".join(s[0] for s in splits)
    out = Path(root) / "raw" / env_variant / f"collection_census_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(census, indent=2))

    print(f"\n=== census ({(time.time()-t0)/60:.1f} min wall clock) ===", flush=True)
    print(
        f"{'variant':<10}{'split':<7}{'kept':<7}{'drawn':<7}"
        f"{'rejected':<10}{'mean pool':<11}{'cpu min':<9}"
    )
    for c in census:
        print(
            f"b{c['num_buttons']:<9}{c['split']:<7}{c['kept']:<7}{c['drawn']:<7}"
            f"{c['rejected']:<10}{c['mean_pool']:<11}{c['cpu_seconds']/60:<9.1f}"
        )
    shortfall = [c for c in census if c["kept"] < _target_for(c["split"])]
    if shortfall:
        print(
            "\nWARNING: targets not met — "
            + ", ".join(
                f"b{c['num_buttons']}/{c['split']}={c['kept']}" for c in shortfall
            )
        )
    print(f"census written to {out}")
    return 1 if shortfall else 0


if __name__ == "__main__":
    raise SystemExit(main())
