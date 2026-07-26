"""DD2D refiner -- the shared backjumping refiner (spec Section 10.2).

Sequential binding with backjumping over a staging skeleton
``[pick(o); place-buffer(o) ...] ++ retrieve(target)``, replayed against a
:class:`~blocks_tamp.dd2d.world.DrawerWorld`:

* ``pick(o)``: bind a grasp of ``o`` at its drawer pose whose fingers clear the remaining
  drawer items + the wall band (``sample-grasp`` + ``CFreeGrasp``). No clear grasp => the
  blocker is itself buried => hard dead-end (drawer-side extraction infeasibility).
* ``place-buffer(o)``: sample a compaction-biased buffer pose (``sample-buffer-pose``) that
  is collision-free AND whose grasp clears the already-staged items (accessibility). After
  ``t`` consecutive failures, **backjump**: undo the previous buffer placement and re-sample
  it (member-order stays fixed; the poses are what backtrack). All attempts count against
  the stream-call budget ``B``.
* ``retrieve(target)``: the target grasp must clear the remaining drawer items + wall.

The expected signature on an infeasible (over-large / mis-chosen) subset -- early
placements succeed, the last fails, backjumps thrash, budget exhausts -- is the cost
structure under study (spec Section 10.2 / D2). Whether the whole skeleton binds within
``B`` is the (noisy) feasibility *label* for a PIGINet example. Returns the shared
:class:`~blocks_tamp.refine.RefineResult` / :class:`~blocks_tamp.refine.BoundStep`, so
``record.build_example`` and the demo's video selection consume it unchanged.
"""

from __future__ import annotations

import random
import time
import warnings

from alphatamp.approaches.spectre.envs.dd2d.refine import (
    BoundStep,
    FailureObservation,
    RefineResult,
)
from alphatamp.approaches.spectre.envs.dd2d.skeleton import Skeleton

from .grasps import finger_rects, has_grasp_witness
from .world import DrawerWorld, StreamCounter, sample_buffer_pose

# hard backstop so a mis-configured "unbounded" run can never thrash forever
_SAFETY_CALLS = 5_000_000


class DD2DRefiner:
    """Backjumping staging refiner conforming to the blocks_tamp refiner protocol.

    Three tunable cost levers (spec P13/P14/P15), all surfaced on the demo/generator so a
    demo or a data-collection run can dial the refinement budget:

    * ``budget`` (B, total stream calls) -- the global cap; ``<= 0`` or ``None`` disables it.
    * ``retry_cap`` (t) -- ``sample-buffer-pose`` calls per ``place-buffer`` step before backjump.
    * ``samples_per_step`` (m_p) -- candidate poses tried *inside* one ``sample-buffer-pose`` call
      (sampler strength / packing quality), passed straight to :func:`sample_buffer_pose`.
    * ``time_budget`` -- wall-clock seconds per plan; ``None`` disables it.

    Refinement stops when the stream-call cap OR the wall-clock cap is hit (whichever first), so
    you can govern by time + per-step attempts *instead of* total stream calls (set ``budget<=0``
    with a ``time_budget``). If both caps are disabled, ``budget`` falls back to 300.
    """

    name = "dd2d-backjump"
    label_source = "refine_buffer_stage"

    def __init__(
        self,
        budget: int | None = 300,
        retry_cap: int = 10,
        samples_per_step: int = 15,
        time_budget: float | None = None,
        **_ignored,
    ) -> None:
        # ``**_ignored`` swallows make_refiner-style kwargs meant for other refiners.
        self.budget = (
            budget if (budget is not None and budget > 0) else None
        )  # <=0 -> uncapped
        self.retry_cap = retry_cap
        self.samples_per_step = samples_per_step
        self.time_budget = (
            time_budget if (time_budget is not None and time_budget > 0) else None
        )
        if self.budget is None and self.time_budget is None:
            warnings.warn(
                "DD2DRefiner: both stream-call budget and time_budget are disabled; "
                "falling back to budget=300 to avoid an unbounded search.",
                stacklevel=2,
            )
            self.budget = 300

    def refine(self, skeleton: Skeleton, scene, seed: int = 0) -> RefineResult:
        rng = random.Random(seed)
        t0 = time.perf_counter()
        plan = list(skeleton.actions)
        n = len(plan)
        counter = StreamCounter()
        world = DrawerWorld(scene, counter)

        def exhausted() -> bool:
            if counter.calls >= _SAFETY_CALLS:
                return True
            if self.budget is not None and counter.calls >= self.budget:
                return True
            if (
                self.time_budget is not None
                and (time.perf_counter() - t0) >= self.time_budget
            ):
                return True
            return False

        committed: list[tuple[int, BoundStep, dict]] = (
            []
        )  # (idx, step, snapshot_before)
        best_reached = 0
        best_steps: list[BoundStep] = []
        idx = 0
        # v3 instrumentation: observation-only accumulators. Nothing below may call the
        # stream counter, draw from ``rng``, or change control flow -- ``n_attempts`` is
        # ``counter.calls``, so one extra stream call would shift every downstream label.
        failures: list[FailureObservation] = []
        n_backjumps = 0
        budget_exit = False

        def note_progress() -> None:
            nonlocal best_reached, best_steps
            if idx > best_reached:
                best_reached = idx
                best_steps = [bs for _, bs, _ in committed]

        # Equivalent to ``while idx < n and not exhausted()`` -- same evaluation order and
        # same number of (side-effect-free) ``exhausted()`` calls -- but it records
        # *which* condition ended the loop, so a budget exit is distinguishable from a
        # query failure. That distinction is invisible in ``failure_action``, which names
        # the deepest step *reached* rather than one that was tested.
        while idx < n:
            if exhausted():
                budget_exit = True
                break
            act = plan[idx]
            if act.name == "pick":
                o = act.args[0]
                st = world.states[o]
                counter.test()  # sample-grasp + CFreeGrasp against the drawer
                obstacles = world.drawer_obstacles(ignore=o)
                g, blocked = has_grasp_witness(st.shape, st.pose, obstacles)
                if g is None:
                    names = world.drawer_obstacle_names(ignore=o)
                    failures.append(
                        FailureObservation(
                            step_index=idx,
                            schema="pick",
                            args=(o,),
                            culprits=tuple(sorted(names[i] for i in blocked)),
                            unmoved=tuple(sorted(world.region_items("drawer"))),
                            n_step=1,
                            exhausted=True,  # every grasp cell was tried
                        )
                    )
                    break  # blocker ungraspable (buried) -> hard dead-end
                snap = world.snapshot()
                world.pick(o)
                committed.append(
                    (
                        idx,
                        BoundStep(
                            act,
                            {"phase": "pick", "item": o, "grasp": g, "pose": st.pose},
                        ),
                        snap,
                    )
                )
                idx += 1
                note_progress()

            elif act.name == "place-buffer":
                o = act.args[0]
                st = world.states[o]
                pose = grasp = None
                ghosts: list[tuple[float, float, float]] = []
                calls_before = counter.calls
                hit_budget = False
                blocked_names: set[str] = set()
                for _ in range(self.retry_cap):
                    if exhausted():
                        hit_budget = True
                        break
                    counter.sample()  # sample-buffer-pose
                    cand = sample_buffer_pose(
                        st.shape,
                        world.buffer_poly,
                        world.buffer_obstacles(),
                        rng,
                        m_p=self.samples_per_step,
                    )
                    if cand is None:
                        continue
                    counter.test()  # CFreeGrasp at the destination (accessibility)
                    obstacles = world.buffer_obstacles()
                    g, blocked = has_grasp_witness(st.shape, cand, obstacles)
                    if g is None:
                        names = world.buffer_obstacle_names()
                        blocked_names.update(names[i] for i in blocked)
                        ghosts.append(cand)  # packs but not graspable there
                        continue
                    pose, grasp = cand, g
                    break
                if pose is None:
                    # No culprit means the sampler never found a *packing* pose at all
                    # (a volume failure); culprits mean it packed but the staged items
                    # blocked the approach (an accessibility failure).
                    failures.append(
                        FailureObservation(
                            step_index=idx,
                            schema="place-buffer",
                            args=(o,),
                            culprits=tuple(sorted(blocked_names)),
                            unmoved=tuple(sorted(world.region_items("drawer"))),
                            n_step=counter.calls - calls_before,
                            exhausted=not hit_budget,
                        )
                    )
                    if not self._backjump(world, committed):
                        break  # nothing to backjump to -> infeasible (joint overflow)
                    n_backjumps += 1
                    # resume at the place-buffer we just undid (its pick is still committed)
                    idx = (committed[-1][0] + 1) if committed else 0
                    continue
                world.place_buffer(o, pose)
                committed.append(
                    (
                        idx,
                        BoundStep(
                            act,
                            {
                                "phase": "place",
                                "item": o,
                                "grasp": grasp,
                                "pose": pose,
                                "ghosts": ghosts,
                            },
                        ),
                        world.snapshot(),
                    )
                )
                idx += 1
                note_progress()

            elif act.name == "retrieve":
                o = act.args[0]
                st = world.states[o]
                counter.test()
                obstacles = world.drawer_obstacles(ignore=o)
                g, blocked = has_grasp_witness(st.shape, st.pose, obstacles)
                if g is None:
                    names = world.drawer_obstacle_names(ignore=o)
                    failures.append(
                        FailureObservation(
                            step_index=idx,
                            schema="retrieve",
                            args=(o,),
                            culprits=tuple(sorted(names[i] for i in blocked)),
                            unmoved=tuple(sorted(world.region_items("drawer"))),
                            n_step=1,
                            exhausted=True,  # every grasp cell was tried
                        )
                    )
                    break  # target still blocked
                committed.append(
                    (
                        idx,
                        BoundStep(
                            act,
                            {
                                "phase": "retrieve",
                                "item": o,
                                "grasp": g,
                                "pose": st.pose,
                            },
                        ),
                        world.snapshot(),
                    )
                )
                idx += 1
                note_progress()
            else:  # pragma: no cover - DD2D emits only pick/place-buffer/retrieve
                break

        elapsed = time.perf_counter() - t0
        if idx == n:  # every step bound -> feasible
            return RefineResult(
                status="feasible",
                steps_bound=n,
                plan_length=n,
                n_attempts=counter.calls,
                failure_action=None,
                bound_plan=[bs for _, bs, _ in committed],
                elapsed=elapsed,
                failures=failures,
                n_backjumps=n_backjumps,
            )
        # Fourth emission site: the loop stopped on the global budget rather than on a
        # failed query, so none of the three sites above fired for the step we were
        # about to try. ``failure_action`` still names ``plan[best_reached]`` -- a step
        # that was never tested -- which is precisely the misreport that made v2.2's
        # proof-demotion unsound (one dd2d_v2 candidate, ``n_attempts=2406``). Emitting
        # an explicit marker makes "no evidence" distinguishable from "lost in
        # conversion", and the v3 registry refuses to promote it to proof tier.
        if budget_exit and idx < n:
            act = plan[idx]
            failures.append(
                FailureObservation(
                    step_index=idx,
                    schema=act.name,
                    args=tuple(act.args),
                    unmoved=tuple(sorted(world.region_items("drawer"))),
                    n_step=0,
                    exhausted=False,
                    budget_exhausted=True,
                )
            )
        failure_action = str(plan[best_reached]) if best_reached < n else None
        return RefineResult(
            status="infeasible",
            steps_bound=best_reached,
            plan_length=n,
            n_attempts=counter.calls,
            failure_action=failure_action,
            bound_plan=best_steps,
            elapsed=elapsed,
            failures=failures,
            n_backjumps=n_backjumps,
            budget_exhausted=budget_exit,
        )

    @staticmethod
    def _backjump(world: DrawerWorld, committed: list) -> bool:
        """Undo committed steps back through the most recent place-buffer so it re-
        samples with fresh randomness.

        Returns False if there is no prior placement to revise.
        """
        while committed:
            idx, step, snap = committed.pop()
            world.restore(snap)
            if step.action.name == "place-buffer":
                return True
        return False
