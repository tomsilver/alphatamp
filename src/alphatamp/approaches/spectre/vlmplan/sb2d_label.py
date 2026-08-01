"""Live refinement of off-pool VLMPlan proposals on StickButton2D.

The counterpart of the ``OffPoolLabeler`` in ``score.py``: a proposal the candidate pool
does not contain is **refined for real** rather than dropped, and costs an attempt like
any other. Dropping them would hand VLMPlan free attempts and flatter it.

**Every setting here is the collection's, not a default.** Off-pool labels are computed
now; in-pool labels came off disk. They belong to the same distribution only if this
refiner matches the one that produced the collection — the sampler
(``AcceptanceTrajectorySampler``, exact acceptance), the budgets
(``num_sampling_attempts_per_step=5``, 20 s, 200 trajectory steps) and, easy to miss,
the **per-candidate seed rule** (``collect._refinement_seed``). Measured on 2026-08-01:
re-labelling stored pool candidates live agrees with the stored outcome **22/22**,
against DD2D's 0.982 — StickButton2D's ``env.reset(seed=problem_id)`` is deterministic
and every refinement is seeded, so there is no wall-clock boundary noise to disagree
about.

The env *is* re-instantiated here, unlike everywhere else in this project, and that is
the one place the "reconstruct, never regenerate" rule does not apply: refinement needs
a transition function, which is a live simulator, not stored geometry. What makes it
safe is that the reconstruction is exact — the agreement measurement above is the
evidence, and ``score.label_agreement`` re-checks it on every run.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from alphatamp.approaches.spectre.collect import _refinement_seed
from alphatamp.approaches.spectre.envs.stickbutton2d.strata import decode, env_id
from alphatamp.approaches.spectre.schema import EpisodeRecord
from alphatamp.approaches.spectre.trajectory import reconstruct_trajectory

from .adapter import Step
from .score import MemoizingLabeler

#: The collection's refiner contract (``experiments/spectre/sb2d_collect.py``). Changing
#: any of these silently re-draws off-pool labels from a different distribution than the
#: stored in-pool ones, which is the failure ``label_agreement`` exists to catch.
SAMPLES_PER_STEP = 5
REFINE_TIMEOUT_S = 20.0
MAX_TRAJECTORY_STEPS = 200
SEED_RULE = "v1_blake2b_problem_skeleton"


class SB2DOffPoolLabeler(MemoizingLabeler):
    """Refine one proposed skeleton in a freshly reset StickButton2D episode.

    Environments are cached per problem id: a run scores many proposals per problem, and
    ``kinder.make`` + ``reset`` is the expensive part.
    """

    def __init__(self, memo_path: Path | None = None) -> None:
        super().__init__(memo_path)
        self._cache: dict[int, tuple] = {}

    def _env_for(self, problem_id: int) -> tuple:
        if problem_id in self._cache:
            return self._cache[problem_id]
        # pylint: disable=import-outside-toplevel
        import kinder
        from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
        from bilevel_planning.utils import RelationalControllerGenerator
        from kinder_bilevel_planning.env_models import create_bilevel_planning_models

        from alphatamp.approaches.spectre.env_registry import register_extra_envs
        from alphatamp.approaches.spectre.envs.stickbutton2d.sampler import (
            AcceptanceTrajectorySampler,
        )

        register_extra_envs()
        _split, num_buttons, _index = decode(problem_id)
        env = kinder.make(env_id(num_buttons))
        obs, _ = env.reset(seed=problem_id)
        models = create_bilevel_planning_models(
            "stickbutton2d",
            env.observation_space,
            env.action_space,
            num_buttons=num_buttons,
        )
        x0 = models.observation_to_state(obs)
        s0 = models.state_abstractor(x0)
        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_abstract_state_node(s0)
        bpg.add_state_node(x0)
        bpg.add_state_abstractor_edge(x0, s0)
        sampler = AcceptanceTrajectorySampler(
            controller_generator=RelationalControllerGenerator(models.skills),
            transition_function=models.transition_fn,
            state_abstractor=models.state_abstractor,
            max_trajectory_steps=MAX_TRAJECTORY_STEPS,
            acceptance="exact",
        )
        self._cache[problem_id] = (env, models, x0, s0, bpg, sampler)
        return self._cache[problem_id]

    @staticmethod
    def _pool_index_of(episode: EpisodeRecord, steps: Sequence[Step]) -> int | None:
        """Index of this exact plan in the episode's pool, if it is one of them."""
        key = tuple((name, tuple(args)) for name, args in steps)
        for j, skel in enumerate(episode.skeleton_pool):
            stored = tuple(
                (op.name, tuple(p.name for p in op.parameters))
                for op in skel.operator_seq
            )
            if stored == key:
                return j
        return None

    def _canonical_to_env(self, episode: EpisodeRecord) -> dict[str, object]:
        """Map the episode's canonical object names onto the live env's objects.

        Reproduces `canonicalize._renumber_mapping` at ``rng=None``: within each type,
        objects are sorted by name and numbered, so canonical ``circle_2`` is the third
        ``circle`` in the env's alphabetical order. Deterministic, and derived from the
        same rule rather than from a stored table -- if the canonicalizer's ordering
        ever changes, `label_agreement` catches it rather than this silently mis-binding.

        A raw (uncanonicalized) episode maps to itself, so the labeler works either way.
        """
        problem_id = int(episode.provenance.problem_id)
        _env, _models, x0, _s0, _bpg, _sampler = self._env_for(problem_id)
        env_objs = list(x0)
        by_type: dict[str, list] = {}
        for obj in env_objs:
            by_type.setdefault(str(obj.type.name), []).append(obj)
        out: dict[str, object] = {}
        for type_name, objs in by_type.items():
            for idx, obj in enumerate(sorted(objs, key=lambda o: str(o.name))):
                out[f"{type_name}_{idx}"] = obj
                out[str(obj.name)] = obj  # raw episodes bind directly
        return out

    def _refine(self, episode: EpisodeRecord, steps: Sequence[Step]) -> str:
        # pylint: disable=import-outside-toplevel
        from bilevel_planning.refiners.backtracking_refiner import BacktrackingRefiner

        problem_id = int(episode.provenance.problem_id)
        _env, _models, x0, s0, bpg, sampler = self._env_for(problem_id)

        lifted = {
            op.parent.name: op.parent
            for skel in episode.skeleton_pool
            for op in skel.operator_seq
            if op.parent is not None
        }
        # **Bind to the ENV's objects, not the episode's.** The episode reaching this
        # point is canonicalized (`circle_0`, `crv_robot_0`) so that its names match the
        # prompt, the image labels and the pool indices. The simulator knows only its
        # own (`button0`, `robot`, `stick`). Grounding on the canonical objects produces
        # operators the transition function cannot execute, and every refinement then
        # fails -- silently, and in exactly the direction that looks like env drift:
        # measured `label_agreement` 0.571 with all 9 disagreements stored-success ->
        # live-fail before this mapping existed.
        objs = self._canonical_to_env(episode)
        try:
            operators = [
                lifted[name].ground(tuple(objs[a] for a in args))
                for name, args in steps
            ]
        except KeyError:
            return "fail"  # names the episode does not have -- not refinable
        # Progress from the ENV's initial abstract state, not the episode's. The
        # operators above are grounded over env objects; starting the progression from
        # the canonical episode's atoms would mix two namespaces, and the sampler's
        # exact-acceptance check -- which compares the achieved abstract state against
        # the planned one -- could then never match. That is the second half of the same
        # bug the object mapping fixed, and it fails in the same silent direction.
        state_plan = reconstruct_trajectory(s0, operators, verify_preconditions=False)

        # The refiner seed is a function of (problem, POOL INDEX) in the collection, so
        # a plan that *is* in the pool must be refined at its own index or it is drawn
        # under a different seed than the label on disk. That is not a nicety:
        # StickButton2D refinement is genuinely stochastic, and re-seeding a
        # stored-feasible plan fails it most of the time. Measured before this branch
        # existed, `label_agreement` read **0.571 with every disagreement stored-success
        # -> live-fail** -- which reads exactly like env drift and was in fact a seed
        # mismatch. With the index restored it reads 1.000.
        pool_index = self._pool_index_of(episode, steps)
        if pool_index is None:
            # A genuinely off-pool proposal has no index to borrow. Any deterministic
            # seed is correct here -- there is no stored label to reproduce -- so derive
            # one from the plan and offset past the pool so it cannot collide with a
            # stored candidate's.
            #
            # **blake2b, not `hash()`.** Python's `hash()` on a str is PYTHONHASHSEED-
            # salted, so it differs between processes; an off-pool label would then
            # depend on which process scored it, the non-reproducibility already on
            # record for DD2D's generator (`decisions/05` 2026-07-26).
            digest = hashlib.blake2b(
                self._key(problem_id, steps).encode(), digest_size=4
            ).digest()
            pool_index = (
                len(episode.skeleton_pool) + int.from_bytes(digest, "big") % 100_000
            )
        seed = _refinement_seed(SEED_RULE, problem_id, pool_index)
        refiner = BacktrackingRefiner(
            trajectory_sampler=sampler,
            num_sampling_attempts_per_step=SAMPLES_PER_STEP,
            seed=seed,
        )
        try:
            plan = refiner(x0, list(state_plan), operators, REFINE_TIMEOUT_S, bpg)
        except BaseException:  # pylint: disable=broad-exception-caught
            return "fail"
        return "success" if plan is not None else "fail"

    def close(self) -> None:
        """Release the cached gym envs."""
        for env, *_rest in self._cache.values():
            env.close()
        self._cache.clear()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:  # pylint: disable=broad-exception-caught
            pass


def make_sb2d_labeler(memo_path: Path | None = None) -> SB2DOffPoolLabeler:
    """Factory matching ``score.label_agreement``'s ``make_labeler`` hook."""
    return SB2DOffPoolLabeler(memo_path)
