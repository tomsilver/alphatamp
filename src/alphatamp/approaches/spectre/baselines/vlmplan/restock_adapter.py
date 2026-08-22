"""Restock3D v2's :class:`~.adapter.EnvAdapter` + off-pool labeler.

``problem`` is a canonicalized ``EpisodeRecord`` (object names identical in the prompt, the
labeled image, and the pool indices). Restock's skills are simpler than StickButton2D's --
``pick`` / ``place_tall`` / ``place_short`` with no chaining -- but two things are disclosed
in text because the abstract model hides them and every trained method reads them from
geometry (so disclosing removes a handicap, not grants an advantage):

* **height / section fit** -- a tall block only fits the taller bottom section; ``place_short``
  of a tall block overflows the short-section ceiling (F3);
* **reach-over order** -- the front grasp reaches north over nearer objects, so a far object is
  blocked until nearer ones are cleared (store south-to-north).

The image is the env's own **oblique** render (reconstructed from the seed) with Set-of-Mark
labels, so a tall block is visually distinct from a cube.
"""

from __future__ import annotations

from typing import Sequence

from PIL.Image import Image

from alphatamp.approaches.spectre.schema import EpisodeRecord

from .adapter import EnvAdapter, Labeler, RawPlan, SkillSpec, Step

_ROBOT_TYPE = "Kinematic3DRobot"
_CUBOID_TYPE = "Kinematic3DCuboid"

_SKILL_TYPES: dict[str, tuple[str, ...]] = {
    "pick": (_ROBOT_TYPE, _CUBOID_TYPE),
    "place_tall": (_ROBOT_TYPE, _CUBOID_TYPE),
    "place_short": (_ROBOT_TYPE, _CUBOID_TYPE),
}

_CONTROLLER_NOTE = """\
What each skill does (the exact rules the low-level executor enforces):
- pick(robot, o): pick object o off the floor. Requires the gripper empty and o on the floor.
- place_tall(robot, o): place the held object on the TALL bottom shelf section.
- place_short(robot, o): place the held object on the SHORT top shelf section.

Two hard geometric constraints that make this a planning problem (both are enforced by real
collision, and are the reason a name-only plan can be infeasible):
1. HEIGHT: a TALL block is too tall for the short section -- place_short(tall block) collides
   the shelf board above it and fails. Tall blocks must go to the tall section; cubes fit
   either. The short/tall section each holds only so many objects before they crowd.
2. REACH-OVER: the arm reaches over nearer objects to grasp a farther one, so pick a far
   object while a nearer one is still on the floor and the reach is blocked. Store objects
   NEAREST-FIRST (south to north) so every pick's path is clear."""


def _as_episode(problem: object) -> EpisodeRecord:
    if not isinstance(problem, EpisodeRecord):
        raise TypeError(f"RestockAdapter expects an EpisodeRecord, got {type(problem)}")
    return problem


class RestockAdapter(EnvAdapter):
    """Restock3D v2 VLMPlan adapter."""

    def __init__(self, with_images: bool = True, image_width_px: int = 768):
        self.with_images = with_images
        self.image_width_px = image_width_px

    # --- vocabulary ---------------------------------------------------------
    def skills(self, problem: object) -> dict[str, SkillSpec]:
        return {n: SkillSpec(n, t) for n, t in _SKILL_TYPES.items()}

    def objects(self, problem: object) -> dict[str, str]:
        return dict(_as_episode(problem).object_registry)

    def type_ancestors(self, type_name: str) -> frozenset[str]:
        return frozenset({type_name})  # flat type set

    # --- prompt content -----------------------------------------------------
    def controllers_str(self, problem: object) -> str:
        lines = ["Skills (ParameterizedController):"]
        for n, ts in _SKILL_TYPES.items():
            args = ", ".join(f"?a{i} - {t}" for i, t in enumerate(ts))
            lines.append(f"  ({n} {args})")
        return "\n".join(lines) + "\n\n" + _CONTROLLER_NOTE

    def typed_objects_str(self, problem: object) -> str:
        return "\n".join(
            f"{n}: {t}" for n, t in sorted(_as_episode(problem).object_registry.items())
        )

    def type_hierarchy_str(self, problem: object) -> str:
        return f"{_CUBOID_TYPE}\n{_ROBOT_TYPE}"  # flat

    def goal_str(self, problem: object) -> str:
        ep = _as_episode(problem)
        goals = sorted({o.name for a in ep.goal_atoms for o in a.objects})
        return "Goal: every object is Stored on a shelf section:\n  " + ", ".join(
            f"Stored({g})" for g in goals
        )

    def init_state_str(self, problem: object) -> str:
        ep = _as_episode(problem)
        lines = ["Initial state (abstract literals):"]
        for a in sorted(
            ep.initial_abstract_state.atoms, key=lambda x: (x.predicate.name,)
        ):
            lines.append(
                "  "
                + a.predicate.name
                + "("
                + ", ".join(o.name for o in a.objects)
                + ")"
            )
        # Geometry disclosure: each object's height class + floor position (y = northward
        # reach; nearest-first order clears the reach path).
        geo = ep.scene_geometry
        if geo is not None:
            lines.append("\nObject geometry (height class + floor position x,y):")
            for o in sorted(geo.objects, key=lambda g: g.name):
                if o.name == "robot":
                    continue
                cls = (
                    "TALL (tall section only)"
                    if o.family == "tall"
                    else "cube (either section)"
                )
                lines.append(
                    f"  {o.name}: {cls}, at (x={o.pose[0]:.2f}, y={o.pose[1]:.2f})"
                )
        return "\n".join(lines)

    def images(self, problem: object) -> list[Image]:
        if not self.with_images:
            return []
        ep = _as_episode(problem)
        stratum = int((ep.provenance.gen_params or {}).get("stratum", 0))
        seed = int(ep.provenance.problem_id)
        try:
            # pylint: disable=import-outside-toplevel
            from PIL import Image as PILImage

            from alphatamp.approaches.spectre.envs.restock3d import render as _render
            from alphatamp.approaches.spectre.envs.restock3d.oracle_v2 import (
                build_v2_bundle,
            )

            bundle = build_v2_bundle(stratum)
            try:
                x0, _ = bundle.sim.reset(seed=seed)
                names = [n for n in bundle.sim.movable_names()]
                h = int(self.image_width_px * 0.75)
                arr = _render.render_labeled_scene(
                    bundle.sim,
                    x0,
                    names,
                    image_width=self.image_width_px,
                    image_height=h,
                )
                return [PILImage.fromarray(arr)]
            finally:
                bundle.sim.close()
        except BaseException:  # pylint: disable=broad-exception-caught
            return []

    # --- output handling ----------------------------------------------------
    def ground(self, raw: RawPlan, problem: object) -> tuple[Step, ...] | None:
        ep = _as_episode(problem)
        goal_objs = {o.name for a in ep.goal_atoms for o in a.objects}
        movables = {n for n, t in ep.object_registry.items() if t == _CUBOID_TYPE}
        holding: str | None = None
        stored: set[str] = set()
        on_floor = set(movables)
        out: list[Step] = []
        for name, args in raw.steps:
            if name not in _SKILL_TYPES:
                return None
            if name == "pick":
                if len(args) != 2 or holding is not None:
                    return None
                obj = args[1]
                if obj not in on_floor:
                    return None
                holding = obj
                on_floor.discard(obj)
            else:  # place_tall / place_short
                if len(args) != 2 or holding is None or holding != args[1]:
                    return None
                stored.add(holding)
                holding = None
            out.append((name, tuple(args)))
        if holding is not None or not goal_objs.issubset(stored):
            return None
        return tuple(out)

    def canonical_key(self, steps: Sequence[Step]) -> tuple[object, ...]:
        # Include the op name (tall vs short is refiner-relevant: F3) + args.
        return tuple((n, tuple(a)) for n, a in steps)

    def plan_str(self, steps: Sequence[Step]) -> str:
        return "\n".join(f"  {n}(" + ", ".join(a) + ")" for n, a in steps)

    def published_order(self, problem: object) -> list[tuple[Step, ...]]:
        ep = _as_episode(problem)
        return [self._skeleton_steps(sk) for sk in ep.skeleton_pool]

    # --- scoring support ----------------------------------------------------
    def pool_index(self, problem: object) -> dict[tuple[object, ...], int]:
        ep = _as_episode(problem)
        out: dict[tuple[object, ...], int] = {}
        for i, sk in enumerate(ep.skeleton_pool):
            out.setdefault(self.canonical_key(self._skeleton_steps(sk)), i)
        return out

    def discretionary_objects(self, steps: Sequence[Step]) -> list[str]:
        return [a[1] for n, a in steps if n == "pick" and len(a) == 2]

    @staticmethod
    def _skeleton_steps(skeleton) -> tuple[Step, ...]:
        return tuple(
            (op.name, tuple(p.name for p in op.parameters))
            for op in skeleton.operator_seq
        )


class RestockOffPoolLabeler(Labeler):
    """Refine one off-pool proposal for real, using the collection's own settings.

    Reconstructs the v2 sim from the episode's ``(stratum recipe key, problem_seed)``
    and refines the proposed store order with the same manual rollout the oracle
    certifier uses (per-step resampling, 18 attempts) -- the deployed v2 refinement
    path. Memoized per (problem, canonical plan). Must clear ``score.label_agreement``
    against the stored in-pool labels before any number is trusted.
    """

    def __init__(self, memo_path=None, attempts_per_step: int = 18):
        self._memo_path = memo_path
        self._attempts = attempts_per_step
        self._memo: dict = {}

    def label(self, episode: object, steps: Sequence[Step]) -> str:
        ep = _as_episode(episode)
        key = (int(ep.provenance.problem_id), tuple((n, tuple(a)) for n, a in steps))
        if key in self._memo:
            return self._memo[key]
        result = self._refine(ep, steps)
        self._memo[key] = result
        self.n_refines += 1
        return result

    def _refine(self, ep: EpisodeRecord, steps: Sequence[Step]) -> str:
        stratum = int((ep.provenance.gen_params or {}).get("stratum", 0))
        seed = int(ep.provenance.problem_id)
        try:
            # pylint: disable=import-outside-toplevel
            from alphatamp.approaches.spectre.envs.restock3d.oracle_v2 import (
                build_v2_bundle,
                refine_skeleton_v2,
            )

            # Convert the proposed pick/place steps into the (obj, section) store order the
            # v2 refiner takes: each place_tall -> section_0, place_short -> section_1.
            order: list[tuple[str, str]] = []
            for n, a in steps:
                if n == "place_tall":
                    order.append((a[1], "section_0"))
                elif n == "place_short":
                    order.append((a[1], "section_1"))
            bundle = build_v2_bundle(stratum)
            try:
                x0, _ = bundle.sim.reset(seed=seed)
                ok, *_ = refine_skeleton_v2(
                    bundle, x0, order, seed=seed, attempts_per_step=self._attempts
                )
                return "success" if ok else "fail"
            finally:
                bundle.sim.close()
        except BaseException:  # pylint: disable=broad-exception-caught
            return "fail"
