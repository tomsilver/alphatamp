"""DD2D implementation of :class:`~.adapter.EnvAdapter` — the only env-aware module here.

The problem object is a **canonicalized** ``EpisodeRecord`` (as returned by
``eda.load_split_episodes``), which is the single source for everything: object names and
types, the STRIPS initial state and goal, the ground-truth scene geometry, the candidate
pool (for the published-order fallback and for pool matching), and the stored refinement
outcomes. Using the canonicalized record — not the raw DD2D JSON — is what keeps object
names (``item_7``) identical across the prompt, the rendered image labels, the pool
indices, and every other method's cache record.

**Geometry is disclosed in text.** DD2D's PDDL is deliberately geometry-blind (the
shortest optimistic plan is literally ``retrieve(target)``), so a prompt carrying only
literals would describe a problem in which every plan is equally good. Giving the model
per-item pose/shape/size is the DD2D analogue of KinDER's object-centric state, and it is
what makes this a fair test of *planning* rather than of clairvoyance.
"""

from __future__ import annotations

import math
from typing import Sequence

from PIL.Image import Image

from alphatamp.approaches.spectre.envs.dd2d.spectre_geometry import reconstruct_scene
from alphatamp.approaches.spectre.envs.dd2d.spectre_operators import OPERATOR_BY_NAME
from alphatamp.approaches.spectre.envs.dd2d.spectre_render import render_labeled_scene
from alphatamp.approaches.spectre.schema import EpisodeRecord, ObjectGeometry
from alphatamp.approaches.spectre.trajectory import reconstruct_trajectory

from .adapter import EnvAdapter, RawPlan, SkillSpec, Step

ITEM_TYPE = "item"

# Matches ``bilevel_planning.LiftedParameterizedController.var_str`` for a controller
# with one ``item`` argument and an empty Box, which is how the KinDER template presents
# skills. Written as a literal rather than built from gymnasium so the prompt text is
# visible in the source a reviewer reads.
_CONTROLLER_LINES = {
    "pick": "pick(types=[item]), params_space=Box([], [], (0,), float64)",
    "place-buffer": (
        "place-buffer(types=[item]), params_space=Box([], [], (0,), float64)"
    ),
    "retrieve": "retrieve(types=[item]), params_space=Box([], [], (0,), float64)",
}

# Deviation 7 (prompts/PROVENANCE.md): what each skill DOES.
#
# The KinDER template lists controller signatures only, relying on the names being
# self-descriptive. In DD2D they are not: `pick` and `retrieve` both plausibly mean "get
# that item out", and the local model duly ended 28/28 otherwise-valid plans with
# `pick(target)` instead of `retrieve(target)`. Every other method in the comparison
# reads these operators' preconditions and effects from the domain, so stating them here
# removes a handicap rather than granting an advantage — the PDDL domain in words, not a
# hint about which subset to stage.
_CONTROLLER_NOTE = """\
What each skill does (these are the exact rules the low-level executor enforces):
- pick(o): lift item o out of the drawer into the gripper. Requires the gripper to be
  empty and o to still be in the drawer. The gripper holds at most one item.
- place-buffer(o): set the held item o down on the buffer. Requires that you are holding
  o. Afterwards the gripper is empty again and o occupies space on the buffer.
- retrieve(o): take the target item o out of the drawer, completing the task. Requires the
  gripper to be empty and o to still be in the drawer. This is the ONLY skill that
  achieves the goal.

Therefore every plan is a sequence of pick/place-buffer PAIRS (a pick must always be
followed immediately by place-buffer for the same item) and ends with a single
retrieve(<target>). Never pick the target, and never put anything after the retrieve.

All three skills take one item and no continuous parameters, so each call ends with an
empty pair of square brackets. Where exactly an item lands on the buffer is chosen
downstream by a motion-planning sampler, not by you."""

# Deviation 4 (prompts/PROVENANCE.md). The PDDL is geometry-blind by construction, so
# nothing in the formal inputs conveys the domain's central fact. The trained methods
# absorb it from labels; stating it in words is disclosure of omitted semantics, not
# leakage of the answer.
_SEMANTICS_DISCLOSURE = (
    "Note on this domain: items can physically obstruct the gripper's access to other "
    "items, and each item you move to the buffer takes up buffer space that later items "
    "must fit around. So both WHICH items you stage and the ORDER you stage them in can "
    "decide whether a plan is physically executable, and two plans that stage the same "
    "items in a different order are different plans."
)


def _steps_of(operator_seq: Sequence[object]) -> tuple[Step, ...]:
    """A pooled skeleton's operator sequence as plain ``(name, args)`` steps."""
    steps: list[Step] = []
    for op in operator_seq:
        name = getattr(op, "name")
        params = getattr(op, "parameters")
        steps.append((str(name), tuple(str(p.name) for p in params)))
    return tuple(steps)


def _extent(geom: ObjectGeometry) -> tuple[float, float]:
    """Item-frame bounding-box width and height, in cm."""
    xs = [p[0] for p in geom.boundary]
    ys = [p[1] for p in geom.boundary]
    return max(xs) - min(xs), max(ys) - min(ys)


def _sort_key(name: str) -> tuple[int, str]:
    """Numeric-aware ordering so ``item_2`` precedes ``item_10`` in the prompt."""
    tail = name.rsplit("_", 1)[-1]
    return (int(tail), name) if tail.isdigit() else (1 << 30, name)


class DD2DAdapter(EnvAdapter):
    """VLMPlan adapter for DD2D drawer-decluttering episodes.

    ``with_images=False`` gives the text-only (LLMPlan) arm from the identical prompt —
    KinDER's controlled +/-image pair, kept available even though only the image arm is
    built out here.
    """

    def __init__(self, with_images: bool = True, image_width_px: int = 1024) -> None:
        self._with_images = with_images
        self._image_width_px = image_width_px

    # --- vocabulary ------------------------------------------------------------

    def skills(self, problem: object) -> dict[str, SkillSpec]:
        del problem  # DD2D's skill set is fixed across episodes
        return {
            name: SkillSpec(name=name, types=(ITEM_TYPE,), num_params=0)
            for name in _CONTROLLER_LINES
        }

    def objects(self, problem: object) -> dict[str, str]:
        episode = _as_episode(problem)
        return dict(episode.object_registry)

    def type_ancestors(self, type_name: str) -> frozenset[str]:
        return frozenset({ITEM_TYPE}) if type_name == ITEM_TYPE else frozenset()

    # --- prompt content --------------------------------------------------------

    def controllers_str(self, problem: object) -> str:
        del problem
        return "\n".join(_CONTROLLER_LINES.values()) + "\n\n" + _CONTROLLER_NOTE

    def typed_objects_str(self, problem: object) -> str:
        objects = self.objects(problem)
        names = sorted(objects, key=_sort_key)
        return "\n".join(f"{name}: {objects[name]}" for name in names)

    def type_hierarchy_str(self, problem: object) -> str:
        del problem
        return ITEM_TYPE

    def goal_str(self, problem: object) -> str:
        episode = _as_episode(problem)
        goal = " ".join(sorted(str(atom) for atom in episode.goal_atoms))
        target = self.target_name(problem)
        return (
            f"{goal}\n\n"
            f"In words: retrieve {target} out of the drawer. {target} is the item drawn "
            f"in RED in the image. It currently cannot be grasped, because neighbouring "
            f"items block every place the two-finger gripper could close on it. To make "
            f"it graspable you may first pick some other items and place them on the "
            f"buffer.\n\n{_SEMANTICS_DISCLOSURE}"
        )

    def init_state_str(self, problem: object) -> str:
        episode = _as_episode(problem)
        literals = "\n".join(
            sorted(
                (str(atom) for atom in episode.initial_abstract_state.atoms), key=str
            )
        )
        return f"{literals}\n\n{self._geometry_str(problem)}"

    def _geometry_str(self, problem: object) -> str:
        """Per-item pose/shape/size plus the container extents, as a table."""
        episode = _as_episode(problem)
        geometry = episode.scene_geometry
        if geometry is None:
            return "(No geometry recorded for this problem.)"
        frame = geometry.frame or {}
        width = float(frame.get("drawer_w", 0.0))
        depth = float(frame.get("drawer_d", 0.0))
        buffer_bounds = next(
            (c.bounds for c in geometry.containers if c.kind == "buffer"), None
        )
        header = [
            "Geometry (all lengths in centimetres, angles in degrees):",
            f"- The drawer interior spans x in [0.00, {width:.2f}], "
            f"y in [0.00, {depth:.2f}], enclosed by a 1.5 cm wall.",
        ]
        if buffer_bounds is not None:
            x0, y0, x1, y1 = (float(v) for v in buffer_bounds)
            header.append(
                f"- The buffer is an open rectangle spanning x in [{x0:.2f}, {x1:.2f}], "
                f"y in [{y0:.2f}, {y1:.2f}] (area {(x1 - x0) * (y1 - y0):.1f}). Staged "
                f"items must all fit inside it without overlapping."
            )
        header.append(
            "- Each item below is given as: name, shape family, bounding box "
            "(width x height), footprint area, whether its outline is concave, its "
            "centre position, and its rotation."
        )
        rows: list[str] = []
        for geom in sorted(geometry.objects, key=lambda g: _sort_key(g.name)):
            w, h = _extent(geom)
            x, y, theta = (float(v) for v in geom.pose)
            rows.append(
                f"{geom.name}: {geom.family}, "
                f"{w:.1f} x {h:.1f}, area {geom.area:.1f}, "
                f"{'concave' if geom.concave else 'convex'}, "
                f"centre ({x:.1f}, {y:.1f}), "
                f"rotated {math.degrees(theta) % 360.0:.0f} deg"
                f"{'   <-- THE TARGET' if geom.is_target else ''}"
            )
        return "\n".join(header) + "\n" + "\n".join(rows)

    def images(self, problem: object) -> list[Image]:
        episode = _as_episode(problem)
        if not self._with_images or episode.scene_geometry is None:
            return []
        scene = reconstruct_scene(episode.scene_geometry)
        return [render_labeled_scene(scene, width_px=self._image_width_px)]

    # --- output handling -------------------------------------------------------

    def ground(self, raw: RawPlan, problem: object) -> tuple[Step, ...] | None:
        """Reject anything the symbolic model says is inapplicable or goal-missing.

        Runs the same ``reconstruct_trajectory`` precondition check the DD2D converter
        uses to validate the collected pool, so a VLM proposal is held to exactly the
        standard a planner-emitted skeleton already meets.
        """
        episode = _as_episode(problem)
        try:
            operators = [
                OPERATOR_BY_NAME[name].ground(
                    tuple(_object_by_name(episode)[arg] for arg in args)
                )
                for name, args in raw.steps
            ]
        except KeyError:
            return None
        try:
            states = reconstruct_trajectory(
                episode.initial_abstract_state, operators, verify_preconditions=True
            )
        except AssertionError:
            return None
        if not episode.goal_atoms.issubset(states[-1].atoms):
            return None
        return raw.steps

    def canonical_key(self, steps: Sequence[Step]) -> tuple[object, ...]:
        """The ordered step tuple.

        Order is load-bearing: a blocker can block another blocker, and each staged item
        eats buffer space the next one must fit around, so the same item *set* staged in
        a different order is a genuinely different plan with a different label.
        Deduplicating on the unordered set would delete real distinctions.
        """
        return tuple((name, tuple(args)) for name, args in steps)

    def plan_str(self, steps: Sequence[Step]) -> str:
        """Render a plan back into the template's line format."""
        return "; ".join(
            f"{name}({', '.join(f'{a}:{ITEM_TYPE}' for a in args)})[]"
            for name, args in steps
        )

    def published_order(self, problem: object) -> list[tuple[Step, ...]]:
        """The pool in planner order — index j is exactly cache index j."""
        episode = _as_episode(problem)
        return [_steps_of(sk.operator_seq) for sk in episode.skeleton_pool]

    # --- DD2D-specific helpers used by score.py --------------------------------

    def target_name(self, problem: object) -> str:
        """Name of the retrieval target."""
        episode = _as_episode(problem)
        if episode.scene_geometry is not None:
            for geom in episode.scene_geometry.objects:
                if geom.is_target:
                    return geom.name
        for atom in episode.goal_atoms:
            if atom.predicate.name == "extracted":
                return str(atom.objects[0].name)
        raise ValueError("no target found on episode")

    def pool_index(self, problem: object) -> dict[tuple[object, ...], int]:
        """Canonical key -> pool index, for matching proposals against stored labels."""
        episode = _as_episode(problem)
        return {
            self.canonical_key(_steps_of(sk.operator_seq)): j
            for j, sk in enumerate(episode.skeleton_pool)
        }

    def discretionary_objects(self, steps: Sequence[Step]) -> list[str]:
        """The items staged to the buffer, in staging order.

        Renamed from ``staged_members`` on 2026-08-01 when the scorer became
        env-agnostic: "staged members" is a DD2D noun, and the shared code needed a name
        that means the same thing on an environment with no buffer.
        """
        return [args[0] for name, args in steps if name == "place-buffer"]

    # Back-compat alias: `staged_members` is the name the DD2D notebook's diagnostics and
    # the archived comparison script use.
    staged_members = discretionary_objects


def _as_episode(problem: object) -> EpisodeRecord:
    if not isinstance(problem, EpisodeRecord):
        raise TypeError(f"DD2DAdapter expects an EpisodeRecord, got {type(problem)}")
    return problem


def _object_by_name(episode: EpisodeRecord) -> dict[str, object]:
    """Name -> the ``Object`` instances the pool's ground operators were built from."""
    objects: dict[str, object] = {}
    for skeleton in episode.skeleton_pool:
        for op in skeleton.operator_seq:
            for param in op.parameters:
                objects.setdefault(param.name, param)
    return objects
