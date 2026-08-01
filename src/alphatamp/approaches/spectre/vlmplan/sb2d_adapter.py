"""StickButton2D implementation of :class:`~.adapter.EnvAdapter`.

Same shape as ``dd2d_adapter.py``, same problem object: a **canonicalized**
``EpisodeRecord``, so object names (``circle_3``) are identical in the prompt, in the
rendered image's labels, in the pool indices and in every other method's cache record.

**Geometry and reach are disclosed in text, and this is the load-bearing choice.**
StickButton2D's symbolic model is reach-blind: ``RobotPressButtonFromNothing`` is
applicable to *every* button, including ones the robot physically cannot touch, because
its base collides with the table. A prompt carrying only literals therefore describes a
problem in which every plan is equally good, and the model would be graded on
clairvoyance rather than planning. Every other method in the comparison reads this from
``scene_geometry`` (SPECTRE through its geometry tokens, PIGINet through pose/shape), so
stating it removes a handicap rather than granting an advantage.

**The skills get a plain-English note** for the same reason DD2D's do, and the SB2D
pilot showed exactly why. Left to infer meaning from the names, the model wrote
``RobotPressButtonFromNothing`` for *every* press — but pressing a button leaves the
robot standing on it, so only the first press can be the ``FromNothing`` variant.
Measured on a b3 problem: **11/11 parsed plans violated a precondition, all of them this
one.** The note therefore spells out the chaining rule, which is a precondition the PDDL
domain already states and which every other method in the comparison reads from that
domain for free — so stating it removes a handicap rather than granting an advantage.
Recorded as a deviation in ``prompts/PROVENANCE.md``.
"""

from __future__ import annotations

from typing import Sequence

from PIL.Image import Image

from alphatamp.approaches.spectre.envs.stickbutton2d.geometry import robot_reach_max_y
from alphatamp.approaches.spectre.envs.stickbutton2d.render import render_labeled_scene
from alphatamp.approaches.spectre.schema import EpisodeRecord
from alphatamp.approaches.spectre.trajectory import reconstruct_trajectory

from .adapter import EnvAdapter, RawPlan, SkillSpec, Step

BUTTON_TYPE = "circle"
STICK_TYPE = "rectangle"
ROBOT_TYPE = "crv_robot"

# kinder's kinematic2d hierarchy: the geometry types share a parent, the robot does not.
_PARENT = {BUTTON_TYPE: "kinematic2d", STICK_TYPE: "kinematic2d"}

_SKILL_TYPES: dict[str, tuple[str, ...]] = {
    "PickStickFromNothing": (ROBOT_TYPE, STICK_TYPE),
    "PickStickFromButton": (ROBOT_TYPE, STICK_TYPE, BUTTON_TYPE),
    "PlaceStick": (ROBOT_TYPE, STICK_TYPE),
    "RobotPressButtonFromNothing": (ROBOT_TYPE, BUTTON_TYPE),
    "RobotPressButtonFromButton": (ROBOT_TYPE, BUTTON_TYPE, BUTTON_TYPE),
    "StickPressButtonFromNothing": (ROBOT_TYPE, STICK_TYPE, BUTTON_TYPE),
    "StickPressButtonFromButton": (ROBOT_TYPE, STICK_TYPE, BUTTON_TYPE, BUTTON_TYPE),
}

_CONTROLLER_NOTE = """\
What each skill does (these are the exact rules the low-level executor enforces):
- RobotPressButtonFromNothing(robot, b): drive to b and press it with the arm. Only works
  if b is within the robot's arm reach (see the geometry below); the robot's base cannot
  drive onto the table.
- RobotPressButtonFromButton(robot, b, from_b): the same press, starting from where the
  robot is already standing over from_b.
- PickStickFromNothing(robot, stick) / PickStickFromButton(robot, stick, from_b): pick up
  the stick. Requires an empty gripper.
- StickPressButtonFromNothing(robot, stick, b) / StickPressButtonFromButton(robot, stick,
  b, from_b): press b with the tip of the held stick. This reaches buttons the bare arm
  cannot. Requires holding the stick.
- PlaceStick(robot, stick): put the stick down. Requires holding it.

**Chaining rule — this is the one that invalidates most plans if you get it wrong.**
`...FromNothing` requires the robot to be over NO button; `...FromButton` requires it to
be over `from_b`. Pressing a button leaves the robot standing on it. So in any plan:

- the FIRST press is `...FromNothing`;
- EVERY LATER press must be `...FromButton(..., b, from_b)` where `from_b` is the button
  pressed immediately before it.

A plan that uses `...FromNothing` twice in a row is inapplicable and will be discarded.
Correct: `RobotPressButtonFromNothing(robot, b1)` then
`RobotPressButtonFromButton(robot, b2, b1)` then
`RobotPressButtonFromButton(robot, b3, b2)`. The same applies to the stick presses, and
`PickStickFromButton(robot, stick, from_b)` is the pick to use when the robot is
standing on `from_b`.

The single hard constraint that makes this a planning problem: **the robot presses every
button it drives over.** A press that also sweeps a not-yet-planned button off-plan is
rejected. So the ORDER you press them in matters, and pressing a near button before a
far one you would pass over it is usually right."""


def _as_episode(problem: object) -> EpisodeRecord:
    if not isinstance(problem, EpisodeRecord):
        raise TypeError(f"SB2DAdapter expects an EpisodeRecord, got {type(problem)}")
    return problem


def _lifted_by_name(episode: EpisodeRecord) -> dict[str, object]:
    """Lifted operators recovered from the pool's own ``GroundOperator.parent``.

    Avoids rebuilding ``SesameModels`` (which would need the env) just to ground a
    proposal — the same trick ``unified_evidence.scene_filters`` uses.
    """
    return {
        op.parent.name: op.parent
        for skel in episode.skeleton_pool
        for op in skel.operator_seq
        if op.parent is not None
    }


def _objects_by_name(episode: EpisodeRecord) -> dict[str, object]:
    return {o.name: o for o in episode.initial_abstract_state.objects}


def _steps_of(operator_seq) -> tuple[Step, ...]:
    return tuple((op.name, tuple(p.name for p in op.parameters)) for op in operator_seq)


class SB2DAdapter(EnvAdapter):
    """VLMPlan's view of StickButton2D."""

    def __init__(self, with_images: bool = True, image_width_px: int = 1024) -> None:
        self._with_images = with_images
        self._image_width_px = image_width_px

    # --- vocabulary ------------------------------------------------------------
    def skills(self, problem: object) -> dict[str, SkillSpec]:
        """The seven kinder operators, as the parser validates them."""
        del problem
        return {
            name: SkillSpec(name=name, types=types)
            for name, types in _SKILL_TYPES.items()
        }

    def objects(self, problem: object) -> dict[str, str]:
        """Object name -> type, straight from the episode's registry."""
        return dict(_as_episode(problem).object_registry)

    def type_ancestors(self, type_name: str) -> frozenset[str]:
        """``type_name`` plus its kinematic2d parent, if it has one."""
        parent = _PARENT.get(type_name)
        return frozenset({type_name} | ({parent} if parent else set()))

    # --- prompt content --------------------------------------------------------
    def controllers_str(self, problem: object) -> str:
        """Skill signatures in the KinDER ``ParameterizedController`` format."""
        del problem
        lines = [
            f"{name}(types=[{', '.join(types)}]), "
            f"params_space=Box([], [], (0,), float64)"
            for name, types in _SKILL_TYPES.items()
        ]
        return "\n".join(lines) + "\n\n" + _CONTROLLER_NOTE

    def typed_objects_str(self, problem: object) -> str:
        """``<object_name>: <type_name>`` lines."""
        return "\n".join(
            f"{name}: {type_name}"
            for name, type_name in sorted(self.objects(problem).items())
        )

    def type_hierarchy_str(self, problem: object) -> str:
        """PDDL-style parent lines."""
        del problem
        return f"{BUTTON_TYPE} {STICK_TYPE} - kinematic2d\n{ROBOT_TYPE}"

    def goal_str(self, problem: object) -> str:
        """Every button pressed, plus what that means physically."""
        episode = _as_episode(problem)
        goal = " ".join(sorted(str(atom) for atom in episode.goal_atoms))
        buttons = sorted(
            n for n, t in episode.object_registry.items() if t == BUTTON_TYPE
        )
        return (
            f"{goal}\n\n"
            f"In words: press all {len(buttons)} buttons ({', '.join(buttons)}). "
            f"They are the RED circles in the image. A button stays pressed once "
            f"pressed, so you never need to press one twice — but pressing one you "
            f"had not planned to press yet invalidates the plan, because the executor "
            f"checks the world against the plan after every step."
        )

    def init_state_str(self, problem: object) -> str:
        """Literals plus the geometry and reach disclosure."""
        episode = _as_episode(problem)
        literals = "\n".join(
            sorted(str(atom) for atom in episode.initial_abstract_state.atoms)
        )
        return f"{literals}\n\n{self._geometry_str(episode)}"

    def _geometry_str(self, episode: EpisodeRecord) -> str:
        """Per-object position/size, the world frame, and the reach boundary."""
        geometry = episode.scene_geometry
        if geometry is None:
            return "(No geometry recorded for this problem.)"
        frame = geometry.frame or {}
        width = float(frame.get("frame_w", frame.get("drawer_w", 3.5)))
        depth = float(frame.get("frame_d", frame.get("drawer_d", 2.5)))
        reach = robot_reach_max_y()
        table = next((c for c in geometry.containers if c.kind == "table"), None)

        header = [
            "Geometry (lengths in metres; x runs right, y runs up):",
            f"- The world spans x in [0.00, {width:.2f}], y in [0.00, {depth:.2f}].",
        ]
        if table is not None:
            tx0, ty0, tx1, ty1 = (float(v) for v in table.bounds)
            header.append(
                f"- A table occupies x in [{tx0:.2f}, {tx1:.2f}], "
                f"y in [{ty0:.2f}, {ty1:.2f}]. The robot's BASE cannot enter it; only "
                f"the arm and the stick can pass over it."
            )
        header.append(
            f"- **Arm reach limit: y = {reach:.3f}.** A button with centre y above "
            f"this CANNOT be pressed by RobotPressButton* — it needs the stick. A "
            f"button below it can be pressed either way."
        )
        header.append("- Each object below: name, type, centre position, size.")

        rows: list[str] = []
        types = episode.object_registry
        for geom in sorted(geometry.objects, key=lambda g: g.name):
            x, y, _theta = (float(v) for v in geom.pose)
            kind = types.get(geom.name, geom.family)
            xs = [px for px, _ in geom.boundary]
            ys = [py for _, py in geom.boundary]
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            note = ""
            if kind == BUTTON_TYPE:
                note = (
                    "   <-- NEEDS THE STICK (above the reach limit)"
                    if y > reach
                    else "   <-- reachable by the bare arm"
                )
            rows.append(
                f"{geom.name}: {kind}, centre ({x:.2f}, {y:.2f}), "
                f"size {w:.2f} x {h:.2f}{note}"
            )
        return "\n".join(header) + "\n" + "\n".join(rows)

    def images(self, problem: object) -> list[Image]:
        """The labelled top-down render, or none for the text-only arm."""
        episode = _as_episode(problem)
        if not self._with_images or episode.scene_geometry is None:
            return []
        return [
            render_labeled_scene(
                episode.scene_geometry,
                episode.object_registry,
                width_px=self._image_width_px,
            )
        ]

    # --- output handling -------------------------------------------------------
    def ground(self, raw: RawPlan, problem: object) -> tuple[Step, ...] | None:
        """Reject anything the symbolic model says is inapplicable or goal-missing.

        Uses the same ``reconstruct_trajectory`` precondition check the collection's own
        pool satisfies, so a VLM proposal is held to exactly the standard a
        planner-emitted skeleton already meets — no more, no less.
        """
        episode = _as_episode(problem)
        lifted = _lifted_by_name(episode)
        objs = _objects_by_name(episode)
        try:
            operators = [
                lifted[name].ground(  # type: ignore[attr-defined]
                    tuple(objs[arg] for arg in args)
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

        Order is the whole problem here: the robot presses everything it drives over, so
        the same set of presses in a different order is a genuinely different plan with
        a different label. Deduplicating on the unordered set would delete the
        distinction the environment is built around.
        """
        return tuple((name, tuple(args)) for name, args in steps)

    def plan_str(self, steps: Sequence[Step]) -> str:
        """Render a plan back into the template's line format."""
        types = _SKILL_TYPES
        out = []
        for name, args in steps:
            arg_types = types.get(name, (BUTTON_TYPE,) * len(args))
            typed = ", ".join(f"{a}:{t}" for a, t in zip(args, arg_types))
            out.append(f"{name}({typed})[]")
        return "; ".join(out)

    def published_order(self, problem: object) -> list[tuple[Step, ...]]:
        """The pool in planner order — index j is exactly cache index j."""
        episode = _as_episode(problem)
        return [_steps_of(sk.operator_seq) for sk in episode.skeleton_pool]

    # --- scoring support -------------------------------------------------------
    def pool_index(self, problem: object) -> dict[tuple[object, ...], int]:
        """Canonical key -> pool index, for matching proposals against stored labels."""
        episode = _as_episode(problem)
        return {
            self.canonical_key(_steps_of(sk.operator_seq)): j
            for j, sk in enumerate(episode.skeleton_pool)
        }

    def discretionary_objects(self, steps: Sequence[Step]) -> list[str]:
        """The press order — the choice this plan actually made.

        DD2D's analogue is the staged-member list. Diagnostics only; never a label. The
        pressed button is argument 1 for the robot presses and argument 2 for the stick
        presses (the stick sits between the robot and the button), and the order must be
        the plan's own -- on this environment the order *is* the decision.
        """
        out: list[str] = []
        for name, args in steps:
            if name.startswith("RobotPressButton"):
                out.append(args[1])
            elif name.startswith("StickPressButton"):
                out.append(args[2])
        return out
