"""StickButton2D reach geometry — which buttons the bare robot can physically press.

This is the one bit of geometry the symbolic model does not carry. kinder's
``RobotPressButtonFromNothing`` / ``RobotPressButtonFromButton`` operators are
applicable to *any* button, but the robot's base has ``ZOrder.ALL`` and so collides with
the table (``ZOrder.FLOOR``): it can never drive onto the table surface. Only the arm
and gripper (``ZOrder.SURFACE``) sweep over it. A button far enough into the table is
therefore unreachable by the bare robot and can only be pressed with the stick.

See ``docs/kinder_stickbutton2d_map.md`` §3 for the full derivation. Everything here is
computed from :class:`StickButton2DEnvConfig` rather than hardcoded, so a config change
(or a differently-configured variant) cannot silently invalidate it.
"""

from __future__ import annotations

from dataclasses import dataclass

from kinder.envs.kinematic2d.stickbutton2d import StickButton2DEnvConfig

_BUTTON_PREFIX = "button"


def robot_reach_max_y(config: StickButton2DEnvConfig | None = None) -> float:
    """Largest button-centre ``y`` the bare robot can press.

    Derivation (all terms from the config, none hardcoded)::

        base-centre max y = table_y0 - base_radius
        reach             = + arm_length + gripper_width / 2 + button_radius

    The base cannot overlap the table, so its centre stops one base-radius short of the
    table's lower edge. From there the arm extends ``arm_length``, the gripper's inner
    face adds half its width, and a button is pressed as soon as its *circle* is touched,
    which buys one more button radius.

    This is the vacuum-off figure. With the vacuum on, the suction body extends a further
    ``gripper_width + suction_width / 2``, which is why the empirical boundary is very
    slightly higher. We deliberately use the conservative (smaller) value: calling a
    reachable button "needs the stick" merely adds a redundant pickup to the heuristic's
    estimate, whereas the opposite error plans a physically impossible press.
    """
    cfg = config if config is not None else StickButton2DEnvConfig()
    table_y0 = float(cfg.table_pose.y)
    return (
        table_y0
        - float(cfg.robot_base_radius)
        + float(cfg.robot_arm_length)
        + float(cfg.robot_gripper_width) / 2.0
        + float(cfg.button_radius)
    )


@dataclass(frozen=True)
class ButtonReach:
    """Per-problem classification of buttons into robot-pressable and stick-only.

    ``needs_stick`` is the set the heuristic keys on: if any of these is still unpressed
    while the hand is empty, a ``PickStick`` is unavoidable and can be counted now.

    ``robot_only`` is the symmetric set — buttons that *cannot* be pressed while holding
    the stick, implying a future ``PlaceStick``. On stock StickButton2D it is empty (the
    stick reaches the whole world, so it is never strictly necessary to put it down),
    and the corresponding heuristic term is then inert by construction rather than by
    special-casing. It exists so the contract is complete and so a variant that does
    have such buttons is handled without a code change.
    """

    needs_stick: frozenset[str]
    robot_only: frozenset[str]
    reach_max_y: float

    @property
    def all_buttons(self) -> frozenset[str]:
        """Every button named in this classification."""
        return self.needs_stick | self.robot_only


def button_positions(state: object) -> dict[str, tuple[float, float]]:
    """``{button_name: (x, y)}`` from a concrete state.

    Buttons are static (``static=1``, and no operator moves them), so reading them once
    from ``x0`` is valid for the whole episode.
    """
    out: dict[str, tuple[float, float]] = {}
    for obj in state:  # type: ignore[attr-defined]
        name = str(obj.name)
        if name.startswith(_BUTTON_PREFIX):
            out[name] = (
                float(state.get(obj, "x")),  # type: ignore[attr-defined]
                float(state.get(obj, "y")),  # type: ignore[attr-defined]
            )
    return out


def robot_start_xy(state: object) -> tuple[float, float]:
    """The robot's base position in a concrete state."""
    for obj in state:  # type: ignore[attr-defined]
        if str(obj.name) == "robot":
            return (
                float(state.get(obj, "x")),  # type: ignore[attr-defined]
                float(state.get(obj, "y")),  # type: ignore[attr-defined]
            )
    raise KeyError("no 'robot' object in state")


def classify_buttons(
    state: object, config: StickButton2DEnvConfig | None = None
) -> ButtonReach:
    """Classify each button in ``state`` by which end-effector can reach it.

    ``state`` is the ``ObjectCentricState`` from ``env_models.observation_to_state`` —
    iterating it yields the scene's objects, and ``.get(obj, "y")`` reads a feature. We
    duck-type rather than import the concrete class so this stays usable from the probe
    harnesses, which build states through several different paths.
    """
    reach = robot_reach_max_y(config)
    needs_stick: set[str] = set()
    robot_only: set[str] = set()
    for obj in state:  # type: ignore[attr-defined]
        name = str(obj.name)
        if not name.startswith(_BUTTON_PREFIX):
            continue
        if float(state.get(obj, "y")) > reach:  # type: ignore[attr-defined]
            needs_stick.add(name)
    return ButtonReach(
        needs_stick=frozenset(needs_stick),
        robot_only=frozenset(robot_only),
        reach_max_y=reach,
    )
