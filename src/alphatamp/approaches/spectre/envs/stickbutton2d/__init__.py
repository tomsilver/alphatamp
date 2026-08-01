"""StickButton2D adapters — SPECTRE's second evaluation environment.

The environment, predicates, operators, controllers and refiner all come from upstream
(``kindergarden`` / ``kinder-bilevel-planning`` / ``bilevel-planning``); nothing here
reimplements them. This package holds only the thin adapters SPECTRE needs on top:

- :mod:`geometry` — the reach classification that tells robot-pressable buttons from
  stick-only ones. This is the one fact kinder's symbolic model does not carry.
- :mod:`heuristic` — a geometry-aware A* heuristic plus a drop-in plan generator.
  Required because ``RelationalHeuristicSearchAbstractPlanGenerator`` ignores its
  ``heuristic_name`` argument and hardcodes hff.
- :mod:`diagnostics` — a per-button achievability probe. Because the goal demands *all*
  N buttons pressed, one unpressable button voids every skeleton; this measures that
  directly in O(N) refinements instead of O(K_max).

See ``docs/kinder_stickbutton2d_map.md`` for the full substrate map and the measured
sparsity findings.
"""

from alphatamp.approaches.spectre.envs.stickbutton2d.geometry import (
    ButtonReach,
    classify_buttons,
    robot_reach_max_y,
)
from alphatamp.approaches.spectre.envs.stickbutton2d.heuristic import (
    button_count_heuristic,
    make_plan_generator,
)

__all__ = [
    "ButtonReach",
    "button_count_heuristic",
    "classify_buttons",
    "make_plan_generator",
    "robot_reach_max_y",
]
