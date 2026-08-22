"""Restock3D **v3** models bundle: the v2 operators/predicates/abstractor over a v3 per-object-dims
sim (:class:`kinematic_env.ObjectCentricRestock3DEnvV3`) with the left-to-right packing controllers
(``place_controller_v3.create_lifted_controllers_v3``).

v3 reuses v2's abstraction verbatim — two place operators ``place_tall``/``place_short`` with
identical abstract effects (section validated by real collision), predicates ``{HandEmpty, Holding,
OnFloor, Stored, OnBuffer}`` — so capacity/height remain invisible to the planner (the intended
false-positive source). Only the sim (per-seed dims) and the place controllers (analytic slotting)
differ; both are injected through ``build_restock3d_v2_models``.
"""

from __future__ import annotations

from gymnasium.spaces import Space

from .generator_v3 import stratum_env_args_v3
from .kinematic_env import ObjectCentricRestock3DEnvV3
from .models_v2 import RestockModelsV2, build_restock3d_v2_models
from .place_controller_v3 import create_lifted_controllers_v3


def create_restock3d_v3_models(
    observation_space: Space,
    action_space: Space,
    stratum: int,
) -> RestockModelsV2:
    """Create the Restock3D v3 models bundle for a banding stratum (0..3)."""
    spec_fn, pose_fn, section_infos, config = stratum_env_args_v3(stratum)
    sim = ObjectCentricRestock3DEnvV3(
        spec_fn, pose_fn, section_infos, config=config, allow_state_access=True
    )
    goal_names = [
        n for n, _, _ in spec_fn(0)
    ]  # all obj_goal names (constant set per stratum)
    return build_restock3d_v2_models(
        sim,
        section_infos,
        goal_names,
        observation_space,
        observation_space.devectorize,  # type: ignore[attr-defined]
        action_space,
        lifted_controllers_factory=create_lifted_controllers_v3,
    )
