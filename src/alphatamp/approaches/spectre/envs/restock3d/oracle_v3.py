"""Restock3D **v3** scene-reconstruction bundle (for the PIGINet comparator).

The low-level image predictor rebuilds each collected v3 scene from ``(stratum, seed)``
to render its oblique crops. v3 generation is **deterministic from seed** (numpy RNG,
not hash-based), so a fresh :meth:`ObjectCentricRestock3DEnvV3.reset(seed)` reproduces
the exact per-object widths / heights / floor positions -- the *reconstruct, never
regenerate* path. Mirrors ``oracle_v2.build_v2_bundle``, differing only in the per-seed-
dims v3 env + v3 stratum args.

Only the sim-bearing bundle is provided (no oracle / certifier): v3 labels come from the
analytic classifier at collection time and from the real refiner at eval time, neither
of which needs a hand-built plan here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .generator_v3 import stratum_env_args_v3
from .kinematic_env import ObjectCentricRestock3DEnvV3, Restock3DEnvConfig
from .models_v2 import RestockAbstractorV2
from .region_geometry import RegionInfo


@dataclass
class V3Bundle:
    """A v3 stratum's built sim + section bands + abstractor (bodies rebuilt per
    ``reset(seed)``)."""

    sim: ObjectCentricRestock3DEnvV3
    section_infos: dict[str, RegionInfo]
    goal_names: list[str]
    abstractor: RestockAbstractorV2


def build_v3_bundle(
    stratum: int, config: Optional[Restock3DEnvConfig] = None
) -> V3Bundle:
    """Build the v3 sim + section bands + abstractor for a stratum (per-seed dims via
    reset)."""
    spec_fn, pose_fn, section_infos, config = stratum_env_args_v3(stratum, config)
    sim = ObjectCentricRestock3DEnvV3(
        spec_fn, pose_fn, section_infos, config=config, allow_state_access=True
    )
    goal_names = [n for n, _, _ in spec_fn(0)]  # constant obj_goal name set per stratum
    return V3Bundle(
        sim, section_infos, goal_names, RestockAbstractorV2(section_infos, goal_names)
    )
