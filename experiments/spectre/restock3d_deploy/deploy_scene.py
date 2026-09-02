"""Hand-specified Restock3D-v3 scenes for the real-robot proof-of-concept deployment.

The evaluation pipeline builds a scene from a ``(seed, stratum)`` pair via
``generator_v3.build_spec_v3``. For a real-robot run the scene is instead
**hand-specified**: the objects are built and measured in the physical world, and the
measurements go into a ``scene.yaml`` file (schema in the sibling ``README.md``). This
module turns that file into the exact env + models a live planning run needs.

Two things make this work without touching the collection code:

- **The scene is baked into the sim at CONSTRUCTION.** The refiner drives the models'
  own PyBullet sim; ``sim.set_state`` only moves object bodies, it never resizes them.
  The F3-critical per-object heights therefore must be present when the movable bodies
  are created (``ObjectCentricRestock3DEnvV3.__init__`` builds from ``spec_fn(0)``). We
  pass a ``spec_fn``/``pose_fn`` that ignore the seed and always return the hand scene,
  so every reset reproduces the measured geometry.
- **Any object count is supported.** The trained ranker is count-agnostic (typed local
  ids + set pooling), and every space is read off the sim we build, so ``n`` is not
  restricted to the trained ``6..9``. The per-count budget still clamps to the nearest
  trained stratum (see ``budget_for_n``); ``n`` far from ``6..9`` is out-of-distribution
  for the ranker (a warning, not an error).

Nothing here is on the collection path; it is deploy-only glue over
``models_v2.build_restock3d_v2_models`` (the assembler ``create_restock3d_v3_models``
uses too).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from alphatamp.approaches.spectre.envs.restock3d import feasibility_v3 as F
from alphatamp.approaches.spectre.envs.restock3d.generator import (
    _EXCLUSION_RADIUS,
    _OBJECT_REGION_X,
    _OBJECT_REGION_Y,
)
from alphatamp.approaches.spectre.envs.restock3d.generator_v3 import (
    _DEPTH_HALF,
    _rgba,
    v3_config,
)
from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import (
    ObjectCentricRestock3DEnv,
    ObjectCentricRestock3DEnvV3,
    ObjectSpec,
    Restock3DEnvConfig,
)
from alphatamp.approaches.spectre.envs.restock3d.models_v2 import (
    RestockModelsV2,
    build_restock3d_v2_models,
)
from alphatamp.approaches.spectre.envs.restock3d.place_controller_v3 import (
    create_lifted_controllers_v3,
)
from alphatamp.approaches.spectre.envs.restock3d.section_geometry import (
    compute_section_infos,
)

#: Default full y-depth of an object (2 * ``generator_v3._DEPTH_HALF``). The generator
#: fixes depth; a hand scene may override it per object, but the trained ranker and the
#: collected geometry assume ~this value, so a different depth is flagged as OOD.
DEFAULT_DEPTH = 2 * _DEPTH_HALF  # 0.05 m

#: The trained strata (block counts) and their real-collection per-stratum budgets. These
#: arrays live otherwise only in ``experiments/spectre/restock3d_v3_real_run_all.sh``
#: (they are NOT in any importable module -- ``strata_v3.BUDGETS`` holds the larger
#: *synthetic* budgets). ``stratum = n - 6``; ``n`` outside 6..9 clamps to the nearest.
_REAL_KMAX = {0: 35, 1: 40, 2: 135, 3: 185}
_REAL_RCAP = {0: 60.0, 1: 65.0, 2: 65.0, 3: 75.0}
_REAL_SAMPLES_PER_STEP = 6


@dataclass(frozen=True)
class DeployObject:
    """One hand-measured object: full extents (m) and floor position (robot frame)."""

    name: str
    width: float  # full x-extent
    height: float  # full z-extent
    depth: float  # full y-extent
    floor_x: float
    floor_y: float


@dataclass(frozen=True)
class DeployScene:
    """A hand-specified Restock3D-v3 scene."""

    objects: tuple[DeployObject, ...]
    config: Restock3DEnvConfig

    @property
    def n(self) -> int:
        return len(self.objects)

    @property
    def stratum(self) -> int:
        """Nearest trained stratum (``clamp(n, 6, 9) - 6``), used only for the budget."""
        return max(6, min(9, self.n)) - 6

    def blocks(self) -> list[F.Block]:
        """The feasibility-classifier view (width + height + floor xy)."""
        return [
            F.Block(o.name, o.width, o.height, o.floor_x, o.floor_y)
            for o in self.objects
        ]


def budget_for_n(n: int) -> tuple[int, float, int]:
    """``(K_max, r_cap_s, samples_per_step)`` default for a scene of ``n`` objects.

    Clamps to the nearest trained stratum's real-collection budget. Overridable by the
    caller (the deploy script exposes ``--k-max`` / ``--refinement-timeout``), mirroring
    the ``arg if not None else default`` pattern in ``restock3d_v3_collect._budget``.
    """
    stratum = max(6, min(9, n)) - 6
    return _REAL_KMAX[stratum], _REAL_RCAP[stratum], _REAL_SAMPLES_PER_STEP


def load_scene(path: str | Path) -> DeployScene:
    """Parse a ``scene.yaml`` (or a dir containing one) into a :class:`DeployScene`.

    Missing object ``name`` defaults to ``obj_goal{i}`` in file order (the names the
    trained models expect); missing ``depth`` defaults to :data:`DEFAULT_DEPTH`.
    Optional ``shelf``/``sections`` blocks override the env config (leave them out to
    match the trained env). Raises ``ValueError`` on a malformed file; geometric
    feasibility is checked separately by :func:`validate_scene`.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "scene.yaml"
    if not path.exists():
        raise ValueError(f"scene file not found: {path}")
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(raw, dict) or "objects" not in raw:
        raise ValueError(f"{path}: expected a mapping with an 'objects' list")
    obj_list = raw["objects"]
    if not isinstance(obj_list, list) or not obj_list:
        raise ValueError(f"{path}: 'objects' must be a non-empty list")

    objects: list[DeployObject] = []
    for i, entry in enumerate(obj_list, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: object #{i} is not a mapping")
        try:
            floor = entry["floor"]
            width = float(entry["width"])
            height = float(entry["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: object #{i} needs numeric 'width'/'height' and 'floor: [x, y]'"
            ) from exc
        if not (isinstance(floor, (list, tuple)) and len(floor) == 2):
            raise ValueError(f"{path}: object #{i} 'floor' must be [x, y]")
        name = str(entry.get("name") or f"obj_goal{i}")
        depth = float(entry.get("depth", DEFAULT_DEPTH))
        objects.append(
            DeployObject(name, width, height, depth, float(floor[0]), float(floor[1]))
        )

    names = [o.name for o in objects]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: object names must be unique, got {names}")

    return DeployScene(objects=tuple(objects), config=_build_config(raw))


def _build_config(raw: dict) -> Restock3DEnvConfig:
    """The v3 env config, plus optional ``shelf``/``sections`` file overrides."""
    config = v3_config()
    shelf = raw.get("shelf") or {}
    sections = raw.get("sections") or {}
    kwargs: dict[str, Any] = {}
    if raw.get("bins"):
        # Staging bins: open-top walled boxes the objects start in, so only the 45deg
        # front grasp can get them out (a horizontal side grasp is walled off). Each
        # entry gives the INNER floor extents and wall height:
        #   bins:
        #     - {x: [-0.81, -0.16], y: [0.675, 0.825], height: 0.10}
        # Walls are collision bodies; motion planning lifts grasped objects over them.
        bins = []
        for i, entry in enumerate(raw["bins"], start=1):
            try:
                (x_lo, x_hi), (y_lo, y_hi) = entry["x"], entry["y"]
                bins.append(
                    (
                        float(x_lo),
                        float(x_hi),
                        float(y_lo),
                        float(y_hi),
                        float(entry["height"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"bin #{i} needs 'x: [lo, hi]', 'y: [lo, hi]' and 'height'"
                ) from exc
        kwargs["bins"] = tuple(bins)
    if "clearances" in sections:
        cl = sections["clearances"]
        kwargs["section_clearances"] = (float(cl[0]), float(cl[1]))
        # A custom shelf also moves the analytic arm-insertion cutoffs (F3): the
        # calibrated rule is ~0.10 m of entry headroom under each section's ceiling
        # board, so the cutoffs follow the clearances unless the scene pins them
        # explicitly (``sections: {cutoffs: [tall, short]}``). The library treats
        # these as train-time constants, so for this proof-of-concept deployment we
        # rebind the module attributes; every consumer (the packing feasibility, the
        # skeleton generator, the refiner's F3 probe via ``collect``'s call-time
        # import, and this file's validator) reads them through the module.
        cutoffs = sections.get("cutoffs") or [float(cl[0]) - 0.10, float(cl[1]) - 0.10]
        F.TALL_CUTOFF = float(cutoffs[0])
        F.SHORT_CUTOFF = float(cutoffs[1])
        F.CUTOFF["tall"] = F.TALL_CUTOFF
        F.CUTOFF["short"] = F.SHORT_CUTOFF
    if shelf:
        # x/y and the bottom placement-surface height are honoured; keep the rest of
        # v3_config.
        from pybullet_helpers.geometry import Pose  # local: geometry dep

        sx = float(shelf.get("x", config.shelf_pose.position[0]))
        sy = float(shelf.get("y", config.shelf_pose.position[1]))
        kwargs["shelf_pose"] = Pose((sx, sy, 0.0))
        if "bottom_surface_z" in shelf:
            kwargs["bottom_surface_z"] = float(shelf["bottom_surface_z"])
    if not kwargs:
        return config
    from dataclasses import replace

    return replace(config, **kwargs)


def validate_scene(scene: DeployScene) -> list[str]:
    """Return human-readable warnings; an empty list means the scene looks clean.

    Warnings (never hard errors -- a real object may legitimately sit outside a soft
    bound): per-object width outside ``[WIDTH_MIN, WIDTH_MAX]`` (finger-aperture cap);
    height above ``TALL_CUTOFF`` (un-storable in either section); a non-default depth
    (out-of-distribution for the ranker); a floor position outside the sampled object
    field or closer than the exclusion radius to a neighbour; ``n`` outside the trained
    ``6..9``; and -- the important one -- **no feasible two-level split**
    (``feasible_ratio == 0`` => the problem is geometrically unsolvable).
    """
    warnings: list[str] = []
    for o in scene.objects:
        if not F.WIDTH_MIN <= o.width <= F.WIDTH_MAX:
            warnings.append(
                f"{o.name}: width {o.width:.3f} m outside trained "
                f"[{F.WIDTH_MIN}, {F.WIDTH_MAX}] (finger aperture ~0.09 m)"
            )
        if o.height > F.TALL_CUTOFF:
            warnings.append(
                f"{o.name}: height {o.height:.3f} m > TALL_CUTOFF {F.TALL_CUTOFF} "
                f"(cannot be stored in either section)"
            )
        if abs(o.depth - DEFAULT_DEPTH) > 1e-6:
            warnings.append(
                f"{o.name}: depth {o.depth:.3f} m != default {DEFAULT_DEPTH} "
                f"(collision is exact, but the ranker trained on ~{DEFAULT_DEPTH} m)"
            )
        x_lo, x_hi = _OBJECT_REGION_X
        y_lo, y_hi = _OBJECT_REGION_Y
        if not (x_lo <= o.floor_x <= x_hi and y_lo <= o.floor_y <= y_hi):
            warnings.append(
                f"{o.name}: floor ({o.floor_x:.2f}, {o.floor_y:.2f}) outside the "
                f"sampled object field x{_OBJECT_REGION_X} y{_OBJECT_REGION_Y}"
            )
    objs = scene.objects
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            dx = objs[i].floor_x - objs[j].floor_x
            dy = objs[i].floor_y - objs[j].floor_y
            if (dx * dx + dy * dy) ** 0.5 < _EXCLUSION_RADIUS:
                warnings.append(
                    f"{objs[i].name} and {objs[j].name} are closer than the exclusion "
                    f"radius {_EXCLUSION_RADIUS} m (may overlap / block the front grasp)"
                )
    if not 6 <= scene.n <= 9:
        warnings.append(
            f"n={scene.n} is outside the trained strata 6..9; the ranker runs "
            f"(count-agnostic) but is OOD, and the budget clamps to stratum "
            f"{scene.stratum}"
        )
    n_feas, _total, _rho = F.feasible_ratio(scene.blocks())
    if n_feas < 1:
        warnings.append(
            "NO FEASIBLE two-level split exists for these objects "
            "(feasible_ratio == 0): the scene is geometrically unsolvable, so no plan "
            "can succeed. Reduce widths or count, or move a tall object under "
            "TALL_CUTOFF."
        )
    return warnings


def build_deploy_models(scene: DeployScene) -> RestockModelsV2:
    """Build the sim + v3 models for a hand scene, and publish the sim to ``collect``.

    Mirrors ``models_v3.create_restock3d_v3_models`` but (a) binds a seed-independent
    ``spec_fn``/``pose_fn`` to the hand scene (so the F3 heights are baked into the
    movable bodies at construction), (b) reads the observation/action spaces off the sim
    itself (so any object count works), and (c) stamps ``collect._restock_extras`` so
    the reused ``_make_plan_generator`` / ``_make_trajectory_sampler`` find the sim,
    section infos and goal names -- exactly as ``_make_env_models`` does on the
    collection path.
    """
    # Deferred so importing this module never pulls in the collection stack (and to avoid
    # a cycle: collect imports the restock env lazily, never this module at top level).
    from alphatamp.approaches.spectre import collect as C

    config = scene.config
    section_infos = compute_section_infos(config)
    object_specs: list[ObjectSpec] = [
        (o.name, (o.width / 2.0, o.depth / 2.0, o.height / 2.0), _rgba(o.height))
        for o in scene.objects
    ]

    def spec_fn(
        _seed: int,
    ) -> list[ObjectSpec]:  # seed-independent: hand scene is fixed
        return object_specs

    def pose_fn(_seed: int) -> dict[str, tuple[float, float]]:
        return {o.name: (o.floor_x, o.floor_y) for o in scene.objects}

    sim = ObjectCentricRestock3DEnvV3(
        spec_fn, pose_fn, section_infos, config=config, allow_state_access=True
    )
    goal_names = [o.name for o in scene.objects]
    bundle = build_restock3d_v2_models(
        sim,
        section_infos,
        goal_names,
        sim.observation_space,  # ObjectCentricStateSpace; stored, never devectorized
        lambda o: o,  # observation_to_state unused here (x0 = sim.get_state())
        sim.action_space,
        lifted_controllers_factory=create_lifted_controllers_v3,
    )
    C._restock_extras.update(  # pylint: disable=protected-access
        sim=bundle.sim,
        region_infos=bundle.section_infos,
        goal_names=bundle.abstractor.goal_object_names(),
    )
    return bundle


def make_x0(sim: ObjectCentricRestock3DEnv, seed: int = 0) -> Any:
    """Reset the sim to the hand scene and return the initial ObjectCentricState."""
    sim.reset(seed=seed)
    return sim.get_state()
