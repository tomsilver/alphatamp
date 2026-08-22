"""Restock3D v2's :class:`~.domain.PIGINetDomain` (continuous packing).

Mirrors ``sb2d_adapter``: one example per (episode, candidate) from the **collected
``EpisodeRecord`` pickles**, labelled by ``outcomes[i].outcome == "success"`` -- the identical
labels SPECTRE trains on. Two things differ from SB2D:

* **Crops carry height.** Restock3D's decisive feature is that a cube and a tall block share a
  2D footprint and differ only in height, so a top-down / schematic crop is blind to the F3
  axis. The crop source is therefore the env's own **oblique** render
  (``envs/restock3d/render.object_crops``), reconstructed from the problem seed
  (``reconstruct, never regenerate``: the seed + committed recipe fully determine the scene).
* The shape scalars stay 2D ``[w, h, area, concave]`` (PIGINet's encoders are unchanged); the
  height signal reaches the network through the *image* channel, which is the whole point of a
  low-level image predictor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from alphatamp.approaches.spectre.baselines.piginet.dataset import POSE_PREDICATE
from alphatamp.approaches.spectre.baselines.piginet.record import PIGINetExample

#: Restock3D v2 vocabulary -> NL glosses for the frozen CLIP-text encoder. Operators /
# predicates come from ``envs/restock3d/models_v2``; object args reach the net through the
# object channel, not text.
GLOSSES: dict[str, str] = {
    # operators
    "pick": "reach in and pick up an object from the floor",
    "place_tall": "place the held object onto the tall bottom shelf section",
    "place_short": "place the held object onto the short top shelf section",
    "place_buffer": "set the held object down on a temporary floor buffer spot",
    # predicates
    "HandEmpty": "the robot gripper is empty and holding nothing",
    "Holding": "the robot is currently holding this object",
    "OnFloor": "this object is still sitting on the floor, not yet stored",
    "Stored": "this object has been placed onto a shelf section",
    "OnBuffer": "this object is on a temporary floor buffer spot",
    POSE_PREDICATE: "an object is located at this position and orientation",
    # object families / types
    "cube": "a short cube that fits either shelf section",
    "tall": "a tall block that only fits the taller bottom section",
    "clutter": "a small movable clutter block",
    "robot": "a mobile robot with an arm that stores objects on a shelf",
    "Kinematic3DCuboid": "a movable box-shaped object",
    "Kinematic3DRobot": "a mobile manipulation robot",
}

VOCAB: list[str] = sorted(GLOSSES)


def _config_scales(env_variant: str) -> tuple[tuple[float, float], np.ndarray]:
    """``(frame_extent, shape_max)`` from the env config (matches the scene-geometry
    frame).

    v3 varies the per-object footprint width (v2 keeps it constant), so its
    ``shape_max`` width/area divisors come from ``feasibility_v3.WIDTH_MAX`` -- a valid
    upper bound over the sampled widths (and the fixed 0.05 depth) -- rather than v2's
    single tall-block footprint. The scene frame (shelf) is unchanged across versions.
    """
    # pylint: disable=import-outside-toplevel
    from alphatamp.approaches.spectre.envs.restock3d.kinematic_env import (
        Restock3DEnvConfig,
    )

    is_v3 = env_variant.startswith("restock3d_v3")
    if is_v3:
        from alphatamp.approaches.spectre.envs.restock3d.generator_v3 import v3_config

        cfg = v3_config()
    else:
        cfg = Restock3DEnvConfig()
    frame = (
        float(cfg.shelf_width),
        float(cfg.shelf_pose.position[1]) + float(cfg.shelf_depth),
    )
    # (w, h, area, concave) divisors for the 2D footprint scalars.
    if is_v3:
        from alphatamp.approaches.spectre.envs.restock3d.feasibility_v3 import WIDTH_MAX

        w = float(
            WIDTH_MAX
        )  # per-object widths up to WIDTH_MAX; depth 0.05 < WIDTH_MAX
    else:
        w = 2.0 * float(cfg.tall_half[0])  # v2: fixed 0.05 x 0.05 footprint
    shape_max = np.array([w, w, w * w, 1.0], dtype=np.float32)
    return frame, shape_max


class RestockDomain:
    """Restock3D v2: metres, pickles + oblique-rendered crops."""

    name = "restock3d"

    def __init__(self, data_root: str | Path, env_variant: str = "restock3d_v2_pilot"):
        self.env_variant = env_variant
        self.root = Path(data_root) / "raw" / env_variant
        self._frame, self._shape_max = _config_scales(env_variant)
        self._geo_cache: dict[str, dict] = {}
        self._stratum_cache: dict[str, int] = {}
        # One reusable reconstruction sim per stratum, reset per seed. Building a fresh
        # PyBullet env for every problem leaks ~0.14 GB/env (connections are not fully
        # freed on close), which OOM'd the CLIP cache build over v3's 600 episodes. The env
        # rebuilds its bodies on reset(seed), so a single env per stratum reconstructs every
        # scene faithfully at a bounded 4-connection cost.
        self._bundle_cache: dict[int, object] = {}

    @property
    def vocab(self):
        return VOCAB

    def gloss(self, word: str) -> str:
        return GLOSSES.get(word, word)

    @property
    def frame_extent(self) -> tuple[float, float]:
        return self._frame

    @property
    def shape_max(self) -> np.ndarray:
        return self._shape_max

    def _episodes(self, split: str):
        # pylint: disable=import-outside-toplevel
        from alphatamp.approaches.spectre.io import list_episodes, load_episode

        for path in list_episodes(self.root / split):
            yield load_episode(path)

    @staticmethod
    def _problem_id(episode) -> str:
        # `_s` separator so the cache driver's `pid.split("_s")[-1]` recovers the integer.
        return f"restock_s{episode.provenance.problem_id}"

    @staticmethod
    def _objects(episode) -> list[dict]:
        geo = episode.scene_geometry
        assert geo is not None, "Restock3D PIGINet needs scene_geometry"
        out = []
        for obj in geo.objects:
            ring = np.asarray(obj.boundary, dtype=np.float32)
            width = float(ring[:, 0].max() - ring[:, 0].min()) if len(ring) else 0.0
            depth = float(ring[:, 1].max() - ring[:, 1].min()) if len(ring) else 0.0
            out.append(
                {
                    "name": obj.name,
                    "category": episode.object_registry.get(obj.name, obj.family),
                    "color": obj.family,
                    "pose": [
                        float(obj.pose[0]),
                        float(obj.pose[1]),
                        float(obj.pose[2]),
                    ],
                    "shape": {
                        "w": width,
                        "h": depth,
                        "area": float(obj.area),
                        "concave": bool(obj.concave),
                    },
                    "is_blocker": False,
                    "start_table": "",
                }
            )
        return out

    def problems(self, split: str) -> Iterator[tuple[str, list[PIGINetExample]]]:
        for episode in self._episodes(split):
            pid = self._problem_id(episode)
            objects = self._objects(episode)
            init = self._literals(episode.initial_abstract_state.atoms) + [
                [POSE_PREDICATE, o["name"]] for o in objects
            ]
            goal = self._literals(episode.goal_atoms)
            stratum = int((episode.provenance.gen_params or {}).get("stratum", 0))
            examples = []
            for i, (skel, out) in enumerate(
                zip(episode.skeleton_pool, episode.outcomes)
            ):
                examples.append(
                    PIGINetExample(
                        problem_id=pid,
                        objects=objects,
                        init_literals=init,
                        goal_literals=goal,
                        task_plan=[
                            [op.name] + [p.name for p in op.parameters]
                            for op in skel.operator_seq
                        ],
                        label=(out.outcome == "success"),
                        label_source="spectre_collection",
                        refine={"outcome": out.outcome},
                        images=[
                            {"object": o["name"], "view": "oblique"} for o in objects
                        ],
                        provenance={
                            "plan_idx": i,
                            "drawer_wh": list(self._frame),
                            "stratum": stratum,
                            "problem_seed": int(episode.provenance.problem_id),
                        },
                    )
                )
            if examples:
                self._geo_cache[pid] = {o["name"]: o for o in objects}
                self._stratum_cache[pid] = stratum
                yield pid, examples

    @staticmethod
    def _literals(atoms) -> list[list[str]]:
        return sorted([a.predicate.name] + [o.name for o in a.objects] for a in atoms)

    def object_names(self, split: str, problem_id: str) -> list[str]:
        if problem_id not in self._geo_cache:
            list(self.problems(split))
        return list(self._geo_cache[problem_id])

    def crops(self, split: str, problem_id: str) -> dict:
        """Per-object **oblique** crops, reconstructed from the problem seed.

        The scene is rebuilt from ``(stratum recipe key, problem_seed)`` -- both committed /
        stored -- and the env's own oblique camera renders it, so a tall block occupies a
        taller silhouette than a cube. Empty ``{}`` on any failure (PIGINet zero-fills).
        """
        if problem_id not in self._stratum_cache:
            list(self.problems(split))
        stratum = self._stratum_cache.get(problem_id)
        if stratum is None:
            return {}
        seed = int(problem_id.split("_s")[-1])
        try:
            # pylint: disable=import-outside-toplevel
            from alphatamp.approaches.spectre.envs.restock3d import render as _render

            # Reuse one reconstruction sim per stratum (reset per seed) -- see _bundle_cache.
            # v3 rebuilds per-object dims from seed (oracle_v3); v2 uses the constant-dims
            # recipe (oracle_v2). Both are deterministic-from-seed, so the crop matches the
            # collected scene ("reconstruct, never regenerate").
            if stratum not in self._bundle_cache:
                if self.env_variant.startswith("restock3d_v3"):
                    from alphatamp.approaches.spectre.envs.restock3d.oracle_v3 import (
                        build_v3_bundle,
                    )

                    self._bundle_cache[stratum] = build_v3_bundle(stratum)
                else:
                    from alphatamp.approaches.spectre.envs.restock3d.oracle_v2 import (
                        build_v2_bundle,
                    )

                    self._bundle_cache[stratum] = build_v2_bundle(stratum)
            sim = self._bundle_cache[stratum].sim  # type: ignore[attr-defined]
            x0, _ = sim.reset(seed=seed)
            names = list(self._geo_cache[problem_id])
            return _render.object_crops(sim, x0, names, crop_px=96)
        except BaseException:  # pylint: disable=broad-exception-caught
            return {}


def make_restock_domain(
    data_root: str | Path, env_variant: str = "restock3d_v2_pilot"
) -> RestockDomain:
    """Build the Restock3D PIGINet domain for ``env_variant``."""
    return RestockDomain(data_root, env_variant)
