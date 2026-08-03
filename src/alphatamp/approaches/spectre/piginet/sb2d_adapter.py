"""StickButton2D's :class:`~.domain.PIGINetDomain`.

Answers the same three questions DD2D's adapter does — vocabulary, numeric scales, how
to enumerate a split — from different storage. StickButton2D has no JSON record tree and
no rendered PNGs; it has ``EpisodeRecord`` pickles carrying ``scene_geometry``, so
examples are built from the pool and crops are **rasterised from the stored rings**.
That direction is required, not preferred: post-hoc geometry comes from stored
``scene_geometry``, never from re-running the environment (``decisions/03`` 2026-07-19,
*reconstruct, never regenerate*).

**The labels are the collection's own.** One example per (episode, candidate), labelled
by ``outcomes[i].outcome == "success"`` — the identical labels SPECTRE v3 trains and is
scored on. That is what makes the comparison a comparison rather than two numbers
produced separately.

**Two crop sources.** :class:`SB2DDomain` (the default, for ``stickbutton2d_v1``)
rasterises each object as a lone polygon on a blank background from the stored ring --
the *schematic* secondary. :class:`SB2DKinderDomain` (for ``stickbutton2d_v1_kinder``)
reads per-object crop PNGs rendered by kinder's own renderer and materialised by
``experiments/spectre/sb2d_render_convert.py``. The kinder crops carry real scene
context (neighbours, stick, the table band, the wall), so they are the deployed source;
the schematic is kept as the secondary. Pick one with :func:`make_sb2d_domain`.

**Known limitation of the schematic source, by construction.** In the schematic every
unpressed button is the same red disc of the same radius drawn on an empty background,
so its CLIP crop is pixel-identical to every other button's; the image channel can
separate only ``{button, stick, robot}`` -- information the type literals already carry.
The kinder source narrows this: it does not make two unpressed buttons *look* different
(they are identical red discs in the real env too), but a per-object crop taken from the
whole scene carries positional context the schematic threw away. ``pose`` and ``shape``
work exactly as on DD2D and remain where this environment's signal mostly lives.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterator

import numpy as np
from kinder.envs.kinematic2d.stickbutton2d import StickButton2DEnvConfig

from alphatamp.approaches.spectre.piginet.dataset import POSE_PREDICATE
from alphatamp.approaches.spectre.piginet.record import PIGINetExample

#: StickButton2D vocabulary -> colloquial NL glosses for the CLIP-text encoder (§IV-A).
#: Operators and predicates come from kinder's own model
#: (``kinder_bilevel_planning/env_models``); object arguments (``button3``,
#: ``stick``) are deliberately absent -- they reach the network through the object
#: channel, not text.
GLOSSES: dict[str, str] = {
    # operators (task-plan actions)
    "PickStickFromNothing": "reach down and pick up the stick from the floor",
    "PickStickFromButton": "pick up the stick while standing over a button",
    "PlaceStick": "put the stick back down on the floor",
    "RobotPressButtonFromNothing": (
        "drive the robot over and press a button with its arm"
    ),
    "RobotPressButtonFromButton": (
        "move from one button to another and press it with the arm"
    ),
    "StickPressButtonFromNothing": (
        "hold out the stick and press a far button with its tip"
    ),
    "StickPressButtonFromButton": (
        "move the held stick from one button to another and press it"
    ),
    # predicates (init / goal literals)
    "Pressed": "this button has been pushed down and is lit",
    "Grasped": "the robot is holding the long stick",
    "HandEmpty": "the robot gripper is empty and holding nothing",
    "AboveNoButton": "the robot is not standing over any button",
    "RobotAboveButton": "the robot arm is positioned directly over this button",
    "StickAboveButton": "the held stick is positioned directly over this button",
    # the shared pose literal (see dataset.POSE_PREDICATE)
    POSE_PREDICATE: "an object is located at this position and orientation",
    # object families / types
    "circle": "a small round push button",
    "rectangle": "a long thin wooden stick used to reach far buttons",
    "crv_robot": "a mobile robot with a circular base and an extending arm",
}

VOCAB: list[str] = sorted(GLOSSES)

#: Crop window in world units. Fixed rather than per-object so scale is *preserved*
#: across crops: a button renders as a small dot and the 1.25-long stick as a long bar,
#: which is the only visual difference the image channel can carry here. Sized to fit
#: the stick.
_CROP_WORLD = 1.4
_CROP_PX = 96

_RGB = {
    "circle": (229, 0, 0),  # button_unpressed_rgb
    "rectangle": (102, 51, 25),  # stick_rgb
    "crv_robot": (120, 120, 200),
}
_BG = (18, 18, 18)


def _config_scales() -> tuple[tuple[float, float], np.ndarray]:
    """``(frame_extent, shape_max)`` derived from the env config, never hardcoded.

    Same discipline as ``envs/stickbutton2d/geometry.py``'s reach limit: a config change
    must not silently invalidate the normalisers. Getting these wrong is the failure
    mode this whole abstraction exists for -- against DD2D's centimetre constants every
    value here underflows to ~0.
    """
    cfg = StickButton2DEnvConfig()
    frame = (
        float(cfg.world_max_x) - float(cfg.world_min_x),
        float(cfg.world_max_y) - float(cfg.world_min_y),
    )
    stick_w, stick_h = float(cfg.stick_shape[0]), float(cfg.stick_shape[1])
    robot_d = 2.0 * float(cfg.robot_base_radius)
    button_d = 2.0 * float(cfg.button_radius)
    shape_max = np.array(
        [
            max(stick_w, robot_d, button_d),
            max(stick_h, robot_d, button_d),
            max(
                stick_w * stick_h,
                math.pi * float(cfg.robot_base_radius) ** 2,
                math.pi * float(cfg.button_radius) ** 2,
            ),
            1.0,
        ],
        dtype=np.float32,
    )
    return frame, shape_max


class SB2DDomain:
    """StickButton2D: metres, a 3.5x2.5 world, pickles + rasterised crops."""

    name = "stickbutton2d"

    def __init__(self, data_root: str | Path, env_variant: str = "stickbutton2d_v1"):
        self.root = Path(data_root) / "raw" / env_variant
        self._frame, self._shape_max = _config_scales()
        self._geo_cache: dict[str, dict] = {}

    @property
    def vocab(self):
        """Glossable words, sorted -- the order that indexes the CLIP-text cache."""
        return VOCAB

    def gloss(self, word: str) -> str:
        """NL phrase for a domain word (falls back to the word itself if unglossed)."""
        return GLOSSES.get(word, word)

    @property
    def frame_extent(self) -> tuple[float, float]:
        """World ``(width, depth)`` in metres."""
        return self._frame

    @property
    def shape_max(self) -> np.ndarray:
        """``[w, h, area, concave]`` divisors in metres."""
        return self._shape_max

    # -- data ----------------------------------------------------------------
    def _episodes(self, split: str):
        # pylint: disable=import-outside-toplevel
        from alphatamp.approaches.spectre.io import list_episodes, load_episode

        for path in list_episodes(self.root / split):
            yield load_episode(path)

    @staticmethod
    def _problem_id(episode) -> str:
        # `_s` separator so the cache driver's `pid.split("_s")[-1]` recovers the
        # integer, which is the convention DD2D's `dd2d_s<seed>` ids already use.
        return f"sb2d_s{episode.provenance.problem_id}"

    @staticmethod
    def _objects(episode) -> list[dict]:
        """Object table: geometry joined with the type registry."""
        geo = episode.scene_geometry
        assert geo is not None, "SB2D PIGINet needs scene_geometry"
        out = []
        for obj in geo.objects:
            ring = np.asarray(obj.boundary, dtype=np.float32)
            width = float(ring[:, 0].max() - ring[:, 0].min())
            height = float(ring[:, 1].max() - ring[:, 1].min())
            out.append(
                {
                    "name": obj.name,
                    "category": episode.object_registry.get(obj.name, obj.family),
                    "color": obj.family,
                    "pose": [float(v) for v in obj.pose],
                    "shape": {
                        "w": width,
                        "h": height,
                        "area": float(obj.area),
                        "concave": bool(obj.concave),
                    },
                    "is_blocker": False,
                    "start_table": "",
                    # Kept for the rasteriser only; `dataset` reads `pose`/`shape` and
                    # ignores it, and SB2D examples are never serialised to disk.
                    "boundary_ring": ring.tolist(),
                }
            )
        return out

    @staticmethod
    def _literals(atoms) -> list[list[str]]:
        return sorted([a.predicate.name] + [o.name for o in a.objects] for a in atoms)

    def problems(self, split: str) -> Iterator[tuple[str, list[PIGINetExample]]]:
        """One example per candidate plan, grouped by problem."""
        for episode in self._episodes(split):
            pid = self._problem_id(episode)
            objects = self._objects(episode)
            # `at-pose` init literals are synthesised for every object, mirroring DD2D.
            # Without them PIGINet would see StickButton2D's two-atom abstract initial
            # state and no positions at all -- i.e. it would stop being a *low-level*
            # predictor, which is the entire reason it is in the comparison.
            init = self._literals(episode.initial_abstract_state.atoms) + [
                [POSE_PREDICATE, o["name"]] for o in objects
            ]
            goal = self._literals(episode.goal_atoms)
            n_buttons = int((episode.provenance.gen_params or {}).get("num_buttons", 0))
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
                            {"object": o["name"], "view": "topdown"} for o in objects
                        ],
                        provenance={
                            "plan_idx": i,
                            "drawer_wh": list(self._frame),
                            "num_buttons": n_buttons,
                        },
                    )
                )
            if examples:
                self._geo_cache[pid] = {o["name"]: o for o in objects}
                yield pid, examples

    def object_names(self, split: str, problem_id: str) -> list[str]:
        """Every object the CLIP cache must carry for this problem."""
        if problem_id not in self._geo_cache:
            list(self.problems(split))
        return list(self._geo_cache[problem_id])

    def crops(self, split: str, problem_id: str) -> dict:
        """Per-object crops rasterised from the stored boundary rings (schematic
        source).

        The secondary crop source: each object is drawn as a lone polygon on a blank
        background, so it carries no scene context. The deployed source is
        :class:`SB2DKinderDomain`, which reads kinder-rendered PNGs instead.
        """
        # pylint: disable=import-outside-toplevel
        from PIL import Image, ImageDraw

        if problem_id not in self._geo_cache:
            list(self.problems(split))
        out = {}
        for name, obj in self._geo_cache[problem_id].items():
            ring = np.asarray(obj["boundary_ring"], dtype=np.float64)
            img = Image.new("RGB", (_CROP_PX, _CROP_PX), _BG)
            half = _CROP_WORLD / 2.0
            px = (ring[:, 0] + half) / _CROP_WORLD * (_CROP_PX - 1)
            # y up in world, row down in image
            py = (half - ring[:, 1]) / _CROP_WORLD * (_CROP_PX - 1)
            ImageDraw.Draw(img).polygon(
                list(zip(px.tolist(), py.tolist())),
                fill=_RGB.get(obj["color"], (200, 200, 200)),
            )
            out[name] = img
        return out


class SB2DKinderDomain(SB2DDomain):
    """StickButton2D PIGINet domain whose crops are **kinder-rendered PNGs** on disk.

    Identical to :class:`SB2DDomain` in every respect -- vocab, normalisers, the object
    table, ``problems``/``object_names`` -- except the crop *source*: instead of drawing
    a lone polygon from the stored ring, it reads the per-object crop PNG that
    ``experiments/spectre/sb2d_render_convert.py`` materialised from kinder's own
    renderer at ``raw/<variant>/<split>/images/<pid>/<obj>.png``. Those crops carry real
    scene context, which the schematic discarded.

    A missing PNG is *omitted* from the returned dict, which ``precompute_clip_cache``
    turns into a zero vector -- the same missing-file semantics DD2D's stored-PNG path
    already has.
    """

    def crops(self, split: str, problem_id: str) -> dict:
        """Per-object crops read from the materialised kinder PNGs."""
        # pylint: disable=import-outside-toplevel
        from PIL import Image

        if problem_id not in self._geo_cache:
            list(self.problems(split))
        # `_problem_id` is `sb2d_s<int>`; the images dir is keyed by the bare int,
        # matching the layout the converter writes.
        pid_int = problem_id.split("_s")[-1]
        images_dir = self.root / split / "images" / pid_int
        out = {}
        for name in self._geo_cache[problem_id]:
            png = images_dir / f"{name}.png"
            if png.exists():
                out[name] = Image.open(png).convert("RGB")
        return out


#: Which crop source each SB2D variant uses. A ``kinder`` variant reads the PNGs
#: materialised by ``sb2d_render_convert.py``; anything else falls back to the schematic
#: rasteriser (the historical default), so :func:`make_sb2d_domain` is a strict superset
#: of constructing :class:`SB2DDomain` directly.
_SB2D_CROP_SOURCE: dict[str, str] = {
    "stickbutton2d_v1": "schematic",
    "stickbutton2d_v1_kinder": "kinder",
}


def make_sb2d_domain(
    data_root: str | Path, env_variant: str = "stickbutton2d_v1"
) -> SB2DDomain:
    """Return the SB2D PIGINet domain for ``env_variant``: kinder-crop or schematic.

    Unknown variants default to the schematic source, matching the behaviour before the
    kinder variant existed.
    """
    if _SB2D_CROP_SOURCE.get(env_variant) == "kinder":
        return SB2DKinderDomain(data_root, env_variant)
    return SB2DDomain(data_root, env_variant)
