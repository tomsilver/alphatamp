"""Convert ``stickbutton2d_v1`` -> ``stickbutton2d_v1_kinder``: same records, kinder
pixels.

The PIGINet baseline's SB2D image crops were rasterised by a custom schematic
(``piginet/sb2d_adapter.SB2DDomain.crops`` -- one object drawn as a lone polygon on a
blank background). For research validity the pixel input a *model* consumes should come
from the environment's own renderer. This converter produces a new env_variant whose
image features come from **kinder's built-in renderer** instead, while every symbolic /
plan / timing / geometry field is copied **verbatim** from the v1 record.

Per problem it materialises two artifacts (the layout DD2D also writes -- per-object
crops plus a ``scene.png`` overview, ``envs/dd2d/dd2d/record_ext.py``)::

    raw/<dst>/<split>/images/<pid>/scene.png    -- full initial scene (env.render())
    raw/<dst>/<split>/images/<pid>/<obj>.png    -- per-object crop (render_2dstate)

**Reconstruct, never regenerate.** The record's plans/outcomes/geometry are not
recomputed; only the pixels are re-rendered by resetting the env from the stored seed
(``env.reset(seed=problem_id)``), which is deterministic on StickButton2D and is the one
sanctioned exception to the rule (see ``vlmplan/sb2d_label.py``). Only
``provenance.env_variant`` is updated; nothing else in the record changes.

Downstream: ``piginet/sb2d_adapter.SB2DKinderDomain`` reads these PNGs. SPECTRE is
image-free (it consumes vector ``scene_geometry``) and is unaffected. Usage::

    python experiments/spectre/sb2d_render_convert.py
    python experiments/spectre/sb2d_render_convert.py splits=[test] overwrite=true
    bash experiments/spectre/spectre_run.sh sb2d_kinder_convert \
        python experiments/spectre/sb2d_render_convert.py
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")  # headless; select before any pyplot import (kinder.envs.utils)
# pylint: disable=wrong-import-position
import hydra  # noqa: E402
import kinder  # noqa: E402
from kinder.envs.utils import render_2dstate  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from PIL import Image  # noqa: E402

from alphatamp.approaches.spectre.collect import episode_path  # noqa: E402
from alphatamp.approaches.spectre.env_registry import (  # noqa: E402
    register_extra_envs,
)
from alphatamp.approaches.spectre.envs.stickbutton2d.strata import (  # noqa: E402
    decode,
    env_id,
)
from alphatamp.approaches.spectre.io import (  # noqa: E402
    atomic_write_pickle_gz,
    list_episodes,
    load_episode,
)

#: World-units side of each per-object crop window. Kept equal to
#: ``piginet/sb2d_adapter._CROP_WORLD`` (asserted by ``test_piginet_sb2d_kinder``) so the
#: kinder crops frame each object at the same scale the schematic did -- the only thing
#: that changes is *what renders inside the window*. Larger is harmless; CLIP resizes to
#: 224 internally, so the crop's pixel size is immaterial.
CROP_WORLD = 1.4


def _atomic_save_png(img: Image.Image, path: Path) -> None:
    """Write a PNG via ``<path>.tmp`` + rename, mirroring
    ``io.atomic_write_pickle_gz``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    img.save(tmp, format="PNG")
    os.replace(tmp, path)


def _render_problem_images(env, episode, images_dir: Path, overwrite: bool) -> None:
    """Render ``scene.png`` + one crop per object into ``images_dir``.

    Idempotent per file (skip-if-exists unless ``overwrite``) so a killed run resumes
    without redoing finished PNGs. The record is written by the caller *after* this
    returns, so a record on disk always implies its images are complete.
    """
    geo = episode.scene_geometry
    assert geo is not None, "SB2D episode has no scene_geometry to render from"
    scene_path = images_dir / "scene.png"
    obj_paths = {o.name: images_dir / f"{o.name}.png" for o in geo.objects}
    need_scene = overwrite or not scene_path.exists()
    need_objs = [o for o in geo.objects if overwrite or not obj_paths[o.name].exists()]
    if not need_scene and not need_objs:
        return

    env.reset(seed=int(episode.provenance.problem_id))
    if need_scene:
        _atomic_save_png(Image.fromarray(env.render()), scene_path)
    if need_objs:
        # Exactly what kinder's ObjectCentricKinematic2DRobotEnv.render() draws: the
        # current object-centric state merged with the constant walls/table. We call the
        # lower-level render_2dstate with a per-object *window* (rather than the full
        # world bounds env.render() uses) so each crop is a native, correctly-scaled view
        # that keeps real local context -- neighbours, stick, the table band, the wall.
        oc = env.unwrapped._object_centric_env  # pylint: disable=protected-access
        state = oc._current_state.copy()  # pylint: disable=protected-access
        state.data.update(oc.initial_constant_state.data)
        cache = oc._static_object_body_cache  # pylint: disable=protected-access
        dpi = int(oc.config.render_dpi)
        half = CROP_WORLD / 2.0
        for obj in need_objs:
            cx, cy = float(obj.pose[0]), float(obj.pose[1])
            rgb = render_2dstate(
                state, cache, cx - half, cx + half, cy - half, cy + half, dpi
            )
            _atomic_save_png(Image.fromarray(rgb), obj_paths[obj.name])


def _fmt_elapsed(sec: float) -> str:
    """``mMsSS`` -- the ``[\\dhms]+`` shape ``spectre_status`` parses as elapsed."""
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s"


def _convert_split(
    data_root: Path,
    src_variant: str,
    dst_variant: str,
    split: str,
    overwrite: bool,
    heartbeat_s: float = 15.0,
) -> tuple[int, int, int]:
    """Convert one split.

    Returns ``(written, skipped, failed)``.
    """
    src_dir = data_root / "raw" / src_variant / split
    paths = list_episodes(src_dir)
    if not paths:
        print(f"[{split}] no episodes under {src_dir}; skipping")
        return (0, 0, 0)

    env_cache: dict[int, object] = {}  # one env per button count, reset per problem
    written = skipped = failed = 0
    start = time.time()
    last_hb = start
    for i, path in enumerate(paths):
        try:
            episode = load_episode(path)
            if episode.scene_geometry is None:
                raise ValueError("no scene_geometry (would be dropped by train/score)")
            pid = int(episode.provenance.problem_id)
            out = episode_path(data_root, dst_variant, split, pid)
            if out.exists() and not overwrite:
                skipped += 1
            else:
                _split, num_buttons, _index = decode(pid)
                if num_buttons not in env_cache:
                    env_cache[num_buttons] = kinder.make(
                        env_id(num_buttons), render_mode="rgb_array"
                    )
                images_dir = (
                    data_root / "raw" / dst_variant / split / "images" / str(pid)
                )
                _render_problem_images(
                    env_cache[num_buttons], episode, images_dir, overwrite
                )
                new_ep = replace(
                    episode,
                    provenance=replace(episode.provenance, env_variant=dst_variant),
                )
                atomic_write_pickle_gz(new_ep, out)
                written += 1
        except Exception as exc:  # pylint: disable=broad-exception-caught
            failed += 1
            print(f"[{split}] {path.name} FAILED: {type(exc).__name__}: {exc}")

        now = time.time()
        if now - last_hb >= heartbeat_s or i + 1 == len(paths):
            done = i + 1
            rate = done / (now - start) if now > start else 0.0
            eta_m = ((len(paths) - done) / rate / 60.0) if rate > 0 else 0.0
            # `[split] elapsed | kept done/total` -- the shape spectre_status parses.
            print(
                f"  [{split}] {_fmt_elapsed(now - start)} | kept {done}/{len(paths)}"
                f"  (wrote {written}, skip {skipped}, fail {failed}) | ETA {eta_m:.1f}m",
                flush=True,
            )
            last_hb = now

    for env in env_cache.values():
        env.close()  # type: ignore[attr-defined]
    print(
        f"[{split}] {len(paths)} episodes: wrote {written}, skip {skipped},"
        f" fail {failed}"
    )
    return (written, skipped, failed)


@hydra.main(config_path="conf", config_name="sb2d_render_convert", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint: v1 -> v1_kinder record copy with kinder-rendered images."""
    register_extra_envs()
    data_root = Path(cfg.data_root)
    src_variant = str(cfg.src_variant)
    dst_variant = str(cfg.dst_variant)
    overwrite = bool(cfg.overwrite)
    splits = cast(list, OmegaConf.to_container(cfg.splits, resolve=True))

    print(
        f"SB2D kinder-render convert: {src_variant} -> {dst_variant} "
        f"(data_root={data_root}, splits={splits}, overwrite={overwrite})"
    )
    totals = [0, 0, 0]
    for split in splits:
        w, s, f = _convert_split(
            data_root, src_variant, dst_variant, str(split), overwrite
        )
        totals[0] += w
        totals[1] += s
        totals[2] += f
    print(f"TOTAL: wrote {totals[0]}, skip {totals[1]}, fail {totals[2]}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
