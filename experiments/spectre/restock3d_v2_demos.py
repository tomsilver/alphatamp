"""Render per-stratum **Restock3D v2** demos (continuous packing).

For each stratum the robot executes the v2 oracle skeleton — tall blocks to the tall section, cubes
balanced across both sections, stored SOUTH-TO-NORTH — with placement x sampled uniformly across each
section's continuous band (no discrete regions). Each pick/place is retried on failure and only a
successful attempt's frames are kept, so the video shows one clean successful episode with objects
spread across the shelf's lateral width and tall blocks on the bottom section.

To keep rendering cheap, each seed is first certified WITHOUT rendering; the first fully-successful
seed is then re-run WITH rendering (deterministic — same seed reproduces the same rollout). Writes one
``demos/v2/demo_r{stratum}.mp4`` per stratum.

Run from the repo root (venv active)::

    python experiments/spectre/restock3d_v2_demos.py --strata 0,1,2,3 --seeds 0,1,2,3
"""

from __future__ import annotations

# --- IKFast needs static LAPACK/BLAS; shim the shared libs (once, cached afterwards). ----------
import glob
import os
import pathlib

_B = os.path.expanduser("~/.cache/alphatamp_ikfast_blas")
os.environ.setdefault("LAPACK_DIR", _B)
os.environ.setdefault("BLAS_DIR", _B)
pathlib.Path(_B).mkdir(parents=True, exist_ok=True)
for _a, (_sd, _pt) in {
    "liblapack.a": ("lapack", "liblapack.so.3*"),
    "libblas.a": ("blas", "libblas.so.3*"),
}.items():
    _lk = pathlib.Path(_B) / _a
    if not (_lk.exists() or _lk.is_symlink()):
        _cs = sorted(
            glob.glob(f"/usr/lib/x86_64-linux-gnu/{_sd}/{_pt}")
            + glob.glob(f"/usr/lib/x86_64-linux-gnu/{_pt}")
        )
        _r = next((c for c in _cs if os.path.isfile(c)), None)
        if _r:
            _lk.symlink_to(_r)

import argparse

import imageio.v2 as iio
import numpy as np
from pybullet_helpers.camera import capture_image

from alphatamp.approaches.spectre.envs.restock3d.oracle_v2 import (
    build_v2_bundle,
    certify_problem,
)

_OUT = pathlib.Path("src/alphatamp/approaches/spectre/envs/restock3d/demos/v2")


def _render(sim) -> np.ndarray:
    return capture_image(
        sim.physics_client_id,
        image_width=640,
        image_height=480,
        **sim.config.get_camera_kwargs(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strata", default="0,1,2,3")
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--attempts", type=int, default=18)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    _OUT.mkdir(parents=True, exist_ok=True)
    for stratum in [int(s) for s in args.strata.split(",")]:
        bundle = build_v2_bundle(stratum)
        out = _OUT / f"demo_r{stratum}.mp4"
        chosen = None
        for seed in seeds:
            result, _ = certify_problem(bundle, seed, attempts_per_step=args.attempts)
            print(
                f"[restock3d_v2_demos] r{stratum} seed={seed}: "
                f"{'SOLVED' if result.certified_feasible else 'partial ('+result.note+')'}",
                flush=True,
            )
            if result.certified_feasible:
                chosen = seed
                break
        if chosen is None:
            print(
                f"[restock3d_v2_demos] r{stratum}: no fully-successful seed in {seeds}; "
                f"rendering seed={seeds[0]} (partial)",
                flush=True,
            )
            chosen = seeds[0]
        _, frames = certify_problem(
            bundle, chosen, attempts_per_step=args.attempts, render=_render
        )
        iio.mimsave(out, frames, fps=20, macro_block_size=16)
        print(
            f"[restock3d_v2_demos] r{stratum}: wrote {out} (seed={chosen})", flush=True
        )


if __name__ == "__main__":
    main()
