"""Convert the DD2D raw_v2 JSON dataset into SPECTRE ``EpisodeRecord`` pickles.

DD2D ships an already-collected PIGINet-style dataset (per-problem directories
of ``NNN.json`` skeleton records). This entry point maps each problem directory
to one ``EpisodeRecord`` and writes it into the standard SPECTRE data layout so
``spectre_build_vocab.py`` / ``spectre_train.py`` consume it unchanged::

    <data_root>/raw/<env_variant>/<split>/episodes/ep_<problem_seed>.pkl.gz

To generate *fresh* DD2D data, run DD2D's own collector
(``python -m alphatamp.approaches.spectre.envs.dd2d.drawer.collect --out-root ...``)
and re-point ``raw_root`` at its output. Usage::

    python experiments/spectre/dd2d_convert.py
    python experiments/spectre/dd2d_convert.py splits=[train] overwrite=true

Existing episode files are skipped unless ``overwrite=true``. Per-problem
failures are reported, not fatal.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import hydra
from omegaconf import DictConfig, OmegaConf

from alphatamp.approaches.spectre.collect import episode_path
from alphatamp.approaches.spectre.envs.dd2d.spectre_convert import convert_problem_dir
from alphatamp.approaches.spectre.io import atomic_write_pickle_gz


def _problem_dirs(split_dir: Path) -> list[Path]:
    """DD2D problem directories under a split (``dd2d_*`` dirs only).

    Skips the split-level ``attempted.log`` / ``manifest.json`` bookkeeping.
    """
    return sorted(p for p in split_dir.glob("dd2d_*") if p.is_dir())


def _convert_split(
    raw_root: Path,
    data_root: Path,
    env_variant: str,
    env_id: str,
    split: str,
    overwrite: bool,
) -> tuple[int, int, int, int]:
    """Convert one split.

    Returns (written, skipped, failed, total_success_skeletons).
    """
    split_dir = raw_root / split
    if not split_dir.exists():
        print(f"[{split}] no such dir {split_dir}; skipping")
        return (0, 0, 0, 0)

    dirs = _problem_dirs(split_dir)
    written = skipped = failed = success_skeletons = 0
    for problem_dir in dirs:
        try:
            ep = convert_problem_dir(
                problem_dir, env_variant=env_variant, split=split, env_id=env_id
            )
        except Exception as exc:  # pylint: disable=broad-exception-caught
            failed += 1
            print(f"[{split}] {problem_dir.name} FAILED: {type(exc).__name__}: {exc}")
            continue

        out = episode_path(data_root, env_variant, split, ep.provenance.problem_id)
        if out.exists() and not overwrite:
            skipped += 1
            continue
        atomic_write_pickle_gz(ep, out)
        written += 1
        success_skeletons += ep.summary.num_success

    print(
        f"[{split}] {len(dirs)} problems: wrote {written}, skipped {skipped},"
        f" failed {failed}"
    )
    return (written, skipped, failed, success_skeletons)


@hydra.main(config_path="conf", config_name="dd2d_convert", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entrypoint for the DD2D -> SPECTRE conversion."""
    raw_root = Path(cfg.raw_root)
    data_root = Path(cfg.data_root)
    env_variant = str(cfg.env.env_variant)
    env_id = str(cfg.env.env_id)
    overwrite = bool(cfg.overwrite)
    splits = cast(list, OmegaConf.to_container(cfg.splits, resolve=True))

    print(
        f"DD2D convert: raw_root={raw_root} -> data_root={data_root}"
        f" env_variant={env_variant} splits={splits} overwrite={overwrite}"
    )

    totals = [0, 0, 0]
    for split in splits:
        written, skipped, failed, _ = _convert_split(
            raw_root, data_root, env_variant, env_id, str(split), overwrite
        )
        totals[0] += written
        totals[1] += skipped
        totals[2] += failed
    print(f"TOTAL: wrote {totals[0]}, skipped {totals[1]}, failed {totals[2]}")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
