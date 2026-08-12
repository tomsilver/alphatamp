"""Step-5 gate (docs/piginet_dd2d_plan.md): the dataset-point visualizer.

Collects one tiny real problem, then renders one of its records with
``inspect_example.visualize_record`` and asserts a non-trivial figure is produced (and that
``write_crops`` wrote the sibling ``scene.png``).
"""

from __future__ import annotations

import glob
import os

from alphatamp.approaches.spectre.envs.dd2d.drawer.collect import (
    DD2DCollectConfig,
    collect_problem,
)
from alphatamp.approaches.spectre.envs.dd2d.drawer.inspect_example import (
    _find_record,
    visualize_record,
)


def test_visualize_record_produces_figure(tmp_path):
    # one small real problem (crowd=0 -> fast) written to a tmp split dir
    split_dir = tmp_path / "train"
    res = collect_problem(
        seed=0,
        stratum=1,
        config=DD2DCollectConfig(crowd=0, time_budget=5.0),
        split_dir=str(split_dir),
    )
    assert res.kept, f"sample problem did not solve (reason={res.reason})"
    pdir = split_dir / res.problem_id

    # write_crops saved the full-scene overview alongside the crops
    assert (pdir / "scene.png").exists()
    assert (pdir / "images").is_dir() and list((pdir / "images").glob("*.png"))

    # a bare dir arg resolves to a record; render it
    record = _find_record(str(pdir))
    out = visualize_record(record, out_path=str(tmp_path / "dp.png"))
    assert os.path.exists(out)
    assert (
        os.path.getsize(out) > 20_000
    )  # a real multi-panel figure, not a blank canvas


def test_find_record_from_split_dir(tmp_path):
    split_dir = tmp_path / "train"
    res = collect_problem(
        seed=0,
        stratum=1,
        config=DD2DCollectConfig(crowd=0, time_budget=5.0),
        split_dir=str(split_dir),
    )
    assert res.kept
    # resolving from the split root should find a record nested under <problem_id>/
    rec = _find_record(str(split_dir))
    assert rec.endswith(".json") and os.path.exists(rec)
    assert (
        len(glob.glob(os.path.join(str(split_dir), "**", "*.json"), recursive=True))
        >= 1
    )
