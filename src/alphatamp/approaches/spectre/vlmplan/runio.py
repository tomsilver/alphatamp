"""Path layout and (de)serialisation shared by the two VLMPlan entry points.

Generation and scoring are separate scripts on purpose — see `score.py` — but they must
agree on where a run lives and on the sequence-file format. Keeping that agreement in the
package rather than importing one script from the other keeps both entry points thin and
makes the round-trip testable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from alphatamp.approaches.spectre import eda
from alphatamp.approaches.spectre.schema import EpisodeRecord

from .adapter import Step


def run_dir(data_root: Path, env_variant: str, run: str) -> Path:
    """``<data_root>/derived/<env_variant>/vlmplan/<run>/``."""
    return data_root / "derived" / env_variant / "vlmplan" / run


def compare_cache_dir(
    data_root: Path, env_variant: str, subdir: str, seed: int
) -> Path:
    """The seeded compare-cache directory this arm's records go in.

    The ``seed_<n>`` layer matches the SPECTRE families' layout, so the existing
    seed-averaging reader in ``dd2d_compare`` handles multi-seed VLMPlan runs unchanged.
    """
    return (
        data_root / "derived" / env_variant / "compare_cache" / subdir / f"seed_{seed}"
    )


def split_dir(data_root: Path, env_variant: str, split: str) -> Path:
    """``<data_root>/raw/<env_variant>/<split>/`` — where the episodes live."""
    return data_root / "raw" / env_variant / split


def select_episodes(
    episodes_dir: Path,
    n_problems: int = 0,
    problem_ids: Sequence[int] = (),
) -> list[EpisodeRecord]:
    """Load a split and take the configured subset, in problem-id order.

    Sorted so that ``n_problems=5`` names the same five problems on every run, and so a
    generation subset and a scoring subset can never silently disagree.
    """
    episodes = sorted(
        eda.load_split_episodes(episodes_dir).episodes,
        key=lambda ep: int(ep.provenance.problem_id),
    )
    wanted = [int(p) for p in problem_ids]
    if wanted:
        by_pid = {int(ep.provenance.problem_id): ep for ep in episodes}
        missing = [p for p in wanted if p not in by_pid]
        if missing:
            raise KeyError(f"problem_ids not present in {episodes_dir}: {missing}")
        return [by_pid[p] for p in wanted]
    return episodes[:n_problems] if n_problems > 0 else episodes


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic write, so an interrupted run never leaves a half-parsed file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def load_generation_stats(path: Path) -> dict[str, Any]:
    """Per-problem generation diagnostics from a sequences file.

    Scoring copies these onto the compare-cache record so the notebook can report
    generation quality (did it stall? was any response truncated?) beside the FP, without
    the notebook needing to read the run directory at all.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    rounds = payload.get("rounds") or []
    return {
        "n_proposed": len(payload.get("proposals") or []),
        "n_rounds": len(rounds),
        "stalled": payload.get("stalled"),
        "hit_max_rounds": payload.get("hit_max_rounds"),
        # Recomputed from the rounds rather than trusting the top-level field, so a
        # sequences file written before this telemetry existed still yields a number.
        "n_truncated": sum(1 for r in rounds if r.get("truncated")),
        "n_malformed": sum(int(r.get("n_malformed") or 0) for r in rounds),
        "n_blocks": sum(int(r.get("n_blocks") or 0) for r in rounds),
        "n_duplicate": sum(int(r.get("n_duplicate") or 0) for r in rounds),
        "n_invalid": sum(int(r.get("n_invalid") or 0) for r in rounds),
    }


def load_proposals(path: Path) -> list[tuple[tuple[Step, ...], int]]:
    """Read a sequences file back into ``[(steps, round_index), ...]``."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    proposals: list[tuple[tuple[Step, ...], int]] = []
    for proposal in payload["proposals"]:
        steps = tuple(
            (str(name), tuple(str(arg) for arg in args))
            for name, args in proposal["steps"]
        )
        proposals.append((steps, int(proposal["round"])))
    return proposals
