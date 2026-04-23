"""Collection configuration: frozen, hashable, YAML round-trippable.

Follows pipeline spec §4 with substrate-aligned field names:
- ``num_sampling_attempts_per_step`` (not ``refinement_max_samples``) — the real
  knob on ``BacktrackingRefiner``.
- ``refinement_timeout_s`` (not ``refinement_wall_clock_cutoff_s``) — the real
  ``timeout`` arg to ``Refiner.__call__``.
- ``abstract_plan_timeout_s`` — time budget for drawing the pool from the A*
  generator; a new field not in the spec but required by the real API.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from omegaconf import OmegaConf

_CONFIG_VERSION = "v1"
_TRACKED_PACKAGES = (
    "bilevel_planning",
    "kinder",
    "kinder_bilevel_planning",
    "kinder_models",
    "relational_structs",
)


def _git_sha() -> str:
    """Return short git sha of the code, or ``"unknown"`` on failure."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _package_versions() -> dict[str, str]:
    """Resolve the current versions of substrate packages via importlib.metadata."""
    out: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "unknown"
    return out


@dataclass(frozen=True)
class CollectionConfig:
    """Frozen spec for one ``(env_variant, split)`` data-collection run."""

    env_id: str
    env_variant: str
    model_name: str
    model_kwargs: dict[str, int | float | str]
    split: Literal["train", "val", "test"]
    num_problems: int
    problem_seed_start: int
    problem_seed_end: int  # exclusive

    # Pool / refinement budgets.
    K_max: int = 50
    abstract_plan_timeout_s: float = 30.0
    refinement_timeout_s: float = 20.0
    num_sampling_attempts_per_step: int = 10
    max_trajectory_steps: int = 100

    # Seeding / provenance.
    heuristic_name: str = "hff"
    refinement_seed_rule: str = "v1_blake2b_problem_skeleton"
    collect_instrumentation: bool = False
    state_path_depth: Literal["s0_sL_only", "full"] = "s0_sL_only"

    # Resolved at creation time.
    config_version: str = _CONFIG_VERSION
    git_sha: str = field(default_factory=_git_sha)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        )
    )
    package_versions: dict[str, str] = field(default_factory=_package_versions)

    def __post_init__(self) -> None:
        if self.state_path_depth != "s0_sL_only":
            raise NotImplementedError(
                "Only state_path_depth='s0_sL_only' (Substage A) is supported in v0.1"
            )
        if self.collect_instrumentation:
            raise NotImplementedError(
                "collect_instrumentation=True requires refiner subclassing;"
                " not implemented in v0.1"
            )
        if self.problem_seed_end <= self.problem_seed_start:
            raise ValueError(
                f"problem_seed_end ({self.problem_seed_end}) must be >"
                f" problem_seed_start ({self.problem_seed_start})"
            )

    def hashable_fields(self) -> dict[str, object]:
        """All fields except ``created_at`` — the basis for ``config_hash``."""
        d = asdict(self)
        d.pop("created_at", None)
        return d

    @property
    def config_hash(self) -> str:
        """First 12 hex chars of sha256 over canonical-JSON hashable fields."""
        payload = json.dumps(self.hashable_fields(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:12]

    def to_yaml(self, path: Path) -> None:
        """Write a canonical YAML representation (including ``created_at``)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = OmegaConf.create(asdict(self))
        with open(path, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg))

    @classmethod
    def from_yaml(cls, path: Path) -> "CollectionConfig":
        """Round-trip a config from YAML, preserving hash-determining fields."""
        raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        assert isinstance(raw, dict)
        # Drop non-init fields that from_yaml should carry through verbatim.
        return cls(**raw)  # type: ignore[arg-type]
