"""The PIGINet training-example schema — the input contract, with no environment in it.

One :class:`PIGINetExample` is exactly the tuple the PIGINet paper feeds its feasibility
predictor ``f(I, pi, G)`` (paper lines 67-76, 197, 223): objects ``O``, initial literals
``I``, goal literals ``G``, a task plan ``pi`` (the skeleton with continuous args
omitted -- paper Table II), per-object segmented images, and a feasibility label.

These two dataclasses moved here from ``envs/dd2d/record.py`` on 2026-08-01 when the
PIGINet stack was lifted out of the DD2D tree to take a second environment. They were
always domain-neutral -- what is DD2D-specific is the *builders* that fill them
(``extract_init_literals``, ``object_table``, ``build_dd2d_example``), which stay in
``envs/dd2d/record.py``. Field-by-field mapping to the original ``fastamp`` contract is
in ``docs/piginet_record_schema.md``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = "1.0"


@dataclass
class ImageRef:
    """A reference to one per-object segmented image (pixels optional)."""

    object: str
    view: str  # "topdown" | "oblique" | ...
    seg_id: int | None = None  # segmentation id within the rendered frame
    bbox: list[int] | None = None  # [row_min, col_min, row_max, col_max]
    path: str | None = None  # file path once pixels are rendered; None = deferred


@dataclass
class PIGINetExample:
    """One (problem, candidate plan) example with its feasibility label."""

    problem_id: str
    objects: list[dict]  # {name, category, color, size, is_blocker, start_table}
    init_literals: list[list[str]]  # I
    goal_literals: list[list[str]]  # G
    task_plan: list[list[str]]  # pi: [operator, *args] per step, no continuous args
    label: bool  # feasibility (positive iff refined within budget)
    label_source: str  # how the label was decided
    refine: dict  # refinement diagnostics
    images: list[dict] = field(default_factory=list)  # list[ImageRef-as-dict]
    provenance: dict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    # -- serialisation -------------------------------------------------------
    def to_json(self, indent: int | None = 2) -> str:
        """JSON text for this example."""
        return json.dumps(asdict(self), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> "PIGINetExample":
        """Parse from JSON text."""
        return cls(**json.loads(text))

    def save(self, path: str) -> None:
        """Write to ``path`` as JSON."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "PIGINetExample":
        """Read from a JSON file."""
        with open(path, encoding="utf-8") as f:
            return cls.from_json(f.read())


__all__ = ["ImageRef", "PIGINetExample", "SCHEMA_VERSION"]
