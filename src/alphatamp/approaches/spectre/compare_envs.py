"""Per-environment configuration for the method-comparison notebook.

The notebook was written for DD2D and grew DD2D assumptions in a dozen small places: a
hardcoded ``env_variant``, strata that mean min-feasible-subset size, a method list
including two SPECTRE-v1 rows, a scene renderer imported from ``envs/dd2d``. Standing up
StickButton2D by copying the file made two of everything, which is how two notebooks
drift.

This is the alternative: one entry per environment, and the notebook reads the entry.
**Adding a third environment is adding an :class:`EnvSpec` here** — no notebook edit, no
fork.

The one thing that genuinely differs and cannot be defaulted is what a *stratum* means.
On DD2D it is the min-feasible-subset size, recovered from a seed band; on StickButton2D
it is the button count, recovered from the same arithmetic because the problem ids were
chosen to make that true. That coincidence is now written down as ``stratum_labels``
rather than left implicit in a formula named for DD2D seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from PIL.Image import Image

from alphatamp.approaches.spectre.schema import EpisodeRecord


@dataclass(frozen=True)
class EnvSpec:
    """Everything the comparison notebook needs to know about one collection."""

    key: str
    """Short name shown in the notebook's environment picker."""

    title: str
    """Heading for section 0."""

    env_variant: str
    """The collection whose ``compare_cache`` holds the rows."""

    stratum_labels: dict[int, str]
    """Stratum index -> the label a reader should see (``"b5"``, ``"s3"``, …)."""

    stratum_meaning: str
    """One line explaining what a stratum *is* here. Printed above every table."""

    stratum_axis_label: str = "stratum"
    """x-axis label for the per-stratum chart."""

    legacy_variant: str | None = None
    """A second collection to graft rows from, for methods never re-run natively."""

    legacy_only: tuple[str, ...] = ()
    """Method names taken from ``legacy_variant`` rather than the primary cache."""

    has_ablations: bool = False
    """Whether the v3 ablation arms (§4) are cached for this collection."""

    caveats: tuple[str, ...] = ()
    """Environment-specific warnings rendered under the summary table.

    These are the things a reader must know before quoting a number, and they belong
    beside the number rather than in a document nobody opens.
    """

    render_scene: Callable[[EpisodeRecord], Image] | None = None
    """Labelled scene render for the §5 planner inspector; ``None`` hides the section."""

    scene_legend: str = ""
    """Markdown describing what the §5 render shows: colours, labels, what to look at.

    Env-specific by nature — DD2D's "the red item is the retrieval target" is meaningless
    on StickButton2D — so it lives beside the renderer rather than in the notebook, where
    it would silently describe the wrong picture.
    """

    method_order: tuple[str, ...] = ()
    """Presentation order; empty = the shared default in ``compare.METHOD_ORDER``."""


def _dd2d_scene(episode: EpisodeRecord) -> Image:
    # pylint: disable=import-outside-toplevel
    from alphatamp.approaches.spectre.envs.dd2d.spectre_geometry import (
        reconstruct_scene,
    )
    from alphatamp.approaches.spectre.envs.dd2d.spectre_render import (
        render_labeled_scene,
    )

    assert episode.scene_geometry is not None
    return render_labeled_scene(reconstruct_scene(episode.scene_geometry))


def _sb2d_scene(episode: EpisodeRecord) -> Image:
    # pylint: disable=import-outside-toplevel
    from alphatamp.approaches.spectre.envs.stickbutton2d.render import (
        render_labeled_scene,
    )

    assert episode.scene_geometry is not None
    return render_labeled_scene(episode.scene_geometry, episode.object_registry)


DD2D = EnvSpec(
    key="dd2d",
    title="DD2D — SPECTRE v3 vs v2, PIGINet, VLMPlan and pure planning",
    env_variant="dd2d_v4",
    stratum_labels={0: "s0", 1: "s1", 2: "s2", 3: "s3"},
    stratum_meaning=(
        "Stratum = **min-feasible-subset size**: how many blockers must be staged "
        "before the target can be retrieved. s0 is trivially feasible; s3 needs three."
    ),
    stratum_axis_label="min-feasible-subset stratum",
    legacy_variant="dd2d_v3",
    # VLMPlan is the only row without a native dd2d_v4 run: regenerating it is two model
    # arms x 100 problems (~10.5 h) to move a row that cannot plausibly shift on a 0.08%
    # label change.
    legacy_only=("VLMPlan-8B", "VLMPlan-32B"),
    has_ablations=True,
    caveats=(
        "The `spectre3` cache row was written before `_V3_ARMS` was repointed to the "
        "unified coverage/waste arm on 2026-07-31, and `_dir_complete` skips a full "
        "directory. Until it is rebuilt with `--force`, this table under-reports v3 "
        "(7.44 here vs 5.78 for the deployed checkpoint).",
    ),
    render_scene=_dd2d_scene,
    scene_legend=(
        "the initial drawer, drawn from the episode's stored geometry. The **red** item "
        "is the retrieval target; blue items are concave; the dark frame is the wall "
        "band; the dashed box is the buffer. Labels are the item index (`item_5` → "
        "`5`), so they match the `stage {…}` sets in the plan table."
    ),
)

SB2D = EnvSpec(
    key="sb2d",
    title="StickButton2D — SPECTRE v3 vs PIGINet, VLMPlan and pure planning",
    env_variant="stickbutton2d_v1",
    stratum_labels={0: "b1", 1: "b2", 2: "b3", 3: "b5"},
    stratum_meaning=(
        "Stratum = **button count** (b1/b2/b3/b5). b1 and b2 are anchors every method "
        "ties on — their pools hold ~2 and 6–34 candidates — so **b3 and b5 carry the "
        "result**, and a pooled 'ALL' mean over strata this unbalanced is not a method "
        "comparison."
    ),
    stratum_axis_label="button count",
    has_ablations=True,
    caveats=(
        "**PIGINet's image channel is degenerate here.** Every unpressed button "
        "renders as the same red disc, so CLIP separates only {button, stick, robot} "
        "— which the type literals already give. Its pose/shape channels do the work. "
        "This bounds what the representation contrast on this environment shows; it is "
        "not evidence about low-level prediction in general.",
        "**PIGINet's `at-pose` literals are synthesised** by the adapter. SB2D's "
        "abstract initial state names no positions, and a low-level predictor with no "
        "coordinates would be a strawman.",
        "**b5's training split is 17 episodes** for every learned method — the "
        "collection was cut at a wall-clock budget — so the b5 column is substantially "
        "a generalisation result. No method is advantaged; none should be quoted as "
        "trained-on-b5.",
        "**Run-to-run noise at fixed seed reaches 1.02 FP here — read §4 against that, "
        "not against the seed sd.** The deployed arm and `abl_cov_rec` are the *same "
        "flags at the same three seeds*, trained twice by accident, and they read 1.69 "
        "vs 2.00 (seed 0: 1.76 vs 2.78). Training is not reproducible from the seed "
        "alone. Every ablation gap in §4 is smaller than that, so the 2x2 does not "
        "separate on this environment and no arm ordering there should be quoted.",
    ),
    render_scene=_sb2d_scene,
    scene_legend=(
        "the initial table scene, drawn from the episode's stored geometry. **Red** "
        "discs are unpressed buttons, the brown bar is the stick, the blue disc is the "
        "robot base. The dashed line is the arm's reach limit — buttons above it need "
        "the stick, which the symbolic model cannot see. Labels are the canonical "
        "object names used by the plan table and the VLMPlan prompt."
    ),
)

#: Registry. Order sets the notebook's default (first entry).
ENVS: dict[str, EnvSpec] = {spec.key: spec for spec in (SB2D, DD2D)}


def get(key: str) -> EnvSpec:
    """Look up an environment spec by key."""
    if key not in ENVS:
        raise KeyError(f"unknown environment {key!r}; known: {sorted(ENVS)}")
    return ENVS[key]


def stratum_label(spec: EnvSpec, stratum: int) -> str:
    """Reader-facing label for a stratum index."""
    return spec.stratum_labels.get(stratum, str(stratum))


def methods_for(spec: EnvSpec, present: Sequence[str]) -> list[str]:
    """Presentation order restricted to the methods actually in the cache."""
    # pylint: disable=import-outside-toplevel
    from alphatamp.approaches.spectre.compare import METHOD_ORDER

    order = spec.method_order or tuple(METHOD_ORDER)
    seen = set(present)
    return [m for m in order if m in seen]


__all__ = [
    "EnvSpec",
    "ENVS",
    "DD2D",
    "SB2D",
    "get",
    "stratum_label",
    "methods_for",
]
