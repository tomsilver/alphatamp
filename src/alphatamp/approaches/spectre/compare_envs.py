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
    """One line explaining what a stratum *is* here.

    Printed above every table.
    """

    stratum_axis_label: str = "stratum"
    """X-axis label for the per-stratum chart."""

    legacy_variant: str | None = None
    """A second collection to graft rows from, for methods never re-run natively."""

    legacy_only: tuple[str, ...] = ()
    """Method names taken from ``legacy_variant`` rather than the primary cache."""

    has_ablations: bool = False
    """Whether the v3 ablation arms (§4) are cached for this collection."""

    has_timing: bool = False
    """Whether the §2b wall-clock section is enabled for this collection.

    True for the DD2D v3/v4 collections, whose pool-method compare records carry the
    per-candidate timing fields written by ``precompute_dd2d_cache``. False for SB2D **by
    choice, not by data**: SB2D episodes *do* carry real per-candidate
    ``refinement_wall_clock_s`` (the shared collector times every refine), but filling
    §2b there needs a per-env refinement cap (the DD2D 2 s cap censors SB2D's ~10 s
    feasible refines), a ``precompute_dd2d_cache.py --env-variant stickbutton2d_v1_kinder
    --methods astar piginet spectre3`` run, and SPECTRE timing sourced from the legacy
    cache — deferred, so §2b shows a stub on SB2D.
    """

    caveats: tuple[str, ...] = ()
    """Environment-specific warnings rendered under the summary table.

    These are the things a reader must know before quoting a number, and they belong
    beside the number rather than in a document nobody opens.
    """

    render_scene: Callable[[EpisodeRecord], Image] | None = None
    """Labelled scene render for the §5 planner inspector; ``None`` hides the
    section."""

    plan_label: Callable[[Sequence[tuple[str, Sequence[str]]]], str] | None = None
    """Render one plan's step sequence (``[(op_name, [args...]), ...]``) into a short
    human-readable string for the §5 inspector.

    Env-specific because the operator vocabulary is: DD2D names its operators
    ``place-buffer`` / ``retrieve``, StickButton2D names them ``RobotPressButton*`` /
    ``StickPressButton*`` / ``PickStick*``. A single hardcoded formatter is why the SB2D
    inspector printed ``retrieve ?`` for every row. ``None`` falls back to a generic
    join. It takes the step sequence (not a skeleton) so it renders both a pooled skeleton
    and a VLMPlan off-pool attempt, which carries steps but no skeleton.
    """

    scene_legend: str = ""
    """Markdown describing what the §5 render shows: colours, labels, what to look at.

    Env-specific by nature — DD2D's "the red item is the retrieval target" is
    meaningless on StickButton2D — so it lives beside the renderer rather than in the
    notebook, where it would silently describe the wrong picture.
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


def _sb2d_kinder_scene(episode: EpisodeRecord) -> Image:
    """kinder's own env pixels with Set-of-Mark labels overlaid — the SAME render the
    VLMPlan prompt attaches, so the inspector shows exactly what the model saw."""
    # pylint: disable=import-outside-toplevel
    from alphatamp.approaches.spectre.envs.stickbutton2d.render import (
        render_kinder_labeled_scene,
    )

    return render_kinder_labeled_scene(episode)


def _short(name: str) -> str:
    """``item_5`` / ``circle_3`` -> the trailing token, matching the render's labels."""
    return name.rsplit("_", 1)[-1]


def _dd2d_plan_label(steps: Sequence[tuple[str, Sequence[str]]]) -> str:
    """``stage {5, 3} → retrieve 10`` from a DD2D plan's ``place-buffer`` / ``retrieve``
    ops."""
    staged = [
        _short(args[0]) for name, args in steps if name == "place-buffer" and args
    ]
    tgt = next(
        (_short(args[0]) for name, args in steps if name == "retrieve" and args), "?"
    )
    head = "stage {" + ", ".join(staged) + "} → " if staged else ""
    return f"{head}retrieve {tgt}"


def _sb2d_plan_label(steps: Sequence[tuple[str, Sequence[str]]]) -> str:
    """The press order + stick handling of a StickButton2D plan.

    The pressed button is arg 1 for an arm press and arg 2 for a stick press (the stick
    sits between robot and button), matching ``SB2DAdapter.discretionary_objects``.
    """
    parts: list[str] = []
    for name, args in steps:
        if name.startswith("PickStick"):
            parts.append("pick stick")
        elif name == "PlaceStick":
            parts.append("place stick")
        elif name.startswith("RobotPressButton") and len(args) >= 2:
            parts.append(f"press {args[1]} (arm)")
        elif name.startswith("StickPressButton") and len(args) >= 3:
            parts.append(f"press {args[2]} (stick)")
    return " → ".join(parts) if parts else "(empty plan)"


DD2D = EnvSpec(
    key="dd2d",
    title="DD2D — SPECTRE vs PIGINet, VLMPlan and pure planning",
    env_variant="dd2d_v4",
    stratum_labels={0: "s0", 1: "s1", 2: "s2", 3: "s3"},
    stratum_meaning=(
        "Stratum = **min-feasible-subset size**: how many blockers must be staged "
        "before the target can be retrieved. s0 is trivially feasible; s3 needs three."
    ),
    stratum_axis_label="min-feasible-subset stratum",
    legacy_variant="dd2d_v3",
    # VLMPlan-32B (the Qwen arm) is the only row without a native dd2d_v4 run -- grafted
    # from dd2d_v3 rather than regenerated (two model arms x 100 problems to move a row
    # that cannot plausibly shift on a 0.08% label change). VLMPlan-GPT5.6 is native to v4.
    legacy_only=("VLMPlan-32B",),
    has_ablations=True,
    has_timing=True,
    caveats=(
        "The §1/§2 `SPECTRE` rows are the **deployed unified coverage/waste** "
        "checkpoint (`checkpoints_v3_unified`), rebuilt 2026-08-01: adaptive 5.78 ± "
        "0.10. The §4 ablation component arms (`abl_*`) predate the 2026-07-31 "
        "unification and score under the **old** coverage/waste definition, so the §4 "
        "`deployed` row (unified, ~5.78) is not directly comparable to the matched "
        "`cov+waste, tokens` arm (~7.90) — the gap conflates the definition change with "
        "the aggregate-records/evidence-attn switches. §4 is a self-contained "
        "matched-settings study; read it internally, not against §1.",
        "**`VLMPlan-GPT5.6` (gpt-5.6-luna, the frontier arm) is the *worst* method here** "
        "— 62.98 ALL, worse than the naive planner order (34.52) and the local Qwen-32B "
        "(23.55). It over-stages confidently on the packing negative control (s0 43.2 vs "
        "astar 0.0) and never stalls, so it accrues many diverse failed off-pool attempts. "
        "It is scored on a stratified 40 (10/stratum); the per-stratum means are comparable "
        "to the 100-problem rows, the stratum-weighted ALL is not read against them "
        "1-for-1. A frontier VLM does not rescue VLMPlan on DD2D — see "
        "`notebook/07` 2026-08-03.",
    ),
    render_scene=_dd2d_scene,
    plan_label=_dd2d_plan_label,
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
        "**VLMPlan-32B is scored on a stratified 40-problem subset (10/stratum), not "
        "the full 100.** The other methods use all 100 test problems. Strata stay "
        "balanced 10/10/10/10, so the per-stratum means and the stratum-weighted ALL "
        "are comparable — only the per-cell n (10 vs 25) differs. The subset was a "
        "compute choice: b3/b5 problems VLMPlan cannot self-solve run to the ~10-round "
        "stall cap (~15–20 min each), so a full run is ~16 h. Its off-pool proposals "
        "are refined for real and charged as attempts, which shows as a heavy right "
        "tail on b3/b5 (b5 self-solve rate 5/10, tail FP 62–150) — not censoring; "
        "nothing hit the 200 cap. VLMPlan lands between astar-dist and the learned "
        "methods: it beats the naive planner order overall (13.2 vs 16.3, via b5 where "
        "astar's default order is worst) but loses to SPECTRE/PIGINet ~7× and is worse "
        "than astar on b1/b2/b3, where its charged failed guesses cost it and the pool "
        "order is already near-optimal.",
        "**§6's `n_proposed` is not comparable to DD2D's.** SB2D generation stops at "
        "the first plan that refines (the 200-plan budget is a ceiling for when they "
        "all fail, not a quota), so the count reads as *plans needed*; DD2D's rows "
        "predate that and read as *plans producible*. The FP columns are unaffected — "
        "the rollout never looks past the first success either way.",
        "**Run-to-run noise at fixed seed reaches 1.02 FP here — read §4 against that, "
        "not against the seed sd.** The deployed arm and `abl_cov_rec` are the *same "
        "flags at the same three seeds*, trained twice by accident, and they read 1.69 "
        "vs 2.00 (seed 0: 1.76 vs 2.78). Training is not reproducible from the seed "
        "alone. Every ablation gap in §4 is smaller than that, so the 2x2 does not "
        "separate on this environment and no arm ordering there should be quoted.",
    ),
    render_scene=_sb2d_scene,
    plan_label=_sb2d_plan_label,
    scene_legend=(
        "the initial table scene, drawn from the episode's stored geometry. **Red** "
        "discs are unpressed buttons, the brown bar is the stick, the blue disc is the "
        "robot base. The dashed line is the arm's reach limit — buttons above it need "
        "the stick, which the symbolic model cannot see. Labels are the canonical "
        "object names used by the plan table and the VLMPlan prompt."
    ),
)

SB2D_KINDER = EnvSpec(
    key="sb2d_kinder",
    title=(
        "StickButton2D (kinder-rendered crops) — SPECTRE v3 vs PIGINet, VLMPlan and "
        "pure planning"
    ),
    env_variant="stickbutton2d_v1_kinder",
    stratum_labels={0: "b1", 1: "b2", 2: "b3", 3: "b5"},
    stratum_meaning=SB2D.stratum_meaning,
    stratum_axis_label="button count",
    # PIGINet + VLMPlan-GPT5.6 are native to the kinder cache; SPECTRE and VLMPlan-32B are
    # image-free / already-collected and byte-identical to `stickbutton2d_v1`, so they are
    # grafted from that legacy cache. astar is rebuilt natively (deterministic and cheap).
    legacy_variant="stickbutton2d_v1",
    legacy_only=("SPECTRE-static", "SPECTRE-adaptive", "VLMPlan-32B"),
    has_ablations=False,  # v3 ablations are SPECTRE-internal; §4 is hidden here
    has_timing=False,
    caveats=(
        "**PIGINet's crops come from kinder's own renderer** (a window on the true scene, "
        "with real context — neighbouring buttons, stick, table, wall), so this is the "
        "variant to read for the representation contrast. It still does not make two "
        "unpressed buttons look different (identical red discs in the real env too), so "
        "the image channel is *less* degenerate but not fully informative; pose/shape do "
        "most of the work.",
        "**Only PIGINet and VLMPlan-GPT5.6 are native here; SPECTRE and VLMPlan-32B are "
        "grafted from `stickbutton2d_v1`** (image-free / byte-identical records). The "
        "comparison that moves is PIGINet vs SPECTRE.",
        "**PIGINet's `at-pose` literals are synthesised** by the adapter — SB2D's abstract "
        "initial state names no positions, and a low-level predictor with no coordinates "
        "would be a strawman.",
        "**b5's training split is 17 episodes** for every learned method (collection cut "
        "at a wall-clock budget), so the b5 column is substantially a generalisation "
        "result — none should be quoted as trained-on-b5.",
        "**`VLMPlan-GPT5.6` (gpt-5.6-luna) is a genuine planner** — 11.85 ALL, self-solves "
        "35/40, beats the naive order (16.29) — but **~6× behind the learned rankers** "
        "(1.69–2.28) and roughly tied with the local Qwen-32B (13.18); it over-thinks b3 "
        "and only wins at b5. Stratified 40 (10/stratum). See `notebook/07` 2026-08-03.",
    ),
    render_scene=_sb2d_kinder_scene,
    plan_label=_sb2d_plan_label,
    scene_legend=(
        "kinder's own initial-scene render (the real env pixels PIGINet's crops come "
        "from) with Set-of-Mark labels overlaid — the exact image the VLMPlan prompt "
        "attaches. **Red** discs are unpressed buttons, the brown bar is the stick, the "
        "purple disc is the robot; labels are the canonical object names used by the plan "
        "table."
    ),
)

#: Registry. Order sets the notebook's default (first entry). Only the kinder-rendered
#: SB2D variant is kept (the schematic `sb2d` entry was retired 2026-08-03).
ENVS: dict[str, EnvSpec] = {spec.key: spec for spec in (SB2D_KINDER, DD2D)}


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
    "SB2D_KINDER",
    "get",
    "stratum_label",
    "methods_for",
]
