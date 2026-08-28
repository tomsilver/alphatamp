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

    True for the DD2D v3/v4 collections and the kinder SB2D variant, whose pool-method
    compare records carry the per-candidate timing fields written by
    ``precompute_dd2d_cache`` from the stored ``refinement_wall_clock_s``. The two live
    variants differ only in the per-candidate refinement cap: DD2D's 2 s sits *above
    the whole feasible distribution* (feasible p95 0.44 s), while SB2D's feasible
    refines run to seconds, so it uses 10 s — above each problem's *fastest*-feasible
    (max 8.84 s, so no problem is censored) but inside the feasible distribution. On
    the kinder SB2D variant SPECTRE's §2b timing is grafted from the
    ``stickbutton2d_v1`` legacy cache, the same as its FP (``legacy_only``). Default
    ``False`` for a freshly onboarded collection until its cache carries the timing
    fields.
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
    join. It takes the step sequence (not a skeleton) so it renders both a pooled
    skeleton and a VLMPlan off-pool attempt, which carries steps but no skeleton.
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
    """Kinder's own env pixels with Set-of-Mark labels overlaid — the SAME render the
    VLMPlan prompt attaches, so the inspector shows exactly what the model saw."""
    # pylint: disable=import-outside-toplevel
    from alphatamp.approaches.spectre.envs.stickbutton2d.render import (
        render_kinder_labeled_scene,
    )

    return render_kinder_labeled_scene(episode)


def _restock_scene(episode: EpisodeRecord) -> Image:
    """Top-down Restock3D scene from stored geometry (no PyBullet in the notebook)."""
    # pylint: disable=import-outside-toplevel
    from alphatamp.approaches.spectre.envs.restock3d.render import (
        render_scene_from_geometry,
    )

    assert episode.scene_geometry is not None
    return render_scene_from_geometry(episode.scene_geometry)


def _restock_short(name: str) -> str:
    """``block_goal1`` -> ``block1`` / ``cube_goal2`` -> ``cube2`` (matches the
    render)."""
    return name.replace("_goal", "")


def _restock_plan_label(steps: Sequence[tuple[str, Sequence[str]]]) -> str:
    """The store order + section of a Restock3D plan.

    Each object is picked then stored via ``place_tall`` / ``place_short``; the ordered
    stores (which object, which section) are the readable signal — the store order is
    the south-to-north / far-first structure feasibility hinges on.
    """
    parts: list[str] = []
    for name, args in steps:
        if name in ("place_tall", "place_short") and args:
            obj = _restock_short(args[-1])
            section = "tall" if name == "place_tall" else "short"
            parts.append(f"{obj}→{section}")
    return " · ".join(parts) if parts else "(no stores)"


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
    # that cannot plausibly shift on a 0.08% label change). VLMPlan-GPT5.6 is native
    # to v4.
    legacy_only=("VLMPlan-32B",),
    has_ablations=True,
    has_timing=True,
    caveats=(
        "The §1/§2 `SPECTRE` rows are the **deployed unified coverage/waste** "
        "checkpoint (`checkpoints_spectre_unified`), rebuilt 2026-08-01: adaptive 5.78 ± "
        "0.10. The §4 ablation component arms (`abl_*`) predate the 2026-07-31 "
        "unification and score under the **old** coverage/waste definition, so the §4 "
        "`deployed` row (unified, ~5.78) is not directly comparable to the matched "
        "`cov+waste, tokens` arm (~7.90) — the gap conflates the definition change with "
        "the aggregate-records/evidence-attn switches. §4 is a self-contained "
        "matched-settings study; read it internally, not against §1.",
        "**`VLMPlan-GPT5.6` (gpt-5.6-terra, the frontier arm) reaches ~parity with the "
        "naive planner order here** — 35.23 ALL vs astar 34.52 — but stays far "
        "behind the "
        "learned rankers (SPECTRE 5.78, PIGINet 17.27) on the packing negative control. "
        "This is the stronger tier *and* the gripper-geometry disclosure (PROVENANCE "
        "deviation 9): together they nearly halve the earlier gpt-5.6-luna row (62.98, "
        "which had been the worst method). The behaviour is bimodal — 14/40 targets are "
        "trivially graspable and solved on the first attempt (FP 0), but when "
        "staging is "
        "needed the model over-stages and floods dozens of off-pool proposals that all "
        "fail geometric refinement (per-problem FP up to 200). Reasoning effort "
        "does not "
        "rescue it: a full-scale medium-effort arm scored 33.5, tied with low "
        "(paired 95% "
        "CI [-18.6, +15.1]). Stratified 40 (10/stratum), single generation seed (bare "
        "mean, like astar). A frontier VLM does not close the packing gap — see "
        "`notebook/07` 2026-08-08.",
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

DD2D_GEN_SHAPEONLY = EnvSpec(
    key="dd2d_gen_shapeonly",
    title=(
        "DD2D shape-only generalization — SPECTRE vs PIGINet "
        "(train dd2d_v4 / test unseen tee+cross at the default 0.7x size, 9–12 blockers)"
    ),
    # The object-generalization test at the current **default** tee/cross size (0.7x
    # linear, shapes._FAMILY_DEFAULT_SCALE; docs/decisions 2026-08-06). Points at the
    # `_sz07` collection, which is a 0.7x draw of exactly this test. The full-size draws
    # (band 5 `dd2d_v4gen_shapeonly` s2=17.27, band 7 `_fresh` s2=5.63) are retained on
    # disk and in the 2026-08-04/06 notebook entries as the historical / size-sweep
    # record; the live comparison uses the default size.
    env_variant="dd2d_v4gen_shapeonly_sz07",
    stratum_labels={0: "s0", 1: "s1", 2: "s2", 3: "s3"},
    stratum_meaning=DD2D.stratum_meaning,
    stratum_axis_label="min-feasible-subset stratum",
    # SPECTRE + PIGINet are scored natively into this collection's cache (train-old
    # checkpoints, test-new episodes, via precompute_dd2d_cache.py --test-variant), so
    # nothing is grafted. VLMPlan and the v3 ablations are out of scope for a gen
    # section.
    legacy_variant=None,
    legacy_only=(),
    has_ablations=False,
    has_timing=True,
    caveats=(
        "**Train-old / test-new, shape ISOLATED, default 0.7x shapes.** SPECTRE and "
        "PIGINet are the dd2d_v4-trained checkpoints scored on 40 held-out problems "
        "(10/stratum) whose ONLY controlled shift is two unseen concave figures — a "
        "`tee` and a `cross`, ≥1 of each forced into every scene, at their **default** "
        "0.7x size (hull footprint tee ~29, cross ~33). Blocker count is held at the "
        "trained 9–12 band (identical to dd2d_v4), so unlike `dd2d_v4gen_shape` (which "
        "also raised the count to 13–15) this does not confound shape with count. No "
        "OOV — a shape family is geometry metadata, not a token. Seed band [6M,7M).",
        "**Shape generalization is essentially free, and SPECTRE beats PIGINet "
        "decisively.** SPECTRE-adaptive ALL 2.79 — at or below the in-dist dd2d_v4 "
        "headline 5.78, so no degradation on unseen shapes. It beats PIGINet (22.68) by "
        "a wide, significant margin (paired bootstrap −19.88, CI [−31.04, −10.07]) and "
        "astar (34.72). The abstract representation wins the SPECTRE-vs-PIGINet "
        "contrast on unseen shapes; the failure-conditioned re-ranking (static ALL "
        "15.00 → adaptive 2.79) does most of the lifting.",
        "**s2 is collection-variance-dominated — read the ALL win, not a single-draw "
        "s2.** This is one 0.7x draw; across draws of this test SPECTRE-adaptive s2 = "
        "17.27 (band 5, full size) / 5.63 (band 7, full size) / 3.17 (here, 0.7x) "
        "while **astar s2 is stable at 14–15** — physical difficulty does not move, "
        "only which ~1.5-solution s2 instances land in the k=200 pool (2026-08-02 "
        "finding). The 0.7x default was chosen because the smaller figures grasp and "
        "pack cleanly while still being unseen (docs 2026-08-06); the object-gen "
        "conclusion — SPECTRE ≫ PIGINet on unseen shapes — is robust across all draws.",
    ),
    render_scene=_dd2d_scene,
    plan_label=_dd2d_plan_label,
    scene_legend=DD2D.scene_legend,
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
    # PIGINet + VLMPlan-GPT5.6 are native to the kinder cache; SPECTRE and VLMPlan-32B
    # are image-free / already-collected and byte-identical to `stickbutton2d_v1`, so
    # they are grafted from that legacy cache. astar is rebuilt natively (deterministic
    # and cheap).
    legacy_variant="stickbutton2d_v1",
    # LAZY is image-free (like SPECTRE), trained + cached on the base stickbutton2d_v1,
    # so it is grafted from that legacy cache rather than recomputed on the kinder
    # re-imaging.
    legacy_only=("SPECTRE-static", "SPECTRE-adaptive", "VLMPlan-32B", "LAZY-adaptive"),
    has_ablations=True,  # single-feature-isolation ablation (§4.3), grafted from the
    # `stickbutton2d_v1` cache (SPECTRE is image-free; arms trained/cached there) — 2026-08-21
    has_timing=True,  # §2b enabled 2026-08-03: per-env 10s cap, SPECTRE timing grafted
    caveats=(
        "**PIGINet's crops come from kinder's own renderer** (a window on the "
        "true scene, "
        "with real context — neighbouring buttons, stick, table, wall), so this is the "
        "variant to read for the representation contrast. It still does not make two "
        "unpressed buttons look different (identical red discs in the real env too), so "
        "the image channel is *less* degenerate but not fully informative; "
        "pose/shape do "
        "most of the work.",
        "**Only PIGINet and VLMPlan-GPT5.6 are native here; SPECTRE and VLMPlan-32B are "
        "grafted from `stickbutton2d_v1`** (image-free / byte-identical records). The "
        "comparison that moves is PIGINet vs SPECTRE.",
        "**PIGINet's `at-pose` literals are synthesised** by the adapter — "
        "SB2D's abstract "
        "initial state names no positions, and a low-level predictor with no "
        "coordinates "
        "would be a strawman.",
        "**b5's training split is 17 episodes** for every learned method "
        "(collection cut "
        "at a wall-clock budget), so the b5 column is substantially a generalisation "
        "result — none should be quoted as trained-on-b5.",
        "**`VLMPlan-GPT5.6` (gpt-5.6-terra) is a genuine planner** — 6.42 ALL, "
        "self-solves "
        "39/40 with 0 censored, and now beats the naive order across the board (astar "
        "16.29): b1 0.0, b2 2.4, b3 0.9 (better than astar's 2.96 — the earlier "
        "gpt-5.6-luna row over-thought b3), only b5 stays hard (22.4). It is the "
        "stronger "
        "tier plus the gripper disclosure (deviation 9), roughly halving luna "
        "(11.85) and "
        "clearing the local Qwen-32B (13.18), though still ~3–4× behind the "
        "learned rankers "
        "(1.69–2.28). Stratified 40 (10/stratum), single generation seed. See "
        "`notebook/07` 2026-08-08.",
        "**§2b wall-clock is under a 10 s per-candidate refinement cap** (vs "
        "DD2D's 2 s): "
        "SB2D feasible refines run to seconds, so the cap clears each problem's "
        "*fastest* "
        "feasible (0 censored) but abandons slower candidates. Because SB2D's *failing* "
        "refines run to the 20 s budget, the cap saves the highest-FP methods "
        "(astar) the "
        "most in absolute seconds — the reverse of DD2D, where it most helped "
        "the low-FP "
        "learned ranker. Refinement dominates the total; plan-gen and inference "
        "are small. "
        "See `notebook/07` 2026-08-03.",
    ),
    render_scene=_sb2d_kinder_scene,
    plan_label=_sb2d_plan_label,
    scene_legend=(
        "kinder's own initial-scene render (the real env pixels PIGINet's crops come "
        "from) with Set-of-Mark labels overlaid — the exact image the VLMPlan prompt "
        "attaches. **Red** discs are unpressed buttons, the brown bar is the stick, the "
        "purple disc is the robot; labels are the canonical object names used by "
        "the plan "
        "table."
    ),
)

DD2D_HOLDOUT = EnvSpec(
    key="dd2d_holdout_s3",
    title=(
        "DD2D held-out stratum — SPECTRE vs PIGINet, VLMPlan and pure planning "
        "(train s0–s2, evaluate the never-trained s3)"
    ),
    env_variant="dd2d_v4_holdout_s3",
    stratum_labels={0: "s0", 1: "s1", 2: "s2", 3: "s3"},
    stratum_meaning=DD2D.stratum_meaning,
    stratum_axis_label="min-feasible-subset stratum",
    # astar + PIGINet + SPECTRE are scored natively with the s0–s2-trained checkpoints;
    # VLMPlan-GPT5.6 is symlinked into this cache from dd2d_v4 (training-free, identical
    # test problems). Nothing is grafted through the notebook, so no legacy variant.
    legacy_variant=None,
    legacy_only=(),
    has_ablations=False,
    has_timing=True,
    caveats=(
        "**Held-out STRATUM generalization.** SPECTRE and PIGINet are trained on s0–s2 "
        "only — the s3 training problems are excluded via `--train-strata 0 1 2`, no "
        "re-collection — then evaluated on all four strata of the standard dd2d_v4 test "
        "split. **s3 is the never-trained held-out stratum, and it is the headline "
        "column.** astar and VLMPlan are training-free and unchanged; VLMPlan-GPT5.6 is "
        "reused verbatim from the deployed dd2d_v4 cache.",
        "**Do not read the pooled ALL as the result.** It averages held-out s3 with "
        "s0–s2, which are in-training-distribution strata (held-out *problems*, seen "
        "*strata*). The s0–s2 columns are the sanity floor; s3 is the generalization "
        "test.",
    ),
    render_scene=_dd2d_scene,
    plan_label=_dd2d_plan_label,
    scene_legend=DD2D.scene_legend,
)

SB2D_KINDER_HOLDOUT = EnvSpec(
    key="sb2d_holdout_b5",
    title=(
        "StickButton2D held-out stratum (kinder-rendered crops) — SPECTRE v3 vs "
        "PIGINet, "
        "VLMPlan and pure planning (train b1/b2/b3, evaluate the never-trained b5)"
    ),
    env_variant="stickbutton2d_v1_kinder_holdout_b5",
    stratum_labels={0: "b1", 1: "b2", 2: "b3", 3: "b5"},
    stratum_meaning=SB2D.stratum_meaning,
    stratum_axis_label="button count",
    # Mirrors SB2D_KINDER: PIGINet (kinder crops) + astar + VLMPlan-GPT5.6 are native to
    # this cache; SPECTRE is image-free and grafted from the instrumented-refiner
    # `stickbutton2d_v1_holdout_b5` cache. Every learned row is the b1/b2/b3-trained
    # checkpoint; VLMPlan-GPT5.6 is reused verbatim (symlinked native).
    legacy_variant="stickbutton2d_v1_holdout_b5",
    legacy_only=("SPECTRE-static", "SPECTRE-adaptive"),
    has_ablations=False,
    has_timing=True,
    caveats=(
        "**Held-out STRATUM generalization.** SPECTRE and PIGINet are trained on "
        "b1/b2/b3 "
        "only — b5 excluded via `--train-strata 0 1 2`, no re-collection — then "
        "evaluated "
        "on all four strata. **b5 is the never-trained held-out stratum, and it is the "
        "headline column.** astar and VLMPlan are training-free; VLMPlan-GPT5.6 "
        "is reused "
        "verbatim.",
        "**The existing b5-training caveat compounds here.** In the standard collection "
        "b5's train split was already only 17 episodes; here b5 is removed from "
        "training "
        "entirely, so b5 is a clean held-out-stratum number for every learned method.",
        "**Do not read the pooled ALL as the result.** It averages held-out b5 with the "
        "in-distribution b1/b2/b3 (b1/b2 are anchors every method ties on). Read b5.",
        "**SPECTRE is grafted from the instrumented `stickbutton2d_v1` refiner cache** "
        "(image-free, so schematic-vs-kinder is irrelevant to it); PIGINet uses "
        "the real "
        "kinder crops. The contrast that moves is PIGINet vs SPECTRE on b5.",
    ),
    render_scene=_sb2d_kinder_scene,
    plan_label=_sb2d_plan_label,
    scene_legend=SB2D_KINDER.scene_legend,
)

RESTOCK3D = EnvSpec(
    key="restock3d",
    title="restock3D — SPECTRE vs PIGINet, LAZY and pure planning",
    env_variant="restock3d_v2",
    # Stratum = the section config (n_tall x n_short), banded 0..4 in the problem id by
    # `strata_v2` (5 strata, NOT the shared 4). The two symmetric light strata (2×2/3×3) plus
    # the crowded asymmetric 4×3 (banding stratum 3) are collected + evaluated here; the
    # remaining crowded strata (3×4 = 2, 4×4 = 4) are still collecting. The notebook groups
    # on `sorted(stratum_labels)` = [0, 1, 3] (note the gap at 2 -- 3×4 is not in yet).
    stratum_labels={0: "2×2", 1: "3×3", 3: "4×3"},
    stratum_meaning=(
        "Stratum = **section config** (n_tall × n_short): how many tall blocks and short "
        "cubes must be stored on the shelf. 2×2 / 3×3 are the symmetric light strata; 4×3 "
        "(four tall + three short) is the first crowded stratum, with larger pools (K≤75) "
        "and more reach-over / F3 pressure. Feasibility hinges on **reach-over** (a nearer "
        "object blocks the front-grasp of a farther one, so the store order must go "
        "far-first / south-to-north) and **F3** (a tall block placed in the short top "
        "section collides the ceiling). Real PyBullet collision decides it."
    ),
    stratum_axis_label="section config (tall×short)",
    has_ablations=False,  # §4 ablation arms not trained for restock3D (deferred)
    has_timing=True,  # per-candidate refinement_wall_clock_s is stored; §2b enabled
    caveats=(
        "**2×2 / 3×3 / 4×3 sections, 3 seeds (0,1,2).** All learned rows (SPECTRE, PIGINet, "
        "LAZY) are 3 seeds so §1/§2 carry an across-seed ±; the remaining crowded strata "
        "(3×4, 4×4) are still collecting in a separate session, so the pooled 'ALL' averages "
        "these three configs (note the stratum axis skips s2 = 3×4). Partial-stratum "
        "stragglers on disk are excluded by the strata-{0,1,3} filter (train + eval).",
        "**Read the crowded 4×3, not the two-stratum ALL.** 2×2 and 3×3 are near-trivial "
        "(small feasible-dense pools every learned method solves in ≈0 attempts), so any "
        "representation / adaptivity contrast, if one exists, lives at **4×3** — the first "
        "stratum with real crowding. All learned methods crush the naive planner order "
        "(astar-dist); see the paired notebook entry for the per-stratum numbers.",
        "**§2b wall-clock is refinement-dominated.** Restock3D refinement is real PyBullet "
        "motion planning: feasible candidates take tens of seconds (2×2 ~32 s / 3×3 ~45 s / "
        "4×3 ~54 s fastest-feasible), while plan-gen and GPU inference are sub-second. The "
        "55 s per-candidate cap sits just above the fastest-feasible distribution (so no "
        "feasible candidate is censored); the FP headline stays uncapped.",
        "**VLMPlan is not run here yet** (deferred). The comparison is the naive planner "
        "order (astar-dist) vs the three learned methods.",
    ),
    render_scene=_restock_scene,
    plan_label=_restock_plan_label,
    scene_legend=(
        "the initial scene, drawn top-down from the episode's stored 3D geometry. **Blue** "
        "footprints are tall blocks, **orange** are short cubes (the height/F3 axis, shown "
        "as colour since both share a footprint from above); the grey dashed box is the "
        "shelf store region; the grey marker is the robot base. y increases south→north — "
        "the store order the plan table lists. Labels (`cube1`, `block2`) match the plan "
        "table's `obj→section` stores."
    ),
)

RESTOCK3D_V3 = EnvSpec(
    key="restock3d_v3",
    title="restock3D-v3 — SPECTRE vs astar (REAL hybrid-prune dataset, INTERMEDIATE)",
    # Repointed 2026-08-27 from the SYNTHETIC restock3d_v3 to the REAL (hybrid-prune) collection.
    # The synthetic result is preserved in docs/git + on disk; the `key` stays "restock3d_v3" so
    # the notebook dropdown / SPECTRE_COMPARE_ENV selector is unchanged.
    env_variant="restock3d_v3_real",
    # Stratum = block count n (6/7/8/9), banded 0..3 on the SHARED 4-stratum band (unlike v2's
    # 5-stratum local band), so compare.stratum_of decodes it with no routing edit.
    stratum_labels={0: "n=6", 1: "n=7", 2: "n=8", 3: "n=9"},
    stratum_meaning=(
        "Stratum = **block count n** (6/7/8/9). v3 varies **per-object widths** (in "
        "[0.02, 0.08] m) and **heights near the short/tall fit cutoff**, so — unlike v2's "
        "interchangeable constant-size blocks — **which** block goes to **which** section "
        "matters (block *selection*, not just order). Feasibility hinges on a lateral "
        "**capacity** budget (Σwidths + gaps ≤ shelf) per section plus the **height cutoffs** "
        "(a too-tall block in the short section is F3) and **reach-over** (store far-first). "
        "Larger n = tighter feasible-split set (fewer valid tall/short assignments)."
    ),
    stratum_axis_label="block count n",
    has_ablations=False,  # SPECTRE+astar intermediate: no ablation arms trained on the real data
    has_timing=False,  # real wall-clock exists but §2b deferred pending a real feasible-tail cap
    caveats=(
        "**⚠️ REAL but INTERMEDIATE dataset — collection is still running.** Labels are real "
        "PyBullet motion planning: **train** is `hybrid_prune` (the analytic classifier prunes "
        "the K_max pool, real MP labels the analytic-feasible candidates + a deterministic 25% "
        "audit of the infeasible ones), **val/test** are full `real`. At this cut **train is "
        "complete (300)** but the hardest stratum **n=9 test is only partially collected "
        "(~6-14/20)** and val ~6-10/10 — so the n=9 column is small-sample-noisy and the pooled "
        "ALL under-weights it. This **replaces** the earlier SYNTHETIC restock3d_v3 comparison; "
        "the analytic result (SPECTRE 11.11 vs PIGINet 38.11) is preserved in docs/git as the "
        "geometry **upper bound** this real cut audits.",
        "**SPECTRE + astar only (no PIGINet / LAZY).** This intermediate retrains only SPECTRE on "
        "the real data (3 seeds, the deployed `--scene-3d --atom-mode profiles --repeat-feats` "
        "recipe verbatim); PIGINet and LAZY are **not yet** retrained on real labels, so the "
        "SPECTRE-vs-PIGINet real-label representation-crossover audit — the point of the real "
        "collection — is **deferred** to the full collection. The §4.3 ablation and §2b "
        "wall-clock sections are disabled for this cut.",
        "**Read the crowded n=8/9, not the pooled ALL.** Smaller n is feasible-denser (easier); "
        "any representation / adaptivity contrast lives at the tighter feasible-split strata "
        "where block *selection* bites — but note n=9 is the least-collected stratum here.",
    ),
    render_scene=_restock_scene,
    plan_label=_restock_plan_label,
    scene_legend=(
        "the initial scene, drawn top-down from the episode's stored 3D geometry. **Blue** "
        "footprints are tall (short-cutoff-exceeding) blocks, **orange** are short-eligible "
        "ones (the height/F3 axis shown as colour since footprints vary in width but not from "
        "above); the grey dashed box is the shelf store region; the grey marker is the robot "
        "base. y increases south→north — the store order the plan table lists."
    ),
)

# ---------------------------------------------------------------------------------------
# compare_methods_simple.py: SPECTRE with the earlier/simple coverage/waste
# (--legacy-coverage) + repeat carrying SB2D. Same episodes as the parent; only the
# SPECTRE rows (§1/§2) and the +scalars ablation arm are native to the simple cache. Every
# baseline (astar/PIGINet/LAZY/VLMPlan) and the definition-invariant static/+records/+recjac
# ablation arms are grafted from the parent cache. docs/decisions/07 2026-08-27.
# ---------------------------------------------------------------------------------------
_SIMPLE_CAVEAT = (
    "**This is the SIMPLE coverage/waste variant** (`--legacy-coverage`: "
    "`coverage=|S∩culprits|/|culprits|`, `waste=|S\\culprits|/|S|`), not the deployed "
    "unified definitions. The SPECTRE rows are trained with the deployed two-stage "
    "residual recipe but this coverage/waste; every baseline is grafted UNCHANGED from "
    "the parent cache, so only the SPECTRE rows and the §4 `+scalars`/`full` arms differ "
    "from the deployed notebook. §2b wall-clock is omitted (has_timing off)."
)
DD2D_SIMPLE = EnvSpec(
    key="dd2d_simple",
    title="DD2D (simple coverage/waste) — SPECTRE vs PIGINet, LAZY, VLMPlan, pure planning",
    env_variant="dd2d_v4_simple",
    stratum_labels=DD2D.stratum_labels,
    stratum_meaning=DD2D.stratum_meaning,
    stratum_axis_label=DD2D.stratum_axis_label,
    legacy_variant="dd2d_v4",
    # Every baseline is byte-identical to the deployed dd2d_v4 run (same episodes, same
    # checkpoints) -> grafted. VLMPlan-32B is omitted (it is itself grafted from dd2d_v3 in
    # the deployed notebook, a second hop).
    legacy_only=("astar-dist", "PIGINet", "LAZY-adaptive", "VLMPlan-GPT5.6"),
    has_ablations=True,
    has_timing=False,
    caveats=(_SIMPLE_CAVEAT,),
    render_scene=DD2D.render_scene,
    plan_label=DD2D.plan_label,
    scene_legend=DD2D.scene_legend,
)
SB2D_SIMPLE = EnvSpec(
    key="sb2d_simple",
    title=(
        "StickButton2D (simple coverage/waste, repeat-on) — SPECTRE vs PIGINet, LAZY, "
        "VLMPlan, pure planning"
    ),
    env_variant="stickbutton2d_v1_simple",
    stratum_labels=SB2D.stratum_labels,
    stratum_meaning=SB2D.stratum_meaning,
    stratum_axis_label=SB2D.stratum_axis_label,
    legacy_variant="stickbutton2d_v1",
    legacy_only=("astar-dist", "PIGINet", "LAZY-adaptive", "VLMPlan-32B"),
    has_ablations=True,
    has_timing=False,
    caveats=(
        _SIMPLE_CAVEAT,
        "**Simple coverage/waste is identically inert on SB2D** (it reads `r.culprits` "
        "only, and SB2D reports `dev_blame`), so SB2D's adaptive signal is carried by "
        "`repeat` — which required RESURRECTING the `step_certificate` probe on the four "
        "press schemas (isolated to `stickbutton2d_v1_simple` via domain `_SB2D_REPEAT`; "
        "the deployed `stickbutton2d_v1` stays inert/byte-reproducible). That probe was "
        "retired 2026-08-26 as UNSOUND (~10.9% of feasible SB2D candidates flagged; the "
        "promising −0.79 FP was 1-seed) — `repeat` here is a learned column the model can "
        "down-weight, not a sound veto. Read accordingly.",
        "**PIGINet is the schematic-crop row** grafted from `stickbutton2d_v1` (2.02), not "
        "the kinder-rendered crops (2.28); both are documented and do not separate. "
        "VLMPlan-GPT5.6 (terra) is omitted (kinder-cache only).",
    ),
    render_scene=SB2D.render_scene,
    plan_label=SB2D.plan_label,
    scene_legend=SB2D.scene_legend,
)

#: Registry. Order sets the notebook's default (first entry). Only the kinder-rendered
#: SB2D variant is kept (the schematic `sb2d` entry was retired 2026-08-03); the two
#: `_holdout_` entries are the held-out-stratum generalization sections (2026-08-09).
#: `restock3d` is the third evaluation environment (2×2 + 3×3, 1 seed; 2026-08-19);
#: `restock3d_v3` is its per-object-dims successor on a SYNTHETIC (analytic-labelled) dataset.
ENVS: dict[str, EnvSpec] = {
    spec.key: spec
    for spec in (
        SB2D_KINDER,
        DD2D,
        DD2D_SIMPLE,
        SB2D_SIMPLE,
        RESTOCK3D,
        RESTOCK3D_V3,
        DD2D_GEN_SHAPEONLY,
        DD2D_HOLDOUT,
        SB2D_KINDER_HOLDOUT,
    )
}


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
    "DD2D_HOLDOUT",
    "SB2D_KINDER_HOLDOUT",
    "RESTOCK3D",
    "get",
    "stratum_label",
    "methods_for",
]
