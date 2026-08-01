# SPECTRE Decisions — Direction pivot

1 entries, 2026-06-25 .. 2026-07-11 (closed). Newest first.
Index and cross-reference tables: [README.md](README.md).

---

<a id="2026-06-25-direction-pivot-representation-question"></a>
## 2026-06-25 — Direction pivot: from adaptive reordering to a representation question

<!--strip-->
> **id** `2026-06-25-direction-pivot-representation-question` · **status** active ·
> **tracks** process, method · **see also** `proposal.md` §0 is the current framing
<!--/strip-->

**Context.** The project's headline had been *adaptive test-time reordering* (the
SPECTRE re-ranker), evaluated on the bespoke RT2D env against the adaptive
baseline B4. Two results undercut that as the lead: the Ψ-ablation (`notebook.md`
2026-06-06) attributes only **~27%** of SPECTRE's margin over B4 to
failure-conditioning — the **static** Φ+σ representation carries ~73%; and the B6
DP-on-counts sweep ([`notebook.md` 2026-06-11](../notebook/01-foundations.md#2026-06-11-b6-exact-h-sweep)) showed lookahead over the count
model is **small, fragile, and saturated**, i.e. not the missing ingredient. A
structural reading of RT2D explains why, and motivates a reframe. Each decision
below is recorded with rationale.

**Decisions.**

(a) **Reframe adaptivity-primary → representation-primary.** The contribution is
now a *representation question for plan-feasibility prediction in
fully-observable (FO), deterministic bilevel TAMP*: what should a feasibility
predictor represent skeletons/problems over? Rationale: the empirics put the
margin in the static representation, not the failure-conditioning. (See
`proposal.md` §0.)

(b) **Demote SPECTRE/reordering to a secondary, composable increment.**
Within-episode failures carry free instance-specific signal, but it is a minority
of the margin; treat the re-ranker as orthogonal to — and combinable with —
whichever representation wins, not as the headline.

(c) **RT2D was effectively partially observable *to the policy* — a mislabeling.**
RT2D was described as FO+deterministic, but the policy π was denied x₀ and the toy
three-gate refiner had **privileged access to the latent z**. To the policy the
problem was therefore effectively partially observable, and the discrete gating
latent had to be **manufactured** — which is why RT2D felt contrived. Rationale:
record so we do not re-derive the bespoke env as if it were a faithful FO TAMP
instance.

(d) **The no-x₀ design was a handicap *in RT2D* — but the nuance matters.**
PIGINet's own ablation shows x₀ carries real signal **in their kitchen
problems**, and PIGINet already works at **150–600 problems**, so the
*data-efficiency* rationale for dropping x₀ does not hold *universally*. This does
**not** establish that x₀ must always be included: whether dropping low-level
state is a helpful abstraction is **domain-dependent**, and there may be problems
where it helps. We are **not committed** either way — the x₀ stance is
experiment-driven. Rationale: avoid over-correcting from "drop x₀ always" to "keep
x₀ always"; both are empirical questions.

(e) **The FO information-ceiling bounds the adaptive component's value.** In
FO+deterministic TAMP, the within-episode refinement failures add **no
information beyond x₀** at the predictor's ceiling (the outcome of every skeleton
is a deterministic function of x₀). This is the structural reason the adaptive
signal is small here, and the structural reason for the pivot. Rationale: it
makes the ~27% finding expected, not a defect of Ψ.

(f) **Reinterpret the 27% finding as "the static representation does the work."**
The ablation is now read as positive evidence for the representation thesis: most
achievable gain is captured by the static ranking, with online updating a small
add-on — consistent with (e).

(g) **Adopt the efficiency/representation framing + crossover prediction +
negative control.** The claim is **efficiency / perception-lightness, not
information access**: under FO+determinism no representation beats an ideal
low-level predictor on information grounds. **Falsifiable prediction (a
hypothesis, not a result):** a *crossover* — in the low-data / weak-perception
regime a well-chosen (richer-than-pixels, cheaper-than-full-state) representation
matches or beats a low-level PIGINet-style predictor on downstream planning
efficiency, while the low-level predictor regains its edge with abundant data +
strong perception. **Negative control:** dense-packing / fine-continuous-fit
domains, where any compressed representation is expected to lose, bound the
claim. *Abstract-first* is the current leading candidate but only one point in a
design space (learned latents, object-centric/graph features, intermediate
symbolic+coarse-geometric states, invented predicates), and may prove too lossy.

(h) **Prefer pre-existing environments that meet a hypothesized-advantage
property wishlist; keep bespoke in scope.** We prefer pre-existing envs *only if*
they exhibit properties we expect to favor a relational/abstract representation,
and keep **bespoke, hand-crafted** envs in scope where they better expose the
advantage. The (open, evolving) property list: (1) feasibility governed by
relational structure the abstraction captures; (2) low-level state
high-dimensional/distracting or hard to extract relational structure from; (3)
perception genuinely limited or costly; (4) object-count/identity generalization;
(5) long horizon / large diverse pool. Planned homes: PIGINet kitchens with
degraded perception, and Khodeir clutter/distractor domains augmented with a
low-level baseline, swept over perception-degradation × training-set size.
Primary metric time-to-first-success; secondary time-to-k. Rationale: the
property combination needed for the *adaptive* claim (shared, refinement-decidable,
instance-specific gating) is rare in pre-existing benchmarks, but the
*representation* claim has real pre-existing homes.

(i) **Freeze the April writeup.** `archive/SPECTRE_WRITEUP_APR_2026.md` is frozen
with a banner (2026-06-25); it reflects the adaptive-reordering framing and is
retained as historical record. Rationale: the living docs (`proposal.md` §0,
this log, `notebook.md`) are the source of truth and must not defer to the frozen
snapshot.

**Consequences.** `proposal.md` now leads with §0 (representation-first), with the
original §1–§6 retained byte-unchanged under "Superseded framing (April 2026)";
`research_lit.md` reframes PIGINet as the low-level static predictor we compare
against and adds a representation lens; [`notebook.md` 2026-06-25](../notebook/02-pivot.md#2026-06-25-psi-ablation-reinterpretation) records the
reinterpretation and the forward sweep. The RT2D env, the SPECTRE model, and the
B1–B6 baselines/code are **unchanged** — this is a framing pivot, not a code
change. No planner/refiner/abstraction change. What survives intact: the
rollout-based model-selection discipline, the PL loss, the F-subset discipline.

---

