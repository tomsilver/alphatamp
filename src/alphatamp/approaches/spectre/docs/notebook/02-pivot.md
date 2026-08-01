# SPECTRE Notebook — Direction pivot

1 entries, 2026-06-25 .. 2026-07-11 (closed). Newest first.
Index and cross-reference tables: [README.md](README.md).

---

<a id="2026-06-25-psi-ablation-reinterpretation"></a>
## 2026-06-25 — Reinterpretation of the Ψ-ablation + direction pivot

<!--strip-->
> **id** `2026-06-25-psi-ablation-reinterpretation` · **status** active · **tracks**
> method, process
<!--/strip-->

- What: no new training run — a reinterpretation of two existing results under a
  fully-observable (FO) information-ceiling lens, and the pivot it motivates.
  Inputs: the 2026-06-06 frozen-context (Ψ) ablation (failure-conditioning ≈ 27%
  of the margin over B4; static Φ+σ ≈ 73%) and the 2026-06-11 B6 sweep (lookahead
  premium small/fragile/saturated, +0.47 h1→h3 n.s. at p=0.23). Structural point:
  in FO+deterministic TAMP every skeleton's outcome is a deterministic function
  of x₀, so within-episode failures add no information beyond x₀ at the
  predictor's ceiling — which *bounds* the adaptive component and explains why
  both the Ψ-ablation and the B6 sweep find adaptivity small.
- Result (interpretation, not a new number): the static *representation*, not the
  failure-conditioning, carries SPECTRE's margin. This reframes the contribution
  from "failure context helps" to a **representation question** for
  plan-feasibility prediction: what substrate (low-level/PIGINet-style vs.
  abstract-first vs. learned-latent/object-centric/invented-predicate) predicts
  refinement feasibility most sample-efficiently and with weakest perception.
  Established: the 27% and B6 numbers above. **Hypothesis (to test):** a
  crossover — a well-chosen representation matches/beats a low-level predictor in
  the low-data/weak-perception regime, losing its edge with abundant data +
  strong perception (efficiency, not access; negative control = dense packing).
- Takeaway / next: two forward experiments. (1) **Perception × training-size
  crossover sweep** on pre-existing homes (PIGINet kitchens with degraded
  perception; Khodeir clutter/distractor) with a low-level (PIGINet-class)
  baseline — metric time-to-first-success (secondary time-to-k) — to test the
  crossover. (2) Cheap **"Measurement B" probe** on a Khodeir domain: are residual
  refinement failures predictable from *earlier-in-episode* failures (controlling
  for plan-prefix logic, so we are not just re-reading deterministic prefix
  implications)? This bounds whether any adaptive headroom survives on natural
  (non-bespoke) domains, complementing the FO information-ceiling argument. Full
  rationale: [`decisions.md` 2026-06-25](../decisions/02-pivot.md#2026-06-25-direction-pivot-representation-question); current framing: `proposal.md` §0.

---

