# SPECTRE Notebook — Foundations

4 entries, 2000-01-01 .. 2026-06-24 (closed). Newest first.
Index and cross-reference tables: [README.md](README.md).

---

<a id="2026-06-11-b6-exact-h-sweep"></a>
## 2026-06-11 — B6 DP-on-counts exact h-sweep (RT2D-n3, K=30, uncensored)

<!--strip-->
> **id** `2026-06-11-b6-exact-h-sweep` · **status** active · **tracks** baselines
<!--/strip-->

- What: exact (unpruned) receding-horizon sweep of the B6 DP-on-counts baseline,
  `h ∈ {1,2,3,4}`, on the 100 RT2D-n3 test problems at the full pool K=30, budget
  30. B6 reuses B4's calibrated NB count model as its `q`-model; `h=1` is B4 by
  construction. Made tractable by **incremental NB scoring** (leaf `O(K³)→O(K²)`):
  h=2 93 s→9 s, h=3 740 s→86 s, h=4 intractable→~10 min, all reproducing the exact
  attempt counts; `h=1 == B4` bitwise. Also added `solvability_at_cap` (gates any
  pool capping). Driver: `experiments/spectre/analyze_spectre.py` /
  `eda.dp_on_counts_baseline`.
- Result: mean attempts 9.62 (h1=B4) → 9.46 (h2) → 9.15 (h3) → 9.14 (h4). **Paired
  per-problem** (Δ = shallower − deeper, win/tie/loss = deeper strictly fewer /
  tie / more): h1→h2 Δ+0.16, 9/56/35, p=0.018; h2→h3 Δ+0.31, 10/87/3, p=0.057;
  h3→h4 Δ+0.01, 2/95/3, p=0.89; h1→h3 Δ+0.47, 10/62/28, **p=0.23 (n.s.)**.
  Solvability-at-cap: test successes sit at every planner depth (solvable@15≈0.46,
  @20≈0.60, ~1.0 only at k=30) — capping the candidate pool would censor real
  successes, so B6 runs uncapped at K=30.
- Takeaway / next: the lookahead premium is **small, fragile, and saturated** —
  the +0.47 mean gain (h1→h3) is not significant (Wilcoxon p=0.23) and comes from
  a few large wins offset by more small regressions (the DP optimizes an imperfect
  count model, so its "smarter" early picks sometimes lose on realized outcomes);
  h4 adds nothing over h3 (p=0.89). Against the ~3.5–4 attempt gap from B6 to
  SPECTRE, **lookahead on the count model is not the missing ingredient** —
  SPECTRE's relational features, not deeper search over counts, are what close the
  gap. Pruning (`m`) and exact full-horizon DP remain available but unmotivated
  here.

<a id="2026-06-06-frozen-context-ablation-rt2d-n3"></a>
## 2026-06-06 — Frozen-context (Ψ) ablation on RT2D-n3

<!--strip-->
> **id** `2026-06-06-frozen-context-ablation-rt2d-n3` · **status** active · **tracks**
> method, env-rt2d
<!--/strip-->

- What: inference-time ablation of the context encoder — at every rollout
  step the scorer's context vector is pinned to the learned empty-F `c₀`
  (the frozen variant is exactly a learned *static* ranker), vs the full
  adaptive pipeline. Same checkpoint (`r3_visit_rate/seed_0`, the headline
  model), deterministic paired rollouts over the 100 test episodes.
  Runner: `experiments/spectre/spectre_ablate_context.py`; artifacts in
  `data/spectre/derived/ablation_context/`.
- Result (attempt budget 20): full **5.55 ± 5.18** vs frozen
  **6.53 ± 5.70** attempts; paired Δ (frozen − full) = **+0.98
  [95% CI +0.39, +1.57]**; per-episode win/tie/loss 32/54/14. At budget 30
  (the analysis notebook's setting; never binds — censoring 0): full 5.67
  (reproduces the headline row exactly) vs frozen 6.73, Δ +1.06
  [+0.45, +1.67]. Placement: frozen (6.73) still far ahead of B4 (9.62)
  and B3 (12.47), landing between B2-FF (6.31) and B1-random (6.81).
  Same-choice agreement: 1.0 at t=1 (by construction — full also uses
  `c₀` at empty F), ~0.89 at t=2–3, ≤0.26 from t=5 on; first divergence
  concentrated at t=4–5; the success-at-K gap is concentrated mid-rollout
  (0.82 vs 0.69 at K=9; identical at K=1–3).
- Takeaway / next: Ψ is **not dead weight** — failure-conditioning buys
  ~1 attempt (CI excludes zero), concentrated mid-rollout once a few
  failures accumulate — but the majority of SPECTRE's margin over B4
  (3.95 attempts) comes from the static Φ+σ ranking (frozen alone beats
  B4 by 2.89). This quantifies the 2026-04-27 "representation, not
  failure-conditioning per se" claim: adaptivity ≈ 27% of the B4 margin.
  Caveats: single seed; inference-time freeze (σ was trained expecting a
  varying c) may understate what a trained-static architecture could do —
  a retrain-without-Ψ variant is the natural follow-up if a sharper
  number is needed.
- Side-finding (eval protocol): the analysis notebook's headline table was
  generated with `ATTEMPT_BUDGET = 30`, not the documented 20 — at 30 the
  budget never binds (pool cap 30, censoring 0), so the headline numbers
  are effectively uncensored. The budget-20 protocol numbers for the same
  checkpoint are 5.55 (full) with 2% censoring. Reconciled 2026-06-07:
  adopted uncensored (budget = pool cap = 30) evaluation as the reporting
  standard — see the [`decisions.md` 2026-06-07](../decisions/README.md) entry; writeup + archive README
  corrected 20 → 30.

---

<a id="2026-06-06-seed-forwarding-bug"></a>
## 2026-06-06 — Multi-seed checkpoints are duplicates (seed-forwarding bug)

<!--strip-->
> **id** `2026-06-06-seed-forwarding-bug` · **status** active · **tracks** tooling
<!--/strip-->

- What: while selecting checkpoints for the Ψ ablation, hashed every
  `best.pt` under `data/spectre/checkpoints/`.
- Result: `c1_baseline_seeds/seed_1` ≡ `seed_2` (identical md5; both save
  `seed: 0` in their config) and legacy `routedtransport2d_n3_v1/seed_1`
  ≡ `seed_2` — the seed override never reached training in those runs.
  Only `heuristic_prior/{seed_0,1,2}` are three genuinely distinct seeds.
- Takeaway / next: no valid multi-seed run of any zero-prior recipe exists
  yet; the ≥3-seed reporting bar is currently unmeetable without retraining.
  Diagnose the slurm/Hydra seed-forwarding path before the next multi-seed
  launch (fix deferred).

---

<a id="2026-04-27-rt2d-n3-paper-snapshot"></a>
## 2026-04-27 — RT2D-n3 paper-snapshot results (writeup)

<!--strip-->
> **id** `2026-04-27-rt2d-n3-paper-snapshot` · **status** active · **tracks**
> evaluation, env-rt2d
<!--/strip-->

- What: full pipeline on RoutedTransport2D-n3-v1 (500/100/100); SPECTRE vs
  Pure Planning (≈ B2), Static Historical (≈ B3), Adaptive Historical (≈ B4).
  Checkpoint selected by rollout-based val selection; eval attempt budget 20.
- Result: attempts to first success ↓ 41–62% and refinement wall-clock
  ↓ 36–57% vs the baselines; Adaptive Historical needs 57.3% more refinement
  time than SPECTRE; success-at-K: ~80% of instances within 9 attempts vs
  ~18 for Adaptive Historical.
- Takeaway / next: the gap over Adaptive Historical (which also conditions on
  F) is attributable to representation — generalizing across structurally
  similar skeletons — not failure-conditioning per se. Spreads are over 100
  test instances, not the ≥ 3-seed bar; multi-seed confirmation pending. Full
  narrative: [`archive/SPECTRE_WRITEUP_APR_2026.md`](../archive/SPECTRE_WRITEUP_APR_2026.md)
  (known-stale points: [`archive/README.md`](../archive/README.md)).
