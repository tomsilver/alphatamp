# Archive — historical SPECTRE spec documents

These are the original spec documents, moved here **byte-unchanged** from the
repo root during the 2026-06 spectre silo refactor. They are historical: the
living, consolidated version is [`../proposal.md`](../proposal.md). Where a
document below disagrees with `proposal.md`, the proposal wins. Section
references like "METHOD §4.1.4" in code docstrings resolve against these files.

| File | One-line historical note |
|---|---|
| `SPECTRE_METHOD_SPEC.md` | Original method spec (Φ/Ψ/σ, PL loss, F-subset rules) targeting the five kinder envs; partially superseded by the RT2D spec. |
| `SPECTRE_RT2D_METHOD_SPEC.md` | RT2D-adapted method/training spec (v1.0); authoritative over the original method spec for the RT2D evaluation; introduced fixes 1–5. |
| `SPECTRE_TRAINING_PIPELINE_SPEC.md` | Original three-layer data-pipeline design (raw episodes / parquet derived / online examples). |
| `SPECTRE_TRAINING_PIPELINE_AS_BUILT.md` | What was actually built vs the pipeline spec — Layer 2 collapsed, live-object schema, divergence log (last synced 2026-04-24). |
| `SPECTRE_EDA_SPEC.md` | Pre-training EDA gates: Group 1 sanity, baselines B1–B5, adaptive premium Δ and headroom H, pass bar. |
| `ROUTED_TRANSPORT2D_SPEC.md` | RoutedTransport2D environment spec (v1): K₃,₃ topology, scene latent, per-problem tags, three-gate refiner. |
| `SYNTHETIC_ENVIRONMENT.md` | Motivation memo for building RT2D — why the kinder kinematic2d envs let a lookup-table baseline win, and the required properties of a replacement env. |

## Snapshots

Dated, frozen, paper-style exports of the whole project state (method +
results + frontier). Unlike the spec docs above they are narrative snapshots,
not design documents. The living docs remain the source of truth and win on
disagreement; staleness annotations live here, not in the snapshot files.

| File | One-line note |
|---|---|
| `SPECTRE_WRITEUP_APR_2026.md` | Paper-style snapshot (2026-04-27) of the method and RT2D-n3 results: attempts to first success ↓41–62% and refinement wall-clock ↓36–57% vs Pure Planning + two memoization baselines; checkpoints selected with rollout-based validation (post-2026-04-25). |

Known-stale points in `SPECTRE_WRITEUP_APR_2026.md`:

- Baselines are named Pure Planning / Static Historical / Adaptive Historical
  (≈ B2 / B3 / B4); the living convention is B1–B5, and B4 is implemented as
  Naive-Bayes log-odds over pairwise failure conditionals, not raw conditional
  frequency.
- Reported spreads are std over the 100 test instances; the current reporting
  bar is mean ± std over ≥ 3 training seeds (multi-seed confirmation pending).
- Two deliberate post-deposit edits to §Training (snapshots are otherwise
  never edited after deposit):
  - 2026-06-06: the sentence originally read "each allowed 30 refinement
    attempts", conflating the RT2D candidate-pool cap (`k_cap = 30` in
    `envs/routedtransport2d/plan_generator.py`; every pooled skeleton is
    refined during collection) with the evaluation attempt budget. The edit
    split the pool cap from the eval budget into two explicit numbers.
  - 2026-06-07: that split initially recorded the eval budget as 20, but the
    headline table, distribution, and success-at-K figures were in fact
    generated at an attempt budget of 30 (`analyze_spectre.ipynb`,
    `ATTEMPT_BUDGET = 30`). At 30 the budget equals the pool cap and never
    binds, so the reported numbers are uncensored; the sentence now reads 30.
    See the 2026-06-07 `decisions.md` entry adopting uncensored evaluation as
    the standard.

## Frozen log monoliths

`decisions.md` and `notebook.md` were single files until 2026-07-29, when they were split
into era chapters under `docs/decisions/` and `docs/notebook/`. The pre-split files are
frozen here byte-for-byte:

- `decisions_2026-07-29_monolithic.md` — 2241 lines, 38 ADRs
- `notebook_2026-07-29_monolithic.md` — 2111 lines, 49 entries

These are not merely historical: `experiments/spectre/decisions_index.py check` compares every
live entry against them on every run, which is what makes the split non-lossy as an enforced
property. Do not edit them — that would weaken the check to a tautology.
