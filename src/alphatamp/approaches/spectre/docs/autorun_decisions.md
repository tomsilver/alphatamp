# SPECTRE v3 — Autonomous Run Log (2026-07-26 22:20 → 2026-07-27 09:00)

Decisions and learnings made **without a human in the loop**, at the user's instruction
(advisor meeting 2026-07-27; user unavailable through the evening). Separate from
`decisions.md` on purpose: everything here was judged by me against the plan
(`SPECTRE_v3_proposal.md` §1 goals, §2 constraints C1–C7 / learnings L1–L9), not agreed
with the user first. **Read this before trusting any number produced in this window.**

Standing instruction for the run, verbatim in substance:
- Goal 1 is **performance**: weakly dominate deployed v2.2 per stratum, or at minimum beat
  it on average. v2.2 is in principle a *lower* bound — v3 sees strictly more.
- **Altering the training process is explicitly permitted** to get there, and must be
  documented.
- Goals 2 and 3 (cleanliness/story, generality) still hold; don't buy performance by
  reintroducing per-environment machinery (C1/L2 — that is what `clears` was).
- Finish the remaining gates; write down whatever is left as loose threads.

Starting position (G6b, committed `3b6c301`), uncensored deployed FP on dd2d_v4 test:

| | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| v2.2 yardstick | **14.66** | 0.00 | **6.20** | 26.00 | **26.44** |
| v3 records+overlap | 16.17 | 0.00 | 8.56 | **22.00** | 34.12 |
| v3 no records (bar) | 19.54 | 0.00 | **3.64** | 33.80 | 40.72 |

So v3 **wins s2** (−4.00) and **loses s1** (+2.36) and **s3** (+7.68). Weak dominance
requires closing s1 and s3 without giving up s2.

---

## Time budget

10.6 h at start. Allocation, revised as it goes:

| Window | Work |
|---|---|
| 22:25–23:15 | G7 (eval-only, already staged) + s1/s3 diagnosis by code reading |
| 23:15–03:30 | Performance experiments — parallel sweeps, ~50 min each |
| 03:30–05:30 | G9 length generalization |
| 05:30–07:30 | G10 geometry interface (best effort; descope if performance is unresolved) |
| 07:30–08:45 | G11 consolidation, docs, CI |

**Priority order if time runs short:** performance > G11 consolidation > G7/G9 > G10.
Rationale: goal 1 is explicit and the advisor meeting needs a coherent story with numbers;
G10 is interface-only work whose own acceptance criterion is "changes nothing", so it is
invisible to the story and the safest thing to defer.

---

## Entries

(newest last, so this reads as a narrative of the run)

### A1 — G7 needed no training, because its arms already existed

G7's preset would have trained `{records ON, overlap ON}` and `{records ON, overlap OFF}`.
Those are **byte-identically configured** to G6b's two record arms (checked against the
stored checkpoint cfgs: same seed, epochs, lr, wl-weight, dropout, augment, selector).
Retraining would have produced a second sample of the same configuration and made it
ambiguous which number is "the" G7 result. **Decision: reuse G6b's checkpoints and run
only the eval-time demotion axis.** Cost: 3 minutes instead of 50.

### A2 — the eval-time demotion switch is a separate boolean, not a third `DemotionMode`

`apply_demotion: bool` rather than `mode="none"`. The modes answer *how much evidence
licenses a sound deduction* (`permissive` / `strict`); this answers *whether to act on the
deduction at all*. Collapsing them would let a future reader believe "none" is a soundness
setting. The proof state still advances when the offset is withheld, so the two arms differ
in exactly one thing — pinned by `test_apply_demotion_false_withholds_only_the_offset`,
which also asserts the switch is not inert.

**Learning (cost me a failing test):** my first version of that test iterated
`list_episodes(_V4)[:12]`. That is the project's own stride-don't-truncate trap — a prefix
is all stratum 0, where the first attempt usually succeeds and demotion never gets to act,
so the test reported "switch is inert". Third time this trap has bitten the project. It is
in `CLAUDE.md`; I still walked into it.

### A3 — `dead` is a disguised shortness cue; this is L4 reappearing as a *feature*

Diagnosing the s1 regression (v3 8.56 vs v2.2 6.20, while v3's *no-evidence* bar is 3.64 —
better than v2.2). Measured on dd2d_v4 train, 4600 candidate/context pairs:

```
corr(dead, |S|)      = -0.284
mean |S| | dead=1    = 1.38      mean |S| | dead=0 = 2.39
P(feasible | dead=1) = 0.0000    <- the rule is soundly firing
```

So `dead=1` predominantly marks **short** candidates. As an outside-the-net offset that is
harmless: it fires only where the deduction actually holds. As a **net input** it is a
free-running correlate the scorer can fit as "short ⇒ bad" and then apply everywhere —
which is precisely **L4** ("un-split evidence harms": consumed crudely, `blocked-at-…`
became a prefer-longer cue and cost +13.5 FP on s1). s1 is the stratum where short is
*correct*, so it takes the damage.

**Decision: test dropping `dead` from the net's features while keeping the demotion offset
outside it (`--overlap-mode jaccard`).** This is a **C5 argument, not a tuning knob** — C5
says deductions act on the ranking, not the representation, and feeding the proof in as a
feature violates the spirit of it. G7 supports the move: with overlap on, the offset is
worth only 0.13 FP, so the net is not relying on the offset to cover for it.

### A4 — training never shows the |F| regime s3 deploys in

`sample_context` caps `|F|` at **8** (inherited unchanged from v2.2 — this is *not* a
v3 regression). An s3 rollout has FP ≈ 34 and is therefore queried at `|F|` up to ~40, a
regime the model has literally never seen. The original RT2D spec had a
`rollout_aligned_mix` precisely so training mass matched the test-time visit distribution;
v2/v3 simplified it away.

**Decision: add `tail_max_f`** — half the non-empty mass stays uniform on `1..8`, half
spreads to `1..40`. Verified it moves mean |F| 2.95 → 8.20 and P(|F|>8) 0 → 0.263 **while
preserving P(|F|=0) ≈ 0.34**, so the static pathway keeps its training mass (P-D). This is
the "alteration to the training process" the user authorised, and it is domain-agnostic
(no DD2D constant enters — 40 is read off the observed FP range, and could be set from the
pool cap).

### A5 — `overlap_mode` zeroes a column instead of narrowing the tensor

So the state-dict shape never changes and the **D-8 exact-absence oracle keeps loading**. A
zeroed column *is* the feature's absence (its weight gets no signal from it), and the
alternative — making `n_overlap_feats` 1 — would have retired the G1 equivalence oracle
early, which the plan reserves for G9.

Also threaded `overlap_mode` through **deployment** and made `spectre_score_v3` read it
back off the checkpoint rather than accept it as an argument: deploying a model under a
different mode than it trained under is a silent train/deploy mismatch (§6.6), and the
kind of thing that would quietly invalidate a comparison.
