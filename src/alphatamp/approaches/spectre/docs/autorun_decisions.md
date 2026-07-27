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

**Revised mid-run.** The user confirmed G9 (length generalization) is secondary to beating
v2.2, and that v2.2 does not need re-deriving. G9's encoder is built and tested but the
*experiment* is not run — which is doubly justified, since its premise does not hold on
DD2D anyway (see A-G9 below). G10 is not attempted.

### Sweep ledger

Each sweep is ~50 min for 3 arms in parallel; arms run concurrently with scoring and with
each other wherever slots allow (5 concurrent training procs at peak, 20 of 32 cores).

| sweep | arms | question |
|---|---|---|
| `g8` | `jac`, `tailF`, `jac_tailF` | does dropping the `dead` feature fix s1, does rollout-aligned |F| fix s3 |
| `p2` | `norec`, `agg` (+2 staged) | the missing G6 cell; does taming the token flood make records work |
| `p3` | `objev`, `objev_norec`, `objev_tailF` | does evidence work when routed through the tag join instead of as tokens |

### A-G9 — the length-generalization premise does not hold on DD2D

Measured before spending a sweep on it. Max plan length in the *candidate pool*, by the
episode's stratum:

| stratum of episode | max plan length in its pool | max step index |
|---|---|---|
| s0, s1, s2 | 9 | 8 |
| s3 | 7 | 6 |

Training on s0–s2 therefore already exercises step indices 0–8, which **covers everything
s3 needs**. The absolute `pos_emb` table is never queried out of range under the
"train s0–s2 / deploy s3" protocol, so that experiment would not have been testing what it
claims. The sinusoidal encoder is kept as future-proofing for longer-horizon domains and as
a generality argument — not as a fix for a live DD2D defect, and the docstring says so.

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

### A6 — **dd2d_v4 carries no post-mortem facts, so the v2.2 yardstick has an inert evidence pathway**

The single most important finding of the run. Counted directly off the collections:

| collection | outcomes | with `post_mortem` facts |
|---|---|---|
| dd2d_v2 | 8000 | 6922 |
| dd2d_v3 | 8000 | 6732 |
| **dd2d_v4** | **8000** | **0** |

The offline harvest (`spectre_harvest.py`, `decisions.md` 2026-07-19) was **never run on
dd2d_v4**. So the v2.2 checkpoint used as the v3 yardstick trained and deploys with its
`FactEncoder` receiving nothing: **v2.2@dd2d_v4 = static + `cand_overlap` + demotion, with
no evidence tokens at all.**

Three consequences, and they reframe the whole G6/G6b arc:

1. The comparison has never been "facts vs records". It is **"no evidence vs records"**.
2. G6's ablation was **confounded**: its "no records" bar also had `cand_overlap` off, so
   the −3.37 attributed to record tokens is really *records + overlap* versus *neither*.
   G7 then measured overlap with records on in both arms. **The cell nobody ran is
   `records OFF, overlap ON`** — which is precisely the v2.2 configuration.
3. Read against that: v2.2 (no evidence + overlap) **14.66** vs v3 (records + overlap)
   **16.17**. On this evidence the record tokens are **net-harmful by ~1.5 FP**, and the
   headline v3-vs-v2.2 gap is not a static-representation problem at all.

**Decision: run the missing cell** (`p2_norec`) rather than infer it. If it lands near
14.66, records are the whole gap and the record pathway needs fixing, not the encoder.

*Not doing:* harvesting facts for dd2d_v4 to make v2.2's pathway live. The user ruled out
re-deriving v2.2, and it would not help — v3's claim is that records *replace* facts and
come free from instrumentation, so v3 having evidence where v2.2 has none is v3's
advantage. The problem is that v3 is losing anyway.

### A7 — one record per failed *sample* is a token flood; §6.1 says one per failed *query*

Measured cause for why records might hurt. The instrumented refiner emits a
`FailureObservation` per failed sample, so a candidate whose `place-buffer(o)` was retried
across many buffer poses contributes hundreds of tokens:

```
records per failed candidate:  mean 2.2, median 1, p90 1, max 290
at |F|=30:  raw mean 226 tokens, p90 542, max 2045
            v2.2's fact pathway fed ~40 for the same |F|
```

Those samples are 99.3% distinct on `(schema, args, step)` — but only in *which pose*
failed, which the token does not encode. So it is ~50× the tokens for no extra encoded
information, and one unlucky candidate can dominate the evidence memory.

**Decision: `--aggregate-records`** — collapse to one record per `(schema, args)`, keeping
the deepest step, summing effort, unioning culprits. Measured: **−88.7% tokens, max
2045 → 37**, which is v2.2's order of magnitude. This is a *faithfulness* fix, not a
tuning knob: §6.1 defines the record as the failing query and its arguments.

### A8 — **the trained model already ignores its record tokens**, and v3 in fact *matches* v2.2

The pivotal measurement of the run, and it overturned my own A6 reading. Added
`suppress_records` (a diagnostic, never a deployment mode) and ran the G6b
records+overlap checkpoint with its evidence memory emptied at every step:

| deploy | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| records ON (as trained) | 16.17 | 0.00 | 8.56 | 22.00 | 34.12 |
| records SUPPRESSED | 16.40 | 0.00 | 8.56 | 22.64 | 34.40 |

**0.23 FP.** The record tokens are doing essentially nothing. So:

- G6's −3.37 "record increment" was **`cand_overlap`**, not records — which agrees with
  G7's independent −5.07 for overlap, and means the G6 headline was mis-attributed.
- A6's inference ("records are net-harmful by ~1.5 FP") was **wrong in mechanism**. They
  are not harmful, they are *inert*. The dd2d_v4 fact-inertness finding stands and still
  matters for how the comparison is described, but it does not explain the gap.

And the gap itself is not what it looked like. Scoring both models on **both** splits:

| | val FP | test FP |
|---|---|---|
| v2.2 yardstick | 17.30 | 14.66 |
| v3 G6b rec+ov | **17.09** | 16.17 |

v3 is slightly *better* on val and worse on test, and the paired bootstrap on test already
had CI [−2.29, +5.72] including 0. **v3 matches v2.2; it does not underperform it.** The
100-episode splits simply do not resolve a ~1.5 FP difference. Beating v2.2 therefore
requires *adding signal*, not repairing a regression — a materially different problem from
the one I started the night with.

### A9 — evidence should enter where the tag join is: on objects, not as free tokens

Diagnosis from A8: the failure is not "evidence is useless" — `cand_overlap`, two compact
scalars per candidate over the *same* failure set, is worth 5 FP. It is that **free-floating
tokens are the wrong shape for this architecture**. The scorer's strength is the tag join
between scene objects and candidate arguments; a record token participates only weakly,
through pooled tag slots, and competes with scene tokens for the same attention.

**Decision: `SceneEncoderV3` (`--obj-evidence`)** — summarise the observed failures onto the
objects they *name*, as 4 scalars per object appended to the scene token input, all in
[0,1] and all zero before any failure:

```
[ frac of failed candidates that manipulate o,
  frac of hint records naming o as an argument,
  frac of hint records naming o as a culprit,
  mean normalized depth of records naming o ]
```

Domain-agnostic by construction (set membership over record fields — no geometry, no
per-environment predicate, C1/L2). Proof-tier records stay excluded exactly as in the token
path, so this does not re-import the L4 "blocked sets are large ⇒ prefer longer" correlate.

Implementation note worth keeping: `_V3Example` **subclasses** `_V2Example` rather than
widening `build_v3_example`'s return, so all three callers keep their two-tuple and
`collate_v2` flows through untouched. Like `sinusoidal_pos`, enabling it changes a
projection width and therefore retires the D-8 oracle; default off stays byte-identical.

**User steer, mid-run:** records should be *the primary driver* of v3's adaptiveness, and
their inertness is to be **fixed, not routed around** — the architecture/training is the
suspect, not the idea. Note `obj_evidence` is not a route-around: it is computed purely
from `FailureRecord` fields, so it is a record *consumption* mechanism. But the steer is
right that the token pathway itself has to be made to work, which is A10.

### A10 — **the root cause: evidence competes with geometry inside one softmax**

`CrossAttentionScorer` builds a single memory and runs one cross-attention over it:

```python
memory = torch.cat([scene_tok, glob, fact_tok], dim=1)   # (B, N + 1 + F, D)
attended, _ = self.attn(cand_emb, memory, memory, key_padding_mask=key_pad)
```

With ~10 scene tokens against up to 2045 record tokens, one softmax has to divide a fixed
attention budget between the geometry that determines feasibility and the evidence. Geometry
is reliably useful; evidence is noisy. **So discarding evidence is the loss-minimizing
policy**, and the model duly learned it — which is exactly what A8 measured. Aggregation
alone does not fix this: it lowers the ratio to ~3:1, still growing with |F|.

This is an architecture defect, not a fact about evidence. The same failure set, presented
as two compact per-candidate scalars (`cand_overlap`), is worth 5 FP — because those bypass
the attention entirely and are concatenated straight at the head.

**Decision: `CrossAttentionScorerV3` (`--evidence-attn`)** — a *second, independent*
cross-attention over the evidence memory, with the head seeing both attended vectors
(`2*D_MODEL → 3*D_MODEL`). Evidence can now be read **without giving up geometry**, so a
useful record no longer has to out-compete the scene to be seen. Domain-agnostic: it changes
how tokens are consumed, not what they are, so it carries to any environment with an
instrumented refiner.

One implementation trap worth recording: a batch row with **no** records yields an all-True
key-padding mask, and `nn.MultiheadAttention` returns **NaN** rather than an empty result for
such a row. Guarded by attending under a mask that always leaves one key live and zeroing
those rows afterwards — the same guard `model.py` already uses. Verified NaN-free with and
without records.
