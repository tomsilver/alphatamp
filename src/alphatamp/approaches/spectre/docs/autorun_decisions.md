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

| sweep | arms | question | outcome |
|---|---|---|---|
| `g8` | `jac`, `tailF`, `jac_tailF` | does dropping `dead` fix s1; does rollout-aligned \|F\| fix s3 | s1 **yes** (8.56→4.84); s3 **no** |
| `p2` | `norec`, `agg` | the missing G6 cell; does taming the token flood help | records cost when trained on; aggregation helps (−0.37) |
| `p3` | `objev`, `jac_objev` | evidence via the tag join instead of as tokens | ~tie; fixes s3 alone but wrecks s1 |
| `p4` | `evattn`, `evattn_agg` | give evidence its own attention channel | ~tie (14.92), real but small |
| `p5` | `jac_cov` | **observed coverage/waste** | **the win** — 8.39, −6.27 |
| `p6` | `all` | coverage + aggregation + attention + jaccard | **best** — 7.56, −7.10 → the deployed config |
| `p7` | `recprimary` | rollout-aligned context mass alone | tie (−0.32) |
| `p9` | `cov_only`, `cov_norec` | is `dead` still harmful? do tokens still add? | `dead` now harmless; tokens worth 0.26 |
| `v3final` | 6 seeds | the reportable number | 7.90 ± 0.61, −6.76 |
| `v3lean` | 6 seeds | is the config without record tokens simpler *and* better? | **no** — 9.18 ± 1.41, tokens are worth 1.28 FP (A17) |
| `v22_yardstick` | 3 seeds | the baseline was 1 seed against v3's 6 | (see A18) |

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

> **Ratified** as ADR [`2026-07-27-dead-is-a-length-proxy`](decisions/06-v3-performance.md#2026-07-27-dead-is-a-length-proxy). This entry keeps the full measurement narrative.

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

The offline harvest (`spectre_harvest.py`, [`decisions.md` 2026-07-19](decisions/README.md)) was **never run on
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

> **Ratified** as ADR [`2026-07-27-record-tokens-are-ignored-at-inference`](decisions/06-v3-performance.md#2026-07-27-record-tokens-are-ignored-at-inference). This entry keeps the full measurement narrative.

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

> **Ratified** as ADR [`2026-07-27-evidence-needs-its-own-attention-channel`](decisions/06-v3-performance.md#2026-07-27-evidence-needs-its-own-attention-channel). This entry keeps the full measurement narrative.

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

### A11 — the missing cell, and a sharper statement than A8

> **Ratified** as ADR [`2026-07-27-record-tokens-are-ignored-at-inference`](decisions/06-v3-performance.md#2026-07-27-record-tokens-are-ignored-at-inference) (jointly with A8). This entry keeps the full measurement narrative.

`p2_norec` (records OFF, **overlap ON**) is the cell the G6 ablation never ran, and it is
the closest v3 analogue of the v2.2 configuration:

| arm | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| p2 `norec` | **15.34** | 0.00 | **4.64** | 26.24 | 30.48 |
| p2 `agg` (records, aggregated) | 15.80 | 0.00 | 5.96 | 27.88 | **29.36** |
| G6b rec+ov (raw records) | 16.17 | 0.00 | 8.56 | **22.00** | 34.12 |
| *v2.2 yardstick* | *14.66* | *0.00* | *6.20* | *26.00* | *26.44* |

Two corrections to A8, both worth having:

1. **Records are not merely inert during training — they cost.** −0.83 FP overall against
   the no-records cell, and **−3.9 FP at s1** (4.64 → 8.56). Yet `suppress_records` showed
   the *deployed* model barely reads them (0.23 FP). Both are true: the token stream is
   ignored at inference but still shapes the weights during training. A stream the model
   learns to discard is not free.
2. **Aggregation genuinely helps records** (−0.37 vs raw; s1 8.56 → 5.96, s3 34.12 →
   **29.36**, the best s3 of any arm). So the flood was a real defect, not a red herring —
   which also means the remaining record pathway is worth fixing rather than abandoning.

**Per stratum the best arms are already spread across configurations**: s1 belongs to
no-records (4.64 < 6.20), s2 to *raw* records (22.00 < 26.00), s3 to *aggregated* records
(29.36, still > 26.44). No single arm holds all three, and s3 remains the only stratum where
nothing v3 does beats the yardstick. That is what P4 (give records their own attention
channel) and P5 (observed coverage/waste) are for.

Against the yardstick, all three are **statistically tied** — norec +0.68 CI [−2.01, +3.86],
agg +1.14 CI [−0.94, +3.44], G6b +1.51 CI [−2.29, +5.72]. None is significantly worse; none
is better. Breaking the tie needs added signal, which is the point of P4–P7.

### A12 — the observed culprit set over-covers, so `coverage` is a blunt instrument

Measured how many *distinct* objects have been reported as culprits after k attempts:

| stratum | \|F\|=1 | 3 | 5 | 10 | 20 | objects actually needed |
|---|---|---|---|---|---|---|
| s1 | 1.7 | 3.5 | 4.6 | 5.4 | 6.5 | 1 |
| s2 | 2.4 | 4.6 | 5.7 | 6.8 | 7.3 | 2 |
| s3 | 1.9 | 4.2 | 5.2 | 6.3 | 7.6 | 3 |

The set **accumulates** as a rollout proceeds, which is what makes it an adaptive signal at
all. But it over-covers by ~2.5× — an object that blocked in *some* configuration is not
necessarily in the minimum feasible subset. So `coverage = |S(c) ∩ culprits| / |culprits|`
cannot distinguish "removes the right three" from "removes any three". It still separated
2.45× at s3, because feasible candidates preferentially remove the *frequently* observed
blockers — which points at the obvious refinement if the plain version underperforms:
**weight each culprit by how often it was reported** rather than treating the set as flat.
Recorded, not implemented — no point adding a second feature before the first is measured.

### A13 — **what actually worked: observe the necessity you cannot predict**

The winning change, and the reason it works, stated for whoever reads this next.

G8 established that `dead` is a **length proxy**: right at s3 where long plans are needed,
wrong at s1 where short ones are. Every attempt to *tune* that proxy traded one stratum for
another. The fix was to stop proxying and state the quantity directly — at s3 three objects
block and the right candidate removes all three — which is a **count**, not a length.

The records already contain it. So:

```
coverage = |S(c) ∩ culprits| / |culprits|        waste = |S(c) \ culprits| / |S(c)|
```

**This is §5.1's necessity conditioning with `p_i` observed instead of predicted.** The
proposal's mechanism needed a head to predict per-object necessity from geometry, and was
cut when D2 showed it addressed the wrong deficit. Once the refiner *reports* culprits, the
same two features fall out of the record with no head, no second loss, and no geometry
routine — and they are **more** C2-legal than the predicted version, because nothing is
inferred by us at all. It is also exactly why this is not `clears` (L2): `clears` was a
geometric routine *we* ran; this is the refiner reporting a collision check it already did.

Result: **8.39 vs 14.66, −6.27 FP, CI [−8.92, −3.74]**, weak dominance at every stratum.

Three things worth carrying forward from *how* this was found:

1. **The diagnostic chain was partly a red herring, and that is fine.** `suppress_records`
   → shared-attention competition → separate channel was a correct diagnosis of why *tokens*
   were inert, and the fix (`evidence_attn`, 14.92) is real but small. The large win came
   from a different question: not "why is the model ignoring evidence?" but "what is the
   feature it is using a bad proxy for?".
2. **Cheap signal checks before expensive sweeps paid for themselves.** Coverage was
   measured to separate feasible from infeasible 2.45× at s3 *before* a single epoch was
   trained. The features that measured weakly (object-evidence's column 4) also
   underperformed in training.
3. **A leakage audit is mandatory for a result this large.** Run before reporting: zero at
   |F|=0, culprits only from failed candidates in the context, deploy loop breaks on success
   before a successful candidate can enter it. 0 violations.

### A14 — records drive it, but through *features*, not through *tokens*

The ablation that decides how to describe the contribution:

| arm | ALL | s0 | s1 | s2 | s3 | vs v2.2 |
|---|---|---|---|---|---|---|
| deployed (coverage + tokens + aggregation + attention) | **7.56** | 0.00 | 1.32 | 15.88 | 13.04 | −7.10 |
| coverage, **no record tokens at all** | 7.82 | 0.00 | 3.48 | 12.28 | 15.52 | −6.84 |
| coverage only (no aggregation/attention) | 8.39 | 0.00 | 2.72 | 12.64 | 18.20 | −6.27 |
| rollout-aligned context, no coverage | 14.34 | 0.00 | 8.04 | 17.48 | 31.84 | −0.32 (n.s.) |
| *v2.2 yardstick* | *14.66* | *0.00* | *6.20* | *26.00* | *26.44* | — |

> ⚠️ **A14's headline conclusion is WRONG and is corrected in A17.** The 0.26 figure is a
> 1-seed artifact; at 6 seeds each the tokens are worth **1.28 FP** and halve the variance.
> The rest of this entry (what the arms are, that coverage carries s2/s3) stands.

**The per-failure token stream looks worth 0.26 FP on top of coverage** — everything else,
the whole −6.84, appearing to come from two scalars per candidate. **This does not survive
more seeds; see A17.**

This is not "records don't matter". `coverage`/`waste` are computed **from
`FailureRecord.culprits`**; nothing else in the system can see which object the refiner's
collision check found blocking. So the record schema is doing the driving. What is marginal
is the *encoding* of a record as its own attention token.

The honest statement for the writeup: **one canonical `FailureRecord` carries the adaptive
signal; its most valuable consumption is as compact per-candidate features, with
per-failure tokens a small further increment.** That is a better result for the
generality claim than the reverse would have been — features over a reported culprit set
need no per-environment token vocabulary at all.

Also worth noting against the earlier diagnosis: rollout-aligned context mass, which looked
like a strong lever on paper (53.7% of training carried no evidence), is a **tie** on its own
(−0.32, n.s.). The measurement was right; the inference that it was limiting was not.

### A19 — two caveats that were only ever said out loud

Both surfaced while explaining the result at the end of the run, after the docs were
written. Recorded so they are not rediscovered as objections.

**1. P-v3-1's bar is cross-collection.** The 17.08 astar-dist target was measured on
**dd2d_v3**; v3's s2 of 13.03 ± 1.52 is on **dd2d_v4**. The two collections differ on only
~0.08% of candidate labels ([`decisions.md` 2026-07-26](decisions/README.md)), so they are comparable at that
level — but it is not literally the same benchmark, and any writeup should say "s2 = 13.03
on dd2d_v4, against a 17.08 bar measured on dd2d_v3" rather than implying one number beat
the other on identical data.

**2. The `clears` defence now rests entirely on *observed vs computed*.** §5.3 of the
proposal defended necessity features against L2's rejected `clears` on the grounds that
necessity was *learned* rather than *given*. That argument no longer applies: nothing is
learned in `coverage`/`waste` — the culprits are read off the record. The defence that
remains is narrower and must be made precisely:

> `clears(S)` answers a **counterfactual** — would removing S unblock the target — for any
> subset, without trying it, by running a geometry routine *we* wrote. `coverage` only
> knows objects **already observed blocking in attempts already made**. It cannot say
> whether removing S will work; it says S contains k of the blockers seen so far. Strictly
> weaker, purely retrospective, and exactly what §6.1 lists `culprits` for.

That holds, but it is now load-bearing where it used to be belt-and-braces. It belongs in
the defence-risk register (§8) as the single most likely reviewer objection.

**3. Unverified hypothesis, flagged as such.** Sampling one s2 episode, a context of 8
failed candidates produced only **1** record token, because most DD2D failures are
`retrieve` — proof-tier, so routed to demotion and excluded from the token path. If that
holds generally it would explain A17's finding that tokens contribute almost entirely at
**s1** (shallower failures, more often hint-tier) while s2/s3 tie. **One episode is not
evidence.** The check is a tier split of emitted records by stratum; it was not run.

### A18 — the v2.2 yardstick is v2.2's *best* seed; the honest comparison is stated against it anyway

The baseline was 1 seed against v3's 6, so I trained two more with v2.2's own recipe
(`train_v2 --evidence --use-overlap`, frozen under D-7 — run, not edited):

| v2.2 seed | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| 0 — *the published yardstick* | **14.66** | 0.00 | 6.20 | 26.00 | 26.44 |
| 1 | 16.57 | 0.00 | 4.76 | 23.84 | 37.68 |
| 2 | 20.57 | 0.00 | **30.04** | 20.52 | 31.72 |
| **mean ± sd** | **17.27 ± 3.02** | 0.00 | **13.67 ± 14.20** | 23.45 ± 2.76 | 31.95 ± 5.62 |

**Seed 0 is v2.2's best of three.** So there are two defensible comparisons:

| | v3 (6 seeds) | v2.2 | verdict |
|---|---|---|---|
| vs the **published seed-0 yardstick** | 7.90 ± 0.61 | 14.66 | **−6.76**; s2/s3 won, s1 a tie |
| vs v2.2's **3-seed mean** | 7.90 ± 0.61 | 17.27 ± 3.02 | −9.37; every stratum won |

**Decision: report the first.** Comparing against a baseline's best seed is the conservative
choice, it is the number already published throughout this project, and quoting the 3-seed
mean would be selecting the framing that flatters v3 after seeing both. The 3-seed mean is
recorded here so the choice is visible rather than silent.

**v2.2's instability is itself a finding, and it is R8's.** Its s1 spread is **±14.20** —
seed 2 lands at 30.04 because `relrank` selected a bad epoch. That is precisely the
miscalibration R8 replaced with uncensored deployed-val-FP. So v2.2's own variance argues
for one of v3's changes; it does not, however, explain v3's margin — §7.1 of `as_built_v3`
shows every v3 arm *without* coverage ties v2.2 despite using the v3 selector.

**Method note that nearly cost a wrong number:** scoring immediately after launching those
two runs read `best.pt` at epoch 10/30 and produced a 3-seed v2.2 of 17.56 that looked
plausible. `train_*` rewrites `best.pt` whenever selection improves, so a checkpoint from a
live job is a mid-training model. `spectre_score_v3` now warns when `best.pt` was written in
the last two minutes.

### A17 — **correction to A14: record tokens are worth 1.28 FP, not 0.26**

A14 concluded, from a single seed, that the per-failure token stream contributed almost
nothing on top of coverage (7.56 with tokens vs 7.82 without) and that the adaptive signal
rode entirely on the compact features. **Six seeds each say otherwise:**

| | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| deployed (**with** record tokens) | **7.90 ± 0.61** | 0.00 | **5.60 ± 3.06** | 13.03 ± 1.52 | 12.96 ± 2.46 |
| lean (**coverage only**, no tokens) | 9.18 ± 1.41 | 0.00 | 10.78 ± 6.47 | 12.91 ± 0.84 | 13.03 ± 2.00 |
| *v2.2* | *14.66* | *0.00* | *6.20* | *26.00* | *26.44* |

- **Tokens are worth 1.28 FP overall**, five times the 1-seed estimate.
- **They mostly buy s1**: 5.60 vs 10.78 — the lean model is *worse than v2.2* there (6.20).
- **They halve the variance**: overall sd 0.61 vs 1.41, s1 sd 3.06 vs 6.47. The token stream
  is not just a small mean improvement, it is what makes the model *stable*.
- Both configurations still beat v2.2 (lean −5.48, CI [−8.52, −2.53]), so the headline
  survives either way — but the deployed config is the right one.

**Two lessons, and the second is the uncomfortable one.**

1. **s2 and s3 genuinely do not need the tokens** (12.91 vs 13.03, 13.03 vs 12.96 — both
   ties). The tokens' entire contribution is at s1, the stratum with the fewest observed
   failures. That is the opposite of where I predicted a per-failure stream would help, and
   it is worth understanding rather than filing away.
2. **I published a wrong conclusion from one seed in the same run in which I had just
   written up A16 about exactly that failure mode.** A14 was stated as fact ("the per-failure
   token stream is worth 0.26 FP") on a 0.26 difference — a quarter of what turned out to be
   the between-seed sd. The rule from A16 (compare the margin to the spread, not to zero)
   applies to *ablations*, not only to headline strata, and I did not apply it.

**Consequence for the writeup:** the contribution is *one canonical record consumed two
ways* — compact per-candidate features (`coverage`/`waste`, carrying s2/s3) and a
per-failure token stream (carrying s1 and the stability). Neither alone is the method.

### A16 — three seeds over-claimed s1; six corrected it

> **Ratified** as ADR [`2026-07-27-margin-must-be-compared-to-seed-sd`](decisions/06-v3-performance.md#2026-07-27-margin-must-be-compared-to-seed-sd). This entry keeps the full measurement narrative.

The final headline, and the one place where running more seeds changed a conclusion rather
than tightening it.

| seed | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| 0 | 7.50 | 0.00 | 1.16 | 15.80 | 13.04 |
| 1 | 7.63 | 0.00 | 2.72 | 12.24 | 15.56 |
| 2 | 7.19 | 0.00 | 7.48 | 12.96 | 8.32 |
| 3 | 8.05 | 0.00 | 6.68 | 11.24 | 14.28 |
| 4 | 8.08 | 0.00 | 6.28 | 12.88 | 13.16 |
| 5 | 8.94 | 0.00 | 9.28 | 13.08 | 13.40 |
| **mean ± sd** | **7.90 ± 0.61** | 0.00 ± 0.00 | **5.60 ± 3.06** | **13.03 ± 1.52** | **12.96 ± 2.46** |
| *v2.2* | *14.66* | *0.00* | *6.20* | *26.00* | *26.44* |

**−6.76 FP, CI [−9.43, −4.40].** Weak dominance holds — nothing regresses — but:

- **s2 and s3 are genuine wins**, ~2× and stable (sd 1.52 and 2.46).
- **s1 is a tie.** +0.60 margin against a 3.06 seed sd = 0.20 sd, and **2 of 6 seeds** beat
  6.20. At 3 seeds it read **3.79 ± 3.29** and I wrote it up as a win.

**The lesson is about the project's own seed rule.** "≥3 seeds to report" is the stated bar,
and on the widest-spread stratum three seeds produced a confident, wrong claim — not because
three is unlucky but because the margin (0.60) was a fifth of the spread (3.06) and nobody
had checked that ratio. **Where a per-stratum claim is load-bearing, compare the margin to
the seed sd, not merely to the baseline.** A sign is not a result.

Worth noting what did *not* move: overall FP was 7.44 ± 0.23 at three seeds and 7.90 ± 0.61
at six — the headline was never in doubt, only the per-stratum breakdown.

### A15 — CI: the pylint debt is real, pre-existing, and I stopped trying to automate it

`pytest --pylint` over the spectre tree reports **371 messages**, dominated by
`line-too-long` (169), `import-outside-toplevel` (92) and `missing-function-docstring`
(83). Of these, **58 are in the v3 modules I own**; the rest is pre-existing debt in older
experiment scripts and tests.

I tried twice to fix the line-length share automatically and reverted both times:

1. Wrapping each long line independently treated `CONST = 4  # trailing comment` as prose
   and would have dropped the `#` — it broke a test, which is how I caught it.
2. A paragraph-aware rewrite (only inside docstrings, whole paragraphs at a time) still
   produced five orphaned lines, because its "is this prose?" heuristic classified a
   sentence beginning *"class as `failure_action`…"* as code.

**Decision: leave the line-length debt, documented.** It is cosmetic (all pylint `C`), it is
mostly not mine, and two automated attempts produced subtly mangled prose in exactly the
documents this project relies on for its reasoning. A deliberate manual pass is the right way
to clear it, not a regex at 01:00.

**Exact CI state at the end of the run**, so nobody has to re-derive it:

| check | state |
|---|---|
| `pytest tests/approaches/spectre/` | **407 pass**, 20 deselected |
| the same with `-m slow` | **19 pass**, 1 skipped — includes the D-8 equivalence oracle |
| `mypy src/alphatamp/approaches/spectre/` | **clean**, 65 files |
| `mypy .` (repo-wide) | **19 errors in 7 files**, all pre-existing — `test_necessity`, `test_v3_equivalence`, `test_domain`, `test_instrumentation_is_observational`, `test_spectre_harvest`, `test_spectre_geometry`, `spectre_d2_s2` |
| `pytest --pylint` over spectre | **371 messages**, 58 in v3 modules, the rest pre-existing |

So **`./run_ci_checks.sh` is not green** — that is the honest state of G11's CI criterion,
and the residual is bounded and enumerated rather than vague.

**Process failure to record:** two `p5_jac_cov` processes raced on one checkpoint path after
I relaunched following a collate crash, so the first reported checkpoint scored 8.57 and then
8.39 as the second run overwrote it. Same config, so the conclusion is unaffected, but the
provenance was muddy — the reportable number is the clean 3-seed re-run. `spectre_sweep.py`
should refuse to start an arm whose checkpoint directory is already being written.

> ⚠️ **Corrected 2026-07-27 — the "clean 3-seed re-run" does not exist.** All three
> `p8_cov_final_s{0,1,2}` runs stopped at **epoch 5 of 30**; their `best.pt` files are
> mid-training stubs. Scored, `p8_cov_final_s0` gives **26.97 ALL with s0 = 36.64**, against
> ~8 for the finished config and 0.00 at s0 for every other arm. **The recommendation above
> is withdrawn**: prefer `p5_jac_cov` (complete, 30 epochs, race notwithstanding) or the
> retrained `checkpoints_v3_abl_cov_rec`. This is why a checkpoint's existence is not
> evidence its run finished — `precompute_dd2d_cache._warn_if_undertrained` now reads the
> training log, and `_is_mid_training` reads `train_v3`'s `.owner` marker and skips.
> See `decisions.md` / [`notebook.md` 2026-07-27](notebook/README.md).
