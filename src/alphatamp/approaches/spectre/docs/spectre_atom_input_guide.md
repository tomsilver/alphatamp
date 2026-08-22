# Adding $s_0$ and Goal Atoms to SPECTRE's Input — Implementation Guide

> **Status: Rung A IMPLEMENTED 2026-08-19** (code + tests + smoke-verify; the 2-arm FP
> evaluation of §6–§7 is deferred, and **Rung B is reserved but not built**, gated behind
> `atom_mode="tokens"` → `NotImplementedError`). See ADR/notebook `decisions/07` /
> `notebook/07` 2026-08-19 and the root `CLAUDE.md`. **Deviations from this guide, as built:**
> (1) 0-ary atoms route to the scorer's global summary token via a new optional `glob_extra`
> kwarg on both scorers (not a widened `glob_feats`, which would re-randomize `glob_proj`);
> (2) per-atom masks are derived in-model from `pred != 0` rather than emitted as separate
> `*_mask` tensors; (3) 0-ary atoms are folded into the shared per-atom array (a real
> predicate id + an all-PAD arg row) and split out in the model, so the batch carries §2's
> index trio with no extra fields; (4) the config knob is `SpectreConfig.atom_mode` +
> `use_init_atoms`/`use_goal_atoms` (per §4), the new module is `AtomProfileEncoder`
> (`encoders.py`), and emission is a single `dataset.atom_emission(cfg)` helper.
>
> **Original epistemic framing (still current):** the **registered prediction is a null on
> DD2D and
> SB2D** — $s_0 = \mathrm{abs}(x_0)$ is derived from geometry the model already
> receives, and this codebase's own history (the facts-as-tokens demotion
> experiment) shows the net under-uses symbolic tokens when a richer signal is
> present. The reasons to build it anyway: (a) it converts "geometry subsumes
> $s_0$" from an assertion into an ablation figure a reviewer can be shown;
> (b) it is the load-bearing prerequisite for domains where the goal is *not*
> summarizable by an object flag (heterogeneous goal predicates, e.g. Restock3D
> configurations); (c) the zero-init design below makes it strictly additive —
> a null costs one training run, not a regression.

---

## 0. Current state (what exists today)

- **Input surface** (`dataset.py` → `SpectreBatch` in `encoders.py`): scene/object
  tokens (tag; boundary descriptor; pose; `obj_rel`; `obj_is_goal`), candidate
  tokens (op schema + position + arg tags), failure-record tokens, `cand_overlap`
  scalars. **No atom tokens anywhere.**
- $s_0$ enters only indirectly (pool grounding, record deltas $s_j - s_0$,
  coverage entailment). $g$ enters only as `obj_is_goal`.
- A `FactEncoder` exists in `model.py` — **built but untrained**, a survivor of the
  cut fact-based *evidence* path, kept only for checkpoint `state_dict`
  compatibility. **Recommendation: do not revive it** (§D5); its semantics were
  failure-facts, not problem-atoms, and overloading it invites confusion.
- Predicate vocabulary already exists (built for record state deltas by walking
  the full STRIPS reconstruction, id 0 = pad/OOV), so goal and init predicates
  are representable today with no vocab change. *(Verify: goal atoms are walked
  during vocab build — they should be, since every $s_L \models g$, but confirm
  in `vocab.py`.)*

**Prerequisite verification (do first, 10 minutes):** confirm the converter-side
data actually carries what we need — the abstract initial atoms (recoverable from
any skeleton's $s_0$ on the `EpisodeRecord` pool) and the goal atoms (stored on
the episode). Confirm atom object-names pass through the same
`canonicalize`/`assign_tags` mapping as everything else, so the training-time
within-type object permutation augmentation permutes atom arguments
*consistently* with scene and candidate tags. If atoms bypassed tag assignment,
augmentation would silently break the tag join — this is the one genuinely
dangerous integration point.

---

## 1. Design decisions

### D1 — Injection site: two rungs, build Rung A first

- **Rung A (recommended first): object-centric atom profiles.** No new token
  set. Each object's scene token gains a fixed-width **atom-profile vector**
  summarizing the atoms that mention it; 0-ary/global atoms go to the existing
  global summary token. Cheapest, preserves the object-centric relational join,
  reuses all SAB machinery, smallest parameter delta.
- **Rung B (roadmap-conditioned): dedicated atom-token memory.** One token
  per atom, consumed by a **third cross-attention channel** in the scorer,
  mirroring the evidence channel. Build this when a target domain on the
  roadmap has **≥2-ary predicates in its init/goal atom language** (checkable
  from the domain design doc before any training run — §7 P3, §9 step 0b), or
  if Rung A shows signal but plateaus. The deferral is confined to this one
  module: the tensor surface (§2) carries full per-atom binding from day one,
  so Rung B is an encoder swap, never a schema change.
- **Why not append atoms to the scene memory as extra tokens in the existing
  channel:** the single-softmax competition finding. One attention softmax over
  ~10 reliably-useful scene tokens plus tens of atom tokens is exactly the
  configuration under which the model previously learned to *discard* the noisier
  token set. If atoms get their own tokens at all, they get their own channel —
  that lesson is paid for.
- **The rungs are not nested — different injection sites, different losses.**
  Rung A feeds atoms in *pre-SAB*, so object-object relational reasoning can
  condition on them, but binding is lost; Rung B preserves binding in per-atom
  tokens, but the atom memory is consulted only at scoring time — the scene
  SABs run atom-blind. Neither representationally contains the other, so
  "Rung B is at least weakly better" is false as stated; a hybrid would
  dominate representationally and is the worst first instrument (largest
  parameter delta, most confounded attribution).

### D2 — Atom encoding (and the binding limitation, stated honestly)

An atom is $p(o_{i_1}, \dots, o_{i_m})$, $m \in \{0, 1, 2, \dots\}$.

- **Rung A profile construction.** For object $o$ and atom set $A$ (either
  $s_0$'s atoms or $g$'s — kept separate, §D4):
  $$\mathrm{profile}_A(o) \;=\; \sum_{\substack{a \in A \\ o \text{ at slot } k \text{ of } a}} \big( \mathrm{predEmb}[p_a] + \mathrm{slotEmb}[k] \big),$$
  i.e. a bag of (predicate, argument-position) pairs. Sum pooling (not mean):
  atom *count* is information. **Argument position must be encoded** — an
  order-insensitive pool would make $\texttt{On}(a,b)$ and $\texttt{On}(b,a)$
  indistinguishable per-object.
- **Known limitation of Rung A — binding ambiguity.** With two same-predicate
  binary atoms $\texttt{On}(a,b)$ and $\texttt{On}(c,d)$, the profiles say "$a$
  is a 1st-arg of On" and "$b$ is a 2nd-arg of On" but *not that they belong to
  the same atom instance* — the pairing is lost. DD2D/SB2D cannot exercise this
  failure (their predicates are overwhelmingly unary), which cuts both ways: the
  limitation costs nothing there, and a Rung B built today could not be
  *validated* there either — its distinguishing capability would ship untested.
  Absent a stressing environment, the validation instrument for Rung B is a
  **synthetic binding probe**: constructed pools whose correct ranking provably
  requires resolving which arguments pair. Rung B's per-atom tokens (predicate
  embedding + arg tags *jointly* in one token) fix binding exactly. Do not paper
  over this in writing — "the target domain's atom language is ≥2-ary" is the
  precise technical statement of when Rung B becomes necessary.
- **Rung B token construction.**
  $$\mathrm{tok}(a) = \mathrm{Linear}\big[\, \mathrm{predEmb}[p_a] \;;\; \textstyle\sum_k (\mathrm{tagEmb}[o_{i_k}] + \mathrm{slotEmb}[k]) \;;\; \mathrm{provEmb}[\mathrm{init/goal}] \,\big].$$

### D3 — 0-ary / global atoms

Atoms with no object arguments (e.g. a hand-empty style predicate) attach to no
object token. Route their profile (same construction, no slot term) into the
**global summary token** the scorer already consumes. This keeps the design total
over atom arities without a special case downstream.

### D4 — Goal provenance is an axis, never a merge

Init atoms and goal atoms are *different kinds of facts* ("true now" vs. "wanted
eventually"). Keep two separate profile vectors per object (Rung A) or a
provenance embedding per token (Rung B). **Keep the existing `obj_is_goal` flag
untouched** — it is deployed, cheap, and its removal would be a second
simultaneous variable.

### D5 — New module, not `FactEncoder`

Name it `AtomProfileEncoder` (Rung A) / `AtomTokenEncoder` (Rung B). Because the
loader rebuilds the model from checkpoint config, a new module gated behind a
default-off switch means **old checkpoints are structurally unaffected** — no
`state_dict` surgery, no `strict=False`. Reviving `FactEncoder` would entangle
this feature with the cut evidence-facts path's semantics and its frozen
parameters in the deployed checkpoint.

### D6 — Zero-init, strict-superset discipline

The atom pathway's final projection into the scene token (Rung A) or the
channel's output projection (Rung B) is **zero-initialized**, following the state
delta branch precedent. Consequences: at initialization the extended model is
*exactly* the baseline (testable, §5); any learned use of atoms is opt-in by
gradient; and a null result is a clean "the optimizer had the pathway and left
it near zero," which is itself reportable.

---

## 2. Data & tensor surface changes (`dataset.py`, `encoders.py`)

New `SpectreBatch` fields (shapes for batch $B$, objects $O$, atoms $N_a$; all
gated behind the config switch, absent ⇒ current behavior bit-identical):

- **Rung A** (profiles precomputed at tensorization *or* assembled in the model
  from index tensors — prefer index tensors so embeddings stay learnable):
  - `init_atom_pred` $(B, N_a)$ — predicate vocab ids (0 = pad/OOV)
  - `init_atom_arg_tags` $(B, N_a, M)$ — object tags per slot, 0-padded to max
    arity $M$
  - `init_atom_mask` $(B, N_a)$
  - `goal_atom_*` — same trio for $g$
- Tensorization order matters: extract atoms **after** `canonicalize` +
  `assign_tags`, from the same episode-local mapping (see the prerequisite check
  in §0).
- OOV predicates at test time map to id 0 by the existing vocab guard — verify
  the guard is applied on this path too.
- Cost note: $N_a$ on DD2D/SB2D is tens of atoms; memory impact negligible
  relative to the 200-candidate pools.

---

## 3. Architecture changes (`model.py`)

### Rung A

1. `AtomProfileEncoder`: embeds `(pred, slot)` pairs, scatter-sums into
   per-object profiles $(B, O, d_{\text{atom}})$ for init and goal separately
   (plus one global profile each for 0-ary atoms), where
   $d_{\text{atom}} \ll d$ is fine (e.g. 16–32).
2. `SceneEncoder` input concat gains
   `[...; init_profile; goal_profile]`, passed through a **zero-init**
   `Linear(2 d_{\text{atom}} \to` scene-input width$)$ *added* to the existing
   pre-SAB projection (additive, mirroring the state delta pattern) — so with
   atoms present but weights at init, scene tokens are unchanged.
3. Global profiles added (zero-init) to the global summary token.
4. Nothing downstream changes: SABs, candidate cross-attention, evidence
   channel, head, loss — all untouched.

### Rung B (deferred)

1. `AtomTokenEncoder` → atom memory $(B, N_a, d)$.
2. Third `MultiheadAttention` channel in `EvidenceCrossAttentionScorer`
   (candidates as queries, atom memory as keys/values), output concatenated at
   the head through a zero-init projection.
3. Head width grows; everything else untouched.

Parameter accounting: report the delta (Rung A is a few thousand params on
$d=64$); if a reviewer-facing capacity-matched comparison is ever needed, match
by trimming $d_{\text{atom}}$, not by touching the shared trunk.

---

## 4. Config & checkpoint plumbing

- `SpectreConfig`: `atom_mode: {"off", "profiles", "tokens"}` (default `"off"`),
  plus `use_goal_atoms: bool`, `use_init_atoms: bool` (both default `True` when
  `atom_mode != "off"` — the decomposed ablation toggles these).
- The **loader invariant is non-negotiable**: `inference.load_checkpoint` reads
  `atom_mode` off the checkpoint, never from the caller, and the deploy-time
  switch set that controls what `dataset.build_example` emits gains this switch —
  a model is never scored on a feature it was not trained on.
- Old checkpoints: `atom_mode` absent ⇒ `"off"` ⇒ module not built ⇒ loading
  and inference bit-identical to today.

---

## 5. Tests (all cheap; the first two are the load-bearing ones)

1. **Off-equivalence**: `atom_mode="off"` ⇒ logits bit-identical to the current
   architecture on a fixed batch (guards the single-variable discipline).
2. **Zero-init equivalence**: `atom_mode="profiles"` at initialization ⇒ logits
   identical to baseline on the same batch (guards D6).
3. **Atom-set permutation invariance**: shuffling atom order leaves logits
   unchanged (atoms are a set).
4. **Arg-slot sensitivity**: $\texttt{On}(a,b)$ vs. $\texttt{On}(b,a)$ produce
   different profiles (the anti-collapse analog for D2).
5. **Augmentation consistency**: within-type object permutation applied to a
   synthetic episode permutes atom arg tags consistently with scene/candidate
   tags (guards §0's dangerous integration point).
6. **OOV predicate** maps to id 0 without error.
7. Existing invariants re-run green: object-order permutation invariance of
   logits, footprint point-permutation invariance, anti-collapse.

---

## 6. Training & evaluation protocol (single-variable)

- **Cheapest probe first**: two arms only — `baseline-static` (exists) vs.
  `+atoms-static` (`atom_mode="profiles"`, both init and goal on). Same data,
  same recipe (AdamW $3\times10^{-4}$, 30 epochs, PL + within-length), same
  selection (uncensored val FP, 5-epoch moving average). Static arms first
  because the question is about the *static* representation; the adaptive
  machinery is unchanged and would only add variance.
- **Decompose only on signal**: if `+atoms` separates from baseline, run
  init-only and goal-only arms to attribute it. If null, stop — do not fish.
- **Probes to read alongside FP**: worst-stratum regret (headline convention);
  the length-fit probe (linear length-$R^2$ and $\eta^2$ on per-episode
  normalized logits) — atom *counts* correlate with clutter, so this channel is
  a fresh route to the length shortcut; and the profile-projection weight norm
  (a near-zero learned norm is direct evidence the optimizer declined the
  pathway).
- 1-seed dev read → 3-seed confirmation before any number is written down, per
  standing convention.

---

## 7. Registered predictions & abort criteria

- **P1 (registered): DD2D null.** $|\Delta \mathrm{FP}|$ within seed noise of
  baseline-static. Rationale: $s_0$'s atoms are functions of geometry already in
  the input; precedent that token-fed symbolic facts go unused.
- **P2 (registered): SB2D null**, same rationale.
- **P3 (conditional, forward-looking):** the channel becomes non-null first in a
  domain whose goals are not object-flag-summarizable (heterogeneous goal
  predicates over the same objects — the Restock3D shape). This is the actual
  payoff case; DD2D/SB2D are the honest-null controls. **Arity caveat
  (registered):** if that domain's init/goal atoms are ≥2-ary (anything
  $\texttt{In}(\mathrm{item}, \mathrm{region})$-shaped), Rung A is predicted
  insufficient *there by construction* — binding loss deletes exactly the
  content such goals carry — so the P3 experiment must run with Rung B, and the
  domain's predicate arities must be read off its design doc before building
  (§9 step 0b).
- **Abort criteria:** (a) `+atoms` *worse* than baseline beyond seed noise on
  uncensored val FP → stop, record, keep `atom_mode="off"` deployed; (b)
  length-$R^2$ materially increases in the `+atoms` arm → the channel is a
  length-shortcut vector; stop and record before any mitigation attempt; (c)
  any test in §5 cannot be made to pass without touching the shared trunk →
  redesign, don't patch.

---

## 8. What this channel will NOT do (scope fence)

Do not expect the network to learn *sound symbolic rules* from atom tokens —
set-containment / universal-AND tests are exactly what attention approximated
poorly in the demotion experiment, and soundness needs the exact discrete test.
If a specific rule over $s_0$ or $g$ is ever wanted (e.g. a goal-entailment
check), compute it as a feature and let the net weight it — the established
pattern (coverage, waste, containment). This channel buys *soft context*, not
rules.

---

## 9. Checklist (build order)

1. §0 prerequisite verification (schema carries atoms; tag mapping covers them).
   **0b.** Read the target third-environment's predicate arities off its design
   doc; if ≥2-ary atoms appear in init/goal, schedule Rung B as the P3 vehicle
   (Rung A remains the DD2D/SB2D ablation vehicle).
2. `SpectreConfig.atom_mode` + loader/switch plumbing (§4), default off.
3. Tensorization (§2) behind the switch.
4. `AtomProfileEncoder` + zero-init injection (§3 Rung A).
5. Tests 1–7 green (§5).
6. Two-arm static run + probes (§6).
7. `notebook.md` entry with the P1/P2 outcome; `decisions.md` entry for
   `atom_mode` regardless of result; supersession note in the methodology doc's
   M1 bullet **only if** the null fails.
8. Rung B: build when a roadmap domain's atom language is ≥2-ary (step 0b) or
   when §6 produces signal that plateaus; if built ahead of any stressing
   environment, validate its binding capability with the synthetic probe (§D2),
   not with DD2D/SB2D runs — those cannot distinguish it from Rung A.
