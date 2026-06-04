# SPECTRE: Method Specification

**Skeleton-Pool Embedding with Contextual Transformer for REordering**

*Version 0.1 — pre-implementation specification. Sections marked* ⚠ *contain decisions that are provisional pending milestone validation.*

---

## 1. Problem Statement

### 1.1 Setting

We operate within a bilevel task-and-motion-planning (TAMP) framework. At test time, given a planning problem, a symbolic planner produces a pool of candidate **skeletons** — sequences of ground symbolic operators that provably achieve the goal under STRIPS semantics. Each candidate skeleton must then be **refined**: a continuous-parameter sampler attempts to instantiate each operator with concrete controller parameters (object grasps, placement poses, motion timings) such that the full resulting trajectory is physically feasible in the environment. Refinement can fail — the sampler may be unable to find valid continuous parameters within its budget — in which case the skeleton is discarded and the planner moves to the next candidate.

A full planning episode is defined by: one problem, a pool of K candidate skeletons S = {s₁, …, s_K}, and a sequence of attempted refinements that terminates either when some skeleton succeeds or when the per-episode attempt budget (≤ 20) is exhausted.

### 1.2 The reordering problem

Skeletons in S are produced in an order determined by the symbolic planner (typically by symbolic cost or length). The default strategy attempts them in this order. A static learned scorer — e.g. PIGINet or a heuristic-score ranker (HSR) — can re-rank S once, before any refinement attempts, using features of each skeleton in isolation.

We ask a sharper question: after t − 1 skeletons have been attempted and failed in the current episode, how should we reorder the remaining candidates ℛ_t ⊆ S to minimize the number of further attempts until the first success? This is an **adaptive reordering** problem: the policy sees a growing sequence of per-skeleton failure observations within the episode and must exploit them.

The quantity we minimize is *time-to-first-success*, measured in refinement attempts (with wall-clock as a secondary metric).

### 1.3 Why this is hard

Four properties of the problem together rule out most off-the-shelf learning-to-rank approaches.

**Open-ended skeleton space.** Skeletons are structured symbolic objects built from a lifted operator vocabulary and problem-specific object bindings. The number of distinct skeletons encountered across problems is unbounded; no fixed vocabulary or one-hot encoding applies. A prior attempt to represent the candidate set as a fixed-size vector with a padded input-slot structure (Attempt 2 below) failed for this reason.

**Variable-size candidate pools.** K varies per problem. The policy must handle pools of any size without architectural changes.

**Limited data budget.** Training data is capped at 500 episodes × ≤ 50 attempts per episode, per environment. Cross-environment transfer is desired.

**Rollout-alignment gap.** The quantity that matters at evaluation is argmax-based (which skeleton gets attempted first), not calibration-based. Standard pointwise losses (BCE over per-skeleton success probabilities) are not rollout-aligned: they can reach high AUROC while providing no improvement on the argmax objective. Attempt 2 achieved ~85% validation AUROC while yielding zero time-to-first-success improvement over a static baseline, exactly because of this misalignment.

### 1.4 Prior attempts ruled out

- **Attempt 1 (BOX):** A bandit-with-covariance approach using a fixed skeleton vocabulary. Failed because the skeleton space is open-ended and the vocabulary-size assumption is unrealistic.
- **Attempt 2 (masked-prediction MLP):** A fixed-size padded input vector with masked-token prediction via pointwise BCE. Failed because of (a) the fixed-size/masked-slot representation, (b) training contexts that included revealed successes (distribution shift from the test-time all-failures-so-far constraint), and (c) pointwise loss not being rollout-aligned.

### 1.5 Success criteria

The method must:
- Beat HSR and PIGINet-static baselines on mean time-to-first-success across the target benchmarks, averaged over ≥ 3 seeds.
- Degrade gracefully on problems where the static prior is already good (no regression on step-1 attempt quality).
- Be plug-and-play with any static prior (including π ≡ 0).
- Transfer across environment variants (e.g. train on -b7, evaluate on -b15) with partial retention of the adaptive gain.

---

## 2. Core Idea

We treat the within-episode failure history as a sequence of observations about a latent per-scene feasibility vector z. The planner's candidate pool exposes K skeletons; each has some true success probability under the (unknown) continuous-parameter constraints of the current scene, and the pattern of failures observed so far is informative about which remaining skeletons are likely to succeed.

Rather than attempt explicit Bayesian inference — which would require modeling the joint distribution over skeletons and z — we **amortize** the posterior update with a learned module. A permutation-invariant context encoder reads the set of failed-skeleton embeddings and produces a fixed-length context vector c_t that summarizes what the failure pattern implies. A scorer then re-ranks the remaining candidates conditioned on c_t.

The training objective — a listwise Plackett-Luce loss over the full set of remaining skeletons — is aligned by construction with the quantity minimized at rollout (probability that argmax selects a successful skeleton). This closes the AUROC-vs-rollout gap that broke Attempt 2.

Three design commitments follow from the framing:

1. **The skeleton encoder is permutation-equivariant over object identities.** Skeletons are typed relational structures; two skeletons that are object-renumbering-equivalent must produce identical embeddings. This enables cross-problem generalization.
2. **The context encoder is permutation-invariant over failure history.** The set of failures is order-independent by construction of the problem (the order we attempted them in is arbitrary and not physically meaningful). A Set-Transformer provides this invariance with sufficient pairwise-interaction capacity for non-trivial posterior approximations.
3. **The loss operates on full lists of remaining candidates, not pairs or individual items.** This directly optimizes the rollout objective.

---

## 3. Method Overview

### 3.1 Architecture diagram

```
                 Candidate skeleton pool  S = {s₁, …, s_K}
                                │
                                ▼
                ┌───────────────────────────────┐
                │    Skeleton Encoder  Φ        │  (shared weights,
                │    applied per skeleton       │   applied K times)
                └───────────────┬───────────────┘
                                │
                                ▼
               { e(s₁), e(s₂), …, e(s_K) }    (64-dim each)
                                │
              ┌─────────────────┴────────────────┐
              ▼                                  ▼
      ℱ_t (failed so far)                ℛ_t (remaining)
              │                                  │
              ▼                                  │
    ┌───────────────────────────┐                │
    │  Context Encoder  Ψ       │                │
    │  Set-Transformer          │                │
    │  SAB × 2  +  PMA_{k=1}    │                │
    └────────────┬──────────────┘                │
                 │                                │
                 ▼                                ▼
                c_t  ────►  ┌────────────────────────────────┐
                            │  Scorer  σ                     │
   π(s) plug-in prior  ────►│  MLP( [ e(s) ; c_t ; π(s) ] )  │
                            └────────────────┬───────────────┘
                                             │
                                             ▼
                                  { σ(s) : s ∈ ℛ_t }  (logits)
                                             │
                       ┌─────────────────────┴────────────────────┐
                       ▼                                          ▼
            TRAINING:                                  INFERENCE:
            Plackett-Luce loss                         argmax over σ(s)
            ℒ = −log( Σ_{s∈SUCC∩ℛ_t} exp σ(s)         → next attempt
                      / Σ_{s∈ℛ_t}   exp σ(s) )
```

### 3.2 Components at a glance

| Component | Role | Learned? | Output |
|---|---|---|---|
| Skeleton Encoder Φ | Map a skeleton (+ initial abstract state) to a fixed-length embedding | Yes | 64-dim vector per skeleton |
| Context Encoder Ψ | Map the set of failed-skeleton embeddings to a context summary | Yes | 64-dim vector per episode step |
| Scorer σ | Score each remaining skeleton given the context and an optional prior | Yes | Scalar logit per skeleton |
| Prior π | External static ranker (PIGINet, HSR, or zero) | No — plug-in | Scalar per skeleton |
| Loss | Optimize rollout-aligned ranking | — | Scalar per (R_t, SUCC, F) triple |

---

## 4. Components

### 4.1 Skeleton Encoder Φ

#### 4.1.1 What it does

Given a skeleton s — a sequence of ground symbolic operators plus the associated sequence of relational abstract states — Φ produces a fixed-length 64-dim embedding e(s). Two skeletons differing only by an object-renumbering of the underlying problem must produce identical embeddings (see §4.1.4 on canonicalization).

#### 4.1.2 Input representation

A skeleton is obtained from the symbolic planner as a sequence of (GroundOperator, RelationalAbstractState) pairs:

    (s₀, g₁, s₁, g₂, s₂, …, g_L, s_L)

where s₀ is the initial abstract state (shared across all candidate skeletons for a given problem), each g_i is a GroundOperator (lifted operator applied to specific objects), and each s_i ≥ 1 is the abstract state reached after applying g_i. Each RelationalAbstractState contains a set of GroundAtoms plus a set of typed Objects. The final state s_L entails the goal atoms (symbolic planner guarantee).

#### 4.1.3 Encoding procedure

**Operator tokens.** Each GroundOperator g = (op_name, (o₁, …, o_k)) is encoded as:
- A learned embedding of op_name (from a fixed-size embedding table keyed by the lifted operator vocabulary, extracted from training data).
- A learned embedding of each argument's typed-local-id (see §4.1.4). Argument embeddings are order-sensitive — argument slot 0 uses a different projection than argument slot 1 — so that PickObstruction(robot, obs:0) and PickObstruction(obs:0, robot) would be distinguished.
- A learned positional embedding for the operator's position in the skeleton.
- These are concatenated and passed through a two-layer MLP (96 → 128 → 64 with ReLU) to produce a 64-dim operator token.

**State tokens.** Each RelationalAbstractState s_i is encoded by a sub-module Φ_s:
- Each GroundAtom (predicate, (arg_1, …, arg_k)) is encoded by concatenating a predicate embedding (from an embedding table keyed by the predicate vocabulary) with the typed-local-id embeddings of its arguments. Variable predicate arity is handled by padding to the maximum arity with a learned null-argument token.
- Atom tokens are pooled via Deep Sets (mean over per-atom MLP outputs) to a single atom-pool vector.
- A **type histogram** is computed from the state's Objects set — an integer count per type. This is concatenated with the atom-pool vector. The type histogram captures objects that exist but do not appear in any atom (e.g. the count of obstructions in the initial state).
- The combined vector passes through a final MLP to produce a 64-dim state token.

**Sequence composition.** Operator tokens and state tokens are interleaved into a sequence:

    [STATE_0, OP_1, STATE_1, OP_2, STATE_2, …, OP_L, STATE_L]

Each token is augmented with a learned **token-type embedding** (two types: STATE and OP) added to its content embedding. Positional embeddings are applied over the interleaved sequence.

**Aggregation.** The token sequence is passed through a 2-layer Transformer encoder (4 heads, hidden dim 64). The output tokens are mean-pooled to yield the 64-dim skeleton embedding e(s).

#### 4.1.4 Canonicalization: object-renumbering for equivariance

Skeletons from different problems reference different concrete objects (obstruction_7 in problem A, obstruction_13 in problem B). To enable cross-problem generalization, we apply per-episode **typed-local-id renumbering**:

1. Extract the set of all objects referenced anywhere in the skeleton — in operators and in every atom of every state in the sequence, plus all objects in the Objects set of every state.
2. Group objects by type.
3. Assign within-type indices (e.g. obstruction:0, obstruction:1, target:0, robot:0) using a canonical ordering.
4. Substitute renumbered ids consistently across every operator argument and every atom.

The canonical ordering is chosen to respect equivariance: two skeletons that are object-renumbering-equivalent must yield identical canonical forms. During training, we apply random within-type permutations of local-ids as data augmentation to prevent the model from overfitting to any specific ordering convention.

#### 4.1.5 Why this design — alternatives considered

- **Fixed-vocabulary skeleton tokens** (Attempt 1): rejected. Skeleton space is open-ended.
- **Sparse fixed-size input vector with masked slots** (Attempt 2): rejected. Requires a maximum pool size; slot-specific learning defeats object equivariance.
- **Pure operator-sequence encoding (no state tokens):** considered. Sufficient to represent actions but discards per-problem information carried by s₀ (object counts, initial configuration) and by s_L (implicit goal). See §8.3 — this is the ablation baseline for the state-path contribution.
- **Graph neural network over the skeleton-induced object graph:** considered, deferred. A GNN would make the relational structure of atoms even more explicit, but the Transformer-over-interleaved-tokens approach recovers most of the same expressivity at lower engineering cost and with a vocabulary-agnostic interface.
- **Set Transformer for atom pooling instead of Deep Sets:** considered, not used. Atoms within a state do not require heavy pairwise reasoning — STRIPS semantics already structures their relations externally. Deep Sets is cheaper and likely sufficient. Upgrade path is trivial if needed.

#### 4.1.6 Inputs and outputs

- **Input:** a `Skeleton` object = sequence of `(GroundOperator, RelationalAbstractState)` pairs plus an initial `RelationalAbstractState` s₀.
- **Output:** a 64-dim real-valued vector e(s).
- **Parameters:** ~100–150k total.

#### 4.1.7 Open design questions

- ⚠ **Substage A vs Substage B** (see §8.1): initial implementation encodes only s₀ (not intermediate or final states). Full interleaved state path is an ablation to be run in M4/M8.
- ⚠ **Atom-pooling capacity:** Deep Sets provisional. If validation AUROC stalls and the state encoder is suspect, upgrade to a small Set Transformer.
- ⚠ **Handling of predicates never seen at training time:** the predicate embedding table is closed over the training vocabulary. Unknown predicates at test time currently trigger an assertion failure; graceful fallback (e.g. "unknown predicate" embedding) is not implemented.

### 4.2 Context Encoder Ψ

#### 4.2.1 What it does

Given the set ℱ_t = {e(s) : s attempted and failed so far in the current episode} of failed-skeleton embeddings, Ψ produces a fixed-length context vector c_t ∈ ℝ^64 that summarizes what the failure pattern implies about which remaining skeletons are likely to succeed.

The input is a *set*, not a sequence: the order in which skeletons were attempted is arbitrary and not physically meaningful. Ψ must therefore be permutation-invariant over its input.

#### 4.2.2 Architecture: Set-Transformer

We use a Set-Transformer (Lee et al., 2019) with the following structure:

    tokens ← [concat(e(sᵢ), ψᵢ) for sᵢ in ℱ_t]    # optional auxiliary features ψᵢ
    tokens ← Linear(d_in → d_model)(tokens)        # d_model = 64
    tokens ← SAB(tokens)                            # self-attention, no positional embeddings
    tokens ← SAB(tokens)                            # second self-attention layer
    c_t    ← PMA_{k=1}(tokens)                     # pool-by-multihead-attention to single vector
    c_t    ← Linear(d_model → 64)(c_t).squeeze()

Key properties:

- **SAB (Set Attention Block):** self-attention over input tokens without positional embeddings. This gives permutation-*equivariance* (permuting input rows permutes output rows in lock-step). Unlike Deep Sets, SABs perform explicit all-pairs interactions, which is required to capture joint-failure signals that are informative only in combination.
- **PMA (Pooling by Multihead Attention):** k = 1 learned seed vector attends over the SAB-enriched tokens, producing a single output vector. This is permutation-*invariant* (the output is independent of input order). Conceptually analogous to a BERT [CLS] token.
- **Empty-history handling:** when ℱ_t = ∅, Ψ outputs a learned parameter vector c₀ ∈ ℝ^64. This is trained alongside the rest of the model. At episode step 1 (always the case when |ℱ_t| = 0), c_t = c₀ for every episode, so step-1 ranking depends only on e(s) and π(s) — giving "reduces to static ranker at step 1" behavior by construction.

#### 4.2.3 Why this design — alternatives considered

- **Deep Sets** (sum of MLP over each input): rejected. Universal approximation requires width that our data budget doesn't support; joint failure reasoning is especially weak.
- **Concatenate-then-MLP with padding** (Attempt 2): rejected. Breaks permutation invariance; fixes a maximum set size.
- **Ordinary Transformer with positional embeddings:** rejected. Positional embeddings make outputs order-dependent, violating permutation invariance.
- **Induced Set Attention Block (ISAB) variant:** considered, not needed. ISAB reduces attention cost from O(n²) to O(nm) via learned inducing points. Our |ℱ_t| ≤ 20 makes O(n²) trivial.

#### 4.2.4 Auxiliary features ψᵢ (optional)

Each failed skeleton's token may optionally be augmented with per-failure auxiliary features:
- Stuck-step index (at which point in the skeleton did refinement fail?)
- Number of sampler retries attempted
- Refiner wall-clock time

These features require refiner instrumentation and are governed by hypothesis H3 in the research brief. ⚠ Whether H3 features are included in the first implementation is deferred to §8.2.

Scale mismatch between e(sᵢ) (64-dim, learned) and ψᵢ (small, hand-crafted, possibly integer-valued) is managed by layer normalization at the Ψ input and by projecting ψᵢ through a small MLP to a comparable dimension before concatenation.

#### 4.2.5 Inputs and outputs

- **Input:** set of pairs {(e(sᵢ), ψᵢ) : sᵢ ∈ ℱ_t}. Variable size, possibly empty.
- **Output:** vector c_t ∈ ℝ^64.
- **Parameters:** ~30–50k.

### 4.3 Scorer σ

#### 4.3.1 What it does

For each remaining candidate s ∈ ℛ_t, σ computes a scalar logit σ(s) = σ(e(s), c_t, π(s)) that determines ranking order.

#### 4.3.2 Architecture

    π_proj ← Linear(1 → 8)(π(s))                    # project scalar prior
    x      ← Concat[e(s); c_t; π_proj]              # 64 + 64 + 8 = 136 dim
    h      ← LayerNorm(Linear(136 → 128)(x)).gelu()
    h      ← LayerNorm(Linear(128 → 64)(h)).gelu()
    σ(s)   ← Linear(64 → 1)(h).squeeze()

#### 4.3.3 Design notes

- **Prior projection.** Concatenating a scalar π(s) directly with 64-dim vectors would give the prior channel negligible gradient weight. Projecting to an 8-dim embedding first balances the contribution. 8 is a tunable hyperparameter.
- **Prior dropout during training.** With probability p_drop (default 0.2), π(s) is replaced with 0 during training. This prevents the scorer from degenerating into "parrot the prior" and ensures that e(s) and c_t carry meaningful signal. This is also what makes FM-A (context collapse to prior) detectable: if the model ignores c_t, prior dropout degrades training loss significantly.
- **Initialization toward the prior.** At the start of training, weights are initialized such that the scorer approximately outputs α·π(s) for small α. This means an untrained SPECTRE behaves similarly to the static ranker, and training learns corrections.

#### 4.3.4 Inputs and outputs

- **Input:** e(s) ∈ ℝ^64, c_t ∈ ℝ^64, π(s) ∈ ℝ (scalar).
- **Output:** σ(s) ∈ ℝ.
- **Parameters:** ~20k.

### 4.4 Plug-in static prior π

#### 4.4.1 Role

π provides a per-skeleton scalar score from any static (context-independent) ranker. Supported options:
- **π ≡ 0:** no prior; scorer relies purely on learned components.
- **HSR (heuristic-score ranker):** existing baseline in the substrate.
- **PIGINet-static:** a separately trained static learned ranker.

π is **not learned jointly** with SPECTRE. It is fixed before SPECTRE training and remains fixed throughout.

#### 4.4.2 Why plug-in

Separating π from the learned components yields two benefits. First, it allows clean comparison: SPECTRE + HSR vs. HSR alone directly measures the *adaptivity premium* — the contribution of context-dependent reasoning on top of an existing static ranker. Second, it allows drop-in compatibility with future static rankers without retraining SPECTRE.

---

## 5. Training Procedure

### 5.1 Data collection

For each training problem:
1. Invoke the symbolic planner to obtain the candidate pool S.
2. Attempt refinement on every s ∈ S in some canonical order (with randomized seeds), regardless of whether some earlier s succeeds. Record outcomes for the full pool.
3. Save per-episode record: (problem_id, S, per-skeleton success/failure, per-skeleton refinement wall-clock, optional refiner instrumentation).

Target dataset size: 500 training episodes, 100 validation, 100 test — per environment.

### 5.2 Training example construction (F-subset sampling)

One raw episode yields many training examples via subset sampling. For each episode e:
- Let SUCC_e = skeletons in S_e that succeeded; FAIL_e = skeletons that failed.
- Sample a subset F ⊆ FAIL_e (uniformly over subsets, including F = ∅ and F = FAIL_e).
- Let R = S_e \ F (the "remaining" pool, which includes SUCC_e entirely plus FAIL_e \ F).
- This (R, SUCC_e ∩ R, F) triple is one training example.

**Critical constraint:** F must contain only failed skeletons, never successful ones. At test time, the set of skeletons already attempted by construction consists entirely of failures (a success would have terminated the episode). Training on contexts F that include successes would create a train-test distribution mismatch. This constraint was violated in Attempt 2 and was a direct cause of its failure.

Because |FAIL_e| can be large (up to ~19 for a 20-attempt episode with one success), subset sampling yields O(2^|FAIL_e|) distinct training examples per episode — a dense per-episode data yield that compensates for the small absolute episode budget.

### 5.3 Loss: Plackett-Luce

For a training example (R, SUCC_R, F):

    c_t      ← Ψ({ (e(s), ψ(s)) : s ∈ F })    or c₀ if F is empty
    logits   ← { σ(s ; e(s), c_t, π(s)) : s ∈ R }
    Z        ← Σ_{s ∈ R}      exp( σ(s) )
    Z_plus   ← Σ_{s ∈ SUCC_R} exp( σ(s) )
    ℒ        ← − log(Z_plus / Z)
             = logsumexp(all logits) − logsumexp(success logits)

Computed in the numerically stable logsumexp form.

The loss is the negative log-probability, under a softmax parameterization, that picking one item from R via the Plackett-Luce choice model yields some element of SUCC_R. Equivalently: the negative log of the top-1 retrieval probability under the softmax-over-logits induced by σ.

Rollout alignment: at test time, we select s_t = argmax_{s ∈ ℛ_t} σ(s). Under temperature → 0, softmax sampling concentrates on the argmax, so P(softmax samples a success) → P(argmax is a success) — exactly the quantity rollout is judged on. Minimizing ℒ is therefore minimizing (an upper bound on) rollout failure probability.

### 5.4 Optimization

Defaults, provisional (⚠ tune in M6):
- Optimizer: AdamW, lr 3e-4, weight decay 0.01.
- Batch: 16 training examples (i.e. 16 (R, SUCC_R, F) triples), accumulated if GPU memory requires it.
- Schedule: cosine decay to 10% of peak, with 500-step linear warmup.
- Total training: 20 epochs over the F-subset distribution (empirically tuned — overfitting onset typically at 10–15 epochs).
- Dropout: 0.1 within Transformer / Set-Transformer / scorer MLPs; 0.2 on π(s) (see §4.3.3).
- Data augmentation: per-example random within-type object renumbering (see §4.1.4).

### 5.5 Multi-seed discipline

Every reported number is the mean ± std across ≥ 3 independent training seeds. Seed variance exceeding half the gap between treatment and baseline is grounds to expand to ≥ 5 seeds or to conclude the gain is noise.

---

## 6. Inference / Test-Time Behavior

### 6.1 Per-episode loop

```
on episode start:
    compute s₀ (initial abstract state)
    generate candidate pool S via symbolic planner
    compute e(s) for all s ∈ S, in parallel    # single Φ forward pass, batched
    F ← ∅
    R ← S
    t ← 1

while not success and t ≤ attempt_budget and R ≠ ∅:
    c_t ← Ψ({ (e(s), ψ(s)) : s ∈ F })  if F ≠ ∅ else c₀
    for s in R:
        compute σ(s ; e(s), c_t, π(s))
    s_t ← argmax_{s ∈ R} σ(s)
    outcome ← refine(s_t)
    if outcome = success: return s_t
    F ← F ∪ {s_t}
    R ← R \ {s_t}
    t ← t + 1

return failure
```

Key properties:
- Φ is called exactly once per episode (batched over all of S). This is the dominant one-time compute cost.
- Ψ is called once per step t (cost O(|F_t|²)).
- σ is called once per remaining skeleton per step. Total σ calls per episode: O(K²) in the worst case, which for K ≤ 30 is negligible.

### 6.2 What is learned vs. hard-coded

| Component | Learned | Hard-coded |
|---|---|---|
| Skeleton → e(s) via Φ | All weights | Canonicalization rule (typed-local-id renumbering) |
| ℱ_t → c_t via Ψ | All weights, including c₀ | Set-Transformer structure (SAB×2 + PMA) |
| Scoring via σ | All weights | Prior projection dimensions; concat order |
| Prior π | (None — π is external) | Choice of π source (HSR, PIGINet, zero) — experimental variable |
| Decision rule | — | argmax over σ |
| Loss | — | Plackett-Luce form |

### 6.3 Integration with the TAMP pipeline

SPECTRE is a drop-in replacement for the static skeleton ranker in the bilevel planner. It does not modify:
- The symbolic planner's candidate-generation procedure.
- The refinement procedure and its outcome semantics.
- The abstraction from continuous states to RelationalAbstractStates.

It adds:
- A call to Φ once per episode at candidate-generation time.
- A call to Ψ and a re-scoring of R at each attempt step.

The integration layer is expected to be < 200 lines in total.

---

## 7. Assumptions and Constraints

### 7.1 Environmental assumptions

- Substrate is `kinder` 2D environments (ClutteredRetrieval2D, ClutteredStorage2D, Obstruction2D, Motion2D, StickButton2D), accessed via the `bilevel_planning` and `kinder_models` APIs. Primary evaluation benchmarks: ClutteredRetrieval2D-o10, ClutteredStorage2D-b7, ClutteredStorage2D-b15.
- The symbolic planner is assumed to produce only candidate skeletons whose final abstract state s_L entails the goal atoms. SPECTRE does not re-check this property.
- The refiner's success/failure outcome is assumed to be deterministic given a fixed random seed. Multi-seed stochastic refiner behavior is handled at evaluation time by averaging over seeds; within a single training episode, one outcome per skeleton is recorded.

### 7.2 Data constraints

- Training budget: 500 problems × ≤ 50 refinement attempts per problem, per environment. Held-out: 100 validation, 100 test, drawn from the same generator with distinct seeds.
- The set of (lifted operator name, argument types) tuples encountered at test time is assumed to be a subset of that encountered at training time. Out-of-vocabulary lifted operators are not currently supported.
- Similarly for the predicate vocabulary used in abstract states (see §4.1.7).

### 7.3 Relaxation of the Phase 1 restriction (explicit)

The original research brief specifies "prior-skeleton history only (no state/goal conditioning)" as a Phase 1 input restriction. SPECTRE as specified here **relaxes this restriction in one direction**: the skeleton encoder conditions on the initial abstract state s₀ (and optionally the full relational-abstract state path), which was not present in the original restriction.

Justification: the Phase 1 restriction was intended to exclude conditioning on *concrete* state (continuous features, object poses, geometry) to preserve cross-environment transfer. Conditioning on the *relational-abstract* state is compatible with cross-env transfer because the relational interface is precisely the shared structure across envs. Additionally, s₀ is already a component of the skeleton object as produced by the planner — we are not pulling in additional pipeline observations.

This relaxation should be verified with the owner of the research brief before committing extensive implementation effort.

### 7.4 Known limitations

- **No uncertainty-aware selection.** The decision rule is greedy argmax, not information-directed sampling. Greedy is defensible within a tight attempt budget but is suboptimal in the full Bayesian decision-theoretic sense.
- **No on-policy correction.** Training F subsets are sampled i.i.d. from FAIL_e, but at test time F grows according to SPECTRE's own decisions. This is an on-policy / off-policy distribution shift (labeled FM-D in the research brief) that may require a DAgger-style correction round if observed to matter empirically.
- **No refinement-time feedback during an attempt.** SPECTRE decides which skeleton to attempt, not how long to let the refiner try before giving up. Attempt cutoffs are controlled externally.
- **Per-episode only.** Adaptivity does not cross episode boundaries; no meta-learning across episodes.

---

## 8. Open Questions

### 8.1 Substage A vs Substage B for state-path encoding — decide by end of M5

Two variants of the skeleton encoder are specified.

- **Substage A (default, simpler):** Include only s₀ as a state token prepended to the operator sequence. Sequence is [STATE_0, OP_1, OP_2, …, OP_L].
- **Substage B (full):** Interleave all state tokens. Sequence is [STATE_0, OP_1, STATE_1, …, OP_L, STATE_L].

Recommendation: implement Substage A first (it captures the largest expected gain at the lowest implementation cost), and run Substage B as an explicit ablation in M8. Report the gap.

### 8.2 Refiner instrumentation (hypothesis H3) — decide by end of M2

Auxiliary features ψᵢ in Ψ require refiner-side instrumentation (per-skeleton stuck-step index, sampler retry counts, etc.). The research brief lists this as medium-priority hypothesis H3.

Recommendation: defer to post-M8. Run M1–M8 with ψᵢ = ∅. If results beat static baselines meaningfully, add ψᵢ in a follow-up round. If results are marginal, ψᵢ is a higher-leverage intervention to try next.

### 8.3 On-policy correction (DAgger round) — decide by end of M8

If evaluation shows the model performs worse at rollout than offline AUROC(t) would predict, the culprit is likely on-policy distribution shift (FM-D). The mitigation is one round of DAgger: run the trained SPECTRE on the training problems, collect the F sets it actually produces, and retrain on this on-policy F distribution.

Recommendation: only implement if M8 uncovers an offline-vs-rollout gap.

### 8.4 Loss variant — decide by end of M6

The current specification uses uniform Plackett-Luce. An alternative is a **cost-weighted pairwise loss**, where pairs are weighted by refinement wall-clock time (making the loss proportional to the wall-clock cost of suboptimal picks, not just the attempt count).

Recommendation: use uniform PL by default; consider the cost-weighted variant only if wall-clock becomes a reported secondary metric and shows a different picture from attempt count.

### 8.5 Unknown-vocabulary graceful fallback — decide before public release

Current design hard-fails on unknown lifted operators or predicates at test time. A graceful fallback (e.g. learned "OOV" embeddings with zero-init) should be added before any evaluation involving held-out environments or tasks.

### 8.6 Atom-pooling capacity — decide if M4 validation fails

Deep Sets is specified as the atom pooler inside Φ_s. If M4 validation shows Φ is underfitting (gradient signal through Φ_s is weak, or state-path ablation shows the state encoder isn't contributing), upgrade to a small SAB + PMA over atoms.

---

## 9. Key Terminology

**Abstract state / RelationalAbstractState.** A symbolic representation of a state as a set of GroundAtoms over a set of typed Objects. Produced by the environment's state abstractor.

**Adaptivity premium.** Mean time-to-first-success of static baseline minus mean time-to-first-success of SPECTRE. The headline evaluation metric.

**Amortized posterior.** A learned function that approximates a Bayesian posterior update without performing explicit inference. Ψ plays this role in SPECTRE.

**AUROC(t).** Area under the ROC curve for predicting skeleton success, restricted to examples where exactly t failures have been observed. Used to verify that context conditioning improves with more observations.

**Bilevel TAMP.** A task-and-motion-planning architecture in which a symbolic (task-level) planner generates candidate skeletons and a continuous (motion-level) refiner attempts to instantiate each one.

**Candidate pool / S.** The set of skeletons returned by the symbolic planner for a given problem.

**Context encoder / Ψ.** The Set-Transformer-based module that maps the failed-skeleton set ℱ_t to the context vector c_t.

**Failed set / ℱ_t.** The set of skeletons attempted and failed so far in the current episode at step t.

**F-subset sampling.** The training-example construction procedure in which F ⊆ FAIL_e is sampled and used as a training-time stand-in for ℱ_t.

**Ground operator / GroundOperator.** A lifted symbolic operator applied to specific objects, with specific preconditions, add-effects, and delete-effects. See `structs.py` and `pddl.py`.

**Ground atom / GroundAtom.** A predicate applied to specific objects, e.g. `Inside(target_block_0, target_region_0)`.

**HSR.** Heuristic-score ranker. An existing static baseline in the substrate.

**ISAB.** Induced Set Attention Block. A scalability variant of SAB, not used in SPECTRE.

**Listwise loss.** A ranking loss defined over full candidate lists rather than pairs or individual items. Plackett-Luce is listwise.

**PIGINet.** Static learned skeleton ranker used as an evaluation baseline.

**Plackett-Luce loss / PL loss.** The listwise loss ℒ = −log(Z_plus / Z) specified in §5.3.

**PMA.** Pooling by Multihead Attention. A permutation-invariant attention-based pooling operation using learned seed vectors.

**Prior / π.** A plug-in static ranker that provides a per-skeleton scalar score. External to the learned model.

**Refinement.** The continuous-parameter instantiation step that follows skeleton selection. Success/failure of refinement is the per-skeleton outcome.

**Remaining set / ℛ_t.** The set of candidate skeletons not yet attempted, at step t of the current episode.

**Rollout alignment.** The property that the training loss and the evaluation metric optimize substantially the same objective. PL loss is rollout-aligned for top-1 selection.

**SAB.** Set Attention Block. Self-attention over a set of tokens with no positional embeddings, producing a permutation-equivariant output.

**Scorer / σ.** The MLP that maps (e(s), c_t, π(s)) to a scalar ranking logit.

**Set Transformer.** An architecture family for permutation-invariant set aggregation, composed of SAB/ISAB layers followed by a PMA pooler.

**Skeleton.** An ordered sequence of (GroundOperator, RelationalAbstractState) pairs from some initial abstract state s₀ to a final abstract state s_L that entails the goal atoms.

**Skeleton encoder / Φ.** The module that maps a skeleton to a fixed-length embedding e(s).

**SPECTRE.** Skeleton-Pool Embedding with Contextual Transformer for REordering. The full method specified by this document.

**Time-to-first-success.** Number of refinement attempts made in an episode until the first successful skeleton is found. The primary evaluation metric, counted in attempts.

**Typed-local-id.** A within-type index assigned to an object during canonicalization (e.g. `obstruction:0`, `target:0`). Replaces concrete object names to enable cross-problem generalization.

**Z, Z_plus.** Normalizing constants in the PL loss. Z = Σ_{s ∈ ℛ_t} exp σ(s); Z_plus = Σ_{s ∈ SUCC ∩ ℛ_t} exp σ(s).

---

*End of specification.*
