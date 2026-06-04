# SPECTRE for RoutedTransport2D: Method & Training Spec

**Implementation-ready specification for the SPECTRE model, training loop, and test-time inference, adapted to the RoutedTransport2D environment.**

_Version 1.0 — supersedes `SPECTRE_METHOD_SPEC.md` for the RT2D evaluation. Where this document and `SPECTRE_METHOD_SPEC.md` disagree, this document is authoritative for the model and training pipeline targeting RT2D. The data-collection pipeline is governed by `SPECTRE_TRAINING_PIPELINE_AS_BUILT.md` unchanged; only the augmentation rule (§4.6) requires a small amendment on the data side._

---

## 1. Purpose and scope

This document specifies:

- The full model architecture (Φ, Ψ, σ).
- The training loop (loss, optimizer, F-subset sampling, regularization, multi-seed protocol).
- The test-time inference loop, sized for the sparse rollout protocol (|F| = 0 → 1 → 2 → … attempts).
- Three deliberate deviations from `SPECTRE_METHOD_SPEC.md` that RT2D forces, with rationale.

It does **not** re-specify:

- The RT2D environment itself (see `ROUTED_TRANSPORT2D_SPEC.md`).
- The data-collection pipeline, episode schema, or `SpectreDataset` (see `SPECTRE_TRAINING_PIPELINE_AS_BUILT.md`).
- Evaluation baselines B1–B5 (see SPECTRE EDA spec).

The end deliverable from this spec is a trained model `M(seed)` and an inference function `select_next_skeleton(R, F, π) → skeleton_index` that the evaluation driver calls once per attempt step.

---

## 2. Problem recap and success criterion

At test time, given a problem, the symbolic planner produces a candidate pool `S = {s₁, …, s_K}` of skeletons (`K = K_pool`, capped at 30 for RT2D). The evaluation loop is the sparse rollout:

```
F ← ∅
R ← S
for t = 1, 2, …, attempt_budget:
    s_t ← select_next_skeleton(R, F, π)        # the model
    outcome ← refine(s_t, scene_latent, tags)  # pre-recorded in episode
    if outcome == success: return t              # time-to-first-success = t
    F ← F ∪ {s_t}
    R ← R \ {s_t}
return failure
```

**The model must produce a useful ranking from t = 1**, i.e. with `|F| = 0`. At t = 2 it has exactly one observed failure to condition on. The model is not allowed to defer using context — every step's argmax matters, and the gradient signal during training must reach all of those step regimes.

The reported metric is **mean time-to-first-success** averaged over the test split and over ≥ 3 training seeds. The headline number is the **adaptivity premium** versus the strongest baseline (B4, the Naive-Bayes log-odds adaptive ranker on canonical-key keys); positive premium is the success criterion.

---

## 3. Architecture overview

```
                                 candidate pool S
                                       │
                        ┌──────────────┼──────────────┐
                        ▼              ▼              ▼
                       s₁             s₂      …      s_K
                        │              │              │
                        └──── Φ ───────┴───── Φ ──────┘     (batched, one pass per episode)
                                       │
                              {e(s) : s ∈ S}  ∈ ℝ^{K × 64}
                                       │
                                       │
   ┌───────────────────────────────────┼────────────────────────────────────┐
   │  per attempt step t:                                                   │
   │                                                                        │
   │   {e(s) : s ∈ F_t} ─── Ψ ──→ c_t ∈ ℝ^64                                │
   │                                                                        │
   │   for s ∈ R_t:                                                         │
   │       σ(e(s), c_t, π(s)) ─── argmax ──→ s_t                            │
   │                                                                        │
   └────────────────────────────────────────────────────────────────────────┘
```

|Module|Role|Trained jointly|Output|Approx. params|
|---|---|---|---|---|
|Skeleton Encoder Φ|skeleton → 64-dim embedding|yes|`e(s) ∈ ℝ^64`|~120k|
|Context Encoder Ψ|failure-set → 64-dim context|yes|`c_t ∈ ℝ^64`|~40k|
|Scorer σ|(e(s), c_t, π(s)) → scalar|yes|`σ(s) ∈ ℝ`|~25k|
|Prior π|external static ranker|no|`π(s) ∈ ℝ`|0 (plug-in)|

Total trainable ≈ 185k. Hidden size `d = 64` throughout. Multi-head attention uses 4 heads.

---

## 4. Skeleton encoder Φ

### 4.1 Input contract

A canonicalized skeleton record consumed by Φ has the following structure (produced by the as-built `canonicalize_episode` + `collate_spectre_batch`, see §10):

```python
@dataclass
class SkeletonInput:
    op_name_ids:        LongTensor   # (L,)            lifted-operator vocab id per op
    op_arg_type_ids:    LongTensor   # (L, A)          type vocab id per (op, arg slot); 0 = pad
    op_arg_local_ids:   LongTensor   # (L, A)          within-type local id per (op, arg slot); 0 = pad
    op_position:        LongTensor   # (L,)            0..L-1
    s0_atom_pred_ids:   LongTensor   # (M0,)           predicate vocab id per atom in s_0; 0 = pad
    s0_atom_arg_type_ids: LongTensor # (M0, P)         type vocab id per (atom, arg slot)
    s0_atom_arg_local_ids: LongTensor # (M0, P)        within-type local id per (atom, arg slot)
    s0_type_histogram:  LongTensor   # (T,)            count per type id
    sL_atom_pred_ids:   LongTensor   # (ML,)
    sL_atom_arg_type_ids: LongTensor # (ML, P)
    sL_atom_arg_local_ids: LongTensor # (ML, P)
    sL_type_histogram:  LongTensor   # (T,)
```

where:

- `L` is per-skeleton operator-sequence length (padded to batch max).
- `A = vocab.max_operator_arity` (5 for RT2D — `TraverseLoadedColor⟨X⟩`).
- `M0`, `ML` are per-state atom counts (padded to batch max).
- `P = vocab.max_predicate_arity` (3 for RT2D — `Connects`).
- `T = len(vocab.types)`.

Vocab id 0 is reserved for `<PAD>` / `<OOV>` everywhere. Local-id 0 is the pad slot (so real local ids are 1, 2, 3, …). Type histogram counts the **leaf type** of each object only.

### 4.2 Operator-token sub-encoder

For each op in the L-step sequence:

```
op_name_emb        ← Embed(|ops|, d_op_name=32)(op_name_ids)              # (L, 32)
arg_type_emb       ← Embed(|types|+1, d_type=8)(op_arg_type_ids)          # (L, A, 8)
arg_local_emb      ← Embed(max_local_id+1, d_local=16)(op_arg_local_ids)  # (L, A, 16)
arg_slot_proj      ← Linear(8+16 → 16) per arg slot (slot-specific weights)# (L, A, 16)
arg_token          ← arg_slot_proj.flatten(start_dim=-2)                  # (L, A*16)
op_pos_emb         ← Embed(max_L, d_pos=16)(op_position)                  # (L, 16)
op_token_in        ← Concat[op_name_emb, arg_token, op_pos_emb]           # (L, 32 + A*16 + 16)
op_token           ← MLP_op(op_token_in)                                  # (L, 64), 2 layers, GELU
```

`MLP_op` is `Linear(D_in → 128) → GELU → Linear(128 → 64)` where `D_in = 32 + A*16 + 16`. The slot-specific projection (`arg_slot_proj` is a `ModuleList` of length `A`) is what makes argument-position semantically meaningful — `Pick(robot, item)` ≠ `Pick(item, robot)`. **A is sourced from `vocab.max_operator_arity` at model-init time.**

Pad-slot args (type id 0) are handled by the embedding tables themselves: `Embed(num_embeddings=|types|+1, padding_idx=0)`. Same for `op_arg_local_ids`.

### 4.3 State-token sub-encoder Φ_s

For each state s_i (= s_0 or s_L), with M_i atoms:

```
atom_pred_emb     ← Embed(|preds|, d_pred=32)(atom_pred_ids)              # (M, 32)
arg_type_emb      ← Embed(|types|+1, d_type=8)(atom_arg_type_ids)         # (M, P, 8)
arg_local_emb     ← Embed(max_local_id+1, d_local=16)(atom_arg_local_ids) # (M, P, 16)
arg_token         ← Concat[arg_type_emb, arg_local_emb].flatten(-2)        # (M, P*24)
atom_token_in     ← Concat[atom_pred_emb, arg_token]                       # (M, 32 + P*24)
atom_token        ← Linear(32 + P*24 → 64)(atom_token_in)                  # (M, 64)
```

`P` is sourced from `vocab.max_predicate_arity` at init time. Pad atoms (pred id 0) are masked out for the next step.

**Atom pooling — Set Transformer, not Deep Sets.** This is fix #1 (see §9.1).

```
atom_tokens       ← LayerNorm(atom_token)                                  # (M, 64)
atom_tokens       ← SAB(atom_tokens, mask=atom_mask)                       # (M, 64); 4 heads
atom_pool         ← PMA_{k=1}(atom_tokens, mask=atom_mask)                 # (1, 64)
                  ← atom_pool.squeeze(0)                                    # (64,)
```

`SAB` is a single Set Attention Block (multihead self-attention + residual + layer-norm + position-wise feed-forward + residual + layer-norm), with no positional embeddings, masked over padded atoms. `PMA_{k=1}` is one learned seed vector that attends over the SAB output.

```
type_hist_emb     ← Linear(T → 16)(type_histogram.float())                 # (16,)
state_token_in    ← Concat[atom_pool, type_hist_emb]                       # (80,)
state_token       ← LayerNorm(Linear(80 → 64)(state_token_in)).gelu()      # (64,)
```

Empty-state edge case: if `M_i == 0` (no atoms), set `atom_pool = 0` and let the type-histogram path carry the signal. This will not occur in RT2D (every state has the static atoms), but the implementation should handle it.

### 4.4 Sequence composition and aggregation

Substage A is the sequence shape: `[STATE_0, OP_1, OP_2, …, OP_L, STATE_L]` (length `L + 2`). Intermediate states are not encoded. Augment each token with a **token-type embedding** (3 ids: `STATE_0 / OP / STATE_L`) and a **sequence-position embedding** over the `L + 2` slots.

```
tokens            ← stack[state_0_token, op_token_1, …, op_token_L, state_L_token]  # (L+2, 64)
type_emb          ← TokenTypeEmbed(...)                                     # (L+2, 64), broadcast-add
pos_emb           ← SeqPosEmbed(...)                                        # (L+2, 64), broadcast-add
tokens            ← LayerNorm(tokens + type_emb + pos_emb)
encoded           ← TransformerEncoder(tokens, mask)                        # (L+2, 64); 2 layers, 4 heads
e(s)              ← mean_pool(encoded, mask)                                # (64,)
```

`TransformerEncoder` uses post-norm, GELU activations, dim-feedforward 256, dropout 0.1.

### 4.5 Canonicalization

Canonicalization (per as-built `canonicalize.py`) renumbers concrete object names to typed local ids. Two skeletons that are object-renumbering equivalent must produce identical `SkeletonInput`s after canonicalization.

The canonical ordering, applied **without augmentation**, is alphabetical by the original object name within each type. This deterministic ordering is used at validation, test, and as the base order at training.

### 4.6 Augmentation — the augmentable / non-augmentable type distinction

This is **fix #2** (see §9.2). Per-type augmentation policy is sourced from the env registry at vocab-build time:

```python
@dataclass(frozen=True)
class TypeAugPolicy:
    augmentable: bool
```

The vocab JSON gains a per-type `augmentable: bool` field. For RT2D:

|Type|augmentable|
|---|---|
|`Robot`|`True` (only one instance, but defaults to `True` for consistency)|
|`Item`|`True`|
|`Zone`|`False`|
|`Passage` (parent), `PassageColorA`, `PassageColorB`, `PassageColorC`|`False`|
|`WidthLevel`|**`False`**|
|`SizeLevel`|**`False`**|
|`GraspMode`|**`False`**|

Rationale per type is in §9.2. For all five `kinder` envs in the original SPECTRE substrate (ClutteredRetrieval2D, ClutteredStorage2D, Obstruction2D, Motion2D, StickButton2D), every problem-instance type is `augmentable=True`, so this flag is backwards-compatible.

Augmentation procedure (training only; off at val/test):

```python
def augment(skeleton_input, type_aug_policy, rng):
    for type_id in range(num_types):
        if type_aug_policy[type_id].augmentable:
            n = max_local_id_for_type[type_id]
            perm = rng.permutation(n) + 1   # +1 to skip pad slot 0
            for tensor in (op_arg_local_ids, s0_atom_arg_local_ids, sL_atom_arg_local_ids):
                mask = (op_arg_type_ids == type_id)
                tensor[mask] = perm[tensor[mask] - 1]
    # type_histogram is invariant under within-type permutation
    return skeleton_input
```

The augmentation RNG is seeded from `(seed, problem_id, epoch, skeleton_idx)` for reproducibility. The same permutation must be applied consistently **across all skeletons in a given training example** (R and F together) and across both s₀ and s_L within each skeleton, so that operator args and atom args stay coherent.

### 4.7 Inputs and outputs

- **Input:** `SkeletonInput` (canonicalized, possibly augmented).
- **Output:** `e(s) ∈ ℝ^64`.
- **Param count:** ~120k (operator sub-encoder ~30k, atom Set-Transformer pool ~40k, sequence transformer ~50k).

---

## 5. Context encoder Ψ

### 5.1 Architecture

Input: a set of failed-skeleton embeddings `{e(sᵢ) : sᵢ ∈ F_t}`, variable-size, possibly empty.

```
if |F_t| == 0:
    c_t ← c_0                                  # learned 64-dim parameter
else:
    tokens ← stack[e(s) : s ∈ F_t]             # (|F|, 64)
    tokens ← LayerNorm(tokens)
    tokens ← SAB(tokens, mask=F_mask)          # (|F|, 64); 4 heads
    tokens ← SAB(tokens, mask=F_mask)          # (|F|, 64)
    c_t    ← PMA_{k=1}(tokens, mask=F_mask)    # (1, 64)
    c_t    ← Linear(64 → 64)(c_t).squeeze(0)   # (64,)
```

`c_0 ∈ ℝ^64` is a learned parameter, initialized to zero, trained jointly with the rest of the model.

### 5.2 Auxiliary features ψᵢ — not used in v1

The v0 spec listed optional per-failure features (stuck-step index, sampler retries, refiner wall-clock). These remain deferred. RT2D's `OutcomeRecord.refiner_metadata` does include `stuck_cause`, `stuck_op_name`, `stuck_step_index`, but using them would inject an additional signal that B4 cannot match for trivial reasons; we want the SPECTRE-vs-B4 gap to be attributable to skeleton structure + tag access, not to refiner introspection. Defer to a follow-up ablation if M8 leaves a residual gap.

### 5.3 Inputs and outputs

- **Input:** variable-size set of vectors in ℝ^64 (possibly empty).
- **Output:** `c_t ∈ ℝ^64`.
- **Param count:** ~40k.

---

## 6. Scorer σ

### 6.1 Architecture

For each remaining candidate `s ∈ R_t`:

```
π_proj   ← Linear(1 → 8)(π(s))                         # (8,)
x        ← Concat[e(s); c_t; π_proj]                    # (136,)
h        ← LayerNorm(Linear(136 → 128)(x)).gelu()       # (128,)
h        ← Dropout(p=0.1)(h)
h        ← LayerNorm(Linear(128 → 64)(h)).gelu()        # (64,)
σ(s)     ← Linear(64 → 1)(h).squeeze(-1)                # scalar
```

### 6.2 Prior dropout

During training, with probability `p_drop = 0.2` (independently per training example, _not_ per skeleton), `π(s)` is replaced with 0 for all `s` in that example. This forces e(s) and c_t to carry meaningful signal and makes context-collapse-to-prior detectable.

### 6.3 Initialization toward the prior

The final `Linear(64 → 1)` weights are initialized to zero, and the final-layer bias is initialized to zero. The `Linear(1 → 8)` weights for `π_proj` are initialized to a small constant `α = 0.1` along the diagonal (plus zero bias). This means an untrained σ outputs a small linear function of `π(s)` — close to behaving like the static ranker.

For RT2D with `π ≡ 0`, this initialization gives σ(s) ≈ 0 for every s at training start, and ranking is uniform random until learning takes over. This is acceptable; the warm-start matters more when a non-zero prior is plugged in.

### 6.4 Inputs and outputs

- **Input:** `e(s) ∈ ℝ^64`, `c_t ∈ ℝ^64`, `π(s) ∈ ℝ`.
- **Output:** `σ(s) ∈ ℝ`.
- **Param count:** ~25k.

---

## 7. Plug-in prior π

For the headline RT2D experiment, use `π(s) ≡ 0` (`ZeroPrior` in the as-built pipeline). The architecture supports any plug-in scalar prior; if a mode-marginal prior is added later (e.g. a static ranker that scores each skeleton by `log Σ_z π(z) · I[skeleton survives mode z]`), it slots in unchanged. Prior-dropout (§6.2) and the init-toward-prior rule (§6.3) apply identically regardless of the prior choice.

---

## 8. Training procedure

### 8.1 Data and dataset length

Training data is produced by `SpectreDataset` (as-built) over the 500-problem RT2D-n3-v1 train split. Each `__getitem__` call returns a `SpectreTrainingExample` containing `(R_skeletons, F_skeletons, R_success_mask, R_priors, …)` after F-subset sampling and (training-only) augmentation. The collate function `collate_spectre_batch` returns a `SpectreBatch` with the tensor shapes documented in §4.1.

**Dataset length is multiplied by F-samples per epoch.** A given training episode admits up to `2^|FAIL_e|` distinct `(R, F)` training examples (one per choice of `F ⊆ FAIL_e`). For RT2D with typical `|FAIL_e| ∈ [5, 25]`, this is 32 to ~33 million distinct examples per episode. The as-built default of one F-sample per `__getitem__` call, combined with `__len__ = num_episodes`, would visit only `num_epochs ≈ 20` of these per problem across the entire training run — severe undersampling of the F-subset distribution and of its cross-product with object-renumbering augmentation.

**Fix #5** (see §9.5): expose `num_f_samples_per_epoch` as a `SpectreDataset` constructor argument, default `8`. This makes:

```python
def __len__(self):
    return self._num_trainable_episodes * self._num_f_samples_per_epoch

def __getitem__(self, i):
    episode_idx  = i // self._num_f_samples_per_epoch
    f_sample_idx = i  % self._num_f_samples_per_epoch
    episode = self._load(episode_idx)
    rng = np.random.default_rng(seed=(self._seed, episode_idx, f_sample_idx, self._epoch))
    F = self._sample_f_subset(episode.fail_indices, rng)
    aug = self._sample_augmentation(episode, rng) if self._augment else None
    return self._build_example(episode, F, aug)
```

With `num_f_samples_per_epoch = 8` and 20 epochs, each problem now yields 160 distinct training examples across the run — an 8× increase in F-subset coverage and in the number of (augmentation, F) pairs the model sees per problem. Total epoch length increases 8× to ~4000 examples per epoch (~250 gradient steps at `B = 16`). Wall-clock per epoch increases roughly 8× as well, so the 20-epoch budget translates to a fixed total compute roughly 8× larger than the as-built baseline.

**Φ-compute sharing within a batch (optimization, not contract).** Since `e(s) = Φ(s)` does not depend on `F`, multiple F-samples drawn from the same episode produce identical `e(s)` values for every `s ∈ S`. A batch sampler that groups F-samples from the same episode into a single batch can compute `Φ` over `S` once and re-use the embeddings across all F-samples in the group. This is a 4–8× speedup on the dominant Φ cost. Recommended implementation: a custom `BatchSampler` that yields chunks of indices `{i, i+1, …, i+G-1}` such that each chunk consists of F-samples from a single episode, with `G ∈ {4, 8}`. The collate function detects the "shared episode" case via the `problem_id` field and computes `Φ` once per group internally. If implementation complexity is a concern, skip this optimization and let `Φ` be recomputed per F-sample — correctness is unaffected; only wall-clock time differs.

The interaction with the augmentation policy (§4.6) is intentional: each F-sample within an episode can independently sample its own augmentation permutation (when `augmentable=True`), giving us the full cross product of augmentations × F-samples per problem per epoch. This is what makes the 8× length multiplier useful — same-augmentation, different-F samples are already useful, but different-augmentation × different-F samples are strictly more so.

### 8.2 F-subset sampling — match training |F| to test-time visit distribution

This is **fix #4** (see §9.4 for rationale). The framing: the training-time distribution over `|F|` should match the test-time _visit_ distribution as closely as possible. At test time, every episode visits `|F| = 0`, most also visit `|F| = 1`, fewer visit `|F| = 2`, and so on — a geometric-decay-shaped distribution. The as-built default (`uniform_subsets`, Bernoulli(0.5) per failure index) puts most mass at `|F| ≈ |FAIL_e|/2` and almost none at `|F| ∈ {0, 1, 2}`, which is the inverse of the test-time mass.

`SpectreDataset` accepts an `f_sampling` argument with four modes; default is `"rollout_aligned_mix"`.

**Mode 1 — `"uniform_subsets"`** (as-built original; retained for ablation):

```python
keep = rng.random(n) < 0.5
return frozenset(i for i, k in zip(fail_indices, keep) if k)
```

**Mode 2 — `"uniform_size"`** (full coverage of all sizes):

```python
size = rng.integers(0, n + 1)             # uniform over {0, 1, ..., n}
return frozenset(rng.choice(fail_indices, size=size, replace=False))
```

**Mode 3 — `"log_normal"`** (small-|F| biased, matches test-time shape):

```python
size = int(round(rng.lognormal(mean=mu, sigma=sigma)))   # mu=0.0, sigma=1.0 default
size = max(0, min(size, n))                              # clip to [0, n]
return frozenset(rng.choice(fail_indices, size=size, replace=False))
```

With default `mu=0`, `sigma=1`, the marginal distribution over `|F|` (before clipping) is approximately `P(|F|=0) ≈ 0.24`, `P(|F|=1) ≈ 0.40`, `P(|F|=2) ≈ 0.16`, `P(|F|=3) ≈ 0.07`, decaying tail. This mirrors the geometric-decay shape of test-time visit frequencies far better than either of the first two modes.

**Mode 4 — `"rollout_aligned_mix"`** (default): a 3-way mixture with weights `(p_subsets, p_size, p_lognormal) = (0.25, 0.25, 0.5)` on `uniform_subsets`, `uniform_size`, and `log_normal` respectively. The weighting reflects an explicit asymmetry: log-normal carries half the mass because it is the only component whose shape matches the test-time visit distribution, while the two uniform components together carry the other half to (a) ensure full coverage of all size classes from `uniform_size` and (b) preserve the gradient regime the original SPECTRE spec was calibrated against from `uniform_subsets`.

The exact weighting is a tunable hyperparameter, not a fixed contract:

```python
@dataclass(frozen=True)
class FSamplingConfig:
    mode: str = "rollout_aligned_mix"
    mix_weights: tuple[float, float, float] = (0.25, 0.25, 0.5)   # subsets, size, log_normal
    log_normal_mu: float = 0.0
    log_normal_sigma: float = 1.0
```

Empirical effective marginal under the default mix, for a representative `|FAIL_e| = 15`:

| `|F|` | mix probability | |---|---| | 0 | ~0.14 | | 1 | ~0.21 | | 2 | ~0.10 | | 3 | ~0.07 | | 4–6 | ~0.04 each | | 7–15 | ~0.05 each (tail) |

So ~52% mass on `|F| ∈ {0,1,2,3}` and ~25% on the long tail `|F| ≥ 7` — materially better matched to test-time rollout shape than 1/3-1/3-1/3 would be (~38% / ~30%), without abandoning long-tail coverage entirely.

```python
def sample_f_subset(fail_indices, rng, config: FSamplingConfig):
    n = len(fail_indices)
    if n == 0:
        return frozenset()
    if config.mode == "rollout_aligned_mix":
        sub_mode = rng.choice(
            ["uniform_subsets", "uniform_size", "log_normal"],
            p=config.mix_weights,
        )
    else:
        sub_mode = config.mode
    if sub_mode == "uniform_subsets":
        keep = rng.random(n) < 0.5
        return frozenset(i for i, k in zip(fail_indices, keep) if k)
    if sub_mode == "uniform_size":
        size = int(rng.integers(0, n + 1))
    elif sub_mode == "log_normal":
        size = int(round(rng.lognormal(
            mean=config.log_normal_mu,
            sigma=config.log_normal_sigma,
        )))
        size = max(0, min(size, n))
    return frozenset(rng.choice(fail_indices, size=size, replace=False).tolist())
```

The default `(0.25, 0.25, 0.5)` weighting and `(μ=0, σ=1)` shape are reasonable for `|FAIL_e| ∈ [5, 25]` (the typical RT2D range). If empirics show the model is underperforming specifically at small `|F|` regimes during validation, the recommended tuning order is:

1. **First**, drop `log_normal_mu` to `-0.3` or `-0.5` (shifts the log-normal mode from `|F|=1` toward `|F|=0` while keeping the mixture shape interpretable).
2. **Only if that is insufficient**, raise the log-normal mix weight to `0.6` (and split the remainder evenly between the two uniforms).

The reverse direction (model underperforming at long-rollout `|F| ≥ 5`) calls for the opposite: raise `log_normal_mu` toward `0.5–1.0`, or shift mix weight toward `uniform_subsets`.

**Empirical handle on whether this works.** AUROC(0), AUROC(1), AUROC(2), AUROC(3) tracked separately during validation (§8.6) tell you directly whether the small-`|F|` regime is being learned. If AUROC(0) or AUROC(1) are flat across epochs while AUROC(3) climbs, the mixture is over-weighting late stages and `mu` should drop (or the log-normal weight in the mix should rise).

### 8.3 Loss — uniform Plackett-Luce

For each training example with remaining set `R`, succeeded-in-R set `SUCC_R = {s ∈ R : s succeeded in the original episode}`, and failed-in-context set `F`:

```
logits   ← {σ(s ; e(s), c(F), π(s))    : s ∈ R}
Z        ← Σ_{s ∈ R}      exp(logits[s])
Z_plus   ← Σ_{s ∈ SUCC_R} exp(logits[s])
ℒ        ← −log(Z_plus / Z)
```

Computed in log-space with `torch.logsumexp` for numerical stability. The batch loss is the mean over `B` examples.

This is the standard top-1-via-PL formulation: `ℒ = −log P(argmax picks any success)`. It is rollout-aligned for the time-to-first-success metric.

Examples with `|SUCC_R| = 0` are filtered out by the dataset (per as-built `SpectreDataset.__init__`); examples with `|R| ≤ 1` are also filtered.

### 8.4 Optimizer and schedule

- Optimizer: `AdamW`, `lr = 3e-4`, `weight_decay = 1e-4`, `betas = (0.9, 0.999)`.
- Batch size: `B = 16` training examples (after F-subset sampling).
- Epoch length with `num_f_samples_per_epoch = 8`: ~500 problems × 8 ≈ 4000 examples per epoch ≈ 250 gradient steps at `B = 16`.
- Total training: 20 epochs (10× the as-built spec's per-problem F coverage, measured in (epoch × num_f_samples_per_epoch)). Empirical overfitting onset is usually 10–15 epochs; checkpoint best-by-val-loss.
- LR schedule: cosine decay to `1e-5` over the full 20-epoch budget (~5000 steps), with 500-step linear warmup at the start.
- Gradient clipping: global norm 1.0.

### 8.5 Regularization and dropout

- Dropout 0.1 on all multihead-attention outputs, all FFN intermediates, and the scorer's hidden layers.
- LayerNorm everywhere standard (post-attention, post-FFN, pre-mlp, pre-Ψ-input, pre-σ-input).
- Prior dropout 0.2 (§6.2).
- Augmentation per §4.6 at every `__getitem__` during training; off at val/test.

### 8.6 Validation and checkpointing

- Validation dataset: same `SpectreDataset` over the val split, with `augment=False` and `f_sampling="rollout_aligned_mix"` (deterministic seed), 4 F-subsamples per val episode for stable loss estimates.
- Validation metrics, every epoch:
    - Mean PL loss.
    - **AUROC(t)** for `t ∈ {0, 1, 2, 3}`: AUROC of σ on labels `success-in-R` restricted to examples with `|F| = t`. Track all four; rising AUROC(t) with `t` is the diagnostic that confirms Ψ is using the failure context.
    - Top-1 hit rate at `t = 0` (proxy for step-1 ranker quality) and at `t = 1, 2, 3`.
- Checkpoint criterion: save the epoch with lowest val PL loss. Tie-break on AUROC(3) (later-step performance is the harder signal).

### 8.7 Multi-seed protocol

Train ≥ 3 independent seeds (default 3, expand to 5 if seed std exceeds half the gap between SPECTRE and the strongest baseline). Each seed independently re-initializes weights, F-sampling RNG, and augmentation RNG. Reported numbers are mean ± std across seeds.

---

## 9. Fixes from the original SPECTRE method spec — rationale

This section documents the four deliberate deviations from `SPECTRE_METHOD_SPEC.md` and why each is necessary for RT2D.

### 9.1 Fix 1 — Atom pooling: Set Transformer, not Deep Sets (§4.3)

**Original spec:** §4.1.3 specifies Deep Sets (mean of per-atom MLP outputs) for atom pooling inside Φ_s, with §8.6 deferring an upgrade to SAB+PMA "if M4 validation fails."

**Why RT2D forces this now:** The RT2D environment is engineered so the _only_ way to outperform B4 (the canonical-key adaptive baseline) is by binding `PassageWidth(p, w)` and `ItemSize(i, s)` static atoms in s₀ to the operator-sequence's `TraverseLoadedColor⟨X⟩(robot, p, …, i)` arguments. Recognizing that a specific `(p, i)` pair is at-risk requires a relational join across atoms with shared arguments. Deep Sets cannot do this: it pools each atom independently through an MLP and then averages, losing the ability to represent argument-level coincidences across atoms.

A Set Transformer (SAB) lets each atom token attend to every other atom token, so the conjunction "this passage's width atom mentions the same id as this item's size atom" is representable in a single layer. PMA pools the result to a single state-token vector.

**Cost:** one SAB block + one PMA seed vector (~40k params). Negligible inference cost — atom counts are O(30) per state, so attention is O(1000) flops per state, dwarfed by the sequence Transformer.

### 9.2 Fix 2 — Per-type augmentable / non-augmentable distinction (§4.6)

**Original spec:** §4.1.4 specifies "random within-type permutations of local-ids as data augmentation" applied to every type uniformly.

**Why RT2D forces this:** RT2D introduces three types whose instances are named constants with semantic identity:

- `WidthLevel` ∈ {narrow, medium, wide} — totally ordered (narrow < medium < wide).
- `SizeLevel` ∈ {small, medium, large} — totally ordered.
- `GraspMode` ∈ {top, side} — semantically distinct (different refinement gates).

The size-vs-width compatibility check (gate 2 in the refiner: succeeds iff `size(item) ≤ width(passage)`) is the ground truth the model must learn, and that gate is _intrinsically ordered_. If the augmentation step randomly permutes the local-ids of `WidthLevel` and `SizeLevel` per training example, the embedding lookup for "narrow" varies arbitrarily across examples, and the model has no stable signal that `WidthLevel:0` corresponds to "the smallest." The same argument applies to `GraspMode`: in one example `GraspMode:0` means "top," in the next it means "side."

Random within-type permutation of these types **destroys the exact signal RT2D is engineered to test for**.

`Zone` is also marked non-augmentable. In RT2D the 6 zones form a fixed K₃,₃ topology; permuting zone ids would relabel `Connects` atoms in a way that, while topologically equivalent, requires the encoder to fully internalize K₃,₃ symmetry to generalize. This is a heavier ask than necessary; freezing zone ids is conservative and removes the variable.

`Passage` and its color subtypes are non-augmentable for the analogous reason — the static `Connects(passage, zone, zone)` topology is fixed across all problems, so within-type permutation gains nothing and risks breaking the topology signal.

**Backwards compatibility:** for the kinder envs (Obstruction2D, etc.), all instance types remain `augmentable=True`. The fix is additive; nothing about the original SPECTRE behavior on those envs changes.

### 9.3 Fix 3 — Vocab-driven dynamic sizing of operator and atom MLPs (§4.2, §4.3)

**Original spec:** §4.1.3 hardcodes operator-token MLP shape `96 → 128 → 64`, predicated on the kinder envs' max operator arity 3 and an implicit predicate arity of 2.

**Why RT2D forces this:** RT2D bumps `max_operator_arity = 5` (`TraverseLoadedColor⟨X⟩` takes `robot, passage, src, dst, item`) and `max_predicate_arity = 3` (`Connects(passage, zone, zone)`). The MLPs must size their input dim from `vocab.max_operator_arity` and `vocab.max_predicate_arity` at model construction time, not from a hardcoded constant.

Concretely:

- Operator-token MLP input dim: `32 + A*16 + 16` where `A = vocab.max_operator_arity`.
- Atom-token Linear input dim: `32 + P*24` where `P = vocab.max_predicate_arity`.

This is a code-shape fix, not a model-design fix; but it must be in place before training, or the model will silently fail on RT2D with arity-out-of- range index errors.

### 9.4 Fix 4 — F-subset sampling: rollout-aligned mix for small-|F| coverage (§8.2)

**Original spec:** §5.2 specifies "sample F uniformly over subsets of FAIL_e," implemented in the as-built pipeline as Bernoulli(0.5) per failure index.

**Why RT2D — and the test-time protocol — force this:** the test-time sparse rollout visits every episode at `|F| = 0` first, most episodes at `|F| = 1`, fewer at `|F| = 2`, and so on — a geometric-decay-shaped visit distribution. Under Bernoulli(0.5) sampling, with `|FAIL_e|` typically in 5–25 for RT2D, the training distribution puts most mass at `|F| ≈ |FAIL_e|/2` and almost none at the small-`|F|` regimes that dominate test-time visits. This is exactly the wrong shape for time-to-first-success optimization.

The right framing is **match the training |F| distribution to the test-time visit distribution**. Three sampling shapes are useful:

- **Uniform-over-subsets** (the original) is heavily peaked at the middle.
- **Uniform-over-size** is flat — covers all sizes equally but doesn't match the test-time decay.
- **Log-normal-over-size** approximates the test-time decay shape directly: `round(LogNormal(0, 1))` puts roughly 24% mass at `|F|=0`, 40% at `|F|=1`, 16% at `|F|=2`, and a decaying tail.

The default `"rollout_aligned_mix"` puts weights `(0.25, 0.25, 0.5)` on `(uniform_subsets, uniform_size, log_normal)` respectively — log_normal in the lead because it is the only component whose shape matches the test-time visit distribution, with the two uniform components splitting the rest to preserve coverage and gradient regimes the model has historically trained well on. The asymmetric weighting is intentional: an equal 1/3-1/3-1/3 mix would put only ~38% mass on `|F| ∈ {0,1,2,3}` for a typical `|FAIL_e| = 15`, while the 0.25/0.25/0.5 weighting puts ~52% there — materially better matched to test-time visit shape, which is heavy at small `|F|`.

We do not push log-normal weight higher than 0.5 by default for three reasons: (1) the per-example _information content_ is highest at medium `|F|` (at `|F| = 0` the loss only depends on Φ + π since `c_t = c_0`; at small `|F|` the context signal is thin), (2) test-time long-rollout robustness requires some training density at `|F| ≥ 5`, since SPECTRE will encounter those regimes when it runs into a hard problem, and (3) hyperparameter conservatism — both the mix weights and the log-normal `μ` parameter affect the small-`|F|` mass, and shifting both in the same direction compounds two unverified guesses.

Pure log-normal is available as a separate mode for ablation if the mix turns out to overweight middle-`|F|` cases empirically. AUROC(t) tracked per t during validation (§8.6) is the diagnostic for whether the mix is right; the recommended tuning order if small-`|F|` AUROC lags is in §8.2.

### 9.5 Fix 5 — F-subsample multiplier exploits per-episode example richness (§8.1)

**Original spec:** Implicit in `SpectreDataset.__len__ = num_episodes` — one F-sample per episode per epoch.

**Why this leaves signal on the table:** Each training episode admits up to `2^|FAIL_e|` distinct `(R, F)` training examples. For RT2D problems with `|FAIL_e| ∈ [5, 25]`, this is 32 to 33 million examples per episode. F-subset sampling is itself a form of data augmentation — it expands one episode into a combinatorially large family of training examples that share the same `Φ`-encoder targets but condition `Ψ` and `σ` on different failure histories.

Sampling one F per episode per epoch and training for 20 epochs visits only ~20 of these per problem. Combined with object-renumbering augmentation (§4.6), the (aug, F) cross-product is similarly undersampled. This is qualitatively different from the kinder envs the original SPECTRE spec was written against — in those envs, the failure modes are simpler and the gradient signal per F-sample is less differentiated, so under-sampling hurts less. RT2D's mode-conditional family structure means each F-sample genuinely teaches the model something new about the per-problem failure pattern.

**The multiplier.** `num_f_samples_per_epoch = 8` (default) gives 160 distinct training examples per problem across the 20-epoch budget — still a small fraction of `2^|FAIL_e|`, but an 8× improvement on the original plan. Larger multipliers (16, 32) are reasonable if compute allows; the diminishing return point is when AUROC(t) at all t stabilizes by mid-training. Smaller multipliers (1–4) are reasonable if compute is tight and the rollout-aligned mix is enough to span the F-distribution.

**The Φ-compute-sharing optimization** is what makes this cheap: encoder forwards are the dominant per-batch cost, and they are identical across all F-samples drawn from the same episode. Grouping F-samples from a single episode into a batch chunk lets us amortize Φ compute across the chunk. With `num_f_samples_per_epoch = 8` and chunked batching, the wall-clock cost per epoch is roughly 1.5–2× the original (not 8×), since Φ is the dominant cost.

**Why default 8 and not higher?** Two reasons. First, gradient correlation within a batch — multiple F-samples from the same episode are not independent, so the effective batch size for SGD purposes is smaller than the nominal `B = 16`. Beyond ~8 samples per episode in a batch, the correlation overwhelms the diversity gain. Second, the mixture shape already gives reasonable coverage of size classes per problem; the multiplier is for filling in the _content_ of each size class, not for broadening it.

---

## 10. Implementation contracts

### 10.1 Vocab schema additions

The vocab JSON gains one field beyond what the as-built pipeline emits:

```json
{
  ...
  "type_aug_policy": {
    "Robot":         { "augmentable": true  },
    "Item":          { "augmentable": true  },
    "Zone":          { "augmentable": false },
    "Passage":       { "augmentable": false },
    "PassageColorA": { "augmentable": false },
    "PassageColorB": { "augmentable": false },
    "PassageColorC": { "augmentable": false },
    "WidthLevel":    { "augmentable": false },
    "SizeLevel":     { "augmentable": false },
    "GraspMode":     { "augmentable": false }
  }
}
```

Defaults to `augmentable=true` for any type missing from the policy (backwards-compatible with the kinder envs). The policy is sourced from a per-env table in `env_registry.py`.

### 10.2 Tensor shapes for `SpectreBatch`

|Tensor|Shape|Notes|
|---|---|---|
|`r_skeleton.op_name_ids`|`(B, R, L)`|R = max remaining-pool size in batch|
|`r_skeleton.op_arg_type_ids`|`(B, R, L, A)`|A = vocab.max_operator_arity|
|`r_skeleton.op_arg_local_ids`|`(B, R, L, A)`||
|`r_skeleton.s0_atom_pred_ids`|`(B, R, M0)`|M0 = max s_0 atom count in batch|
|`r_skeleton.s0_atom_arg_type_ids`|`(B, R, M0, P)`|P = vocab.max_predicate_arity|
|`r_skeleton.s0_atom_arg_local_ids`|`(B, R, M0, P)`||
|`r_skeleton.s0_type_histogram`|`(B, R, T)`|T = num types|
|`r_skeleton.sL_*`|analogous to s0_*||
|`r_priors`|`(B, R)`|float|
|`r_success_mask`|`(B, R)`|bool, used in loss|
|`r_pool_mask`|`(B, R)`|bool, true for valid pool slots|
|`f_skeleton.*`|`(B, F, …)`|F = max failure-set size in batch|
|`f_pool_mask`|`(B, F)`|bool|

Per spec §3.7 of `SPECTRE_TRAINING_PIPELINE_AS_BUILT.md`, the s₀ replication "per-skeleton" is done at collate time even though the dataset stores it once.

### 10.3 Forward-pass interface

```python
class SpectreModel(nn.Module):
    def __init__(self, vocab: Vocab, type_aug_policy: dict[str, TypeAugPolicy]): ...

    def encode_skeletons(
        self,
        skeleton_input: SkeletonInputBatch,  # (B, K, ...) tensors
    ) -> Tensor:                              # (B, K, 64)
        """Φ. Batched over both batch dim B and pool dim K."""

    def encode_context(
        self,
        f_embeddings: Tensor,                 # (B, F, 64)
        f_mask: Tensor,                       # (B, F) bool
    ) -> Tensor:                              # (B, 64)
        """Ψ. Returns c_0 broadcast for examples where f_mask is all-False."""

    def score(
        self,
        r_embeddings: Tensor,                 # (B, R, 64)
        c: Tensor,                            # (B, 64) or (B, 1, 64) broadcastable
        r_priors: Tensor,                     # (B, R)
        prior_dropout: bool = False,
    ) -> Tensor:                              # (B, R)
        """σ. Optionally applies prior dropout."""

    def forward(self, batch: SpectreBatch) -> Tensor:  # (B, R) logits
        e_R = self.encode_skeletons(batch.r_skeleton_input)
        e_F = self.encode_skeletons(batch.f_skeleton_input)
        c   = self.encode_context(e_F, batch.f_pool_mask)
        return self.score(e_R, c, batch.r_priors, prior_dropout=self.training)
```

### 10.4 Loss implementation

```python
def plackett_luce_loss(
    logits: Tensor,           # (B, R)
    success_mask: Tensor,     # (B, R) bool
    pool_mask: Tensor,        # (B, R) bool
) -> Tensor:                  # scalar
    masked_logits = logits.masked_fill(~pool_mask, -float("inf"))
    Z      = torch.logsumexp(masked_logits,                            dim=-1)
    Z_plus = torch.logsumexp(masked_logits.masked_fill(~success_mask, -float("inf")), dim=-1)
    return -(Z_plus - Z).mean()
```

### 10.5 Test-time inference

```python
@dataclass
class InferenceState:
    e_S: Tensor                        # (K, 64), computed once at episode start
    priors: Tensor                     # (K,)
    pool_mask: BoolTensor              # (K,) — false where attempted
    fail_indices: list[int]

def select_next_skeleton(state, model) -> int:
    # Encode failure context
    if state.fail_indices:
        f_emb = state.e_S[state.fail_indices].unsqueeze(0)             # (1, |F|, 64)
        f_mask = torch.ones(1, len(state.fail_indices), dtype=bool)
        c = model.encode_context(f_emb, f_mask)                         # (1, 64)
    else:
        c = model.empty_context.unsqueeze(0)                            # (1, 64)
    # Score remaining pool
    e_R = state.e_S.unsqueeze(0)                                        # (1, K, 64)
    p_R = state.priors.unsqueeze(0)                                     # (1, K)
    logits = model.score(e_R, c, p_R, prior_dropout=False)              # (1, K)
    logits = logits.masked_fill(~state.pool_mask.unsqueeze(0), -float("inf"))
    return int(logits.argmax(dim=-1).item())
```

The episode-start cost is one batched Φ forward pass over K skeletons. Each subsequent step is one Ψ forward (over a set of size t-1) plus a broadcasted σ over K logits — both negligible for K ≤ 30 and t ≤ 30.

---

## 11. Smoke tests and acceptance criteria

### 11.1 Pre-training smoke tests

1. **Forward shape check.** Construct a `SpectreBatch` from 2 random training examples. Confirm `model(batch)` returns shape `(B, R)` finite floats. Confirm gradient flows to all parameters via `torch.autograd.grad`.
    
2. **Empty-F path.** Construct a batch where all `f_pool_mask` rows are all-False. Confirm `encode_context` returns `c_0` broadcast — i.e. `c[i] == model.c_0` for all i.
    
3. **Augmentation invariance under non-augmentable types.** Permute `WidthLevel` local ids by hand in a `SkeletonInput`. Confirm the training Dataset's augmentation does **not** further permute them (i.e. WidthLevel local ids in the augmented output match what was passed in).
    
4. **Augmentation equivariance under augmentable types.** Two skeletons that differ only by `Item` renumbering must produce e(s) values that are equal up to a tolerance of 1e-5 (after the model is in eval mode and augmentation is off).
    
5. **Vocab arity sourcing.** Construct a vocab with `max_operator_arity = 7` and `max_predicate_arity = 4`. Build a model. Confirm the operator-token MLP and atom-token Linear have input dimensions matching the formulas in §4.2 and §4.3.
    
6. **PL loss behavior.** With logits `[1e6, 0, 0]` and success_mask `[True, False, False]`, loss → 0. With logits `[0, 1e6, 0]` and the same mask, loss → +∞ (numerically: `~1e6`). With uniform logits, loss = `log(K / |SUCC_R|)`.
    

### 11.2 During-training acceptance bar

After one epoch on the train split, **AUROC(0) > 0.55** is the minimum bar (otherwise Φ is broken). After 5 epochs, **AUROC(3) > AUROC(0) by at least 0.05** is the minimum bar (otherwise Ψ is not contributing — context collapse). Both conditions checked in the validation log; tripping either fails the training run.

### 11.3 End-to-end acceptance bar

On the held-out test split, averaged over ≥ 3 training seeds:

- **Mean time-to-first-success** beats B3 (static heuristic) by ≥ 1 attempt. This is the kinder bar.
- **Mean time-to-first-success** beats B4 (Naive-Bayes log-odds adaptive) by ≥ 0.3 attempts. This is the headline bar that justifies the RT2D benchmark design.
- **Step-1 success rate** (probability of picking a successful skeleton at `|F| = 0`) is no worse than B2 (mode-marginal greedy). This guards against losing the static-prior signal entirely.

Failing the headline bar is grounds to revisit fixes 1 and 4 (atom pooling upgrade may need a second SAB layer; F-subset sampling may need further rebalancing).

---

## 12. Open questions deliberately not specified

These are not blockers for implementation but should be revisited if the acceptance bar is missed:

- **Φ_s atom pooling depth.** §4.3 specifies one SAB + PMA. If RT2D acceptance is missed, try two SABs (closer to the original Set Transformer paper).
- **Ψ depth.** §5.1 specifies two SABs + PMA. One SAB may be sufficient given `|F| ≤ 30`; ablate if the param budget needs trimming.
- **Loss variant.** §8.3 specifies uniform PL. Cost-weighted PL (weight pairs by refinement wall-clock) is a simple alternative if the secondary wall-clock metric matters.
- **DAgger correction.** Not implemented. If acceptance shows a test-time gap that offline AUROC(t) does not predict, train one round of DAgger on F sets the trained SPECTRE actually produces.

---

_End of specification. Companion documents: `SPECTRE_METHOD_SPEC.md` (original method spec, partially superseded); `ROUTED_TRANSPORT2D_SPEC.md` (environment); `SPECTRE_TRAINING_PIPELINE_AS_BUILT.md` (data pipeline, unchanged except for the per-type augmentation policy in §10.1)._