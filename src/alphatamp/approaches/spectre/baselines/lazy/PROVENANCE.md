# LAZY baseline — provenance and deviations from the paper

This package re-implements **LAZY** (Khodeir, Sonwane, Hari, Shkurti, *Policy-Guided Lazy
Search with Feedback for Task and Motion Planning*; reference code in
`baselines/drake-tamp/`, `lifted_merged` branch) as an adaptive baseline over SPECTRE's
fixed candidate-skeleton pool. The reference code is ~80% blocks-world / PDDLStream
specific and cannot be reused directly (its feasibility runs over a computation graph
SPECTRE has no analog for, and it does integrated incremental search). Only the GATv2
policy, the behaviour-cloning cross-entropy, and the `π̄ = π·ϕ/Σ` reweighting carry over.

This file enumerates every deviation from the paper, in the spirit of
`vlmplan/prompts/PROVENANCE.md`, so the appendix can state them precisely.

## What is faithful

- **GATv2 policy.** The object-relation encoder is the literal `torch_geometric.nn.GATv2Conv`
  (2 message-passing layers), the same layer the reference uses (`learning/policy.py`).
- **Per-action policy over a search structure.** π scores candidate *next-operators* at each
  node of a tree, and `π(skeleton) = Π π(op|node)` along the path — the paper's per-action
  policy and path probability, not a single per-skeleton score.
- **Behaviour cloning.** Cross-entropy over the candidate next-operators at each node with
  the demonstrated next-operator as target (`learning/policy.py::train_model`), one example
  per (feasible-leaf, node) decision.
- **Feasibility feedback.** `ϕ = (succ+1)/(att+1)` keyed on a renaming-invariant per-operator
  key (the analog of `utils.anonymise`), combined online as `π̄ = π·ϕ/Σ π·ϕ` with a
  LevinTS-style `1/path_prob` priority (`learning/policy.py::FeedbackPolicy`), updated as
  skeletons fail refinement (`a_star.py::repeated_a_star`).

## Deviations

1. **Fixed-pool re-ranker, not integrated search.** Every method in this comparison
   re-ranks a fixed pool of pre-enumerated skeletons with pre-computed refinement outcomes.
   LAZY's prefix tree is therefore built over the pool's canonicalized operator sequences
   (a trie), not grown by an online successor generator. Consequence: LAZY cannot propose an
   off-pool skeleton (VLMPlan is the method that does); it is scored on the same footing and
   FP metric as SPECTRE/PIGINet/astar.

2. **Feasibility is per-operator, not per-stream over a computation graph.** SPECTRE has no
   PDDLStream computation graph or ancestral sampler, so ϕ is keyed on the per-operator
   canonical key `(op_name, typed-local args)` and fit from per-skeleton refinement outcomes
   with per-operator failure attribution (`failure_record.records_for_candidate`), rather
   than on anonymised stream-output CG keys with per-stream attempt/success counts.

3. **Attribution on environments with no class-1 channel.** On StickButton2D there is no
   collision-culprit channel (kinder's check returns a bool), so failure attribution uses the
   failing `(schema, step_index)` plus the inferred `dev_blame`, with a whole-skeleton
   suffix-blame fallback when nothing is named. On DD2D the refiner names culprits directly.

4. **Action-selection head.** The reference scores actions with a *third* cross-attention
   `GATv2Conv` from graph nodes to action nodes. Ours scores each candidate with an MLP over
   `[op-schema embedding ‖ mean argument-node GAT embeddings ‖ mean-pooled graph context]`
   (attention-pool + MLP). The object encoder — the GAT policy proper — is the literal
   `GATv2Conv`; only the action head differs, for batching robustness. Because each candidate
   operator's arguments are explicit, the argument-node embeddings already carry "which
   objects", so this is a cross-attention in spirit.

5. **`ϕ` without the reference attempt-adaptive multiplier.** The reference reweights
   `ϕ = (1 + succ·m)/(1 + att·m)` with `m = 10^(1 + att//10)` (`FeedbackPolicy`). We use the
   plain `(succ+1)/(att+1)` (the paper's III-E formula), i.e. `m=1`.

6. **No object-permutation augmentation (v1).** Episodes are canonicalized deterministically
   (`canonicalize_episode(rng=None)`), matching `eda.load_split_episodes`. Within-type
   permutation augmentation is available in the substrate but not used here.

7. **Selection metric.** Checkpoints are selected on val rollout-FP (the SPECTRE project
   arbiter), not on BC validation cross-entropy. The BC cross-entropy has a label-conflict
   floor (multiple feasible plans diverge at the high-fanout root), so it is not a good
   selection signal on its own.

## Environment / install note (2026-08-09)

`torch_geometric` 2.8.0 (core only; no compiled `torch-scatter`/`-sparse`) was added to the
project. It runs on the RTX 5090 (Blackwell sm_120, torch 2.13+cu130) — `GATv2Conv` forward
and backward verified on-device with finite gradients (G0 gate). `scatter(reduce='max')`
falls back to a native implementation and prints one benign UserWarning.
