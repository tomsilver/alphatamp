"""LAZY — policy-guided lazy search with feedback (Khodeir et al), re-implemented over
SPECTRE's fixed candidate-skeleton pool as an adaptive baseline.

LAZY's reference implementation (``baselines/drake-tamp/``, ``lifted_merged`` branch) is
~80% blocks-world / PDDLStream-specific: its native feasibility runs over a computation
graph SPECTRE has no analog for, and it does integrated incremental search rather than
re-ranking a fixed pool. Every method in this comparison instead re-ranks a fixed pool of
pre-enumerated skeletons with pre-computed refinement outcomes. So LAZY is re-implemented
here; only its GATv2 policy, behaviour-cloning cross-entropy, and π̄=π·ϕ/Σ reweighting
math carry over. Deviations from literal LAZY are enumerated in ``PROVENANCE.md``.

The method has three pillars, one module each:

- **prefix-tree policy** (``tree`` + ``graph`` + ``model``): a GAT policy over a prefix
  tree of the pool. At each tree node (a shared operator prefix) the policy scores the
  candidate next-operators; π(op|node)=softmax; π(skeleton)=∏ per-action probs.
- **feasibility statistics ϕ** (``feasibility``): ϕ=(succ+1)/(att+1) keyed on shared
  operator substructure, updated online as skeletons fail.
- **online reweighting** (``rollout``): π̄(op|n)=π·ϕ/Σ, LevinTS path-product priority,
  producing the realized attempt order and time-to-first-success.
"""
