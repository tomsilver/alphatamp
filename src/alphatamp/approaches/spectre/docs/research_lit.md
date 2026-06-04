# SPECTRE: Related Literature Review

**Topic:** Adaptive skeleton reordering in Task-and-Motion Planning — Set Transformers, listwise ranking, failure-conditioned context encoders

*Generated 2026-04-27 via `/research-lit @SPECTRE_METHOD_SPEC.md`*

---

## Literature Table

| Paper | Venue | Method | Key Result | Relevance to SPECTRE |
|---|---|---|---|---|
| **PIGINet** Yang et al. 2023 [2211.01576] | RSS 2023 | Transformer over skeleton token sequences predicts refinement feasibility; static pre-sort before refinement | 10–80% planning runtime reduction on kitchen TAMP | **Direct predecessor / baseline π.** Same token-sequence skeleton encoding; SPECTRE adds failure-conditioned context and listwise loss |
| **LAZY / Policy-Guided Search** Khodeir et al. 2023 [2210.14055] | — | PDDLStream search queue reordered lazily using geometric motion samples + learned goal-directed policy | Speedups on 7-DOF manipulation; adapts during search | Adaptive skeleton-queue ordering — same problem framing; no context encoder or PL loss |
| **Effort Allocation / Metareasoning** Sung et al. 2024 [2410.05828] | — | Models skeleton-selection as MDP; DP_Rerun algorithm allocates refinement budget across candidates | Near-MCTS quality with negligible overhead | **Closest in spirit to SPECTRE's adaptive loop.** DP approach vs. learned context encoder; no within-episode failure encoding |
| **Learning to Search in TAMP** Khodeir et al. 2023 [2111.13144] | — | GNN heuristic over PDDLStream fact/object expansion ordering | Speedups over LAZY on real 7-DOF tasks | Coarser than skeleton-level ranking; learns a static search heuristic |
| **Geometric TAMP Rank Function** Kim et al. 2021 [2203.04605] | IJRR | Learned rank function for geometric TAMP discrete search + learned sampler for motion level | Efficient bilevel planning with generalization | Earliest explicit learned skeleton ranker in TAMP |
| **Anticipatory TAMP** Dhakal et al. 2024 [2407.13694] | — | Learned future-cost predictor guides plan selection beyond immediate feasibility | Gains in sim + real manipulation | Implicit adaptive skeleton preference, looks ahead rather than back at failures |
| **Neural Feasibility Checking** Xu et al. 2022 [2203.10568] | — | CNN predicts kinematic feasibility of actions, replaces expensive motion checks | Reduced planning time in manipulation domains | Feasibility-prediction framing; image-level not skeleton-level |
| **Integrated TAMP Survey** Garrett et al. 2021 [2010.01083] | AAAI 2021 | Survey of bilevel TAMP architectures: skeleton-then-refine, backtracking | Canonical reference for skeleton-refine loop | Background; defines the problem SPECTRE operates in |
| **Set Transformer** Lee et al. 2019 [1810.00825] | ICML 2019 | SAB (set attention block) + PMA (pooling by multihead attention); permutation-invariant set encoding | O(nm) complexity via inducing points; universal approximator over sets | **Direct architecture for Ψ.** SPECTRE uses SAB×2 + PMA_{k=1} over ℱ_t |
| **Set-Encoder Passage Re-Ranking** Deckers et al. 2024 [2404.06912] | — | Set Transformer applied to cross-encoder re-ranking to eliminate position bias | Consistent gains over standard cross-encoders on TREC | Shows Set Transformer generalizes to listwise ranking tasks beyond the original paper |
| **Plackett-Luce LTR** Xia et al. 2019 [1909.06722] | — | ListMLE: PL probability over permutations as listwise ranking loss | Matches/surpasses pointwise/pairwise baselines on retrieval | **Theoretical basis for SPECTRE's ℒ** |
| **LiPO: Listwise Preference Optimization** Liu et al. 2024 [2402.01878] | — | PL + other LTR objectives applied to LLM alignment over ranked response lists | Outperforms pairwise DPO for policy optimization | Demonstrates listwise > pointwise for argmax-aligned objectives |
| **Rank Not Estimate** Ferber et al. 2023 [2310.19463] | NeurIPS 2023 | Proves A*/GBFS need only rank states correctly, not estimate cost; proposes ranking losses tailored to each | Validated on IPC planning benchmarks | **Theoretical grounding for PL over regression** in planning contexts |
| **Learning to Rank for Planning Heuristics** (2016) [1608.01302] | IJCAI 2016 | Applies LTR to synthesize planning heuristics for classical planning | Earlier LTR + planning connection | Classical-planning predecessor to SPECTRE's approach |
| **PL Partitioned Preference** Ma et al. 2021 [2006.05067] | AISTATS 2021 | PL estimation with only group-level (partition) labels; fast numerical integration | Orders of magnitude faster than Monte Carlo PL | Relevant when SUCC vs. FAIL partition is the only supervision signal |
| **Algorithm Distillation** Laskin et al. 2023 [2210.14215] | ICLR 2023 | Causal transformer over episode history enables in-context RL improvement without gradient updates | Policy improves purely in-context across multi-task | **Analogous to Ψ:** amortizes history into context rather than per-episode re-optimization |
| **Predicate Invention for Bilevel Planning** Silver et al. 2023 [2203.09634] | AAAI 2023 | Hill-climb over predicate grammars; objective aligned with bilevel planning efficiency | Generalizes to more objects, longer horizons | Same bilevel-planning substrate; shows surrogate-objective alignment matters |
| **NSRTs** Chitnis et al. 2021 [2105.14074] | — | Neuro-symbolic relational transition models; symbolic planner outer loop + neural samplers inner loop | Generalizes to 60-action tasks with unseen objects | Same substrate group (Silver/Chitnis/Kaelbling); provides system context |
| **Learning Symbolic Operators for TAMP** Silver et al. 2021 [2103.00589] | — | Bottom-up relational learning discovers PDDL-style operators from trajectory data | Outperforms GNN baselines across three robotic domains | Same substrate group; context on operator representations |
| **Learning When to Quit** Sung et al. 2021 [2103.04374] | — | Meta-reasoning MDP for when to stop anytime motion planning | Generalizes across environment distributions | Related meta-reasoning framing; stopping vs. reordering |
| **GNN Robot Manipulation** Lin et al. 2021 [2102.13177] | — | GNN policy over object graphs; permutation-invariant; generalizes over object counts | Generalizes from 20 demos across 3 manipulation tasks | GNN equivariance as alternative to SPECTRE's typed-local-id canonicalization |
| **Neural-Guided Runtime Prediction** (2022) [2207.14422] | — | GNN predicts planner runtime from relational problem graph for planner selection | Speedups via planner routing | GNN-based skeleton/planner selection closely related to SPECTRE's scoring |
| **Differentiable GPU-Parallelized TAMP** Shen et al. 2024 [2411.11833] | — | GPU-parallel evaluation of thousands of skeleton continuous solutions simultaneously | Sidesteps sequential selection by parallelizing refinement | Alternative to ranking: try all in parallel rather than select best |

---

## Narrative Synthesis

### 1. The Skeleton Ranking Problem and Prior Approaches

The TAMP bilevel architecture (Garrett et al. 2021, [2010.01083]) separates symbolic skeleton generation from continuous parameter refinement. When the symbolic planner produces multiple candidates, which to attempt first becomes a secondary optimization problem. Early approaches relied on planner-internal cost heuristics; Kim et al. (2021, [2203.04605]) introduced the first explicitly *learned* rank function in geometric TAMP. PIGINet (Yang et al. 2023, [2211.01576]) is the current state of the art: a Transformer over the skeleton's (operator, state) token sequence predicts refinement feasibility, cutting planning time 10–80%. Its key limitation — directly motivating SPECTRE — is that PIGINet is a **static** ranker: it scores each skeleton once, in isolation, before any refinement is attempted. It cannot update its beliefs when it learns that skeleton 1 failed.

Khodeir et al.'s LAZY (2023, [2210.14055]) reorders the skeleton queue *lazily* using geometric motion samples as they arrive — the closest existing work to online adaptation. However, it has no learned context encoder and cannot generalize the failure signal across problem instances. Sung et al. (2024, [2410.05828]) formalize budget-aware skeleton selection as a metareasoning MDP (DP_Rerun), achieving near-MCTS quality. This is the closest prior approach in spirit to SPECTRE but frames adaptation via dynamic programming over time-cost estimates rather than a learned failure-history encoder. None of these methods use a permutation-invariant set encoder over the failure set ℱ_t, which is SPECTRE's core mechanism.

### 2. Architecture: Set Transformer and Permutation Invariance

The Set Transformer (Lee et al. 2019, [1810.00825]) is the direct architectural foundation for SPECTRE's context encoder Ψ. Its SAB (Set Attention Block) achieves permutation-equivariant token mixing without positional embeddings, and PMA (Pooling by Multihead Attention) produces a permutation-invariant fixed-size output — exactly the requirements for encoding the unordered failure set ℱ_t. The original paper reduces complexity from O(n²) to O(nm) via ISAB; SPECTRE uses standard SAB because |ℱ_t| ≤ 20 makes quadratic cost trivial. Deckers et al. 2024 ([2404.06912]) show the same architecture applies to listwise re-ranking in IR, eliminating position bias — an analogous goal to SPECTRE's rollout-alignment objective.

For the skeleton encoder Φ, the key design question is equivariance over object identities. GNN-based approaches (Lin et al. 2021, [2102.13177]; Khodeir et al. 2023, [2111.13144]) achieve this via graph structure. SPECTRE instead uses typed-local-id canonicalization, which achieves the same equivariance via a preprocessing step and enables a simpler Transformer encoder — lower engineering cost with comparable expressivity, as the spec argues.

### 3. Loss Function: Why Listwise Plackett-Luce Over Pointwise BCE

The Plackett-Luce listwise loss (Xia et al. 2019, [1909.06722]) defines SPECTRE's training objective. Three papers from different communities converge on the same conclusion:

- **LiPO** (Liu et al. 2024, [2402.01878]) demonstrates for LLM alignment that listwise PL objectives consistently outperform pairwise DPO because PL directly optimizes the argmax-selection distribution.
- **Rank Not Estimate** (Ferber et al. 2023, [2310.19463]) proves formally that in planning search, heuristics need to rank correctly, not estimate absolute cost — precisely the argument for PL over regression or BCE in SPECTRE's context.
- **Ma et al. 2021** ([2006.05067]) show that when supervision is only at the partition level (successes vs. failures, not full orderings), the PL estimator remains tractable — directly applicable to SPECTRE's F-subset sampling where the SUCC/FAIL partition is the only label.

This three-way convergence from information retrieval, classical planning theory, and LTR statistics provides strong justification for SPECTRE's loss choice that Attempt 2 (pointwise BCE) lacked.

### 4. Amortized Adaptation via Context Encoders

SPECTRE's core claim is that a learned module Ψ can amortize episode-level posterior updates over the failure set ℱ_t. Algorithm Distillation (Laskin et al. 2023, [2210.14215]) is the most direct conceptual parallel in the RL literature: a causal Transformer over an agent's episode history enables in-context policy improvement without any gradient update at test time — the same "read the history, produce a better prior" loop. The key difference is that SPECTRE's Ψ is a *set* encoder (permutation-invariant over ℱ_t) rather than a sequence encoder, which is theoretically more correct since the failure order carries no information.

### 5. Gaps and SPECTRE's Unique Contributions

No existing paper found combines: (a) failure-set context encoding via a permutation-invariant set encoder, (b) applied to TAMP skeleton reordering, (c) with a listwise Plackett-Luce loss. The closest combinations are:

- **PIGINet + failure context** → would require adding Ψ, which PIGINet lacks
- **LAZY + PL loss** → would require replacing its geometric heuristic with a trained encoder
- **Algorithm Distillation + TAMP** → would require set-not-sequence encoding and skeleton-specific tokenization

SPECTRE's combination is genuinely novel. The on-policy correction concern (DAgger round, §8.3 of the spec) is a known gap in offline LTR approaches generally — Learning to Search via Retrospective Imitation ([1804.00846]) would be an appropriate citation if an on-policy round is added.

---

## Recommended Citation Clusters

| Claim / Section | Papers |
|---|---|
| Problem setting (bilevel TAMP) | Garrett et al. 2021 [2010.01083] |
| Skeleton ranker baselines | PIGINet [2211.01576], Kim et al. 2021 [2203.04605] |
| Adaptive selection prior work | Sung et al. 2024 [2410.05828], LAZY [2210.14055] |
| Architecture — set encoding | Lee et al. 2019 [1810.00825] |
| Loss — Plackett-Luce | Xia et al. 2019 [1909.06722], Ferber et al. 2023 [2310.19463], Ma et al. 2021 [2006.05067] |
| Amortized context / in-context adaptation | Laskin et al. 2023 [2210.14215] |
| Substrate context | Silver et al. 2021 [2103.00589], Chitnis et al. 2021 [2105.14074], Silver et al. 2023 [2203.09634] |

---

## Full Reference List

- [1608.01302] Learning to Rank for Synthesizing Planning Heuristics (IJCAI 2016) https://arxiv.org/abs/1608.01302
- [1804.00846] Learning to Search via Retrospective Imitation (2018) https://arxiv.org/abs/1804.00846
- [1810.00825] Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks — Lee et al. (ICML 2019) https://arxiv.org/abs/1810.00825
- [1909.06722] Plackett-Luce Model for Learning-to-Rank Task — Xia et al. (2019) https://arxiv.org/abs/1909.06722
- [2006.05067] Learning-to-Rank with Partitioned Preference: Fast Estimation for the Plackett-Luce Model — Ma et al. (AISTATS 2021) https://arxiv.org/abs/2006.05067
- [2010.01083] Integrated Task and Motion Planning — Garrett et al. (AAAI 2021) https://arxiv.org/abs/2010.01083
- [2102.13177] Efficient and Interpretable Robot Manipulation with Graph Neural Networks — Lin et al. (2021) https://arxiv.org/abs/2102.13177
- [2103.00589] Learning Symbolic Operators for Task and Motion Planning — Silver et al. (2021) https://arxiv.org/abs/2103.00589
- [2103.04374] Learning When to Quit: Meta-Reasoning for Motion Planning — Sung et al. (2021) https://arxiv.org/abs/2103.04374
- [2105.14074] Learning Neuro-Symbolic Relational Transition Models for Bilevel Planning — Chitnis et al. (2021) https://arxiv.org/abs/2105.14074
- [2111.13144] Learning to Search in Task and Motion Planning with Streams — Khodeir et al. (2023) https://arxiv.org/abs/2111.13144
- [2203.04605] Representation, Learning, and Planning Algorithms for Geometric TAMP — Kim et al. (IJRR 2021) https://arxiv.org/abs/2203.04605
- [2203.09634] Predicate Invention for Bilevel Planning — Silver et al. (AAAI 2023) https://arxiv.org/abs/2203.09634
- [2203.10568] Accelerating Integrated Task and Motion Planning with Neural Feasibility Checking — Xu et al. (2022) https://arxiv.org/abs/2203.10568
- [2203.13913] SpeqNets: Sparsity-aware Permutation-equivariant Graph Networks — Morris et al. (2022) https://arxiv.org/abs/2203.13913
- [2207.14422] Neural-Guided Runtime Prediction of Planners for Improved Motion and Task Planning with GNNs (2022) https://arxiv.org/abs/2207.14422
- [2210.14055] Policy-Guided Lazy Search with Feedback for Task and Motion Planning — Khodeir et al. (2023) https://arxiv.org/abs/2210.14055
- [2210.14215] In-Context Reinforcement Learning with Algorithm Distillation — Laskin et al. (ICLR 2023) https://arxiv.org/abs/2210.14215
- [2211.01576] Sequence-Based Plan Feasibility Prediction for Efficient Task and Motion Planning (PIGINet) — Yang et al. (RSS 2023) https://arxiv.org/abs/2211.01576
- [2310.19463] Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal — Ferber et al. (NeurIPS 2023) https://arxiv.org/abs/2310.19463
- [2402.01878] LiPO: Listwise Preference Optimization through Learning-to-Rank — Liu et al. (2024) https://arxiv.org/abs/2402.01878
- [2404.02817] A Survey of Optimization-based Task and Motion Planning: From Classical To Learning Approaches — Zhao et al. (2024) https://arxiv.org/abs/2404.02817
- [2404.06912] Set-Encoder: Permutation-Invariant Inter-Passage Attention for Listwise Passage Re-Ranking — Deckers et al. (2024) https://arxiv.org/abs/2404.06912
- [2407.13694] Anticipatory Task and Motion Planning — Dhakal et al. (2024) https://arxiv.org/abs/2407.13694
- [2410.05828] Effort Allocation for Deadline-Aware Task and Motion Planning: A Metareasoning Approach — Sung et al. (2024) https://arxiv.org/abs/2410.05828
- [2411.11833] Differentiable GPU-Parallelized Task and Motion Planning — Shen et al. (2024) https://arxiv.org/abs/2411.11833
