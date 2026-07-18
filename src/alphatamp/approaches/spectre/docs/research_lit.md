# SPECTRE: Related Literature Review (v2)

**Topic:** Learned test-time reordering for bilevel TAMP — failure-conditioned
context encoding, set/permutation-invariant architectures, listwise (Plackett–Luce)
ranking, and the problem framings adjacent to all three.

*Rewritten 2026-06-08. Supersedes the 2026-04-27 version. The living docs
(`proposal.md` / `decisions.md` / `notebook.md`) carry the foundational ideas
for this method; this file positions the method against the literature.*

> **2026-06-25 — positioning shift (see `proposal.md` §0 / `decisions.md`
> 2026-06-25).** The project reframed from *learned test-time reordering* to a
> **representation question** for plan-feasibility prediction: low-level
> (PIGINet-style) vs. abstract-first / learned-latent / object-centric /
> invented-predicate substrates, under realistic data/perception budgets.
> Consequently **PIGINet is reframed as the low-level static feasibility
> predictor we compare against** (not "adaptive prior work"), and adaptive
> reordering is the secondary increment. The table/section text below still
> carries the reordering-era emphasis where untouched; the representation lens is
> added in §2 and the boundary/evaluation entries are updated. Edits here are
> targeted, not a rewrite.

> **How to read the table.** The **Tier** column is the context-budget signal.
> *Core* = load-bearing for the method's positioning (keep always). *Supporting* =
> cite when the relevant sub-claim is made. *Background* = one-line framing only.
> *Conditional* = pull in only if a specific pivot/extension is taken.
> *Frame* = a problem framing the current writeup omits (see §6); cite if that
> framing is adopted.

---

## 0. Corrections carried over from v1 (read first)

Three things in the prior version are inaccurate and have been fixed below. They
are flagged here because they affect *which* citation is authoritative, not just
wording.

1. **The Plackett–Luce / ListMLE loss was attributed to the wrong author.** v1's
   "Xia et al. 2019 [1909.06722]" is **Tian Xia, Shaodan Zhai, Shaojun Wang
   (Wright State)** — a name collision. The primary source for the listwise PL
   likelihood loss SPECTRE actually uses is **Fen Xia, Tie-Yan Liu, Jue Wang,
   Wensheng Zhang, Hang Li, "Listwise Approach to Learning to Rank: Theory and
   Algorithm," ICML 2008** (ListMLE), with **ListNet (Cao et al. 2007)** as the
   origin of the listwise approach and **Plackett (1975) / Luce (1959)** as the
   statistical origin of the PL model. Cite the 2008 paper as the loss's source.

2. **PIGINet is not a pure operator/state-token encoder.** It tokenizes the plan
   skeleton *together with the concrete initial state and goal*, fusing CLIP
   image, text, and continuous-value embeddings. This sharpens — rather than
   weakens — SPECTRE's positioning: SPECTRE deliberately excludes `x₀` and works
   from symbolic structure alone, so PIGINet is the natural *`x₀`-conditioned*
   static ranker that SPECTRE's future-work prior would generalize.

3. **"Sung 2024 is closest in spirit to SPECTRE" overstates the link.** Effort
   Allocation is a metareasoning MDP (DP_Rerun) that allocates a compute/time
   budget across options under a deadline; it does not learn a representation
   over failure *structure*. It is the budget/metareasoning neighbor, not the
   representation-learning neighbor.

A fourth, non-citation point that should shape the related-work emphasis is in §6.

---

## 1. Literature Table

### Core — skeleton ranking / feasibility prediction (the baselines and direct line)

| Paper | Venue | Tier | What it does | Relation to SPECTRE |
|---|---|---|---|---|
| **PIGINet** — Yang, Garrett, Lozano-Pérez, Kaelbling, Fox 2023 [2211.01576] | RSS 2023 | Core | Transformer over (skeleton, goal, `x₀`) with fused CLIP image/text/value tokens predicts refinement feasibility; static pre-sort; trains on **150–600 problems** | **The low-level static feasibility predictor we compare against** (representation pivot, 2026-06-25). `x₀` = multi-camera images + relational literals over the concrete initial state. The crossover hypothesis: a richer-than-pixels / cheaper-than-full-state representation should match or beat it in the low-data / weak-perception regime, PIGINet regaining its edge with abundant data + strong perception. PIGINet's x₀-ablation (x₀ carries signal *in their kitchens*) is why our x₀ stance is open, not "drop x₀" |
| **LAZY / Policy-Guided Lazy Search** — Khodeir et al. 2023 [2210.14055] | — | Core | Reorders PDDLStream queue lazily from geometric samples + goal-directed policy; updates within search; **kitchen rearrangement with clutter/distractors** | **Closest *adaptive* prior work / B4 analog.** Uses frequency counts of co-failure, not a learned structural encoder. **Evaluation candidate (2026-06-25):** its clutter/distractor 7-DoF domains are a pre-existing home for the representation sweep (augment with a low-level baseline) |
| **Geometric TAMP rank function (SAHS)** — Kim et al. 2018 / 2021 [2203.04605] | NeurIPS-WS 2018 → IJRR 2021 | Core | Learned score-space rank function for discrete TAMP search + learned sampler | **Earliest explicit learned skeleton ranker.** Score-space (2018) is the origin; IJRR (2021) the journal extension |
| **Neural Feasibility Checking** — Xu et al. 2022 [2203.10568] | — | Supporting | NN predicts kinematic feasibility, replaces motion checks | Feasibility-prediction, action-level not skeleton-level |
| **Learning Feasibility for TAMP (tabletop)** — Wells, Dantam, Shrivastava, Kavraki 2019 | IEEE RA-L 2019 | Supporting | Learns a feasibility classifier used as a *search-ordering heuristic* | **Foundational** learned-feasibility-as-ordering; predecessor to PIGINet/Xu |
| **Deep Visual Heuristics** — Driess, Oguz, Ha, Toussaint 2020 | ICRA 2020 | Supporting | Learns feasibility of mixed-integer manipulation programs from images | Foundational learned feasibility heuristic; image-conditioned |
| **Extended Tree Search for TAMP** — Ren, Chalvatzaki, Peters 2021 [2103.05456] | — | Supporting | Maintains an explicit skeleton space (top-k planner) and estimates each skeleton's *value* via MCTS-style search | **Search-based** alternative to learned ranking for the *same* skeleton-selection problem |
| **Planning with Learned Object Importance** — Silver et al. 2021 [2009.05613, *verify*] | ICRA 2021 | Supporting | GNN predicts object relevance to prune large problem instances | Selection-by-relevance; complementary pruning signal |
| **Anticipatory TAMP** — Dhakal et al. 2024 [2407.13694] | — | Supporting | Learned future-cost predictor guides plan selection beyond immediate feasibility | Looks *ahead*; SPECTRE looks *back* at failures — useful contrast |

### Core — adaptive selection / metareasoning

| Paper | Venue | Tier | What it does | Relation to SPECTRE |
|---|---|---|---|---|
| **Effort Allocation / Metareasoning** — Sung, Shperberg, Wang, Stone 2024 [2410.05828] | TRO (under review) | Supporting | MDP over option-refinement budget under a deadline; DP_Rerun ≈ MCTS quality, negligible overhead | Budget/metareasoning neighbor; **not** representation learning over failure structure |
| **Learning to Search in TAMP with Streams** — Khodeir et al. 2023 [2111.13144] | — | Supporting | GNN heuristic over PDDLStream fact/object expansion order | Coarser than skeleton-level; static search heuristic |

### Core — architecture (set / permutation-invariant encoders)

| Paper | Venue | Tier | What it does | Relation to SPECTRE |
|---|---|---|---|---|
| **Set Transformer** — Lee et al. 2019 [1810.00825] | ICML 2019 | Core | SAB + PMA; permutation-invariant set encoding via inducing points | **Direct architecture** for Ψ and for Φ's atom pooling |
| **Deep Sets** — Zaheer et al. 2017 [1703.06114] | NeurIPS 2017 | Background | Characterizes permutation-invariant functions (sum-decomposition) | **Foundational** justification for why Ψ/atom-pooling *can* be invariant; the design choice (Set Transformer over Deep Sets, per `decisions.md` RT2D-fix-1) is precisely a Deep-Sets-vs-attention call |
| **Set-Encoder passage re-ranking** — Deckers et al. 2024 [2404.06912] | — | Supporting | Set Transformer for listwise re-ranking; removes position bias | Direct precedent for Set-Transformer-based *listwise re-ranking* |

### Core — loss (listwise ranking)

| Paper | Venue | Tier | What it does | Relation to SPECTRE |
|---|---|---|---|---|
| **ListMLE** — Fen Xia, Liu, Wang, Zhang, Li 2008 | ICML 2008 | Core | Listwise PL likelihood loss; consistency/soundness analysis | **Primary source for SPECTRE's loss** |
| **ListNet** — Cao, Qin, Liu, Tsai, Li 2007 | ICML 2007 | Background | Origin of the listwise approach (PL top-one probability) | Lineage of the listwise objective |
| **Plackett–Luce model** — Plackett 1975 / Luce 1959 | Appl. Stat. / book | Background | The permutation/choice model underlying the loss | Statistical origin of `P(argmax ∈ SUCC)` |
| **Rank, not estimate** — Ferber et al. 2023 [2310.19463] | NeurIPS 2023 | Core | Proves search heuristics need correct *ranking*, not calibrated cost; proposes ranking losses | **Theoretical grounding** for PL over regression/BCE in a planning loop |
| **LiPO** — Tianqi Liu et al. 2024 [2402.01878] | — | Supporting | Listwise PL beats pairwise DPO for argmax-aligned LLM alignment | Listwise > pointwise/pairwise for argmax objectives, in another domain |
| **PL with partitioned preference** — Ma et al. 2021 [2006.05067] | AISTATS 2021 | Conditional | Tractable PL estimation from group/partition labels | Relevant *only if* supervision collapses to a SUCC/FAIL partition rather than the current per-skeleton outcomes |
| **LTR for planning heuristics** — Garrett et al. 2016 [1608.01302] | IJCAI 2016 | Supporting | Applies LTR to synthesize classical-planning heuristics | Classical-planning predecessor of the rank-the-search idea |

### Supporting — amortized / in-context adaptation

| Paper | Venue | Tier | What it does | Relation to SPECTRE |
|---|---|---|---|---|
| **Algorithm Distillation** — Laskin et al. 2023 [2210.14215] | ICLR 2023 | Supporting | Causal Transformer over episode history → in-context RL improvement, no gradient updates | Conceptual parallel for Ψ ("read the history, emit a better prior"); SPECTRE's Ψ is a *set* not sequence encoder |

### Background — TAMP substrate and foundations

| Paper | Venue | Tier | What it does | Relation to SPECTRE |
|---|---|---|---|---|
| **Integrated TAMP (survey)** — Garrett et al. 2021 [2010.01083] | AAAI 2021 | Background | Canonical survey of bilevel skeleton-then-refine TAMP | Defines the loop SPECTRE operates in |
| **PDDLStream** — Garrett et al. 2020 [1802.08705] | ICAPS 2020 | Background | Integrates PDDL planners with blackbox samplers | Standard substrate; the queue LAZY reorders |
| **STRIPS** — Fikes & Nilsson 1971 | AIJ 1971 | Background | Symbolic action representation (preconditions/effects) | Why intermediate skeleton states are redundant given `s₀ + a₁:L` (Φ design) |
| **Predicate Invention for Bilevel Planning** — Silver et al. 2023 [2203.09634] | AAAI 2023 (v3 2025) | Background | Learns predicates by optimizing a surrogate aligned with planning efficiency | **Surrogate-objective alignment** precedent — the same discipline SPECTRE applies in choosing a rollout-aligned loss |
| **NSRTs** — Chitnis et al. 2021 [2105.14074] | — | Background | Neuro-symbolic relational transition models for bilevel planning | System context for the operator/sampler substrate |
| **SLAP** — Y. I. Liu, Li, Eysenbach, Silver 2025 [2511.01107] | — | Background | Discovers new options via RL shortcuts in the abstract graph | Same group/setting (deterministic, fully-observable); orthogonal to ranking |
| **TAMPURA** — Curtis et al. 2024 [2403.10454] | — | Background | TAMP with uncertainty + risk awareness; reasons in belief space at task + controller level | **The partially-observable boundary — out of scope for our FO claim.** Explicitly notes the typical TAMP formulation assumes full observability + deterministic effects; TAMPURA relaxes exactly that. Marks where the representation-efficiency claim no longer applies on information grounds |
| **KinDER** — *Physical Reasoning Benchmark for Robot Learning and Planning* 2026 [2604.25788] | RSS 2026 | Background | 25 procedurally-generated 2D/3D envs + 13 baselines; isolates five physical-reasoning axes: spatial relations, nonprehensile manipulation, tool use, combinatorial geometric constraints, dynamic constraints | Source of the kinder 2D envs (ClutteredStorage2D etc.). **Observability and determinism are *orthogonal* to its five axes** — KinDER varies physical-reasoning difficulty, not the FO/deterministic assumption SPECTRE works under; an env can be hard on a KinDER axis yet still FO+deterministic |
| **Sampling-based optimal motion planning (RRT\*/PRM\*)** — Karaman & Frazzoli 2011 [1105.1186] | IJRR 2011 | Background | Asymptotically optimal sampling-based motion planning | The continuous layer the refiner invokes |
| **Collision-free path planning** — Lozano-Pérez & Wesley 1979 | CACM 1979 | Background | Configuration-space path planning among obstacles | Origin of the motion-planning half of TAMP |
| **PRM** — Kavraki et al. 1996 | IEEE T-RA 1996 | Background | Probabilistic roadmaps | Classic motion-planning substrate |

### Conditional / Frame

| Paper | Venue | Tier | When to cite |
|---|---|---|---|
| **DAgger** — Ross, Gordon, Bagnell 2011 | AISTATS 2011 | Conditional | If an on-policy correction round is added (proposal §6 "DAgger round") |
| **Retrospective Imitation** — Sun et al. 2018 [1804.00846] | — | Conditional | If on-policy *search* imitation specifically is used |
| **Sequential design of experiments** — Chernoff 1959 | Ann. Math. Stat. 1959 | Frame | If the latent-inference framing (§6) is adopted — foundational |
| **Active sequential hypothesis testing** — Naghshvar & Javidi 2013 | Ann. Statistics 2013 | Frame | Modern anchor for "which experiment next to identify the latent" |
| **Best-arm identification / pure exploration** — Garivier & Kaufmann 2016 | COLT 2016 | Frame | If the selection problem is cast as fixed-confidence BAI over families |

### Dropped from v1 (and why)

| Paper | Reason |
|---|---|
| **SpeqNets** — Morris et al. 2022 [2203.13913] | Listed but never used in table or narrative; SPECTRE uses no GNN. Vestigial. |
| **Neural-Guided Runtime Prediction** 2022 [2207.14422] | Planner-level routing, not skeleton-level; the GNN-selection point is already made. Redundant. |
| **Optimization-based TAMP survey** — Zhao et al. 2024 [2404.02817] | Second survey; Garrett 2021 covers the needed background. Keep only if the optimization-as-TAMP angle becomes relevant. |
| **GNN robot manipulation** — Lin et al. 2021 [2102.13177] | Demoted to optional: the equivariance-design contrast (GNN structure vs. typed-local-id canonicalization) needs at most one citation; Deep Sets + Set Transformer carry the architectural argument. Reinstate if the canonicalization-vs-GNN tradeoff is argued explicitly. |
| **Learning When to Quit** — Sung et al. 2021 [2103.04374] | Subsumed by Effort Allocation 2024 (same author, same metareasoning theme). Keep only if *stopping* (vs. reordering) is discussed. |
| **Learning Symbolic Operators** — Silver et al. 2021 [2103.00589] | Three substrate papers (this + NSRTs + Predicate Invention) overcount; NSRTs + Predicate Invention suffice for system context. |

---

## 2. Skeleton ranking: the actual line of prior work

The bilevel decomposition (Garrett et al. 2021) makes skeleton ordering a secondary
optimization once the symbolic planner enumerates a goal-achieving pool (typically
via top-k/diverse planning). The learned-ranking line runs: learned *feasibility
classifiers used as search-ordering heuristics* (Wells et al. 2019; Driess et al.
2020) → learned *score-space rank functions* (Kim et al. 2018/2021) → *Transformer
feasibility predictors* over the whole skeleton (PIGINet, 2023). All of these are
**static**: each candidate is scored once, in isolation, before any refinement.

Two threads break from static scoring. **Search-based selection** maintains an
explicit skeleton space and estimates per-skeleton value online via tree search
(Extended Tree Search, Ren et al. 2021) — adaptive, but not amortized across
problems by a learned encoder. **Lazy/online reordering** (LAZY, Khodeir et al.
2023) updates the queue from geometric samples as they arrive, using co-failure
*frequency counts* that do not generalize across structurally similar skeletons.
SPECTRE's claim sits exactly here: a learned encoder that generalizes failure
structure across skeletons, rather than counting co-failures per canonical key.

**Representation lens (added 2026-06-25).** Under the current framing the line of
prior work is read differently: the static feasibility predictors above differ
not only in *being* static but in *what they represent the problem over*. PIGINet
predicts from the **low-level** initial state (multi-camera images + relational
literals); the open question is whether a **richer-than-pixels,
cheaper-than-full-state** substrate — *abstract-first* (abstract state + skeleton
structure; the current leading candidate but possibly too lossy), learned
latents, object-centric / graph features, intermediate symbolic-plus-coarse-
geometric states, or invented predicates — predicts refinement feasibility more
sample-efficiently and with weaker perception. This is an **efficiency /
perception-lightness** claim, not an information-access claim: under
fully-observable, deterministic TAMP no representation beats an ideal low-level
predictor on information grounds (cf. the FO information-ceiling, `decisions.md`
2026-06-25). The abstraction-learning neighbor is **predicate invention** (Silver
et al. 2023, [2203.09634]): it learns *which* relational predicates make planning
efficient — the same "what to represent over" question, applied to abstraction
construction rather than feasibility scoring. The negative control is dense
packing / fine continuous fit, where compression is expected to lose.

## 3. Architecture: set encoders and permutation invariance

Deep Sets (Zaheer et al. 2017) is the foundational result — any permutation-invariant
function admits a sum-decomposition — and is the right citation for *why* Ψ and the
atom pooling can be invariant at all. The Set Transformer (Lee et al. 2019) is the
specific instantiation SPECTRE uses (SAB for equivariant mixing, PMA for invariant
pooling). The RT2D-fix-1 decision to use Set-Transformer attention over Deep-Sets
pooling for the `PassageWidth`×`ItemSize` relational join is a concrete Deep-Sets-vs-
attention expressivity call, so both belong in the architecture story. Deckers et al.
(2024) is the closest precedent for using this family specifically for *listwise
re-ranking*.

## 4. Loss: listwise Plackett–Luce, and why not pointwise

SPECTRE's loss is the ListMLE/PL likelihood loss (**Fen Xia et al. 2008**; lineage
ListNet, Cao et al. 2007; model Plackett 1975 / Luce 1959), specialized to
`−log P(argmax over R picks a success)`. Three independent results support the
listwise-over-pointwise choice that Attempt-2 (BCE) lacked: Ferber et al. (2023)
prove that in search, heuristics need only *rank* correctly, not estimate cost —
the cleanest theoretical argument for PL over regression/BCE in a planning loop;
LiPO (Tianqi Liu et al. 2024) shows listwise PL beats pairwise DPO for
argmax-aligned objectives; and the LTR-for-heuristics line (Garrett et al. 2016)
is the in-domain precedent. Ma et al. (2021) is only relevant if supervision ever
degrades to a SUCC/FAIL partition — the current pipeline has per-skeleton outcomes,
so it is conditional, not core.

## 5. Amortized adaptation

Algorithm Distillation (Laskin et al. 2023) is the cleanest conceptual parallel for
Ψ: a network that ingests interaction history and emits an improved policy with no
test-time gradient step. The honest difference is that AD is a *sequence* model over
RL episodes; Ψ is a *set* encoder over an unordered failure set, which is the
theoretically correct inductive bias (failure order is uninformative). Treat this as
an analogy that motivates the architecture, not as a result about TAMP.

## 6. The framing the writeup omits — and what the ablation implies

**Analysis, not established prior work.** The following is a reframing I am
proposing for brainstorming; none of these papers address SPECTRE's problem
directly.

RT2D's generative story is a discrete latent `z = (BlockedColor, BlockedGrasp)`
that gates which skeleton *family* refines. Under that story, **adaptive reordering
from failures is posterior inference over `z`**: each failed skeleton is an
observation that rules out families, and the optimal next pick is the one most
likely to succeed under the current posterior — equivalently, the cheapest
experiment that most reduces uncertainty about `z`. This is the language of
**active sequential hypothesis testing / optimal experimental design** (Chernoff
1959; Naghshvar & Javidi 2013) and **best-arm identification / pure exploration**
(Garivier & Kaufmann 2016). The B4 baseline (Naive-Bayes log-odds over pairwise
failure conditionals) is precisely the *explicit, tabular* Bayes version of this;
SPECTRE's Ψ is an *amortized, learned* approximation to the same posterior.

Why this matters for the project, concretely:

- **It explains the Ψ-ablation (notebook 2026-06-06).** Failure-conditioning buys
  only ~1 attempt (~27% of the margin over B4); ~73% comes from the static Φ+σ
  ranking. If `z` is low-dimensional and the marginal family-success prior is
  already informative, the *static* posterior (the prior over `z` before any
  failure) captures most of the achievable gain, and online updating adds little.
  That is the expected behavior of an easy inference problem — not evidence that Ψ
  is broken.

- **It reprioritizes the related-work emphasis.** Because the static representation
  is doing most of the work, the *primary* baseline to beat is a strong **static**
  ranker (PIGINet-class), not adaptive prior work (LAZY/Sung). The current draft
  spends more positioning on adaptivity than the empirics justify. Lead with the
  static-ranking line; treat adaptivity as the (smaller, real) increment.

- **It surfaces a "right tool" question to settle before scaling.** Two cheap
  probes would isolate whether SPECTRE is the right instrument: (i) report the
  Bayes-optimal oracle that knows `z` and the family→`z` map — the gap between B4
  and that oracle bounds *all* achievable adaptivity, and if it is small the
  adaptive story has little headroom on RT2D by construction; (ii) an explicit
  information-gain selector (pick the skeleton maximizing expected posterior
  entropy reduction over families) as a *non-learned adaptive* baseline stronger
  than B4 — if it matches SPECTRE's adaptive increment, the learned Ψ is buying
  amortization/generalization, not better inference, which is a sharper and more
  defensible claim. The GPU-parallel refinement paradigm (Shen et al. 2024) is the
  other "right tool" pressure: if refinement parallelizes, the sequential ordering
  objective weakens.

This framing is offered as a lens for pivots, and as pre-emption for the obvious
reviewer question ("isn't this amortized Bayesian inference over the latent?"). The
strongest version of SPECTRE's contribution is probably *not* "failure context
helps" (the ablation makes that the minor effect) but "a learned set-encoder over
skeleton structure generalizes the success posterior across a combinatorial
skeleton space where tabular conditioning cannot" — which is a representation claim,
testable against the BAI/info-gain baselines above.

**2026-06-25 update.** This section's own closing intuition is now the project's
lead: the contribution is the **representation** claim, and the latent-inference /
BAI / info-gain framing (which is about the *adaptive* increment) is demoted with
it. The active comparison is therefore against the **low-level static predictor**
(PIGINet-class), not against stronger adaptive baselines, and the falsifiable
prediction is the perception × data **crossover**, not a tighter posterior. The
latent-inference material above is retained as the record of the
reordering-era analysis and as reviewer pre-emption. See `proposal.md` §0 /
`decisions.md` 2026-06-25.

---

## 7. Recommended citation clusters

| Claim / Section | Papers |
|---|---|
| Problem setting (bilevel TAMP) | Garrett et al. 2021 [2010.01083]; PDDLStream [1802.08705]; STRIPS 1971 |
| Static skeleton ranking (baselines) | PIGINet [2211.01576]; Kim 2018/2021 [2203.04605]; Wells 2019; Driess 2020 |
| Adaptive / search-based selection | LAZY [2210.14055]; Extended Tree Search [2103.05456]; Sung 2024 [2410.05828] |
| Architecture (set encoding) | Lee 2019 [1810.00825]; Zaheer 2017 [1703.06114]; Deckers 2024 [2404.06912] |
| Loss (listwise PL) | Fen Xia 2008 (ListMLE); Ferber 2023 [2310.19463]; LiPO [2402.01878] |
| Amortized adaptation | Laskin 2023 [2210.14215] |
| Latent-inference framing (if adopted) | Chernoff 1959; Naghshvar & Javidi 2013; Garivier & Kaufmann 2016 |
| Substrate context | Predicate Invention [2203.09634]; NSRTs [2105.14074] |
| "Right tool" pressure | Shen 2024 [2411.11833] |

---

## 8. Full reference list

Verified via search (2026-06-08): PIGINet, Effort Allocation, ListMLE-vs-Tian-Xia
identity, SLAP, Predicate Invention venue, Extended Tree Search, Wells/Driess
existence, active-testing anchors. arXiv IDs marked *[verify]* were not
re-confirmed and should be checked before use.

- [1105.1186] Sampling-based Algorithms for Optimal Motion Planning — Karaman & Frazzoli (IJRR 2011) https://arxiv.org/abs/1105.1186
- [1608.01302] Learning to Rank for Synthesizing Planning Heuristics — Garrett et al. (IJCAI 2016) https://arxiv.org/abs/1608.01302
- [1703.06114] Deep Sets — Zaheer et al. (NeurIPS 2017) https://arxiv.org/abs/1703.06114
- [1802.08705] PDDLStream: Integrating Symbolic Planners and Blackbox Samplers — Garrett et al. (ICAPS 2020) https://arxiv.org/abs/1802.08705
- [1804.00846] Learning to Search via Retrospective Imitation — Sun et al. (2018) https://arxiv.org/abs/1804.00846
- [1810.00825] Set Transformer — Lee et al. (ICML 2019) https://arxiv.org/abs/1810.00825
- [2006.05067] Learning-to-Rank with Partitioned Preference (fast PL estimation) — Ma et al. (AISTATS 2021) https://arxiv.org/abs/2006.05067
- [2009.05613] Planning with Learned Object Importance in Large Problem Instances using GNNs — Silver et al. (ICRA 2021) *[verify id]* https://arxiv.org/abs/2009.05613
- [2010.01083] Integrated Task and Motion Planning — Garrett et al. (AAAI 2021) https://arxiv.org/abs/2010.01083
- [2103.05456] Extended Tree Search for Robot Task and Motion Planning — Ren, Chalvatzaki, Peters (2021) https://arxiv.org/abs/2103.05456
- [2105.14074] Learning Neuro-Symbolic Relational Transition Models for Bilevel Planning (NSRTs) — Chitnis et al. (2021) https://arxiv.org/abs/2105.14074
- [2111.13144] Learning to Search in Task and Motion Planning with Streams — Khodeir et al. (2023) https://arxiv.org/abs/2111.13144
- [2203.04605] Representation, Learning, and Planning Algorithms for Geometric TAMP (SAHS) — Kim et al. (IJRR 2021; origin: score-space, NeurIPS-WS 2018) https://arxiv.org/abs/2203.04605
- [2203.09634] Predicate Invention for Bilevel Planning — Silver et al. (AAAI 2023; v3 2025) https://arxiv.org/abs/2203.09634
- [2203.10568] Accelerating Integrated Task and Motion Planning with Neural Feasibility Checking — Xu et al. (2022) https://arxiv.org/abs/2203.10568
- [2210.14055] Policy-Guided Lazy Search with Feedback for TAMP (LAZY) — Khodeir et al. (2023) https://arxiv.org/abs/2210.14055
- [2210.14215] In-Context Reinforcement Learning with Algorithm Distillation — Laskin et al. (ICLR 2023) https://arxiv.org/abs/2210.14215
- [2211.01576] Sequence-Based Plan Feasibility Prediction for Efficient TAMP (PIGINet) — Yang, Garrett, Lozano-Pérez, Kaelbling, Fox (RSS 2023) https://arxiv.org/abs/2211.01576
- [2310.19463] Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal — Ferber et al. (NeurIPS 2023) https://arxiv.org/abs/2310.19463
- [2402.01878] LiPO: Listwise Preference Optimization through Learning-to-Rank — T. Liu et al. (2024) https://arxiv.org/abs/2402.01878
- [2403.10454] Task and Motion Planning with Uncertainty and Risk Awareness (TAMPURA) — Curtis et al. (2024) https://arxiv.org/abs/2403.10454
- [2404.06912] Set-Encoder: Permutation-Invariant Inter-Passage Attention for Listwise Re-Ranking — Deckers et al. (2024) https://arxiv.org/abs/2404.06912
- [2407.13694] Anticipatory Task and Motion Planning — Dhakal et al. (2024) https://arxiv.org/abs/2407.13694
- [2410.05828] Effort Allocation for Deadline-Aware TAMP: A Metareasoning Approach — Sung, Shperberg, Wang, Stone (2024) https://arxiv.org/abs/2410.05828
- [2511.01107] SLAP: Shortcut Learning for Abstract Planning — Y. I. Liu, Li, Eysenbach, Silver (2025) https://arxiv.org/abs/2511.01107
- [2604.25788] KinDER: A Physical Reasoning Benchmark for Robot Learning and Planning — Princeton Robot Planning and Learning group (RSS 2026) https://arxiv.org/abs/2604.25788
- ListMLE: Listwise Approach to Learning to Rank — Theory and Algorithm — Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, Hang Li (ICML 2008)
- ListNet: Learning to Rank — From Pairwise Approach to Listwise Approach — Cao, Qin, Liu, Tsai, Li (ICML 2007)
- The Analysis of Permutations — Plackett (Applied Statistics, 1975); Individual Choice Behavior — Luce (1959)
- Learning Feasibility for Task and Motion Planning in Tabletop Environments — Wells, Dantam, Shrivastava, Kavraki (IEEE RA-L 2019)
- Deep Visual Heuristics: Learning Feasibility of Mixed-Integer Programs for Manipulation Planning — Driess, Oguz, Ha, Toussaint (ICRA 2020)
- STRIPS: A New Approach to the Application of Theorem Proving to Problem Solving — Fikes & Nilsson (AIJ 1971)
- An Algorithm for Planning Collision-Free Paths Among Polyhedral Obstacles — Lozano-Pérez & Wesley (CACM 1979)
- Probabilistic Roadmaps for Path Planning in High-Dimensional Configuration Spaces — Kavraki, Švestka, Latombe, Overmars (IEEE T-RA 1996)
- A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger) — Ross, Gordon, Bagnell (AISTATS 2011)
- Sequential Design of Experiments — Chernoff (Annals of Mathematical Statistics, 1959)
- Active Sequential Hypothesis Testing — Naghshvar & Javidi (Annals of Statistics, 2013)
- Optimal Best Arm Identification with Fixed Confidence — Garivier & Kaufmann (COLT 2016)

### Name-collision warnings (kept here to prevent re-introduction)

- **"Xia" (LTR):** Fen Xia (ICML 2008, ListMLE — *the* primary source) ≠ Tian Xia
  (Wright State, arXiv 1909.06722 — a later non-linear PL preprint, not the origin).
- **"Liu":** Tianqi Liu (Google, LiPO) ≠ Y. Isabel Liu (Princeton, SLAP).
- **"Silver" / Predicate Invention:** arXiv 2203.09634 = AAAI 2023; the v3 revision
  is dated 2025 — cite as 2023 (venue) unless you mean the revised arXiv.
