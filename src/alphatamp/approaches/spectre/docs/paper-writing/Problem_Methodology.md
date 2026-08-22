# Problem Formulation

> **Framing.** A candidate skeleton that fails refinement is not just a failed attempt — it is an _observation_ about the specific problem in front of us (which objects got in the way, what actually changed in the world). Existing methods largely throw this signal away: static rankers never look at what failed, and even adaptive methods use only a coarse "it failed" bit. **SPECTRE's thesis is that a re-ranker which represents each failure in structured detail, and is trained to exploit it, extracts far more of this signal.** The Problem Formulation below builds up to that claim; the Methodology delivers on it.

### Setting: deterministic, fully-observable TAMP

- We study **task-and-motion planning (TAMP)**: a robot must find a sequence of actions that reaches a goal, where each action has both a _discrete_ choice (which operator — pick, place, …) and a _continuous_ choice (exact grasp, placement pose, …).
- States and actions are continuous: state space $\mathcal{X}$, action space $\mathcal{U}$. The world is **deterministic and fully observable** — a known transition function $f : \mathcal{X} \times \mathcal{U} \to \mathcal{X}$ (e.g. a physics simulator) says exactly what the next state is. Each state contains a finite set of typed **objects** $\mathcal{O}$ with geometric attributes (shape, pose).
- A **problem instance** is a pair $(x_0, g)$: an initial state $x_0 \in \mathcal{X}$ and a **goal** $g$, a conjunction of ground atoms over objects (e.g. $\texttt{InDrawer}(o_5)$); a state $x$ satisfies the goal iff its abstraction entails $g$ (abstraction defined below). Instances are drawn from a distribution $\mathcal{D}$; we train on a sample from $\mathcal{D}$ and evaluate on held-out instances from the same $\mathcal{D}$.
- _Scope assumptions (stated up front, revisited in Limitations):_ the domains are fully observable, deterministic, and **kinematic** (feasibility is about reachability and collision, not dynamics/forces). Every evaluation instance admits at least one refinable skeleton — instances with none are discarded at data collection — so "attempts to first success" (below) is finite and well-defined.

### Bilevel TAMP and skeleton refinement

- We use the standard **bilevel** decomposition of TAMP: a fast _symbolic_ layer proposes a plan outline, and a slower _continuous_ layer tries to fill in the geometric details.
- A **relational abstraction** $\mathrm{abs}$ maps a concrete state $x$ to an **abstract state** $s = \mathrm{abs}(x)$ — a set of true/false facts (ground atoms) over the objects, e.g. $\texttt{Holding}(o_2)$ or $\texttt{OnBuffer}(o_1)$. This is the "symbolic view" of the world.
- A symbolic planner produces a **skeleton** $$\sigma = (s_0,; a_1, s_1,; \dots,; a_L, s_L),$$ an alternating sequence of **grounded operators** $a_1,\dots,a_L$ (each a discrete action with its object arguments filled in) and the **abstract states** they induce under STRIPS semantics. Steps within a skeleton are indexed by $j \in {1,\dots,L}$. Here $s_0 = \mathrm{abs}(x_0)$ is the abstract initial state and the final state $s_L$ entails the goal $g$. Intuitively, a skeleton is a _plan outline_: "pick $o_1$, put it on the buffer, then retrieve the target" — correct symbolically, but not yet a real motion.
- Each operator $a_j$ carries a **continuous parameter domain** $\Omega_j$ (grasps, placement poses, …). A full assignment $\omega = (\omega_1, \dots, \omega_L) \in \Omega(\sigma) := \Omega_1 \times \cdots \times \Omega_L$ turns the skeleton into executable controllers whose execution under $f$ from $x_0$ yields concrete states $x_1,\dots,x_L$.
- **Refinement** is the search for such an $\omega$ that is _consistent_ with the skeleton, i.e. $\mathrm{abs}(x_j) = s_j$ for every step $j$. A sampler tries candidate parameters within a fixed budget; if no consistent $\omega$ is found, the skeleton is declared **infeasible** and we move on to the next one. Classically, refinement is treated as a black-box test $$\mathrm{refine}(\sigma;, x_0) ;\in; \Omega(\sigma) ,\cup, {\bot}$$ — either a satisficing assignment $\omega$ (the skeleton is now an executable plan) or an _uninformative_ failure $\bot$. Our formulation enriches exactly the failure branch: $$\mathrm{refine}(\sigma;, x_0) ;\in; \big({\texttt{succ}} \times \Omega(\sigma)\big) ,\cup, \big({\texttt{fail}} \times \mathcal{E}\big).$$ On success the refiner returns the satisficing $\omega$ as before; on failure it returns **failure evidence** $e \in \mathcal{E}$ in place of $\bot$: a set of structured records of _what the refiner already computed while failing_ — which continuous query died, at which step $j$, which objects it named (Methodology M1). Emitting $e$ costs nothing extra: it is observation-only instrumentation of work the refiner performs anyway — the formulation differs from the classical one only in _not discarding_ this byproduct. Refinement is where the real cost lives — a single failed refinement can burn seconds of collision-checking and sampling. _(Refinement is randomized through its sampler; at data-collection time we fix a stable per-skeleton seed so labels are reproducible, and at deployment we report across-seed variation. A successful attempt may also burn failed samples internally before finding $\omega$; the episode ends at success, so no consumer ever reads evidence from a successful candidate — cf. the training-context invariant, M4.)_

### Candidate pool and adaptive reordering

- For a problem $(x_0, g)$, the symbolic planner returns not one skeleton but a **candidate pool** $\mathcal{S} = {\sigma_1, \dots, \sigma_K}$ of $K$ goal-reaching skeletons, in some default order (e.g. increasing heuristic cost). Symbolically they all reach the goal; the planner _cannot_ tell which are geometrically refinable, because that depends on continuous fit it never computes.
    
- A **planning episode** processes the pool as a sequence of refinement attempts, indexed by $t = 1, 2, \dots$ (note: $t$ indexes _attempts across the episode_; $j$ indexes _steps within a skeleton_). Because refinement dominates wall-clock, the **order** in which we try skeletons is what determines cost: finding a refinable skeleton on attempt 1 vs. attempt 10 is the difference between a fast solve and a slow one.
    
- At attempt $t$, the agent has a **failure history** $$\mathcal{F}_{t-1} = \{ (\hat{\sigma}_1, e_1), \dots, (\hat{\sigma}_{t-1}, e_{t-1}) \}$$ the skeletons already tried — all of which failed refinement (a success would have ended the episode) — _paired with the evidence each failure returned_. The next skeleton is picked from the **remaining set** $\mathcal{R}_t = \mathcal{S} \setminus {\sigma : (\sigma, \cdot) \in \mathcal{F}_{t-1}}$. A **selection policy** is therefore a map $$\pi : \big(x_0,; g,; \mathcal{S},; \mathcal{F}_{t-1}\big) ;\longmapsto; \hat{\sigma}_t \in \mathcal{R}_t.$$ The arguments are exactly the information available at attempt $t$: the instance itself, the pool, and everything observed so far this episode.
    
- **Method families differ in which arguments they use, and at what fidelity.** This single signature places every comparator in one class:
    
    |policy family|uses $x_0$?|uses $\mathcal{F}_{t-1}$?|
    |---|---|---|
    |planner default order (astar/FF)|no (only $s_0$, via the skeletons)|no|
    |static learned rankers (PIGINet-style)|yes — low-level (scene image)|no|
    |zero-shot VLM ordering (VLMPlan)|yes — low-level (scene image)|no|
    |adaptive learned search (LAZY)|problem/plan structure|coarse — a scalar feasibility statistic|
    |**SPECTRE**|yes — object-centric geometry|**yes — structured records**|
    
    **Static rankers** are the special case where $\pi$ ignores $\mathcal{F}_{t-1}$ and fixes an order once, before any attempt. The **adaptive** setting we study is strictly more general: $\pi$ may _re-rank_ the remaining pool after every observed failure.
    
- _Representation choice (how $\pi$ sees $x_0$)._ SPECTRE conditions on $x_0$ through a **mid-level, object-centric geometric summary** — per-object boundary outlines and poses, plus goal flags from $g$ — rather than raw pixels (the PIGINet route) or privileged full simulator state. The _registered hypothesis_ (not yet an established result) is that this representation is richer-than-pixels where geometry decides feasibility, yet cheaper and more perception-attainable than full state. _Scope note:_ all current numbers compute this summary from exact simulator geometry; robustness to estimated geometry is examined separately (Limitations).
    

### Problem statement (objective)

- A policy $\pi$ unrolls an episode by the recursion $$\hat{\sigma}_t = \pi(x_0,, g,, \mathcal{S},, \mathcal{F}_{t-1}), \qquad (o_t,, y_t) = \mathrm{refine}(\hat{\sigma}_t;, x_0), \qquad \mathcal{F}_t = \mathcal{F}_{t-1} \cup {(\hat{\sigma}_t, y_t)} ;\text{ if } o_t = \texttt{fail},$$ with $\mathcal{F}_0 = \emptyset$, stopping when some $o_t = \texttt{succ}$ — at which point $(\hat{\sigma}_t,, \omega = y_t)$ is the executable plan the episode returns — or when the pool is exhausted.
- **Primary metric — attempts to first success.** $$\mathrm{Attempts}(\pi;, x_0, g) = \inf{, t \geq 1 : o_t = \texttt{succ} ,}.$$ We report the number of **failed** attempts before the first success, $\mathrm{FP} = \mathrm{Attempts} - 1$ (so a first-attempt solve scores $0$), averaged over test problems. Evaluation is **uncensored**: the budget equals the pool size $K$, so it never cuts an episode short — the metric measures the full tail. _(Note: this replaces the older "budget 30" convention from an earlier environment; current pools hold up to $\sim!200$ candidates.)_
- **Secondary metric — wall-clock time to first success.** $$T(\pi;, x_0, g) = \sum_{t=1}^{\mathrm{Attempts}(\pi;, x_0, g)} \mathrm{refineTime}(\hat{\sigma}_t),$$ the total refinement time (plus plan-generation and inference overhead) until the first success. Attempts and time differ because failures are not equally cheap — a near-feasible skeleton can be expensive to reject. We report time under a per-candidate refinement cap $\tau$ (a deployment knob, Methodology M5), under which attempts is a good hardware-independent surrogate for time.
- **Learning objective.** We seek the policy minimizing expected cost over the task distribution: $$\pi^{\star} = \arg\min_{\pi}; \mathbb{E}_{(x_0, g) \sim \mathcal{D}}\big[, T(\pi;, x_0, g) ,\big].$$

### Why failure history? The observation that motivates SPECTRE

- **The information sitting unused.** When a skeleton fails refinement, the refiner has already computed _why_: the specific objects whose geometry blocked the attempted motion, or the specific way the achieved abstract state deviated from the plan. That is instance-specific evidence about which _other_ skeletons in the pool are likely to fail for the same reason.
- **How prior methods leave it on the table.**
    - _Static feasibility rankers_ (PIGINet-style low-level predictors, learned static scorers, the planner's own heuristic order) fix the order before any attempt and never see $\mathcal{F}$ at all.
    - _Adaptive search methods_ condition on failures only coarsely — e.g. a scalar feasibility feedback — rather than on the structured content of each failure.
- **SPECTRE's claim.** Represent each failure as a **structured record** (failed query, the objects it named, the resulting state change), re-rank the whole remaining pool on the _set_ of such records, and _train the model to use them_ — and the policy improves markedly over methods that discard or under-use this signal.
- **Evidence (forward reference to Experiments).** On DD2D (a drawer-declutter packing/retrieval domain), SPECTRE-adaptive reduces mean failed attempts to first success to $6.29 \pm 0.31$, versus $17.27$ for the best static learned predictor (PIGINet), $23.26$ for a learned _adaptive_ competitor (LAZY), $34.52$ for the planner's default order, and $19.80$ for its own static ablation (SPECTRE-static) — with the margin _widening_ under distribution shift (held-out difficulty stratum: $9.97$ vs. PIGINet's $85.89$). Crucially, the decomposition is literal about the thesis: SPECTRE and the static rankers solve the _same_ episodes on the first attempt; **the entire margin appears after the first observed failure** — exactly what a failure-conditioned method should buy, and independent corroboration that no feature leaks feasibility. The failure-utilization gain (adaptive over static) recurs on the second environment, StickButton2D, where the _representation_ contrast between learned methods does not separate — the two claims are distinguished throughout.

### Notation summary

|symbol|type / meaning|
|---|---|
|$\mathcal{X}, \mathcal{U}, f$|concrete states, actions, known deterministic transition|
|$\mathcal{O}$|typed objects of the instance|
|$(x_0, g) \sim \mathcal{D}$|problem instance: initial state, goal atoms|
|$\mathrm{abs}(x) = s$|relational abstraction; abstract state = set of ground atoms|
|$\sigma = (s_0, a_1, s_1, \dots, a_L, s_L)$|skeleton; steps indexed $j \in {1..L}$|
|$\Omega_j,\ \Omega(\sigma),\ \omega$|step-$j$ parameter domain; their product; full assignment $\omega \in \Omega(\sigma)$|
|$\mathcal{S} = {\sigma_1 .. \sigma_K}$|candidate pool, size $K$|
|$t$|episode attempt index (distinct from step index $j$)|
|$e \in \mathcal{E}$|failure evidence: set of failure records from one refinement|
|$\mathrm{refine}(\sigma; x_0)$|returns $(\texttt{succ},, \omega \in \Omega(\sigma))$ or $(\texttt{fail},, e \in \mathcal{E})$|
|$\mathcal{F}_{t-1}$|failure history: set of (skeleton, evidence) pairs|
|$\mathcal{R}_t$|remaining (untried) candidates at attempt $t$|
|$\pi(x_0, g, \mathcal{S}, \mathcal{F}_{t-1})$|selection policy|
|$\mathcal{K}_{\mathcal{F}} \subseteq \mathcal{O}$|culprit pool: blamed, actionable, non-universal objects (M3)|
|$\tau$|per-candidate refinement cap (seconds), deployment knob|
|$\Phi,\ \Psi,\ \ell(\sigma)$|skeleton encoder, record encoder, per-candidate logit|

---

# Methodology

**Overview.** SPECTRE (_Skeleton-Pool Embedding with Contextual Transformer for REordering_) is a learned selection policy $\pi$ with two cooperating halves, trained end-to-end:

1. a **static scorer** — an object-centric, relational encoder that reads each candidate skeleton and the scene geometry, and produces a feasibility score even before any attempt (this is the representation that orders the _first_ pick); and
2. an **adaptive re-ranker** — a failure-record encoder that, after each failed attempt, turns the failure history $\mathcal{F}_{t-1}$ into structured evidence and _re-scores_ the remaining pool.

The whole model is trained to a single rollout-aligned **listwise** loss so that the top-scored skeleton is likely to be the one that refines. We use hidden width $d = 64$ and 4-head attention throughout (a compact model, on the order of a few hundred thousand parameters). **[Figure 1: architecture — scene/skeleton encoders → static score; failure records → evidence attention → re-rank.]**

_The subsections below give the exact inputs, encoders, layers, training, and inference. Source pointers in italics are for verification and can be dropped in the paper draft._

### M1 — Input representation: what the model sees

For each problem the model consumes three kinds of tokens. All object identities are carried by a shared **tag** (a small integer id per object), reused across every slot below — this tag-join is what lets an operator argument, a blocking object, and a scene shape all refer to the _same_ object.

- **Scene / object tokens** — one per object in $\mathcal{O}$. These carry the model's view of the concrete initial state $x_0$: per object,
    - a **point set** describing the object's shape — a $32$-point **boundary ring** in 2D (arc-length resampled so it is order- and start-invariant), or a $32$-point surface **point cloud** in 3D (used in the 3D environment Restock3D, where a small cube and a tall block share the same footprint and differ only in _height_, so a 2D outline would be blind to what makes them behave differently). Each point additionally carries local **differential features** computed from its neighbours — an outward-facing **surface normal**, a **curvature** (2D) or **surface-variation** (3D) measure, and a **flatness** measure — so the shape encoder sees local geometry (edges, corners, pockets), not just a bag of coordinates;
    - its **pose** $(x/\text{scale},, y/\text{scale},, \theta)$ in 2D (and $(x,y,z,\text{yaw})$ in 3D), normalized by the scene frame size;
    - three domain-agnostic **shape scalars** $[,\text{area},, \sin\theta,, \cos\theta,]$;
    - a coarse **goal flag** $\texttt{obj\_is\_goal} \in {0,1}$ marking objects named anywhere in the goal literals of $g$;
    - an **atom profile** — a pooled embedding of the abstract facts (ground atoms) that name this object, drawn from _both_ the **initial abstract state** $s_0$ and the **goal** $g$, and kept on separate "true now" vs. "wanted" channels. Each fact contributes its **predicate identity** and the object's **argument slot**, so $\texttt{On}(a,b)$ and $\texttt{On}(b,a)$ are distinguished — information the coarse goal flag discards. This is how the symbolic state $s_0$ and goal $g$ enter the model _directly_, rather than only implicitly through the skeletons. (Facts with no object argument, e.g. $\texttt{HandEmpty}$, fold into a small global summary instead.)
    - _This is deliberately a mid-level surface: geometric and relational enough to decide reachability and collision, but object-centric rather than pixel- or full-state-level. The columns are domain-agnostic by construction — no target-anchored offsets, no per-environment flags — so the same encoder serves every environment. Note that the scene tokens (geometry and atom profiles alike) are shared by all candidates in a pool; they do not by themselves rank candidates, but each candidate reads them differentially through attention (M2–M3), and they carry the cross-problem signal a learned ranker generalizes over. Scope note: shapes and poses are currently read exactly from the simulator; the fixed geometric interface is where estimated perception would plug in._
- **Candidate (skeleton) tokens** — for each candidate, one token per operator step, holding the operator's **schema id** (which operator), its **position** in the plan, and its **argument slots** (object tags). A candidate is thus a short sequence of typed operator tokens.
- **Failure-record tokens** — the adaptive input; one token per observed failure in $\mathcal{F}_{t-1}$, holding: the **schema** of the failed query; the objects it was **about** (arg tags) kept separate from the objects observed to **block** it (culprit tags); scalar context $[j/L, \log(1{+}\text{effort})/10, \text{exhausted}, \text{effort\_is\_total}]$ (how deep in the plan it failed, how hard the sampler tried, whether it exhausted its retries); and an **abstract-state delta** $s_j - s_0$ (which atoms the failed prefix added/deleted, on separate role axes). The refiner emits one record per failed _sample_; the deployed configuration **aggregates** to one record per (schema, args) — deepest step, summed effort, unioned culprits — cutting tokens by $\sim!89$ with nothing the token _encodes_ lost.

_Sources: `dataset.py` (tensorization; per-point features `compute_point_feats`; atom profiles `_atom_profile_arrays`), `encoders.py` (`PointSetEncoder`, `AtomProfileEncoder`), `model.py`, `failure_record.py`, `tags.py`, `envs/restock3d/scene_geometry.py` (3D point cloud); aggregation `--aggregate-records`._

### M2 — Static scoring (the representation)

This half produces a per-candidate feasibility logit from geometry and plan structure alone — it is the ranking used on the first attempt, when $\mathcal{F} = \emptyset$.

- **Scene encoder.** Each object's point set (M1) is summarized into a fixed-length shape descriptor by a small **point-set network**: it lifts each point's features, runs one **EdgeConv** layer — a graph-convolution step in which every point exchanges information with its nearest neighbours, so the descriptor captures local structure like edges and pockets rather than isolated points — a Set-Attention Block over the points, and an attention-pooling to one descriptor. This descriptor is concatenated with the object's tag embedding, pose, shape scalars, and goal flag; the object's **atom profile** (M1) is added in; the result is projected to width $d=64$ and passed through **two Set-Attention Blocks (SAB)** so objects attend to one another. (An SAB is a permutation-invariant transformer layer over a _set_ — no positional order — which lets the model reason relationally, e.g. "this object sits between the gripper and the target." The point-set network is dimension-generic: the same code path encodes a 2D boundary ring and a 3D point cloud.)
- **Skeleton encoder $\Phi$.** Each operator step is embedded as (schema embedding $+$ learned position $+$ projected argument-tag features), and the steps are pooled by a **Pooling-by-Multihead-Attention (PMA)** block into one vector $e(\sigma) \in \mathbb{R}^{d}$ per candidate. (PMA is a learned attention-pooling that turns a variable-length set/sequence into a single vector.) $\Phi$ does **not** depend on $\mathcal{F}$, so it runs **once per episode**.
- **Geometry scoring channel.** Each candidate's $e(\sigma)$ **cross-attends** over the scene tokens (plus a small global summary token), producing a geometry-conditioned candidate vector. A per-candidate logit follows from the head (M3). Trained alone, this is already a competent static feasibility ranker.

_Sources: `encoders.py` (`SceneEncoder`, `PointSetEncoder`, `AtomProfileEncoder`, `CandidateEncoder`, `EdgeConv`), `layers.py` (SAB/PMA/MultiSeedPMA), `model.py`._

### M3 — Adaptive scoring (the contribution)

This half is what reads the failure history and re-ranks. Everything here is **exactly zero when $\mathcal{F} = \emptyset$**, so the first attempt is purely static and the failure signal only accrues as the episode observes failures.

- **Record encoder $\Psi$.** Each failure record becomes one token. The objects the query was _about_ and the objects that _blocked_ it are pooled into **separate** slots (role separation is load-bearing — "the prefix moved $o_1$ to the buffer" and "$o_1$ blocked the grasp" are different facts about the same object; pooling both into one slot would say only "these objects are associated with this failure"). The scalar context and the abstract-state delta are folded in; the delta branch is **additive and zero-initialized**, so an empty record contributes nothing.
- **Separate evidence-attention channel.** Candidates cross-attend over the failure tokens in their **own** attention channel, _distinct_ from the geometry channel. This matters empirically: with a single attention softmax shared over $\sim!10$ scene tokens and up to hundreds of record tokens, the reliably-useful geometry crowded out the noisier, more numerous failure tokens and the model learned to ignore them. Two channels remove that competition.
- **Observed coverage and waste** (per candidate). These two scalar features are the crux of how a failure translates into a re-ranking signal. They are the **unified definition** (authoritative: `unified_evidence.py`; spec `docs/unified_culprits_coverage_waste.md`), computed only from objects the refiner actually _reported_, over the filtered **culprit pool** $$\mathcal{K}_{\mathcal{F}} ;=; (\mathrm{Actionable} \setminus \mathrm{Universal}) ,\cap \bigcup_{(\sigma, e) \in \mathcal{F}} \mathrm{blame}(e),$$ where an object is _actionable_ iff some operator's effects mention it, and _universal_ iff it appears in every ground operator instance (behaviorally, the robot — excluded because it would spuriously justify every step). Read the two features as a **recall / precision** pair:
    - $\text{coverage}(\sigma)$ — _recall over the failures' stories_: the fraction of blamed objects in $\mathcal{K}_{\mathcal{F}}$ that candidate $\sigma$ **discharges before it re-enters the situation that named them**. (Where the abstraction can express the hazard, this is a state-entailment test against the candidate's own STRIPS-predicted states $\hat{s}_j(\sigma)$; where it cannot, a "does $\sigma$ deal with the object before retrying the same step" index test.) High coverage = "this candidate addresses exactly what went wrong."
    - $\text{waste}(\sigma)$ — _precision over unexplained work_: among $\sigma$'s **superfluous** steps (those its own causal chain to the goal cannot justify, found by a backward-relevance pass), the fraction that answer to **no** blamed object. High waste = "this candidate does pointless work that the evidence does not motivate." Waste **abstains** (returns $0$) when $\mathcal{K}_{\mathcal{F}} = \emptyset$ — otherwise every candidate would read $1.0$ from zero evidence.
    - Both are $0$ at $|\mathcal{F}| = 0$. _(On simple cases — e.g. DD2D's stage-then-retrieve plans — these reduce to an intuitive object-set recall / precision; the deployed definition is the causal one above, which is what makes them meaningful when tools and multi-step plans are involved.)_
- **Scoring head.** The final per-candidate logit is an MLP over the concatenation $$\ell(\sigma) = \mathrm{MLP}\big[; e(\sigma);;; \text{geometry-attended};;; \text{evidence-attended};;; \underbrace{[\text{jaccard}, \text{coverage}, \text{waste}]}_{\text{overlap features}} ;\big],$$ i.e. what the candidate is, what geometry it sees, what the failures say, and three scalar overlap features — jaccard being a plain set-overlap between the candidate's grounded content and that of already-failed candidates, a coarse "similar to what already failed" signal that coverage/waste refine. _(The feature block has a fourth column, `dead`, zeroed in the deployed configuration.)_ **The ranking is produced entirely by the network** — at inference nothing outside it reorders the pool.

_Sources: `model.py` (`RecordEncoder`, `EvidenceCrossAttentionScorer`, head), `unified_evidence.py`, wired at `dataset.py`; overlap block `cand_overlap`._

### M4 — Training

- **Dataset gathering.** For each training problem we enumerate its candidate pool and refine **every** skeleton — not stopping at the first success (_non-short-circuiting_) — with a **stable per-skeleton seed** so labels are reproducible, and we store one `EpisodeRecord` per problem (the pool, each skeleton's success/fail outcome, the observed failure evidence, and the scene geometry). Problems with no feasible skeleton are discarded.
    - **Splits:** $400$ train / $100$ val / $100$ test per environment.
    - **Stratification by difficulty:** each split is balanced across difficulty strata (DD2D: minimum number of objects that must be staged, $s0$–$s3$; SB2D: number of buttons, $b1/b2/b3/b5$), placed in disjoint seed bands. _(Because strata are contiguous bands, any subset must be taken by_ striding _the data, never truncating — a prefix would be all-easy.)_
- **Vocabulary and canonicalization.** Operator/predicate/type vocabularies are built from the **training split only** (reserving id $0$ for padding / out-of-vocabulary), by walking the full STRIPS reconstruction of each skeleton so predicates appearing only in intermediate states are captured. Object identities are canonicalized to typed **tags**; during training only, objects are randomly permuted within their type as **data augmentation**.
- **Forming a training example.** From each episode we build, online (freshly each epoch), a triple $$(\mathcal{R}, \mathrm{SUCC} \cap \mathcal{R}, \mathcal{F})$$ where $\mathcal{F}$ is a sampled **failure context** — (skeleton, evidence) pairs drawn strictly from the episode's failures, never a success — $\mathcal{R}$ is the pool minus the skeletons in $\mathcal{F}$, and $\mathrm{SUCC} \cap \mathcal{R}$ are the still-available successes we want ranked on top. The context is sampled to match deployment: with probability $\approx 0.35$ it is **empty** (deployment always starts at $|\mathcal{F}|=0$), otherwise its size is drawn uniformly in $[1,, \min(8,, |\mathrm{FAIL}|)]$, with additional evidence dropout. _(This simple size-uniform sampler is what the deployed code uses — not the more elaborate mixture some early design notes describe.)_
- **Loss — listwise Plackett–Luce.** We train the scores so that an argmax over the remaining set lands on a success: $$\mathcal{L} = \operatorname{logsumexp}_{\sigma \in \mathcal{R}} \ell(\sigma); -; \operatorname{logsumexp}_{\sigma \in \mathrm{SUCC} \cap \mathcal{R}} \ell(\sigma) ;=; -\log \Pr[\text{argmax picks a success}],$$ where $\ell(\sigma)$ is the candidate's logit. This is the training-time analog of the time-to-first-success objective, which is exactly why it is used instead of pointwise binary classification (BCE): BCE scores each skeleton in isolation and is **not** aligned with "did the _top_ pick succeed." A second copy of the loss is applied **within plan-length buckets** (bucket key: operator count), so the model cannot cheat by using plan length as a proxy for feasibility.
- **Optimization.** AdamW, learning rate $3\times10^{-4}$, cosine schedule with 2 warmup epochs, 30 epochs, batch size 8, dropout $0.1$, weight decay $5\times10^{-4}$, gradient clipping at $1.0$.
- **Model selection (rollout-based, uncensored).** We do **not** select checkpoints by validation loss or AUROC. Instead we run the _actual deployed re-ranking rollout_ on the validation split and select the epoch with the lowest mean failed attempts, **uncensored** (run to pool exhaustion), smoothed over a 5-epoch moving average. The hard-won lesson: a selection metric censored below the tail where models actually differ will rate different models identical — _stable selection curves are not evidence of a good selector._

_Sources: `collect.py`, `strata.py`, `vocab.py`, `canonicalize.py`, `dataset.py` (`sample_context`), `loss.py`, `train.py` (`--select-window 5`); `EpisodeRecord` in `schema.py`._

### M5 — Test-time inference

- **The deployed rollout.** Given a problem's candidate pool and the trained model:
    1. encode the scene from $(x_0, g)$ and run the skeleton encoder $\Phi$ **once** over the whole pool;
    2. score all remaining candidates conditioned on the current failure history $\mathcal{F}$ (empty on the first pass ⇒ a purely static score);
    3. pick the **argmax** among untried candidates and attempt refinement;
    4. if it **succeeds**, stop — the returned $\omega$ completes the executable plan; if it **fails**, add $(\hat{\sigma}_t, e_t)$ to $\mathcal{F}$, recompute the failure records / coverage / waste, and **re-score** the remaining pool;
    5. repeat until success or the pool is exhausted. Only the cheap adaptive half re-runs per attempt; the expensive scene/skeleton encoding is amortized.
- **Per-candidate refinement cap $\tau$ (wall-clock knob).** At deployment each skeleton is refined for at most $\tau$ seconds before moving on (DD2D $\tau = 2\text{ s}$, SB2D $\tau = 10\text{ s}$), so a slow near-feasible trap costs $\tau$ rather than the full budget. This is a test-time accounting choice applied uniformly to every method; the attempts-based metric is reported uncensored.
- **Worked example (DD2D, illustrative).** Target behind blockers $o_1, o_2, o_3$; candidates stage different subsets before retrieving. Attempt 1 is the static pick — say it stages $o_3$ and its retrieval grasp collides, and the refiner names $o_1$ as the blocker. Now coverage/waste re-score the pool: candidates that stage $o_1$ before retrieving gain coverage, candidates that only do unrelated staging gain waste. The re-ranked next pick stages $o_1$; if it in turn reveals $o_2$, the next re-rank prefers staging both — each observed failure moves the policy one rung up the escalation ladder.

_Sources: `inference.py` (deployed rollout, refinement cap)._

### M6 — Porting SPECTRE to a new environment (domain-agnostic contract)

A central design goal is that SPECTRE's failure-record representation is **environment-general**: it names no drawer, button, stick, or shelf. Porting to a new environment requires only two things, and no per-environment predicate, feature, or fact vocabulary.

- **(1) A converter to `EpisodeRecord`.** Everything downstream (vocab, dataset, training, evaluation) consumes only serialized `EpisodeRecord`s: the candidate pool, per-skeleton outcomes, scene geometry, and goal atoms. A new environment supplies a converter from its own data/collector to this schema.
- **(2) Observation-only refiner instrumentation.** At each site where a continuous query fails, the refiner emits _what it already computed_ — never an extra check (an added stream call would shift the labels). This is a hard invariant, verified by differential replay.
- **The domain spec.** A tiny `DomainSpec` lets the loss and features derive `manipulated` objects, `goal_objects`, and the plan-length key **from the operator schema itself** — no hand-written per-object geometry. An environment may declare nothing at all (`EMPTY_SPEC`) and still run the full learned method — SB2D does.
- **Two classes of failure evidence** (the key to generality). Environments differ in _how_ a refinement can fail, and the record schema covers both, with an empty channel provably inert so no consumer branches on the environment:
    - **Class 1 — the refiner names the blockers** (a collision check reports which objects it hit). Example: DD2D. These populate the **culprit** slot directly.
    - **Class 2 — the refiner only reports a state deviation** (the motion executed but the achieved abstract state differs from the plan; the blocker is not named). Example: StickButton2D, whose collision check returns only a boolean. Blame is read from the **collateral** deviation — the raw deviation minus the failed step's own declared effects — so a failure that merely didn't achieve its own effects blames nobody and rides the record token as a burned query instead. This two-class design is exactly what let the _same_ record representation carry from DD2D (class 1) to StickButton2D (class 2) with no change to the model.
- **A third environment — Restock3D (3D).** To test the contract beyond 2D, SPECTRE is being ported to Restock3D, a kinematic-PyBullet shelf-restocking domain where feasibility is _real_ collision (no toy gate): a robot stores floor-staged cubes and tall blocks into a shelf with a tall/bottom and a short/top section. It exercises exactly the two mechanisms above — the same **observation-only** instrumentation and the same **class-1** culprits (here, the section residents an over-packed candidate collides with) — plus the 3D **point-cloud** input (M1), which is what lets the model see the one distinction that decides feasibility: an upright tall block fits the tall section but collides the ceiling of the short one (a height difference a 2D footprint cannot represent). The failure structure that makes ordering matter is a "far-is-harder" reach-over (a nearer object blocks the diagonal approach to a farther one), cleared by a south-to-north store order. _This environment is wired end-to-end and in data collection; it is reported here as a methodological demonstration of the porting contract in 3D — no head-to-head results yet._

_Sources: `docs/porting_guide.md`, `envs/dd2d/spectre_convert.py`, `unified_evidence.py`, `domain.py`, `envs/restock3d/` (`scene_geometry.py`, `instrumented_refiner.py`, `models_v2.py`)._

### M7 — The full SPECTRE algorithm

```
Input:
  problem (x0, g);  candidate pool S = {σ_1, ..., σ_K}  (symbolic planner output)
  trained SPECTRE model  M
  optional refinement cap  τ  (per-candidate seconds; None = uncapped)

State:
  F     ← {}                    # failure history: (skeleton, evidence) pairs
  tried ← {}                    # skeletons attempted so far (ordered)

Procedure:
  E ← encode(x0, g, S)          # scene tokens from (x0, g) + Φ(σ) per candidate — ONCE
  while |tried| < K:
      feats ← evidence features for each σ ∈ S given F
              # F = {}  ⇒  all adaptive features are zero  ⇒  purely STATIC score
      ℓ ← M(E, feats)           # per-candidate feasibility logits
      σ* ← argmax of ℓ over S \ tried        # never retry a candidate
      outcome, y ← refine(σ*; x0, budget capped at τ)   # y = ω on succ, evidence e on fail
      tried ← tried + [σ*]
      if outcome = succ: break               # (and, if capped, within τ seconds)
      F ← F ∪ {(σ*, y)}

Output:
  executable plan          = (σ*, ω)  from the successful final attempt (if any)
  realized attempt order   = tried
  attempts to first success = |tried|        (metric FP = |tried| − 1)
```

- **Reading the algorithm against the thesis.** The first iteration uses only the static representation (all adaptive features are zero). Every iteration _after_ a failure feeds the structured record back in, so the policy is a genuine **re-ranker**, not a fixed order — which is precisely where its advantage over static and coarse-adaptive methods comes from.

_Sources: `inference.py` (deployed rollout), `model.py` (forward), `train.py` (selection)._