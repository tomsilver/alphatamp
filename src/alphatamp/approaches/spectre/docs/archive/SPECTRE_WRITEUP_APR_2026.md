> **STATUS: FROZEN 2026-06-25.** Reflects the April 2026 adaptive-reordering
> framing. Superseded by the representation-first direction in
> [`../proposal.md`](../proposal.md) §0 / [`../decisions.md`](../decisions.md)
> 2026-06-25. Retained as historical record — **do not edit the body.**

# Introduction

One long-standing goal of robotics is to build agents that can solve
tasks requiring both abstract decision-making and precise physical
interaction with the world. Achieving this goal requires reasoning
jointly over two very different spaces: a discrete, symbolic space of
high-level decisions (e.g., which object to pick up, in what order to
visit a set of locations), and a continuous space of grasps,
trajectories, and contact-rich motions. Task and motion planning (TAMP)
addresses this challenge by coupling a symbolic task planner with a
continuous motion planner, and the bilevel planning
framework [@garrettIntegratedTaskMotion2021; @silverPredicateInventionBilevel2025]
has emerged as a standard recipe: a symbolic planner proposes
goal-achieving *skeletons* (sequences of grounded operators and abstract
states), and a continuous *refiner* attempts to instantiate each
skeleton with concrete controller parameters until one succeeds.

The two halves of this loop generally have very asymmetric costs.
Generating a skeleton from a symbolic planner is typically inexpensive
(milliseconds on contemporary PDDL solvers) and the planner can often
enumerate many goal-achieving skeletons cheaply. Refinement is the
bottleneck: each attempt invokes continuous samplers, motion planners,
etc., and a single failed refinement can cost seconds to minutes. A
planning episode may involve attempting many skeletons before finding
one that refines, and the cumulative wall-clock cost is dominated almost
entirely by these refinement calls. Because of this asymmetry, the order
in which the candidate pool is attempted is a key determinant of total
planning time: finding a refinable skeleton on the first attempt rather
than the fifth can mean the difference between a one-second planning
episode and a one-minute one. However, the symbolic planner cannot solve
this problem on its own, since the failure modes that distinguish
refinable from unrefinable skeletons are continuous-geometric and not
visible at the symbolic level.

A natural response is to learn, from experience, which skeletons are
likely to refine. Existing approaches to this problem fall into two
broad camps. *Static rankers* such as
PIGINet [@yangPIGINetSequenceBasedPlan2023] score every candidate once,
before any refinement is attempted, and commit to that ordering for the
rest of the episode. These methods learn powerful skeleton
representations but do not update their beliefs as the episode unfolds,
so a single early failure that should rule out an entire family of
related skeletons is wasted information. *Adaptive* approaches, such as
LAZY [@khodeirLearningSearchTask2023], do update the queue as samples
arrive, but they rely on tabular co-failure statistics that do not
necessarily generalize across skeletons. Neither camp exploits the
central observation that motivates this work: when a refinement attempt
fails, the *structure* of the failed skeleton is informative about which
other candidates in the pool are likely to share similar behavior.

We present [Spectre]{.smallcaps} (**S**keleton-**P**ool **E**mbedding
with **C**ontextual **T**ransformer for **Re**ordering), a learned
selection policy that closes this gap. [Spectre]{.smallcaps} encodes
each candidate skeleton with a shared Transformer encoder, summarises
the set of previously observed failures with a permutation-invariant Set
Transformer [@leeSetTransformerFramework2019], and combines the two into
a context-conditioned ranker that re-scores the remaining pool after
every failed attempt. Crucially, [Spectre]{.smallcaps} is a drop-in
addition to existing bilevel planners: the symbolic planner and the
refiner are unmodified, and the test-time interface is a single
$\arg\max$ over the remaining set per attempt.

To evaluate adaptive reordering, we introduce
[RoutedTransport]{.smallcaps}, a mobile-manipulation domain explicitly
constructed to require within-episode adaptation: each instance carries
a discrete latent that gates skeleton refinability, no single
"catch-all" skeleton solves the instance distribution, and the candidate
space is large enough to defeat pure memoisation. On this environment,
[Spectre]{.smallcaps} reduces the number of refinement attempts to first
success by $41$--$62\%$ and wall-clock refinement time by $36$--$57\%$
over a no-learning baseline and two statistics-based memoisation
baselines.

In summary, our contributions are:

- We formalise *adaptive skeleton reordering* as a sequential decision
  problem layered on top of standard bilevel TAMP, making the failure
  context a first-class input to the selection policy
  (Section [3](#sec:problem){reference-type="ref"
  reference="sec:problem"}).

- We propose [Spectre]{.smallcaps}, a Transformer-based selection policy
  that conditions on the failure history via a permutation-invariant Set
  Transformer, trained with a listwise loss aligned with the
  time-to-first-success objective and deployable as a drop-in
  replacement for any static skeleton ranker
  (Section [4](#sec:method){reference-type="ref"
  reference="sec:method"}).

- We introduce [RoutedTransport]{.smallcaps}, a TAMP environment
  designed to expose latent structure that only adaptive policies can
  exploit, and demonstrate that [Spectre]{.smallcaps} substantially
  outperforms both non-learned and statistics-based adaptive baselines
  on it (Section [5](#sec:experiments){reference-type="ref"
  reference="sec:experiments"}).

# Related Work

**Task and Motion Planning.** Task and Motion Planning (TAMP) addresses
the challenge of controlling robotics that have to reason both over
abstract, symbolic decisions and the continuous physical world. Task
planning ([@fikesStripsNewApproach1971]) finds a discrete sequence of
actions, each described by preconditions and effects over symbolic
state, that transitions a robot from an initial configuration to a goal
state. Motion planning
([@lozano-perezAlgorithmPlanningCollisionfree1979];
[@karamanSamplingbasedAlgorithmsOptimal2011];
[@kavrakiProbabilisticRoadmapsPath1996]) aims to generate collision-free
continuous trajectories through the robot's higher dimensional state
space. These two problems are tightly coupled, and cannot be solved in
isolation; the feasibility of a symbolic action sequence depends
entirely on whether collision-free motions exist to realize it, and the
precise motions taken depends in turn on which high level actions are
chosen. The TAMP bilevel planning framework
([@garrettIntegratedTaskMotion2021]) splits robotic planning into 2
stages: high level abstract planning to produce symbolic skeletons and
low level continuous parameter refinement. Generally, the symbolic
planner will produce multiple candidates, and the problem of which one
to attempt first therefore becomes a secondary optimization problem. Our
approach specifically learns to adaptively reorder these candidates
based on the previous failures during test-time. Foundational frameworks
like PDDLStream ([@garrettPDDLStreamIntegratingSymbolic2020a]) integrate
symbolic PDDL planners with blackbox continuous samplers, and remain a
standard substrate for TAMP research.

**Learning for TAMP.** SAHS ([@RepresentationLearningPlanning]) is a
method for learning to rank abstract actions at the discrete task level,
acting as a learned heuristic for the planner's search queue to select
the next best move in geometric TAMP. Other learned approaches
([@xuAcceleratingIntegratedTask2022];
[@silverPlanningLearnedObject2021]) attempt to reduce planning in TAMP
by predicting the feasibility of actions, or learning how to identify
the most relevant objects. PIGINet ([@yangPIGINetSequenceBasedPlan2023])
is one of the most promising state of the art methods, using a
transformer over a given task plan, initial literals, and goal literals
to score a diverse set of candidate plans based on predicted
feasibility, cutting planning time by $10 \textendash80\%$. Its key
limitation, directly motivating SPECTRE, is that PIGINet is a static
ranker; it scores each skeleton once, in isolation, before any
refinement is attempted, and it cannot update its beliefs when it learns
about skeletons that fail the refinement process. In contrast, our
method aims to adapt online to specific problem instances by learning
from refinement failures. LAZY ([@khodeirLearningSearchTask2023]) is one
of the closest existing works to the online skeleton adaptation problem;
it reorders the skeleton queue lazily using geometric motion samples as
they arrive. However, their mechanism is based on frequency counts of
success and failures, and does not generalize structural failure
patterns.

# Problem Formulation {#sec:problem}

Following previous works in TAMP ([@garrettIntegratedTaskMotion2021];
[@liuSLAPShortcutLearning2026]), we develop our approach in
fully-observable and deterministic environments with continuous states
and actions. Given a state $x \in \mathcal{X}$ and an action
$u \in \mathcal{U}$, the next state is determined by a known transition
function $f : \mathcal{X} \times \mathcal{U} \to \mathcal{X}$ (e.g., a
physics simulator). A problem instance is a pair $(x_0, g)$ where
$x_0 \in \mathcal{X}$ is an initial state and $g \subseteq \mathcal{X}$
is a goal. We assume a distribution $\mathcal{D}$ over problem instances
and access to a set of training instances drawn from $\mathcal{D}$;
methods are evaluated on held-out instances from the same distribution.

#### Bilevel TAMP and skeleton refinement.

We adopt the standard bilevel decomposition of TAMP
([@garrettIntegratedTaskMotion2021];
[@silverPredicateInventionBilevel2025]). We assume a relational state
abstraction $\mathrm{abs} : \mathcal{X} \to
\mathcal{S}$, where each $s \in \mathcal{S}$ is a set of ground atoms
over typed objects. A symbolic planner produces a *skeleton*
$$\begin{equation}
    \sigma \;=\; \big(\, s_0,\; a_1, s_1,\; \ldots,\; a_L, s_L \,\big),
\end{equation}$$ a sequence of grounded operators $a_1, \ldots, a_L$
together with the abstract states they induce under STRIPS semantics,
where $s_0 = \mathrm{abs}(x_0)$ and $s_L$ entails $g$. Each operator
$a_t$ has a continuous parameter domain $\Omega_t$ (encoding e.g. grasps
and placement poses); a complete assignment
$\omega = (\omega_1, \ldots, \omega_L)$ grounds each $a_t$ into a
controller whose sequential execution under $f$ yields realized states
$x_1, \ldots, x_L$. *Refinement* searches for an $\omega$ such that
$\mathrm{abs}(x_t) = s_t$ for all $t$, using a sampler $\eta$ within a
fixed budget; if none is found, the skeleton is *infeasible* and the
next one is attempted.

#### Candidate pool and adaptive reordering.

For a problem $(x_0, g)$, the symbolic planner returns a *candidate
pool* $\mathcal{S} = \{\sigma_1, \ldots, \sigma_K\}$ of $K$
goal-achieving skeletons in some order (e.g., increasing heuristic
cost). A planning episode proceeds as a sequence of refinement attempts.
At step $t \in \{1, 2, \ldots\}$, the agent has observed a *failure
history* $\mathcal{F}_{t-1} \subseteq \mathcal{S}$ of previously
attempted skeletons that all failed refinement (a successful refinement
terminates the episode) and selects the next skeleton from the
*remaining set*
$\mathcal{R}_t = \mathcal{S} \setminus \mathcal{F}_{t-1}$. A selection
policy is therefore a mapping $$\begin{equation}
    \pi : \big(\mathcal{S},\, \mathcal{F}_{t-1}\big)
    \;\longmapsto\; \sigma_t \in \mathcal{R}_t .
\end{equation}$$ Note that $\pi$ does *not* condition on the concrete
initial state $x_0$; the abstract initial state
$s_0 = \mathrm{abs}(x_0)$ and the goal $g$ are still available
implicitly, since each skeleton $\sigma \in \mathcal{S}$ encodes both as
the first and last elements of its abstract-state sequence. Conditioning
on $x_0$ would be a strict generalization, and is a natural extension;
we purposefully exclude it in order to isolate the contribution of the
failure history $\mathcal{F}_{t-1}$ and to understand the usefulness of
learning from symbolic structure alone. Static rankers
([@yangPIGINetSequenceBasedPlan2023]) correspond to the further
restriction in which $\pi$ depends on $\mathcal{S}$ alone and ignores
$\mathcal{F}_{t-1}$; we are interested in the strictly more general
*adaptive* setting.

#### Objective.

A policy $\pi$ induces a sequence of skeleton selections via the
recursion $$\begin{equation}
    \hat{\sigma}_t \;=\; \pi(\mathcal{S},\, \mathcal{F}_{t-1}),
    \qquad
    \mathcal{F}_t \;=\; \mathcal{F}_{t-1} \cup \{\hat{\sigma}_t\},
    \qquad \mathcal{F}_0 = \emptyset,
\end{equation}$$ continued until a $\hat{\sigma}_t$ succeeds under
refinement (at which point the episode terminates) or until
$\mathcal{F}_t = \mathcal{S}$. The *attempts to first successful
refinement* under $\pi$ is $$\begin{equation}
\label{objective}
    \mathrm{Attempts}(\pi;\, x_0, g) \;=\; \inf\big\{\, t \geq 1 \;:\;
    \mathrm{refine}(\hat{\sigma}_t) = \mathtt{success} \,\big\}
\end{equation}$$

and the *time to first successful refinement* is:

$$\begin{equation}
T(\pi;\, x_0, g) = \sum_{i = 1}^{\mathrm{Attempts}(\pi;\,x_0, g)} \mathrm{refinementTime(\hat{\sigma}_t)}
\end{equation}$$

Our objective is to learn a policy that minimizes the expected time to
first successful refinement over the task distribution:
$$\begin{equation}
    \pi^{\star} \;=\; \arg\min_{\pi} \;
    \mathbb{E}_{(x_0, g) \sim \mathcal{D}}\!\big[\, T(\pi;\, x_0, g) \,\big].
    \label{eq:objective}
\end{equation}$$

# Method {#sec:method}

We propose [Spectre]{.smallcaps} (*Skeleton-Pool Embedding with
Contextual Transformer for REordering*), a learned selection policy
$\pi$ for the adaptive-reordering setting of
Section [3](#sec:problem){reference-type="ref" reference="sec:problem"}.
After every refinement attempt, [Spectre]{.smallcaps} conditions on the
failure history $\mathcal{F}_{t-1}$ and re-ranks the remaining set
$\mathcal{R}_t$, training end-to-end against a listwise loss aligned
with the time-to-first-success objective in
Eq. ([\[eq:objective\]](#eq:objective){reference-type="ref"
reference="eq:objective"}).

## Architecture {#sec:method:arch}

[Spectre]{.smallcaps} factorises $\pi$ into three trainable modules
(Fig. [1](#fig:architecture_diagram){reference-type="ref"
reference="fig:architecture_diagram"}): a *skeleton encoder*
$\Phi : \sigma \mapsto e(\sigma) \in \mathbb{R}^{d}$, a *context
encoder* $\Psi : \mathcal{F}_{t-1} \mapsto c_t \in \mathbb{R}^{d}$, and
a *scorer* $\rho$ that maps a candidate-context pair to a scalar logit
via a 2-layer MLP on $[e(\sigma);\, c_t]$. The selection at step $t$ is
$$\begin{equation}
\pi(\mathcal{S},\, \mathcal{F}_{t-1})
\;=\; \arg\max_{\sigma \in \mathcal{R}_t}\,
\rho\bigl(e(\sigma),\, c_t\bigr).
\label{eq:argmax}
\end{equation}$$ We use $d = 64$ throughout and four-head attention.
Note that $\Phi$ does not depend on $\mathcal{F}_{t-1}$, so it runs
*once per episode* on the entire candidate pool; per-step inference is
then a single $\Psi$ forward and a broadcast scorer over
$\mathcal{R}_t$. $\Psi$ is intentionally constructed as
permutation-invariant, matching the intuition that the order in which
skeletons fail carries no signal beyond the identity of the failed set.

![The three trainable modules of [Spectre]{.smallcaps}: a per-skeleton
encoder $\Phi$ (run once per episode), a permutation-invariant context
encoder $\Psi$ over the failed set $\mathcal{F}_{t-1}$, and a scorer
$\rho$ that produces ranking logits over
$\mathcal{R}_t$.](./iclr2026/imgs/SPECTRE.pdf){#fig:architecture_diagram
width="100%"}

## Skeleton encoder $\Phi$ {#sec:method:phi}

A skeleton $\sigma = (s_0, a_1, s_1, \dots, a_L, s_L)$ is encoded by
concatenating operator tokens with relational-state tokens and
processing the result with a small Transformer encoder. We first
canonicalise $\sigma$, replacing each concrete object name by a typed
local id (e.g. `Item:1`); two skeletons that differ only by an object
renumbering then produce identical inputs.

#### Operator tokens.

Each grounded operator $a_l = (o, (\alpha_1, \dots, \alpha_A))$ is
encoded by concatenating an embedding of the lifted operator $o$,
slot-specific embeddings of each argument's typed local id, and a
position embedding for $l$, then projecting through a 2-layer MLP. Slot
specificity preserves argument order; for instance Pick(robot, item)
$\neq$ Pick(item, robot).

#### State tokens.

Each abstract state $s_l$ is a set of ground atoms over typed objects.
We embed each atom from its predicate id and slot-specific argument
embeddings, pool the atom set with a Set Attention Block (SAB) followed
by a Pooling-by-Multihead-Attention (PMA)
([@leeSetTransformerFramework2019]), and concatenate the pooled vector
with a per-type object-count histogram before projecting to
$\mathbb{R}^{d}$.

#### Sequence aggregation.

The input to the sequence-level encoder is the sequence
$[\mathrm{tok}(s_0),\, \mathrm{tok}(a_1),\, \dots,\,
  \mathrm{tok}(a_L),\, \mathrm{tok}(s_L)]$. A learned token-type
embedding distinguishes `STATE_0`, `OP`, and `STATE_L` positions, with
sequence-position embeddings added on top. A 2-layer Transformer encoder
processes the sequence and we mean-pool the result to obtain
$e(\sigma) \in \mathbb{R}^{d}$. We encode only the endpoints $s_0, s_L$
rather than every intermediate $s_l$: under STRIPS semantics the
intermediate states are determined by $s_0$ and $a_{1:L}$, so these
intermediate states are redundant, carrying no information not already
in $s_0$ and the operator sequence.

## Context encoder $\Psi$ {#sec:method:psi}

$\Psi$ summarises the failure history $\mathcal{F}_{t-1}$ into a single
vector $c_t \in \mathbb{R}^{d}$, with permutation invariance over the
unordered failed set. We use a two-layer Set Transformer
([@leeSetTransformerFramework2019]): $$\begin{equation}
c_t \;=\; \mathrm{Linear}\Bigl(
  \mathrm{PMA}_1\bigl(
    \mathrm{SAB}\bigl(
      \mathrm{SAB}\bigl(
        \{e(\sigma') : \sigma' \in \mathcal{F}_{t-1}\}
      \bigr)
    \bigr)
  \bigr)
\Bigr).
\label{eq:psi}
\end{equation}$$ The empty-set case $\mathcal{F}_{t-1} = \emptyset$ at
$t = 1$ is handled by replacing the output with a learned vector
$c_{\emptyset} \in \mathbb{R}^{d}$, so that [Spectre]{.smallcaps}
defines a non-trivial first-step ranking via $\Phi$ alone.

## Training {#sec:method:training}

We gather an offline dataset of 500 training, 100 validation, and 100
held-out test problems. For each problem the symbolic planner's candidate
pool is capped at 30 skeletons, and every pooled skeleton is refined
(non-short-circuiting) to annotate its outcome; evaluation rollouts use an
attempt budget of 30 — equal to the candidate-pool cap, so the budget never
binds and the reported attempt counts are uncensored.

#### Training Objective.

We optimise the listwise Plackett--Luce loss
([@xiaPlackettLuceModelLearningtorank2019]) restricted to the remaining
pool. For a training example with remaining set $\mathcal{R}$, in-pool
successful skeletons $\mathcal{R}^{+} = \{\sigma \in \mathcal{R} :
\mathrm{refine}(\sigma) = \mathtt{success}\}$, and conditioning failure
subset $F$, $$\begin{equation}
\mathcal{L}(\mathcal{R}, \mathcal{R}^{+}, F)
\;=\; -\log
\frac{\sum_{\sigma \in \mathcal{R}^{+}}
      \exp \rho\bigl(e(\sigma),\, \Psi(F)\bigr)}
     {\sum_{\sigma \in \mathcal{R}}
      \exp \rho\bigl(e(\sigma),\, \Psi(F)\bigr)}.
\label{eq:pl-loss}
\end{equation}$$ This is the negative log-probability that PL sampling
selects any successful skeleton. Optimizing this objective rewards
ranking successful skeletons higher, and penalizes failed skeletons
having large logits relative to the successful ones.

#### Rollout-aligned $F$-subset sampling.

For each training problem $e$ we record the set
$\mathrm{FAIL}_e \subseteq \mathcal{S}_e$ of skeletons on which the
refiner fails. A training example is then an instance of
[\[eq:pl-loss\]](#eq:pl-loss){reference-type="eqref"
reference="eq:pl-loss"} with $F \subseteq \mathrm{FAIL}_e$. Naively
sampling $F$ by taking a binomial distribution over elements of
$\mathrm{FAIL}_e$ concrentrates $|F|$ near $|\mathrm{FAIL}_e|/2$, and
uniform sampling oversamples the right tail of the distribution. This
fails to align with the test-time visit distribution, which is generally
heavily biased towards small $|F|$ (especially if we aim to find a
success refinement as soon as possible).

We instead draw $|F|$ from a three-component mixture, to ensure coverage
of various sizes of $|F|$, while ensuring that the distribution of $|F|$
seen in training is reflective to what we hope to see at test-time:

$$\begin{equation}
P(|F| = k)
\;=\; w_{\mathrm{u}}\!\cdot\!\mathrm{Bin}\bigl(k;\, n,\, \tfrac{1}{2}\bigr)
\;+\; w_{\mathrm{s}}\!\cdot\!\mathrm{Unif}_{\{0,\dots,n\}}(k)
\;+\; w_{\ell}\!\cdot\!\mathrm{LogN}_{n}(k;\, \mu,\, \sigma_{\!\ell}),
\label{eq:fsampling}
\end{equation}$$ with $n = |\mathrm{FAIL}_e|$, weights
$(w_{\mathrm{u}}, w_{\mathrm{s}}, w_{\ell}) = (0.25, 0.25, 0.5)$. Here,
$\mathrm{LogN}_n$ is a clipped, rounded log-normal with
$(\mu, \sigma_{\!\ell}) = (0, 1)$, $\mathrm{Bin}$ signifies a binomial
distribution, and $\mathrm{Unif}$ signifies a uniform distribution over
$\{0\dots n\}$. The log-normal component is the only one whose marginal
shape accurately reflects the geometric-decay test-time visit
distribution and carries half the mass for that reason; the two other
components preserve coverage of larger-$|F|$ regimes that arise on hard
problems[^2].

## Test-time inference {#sec:method:inference}

At test time, $\Phi$ is applied once to the candidate pool $\mathcal{S}$
to obtain $\{e(\sigma) : \sigma \in \mathcal{S}\}$. Per attempt step,
$\Psi$ is invoked on the failed-set embedding cache, and $\rho$
broadcasts over $\mathcal{R}_t$.
Algorithm [\[alg:rollout\]](#alg:rollout){reference-type="ref"
reference="alg:rollout"} summarises the loop, which realises $\pi$ via
Eq. ([\[eq:argmax\]](#eq:argmax){reference-type="ref"
reference="eq:argmax"}). [Spectre]{.smallcaps} therefore works as a
drop-in method on top of existing TAMP frameworks -- the symbolic
planner, the abstraction $\mathrm{abs}$, and the refiner are untouched.

:::: algorithm
::: algorithmic
**Input:** candidate pool $\mathcal{S}$, attempt budget $T_{\max}$.
$\{e(\sigma)\}_{\sigma \in \mathcal{S}} \leftarrow \Phi(\mathcal{S})$
$\triangleright$ one batched forward pass
$\mathcal{F} \leftarrow \emptyset$, $\mathcal{R} \leftarrow \mathcal{S}$
$c_t \leftarrow \Psi\bigl(\{e(\sigma') : \sigma' \in \mathcal{F}\}\bigr)$
$\triangleright$ returns $c_{\emptyset}$ if $\mathcal{F} = \emptyset$
$\sigma_t \leftarrow
           \arg\max_{\sigma \in \mathcal{R}}\,
           \rho\bigl(e(\sigma),\, c_t\bigr)$
$\mathit{outcome} \leftarrow \mathrm{refine}(\sigma_t)$ **return** $t$
$\mathcal{F} \leftarrow \mathcal{F} \cup \{\sigma_t\}$,
$\mathcal{R} \leftarrow \mathcal{R} \setminus \{\sigma_t\}$ **return**
$\mathtt{failure}$
:::
::::

# Experiments {#sec:experiments}

We evaluate [Spectre]{.smallcaps} on a TAMP environment designed to
require adaptive reordering, comparing against a no-learning planner and
two statistics-based memoization baselines.

#### Environment.

We introduce [RoutedTransport]{.smallcaps}, a mobile-manipulation
environment in which a robot must transport $N$ items from per-item
source zones to per-item target zones. The workspace is partitioned into
6 zones connected by 9 passages; the zone-passage incidence graph is the
complete bipartite graph $K_{3,3}$, with passages partitioned into 3
colour classes. The environment is constructed so as to satisfy several
key properties that some prior 2D TAMP benchmarks
([@KinDERPhysicalReasoning]) do not, each motivated by a failure mode of
prior memoization-style baselines:

- **No universal solution.** Benchmarks such as Obstruction2D and
  ClutteredStorage2D ([@KinDERPhysicalReasoning]) admit a small number
  of "catch-all" skeletons that solve nearly every instance: in
  ClutteredStorage2D, for example, the skeleton that first clears all
  shelf positions and then re-places the target succeeds on essentially
  any initial configuration provided the continuous samplers are
  adequate. On such environments adaptive reordering offers no headroom
  over a static lookup of frequently-successful skeletons.
  [RoutedTransport]{.smallcaps} is constructed so that no single
  skeleton covers the instance distribution.

- **Per-instance latent structure.** Each instance carries a two-axis
  discrete latent that gates skeleton refinability:
  $\mathrm{BlockedColor} \in \{1, 2, 3\}$ disables one passage colour
  class for the instance (modelling, e.g., a passage out of service on
  the day), and $\mathrm{BlockedGrasp} \in \{\mathrm{top},
  \mathrm{side}\}$ disables one grasp mode. The product space yields six
  latent modes, each inducing a distinct subset of refinable skeletons.
  The mode distribution is non-uniform, penalising any policy that
  selects skeletons greedily by marginal training-set success rate.

- **Large skeleton space.** The number of distinct goal-achieving
  skeletons grows combinatorially with $N$, so the training and test
  skeleton supports are only partially overlapping and pure memoization
  is structurally limited.

#### Baselines.

We compare [Spectre]{.smallcaps} against three baselines:

- [Pure Planning]{.smallcaps}: standard bilevel planning with no learned
  component; skeletons are attempted in the symbolic planner's
  cost-ordered sequence.

- [Static Historical]{.smallcaps}: a static lookup baseline that
  estimates each skeleton's marginal success rate
  $\hat{p}(\sigma) = (\#\,\text{successes}) / (\#\,\text{appearances})$
  from the training data, with Laplace smoothing (default
  $\hat{p} = \tfrac{1}{2}$ for unseen skeletons), and ranks
  $\mathcal{R}_t$ by $\hat{p}$ at every step.

- [Adaptive Historical]{.smallcaps}: the strongest baseline. It
  estimates the conditional frequency
  $\hat{p}(\sigma \mid \mathcal{F}_{t-1}) = \mathbb{P}\!\left[
  \mathrm{refine}(\sigma) = \mathtt{success} \,\big|\,
  \mathcal{F}_{t-1}\right]$ and ranks $\mathcal{R}_t$ accordingly. This
  baseline captures all signal extractable from raw counts of
  skeleton-level co-failure patterns, isolating the contribution of
  [Spectre]{.smallcaps}'s representation learning over and above tabular
  conditioning.

#### Protocol.

We evaluate on a held-out test bed of $100$ unseen problems drawn from
the same distribution as the training set. For each method and each
instance we run the rollout of
Algorithm [\[alg:rollout\]](#alg:rollout){reference-type="ref"
reference="alg:rollout"} until first successful refinement and record
(i) the number of attempts $T$ to first success and (ii) the cumulative
refinement wall-clock time.

#### Headline results.

Figure [2](#fig:table_comparison){reference-type="ref"
reference="fig:table_comparison"} reports means and standard deviations
of both metrics. [Spectre]{.smallcaps} attains the lowest mean attempts
and the lowest mean refinement time of any method evaluated; the
strongest baseline, [Adaptive Historical]{.smallcaps}, requires $57.3\%$
more refinement time on average. Since [Adaptive Historical]{.smallcaps}
already conditions on $\mathcal{F}_{t-1}$, the gap is attributable is
not solely to the use of the failure context, but specifically to
[Spectre]{.smallcaps}'s ability to generalize across skeletons via its
learned representation, a capability that count-based conditioning
cannot match on a combinatorially large skeleton space with limited
skeleton coverage.

<figure id="fig:table_comparison" data-latex-placement="t">
<embed src="./iclr2026/imgs/method_comparison_table.pdf"
style="width:100.0%" />
<figcaption>Attempts and refinement time to first successful refinement
on <span class="smallcaps">RoutedTransport</span> (mean <span
class="math inline">±</span> standard deviation over <span
class="math inline">100</span> held-out instances). <span
class="smallcaps">Spectre</span> reduces both quantities relative to all
baselines; the strongest baseline (<span class="smallcaps">Adaptive
Historical</span>) requires <span class="math inline">57.3%</span> more
refinement time on average.</figcaption>
</figure>

#### Distributional analysis.

Figures [3](#fig:attempts_dist){reference-type="ref"
reference="fig:attempts_dist"}
and [4](#fig:refinement_time_dist){reference-type="ref"
reference="fig:refinement_time_dist"} show the per-instance
distributions over the test set. Moving from [Pure Planning]{.smallcaps}
to the memoization baselines to [Spectre]{.smallcaps}, the distributions
concentrate progressively at small values and exhibit thinner upper
tails. The reduction in tail mass is especially notable, as the tail end
represents problems that take significantly more attempts and total
time.

<figure id="fig:attempts_dist" data-latex-placement="t">
<embed src="./iclr2026/imgs/attempts_dist.pdf" style="width:100.0%" />
<figcaption>Distribution of attempts to first successful refinement, by
method. Distributions concentrate at lower values and the upper tail
thins as we move from <span class="smallcaps">Pure Planning</span>
through the memoization baselines to <span
class="smallcaps">Spectre</span>.</figcaption>
</figure>

<figure id="fig:refinement_time_dist" data-latex-placement="t">
<embed src="./iclr2026/imgs/refinement_time_dist.pdf"
style="width:100.0%" />
<figcaption>Distribution of refinement wall-clock time to first
successful refinement, by method. <span class="smallcaps">Spectre</span>
both concentrates the bulk of the distribution at lower times and
reduces upper-tail mass relative to all baselines.</figcaption>
</figure>

#### Cumulative success curves.

Figure [5](#fig:cumulative_success){reference-type="ref"
reference="fig:cumulative_success"} reports the success-at-$K$ curve:
the fraction of test instances solved within the first $K$ refinement
attempts. [Spectre]{.smallcaps} reaches most given coverage levels in
roughly half as many attempts as the best baseline; for example,
[Spectre]{.smallcaps} solves $\sim\!80\%$ of instances within $9$
attempts, while [Adaptive Historical]{.smallcaps} requires $\sim\!18$ to
reach the same level.

<figure id="fig:cumulative_success" data-latex-placement="t">
<embed src="./iclr2026/imgs/cumulative_success_rate.pdf"
style="width:100.0%" />
<figcaption>Cumulative success rate as a function of attempt budget
<span class="math inline"><em>K</em></span> on <span
class="smallcaps">RoutedTransport</span>. <span
class="smallcaps">Spectre</span> reaches each coverage level in roughly
half as many attempts as the strongest baseline.</figcaption>
</figure>

# Discussion and Conclusion {#sec:discussion}

We have presented [Spectre]{.smallcaps}, a learned selection policy for
the skeleton-reordering inner loop of bilevel TAMP.
[Spectre]{.smallcaps} conditions on the failure history
$\mathcal{F}_{t-1}$ via a small Transformer-based skeleton encoder and a
permutation-invariant Set Transformer context encoder, and is trained
with a listwise loss aligned with the time-to-first-success objective.
On [RoutedTransport]{.smallcaps}, even the strongest baseline requires
$57.3\%$ more time relative to [Spectre]{.smallcaps}.

#### Strengths.

The gain over [Adaptive Historical]{.smallcaps} is informative: that
baseline *also* conditions on $\mathcal{F}_{t-1}$, but does so via
tabular conditioning on raw co-failure counts. The remaining gap is
therefore attributable to representation -- [Spectre]{.smallcaps} learns
to generalise across skeletons that share structural features but never
co-occur in any single training rollout, an extrapolation that
count-based methods cannot make on a combinatorially large skeleton
space. One key strength of [SPECTRE]{.smallcaps} is that it is a
*drop-in* replacement for any static skeleton ranker: the symbolic
planner, the abstraction $\mathrm{abs}$, and the refiner are untouched,
and the test-time interface is a single $\arg\max$ over $\mathcal{R}_t$.
Additionally, the rollout-aligned $F$-subset sampling acts as built-in
data augmentation: each training problem with $n = |\mathrm{FAIL}_e|$
failed skeletons admits up to $2^n$ distinct training examples, so
[Spectre]{.smallcaps} extracts more gradient signal per collected
rollout.

#### Limitations.

Three limitations of the present work are worth stating.

*Sample complexity.* The training pipeline requires exhaustive
per-problem failure annotation -- every skeleton in the candidate pool
of every training problem is refined to determine its outcome -- which
is the dominant collection cost. We have not characterised how
[Spectre]{.smallcaps}'s performance scales with training-set size, and
an explicit data-efficiency study (varying $|\mathcal{D}|$ over one to
two orders of magnitude) is one of the most informative experiments we
have yet to run.

*Fixed-dimensional context summary.* The context encoder $\Psi$ pools
$\mathcal{F}_{t-1}$ into a single vector $c_t \in \mathbb{R}^d$. As
$|\mathcal{F}_{t-1}|$ grows, distinct failure patterns must compete for
capacity in a fixed-size summary, and we have not directly verified that
this representation remains sharp at large failure counts. A natural
alternative is to expose the per-failure embeddings to the scorer
directly. We expect this to matter most on long-horizon episodes;
however, we believe we should test this empirically.

*Problem suitability.* Adaptive reordering is only useful when the
candidate pool contains substantively distinct refinement strategies. On
environments such as ClutteredStorage2D, a single "catch-all" skeleton
solves nearly every instance; in that regime, static memoization is
already near-optimal and [Spectre]{.smallcaps} offers no measurable
benefit. [Spectre]{.smallcaps} should be understood as a method for
problems with non-trivial latent variation in refinability, which is the
regime that [RoutedTransport]{.smallcaps} was constructed to expose,
rather than a universal upgrade.

#### Future work.

One immediate extension is to incorporate a learned static ranker,
e.g. a sequence-based feasibility predictor in the style of
PIGINet [@yangPIGINetSequenceBasedPlan2023], as an auxiliary input to
the scorer. Because such predictors condition on the concrete initial
state $x_0$ -- a conditioning channel the present method deliberately
excludes (Section [3](#sec:problem){reference-type="ref"
reference="sec:problem"}) -- this would be a strict generalisation of
the current setup, combining state-grounded priors with the
failure-context conditioning [Spectre]{.smallcaps} already provides.

A second priority is broader empirical coverage. The present results
establish proof-of-concept on a single environment; validating on
additional TAMP domains -- particularly ones with qualitatively
different failure structure than the discrete-latent setup of
[RoutedTransport]{.smallcaps} -- is required before the method can be
claimed as general. We are also yet to test *compositional*
generalisation, i.e. training on one configuration of
$(N, |\text{zones}|, |\text{passages}|)$ and testing on a different one.
The architecture is designed to support this: typed local-id embeddings,
set-based atom pooling, and sequence-level processing over
variable-length skeletons all factor cleanly across object counts.
Whether the trained network actually generalises in this direction,
however, is not something we have shown.

We view [Spectre]{.smallcaps} as a proof of concept that representation
learning over the failure context is a viable and high-leverage addition
to bilevel TAMP. The headline gains over a non-trivial adaptive baseline
that already has access to the same conditioning information indicate
that the structure of skeletons -- not merely their co-failure
statistics -- carries useful signal about which candidate to try next,
and that this signal can be extracted at modest training cost.

# Acknowledgements

Thank you to Prof. Tom Silver and Yixuan Huang for all of your insight,
support, and feedback! I'm grateful to have been able to meet with you
nearly every week, and iterate upon my ideas. Also, thank you to
Skywalker Li and Richard Zhou, the other members of the \"AlphaTAMP\"
team, for your insights during our weekly mini-group meetings.

# Appendix

# The [RoutedTransport]{.smallcaps} environment {#app:routedtransport}

This appendix supplements the body's description of
[RoutedTransport]{.smallcaps}.

## Types, predicates, and operators {#app:rt:pddl}

The typed object set comprises one $\mathtt{Robot}$, $N$
$\mathtt{Item}$s, six $\mathtt{Zone}$s partitioned into $L$- and
$R$-sides, and nine $\mathtt{Passage}$s with three subtypes
$\mathtt{Passage}_A, \mathtt{Passage}_B, \mathtt{Passage}_C$.

Dynamic predicates are
$\mathtt{At}, \mathtt{ItemAt}, \mathtt{HandEmpty}, \mathtt{Holding}$,
and a pair $\mathtt{HeldGraspTop} / \mathtt{HeldGraspSide}$ tracking the
grasp mode currently in use. Static predicates -- which appear in $s_0$
and propagate unchanged through every abstract state along a skeleton --
are $\mathtt{Connects}(\mathtt{Passage}, \mathtt{Zone}, \mathtt{Zone})$,
$\mathtt{PassageWidth}(\mathtt{Passage}, w)$, and
$\mathtt{ItemSize}(\mathtt{Item}, s)$, with
$w \in \{\mathtt{narrow}, \mathtt{medium}, \mathtt{wide}\}$ and
$s \in \{\mathtt{small}, \mathtt{medium}, \mathtt{large}\}$.

The lifted operator set has eight elements:
$\mathtt{Pick}/\mathtt{Place}$ split by grasp mode (4 operators);
$\mathtt{TraverseEmpty}$ generic over passage subtype (1);
$\mathtt{TraverseLoadedColor}_X$ for $X \in \{A, B, C\}$ (3). Pick sets
the corresponding $\mathtt{HeldGrasp\!*}$ atom that the matching
$\mathtt{Place}$ requires, coupling pick and place modes within a
skeleton. Loaded traversal is intentionally *not* gated on the held
grasp at the symbolic level (the chassis moves while the gripper
continues to hold), keeping the operator count to eight.

## Latent, tags, and skeleton families {#app:rt:latent-tags}

Each instance carries a latent
$z = (\mathrm{BlockedColor}, \mathrm{BlockedGrasp}) \in \{A, B, C\}
\times \{\mathtt{top}, \mathtt{side}\}$, sampled from a non-uniform
factorised prior that strongly favours $(A, \mathtt{top})$. The
non-uniformity is intentional: it gives a marginally-greedy ranker a
non-trivial signal to chase, while ensuring that the greedy choice is
wrong on most instances.

Each instance also carries per-passage width tags and per-item size
tags, sampled i.i.d. from non-uniform priors and emitted as
$\mathtt{PassageWidth}$ / $\mathtt{ItemSize}$ atoms in $s_0$. A loaded
traversal of item $i$ through passage $p$ is *size-compatible* iff
$\mathtt{size}(i) \leq \mathtt{width}(p)$ under the orderings
$\mathtt{small} < \mathtt{medium} 
\mathtt{large}$ and $\mathtt{narrow} < \mathtt{medium} 
\mathtt{wide}$.

The combination of (i) every skeleton using exactly two color classes
and (ii) the single-grasp-mode constraint partitions the candidate pool
into six *families*, indexed by the unordered loaded color pair and the
grasp mode. Each family succeeds under exactly one of the six $z$-modes
and fails under the other five. A single failed refinement, in the
absence of tag confounding, would suffice to rule out the failing family
and re-weight a uniform prior over the remaining five. The role of the
per-problem tags is to inject confounding: a failure may be due to
mode-conflict or to a tag-incompatible loaded traversal, and tabular
failure-conditioning over canonical operator-sequence keys cannot
distinguish the two. [Spectre]{.smallcaps}'s skeleton encoder reads the
tag atoms in $s_0$ and can in principle disentangle the two failure
causes.

[^1]: Advised by Prof. Tom Silver

[^2]: We plan to ablate this mixture, but we are using it for now
    because it addresses the intuition behind rollout-alignment.
