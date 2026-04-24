# SPECTRE Training Data EDA Specification

*Companion to `SPECTRE_METHOD_SPEC.md`. Assumes that context.*

---

## 1. Purpose

Before committing full SPECTRE training on a given `kinder` environment, validate on the 500-episode training collection that:

1. Episodes are **structurally valid and non-trivial** — there are successes to find, default order doesn't find them for free, and the symbolic planner produces meaningfully varied candidate pools across problems.
2. **Baseline bracket is well-spread** — a static-historical ranker leaves measurable room above it (relative to oracle) for SPECTRE to fill.
3. **Adaptive signal exists** — outcomes of failed skeletons carry information about remaining skeletons, beyond what per-skeleton marginal success rates capture. Without this, Ψ has nothing to learn.

The tests below are the minimum set to answer these three questions. They are organized into three groups, to be run in order.

---

## 2. Data schema

Per SPEC §5.1, each episode record contains `problem_id`, environment tag, the candidate pool `S_e` (in canonical planner-generation order), per-skeleton success/failure outcomes, per-skeleton refinement wall-clock, and the initial abstract state `s₀`.

**Canonical skeleton key.** Throughout the adaptive-signal and historical-baseline tests, two skeletons are "the same" after applying SPEC §4.1.4 typed-local-id renumbering to the operator sequence. Denote this key `key(s)`. This matches what the historical lookup-table baselines see as equivalent, and matches the equivalence class Φ is implicitly trained against.

**Derived quantities.** For episode `e`: `K_e = |S_e|`; `SUCC_e`, `FAIL_e`, `n_succ_e`; `T_default_e` = 1 + index of first success under the planner's canonical order, or `∞` if no success is in `S_e`. Attempt budget is 20 (SPEC §1.1).

**Exclusion discipline.** Episodes with `n_succ_e = 0` are reported as a count but excluded from time-to-first-success aggregates in Group 2. They are retained for diversity and skeleton-count analyses in Group 1.

**Censoring.** Any simulated traversal that exceeds 20 attempts without success contributes `T = 21` (labeled censored) and wall-clock summed to the 20th attempt. No silent imputation.

---

## 3. Group 1 — Episode sanity

Four cheap structural checks. Run first.

### 3.1 Pool cap confirmation

- **Computation:** fraction of episodes where `K_e` equals the planner's pool cap (e.g. 50).
- **Output:** one scalar per environment.
- **Interpretation:** The planner is expected to saturate the cap on essentially every problem; this confirms that expectation. If the fraction is materially below 1, some problems yielded small pools and those episodes should be inspected before aggregating.

### 3.2 Cross-problem skeleton diversity (headline)

Question: given the cap is always hit, are the ~50 skeletons per problem substantially the same ~50 skeletons across problems, or do they differ?

- **Computation:**
  - `U = |{key(s) : s ∈ S_e, over all training episodes e}|`.
  - `N_slots = Σ_e K_e` (total skeleton occurrences).
  - **Rarefaction curve:** process episodes in random order (average over 10 shuffles); record cumulative `U^{(i)}` after `i` episodes, for `i = 1, …, 500`.
  - **Pool Jaccard histogram:** sample 10,000 random episode pairs `(e, e')` and compute `J = |key(S_e) ∩ key(S_{e'})| / |key(S_e) ∪ key(S_{e'})|`.
- **Output:** scalar `U`, ratio `U / N_slots`, rarefaction curve, Jaccard histogram.
- **Interpretation:**
  - `U` near `N_slots` / Jaccard near 0: every problem gets a problem-specific pool. Φ must generalize via embeddings; historical baselines will be noisy.
  - `U` near 50 / Jaccard near 1: all problems share essentially the same pool. Historical baselines collapse to a global ranking; Φ has little to generalize over; SPECTRE's per-problem adaptivity must come entirely from s₀ conditioning or from within-episode failure context.
  - Rarefaction saturating well before episode 500 indicates the planner has a fixed repertoire it exhausts early — same implication as `U` near 50.

### 3.3 Episode success rate

- **Computation:** fraction of episodes with `n_succ_e ≥ 1`; distribution of `n_succ_e / K_e`.
- **Output:** scalar; histogram.
- **Interpretation:** Episodes with zero successes are unusable for time-to-first-success metrics. The success-fraction histogram distinguishes "mostly easy" environments (oracle ≈ 1, no headroom) from "sparse success" environments (oracle > 1 even with perfect knowledge).

### 3.4 Default-order budget exhaustion

- **Computation:** fraction of episodes with `T_default_e > 20`.
- **Output:** one scalar per environment.
- **Interpretation:** The tail SPECTRE is trying to shrink. Near 0 means default order is already near-optimal and any improvement is capped at small absolute value; near 1 means the budget itself is a binding constraint and the per-skeleton data is heavily right-censored.

---

## 4. Group 2 — Baselines

Five rankers, each producing attempt-count and cumulative-wall-clock histograms over the same set of episodes. All overlaid on two summary charts (one per metric) per environment.

### 4.1 Random floor (B1)

- **Computation:** closed form under uniform random permutation: `E[T_rand_e] = (K_e + 1) / (n_succ_e + 1)` when `n_succ_e ≥ 1`. For wall-clock, Monte-Carlo 100 permutations per episode.
- **Purpose:** Anchors the bottom of the bracket. Any baseline indistinguishable from this is useless.

### 4.2 Default order (B2) — user's existing histogram

- **Computation:** traverse `S_e` in planner-generation order; record attempts and cumulative wall-clock to first success.
- **Purpose:** The baseline SPECTRE must beat at deployment.

### 4.3 Static-historical (B3) — user's existing histogram

- **Computation:** `p̂(k) = (successes + 1) / (appearances + 2)` (Laplace smoothing) from training data. For each episode, sort `S_e` by `p̂(key(s))` descending, ties broken by default order; simulate traversal.
- **Held-out variant:** compute `p̂` on training only, evaluate on validation. Report both; the held-out number is the honest one.
- **Purpose:** The static-ranker baseline that SPECTRE must strictly improve upon to justify its added complexity over PIGINet/HSR.

### 4.4 Adaptive-historical (B4)

- **Computation:**
  - For each ordered pair `(k, k')`, compute `p̂(k | k' failed) = (# episodes where k appeared, k' appeared and failed, k succeeded) / (# episodes where k appeared and k' appeared and failed)`, with add-one smoothing.
  - For each episode, simulate greedy adaptive traversal: start `F = ∅`, `R = S_e`; at each step rank `R` by a Naive-Bayes log-odds combination of marginal `p̂(k)` and pairwise conditionals over `k' ∈ F`; pick top, observe outcome, update `F`, repeat.
- **Purpose:** **The central test of this spec.** The gap B3 − B4 is an empirical lower bound on the adaptivity premium SPECTRE can achieve. See §5.1 for the decision-relevant interpretation.

### 4.5 Oracle ceiling (B5)

- **Computation:** for each episode with `n_succ_e ≥ 1`, attempts-oracle = 1, wall-clock-oracle = `min{refine_times[i] : i ∈ SUCC_e}`.
- **Purpose:** Anchors the top of the bracket. The gap B2 − B5 is the total headroom available to any ranker — upper-bounds the absolute gain SPECTRE can produce.

---

## 5. Group 3 — Adaptive signal diagnostics

Two scalars distilled from Group 2, each with a CI.

### 5.1 Adaptive premium (headline scalar)

- **Computation:** `Δ = mean(B3_T) − mean(B4_T)` on attempts, same on wall-clock. 95% bootstrap CI over episodes (10,000 resamples).
- **Output:** two numbers with CI, per environment.
- **Interpretation:**
  - `Δ > 0`, CI excluding zero: within-episode failure context carries exploitable signal. SPECTRE's Ψ has something to learn. Magnitude of `Δ` is a conservative estimate of the adaptivity premium.
  - `Δ ≈ 0` with CI spanning zero: failure context provides no signal beyond skeleton marginals. **Blocking result** — SPECTRE reduces to a static ranker and should not be preferred over simpler alternatives in this environment.
  - `Δ < 0`: data collection or canonical-key computation is likely buggy (adaptive-historical strictly subsumes static-historical in expectation under correct conditioning). Investigate before proceeding.
- **Caveat:** this is a lower bound. SPECTRE's learned Ψ+Φ can exploit structure a discrete pair-frequency table cannot; the true premium may be larger. `Δ = 0` is disqualifying; positive `Δ` is encouraging, not a guarantee.

### 5.2 Headroom (second scalar)

- **Computation:** `H = mean(B2_T) − mean(B5_T)` on attempts, same on wall-clock, with 95% CI.
- **Output:** two numbers with CI, per environment.
- **Interpretation:** Caps the total possible improvement. An environment with `H < 2` attempts on average is a low-value training target — even a perfect SPECTRE produces only small absolute gains there.

---

## 6. Pass bar for proceeding to SPECTRE training

All five conditions must hold (per environment):

1. 3.1 — pool cap is saturated on ≥ 95% of episodes (confirms planner behavior).
2. 3.2 — rarefaction curve has not saturated by episode 500 **or** `U` is meaningfully larger than the pool cap (e.g., ≥ 4× the pool cap). Calibrate the exact threshold on pilot data; the qualitative requirement is that the canonical-skeleton set is not trivially small.
3. 3.3 — episode success rate ≥ 50%.
4. 3.4 — default-order budget-exhaustion fraction ≥ 10%.
5. 5.1 — adaptive premium `Δ > 0` with 95% CI excluding zero, on the attempts metric.

Secondary (flag but non-blocking):

- 5.2 — headroom `H ≥ 2` attempts. If below, SPECTRE's absolute gain will be small even if relative gain is large; may affect cost-benefit of deployment but not technical feasibility.

Environments failing any primary condition should be documented with the failing metric, its value, and the decision (proceed, reconfigure generator, or drop from benchmark set) recorded explicitly before continuing.
