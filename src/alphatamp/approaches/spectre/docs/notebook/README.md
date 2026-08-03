# SPECTRE — Running EDA / Experiment Log

Append-only lab notebook: dated entries for EDA findings, training runs, ablations, and dead
ends. Keep entries short — what was run, the headline number(s), and the takeaway.
Conclusions that change the method belong in [`proposal.md`](../proposal.md); decisions with
lasting consequences belong in [`decisions/`](../decisions/README.md). Plots/tables generated
by `experiments/spectre/analyze_spectre.py` (a marimo notebook) can be referenced by path.

Entries are grouped into **era chapters** — the same chapters, with the same boundaries, as
the decision log — newest first within each chapter. Everything below the marker is generated
by `experiments/spectre/decisions_index.py index`; edit the chapters, not the tables.

The pre-split single file is frozen at
[`../archive/notebook_2026-07-29_monolithic.md`](../archive/notebook_2026-07-29_monolithic.md)
and every entry in it is preserved byte-for-byte here.

## Format

```
## YYYY-MM-DD — short title
- What: ...
- Result: ...
- Takeaway / next: ...
```

Add one with:

```bash
python experiments/spectre/decisions_index.py new --log notebook --title "..." --tracks method
python experiments/spectre/decisions_index.py index
```

## Reading a retrieved entry

**This log is where retracted numbers live.** Check an entry's status strip before quoting any
figure from it — several entries record results that were later corrected (the dd2d_v3
`13.68`, G6's arm levels, the PIGINet-wins table). The **Do not quote** table below lists them
with their replacements. Historical entries are append-only: to correct one, add a new entry
and mark the old one, never edit it in place.

<!--BEGIN GENERATED-->

## Chapters

| Chapter | Entries | Span | State |
|---|---|---|---|
| [07-stickbutton2d](07-stickbutton2d.md) — StickButton2D as a second environment | 12 | 2026-08-01 .. 2026-08-03 | **open** |
| [06-v3-performance](06-v3-performance.md) — v3 performance push | 12 | 2026-07-27 .. 2026-07-31 | closed |
| [05-v3-migration](05-v3-migration.md) — v3 migration | 8 | 2026-07-26 .. 2026-07-26 | closed |
| [04-comparison](04-comparison.md) — Method comparison and VLMPlan | 10 | 2026-07-23 .. 2026-07-25 | closed |
| [03-dd2d-v2.2](03-dd2d-v2.2.md) — DD2D integration and v2.2 | 18 | 2026-07-12 .. 2026-07-20 | closed |
| [02-pivot](02-pivot.md) — Direction pivot | 1 | 2026-06-25 .. 2026-06-25 | closed |
| [01-foundations](01-foundations.md) — Foundations | 4 | 2026-04-27 .. 2026-06-11 | closed |

## All entries, newest first

| Date | Entry | Tracks | Status |
|---|---|---|---|
| 2026-08-03 | [2026-08-03-vlmplan-frontier-vlm-gpt-5-6-luna-strong](07-stickbutton2d.md#2026-08-03-vlmplan-frontier-vlm-gpt-5-6-luna-strong) | baselines, evaluation, env-stickbutton2d, env-dd2d |  |
| 2026-08-02 | [2026-08-02-dd2d-s1-wall-clock-blow-up-diagnosed-per-candidate](07-stickbutton2d.md#2026-08-02-dd2d-s1-wall-clock-blow-up-diagnosed-per-candidate) | method, evaluation, env-dd2d |  |
| 2026-08-02 | [2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s](07-stickbutton2d.md#2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s) | baselines, env-stickbutton2d, data |  |
| 2026-08-02 | [2026-08-02-dd2d-wall-clock-first-success-fp-flatters](07-stickbutton2d.md#2026-08-02-dd2d-wall-clock-first-success-fp-flatters) | evaluation, env-dd2d, tooling |  |
| 2026-08-02 | [2026-08-02-s2-ood-degradation-pool-composition-artifact-model](07-stickbutton2d.md#2026-08-02-s2-ood-degradation-pool-composition-artifact-model) | env-dd2d, evaluation, method |  |
| 2026-08-01 | [2026-08-01-dd2d-generalization-v3-vs-astar-unseen](07-stickbutton2d.md#2026-08-01-dd2d-generalization-v3-vs-astar-unseen) | env-dd2d, evaluation, method |  |
| 2026-08-01 | [2026-08-01-dd2d-compare-cache-rebuilt-unified-coverage](07-stickbutton2d.md#2026-08-01-dd2d-compare-cache-rebuilt-unified-coverage) | evaluation, env-dd2d |  |
| 2026-08-01 | [2026-08-01-sb2d-vlmplan-32b-capable-tail-limited-beats-astar](07-stickbutton2d.md#2026-08-01-sb2d-vlmplan-32b-capable-tail-limited-beats-astar) | baselines, evaluation, env-stickbutton2d |  |
| 2026-08-01 | [2026-08-01-sb2d-ablation-arms-training-not-reproducible-from-seed](07-stickbutton2d.md#2026-08-01-sb2d-ablation-arms-training-not-reproducible-from-seed) | evaluation, baselines, env-stickbutton2d |  |
| 2026-08-01 | [2026-08-01-sb2d-comparison-v3-piginet-indistinguishable-adaptivity](07-stickbutton2d.md#2026-08-01-sb2d-comparison-v3-piginet-indistinguishable-adaptivity) | baselines, evaluation, method, env-stickbutton2d |  |
| 2026-08-01 | [2026-08-01-sb2d-collection-b1-b5-bracket-v3-1](07-stickbutton2d.md#2026-08-01-sb2d-collection-b1-b5-bracket-v3-1) | method, evaluation, baselines, env-stickbutton2d |  |
| 2026-08-01 | [2026-08-01-stickbutton2d-stood-up-pool-shape-evidence](07-stickbutton2d.md#2026-08-01-stickbutton2d-stood-up-pool-shape-evidence) | method, data, env-stickbutton2d, evaluation |  |
| 2026-07-31 | [2026-07-31-unified-coverage-waste-ab-5-83](06-v3-performance.md#2026-07-31-unified-coverage-waste-ab-5-83) | method, evaluation, env-dd2d |  |
| 2026-07-31 | [2026-07-31-unified-coverage-waste-probes](06-v3-performance.md#2026-07-31-unified-coverage-waste-probes) | method, evaluation, env-stickbutton2d |  |
| 2026-07-30 | [2026-07-30-demotion-cut-authoritative-v3-7-44](06-v3-performance.md#2026-07-30-demotion-cut-authoritative-v3-7-44) | evaluation, method |  |
| 2026-07-30 | [2026-07-30-proof-demotion-priced-0-23-fp-deployed](06-v3-performance.md#2026-07-30-proof-demotion-priced-0-23-fp-deployed) | evaluation, method | **partly superseded** |
| 2026-07-29 | [2026-07-29-stickbutton2d-b5-reaches-75](06-v3-performance.md#2026-07-29-stickbutton2d-b5-reaches-75) | env-stickbutton2d |  |
| 2026-07-28 | [2026-07-28-stickbutton2d-feasibility-b1-b3](06-v3-performance.md#2026-07-28-stickbutton2d-feasibility-b1-b3) | env-stickbutton2d, data | **partly superseded** |
| 2026-07-28 | [2026-07-28-dd2d-comparison-3-seeds](06-v3-performance.md#2026-07-28-dd2d-comparison-3-seeds) | evaluation, baselines |  |
| 2026-07-28 | [2026-07-28-state-delta-ties-6-seeds](06-v3-performance.md#2026-07-28-state-delta-ties-6-seeds) | method, evaluation |  |
| 2026-07-27 | [2026-07-27-comparison-retargeted-two-stale-bugs](06-v3-performance.md#2026-07-27-comparison-retargeted-two-stale-bugs) | tooling, evaluation |  |
| 2026-07-27 | [2026-07-27-p5-observed-coverage-waste](06-v3-performance.md#2026-07-27-p5-observed-coverage-waste) | method, evaluation |  |
| 2026-07-27 | [2026-07-27-p2-missing-g6-cell](06-v3-performance.md#2026-07-27-p2-missing-g6-cell) | method, evaluation |  |
| 2026-07-27 | [2026-07-27-g8-dropping-dead-fixes-s1](06-v3-performance.md#2026-07-27-g8-dropping-dead-fixes-s1) | method |  |
| 2026-07-26 | [2026-07-26-g0-g1-instrumentation-13-68-unreproducible](05-v3-migration.md#2026-07-26-g0-g1-instrumentation-13-68-unreproducible) | method, env-dd2d, data | amended |
| 2026-07-26 | [2026-07-26-g7-p-v3-3-falsified](05-v3-migration.md#2026-07-26-g7-p-v3-3-falsified) | method |  |
| 2026-07-26 | [2026-07-26-g6b-uncensoring-the-selector](05-v3-migration.md#2026-07-26-g6b-uncensoring-the-selector) | evaluation |  |
| 2026-07-26 | [2026-07-26-g6-record-tokens](05-v3-migration.md#2026-07-26-g6-record-tokens) | method, evaluation | **retracted** |
| 2026-07-26 | [2026-07-26-g5-one-failurerecord](05-v3-migration.md#2026-07-26-g5-one-failurerecord) | method |  |
| 2026-07-26 | [2026-07-26-d2-advantage-is-length-calibration](05-v3-migration.md#2026-07-26-d2-advantage-is-length-calibration) | method, evaluation |  |
| 2026-07-26 | [2026-07-26-canonicalize-not-idempotent](05-v3-migration.md#2026-07-26-canonicalize-not-idempotent) | data, tooling |  |
| 2026-07-26 | [2026-07-26-vlmplan-scale-comparison](05-v3-migration.md#2026-07-26-vlmplan-scale-comparison) | baselines |  |
| 2026-07-25 | [2026-07-25-vlmplan-8b-test-split](04-comparison.md#2026-07-25-vlmplan-8b-test-split) | baselines |  |
| 2026-07-25 | [2026-07-25-rejected-reasoning-model-arm](04-comparison.md#2026-07-25-rejected-reasoning-model-arm) | baselines |  |
| 2026-07-25 | [2026-07-25-v3-reversal-was-short-first-prior](04-comparison.md#2026-07-25-v3-reversal-was-short-first-prior) | method, baselines |  |
| 2026-07-24 | [2026-07-24-retrained-all-three-on-v3](04-comparison.md#2026-07-24-retrained-all-three-on-v3) | baselines, evaluation | **superseded** |
| 2026-07-24 | [2026-07-24-vlmplan-baseline-smoke-tested](04-comparison.md#2026-07-24-vlmplan-baseline-smoke-tested) | baselines | **partly superseded** |
| 2026-07-24 | [2026-07-24-post-grasp-sanity-astar](04-comparison.md#2026-07-24-post-grasp-sanity-astar) | baselines, env-dd2d |  |
| 2026-07-24 | [2026-07-24-grasp-reaches-concavities](04-comparison.md#2026-07-24-grasp-reaches-concavities) | env-dd2d |  |
| 2026-07-24 | [2026-07-24-grasp-fixed-contacts-material](04-comparison.md#2026-07-24-grasp-fixed-contacts-material) | env-dd2d |  |
| 2026-07-23 | [2026-07-23-adaptive-traces-carry-step-scores](04-comparison.md#2026-07-23-adaptive-traces-carry-step-scores) | tooling |  |
| 2026-07-23 | [2026-07-23-concave-grasp-sanity-demo](04-comparison.md#2026-07-23-concave-grasp-sanity-demo) | env-dd2d |  |
| 2026-07-20 | [2026-07-20-recomputed-comparison-cache-6-methods](03-dd2d-v2.2.md#2026-07-20-recomputed-comparison-cache-6-methods) | baselines, tooling |  |
| 2026-07-19 | [2026-07-19-observed-vs-computed-demotion](03-dd2d-v2.2.md#2026-07-19-observed-vs-computed-demotion) | method, env-dd2d |  |
| 2026-07-19 | [2026-07-19-fixing-v2-length-bias](03-dd2d-v2.2.md#2026-07-19-fixing-v2-length-bias) | method, evaluation |  |
| 2026-07-19 | [2026-07-19-in-distribution-main-table](03-dd2d-v2.2.md#2026-07-19-in-distribution-main-table) | evaluation, baselines |  |
| 2026-07-19 | [2026-07-19-v2-2-1-step-11-learned](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-11-learned) | method |  |
| 2026-07-19 | [2026-07-19-v2-2-1-step-10-proof-demotion](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-10-proof-demotion) | method |  |
| 2026-07-19 | [2026-07-19-v2-2-1-step-9-v2-static](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-9-v2-static) | method |  |
| 2026-07-19 | [2026-07-19-lambda-star-correction-to-0-8](03-dd2d-v2.2.md#2026-07-19-lambda-star-correction-to-0-8) | data, env-dd2d |  |
| 2026-07-19 | [2026-07-19-v2-2-1-step-8-v2](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-8-v2) | method |  |
| 2026-07-19 | [2026-07-19-v2-2-1-step-7-episode-local](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-7-episode-local) | method, data |  |
| 2026-07-19 | [2026-07-19-v2-2-1-step-6-comparison](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-6-comparison) | evaluation |  |
| 2026-07-19 | [2026-07-19-v2-2-1-step-5a-post-mortem](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-5a-post-mortem) | data |  |
| 2026-07-18 | [2026-07-18-v2-2-1-step-4-gate](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-4-gate) | evaluation, env-dd2d | **partly superseded** |
| 2026-07-18 | [2026-07-18-v2-2-1-step-3-schema](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-3-schema) | data |  |
| 2026-07-18 | [2026-07-18-v2-2-1-step-2-arrangement-complete](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-2-arrangement-complete) | env-dd2d |  |
| 2026-07-18 | [2026-07-18-v2-2-1-step-1-dd2d](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-1-dd2d) | env-dd2d, data |  |
| 2026-07-13 | [2026-07-13-first-training-run-on-dd2d](03-dd2d-v2.2.md#2026-07-13-first-training-run-on-dd2d) | method, env-dd2d |  |
| 2026-07-12 | [2026-07-12-dd2d-wired-via-converter](03-dd2d-v2.2.md#2026-07-12-dd2d-wired-via-converter) | env-dd2d, data |  |
| 2026-06-25 | [2026-06-25-psi-ablation-reinterpretation](02-pivot.md#2026-06-25-psi-ablation-reinterpretation) | method, process |  |
| 2026-06-11 | [2026-06-11-b6-exact-h-sweep](01-foundations.md#2026-06-11-b6-exact-h-sweep) | baselines |  |
| 2026-06-06 | [2026-06-06-frozen-context-ablation-rt2d-n3](01-foundations.md#2026-06-06-frozen-context-ablation-rt2d-n3) | method, env-rt2d |  |
| 2026-06-06 | [2026-06-06-seed-forwarding-bug](01-foundations.md#2026-06-06-seed-forwarding-bug) | tooling |  |
| 2026-04-27 | [2026-04-27-rt2d-n3-paper-snapshot](01-foundations.md#2026-04-27-rt2d-n3-paper-snapshot) | evaluation, env-rt2d |  |

## By track

- **method** — [2026-08-02-dd2d-s1-wall-clock-blow-up-diagnosed-per-candidate](07-stickbutton2d.md#2026-08-02-dd2d-s1-wall-clock-blow-up-diagnosed-per-candidate), [2026-08-02-s2-ood-degradation-pool-composition-artifact-model](07-stickbutton2d.md#2026-08-02-s2-ood-degradation-pool-composition-artifact-model), [2026-08-01-dd2d-generalization-v3-vs-astar-unseen](07-stickbutton2d.md#2026-08-01-dd2d-generalization-v3-vs-astar-unseen), [2026-08-01-sb2d-comparison-v3-piginet-indistinguishable-adaptivity](07-stickbutton2d.md#2026-08-01-sb2d-comparison-v3-piginet-indistinguishable-adaptivity), [2026-08-01-sb2d-collection-b1-b5-bracket-v3-1](07-stickbutton2d.md#2026-08-01-sb2d-collection-b1-b5-bracket-v3-1), [2026-08-01-stickbutton2d-stood-up-pool-shape-evidence](07-stickbutton2d.md#2026-08-01-stickbutton2d-stood-up-pool-shape-evidence), [2026-07-31-unified-coverage-waste-ab-5-83](06-v3-performance.md#2026-07-31-unified-coverage-waste-ab-5-83), [2026-07-31-unified-coverage-waste-probes](06-v3-performance.md#2026-07-31-unified-coverage-waste-probes), [2026-07-30-demotion-cut-authoritative-v3-7-44](06-v3-performance.md#2026-07-30-demotion-cut-authoritative-v3-7-44), [2026-07-30-proof-demotion-priced-0-23-fp-deployed](06-v3-performance.md#2026-07-30-proof-demotion-priced-0-23-fp-deployed), [2026-07-28-state-delta-ties-6-seeds](06-v3-performance.md#2026-07-28-state-delta-ties-6-seeds), [2026-07-27-p5-observed-coverage-waste](06-v3-performance.md#2026-07-27-p5-observed-coverage-waste), [2026-07-27-p2-missing-g6-cell](06-v3-performance.md#2026-07-27-p2-missing-g6-cell), [2026-07-27-g8-dropping-dead-fixes-s1](06-v3-performance.md#2026-07-27-g8-dropping-dead-fixes-s1), [2026-07-26-g0-g1-instrumentation-13-68-unreproducible](05-v3-migration.md#2026-07-26-g0-g1-instrumentation-13-68-unreproducible), [2026-07-26-g7-p-v3-3-falsified](05-v3-migration.md#2026-07-26-g7-p-v3-3-falsified), [2026-07-26-g6-record-tokens](05-v3-migration.md#2026-07-26-g6-record-tokens), [2026-07-26-g5-one-failurerecord](05-v3-migration.md#2026-07-26-g5-one-failurerecord), [2026-07-26-d2-advantage-is-length-calibration](05-v3-migration.md#2026-07-26-d2-advantage-is-length-calibration), [2026-07-25-v3-reversal-was-short-first-prior](04-comparison.md#2026-07-25-v3-reversal-was-short-first-prior), [2026-07-19-observed-vs-computed-demotion](03-dd2d-v2.2.md#2026-07-19-observed-vs-computed-demotion), [2026-07-19-fixing-v2-length-bias](03-dd2d-v2.2.md#2026-07-19-fixing-v2-length-bias), [2026-07-19-v2-2-1-step-11-learned](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-11-learned), [2026-07-19-v2-2-1-step-10-proof-demotion](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-10-proof-demotion), [2026-07-19-v2-2-1-step-9-v2-static](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-9-v2-static), [2026-07-19-v2-2-1-step-8-v2](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-8-v2), [2026-07-19-v2-2-1-step-7-episode-local](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-7-episode-local), [2026-07-13-first-training-run-on-dd2d](03-dd2d-v2.2.md#2026-07-13-first-training-run-on-dd2d), [2026-06-25-psi-ablation-reinterpretation](02-pivot.md#2026-06-25-psi-ablation-reinterpretation), [2026-06-06-frozen-context-ablation-rt2d-n3](01-foundations.md#2026-06-06-frozen-context-ablation-rt2d-n3)
- **evaluation** — [2026-08-03-vlmplan-frontier-vlm-gpt-5-6-luna-strong](07-stickbutton2d.md#2026-08-03-vlmplan-frontier-vlm-gpt-5-6-luna-strong), [2026-08-02-dd2d-s1-wall-clock-blow-up-diagnosed-per-candidate](07-stickbutton2d.md#2026-08-02-dd2d-s1-wall-clock-blow-up-diagnosed-per-candidate), [2026-08-02-dd2d-wall-clock-first-success-fp-flatters](07-stickbutton2d.md#2026-08-02-dd2d-wall-clock-first-success-fp-flatters), [2026-08-02-s2-ood-degradation-pool-composition-artifact-model](07-stickbutton2d.md#2026-08-02-s2-ood-degradation-pool-composition-artifact-model), [2026-08-01-dd2d-generalization-v3-vs-astar-unseen](07-stickbutton2d.md#2026-08-01-dd2d-generalization-v3-vs-astar-unseen), [2026-08-01-dd2d-compare-cache-rebuilt-unified-coverage](07-stickbutton2d.md#2026-08-01-dd2d-compare-cache-rebuilt-unified-coverage), [2026-08-01-sb2d-vlmplan-32b-capable-tail-limited-beats-astar](07-stickbutton2d.md#2026-08-01-sb2d-vlmplan-32b-capable-tail-limited-beats-astar), [2026-08-01-sb2d-ablation-arms-training-not-reproducible-from-seed](07-stickbutton2d.md#2026-08-01-sb2d-ablation-arms-training-not-reproducible-from-seed), [2026-08-01-sb2d-comparison-v3-piginet-indistinguishable-adaptivity](07-stickbutton2d.md#2026-08-01-sb2d-comparison-v3-piginet-indistinguishable-adaptivity), [2026-08-01-sb2d-collection-b1-b5-bracket-v3-1](07-stickbutton2d.md#2026-08-01-sb2d-collection-b1-b5-bracket-v3-1), [2026-08-01-stickbutton2d-stood-up-pool-shape-evidence](07-stickbutton2d.md#2026-08-01-stickbutton2d-stood-up-pool-shape-evidence), [2026-07-31-unified-coverage-waste-ab-5-83](06-v3-performance.md#2026-07-31-unified-coverage-waste-ab-5-83), [2026-07-31-unified-coverage-waste-probes](06-v3-performance.md#2026-07-31-unified-coverage-waste-probes), [2026-07-30-demotion-cut-authoritative-v3-7-44](06-v3-performance.md#2026-07-30-demotion-cut-authoritative-v3-7-44), [2026-07-30-proof-demotion-priced-0-23-fp-deployed](06-v3-performance.md#2026-07-30-proof-demotion-priced-0-23-fp-deployed), [2026-07-28-dd2d-comparison-3-seeds](06-v3-performance.md#2026-07-28-dd2d-comparison-3-seeds), [2026-07-28-state-delta-ties-6-seeds](06-v3-performance.md#2026-07-28-state-delta-ties-6-seeds), [2026-07-27-comparison-retargeted-two-stale-bugs](06-v3-performance.md#2026-07-27-comparison-retargeted-two-stale-bugs), [2026-07-27-p5-observed-coverage-waste](06-v3-performance.md#2026-07-27-p5-observed-coverage-waste), [2026-07-27-p2-missing-g6-cell](06-v3-performance.md#2026-07-27-p2-missing-g6-cell), [2026-07-26-g6b-uncensoring-the-selector](05-v3-migration.md#2026-07-26-g6b-uncensoring-the-selector), [2026-07-26-g6-record-tokens](05-v3-migration.md#2026-07-26-g6-record-tokens), [2026-07-26-d2-advantage-is-length-calibration](05-v3-migration.md#2026-07-26-d2-advantage-is-length-calibration), [2026-07-24-retrained-all-three-on-v3](04-comparison.md#2026-07-24-retrained-all-three-on-v3), [2026-07-19-fixing-v2-length-bias](03-dd2d-v2.2.md#2026-07-19-fixing-v2-length-bias), [2026-07-19-in-distribution-main-table](03-dd2d-v2.2.md#2026-07-19-in-distribution-main-table), [2026-07-19-v2-2-1-step-6-comparison](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-6-comparison), [2026-07-18-v2-2-1-step-4-gate](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-4-gate), [2026-04-27-rt2d-n3-paper-snapshot](01-foundations.md#2026-04-27-rt2d-n3-paper-snapshot)
- **data** — [2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s](07-stickbutton2d.md#2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s), [2026-08-01-stickbutton2d-stood-up-pool-shape-evidence](07-stickbutton2d.md#2026-08-01-stickbutton2d-stood-up-pool-shape-evidence), [2026-07-28-stickbutton2d-feasibility-b1-b3](06-v3-performance.md#2026-07-28-stickbutton2d-feasibility-b1-b3), [2026-07-26-g0-g1-instrumentation-13-68-unreproducible](05-v3-migration.md#2026-07-26-g0-g1-instrumentation-13-68-unreproducible), [2026-07-26-canonicalize-not-idempotent](05-v3-migration.md#2026-07-26-canonicalize-not-idempotent), [2026-07-19-lambda-star-correction-to-0-8](03-dd2d-v2.2.md#2026-07-19-lambda-star-correction-to-0-8), [2026-07-19-v2-2-1-step-7-episode-local](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-7-episode-local), [2026-07-19-v2-2-1-step-5a-post-mortem](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-5a-post-mortem), [2026-07-18-v2-2-1-step-3-schema](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-3-schema), [2026-07-18-v2-2-1-step-1-dd2d](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-1-dd2d), [2026-07-12-dd2d-wired-via-converter](03-dd2d-v2.2.md#2026-07-12-dd2d-wired-via-converter)
- **env-dd2d** — [2026-08-03-vlmplan-frontier-vlm-gpt-5-6-luna-strong](07-stickbutton2d.md#2026-08-03-vlmplan-frontier-vlm-gpt-5-6-luna-strong), [2026-08-02-dd2d-s1-wall-clock-blow-up-diagnosed-per-candidate](07-stickbutton2d.md#2026-08-02-dd2d-s1-wall-clock-blow-up-diagnosed-per-candidate), [2026-08-02-dd2d-wall-clock-first-success-fp-flatters](07-stickbutton2d.md#2026-08-02-dd2d-wall-clock-first-success-fp-flatters), [2026-08-02-s2-ood-degradation-pool-composition-artifact-model](07-stickbutton2d.md#2026-08-02-s2-ood-degradation-pool-composition-artifact-model), [2026-08-01-dd2d-generalization-v3-vs-astar-unseen](07-stickbutton2d.md#2026-08-01-dd2d-generalization-v3-vs-astar-unseen), [2026-08-01-dd2d-compare-cache-rebuilt-unified-coverage](07-stickbutton2d.md#2026-08-01-dd2d-compare-cache-rebuilt-unified-coverage), [2026-07-31-unified-coverage-waste-ab-5-83](06-v3-performance.md#2026-07-31-unified-coverage-waste-ab-5-83), [2026-07-26-g0-g1-instrumentation-13-68-unreproducible](05-v3-migration.md#2026-07-26-g0-g1-instrumentation-13-68-unreproducible), [2026-07-24-post-grasp-sanity-astar](04-comparison.md#2026-07-24-post-grasp-sanity-astar), [2026-07-24-grasp-reaches-concavities](04-comparison.md#2026-07-24-grasp-reaches-concavities), [2026-07-24-grasp-fixed-contacts-material](04-comparison.md#2026-07-24-grasp-fixed-contacts-material), [2026-07-23-concave-grasp-sanity-demo](04-comparison.md#2026-07-23-concave-grasp-sanity-demo), [2026-07-19-observed-vs-computed-demotion](03-dd2d-v2.2.md#2026-07-19-observed-vs-computed-demotion), [2026-07-19-lambda-star-correction-to-0-8](03-dd2d-v2.2.md#2026-07-19-lambda-star-correction-to-0-8), [2026-07-18-v2-2-1-step-4-gate](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-4-gate), [2026-07-18-v2-2-1-step-2-arrangement-complete](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-2-arrangement-complete), [2026-07-18-v2-2-1-step-1-dd2d](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-1-dd2d), [2026-07-13-first-training-run-on-dd2d](03-dd2d-v2.2.md#2026-07-13-first-training-run-on-dd2d), [2026-07-12-dd2d-wired-via-converter](03-dd2d-v2.2.md#2026-07-12-dd2d-wired-via-converter)
- **env-rt2d** — [2026-06-06-frozen-context-ablation-rt2d-n3](01-foundations.md#2026-06-06-frozen-context-ablation-rt2d-n3), [2026-04-27-rt2d-n3-paper-snapshot](01-foundations.md#2026-04-27-rt2d-n3-paper-snapshot)
- **env-stickbutton2d** — [2026-08-03-vlmplan-frontier-vlm-gpt-5-6-luna-strong](07-stickbutton2d.md#2026-08-03-vlmplan-frontier-vlm-gpt-5-6-luna-strong), [2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s](07-stickbutton2d.md#2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s), [2026-08-01-sb2d-vlmplan-32b-capable-tail-limited-beats-astar](07-stickbutton2d.md#2026-08-01-sb2d-vlmplan-32b-capable-tail-limited-beats-astar), [2026-08-01-sb2d-ablation-arms-training-not-reproducible-from-seed](07-stickbutton2d.md#2026-08-01-sb2d-ablation-arms-training-not-reproducible-from-seed), [2026-08-01-sb2d-comparison-v3-piginet-indistinguishable-adaptivity](07-stickbutton2d.md#2026-08-01-sb2d-comparison-v3-piginet-indistinguishable-adaptivity), [2026-08-01-sb2d-collection-b1-b5-bracket-v3-1](07-stickbutton2d.md#2026-08-01-sb2d-collection-b1-b5-bracket-v3-1), [2026-08-01-stickbutton2d-stood-up-pool-shape-evidence](07-stickbutton2d.md#2026-08-01-stickbutton2d-stood-up-pool-shape-evidence), [2026-07-31-unified-coverage-waste-probes](06-v3-performance.md#2026-07-31-unified-coverage-waste-probes), [2026-07-29-stickbutton2d-b5-reaches-75](06-v3-performance.md#2026-07-29-stickbutton2d-b5-reaches-75), [2026-07-28-stickbutton2d-feasibility-b1-b3](06-v3-performance.md#2026-07-28-stickbutton2d-feasibility-b1-b3)
- **baselines** — [2026-08-03-vlmplan-frontier-vlm-gpt-5-6-luna-strong](07-stickbutton2d.md#2026-08-03-vlmplan-frontier-vlm-gpt-5-6-luna-strong), [2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s](07-stickbutton2d.md#2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s), [2026-08-01-sb2d-vlmplan-32b-capable-tail-limited-beats-astar](07-stickbutton2d.md#2026-08-01-sb2d-vlmplan-32b-capable-tail-limited-beats-astar), [2026-08-01-sb2d-ablation-arms-training-not-reproducible-from-seed](07-stickbutton2d.md#2026-08-01-sb2d-ablation-arms-training-not-reproducible-from-seed), [2026-08-01-sb2d-comparison-v3-piginet-indistinguishable-adaptivity](07-stickbutton2d.md#2026-08-01-sb2d-comparison-v3-piginet-indistinguishable-adaptivity), [2026-08-01-sb2d-collection-b1-b5-bracket-v3-1](07-stickbutton2d.md#2026-08-01-sb2d-collection-b1-b5-bracket-v3-1), [2026-07-28-dd2d-comparison-3-seeds](06-v3-performance.md#2026-07-28-dd2d-comparison-3-seeds), [2026-07-26-vlmplan-scale-comparison](05-v3-migration.md#2026-07-26-vlmplan-scale-comparison), [2026-07-25-vlmplan-8b-test-split](04-comparison.md#2026-07-25-vlmplan-8b-test-split), [2026-07-25-rejected-reasoning-model-arm](04-comparison.md#2026-07-25-rejected-reasoning-model-arm), [2026-07-25-v3-reversal-was-short-first-prior](04-comparison.md#2026-07-25-v3-reversal-was-short-first-prior), [2026-07-24-retrained-all-three-on-v3](04-comparison.md#2026-07-24-retrained-all-three-on-v3), [2026-07-24-vlmplan-baseline-smoke-tested](04-comparison.md#2026-07-24-vlmplan-baseline-smoke-tested), [2026-07-24-post-grasp-sanity-astar](04-comparison.md#2026-07-24-post-grasp-sanity-astar), [2026-07-20-recomputed-comparison-cache-6-methods](03-dd2d-v2.2.md#2026-07-20-recomputed-comparison-cache-6-methods), [2026-07-19-in-distribution-main-table](03-dd2d-v2.2.md#2026-07-19-in-distribution-main-table), [2026-06-11-b6-exact-h-sweep](01-foundations.md#2026-06-11-b6-exact-h-sweep)
- **tooling** — [2026-08-02-dd2d-wall-clock-first-success-fp-flatters](07-stickbutton2d.md#2026-08-02-dd2d-wall-clock-first-success-fp-flatters), [2026-07-27-comparison-retargeted-two-stale-bugs](06-v3-performance.md#2026-07-27-comparison-retargeted-two-stale-bugs), [2026-07-26-canonicalize-not-idempotent](05-v3-migration.md#2026-07-26-canonicalize-not-idempotent), [2026-07-23-adaptive-traces-carry-step-scores](04-comparison.md#2026-07-23-adaptive-traces-carry-step-scores), [2026-07-20-recomputed-comparison-cache-6-methods](03-dd2d-v2.2.md#2026-07-20-recomputed-comparison-cache-6-methods), [2026-06-06-seed-forwarding-bug](01-foundations.md#2026-06-06-seed-forwarding-bug)
- **process** — [2026-06-25-psi-ablation-reinterpretation](02-pivot.md#2026-06-25-psi-ablation-reinterpretation)

## ID resolution

Where each gate / revision / prediction / constraint is decided.

| ID | Decided in |
|---|---|
| `D2` | [2026-07-26-d2-advantage-is-length-calibration](05-v3-migration.md#2026-07-26-d2-advantage-is-length-calibration) |
| `G0` | [2026-07-18-v2-2-1-step-4-gate](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-4-gate) |
| `G5` | [2026-07-26-g5-one-failurerecord](05-v3-migration.md#2026-07-26-g5-one-failurerecord) |
| `G6` | [2026-07-26-g6-record-tokens](05-v3-migration.md#2026-07-26-g6-record-tokens) |
| `G6b` | [2026-07-26-g6b-uncensoring-the-selector](05-v3-migration.md#2026-07-26-g6b-uncensoring-the-selector) |
| `P4` | [2026-07-19-v2-2-1-step-10-proof-demotion](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-10-proof-demotion) |
| `P5` | [2026-07-19-v2-2-1-step-11-learned](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-11-learned) |
| `P16` | [2026-07-18-v2-2-1-step-2-arrangement-complete](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-2-arrangement-complete) |
| `P19` | [2026-07-18-v2-2-1-step-2-arrangement-complete](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-2-arrangement-complete) |
| `P-v3-3` | [2026-07-26-g7-p-v3-3-falsified](05-v3-migration.md#2026-07-26-g7-p-v3-3-falsified) |

## Do not quote

Conclusions and numbers that later entries retracted or replaced. Check here before citing any figure from a historical entry.

| Entry | Status | What replaced it |
|---|---|---|
| [2026-07-30-proof-demotion-priced-0-23-fp-deployed](06-v3-performance.md#2026-07-30-proof-demotion-priced-0-23-fp-deployed) | **partly superseded** | [2026-07-30-demotion-cut-authoritative-v3-7-44](06-v3-performance.md#2026-07-30-demotion-cut-authoritative-v3-7-44) |
| [2026-07-28-stickbutton2d-feasibility-b1-b3](06-v3-performance.md#2026-07-28-stickbutton2d-feasibility-b1-b3) | **partly superseded** | [2026-07-29-stickbutton2d-b5-reaches-75](06-v3-performance.md#2026-07-29-stickbutton2d-b5-reaches-75) |
| [2026-07-26-g0-g1-instrumentation-13-68-unreproducible](05-v3-migration.md#2026-07-26-g0-g1-instrumentation-13-68-unreproducible) | amended | ⚠️ **AMENDED** — the cause was double canonicalization in the cache builder, not code staleness, and the corrected figur |
| [2026-07-26-g6-record-tokens](05-v3-migration.md#2026-07-26-g6-record-tokens) | **retracted** | [2026-07-26-g6b-uncensoring-the-selector](05-v3-migration.md#2026-07-26-g6b-uncensoring-the-selector) |
| [2026-07-24-retrained-all-three-on-v3](04-comparison.md#2026-07-24-retrained-all-three-on-v3) | **superseded** | [2026-07-25-v3-reversal-was-short-first-prior](04-comparison.md#2026-07-25-v3-reversal-was-short-first-prior) |
| [2026-07-24-vlmplan-baseline-smoke-tested](04-comparison.md#2026-07-24-vlmplan-baseline-smoke-tested) | **partly superseded** | [2026-07-25-vlmplan-8b-test-split](04-comparison.md#2026-07-25-vlmplan-8b-test-split) |
| [2026-07-18-v2-2-1-step-4-gate](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-4-gate) | **partly superseded** | [2026-07-19-lambda-star-correction-to-0-8](03-dd2d-v2.2.md#2026-07-19-lambda-star-correction-to-0-8) |

## Legacy citation resolution

Code docstrings cite this log as `` `notebook.md` <date> ``. Dates collide, so this table resolves each to the entries on that date.

| Cited date | Entries |
|---|---|
| 2026-08-03 | [2026-08-03-vlmplan-frontier-vlm-gpt-5-6-luna-strong](07-stickbutton2d.md#2026-08-03-vlmplan-frontier-vlm-gpt-5-6-luna-strong) |
| 2026-08-02 | [2026-08-02-dd2d-s1-wall-clock-blow-up-diagnosed-per-candidate](07-stickbutton2d.md#2026-08-02-dd2d-s1-wall-clock-blow-up-diagnosed-per-candidate), [2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s](07-stickbutton2d.md#2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s), [2026-08-02-dd2d-wall-clock-first-success-fp-flatters](07-stickbutton2d.md#2026-08-02-dd2d-wall-clock-first-success-fp-flatters), [2026-08-02-s2-ood-degradation-pool-composition-artifact-model](07-stickbutton2d.md#2026-08-02-s2-ood-degradation-pool-composition-artifact-model) |
| 2026-08-01 | [2026-08-01-dd2d-generalization-v3-vs-astar-unseen](07-stickbutton2d.md#2026-08-01-dd2d-generalization-v3-vs-astar-unseen), [2026-08-01-dd2d-compare-cache-rebuilt-unified-coverage](07-stickbutton2d.md#2026-08-01-dd2d-compare-cache-rebuilt-unified-coverage), [2026-08-01-sb2d-vlmplan-32b-capable-tail-limited-beats-astar](07-stickbutton2d.md#2026-08-01-sb2d-vlmplan-32b-capable-tail-limited-beats-astar), [2026-08-01-sb2d-ablation-arms-training-not-reproducible-from-seed](07-stickbutton2d.md#2026-08-01-sb2d-ablation-arms-training-not-reproducible-from-seed), [2026-08-01-sb2d-comparison-v3-piginet-indistinguishable-adaptivity](07-stickbutton2d.md#2026-08-01-sb2d-comparison-v3-piginet-indistinguishable-adaptivity), [2026-08-01-sb2d-collection-b1-b5-bracket-v3-1](07-stickbutton2d.md#2026-08-01-sb2d-collection-b1-b5-bracket-v3-1), [2026-08-01-stickbutton2d-stood-up-pool-shape-evidence](07-stickbutton2d.md#2026-08-01-stickbutton2d-stood-up-pool-shape-evidence) |
| 2026-07-31 | [2026-07-31-unified-coverage-waste-ab-5-83](06-v3-performance.md#2026-07-31-unified-coverage-waste-ab-5-83), [2026-07-31-unified-coverage-waste-probes](06-v3-performance.md#2026-07-31-unified-coverage-waste-probes) |
| 2026-07-30 | [2026-07-30-demotion-cut-authoritative-v3-7-44](06-v3-performance.md#2026-07-30-demotion-cut-authoritative-v3-7-44), [2026-07-30-proof-demotion-priced-0-23-fp-deployed](06-v3-performance.md#2026-07-30-proof-demotion-priced-0-23-fp-deployed) |
| 2026-07-29 | [2026-07-29-stickbutton2d-b5-reaches-75](06-v3-performance.md#2026-07-29-stickbutton2d-b5-reaches-75) |
| 2026-07-28 | [2026-07-28-stickbutton2d-feasibility-b1-b3](06-v3-performance.md#2026-07-28-stickbutton2d-feasibility-b1-b3), [2026-07-28-dd2d-comparison-3-seeds](06-v3-performance.md#2026-07-28-dd2d-comparison-3-seeds), [2026-07-28-state-delta-ties-6-seeds](06-v3-performance.md#2026-07-28-state-delta-ties-6-seeds) |
| 2026-07-27 | [2026-07-27-comparison-retargeted-two-stale-bugs](06-v3-performance.md#2026-07-27-comparison-retargeted-two-stale-bugs), [2026-07-27-p5-observed-coverage-waste](06-v3-performance.md#2026-07-27-p5-observed-coverage-waste), [2026-07-27-p2-missing-g6-cell](06-v3-performance.md#2026-07-27-p2-missing-g6-cell), [2026-07-27-g8-dropping-dead-fixes-s1](06-v3-performance.md#2026-07-27-g8-dropping-dead-fixes-s1) |
| 2026-07-26 | [2026-07-26-g0-g1-instrumentation-13-68-unreproducible](05-v3-migration.md#2026-07-26-g0-g1-instrumentation-13-68-unreproducible), [2026-07-26-g7-p-v3-3-falsified](05-v3-migration.md#2026-07-26-g7-p-v3-3-falsified), [2026-07-26-g6b-uncensoring-the-selector](05-v3-migration.md#2026-07-26-g6b-uncensoring-the-selector), [2026-07-26-g6-record-tokens](05-v3-migration.md#2026-07-26-g6-record-tokens), [2026-07-26-g5-one-failurerecord](05-v3-migration.md#2026-07-26-g5-one-failurerecord), [2026-07-26-d2-advantage-is-length-calibration](05-v3-migration.md#2026-07-26-d2-advantage-is-length-calibration), [2026-07-26-canonicalize-not-idempotent](05-v3-migration.md#2026-07-26-canonicalize-not-idempotent), [2026-07-26-vlmplan-scale-comparison](05-v3-migration.md#2026-07-26-vlmplan-scale-comparison) |
| 2026-07-25 | [2026-07-25-vlmplan-8b-test-split](04-comparison.md#2026-07-25-vlmplan-8b-test-split), [2026-07-25-rejected-reasoning-model-arm](04-comparison.md#2026-07-25-rejected-reasoning-model-arm), [2026-07-25-v3-reversal-was-short-first-prior](04-comparison.md#2026-07-25-v3-reversal-was-short-first-prior) |
| 2026-07-24 | [2026-07-24-retrained-all-three-on-v3](04-comparison.md#2026-07-24-retrained-all-three-on-v3), [2026-07-24-vlmplan-baseline-smoke-tested](04-comparison.md#2026-07-24-vlmplan-baseline-smoke-tested), [2026-07-24-post-grasp-sanity-astar](04-comparison.md#2026-07-24-post-grasp-sanity-astar), [2026-07-24-grasp-reaches-concavities](04-comparison.md#2026-07-24-grasp-reaches-concavities), [2026-07-24-grasp-fixed-contacts-material](04-comparison.md#2026-07-24-grasp-fixed-contacts-material) |
| 2026-07-23 | [2026-07-23-adaptive-traces-carry-step-scores](04-comparison.md#2026-07-23-adaptive-traces-carry-step-scores), [2026-07-23-concave-grasp-sanity-demo](04-comparison.md#2026-07-23-concave-grasp-sanity-demo) |
| 2026-07-20 | [2026-07-20-recomputed-comparison-cache-6-methods](03-dd2d-v2.2.md#2026-07-20-recomputed-comparison-cache-6-methods) |
| 2026-07-19 | [2026-07-19-observed-vs-computed-demotion](03-dd2d-v2.2.md#2026-07-19-observed-vs-computed-demotion), [2026-07-19-fixing-v2-length-bias](03-dd2d-v2.2.md#2026-07-19-fixing-v2-length-bias), [2026-07-19-in-distribution-main-table](03-dd2d-v2.2.md#2026-07-19-in-distribution-main-table), [2026-07-19-v2-2-1-step-11-learned](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-11-learned), [2026-07-19-v2-2-1-step-10-proof-demotion](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-10-proof-demotion), [2026-07-19-v2-2-1-step-9-v2-static](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-9-v2-static), [2026-07-19-lambda-star-correction-to-0-8](03-dd2d-v2.2.md#2026-07-19-lambda-star-correction-to-0-8), [2026-07-19-v2-2-1-step-8-v2](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-8-v2), [2026-07-19-v2-2-1-step-7-episode-local](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-7-episode-local), [2026-07-19-v2-2-1-step-6-comparison](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-6-comparison), [2026-07-19-v2-2-1-step-5a-post-mortem](03-dd2d-v2.2.md#2026-07-19-v2-2-1-step-5a-post-mortem) |
| 2026-07-18 | [2026-07-18-v2-2-1-step-4-gate](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-4-gate), [2026-07-18-v2-2-1-step-3-schema](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-3-schema), [2026-07-18-v2-2-1-step-2-arrangement-complete](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-2-arrangement-complete), [2026-07-18-v2-2-1-step-1-dd2d](03-dd2d-v2.2.md#2026-07-18-v2-2-1-step-1-dd2d) |
| 2026-07-13 | [2026-07-13-first-training-run-on-dd2d](03-dd2d-v2.2.md#2026-07-13-first-training-run-on-dd2d) |
| 2026-07-12 | [2026-07-12-dd2d-wired-via-converter](03-dd2d-v2.2.md#2026-07-12-dd2d-wired-via-converter) |
| 2026-06-25 | [2026-06-25-psi-ablation-reinterpretation](02-pivot.md#2026-06-25-psi-ablation-reinterpretation) |
| 2026-06-11 | [2026-06-11-b6-exact-h-sweep](01-foundations.md#2026-06-11-b6-exact-h-sweep) |
| 2026-06-06 | [2026-06-06-frozen-context-ablation-rt2d-n3](01-foundations.md#2026-06-06-frozen-context-ablation-rt2d-n3), [2026-06-06-seed-forwarding-bug](01-foundations.md#2026-06-06-seed-forwarding-bug) |
| 2026-04-27 | [2026-04-27-rt2d-n3-paper-snapshot](01-foundations.md#2026-04-27-rt2d-n3-paper-snapshot) |

<!--END GENERATED-->
