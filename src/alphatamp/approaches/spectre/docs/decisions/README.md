# SPECTRE — Decision Log (ADR style)

One entry per consequential decision: **context → decision → consequences**. Entries are
grouped into **era chapters**, newest first within each chapter. This README is generated
from those chapters — everything below the marker is rebuilt by
`experiments/spectre/decisions_index.py index`, so edit the chapters, not the tables.

The pre-split single file is frozen at
[`../archive/decisions_2026-07-29_monolithic.md`](../archive/decisions_2026-07-29_monolithic.md)
and every entry in it is preserved byte-for-byte here; `decisions_index.py check` proves it
on every run.

## How to cite an entry

Use the entry **id**, which is stable forever and appears in each entry's metadata strip:

```
[2026-07-26-selection-metric-never-censored](../decisions/05-v3-migration.md#2026-07-26-selection-metric-never-censored)
```

Older code docstrings cite this log as `` `decisions.md` <date> ``. Dates collide — 2026-07-19
alone has six entries — so the **Legacy citation resolution** table below maps each cited date
to the entries on it.

## Reading a retrieved entry

Every entry carries a fenced metadata strip under its heading with its id, status, tracks and
cross-references. **Check the status before quoting any number.** An entry marked
`superseded`, `partially-superseded`, `retracted` or `amended` also carries a banner naming
what replaced it, and the **Do not quote** table below lists them all in one place.

## Adding to the log

| You want to | Do this |
|---|---|
| record a new decision | `python experiments/spectre/decisions_index.py new --log decisions --title "..." --tracks method,evaluation`, write context → decision → consequences, then run `... index` |
| change what an old decision says | **Never edit its body.** Add a new entry, set `supersedes` on it and `superseded_by` + a `status` + a banner on the old one, then `... index`. `check` enforces the symmetry and refuses silent edits to historical entries. |
| start a new chapter | At a named phase boundary (new proposal doc, new env-variant generation, version bump), or when the open chapter passes ~650 lines / 12 entries — `check` warns. Add it to `_ERAS` in `doclog.py`. |
| promote an `autorun_decisions.md` entry | Only if it changed a convention, architecture or invariant **and** no existing ADR states that decision. Promotion is a *ratification*: write the ADR, set `ratifies`, and add a forward pointer in the autorun entry — never move its text. |

Doc-routing (which log an entry belongs in at all) is in the root
[`CLAUDE.md`](../../CLAUDE.md) under "Documentation discipline".

## Standing invariants

The rules that must not be broken without a new ADR. Each links to the entry that set it.

| Invariant | Set by |
|---|---|
| Loss is **listwise Plackett-Luce only** (global + within-length buckets); never pointwise BCE | pre-refactor; [2026-07-19](03-dd2d-v2.2.md#2026-07-19-v2-ranker-fix-length-bias-generalizably) |
| `F ⊆ FAIL_e` strictly — the failure context never contains a success | [pre-refactor](99-pre-refactor.md) |
| Vocab is built from **train only**; id 0 = `<PAD>`/`<OOV>` | [pre-refactor](99-pre-refactor.md) |
| Evaluation is **uncensored**: attempt budget = candidate-pool cap | [2026-06-07](01-foundations.md#2026-06-07-uncensored-evaluation-at-pool-cap) |
| A selection metric is **never censored below the tail that separates the models** | [2026-07-26](05-v3-migration.md#2026-07-26-selection-metric-never-censored) |
| **Stride, never truncate** — episodes are stored in seed order, so `paths[:N]` yields only easy strata | [2026-07-27](06-v3-performance.md#2026-07-27-cross-collection-grafting-coverage-mode) |
| **Reconstruct, never regenerate** — post-hoc geometry comes from stored `scene_geometry` | [2026-07-19](03-dd2d-v2.2.md#2026-07-19-reconstruct-never-regenerate) |
| `canonicalize_episode` is **not idempotent** — always tensorize from raw episodes | [2026-07-26](05-v3-migration.md#2026-07-26-v3-migration-g0-g2) |
| **A checkpoint is not a result until its training log says the run finished** | [2026-07-27](06-v3-performance.md#2026-07-27-cross-collection-grafting-coverage-mode) |
| **Re-score the frozen baseline under new code before training anything** | [2026-07-28](06-v3-performance.md#2026-07-28-state-delta-on-record-ties) |
| A new v3 feature is an **additive zero-initialized branch**, never a widened `Linear` | [2026-07-28](06-v3-performance.md#2026-07-28-state-delta-on-record-ties) |
| Refiner instrumentation is **observation-only** — `n_attempts` *is* `counter.calls` | [2026-07-26](05-v3-migration.md#2026-07-26-v3-migration-g0-g2) |
| Deductions act on the **ranking**, not the representation (C5) | [2026-07-27](06-v3-performance.md#2026-07-27-dead-is-a-length-proxy) |
| Rebuild **both arms** of a paired comparison in the same pass | [2026-07-27](06-v3-performance.md#2026-07-27-cross-collection-grafting-coverage-mode) |
| A load-bearing per-stratum margin is compared to the **seed sd**, not just the baseline | [2026-07-27](06-v3-performance.md#2026-07-27-margin-must-be-compared-to-seed-sd) |
| Paper numbers are mean ± std over **≥ 3 seeds**; dev runs 1 seed + paired bootstrap | [2026-07-27](06-v3-performance.md#2026-07-27-margin-must-be-compared-to-seed-sd) |

<!--BEGIN GENERATED-->

## Chapters

| Chapter | Entries | Span | State |
|---|---|---|---|
| [07-stickbutton2d](07-stickbutton2d.md) — StickButton2D as a second environment | 3 | 2026-08-01 .. 2026-08-01 | **open** |
| [06-v3-performance](06-v3-performance.md) — v3 performance push | 12 | 2026-07-27 .. 2026-07-31 | closed |
| [05-v3-migration](05-v3-migration.md) — v3 migration | 4 | 2026-07-26 .. 2026-07-26 | closed |
| [04-comparison](04-comparison.md) — Method comparison and VLMPlan | 8 | 2026-07-23 .. 2026-07-25 | closed |
| [03-dd2d-v2.2](03-dd2d-v2.2.md) — DD2D integration and v2.2 | 12 | 2026-07-12 .. 2026-07-20 | closed |
| [02-pivot](02-pivot.md) — Direction pivot | 1 | 2026-06-25 .. 2026-06-25 | closed |
| [01-foundations](01-foundations.md) — Foundations | 7 | 2026-06-04 .. 2026-06-11 | closed |
| [99-pre-refactor](99-pre-refactor.md) — pre-refactor | — | 2026-04 | closed |

## All entries, newest first

| Date | Entry | Tracks | Status |
|---|---|---|---|
| 2026-08-01 | [2026-08-01-piginet-lifted-env-agnostic-package-per-env-adapters](07-stickbutton2d.md#2026-08-01-piginet-lifted-env-agnostic-package-per-env-adapters) | baselines, tooling, env-stickbutton2d |  |
| 2026-08-01 | [2026-08-01-both-evidence-classes-stay-wired-stickbutton2d](07-stickbutton2d.md#2026-08-01-both-evidence-classes-stay-wired-stickbutton2d) | method, data, env-stickbutton2d |  |
| 2026-08-01 | [2026-08-01-acyclic-pool-filter-pooled-stickbutton2d-v1](07-stickbutton2d.md#2026-08-01-acyclic-pool-filter-pooled-stickbutton2d-v1) | method, data, env-stickbutton2d |  |
| 2026-07-31 | [2026-07-31-unified-coverage-waste-is-the-deployed-definition](06-v3-performance.md#2026-07-31-unified-coverage-waste-is-the-deployed-definition) | method, evaluation, env-dd2d, env-stickbutton2d |  |
| 2026-07-30 | [2026-07-30-proof-tier-demotion-cut-deployed-method-v3](06-v3-performance.md#2026-07-30-proof-tier-demotion-cut-deployed-method-v3) | method, evaluation |  |
| 2026-07-29 | [2026-07-29-stickbutton2d-heuristic-distance-term](06-v3-performance.md#2026-07-29-stickbutton2d-heuristic-distance-term) | env-stickbutton2d, data |  |
| 2026-07-28 | [2026-07-28-stickbutton2d-subclass-plan-generator](06-v3-performance.md#2026-07-28-stickbutton2d-subclass-plan-generator) | env-stickbutton2d, data | **partly superseded** |
| 2026-07-28 | [2026-07-28-state-delta-deployed-3-seed-protocol](06-v3-performance.md#2026-07-28-state-delta-deployed-3-seed-protocol) | method, evaluation, baselines |  |
| 2026-07-28 | [2026-07-28-state-delta-on-record-ties](06-v3-performance.md#2026-07-28-state-delta-on-record-ties) | method |  |
| 2026-07-27 | [2026-07-27-cross-collection-grafting-coverage-mode](06-v3-performance.md#2026-07-27-cross-collection-grafting-coverage-mode) | tooling, evaluation, env-dd2d |  |
| 2026-07-27 | [2026-07-27-necessity-observed-not-predicted](06-v3-performance.md#2026-07-27-necessity-observed-not-predicted) | method, evaluation, env-dd2d |  |
| 2026-07-27 | [2026-07-27-dead-is-a-length-proxy](06-v3-performance.md#2026-07-27-dead-is-a-length-proxy) | method |  |
| 2026-07-27 | [2026-07-27-record-tokens-are-ignored-at-inference](06-v3-performance.md#2026-07-27-record-tokens-are-ignored-at-inference) | method, evaluation |  |
| 2026-07-27 | [2026-07-27-evidence-needs-its-own-attention-channel](06-v3-performance.md#2026-07-27-evidence-needs-its-own-attention-channel) | method |  |
| 2026-07-27 | [2026-07-27-margin-must-be-compared-to-seed-sd](06-v3-performance.md#2026-07-27-margin-must-be-compared-to-seed-sd) | evaluation, process |  |
| 2026-07-26 | [2026-07-26-selection-metric-never-censored](05-v3-migration.md#2026-07-26-selection-metric-never-censored) | evaluation, method |  |
| 2026-07-26 | [2026-07-26-necessity-conditioning-cut](05-v3-migration.md#2026-07-26-necessity-conditioning-cut) | method |  |
| 2026-07-26 | [2026-07-26-dd2d-generator-pythonhashseed-dependent](05-v3-migration.md#2026-07-26-dd2d-generator-pythonhashseed-dependent) | env-dd2d, data |  |
| 2026-07-26 | [2026-07-26-v3-migration-g0-g2](05-v3-migration.md#2026-07-26-v3-migration-g0-g2) | method, data, env-dd2d | amended |
| 2026-07-25 | [2026-07-25-vlmplan-v3-test-split-two-arms](04-comparison.md#2026-07-25-vlmplan-v3-test-split-two-arms) | baselines, tooling |  |
| 2026-07-25 | [2026-07-25-v3-headline-reversal-was-training-artifact](04-comparison.md#2026-07-25-v3-headline-reversal-was-training-artifact) | method, baselines, env-dd2d |  |
| 2026-07-24 | [2026-07-24-dd2d-comparison-retargeted-v3](04-comparison.md#2026-07-24-dd2d-comparison-retargeted-v3) | baselines, evaluation, env-dd2d | **partly superseded** |
| 2026-07-24 | [2026-07-24-vlmplan-baseline-protocol](04-comparison.md#2026-07-24-vlmplan-baseline-protocol) | baselines, env-dd2d | **partly superseded** |
| 2026-07-24 | [2026-07-24-dd2d-collector-guarantees-exact-per-stratum-counts](04-comparison.md#2026-07-24-dd2d-collector-guarantees-exact-per-stratum-counts) | data, env-dd2d |  |
| 2026-07-24 | [2026-07-24-grasp-internal-concave-grasps](04-comparison.md#2026-07-24-grasp-internal-concave-grasps) | env-dd2d |  |
| 2026-07-24 | [2026-07-24-grasp-model-contacts-material](04-comparison.md#2026-07-24-grasp-model-contacts-material) | env-dd2d |  |
| 2026-07-23 | [2026-07-23-adaptive-traces-persist-step-scores](04-comparison.md#2026-07-23-adaptive-traces-persist-step-scores) | tooling |  |
| 2026-07-20 | [2026-07-20-dd2d-comparison-notebook-piginet-bce](03-dd2d-v2.2.md#2026-07-20-dd2d-comparison-notebook-piginet-bce) | baselines, tooling, evaluation | **partly superseded** |
| 2026-07-19 | [2026-07-19-demotion-signal-flag-default-observed](03-dd2d-v2.2.md#2026-07-19-demotion-signal-flag-default-observed) | method, env-dd2d |  |
| 2026-07-19 | [2026-07-19-v2-ranker-fix-length-bias-generalizably](03-dd2d-v2.2.md#2026-07-19-v2-ranker-fix-length-bias-generalizably) | method, evaluation |  |
| 2026-07-19 | [2026-07-19-step-11-typed-evidence-harvest](03-dd2d-v2.2.md#2026-07-19-step-11-typed-evidence-harvest) | method, data, env-dd2d |  |
| 2026-07-19 | [2026-07-19-reconstruct-never-regenerate](03-dd2d-v2.2.md#2026-07-19-reconstruct-never-regenerate) | data, env-dd2d, process |  |
| 2026-07-19 | [2026-07-19-lambda-star-corrected-to-0-8](03-dd2d-v2.2.md#2026-07-19-lambda-star-corrected-to-0-8) | data, env-dd2d |  |
| 2026-07-19 | [2026-07-19-decouple-harvest-from-collection](03-dd2d-v2.2.md#2026-07-19-decouple-harvest-from-collection) | data, process |  |
| 2026-07-18 | [2026-07-18-gate-g0-passes-size-control-mandatory](03-dd2d-v2.2.md#2026-07-18-gate-g0-passes-size-control-mandatory) | evaluation, env-dd2d | **partly superseded** |
| 2026-07-18 | [2026-07-18-schema-geometry-evidence-layer](03-dd2d-v2.2.md#2026-07-18-schema-geometry-evidence-layer) | data |  |
| 2026-07-18 | [2026-07-18-dd2d-negative-packing-certificate](03-dd2d-v2.2.md#2026-07-18-dd2d-negative-packing-certificate) | env-dd2d, data |  |
| 2026-07-18 | [2026-07-18-modernize-pin-substrate-deps](03-dd2d-v2.2.md#2026-07-18-modernize-pin-substrate-deps) | infra |  |
| 2026-07-12 | [2026-07-12-dd2d-integration-converter-not-native-env](03-dd2d-v2.2.md#2026-07-12-dd2d-integration-converter-not-native-env) | env-dd2d, data |  |
| 2026-06-25 | [2026-06-25-direction-pivot-representation-question](02-pivot.md#2026-06-25-direction-pivot-representation-question) | process, method |  |
| 2026-06-11 | [2026-06-11-b6-higher-horizons-incremental-scoring](01-foundations.md#2026-06-11-b6-higher-horizons-incremental-scoring) | baselines, tooling |  |
| 2026-06-08 | [2026-06-08-dp-on-counts-b6-baseline](01-foundations.md#2026-06-08-dp-on-counts-b6-baseline) | baselines |  |
| 2026-06-07 | [2026-06-07-analysis-notebook-converted-marimo](01-foundations.md#2026-06-07-analysis-notebook-converted-marimo) | tooling |  |
| 2026-06-07 | [2026-06-07-uncensored-evaluation-at-pool-cap](01-foundations.md#2026-06-07-uncensored-evaluation-at-pool-cap) | evaluation |  |
| 2026-06-06 | [2026-06-06-documentation-discipline-codified](01-foundations.md#2026-06-06-documentation-discipline-codified) | process |  |
| 2026-06-06 | [2026-06-06-dated-writeup-snapshots](01-foundations.md#2026-06-06-dated-writeup-snapshots) | process |  |
| 2026-06-04 | [2026-06-04-silo-refactor-scope-placement](01-foundations.md#2026-06-04-silo-refactor-scope-placement) | process, infra |  |

## By track

- **method** — [2026-08-01-both-evidence-classes-stay-wired-stickbutton2d](07-stickbutton2d.md#2026-08-01-both-evidence-classes-stay-wired-stickbutton2d), [2026-08-01-acyclic-pool-filter-pooled-stickbutton2d-v1](07-stickbutton2d.md#2026-08-01-acyclic-pool-filter-pooled-stickbutton2d-v1), [2026-07-31-unified-coverage-waste-is-the-deployed-definition](06-v3-performance.md#2026-07-31-unified-coverage-waste-is-the-deployed-definition), [2026-07-30-proof-tier-demotion-cut-deployed-method-v3](06-v3-performance.md#2026-07-30-proof-tier-demotion-cut-deployed-method-v3), [2026-07-28-state-delta-deployed-3-seed-protocol](06-v3-performance.md#2026-07-28-state-delta-deployed-3-seed-protocol), [2026-07-28-state-delta-on-record-ties](06-v3-performance.md#2026-07-28-state-delta-on-record-ties), [2026-07-27-necessity-observed-not-predicted](06-v3-performance.md#2026-07-27-necessity-observed-not-predicted), [2026-07-27-dead-is-a-length-proxy](06-v3-performance.md#2026-07-27-dead-is-a-length-proxy), [2026-07-27-record-tokens-are-ignored-at-inference](06-v3-performance.md#2026-07-27-record-tokens-are-ignored-at-inference), [2026-07-27-evidence-needs-its-own-attention-channel](06-v3-performance.md#2026-07-27-evidence-needs-its-own-attention-channel), [2026-07-26-selection-metric-never-censored](05-v3-migration.md#2026-07-26-selection-metric-never-censored), [2026-07-26-necessity-conditioning-cut](05-v3-migration.md#2026-07-26-necessity-conditioning-cut), [2026-07-26-v3-migration-g0-g2](05-v3-migration.md#2026-07-26-v3-migration-g0-g2), [2026-07-25-v3-headline-reversal-was-training-artifact](04-comparison.md#2026-07-25-v3-headline-reversal-was-training-artifact), [2026-07-19-demotion-signal-flag-default-observed](03-dd2d-v2.2.md#2026-07-19-demotion-signal-flag-default-observed), [2026-07-19-v2-ranker-fix-length-bias-generalizably](03-dd2d-v2.2.md#2026-07-19-v2-ranker-fix-length-bias-generalizably), [2026-07-19-step-11-typed-evidence-harvest](03-dd2d-v2.2.md#2026-07-19-step-11-typed-evidence-harvest), [2026-06-25-direction-pivot-representation-question](02-pivot.md#2026-06-25-direction-pivot-representation-question)
- **evaluation** — [2026-07-31-unified-coverage-waste-is-the-deployed-definition](06-v3-performance.md#2026-07-31-unified-coverage-waste-is-the-deployed-definition), [2026-07-30-proof-tier-demotion-cut-deployed-method-v3](06-v3-performance.md#2026-07-30-proof-tier-demotion-cut-deployed-method-v3), [2026-07-28-state-delta-deployed-3-seed-protocol](06-v3-performance.md#2026-07-28-state-delta-deployed-3-seed-protocol), [2026-07-27-cross-collection-grafting-coverage-mode](06-v3-performance.md#2026-07-27-cross-collection-grafting-coverage-mode), [2026-07-27-necessity-observed-not-predicted](06-v3-performance.md#2026-07-27-necessity-observed-not-predicted), [2026-07-27-record-tokens-are-ignored-at-inference](06-v3-performance.md#2026-07-27-record-tokens-are-ignored-at-inference), [2026-07-27-margin-must-be-compared-to-seed-sd](06-v3-performance.md#2026-07-27-margin-must-be-compared-to-seed-sd), [2026-07-26-selection-metric-never-censored](05-v3-migration.md#2026-07-26-selection-metric-never-censored), [2026-07-24-dd2d-comparison-retargeted-v3](04-comparison.md#2026-07-24-dd2d-comparison-retargeted-v3), [2026-07-20-dd2d-comparison-notebook-piginet-bce](03-dd2d-v2.2.md#2026-07-20-dd2d-comparison-notebook-piginet-bce), [2026-07-19-v2-ranker-fix-length-bias-generalizably](03-dd2d-v2.2.md#2026-07-19-v2-ranker-fix-length-bias-generalizably), [2026-07-18-gate-g0-passes-size-control-mandatory](03-dd2d-v2.2.md#2026-07-18-gate-g0-passes-size-control-mandatory), [2026-06-07-uncensored-evaluation-at-pool-cap](01-foundations.md#2026-06-07-uncensored-evaluation-at-pool-cap)
- **data** — [2026-08-01-both-evidence-classes-stay-wired-stickbutton2d](07-stickbutton2d.md#2026-08-01-both-evidence-classes-stay-wired-stickbutton2d), [2026-08-01-acyclic-pool-filter-pooled-stickbutton2d-v1](07-stickbutton2d.md#2026-08-01-acyclic-pool-filter-pooled-stickbutton2d-v1), [2026-07-29-stickbutton2d-heuristic-distance-term](06-v3-performance.md#2026-07-29-stickbutton2d-heuristic-distance-term), [2026-07-28-stickbutton2d-subclass-plan-generator](06-v3-performance.md#2026-07-28-stickbutton2d-subclass-plan-generator), [2026-07-26-dd2d-generator-pythonhashseed-dependent](05-v3-migration.md#2026-07-26-dd2d-generator-pythonhashseed-dependent), [2026-07-26-v3-migration-g0-g2](05-v3-migration.md#2026-07-26-v3-migration-g0-g2), [2026-07-24-dd2d-collector-guarantees-exact-per-stratum-counts](04-comparison.md#2026-07-24-dd2d-collector-guarantees-exact-per-stratum-counts), [2026-07-19-step-11-typed-evidence-harvest](03-dd2d-v2.2.md#2026-07-19-step-11-typed-evidence-harvest), [2026-07-19-reconstruct-never-regenerate](03-dd2d-v2.2.md#2026-07-19-reconstruct-never-regenerate), [2026-07-19-lambda-star-corrected-to-0-8](03-dd2d-v2.2.md#2026-07-19-lambda-star-corrected-to-0-8), [2026-07-19-decouple-harvest-from-collection](03-dd2d-v2.2.md#2026-07-19-decouple-harvest-from-collection), [2026-07-18-schema-geometry-evidence-layer](03-dd2d-v2.2.md#2026-07-18-schema-geometry-evidence-layer), [2026-07-18-dd2d-negative-packing-certificate](03-dd2d-v2.2.md#2026-07-18-dd2d-negative-packing-certificate), [2026-07-12-dd2d-integration-converter-not-native-env](03-dd2d-v2.2.md#2026-07-12-dd2d-integration-converter-not-native-env)
- **env-dd2d** — [2026-07-31-unified-coverage-waste-is-the-deployed-definition](06-v3-performance.md#2026-07-31-unified-coverage-waste-is-the-deployed-definition), [2026-07-27-cross-collection-grafting-coverage-mode](06-v3-performance.md#2026-07-27-cross-collection-grafting-coverage-mode), [2026-07-27-necessity-observed-not-predicted](06-v3-performance.md#2026-07-27-necessity-observed-not-predicted), [2026-07-26-dd2d-generator-pythonhashseed-dependent](05-v3-migration.md#2026-07-26-dd2d-generator-pythonhashseed-dependent), [2026-07-26-v3-migration-g0-g2](05-v3-migration.md#2026-07-26-v3-migration-g0-g2), [2026-07-25-v3-headline-reversal-was-training-artifact](04-comparison.md#2026-07-25-v3-headline-reversal-was-training-artifact), [2026-07-24-dd2d-comparison-retargeted-v3](04-comparison.md#2026-07-24-dd2d-comparison-retargeted-v3), [2026-07-24-vlmplan-baseline-protocol](04-comparison.md#2026-07-24-vlmplan-baseline-protocol), [2026-07-24-dd2d-collector-guarantees-exact-per-stratum-counts](04-comparison.md#2026-07-24-dd2d-collector-guarantees-exact-per-stratum-counts), [2026-07-24-grasp-internal-concave-grasps](04-comparison.md#2026-07-24-grasp-internal-concave-grasps), [2026-07-24-grasp-model-contacts-material](04-comparison.md#2026-07-24-grasp-model-contacts-material), [2026-07-19-demotion-signal-flag-default-observed](03-dd2d-v2.2.md#2026-07-19-demotion-signal-flag-default-observed), [2026-07-19-step-11-typed-evidence-harvest](03-dd2d-v2.2.md#2026-07-19-step-11-typed-evidence-harvest), [2026-07-19-reconstruct-never-regenerate](03-dd2d-v2.2.md#2026-07-19-reconstruct-never-regenerate), [2026-07-19-lambda-star-corrected-to-0-8](03-dd2d-v2.2.md#2026-07-19-lambda-star-corrected-to-0-8), [2026-07-18-gate-g0-passes-size-control-mandatory](03-dd2d-v2.2.md#2026-07-18-gate-g0-passes-size-control-mandatory), [2026-07-18-dd2d-negative-packing-certificate](03-dd2d-v2.2.md#2026-07-18-dd2d-negative-packing-certificate), [2026-07-12-dd2d-integration-converter-not-native-env](03-dd2d-v2.2.md#2026-07-12-dd2d-integration-converter-not-native-env)
- **env-stickbutton2d** — [2026-08-01-piginet-lifted-env-agnostic-package-per-env-adapters](07-stickbutton2d.md#2026-08-01-piginet-lifted-env-agnostic-package-per-env-adapters), [2026-08-01-both-evidence-classes-stay-wired-stickbutton2d](07-stickbutton2d.md#2026-08-01-both-evidence-classes-stay-wired-stickbutton2d), [2026-08-01-acyclic-pool-filter-pooled-stickbutton2d-v1](07-stickbutton2d.md#2026-08-01-acyclic-pool-filter-pooled-stickbutton2d-v1), [2026-07-31-unified-coverage-waste-is-the-deployed-definition](06-v3-performance.md#2026-07-31-unified-coverage-waste-is-the-deployed-definition), [2026-07-29-stickbutton2d-heuristic-distance-term](06-v3-performance.md#2026-07-29-stickbutton2d-heuristic-distance-term), [2026-07-28-stickbutton2d-subclass-plan-generator](06-v3-performance.md#2026-07-28-stickbutton2d-subclass-plan-generator)
- **baselines** — [2026-08-01-piginet-lifted-env-agnostic-package-per-env-adapters](07-stickbutton2d.md#2026-08-01-piginet-lifted-env-agnostic-package-per-env-adapters), [2026-07-28-state-delta-deployed-3-seed-protocol](06-v3-performance.md#2026-07-28-state-delta-deployed-3-seed-protocol), [2026-07-25-vlmplan-v3-test-split-two-arms](04-comparison.md#2026-07-25-vlmplan-v3-test-split-two-arms), [2026-07-25-v3-headline-reversal-was-training-artifact](04-comparison.md#2026-07-25-v3-headline-reversal-was-training-artifact), [2026-07-24-dd2d-comparison-retargeted-v3](04-comparison.md#2026-07-24-dd2d-comparison-retargeted-v3), [2026-07-24-vlmplan-baseline-protocol](04-comparison.md#2026-07-24-vlmplan-baseline-protocol), [2026-07-20-dd2d-comparison-notebook-piginet-bce](03-dd2d-v2.2.md#2026-07-20-dd2d-comparison-notebook-piginet-bce), [2026-06-11-b6-higher-horizons-incremental-scoring](01-foundations.md#2026-06-11-b6-higher-horizons-incremental-scoring), [2026-06-08-dp-on-counts-b6-baseline](01-foundations.md#2026-06-08-dp-on-counts-b6-baseline)
- **tooling** — [2026-08-01-piginet-lifted-env-agnostic-package-per-env-adapters](07-stickbutton2d.md#2026-08-01-piginet-lifted-env-agnostic-package-per-env-adapters), [2026-07-27-cross-collection-grafting-coverage-mode](06-v3-performance.md#2026-07-27-cross-collection-grafting-coverage-mode), [2026-07-25-vlmplan-v3-test-split-two-arms](04-comparison.md#2026-07-25-vlmplan-v3-test-split-two-arms), [2026-07-23-adaptive-traces-persist-step-scores](04-comparison.md#2026-07-23-adaptive-traces-persist-step-scores), [2026-07-20-dd2d-comparison-notebook-piginet-bce](03-dd2d-v2.2.md#2026-07-20-dd2d-comparison-notebook-piginet-bce), [2026-06-11-b6-higher-horizons-incremental-scoring](01-foundations.md#2026-06-11-b6-higher-horizons-incremental-scoring), [2026-06-07-analysis-notebook-converted-marimo](01-foundations.md#2026-06-07-analysis-notebook-converted-marimo)
- **infra** — [2026-07-18-modernize-pin-substrate-deps](03-dd2d-v2.2.md#2026-07-18-modernize-pin-substrate-deps), [2026-06-04-silo-refactor-scope-placement](01-foundations.md#2026-06-04-silo-refactor-scope-placement)
- **process** — [2026-07-27-margin-must-be-compared-to-seed-sd](06-v3-performance.md#2026-07-27-margin-must-be-compared-to-seed-sd), [2026-07-19-reconstruct-never-regenerate](03-dd2d-v2.2.md#2026-07-19-reconstruct-never-regenerate), [2026-07-19-decouple-harvest-from-collection](03-dd2d-v2.2.md#2026-07-19-decouple-harvest-from-collection), [2026-06-25-direction-pivot-representation-question](02-pivot.md#2026-06-25-direction-pivot-representation-question), [2026-06-06-documentation-discipline-codified](01-foundations.md#2026-06-06-documentation-discipline-codified), [2026-06-06-dated-writeup-snapshots](01-foundations.md#2026-06-06-dated-writeup-snapshots), [2026-06-04-silo-refactor-scope-placement](01-foundations.md#2026-06-04-silo-refactor-scope-placement)

## ID resolution

Where each gate / revision / prediction / constraint is decided.

| ID | Decided in |
|---|---|
| `A6` | [2026-07-27-record-tokens-are-ignored-at-inference](06-v3-performance.md#2026-07-27-record-tokens-are-ignored-at-inference) |
| `A15` | [2026-07-27-cross-collection-grafting-coverage-mode](06-v3-performance.md#2026-07-27-cross-collection-grafting-coverage-mode) |
| `C2` | [2026-07-27-necessity-observed-not-predicted](06-v3-performance.md#2026-07-27-necessity-observed-not-predicted) |
| `C5` | [2026-07-27-necessity-observed-not-predicted](06-v3-performance.md#2026-07-27-necessity-observed-not-predicted) |
| `D2` | [2026-07-26-necessity-conditioning-cut](05-v3-migration.md#2026-07-26-necessity-conditioning-cut) |
| `D4` | [2026-07-26-necessity-conditioning-cut](05-v3-migration.md#2026-07-26-necessity-conditioning-cut) |
| `D-7` | [2026-07-26-v3-migration-g0-g2](05-v3-migration.md#2026-07-26-v3-migration-g0-g2) |
| `D-8` | [2026-07-28-state-delta-on-record-ties](06-v3-performance.md#2026-07-28-state-delta-on-record-ties), [2026-07-26-v3-migration-g0-g2](05-v3-migration.md#2026-07-26-v3-migration-g0-g2) |
| `G0` | [2026-07-26-v3-migration-g0-g2](05-v3-migration.md#2026-07-26-v3-migration-g0-g2), [2026-07-18-gate-g0-passes-size-control-mandatory](03-dd2d-v2.2.md#2026-07-18-gate-g0-passes-size-control-mandatory) |
| `G1` | [2026-07-26-v3-migration-g0-g2](05-v3-migration.md#2026-07-26-v3-migration-g0-g2) |
| `G2` | [2026-07-26-v3-migration-g0-g2](05-v3-migration.md#2026-07-26-v3-migration-g0-g2) |
| `G6` | [2026-07-27-record-tokens-are-ignored-at-inference](06-v3-performance.md#2026-07-27-record-tokens-are-ignored-at-inference), [2026-07-26-selection-metric-never-censored](05-v3-migration.md#2026-07-26-selection-metric-never-censored) |
| `G6b` | [2026-07-26-selection-metric-never-censored](05-v3-migration.md#2026-07-26-selection-metric-never-censored) |
| `G7` | [2026-07-27-necessity-observed-not-predicted](06-v3-performance.md#2026-07-27-necessity-observed-not-predicted) |
| `G8` | [2026-07-27-necessity-observed-not-predicted](06-v3-performance.md#2026-07-27-necessity-observed-not-predicted) |
| `L2` | [2026-07-27-necessity-observed-not-predicted](06-v3-performance.md#2026-07-27-necessity-observed-not-predicted) |
| `L4` | [2026-07-27-dead-is-a-length-proxy](06-v3-performance.md#2026-07-27-dead-is-a-length-proxy) |
| `P4` | [2026-07-19-reconstruct-never-regenerate](03-dd2d-v2.2.md#2026-07-19-reconstruct-never-regenerate) |
| `P5` | [2026-07-19-step-11-typed-evidence-harvest](03-dd2d-v2.2.md#2026-07-19-step-11-typed-evidence-harvest) |
| `P16` | [2026-07-18-dd2d-negative-packing-certificate](03-dd2d-v2.2.md#2026-07-18-dd2d-negative-packing-certificate) |
| `P19` | [2026-07-18-dd2d-negative-packing-certificate](03-dd2d-v2.2.md#2026-07-18-dd2d-negative-packing-certificate) |
| `P-v3-1` | [2026-07-26-necessity-conditioning-cut](05-v3-migration.md#2026-07-26-necessity-conditioning-cut) |
| `P-v3-3` | [2026-07-27-necessity-observed-not-predicted](06-v3-performance.md#2026-07-27-necessity-observed-not-predicted) |
| `R1` | [2026-07-26-v3-migration-g0-g2](05-v3-migration.md#2026-07-26-v3-migration-g0-g2) |
| `R2` | [2026-07-26-v3-migration-g0-g2](05-v3-migration.md#2026-07-26-v3-migration-g0-g2) |
| `R7` | [2026-07-27-necessity-observed-not-predicted](06-v3-performance.md#2026-07-27-necessity-observed-not-predicted) |
| `R8` | [2026-07-26-selection-metric-never-censored](05-v3-migration.md#2026-07-26-selection-metric-never-censored) |
| `R9` | [2026-07-26-v3-migration-g0-g2](05-v3-migration.md#2026-07-26-v3-migration-g0-g2) |

## Do not quote

Conclusions and numbers that later entries retracted or replaced. Check here before citing any figure from a historical entry.

| Entry | Status | What replaced it |
|---|---|---|
| [2026-07-28-stickbutton2d-subclass-plan-generator](06-v3-performance.md#2026-07-28-stickbutton2d-subclass-plan-generator) | **partly superseded** | [2026-07-29-stickbutton2d-heuristic-distance-term](06-v3-performance.md#2026-07-29-stickbutton2d-heuristic-distance-term) |
| [2026-07-26-v3-migration-g0-g2](05-v3-migration.md#2026-07-26-v3-migration-g0-g2) | amended | ⚠️ **AMENDED** — the 13.68-vs-14.50 oracle discrepancy recorded here was double canonicalization, and the corrected figu |
| [2026-07-24-dd2d-comparison-retargeted-v3](04-comparison.md#2026-07-24-dd2d-comparison-retargeted-v3) | **partly superseded** | [2026-07-25-v3-headline-reversal-was-training-artifact](04-comparison.md#2026-07-25-v3-headline-reversal-was-training-artifact) |
| [2026-07-24-vlmplan-baseline-protocol](04-comparison.md#2026-07-24-vlmplan-baseline-protocol) | **partly superseded** | [2026-07-25-vlmplan-v3-test-split-two-arms](04-comparison.md#2026-07-25-vlmplan-v3-test-split-two-arms) |
| [2026-07-20-dd2d-comparison-notebook-piginet-bce](03-dd2d-v2.2.md#2026-07-20-dd2d-comparison-notebook-piginet-bce) | **partly superseded** | [2026-07-23-adaptive-traces-persist-step-scores](04-comparison.md#2026-07-23-adaptive-traces-persist-step-scores) |
| [2026-07-18-gate-g0-passes-size-control-mandatory](03-dd2d-v2.2.md#2026-07-18-gate-g0-passes-size-control-mandatory) | **partly superseded** | [2026-07-19-lambda-star-corrected-to-0-8](03-dd2d-v2.2.md#2026-07-19-lambda-star-corrected-to-0-8) |

## Legacy citation resolution

Code docstrings cite this log as `` `decisions.md` <date> ``. Dates collide, so this table resolves each to the entries on that date.

| Cited date | Entries |
|---|---|
| 2026-08-01 | [2026-08-01-piginet-lifted-env-agnostic-package-per-env-adapters](07-stickbutton2d.md#2026-08-01-piginet-lifted-env-agnostic-package-per-env-adapters), [2026-08-01-both-evidence-classes-stay-wired-stickbutton2d](07-stickbutton2d.md#2026-08-01-both-evidence-classes-stay-wired-stickbutton2d), [2026-08-01-acyclic-pool-filter-pooled-stickbutton2d-v1](07-stickbutton2d.md#2026-08-01-acyclic-pool-filter-pooled-stickbutton2d-v1) |
| 2026-07-31 | [2026-07-31-unified-coverage-waste-is-the-deployed-definition](06-v3-performance.md#2026-07-31-unified-coverage-waste-is-the-deployed-definition) |
| 2026-07-30 | [2026-07-30-proof-tier-demotion-cut-deployed-method-v3](06-v3-performance.md#2026-07-30-proof-tier-demotion-cut-deployed-method-v3) |
| 2026-07-29 | [2026-07-29-stickbutton2d-heuristic-distance-term](06-v3-performance.md#2026-07-29-stickbutton2d-heuristic-distance-term) |
| 2026-07-28 | [2026-07-28-stickbutton2d-subclass-plan-generator](06-v3-performance.md#2026-07-28-stickbutton2d-subclass-plan-generator), [2026-07-28-state-delta-deployed-3-seed-protocol](06-v3-performance.md#2026-07-28-state-delta-deployed-3-seed-protocol), [2026-07-28-state-delta-on-record-ties](06-v3-performance.md#2026-07-28-state-delta-on-record-ties) |
| 2026-07-27 | [2026-07-27-cross-collection-grafting-coverage-mode](06-v3-performance.md#2026-07-27-cross-collection-grafting-coverage-mode), [2026-07-27-necessity-observed-not-predicted](06-v3-performance.md#2026-07-27-necessity-observed-not-predicted), [2026-07-27-dead-is-a-length-proxy](06-v3-performance.md#2026-07-27-dead-is-a-length-proxy), [2026-07-27-record-tokens-are-ignored-at-inference](06-v3-performance.md#2026-07-27-record-tokens-are-ignored-at-inference), [2026-07-27-evidence-needs-its-own-attention-channel](06-v3-performance.md#2026-07-27-evidence-needs-its-own-attention-channel), [2026-07-27-margin-must-be-compared-to-seed-sd](06-v3-performance.md#2026-07-27-margin-must-be-compared-to-seed-sd) |
| 2026-07-26 | [2026-07-26-selection-metric-never-censored](05-v3-migration.md#2026-07-26-selection-metric-never-censored), [2026-07-26-necessity-conditioning-cut](05-v3-migration.md#2026-07-26-necessity-conditioning-cut), [2026-07-26-dd2d-generator-pythonhashseed-dependent](05-v3-migration.md#2026-07-26-dd2d-generator-pythonhashseed-dependent), [2026-07-26-v3-migration-g0-g2](05-v3-migration.md#2026-07-26-v3-migration-g0-g2) |
| 2026-07-25 | [2026-07-25-vlmplan-v3-test-split-two-arms](04-comparison.md#2026-07-25-vlmplan-v3-test-split-two-arms), [2026-07-25-v3-headline-reversal-was-training-artifact](04-comparison.md#2026-07-25-v3-headline-reversal-was-training-artifact) |
| 2026-07-24 | [2026-07-24-dd2d-comparison-retargeted-v3](04-comparison.md#2026-07-24-dd2d-comparison-retargeted-v3), [2026-07-24-vlmplan-baseline-protocol](04-comparison.md#2026-07-24-vlmplan-baseline-protocol), [2026-07-24-dd2d-collector-guarantees-exact-per-stratum-counts](04-comparison.md#2026-07-24-dd2d-collector-guarantees-exact-per-stratum-counts), [2026-07-24-grasp-internal-concave-grasps](04-comparison.md#2026-07-24-grasp-internal-concave-grasps), [2026-07-24-grasp-model-contacts-material](04-comparison.md#2026-07-24-grasp-model-contacts-material) |
| 2026-07-23 | [2026-07-23-adaptive-traces-persist-step-scores](04-comparison.md#2026-07-23-adaptive-traces-persist-step-scores) |
| 2026-07-20 | [2026-07-20-dd2d-comparison-notebook-piginet-bce](03-dd2d-v2.2.md#2026-07-20-dd2d-comparison-notebook-piginet-bce) |
| 2026-07-19 | [2026-07-19-demotion-signal-flag-default-observed](03-dd2d-v2.2.md#2026-07-19-demotion-signal-flag-default-observed), [2026-07-19-v2-ranker-fix-length-bias-generalizably](03-dd2d-v2.2.md#2026-07-19-v2-ranker-fix-length-bias-generalizably), [2026-07-19-step-11-typed-evidence-harvest](03-dd2d-v2.2.md#2026-07-19-step-11-typed-evidence-harvest), [2026-07-19-reconstruct-never-regenerate](03-dd2d-v2.2.md#2026-07-19-reconstruct-never-regenerate), [2026-07-19-lambda-star-corrected-to-0-8](03-dd2d-v2.2.md#2026-07-19-lambda-star-corrected-to-0-8), [2026-07-19-decouple-harvest-from-collection](03-dd2d-v2.2.md#2026-07-19-decouple-harvest-from-collection) |
| 2026-07-18 | [2026-07-18-gate-g0-passes-size-control-mandatory](03-dd2d-v2.2.md#2026-07-18-gate-g0-passes-size-control-mandatory), [2026-07-18-schema-geometry-evidence-layer](03-dd2d-v2.2.md#2026-07-18-schema-geometry-evidence-layer), [2026-07-18-dd2d-negative-packing-certificate](03-dd2d-v2.2.md#2026-07-18-dd2d-negative-packing-certificate), [2026-07-18-modernize-pin-substrate-deps](03-dd2d-v2.2.md#2026-07-18-modernize-pin-substrate-deps) |
| 2026-07-12 | [2026-07-12-dd2d-integration-converter-not-native-env](03-dd2d-v2.2.md#2026-07-12-dd2d-integration-converter-not-native-env) |
| 2026-06-25 | [2026-06-25-direction-pivot-representation-question](02-pivot.md#2026-06-25-direction-pivot-representation-question) |
| 2026-06-11 | [2026-06-11-b6-higher-horizons-incremental-scoring](01-foundations.md#2026-06-11-b6-higher-horizons-incremental-scoring) |
| 2026-06-08 | [2026-06-08-dp-on-counts-b6-baseline](01-foundations.md#2026-06-08-dp-on-counts-b6-baseline) |
| 2026-06-07 | [2026-06-07-analysis-notebook-converted-marimo](01-foundations.md#2026-06-07-analysis-notebook-converted-marimo), [2026-06-07-uncensored-evaluation-at-pool-cap](01-foundations.md#2026-06-07-uncensored-evaluation-at-pool-cap) |
| 2026-06-06 | [2026-06-06-documentation-discipline-codified](01-foundations.md#2026-06-06-documentation-discipline-codified), [2026-06-06-dated-writeup-snapshots](01-foundations.md#2026-06-06-dated-writeup-snapshots) |
| 2026-06-04 | [2026-06-04-silo-refactor-scope-placement](01-foundations.md#2026-06-04-silo-refactor-scope-placement) |

<!--END GENERATED-->
