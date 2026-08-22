# SPECTRE — Project Context

Authoritative context for the SPECTRE project (branch `adaptive` lineage).
Monorepo-general conventions live in the root `CLAUDE.md`; everything
spectre-specific lives here and in the imported docs:

@docs/proposal.md
@docs/decisions/README.md

## What this is

**Direction pivot, 2026-06-25 (see [`docs/proposal.md`](docs/proposal.md) §0 and
[`docs/decisions.md`](docs/decisions.md) 2026-06-25).** SPECTRE's contribution is
now a **representation question** for plan-feasibility prediction in
fully-observable, deterministic bilevel TAMP: *what should a feasibility
predictor represent skeletons and problems over?* The hypothesis (falsifiable, not
proven) is that a richer-than-pixels, cheaper-than-full-state representation
predicts refinement feasibility more sample-efficiently and with weaker perception
than a low-level (PIGINet-style) predictor over the concrete initial state — with
a **crossover** in the low-data / weak-perception regime (efficiency, not
information access). *Abstract-first* is the current leading candidate, one point
in a design space (learned latents, object-centric/graph features, invented
predicates, …). The **adaptive skeleton re-ranker** described below is now a
**secondary, composable** increment, not the headline: our own ablation
attributes only ~27% of the margin over B4 to failure-conditioning, the static
representation the rest (`docs/notebook.md` 2026-06-06/2026-06-25).

Mechanically, the SPECTRE re-ranker is a learned model for bilevel TAMP: given a
pool of candidate skeletons and the set of skeletons that have already failed
refinement, it picks the next skeleton to try. Candidate method = SPECTRE;
baselines are B1–B5 (random, default order, static-historical,
adaptive-historical, oracle) — never describe spectre-specific code or labels
as a "baseline". Re-ranker metric: mean time-to-first-success vs B4, mean ± std
over ≥ 3 seeds, evaluated uncensored at attempt budget 30 (= the candidate-pool
cap, so the budget never binds; [`decisions.md` 2026-06-07](docs/decisions/README.md)). Model selection
(`val_rollout_attempts`) stays at its own budget 20. RoutedTransport2D-n3-v1
(in-package) is the bespoke env behind the re-ranker results; under the pivot,
evaluation prefers **pre-existing environments meeting the representation-
advantage property wishlist** (`docs/proposal.md` §0), with bespoke still in
scope. ClutteredStorage2D-b5/b7 and StickButton2D-b5 collections are historical.

## Where everything lives

| Piece | Path |
|---|---|
| Package (model, dataset, collection, EDA) | `src/alphatamp/approaches/spectre/` — do not move; it IS `alphatamp.approaches.spectre` |
| RT2D environment (archived 2026-08-12) | RoutedTransport2D and its re-ranker results are historical; the `envs/routedtransport2d/` env, tests and configs were removed from the tree in the publication refactor (kept only in the local pre-refactor snapshot). Design specs remain in `docs/archive/` |
| DD2D environment (migrated) + JSON→EpisodeRecord converter | `src/alphatamp/approaches/spectre/envs/dd2d/` (raw_v2 dataset + `MIGRATION_DD2D.md`); the migrated drawer env is `envs/dd2d/drawer/` (flattened from the confusingly-nested `envs/dd2d/dd2d/` in the 2026-08-12 refactor), with `spectre_operators.py` (drawer substrate) + `spectre_convert.py` (converter) at `envs/dd2d/`. Wired as env_variants `dd2d_v2`/`dd2d_v3`/`dd2d_v4` (re-collections after the grasp/instrumentation changes), **not** a native SesameModels env — see `docs/decisions.md` 2026-07-12 |
| **StickButton2D** — the second evaluation environment | `src/alphatamp/approaches/spectre/envs/stickbutton2d/` — thin adapters over kinder's own env and refiner: `heuristic.py` (geometry-aware A* + the acyclic pool filter), `scene_geometry.py`, `sampler.py`/`instrumented_refiner.py` (class-2 evidence), `strata.py` (pooled-variant problem ids), `geometry.py`, `diagnostics.py`. Collected as env_variant **`stickbutton2d_v1`** (b1/b2/b3/b5 pooled, button count = stratum; b10 dropped). Entry points `experiments/spectre/sb2d_{collect,baselines}.py` + `sb2d_finalize.sh` |
| **Restock3D** — the third evaluation environment (3D / **kinematic PyBullet**) | `src/alphatamp/approaches/spectre/envs/restock3d/` — kinematic PyBullet (MuJoCo-direct superseded 2026-08-14). Store floor cubes + tall blocks into single-object regions on **ONE shelf** with a **tall section (bottom)** + **short section (top)**. **FULLY-LATERAL REBUILD 2026-08-17** (ADR/notebook `decisions/07` / `notebook/07` 2026-08-17): three **disjoint x-bands** (buffer \| objects \| shelf, left→right) so the base slides laterally in a clear **southern corridor** and never crosses the object field — resolves the base **phase-through** for real. `kinematic_env.py` (`ObjectCentricRestock3DEnv`; **`check_base_collisions=True`**), `place_controller.py` (**front-grasp pick + translate-only region place for ALL objects** — cubes + blocks; translate-only preserves the axis-aligned floor orientation so cubes land **upright**, unlike the old analytic cube place which leaked the front grasp's 45° into the symmetric cube; `get_base_plan` **no fallback**), `models.py` (`Pick`/`Place(obj,region)`, `InRegion`/`Stored`, **no `Clear`**), `instrumented_refiner.py` (**real-collision** F2/F3 probes, observation-only; `grasp_blockers` front-grasp), `generator.py` (**region rejection sampling** — 0.6×0.6 band, 0.12 m exclusion radius, random object-type order, axis-aligned, deterministic), `oracle.py` (**south-to-north** store order), `region_geometry.py`, `strata.py`. **Feasibility = real PyBullet collision** (F3 = upright block collides the short-section ceiling; **reach-over** = front grasp reaches north over a nearer object → back object blocked until nearer ones cleared; naive order fails, south-to-north succeeds = "far is harder"). Env_variant **`restock3d_v1`** (`spectre/Restock3D-r{0..3}-v0`); oracle certifies sampled r0–r3 **4/4**. Entry points `experiments/spectre/restock3d_{stage0,collect,difficulty,demos}.py`. **Taxonomy = F2 + F3 + reach-over; F1 clutter RETIRED** (the front grasp is not obstructed by a floor neighbour at the grasp config — verified by sweep; blockers dropped, `CLUTTER_PER_STRATUM=0`, buffer/relocation machinery kept **inert**). **Deferred:** eager `reach_blockers` relation + K_max re-calibration, coverage/waste, learned baselines, full collection. **v2 — continuous-packing variant (2026-08-17, ADR/notebook `decisions/07`/`notebook/07` 2026-08-17; additive, v1 untouched/coexists):** placement is a **continuous geometric packing** instead of discrete region assignment — **two place operators** `place_tall`/`place_short` (identical abstract effects `add {HandEmpty, Stored}`; the tall/short choice is a symbolic token validated by real collision, `place_short(block)`→F3), **uniform x-band sampling** (analytic band `[0.139, 0.661]` = board minus 0.04 end margin) with no discrete regions, predicates `{HandEmpty, Holding, OnFloor, Stored}` (**`InRegion` dropped**, `Stored` purely geometric), crowding is emergent (overlap→collision→resample). New modules `section_geometry.py`/`models_v2.py`/`place_controller_v2.py` (`SectionFrontPlaceController` reuses v1's translate-only place via a synthetic internal region)/`oracle_v2.py` (section assignment + south-to-north + manual rollout-with-resampling certifier, no collection pipeline); env fed 2 wide `RegionInfo` bands so `kinematic_env.py` is unchanged. Entry points `experiments/spectre/restock3d_v2_{stage0,oracle,demos}.py`; tests `test_restock3d_v2_{geometry,models,oracle}.py`. Milestone gates pass (Stage-0 4/4; oracle certifies r0–r3). **v2 geometry-informed plan-generation prior (2026-08-18, ADR/notebook `decisions/07`/`notebook/07` 2026-08-18):** `plan_generator_v2.py` (`GeometryGuidedRestockPlanGenerator`) subclasses the stock hff generator with a nearest-first **pick cost** `1+lambda*|{o' unpicked OnFloor : d(o')<d(o)}|` (lambda=1, `d(o)`=object y = northward reach; total plan penalty = Kendall-tau vs the south-to-north oracle order), overriding only the edge cost. Eval `restock3d_v2_heuristic_eval.py` (10 r3 problems, **enumeration only, no refinement**; match = oracle pick order + both talls `place_tall`) generates the oracle plan in **~15-26 attempts / 100%** vs geometry-blind hff **~4000 / 50-80%** (~150-275x, ~200x central, over 3 K=10000 replicates; hff is *worse* than uniform-random - clustered A* enumeration). Per-problem indices are `PYTHONHASHSEED`-dependent, so quote **aggregates**. A plan-generation prior, distinct from the deferred eager section-capacity heuristic. **Sweep** `restock3d_v2_heuristic_sweep.py` (parallel, ~25 workers / 0.8x CPU, single-thread BLAS) runs both metrics over a (n_tall x n_short)=1-4x1-4 grid (5-per-section overflows): Table A geometry plan-gen attempts (100% success; mean ~2x per tall, heavy-tailed) + Table B oracle-plan refinement solve-rate/wall-clock (`refine_skeleton_v2` gained an optional `max_seconds` cap; 18 retries/step, 90s cap never bound). Solve-rate degrades with objects-per-section (4+4=4/10, all failures genuine overflow), wall-clock 15-60s; results in `data/spectre/derived/restock3d_v2/heuristic_sweep_results.md` (`decisions/07`/`notebook/07` 2026-08-18). **v2 SPECTRE 3D collection pipeline — BUILT + verified 2026-08-18 (ADR/notebook `decisions/07`/`notebook/07` 2026-08-18; port map `docs/restock3d_port_map.md`):** the representation decision is a **full 3D point cloud** — cube and tall block share a 2D footprint and differ only in height (the F3 axis), so a 2D-footprint scene would be blind to it. Built as gated increments: `envs/restock3d/scene_geometry.py` (**analytic axis-aligned-box point cloud** per object; `schema.ObjectGeometry` gained optional `point_cloud`/`pose_z`/`height`, `None`⇒2D envs byte-unchanged); a **config-gated 3D encoder widening** (`SpectreConfig.point_dim`/`pose_dim`, `--scene-3d`; `FootprintEncoder` input `Linear(2→3)`, `pose_proj(3→4)`, `sample_point_cloud`; inference derives it from `model.cfg.point_dim`, `load_checkpoint` from saved `scene_3d`); the **v2 instrumented refiner** (`_probe_place_v2` on `place_tall`/`place_short`, F3=ceiling collision, **F2=section-resident enumeration**); **collection registration** (`model_name="restock3d_v2"` collect.py dispatch, `Restock3DV2Env` gym wrapper, `strata_v2.py` banding-stratum 0–3 → committed `generator.STRATA` recipe keys 10–13 = 1×1..4×4, recipe in `model_kwargs` so `config_hash`+`git_sha` pin it, domain+aug entries, `conf/env/restock3d_v2_pilot.yaml`); the **v2 default pool generator = the geometry-guided prior** (feasible-first + F3 negatives). Pilot collector `experiments/spectre/restock3d_v2_collect.py` (env_variant **`restock3d_v2_pilot`**). **Verified end-to-end:** `collect_episode`→valid `EpisodeRecord` (pool, 3D geometry, F3 negatives), **SPECTRE `--scene-3d` train→`best.pt`→load(`point_dim=3`)→rollout**; tests `test_restock3d_v2_scene_geometry.py`/`test_scene_3d_widening.py`/`test_restock3d_v2_refiner.py` (407 fast green, 2D path unchanged). **Cost:** full-pool refinement is expensive (F3-heavy pools, real MP); calibrated to **18 retries + small `K_max`** (10 rejects even feasible skeletons); 4×4 is the slow/low-yield corner. **Baseline ports (2026-08-18):** shared oblique renderer `envs/restock3d/render.py` (height-visible; world→pixel projection for Set-of-Mark + per-object crops) DONE; PIGINet `baselines/piginet/restock_adapter.py` (+`--domain restock3d`) DONE; LAZY `make_lazy_domain` restock branch DONE (graph `geom_dim` 8→9 height feature **DONE 2026-08-18** — the F3 axis is visible to the GAT, cube 0.21 vs tall 1.0; DD2D/SB2D stay 8, checkpoint self-describes via `node_dim`); VLMPlan adapter+labeler pending. **v2 FULL COLLECTION pipeline — BUILT 2026-08-18 (ADR/notebook `decisions/07`/`notebook/07` 2026-08-18):** the real dataset is **5 strata × 50/15/15 = 250/75/75** — symmetric 2×2/3×3 + crowded 3×4/4×3/4×4 (env_variant **`restock3d_v2`**, recipe keys 11/12/14/15/13; **new asymmetric keys 14=3×4, 15=4×3**). **Per-stratum budgets** (`strata_v2.BUDGETS`): K_max **20/40/75/75/75**, r_cap **40/70/80/90 s** — re-calibrated on the **collection-path** feasible-solve tail, NOT the sweep's Table B (which timed the oracle certifier `refine_skeleton_v2`; collection uses `BacktrackingRefiner`, where **infeasible candidates do NOT fail fast** — backtracking re-descends — so per-problem cost ≈ K_max×r_cap; the run is ~20–30 h at ~29 workers). K_max=75 (not 100) on the crowded strata: Table A capture rates show 100 wasted on 3×4/4×4. **Banding collision fixed** (`V2_STRATUM_BAND = SPLIT_BAND//5` + a v2-local `stratum_of`, since a 5th stratum overflowed compare's `//4` band; shared consumers stay 4-stratum — route them to the v2 decoder before per-stratum analysis). Collector `restock3d_v2_collect.py` gained `--test`, per-stratum budgets, a **dynamic top-up reject-resample loop** (keep target in flight until target kept; 4×4 is ~40–55% packable) + per-stratum heartbeat + exact census trim. Verified pre-launch (banding injective, new gym ids build 7-obj, generator emits feasible skeletons for the asymmetric configs, 3×4 seed packable/oracle-45s, LAZY 9-dim height populates). Protocol = DD2D/SB2D (**no oracle in collection**); scope = collect + vocab + LAZY widening + low-epoch smoke-train all 3 (VLMPlan deferred). **`compare_envs` EnvSpec DONE 2026-08-19** (ADR/notebook `decisions/07`/`notebook/07` 2026-08-19): restock3D onboarded to `compare_methods.py` — SPECTRE (3D point-set + init/goal atoms fully ON: `--scene-3d --atom-mode profiles` + full PointSetEncoder), PIGINet (`piginet_s{seed}` seed axis), LAZY all trained + eval-cached vs the naive planner order. **Scope grows as strata land: now 2×2 + 3×3 + 4×3 (banding strata {0,1,3}), 3 seeds, refine-cap 55 s** (was {0,1}/50 s at onboarding 2026-08-19; ADR/notebook `decisions/07`/`notebook/07` **2026-08-20**). Result (3 seeds, ± across-seed; astar 8.78 ALL FP ≫ learned): **LAZY 0.19 ± 0.01 dominates** (4×3 Δ −3.57 CI[−5.93,−1.63] vs SPECTRE), **SPECTRE 1.44 edges PIGINet 1.96** — the gap is at the crowded **4×3** (SPECTRE 4.1 vs PIGINet 6.0, paired Δ −1.8 CI[−3.87,+0.10], first hint of the §0 representation crossover but CI still includes 0; 2×2/3×3 tied/near-trivial), and **adaptivity is inert** (static 1.44 ≈ adaptive 1.47); §2b wall-clock refinement-dominated (LAZY 53 s < SPECTRE 141 s < PIGINet 220 s ≪ astar 561 s). Wiring: `compare.stratum_of(seed, env_variant)` now routes `restock3d_v2` to the 5-stratum `strata_v2.stratum_of` (the shared 4-stratum decoder collapses 2×2/3×3), threaded into the SPECTRE/LAZY/PIGINet training-strata filters, the precompute cache stamper (+`{0,1}` test filter, `_REFINE_CAP_S=50`), and notebook §5; sim-free `render_scene_from_geometry` for the §5 visualizer. Fixed two latent scene_3d/atom bugs in `cache_spectre3` (warmup/static paths did not thread pointset/atom emission from `model.cfg`) and `cache_lazy` (default `geom_dim=8` not `domain.geom_dim=9`). **Still deferred:** eager section-capacity heuristic, VLMPlan full run (billed backend), the remaining crowded strata **3×4 + 4×4** (still collecting; 4×3 done 2026-08-20) — where more crowding + the asymmetric 3×4 should show whether the SPECTRE > PIGINet edge at 4×3 becomes significant. **v2 collection RE-ARCHITECTED to SEQUENTIAL per-stratum jobs (2026-08-19, ADR/notebook `decisions/07`/`notebook/07` 2026-08-19):** the single mixed 5-stratum job OOM-killed the desktop (28 workers × ~4 GB heavy peak > 59 GB) and its mixed-block per-worker peak was unpredictable. Now `experiments/spectre/restock3d_v2_run_all.sh` runs the collector **once per stratum, gated** (`strata_v2.SEQUENTIAL_ORDER=(0,1,3,2,4)`), so each process has one block count → uniform predictable RAM + full reclamation between jobs. **Per-stratum RAM-sized workers** `min(0.85·CPU, 0.85·freeRAM/PER_WORKER_GB[s])` floor-guarded (2×2≈27, 3×3≈15, heavy≈10; `PER_WORKER_GB={0:1.7,1:3.0,2:4.5…}` validated live by a new `wRSSmax` heartbeat + memory watchdog backstop). **Heavy strata halved to 25/10/10** (light stay 50/15/15) via per-stratum `strata_v2.SIZES` (replaces uniform `PER_CONFIG`): **175/60/60 = 295** (was 250/75/75). Collector auto-sizes + resumes via an on-disk **pre-scan** (budgets/generator/schema unchanged, so the 24 mixed-run episodes are retained). Est. ~1.5 days. Downstream is count-agnostic; the uneven sizing is safe. The user-uploaded `front-grasp-tall-block/` is reference-only (not imported). **v3 direction — calibration study 2026-08-20 (notebook `07` 2026-08-20; findings `docs/restock3d_v3_calibration.md`):** v2 proved **too easy** for the baselines (LAZY near-oracle), so v3 will vary **block x-widths** (selection matters) + **heights near the short/tall cutoff**. Before building v3, the standalone harness `experiments/spectre/restock3d_v3_calibrate.py` (3 process-isolated sweeps over the real controllers/env; no prod edits) mapped the pick/place envelope: **tall-section height 0.05–0.23 m; short section is CUBE-ONLY (0.05 m; gripper needs ~0.10 m above the block, so the 0.15 m short clearance is too tight for a height range — flagged for a v3 env change)**; **width capped by the finger aperture ≈92 mm** (the kinematic sim is width-permissive — attach-grasp excludes the target — so v3 must cap width in the generator, ~0.08 m safe); **left-to-right packing edge gap ≥60 mm** (empirical min 33–50 mm, ~5–8× the naive finger-thickness estimate); and the user's **center-grasp idea was rejected** (the production height-adaptive `front_grasp_transform` is best). Measurement only — v3 measurement was the pre-build study. **v3 BUILT through the pre-collection gates 2026-08-20 (ADR/notebook `decisions/07`/`notebook/07` 2026-08-20):** additive env_variant **`restock3d_v3`** (v2 frozen as negative control) — **per-object widths [0.02,0.08] + heights near the 0.12/0.17 cutoffs**, re-balanced **(0.27,0.22)** partition. New modules `feasibility_v3.py` (single source of truth: capacity formula `Σw+0.06(n−1)+2·0.04≤0.50`, cutoffs, split enumeration, greedy hand-rules, `classify_skeleton` emitting `failure_metadata`-shaped dicts), `generator_v3.py`+`strata_v3.py` (role-banded sampling + split-enumeration acceptance; **4 strata n=6/7/8/9 on the shared 4-band** so `compare.stratum_of` needs no edit), `place_controller_v3.py` (production L2R packer, state-reading slots), `models_v3.py` (reuses v2 operators/abstractor), `ObjectCentricRestock3DEnvV3` (per-seed body rebuild). Collection uses an **analytic refinability classifier** (pure geometry, no MP); the **real refiner stays the eval instrument** with a v3-only arm-insertion **F3-parity** probe in `instrumented_refiner._probe_place_v2` (guarded, v2 byte-identical). `collect.py`/`env_registry`/`domain`/`scene_geometry` wired for `restock3d_v3`; verified end-to-end via `collect_episode`. **Gates cleared:** G3 hard strata defeat both greedy rules 100% (culprits across 8–9 objects); G2 static ceiling 1.00 clear / ~0.88 near-threshold (not saturated); **G1 analytic↔real ~100% under a label-aware budget** (infeasible 53/53, feasible confirmed) — after correcting a flat-10 s-cap artifact (**real v3 refinement ~40 s/candidate → eval budget ≥~60 s**). Entry points `experiments/spectre/restock3d_v3_{calibrate_generator,gates}.py`; tests `test_restock3d_v3_{feasibility,packing_parity,cutoffs(slow),refiner_f3,generator}.py`. **Deferred to the collection pass:** real collection (analytic labels + 5% audit), training, comparison wiring (compare_envs EnvSpec, precompute cache, PIGINet v3 scene reconstruction); LAZY needs no edit (`startswith("restock3d")`→geom_dim 9). **v3 SYNTHETIC dataset + comparison DONE 2026-08-21 (ADR/notebook `decisions/07`/`notebook/07` 2026-08-21):** collected **fully synthetically** — `CollectionConfig.refiner_mode="analytic"` (new additive flag; `collect._restock3d_analytic_outcome` labels via `classify_skeleton`, synthesizes wall-clock: fail=r_cap, success=U[0.6,0.8]·r_cap), geometry-prior pool, `restock3d_v3_collect.py`+`restock3d_v3_run_all.sh` (one stratum/process, per-stratum workers 16/12/6/4, outer refill loop for memory-watchdog pauses), **400/100/100** over n=6/7/8/9 (`strata_v3.BUDGETS` K_max 40/60/150/200, r_cap 50/70/90/110; `SIZES` 100/25/25). Vocab→SPECTRE (deployed `--scene-3d --atom-mode profiles` recipe, 3 seeds, `checkpoints_spectre_atoms/restock3d_v3`)+PIGINet (`oracle_v3.build_v3_bundle` reconstruction, `restock_adapter` v3 `crops`/`_config_scales` branch)+LAZY (v3 normalizers `[0.08,0.05,0.004,1,0.17]`), 3 seeds each. Comparison = `RESTOCK3D_V3` EnvSpec (strata n=6..9, shared 4-band, `has_timing`) + `precompute_dd2d_cache` keys (`_PIGINET_PATHS`/`_V3_ARM_OVERRIDES`/`_REFINE_CAP_S=90`). **Result (3 seeds, test n=100): the §0 representation crossover appears DECISIVELY** — low-level **PIGINet 38.11±1.01 ≈ astar 38.41** (no better than the naive order), both abstract rankers ~3.4× better (**SPECTRE 11.11±0.98, LAZY 11.79±0.08**), paired SPECTRE−PIGINet **−27.00 [−32.97,−21.41]** *growing with crowding* (s2 −43, s3 −48); **SPECTRE≈LAZY** (−0.68 [−3.25,+1.84]), **adaptivity inert** (adaptive≈static +0.06). Far stronger than v2 (PIGINet 1.96≈SPECTRE 1.44) — v3 difficulty is capacity/height/**selection**. **Bug fixed:** `render.object_crops` crashed on the robot (no `pose_x`) → restock PIGINet crops were silently all-zero for v2 *and* v3 (dead image channel); now skips un-poseable objects; also **reuse one reconstruction bundle per stratum** (was leaking ~0.14 GB/env → OOM over 600 episodes). **coverage/waste verified live+correct on the analytic path** (culprit pool non-empty, polarity right, leakage-safe; F3-dominant so less signal than v2). **⚠️ SYNTHETIC — read FP + §2b as an upper bound**, not a real-refiner result (analytic labels = exact geometry, no MP noise, favours the geometry-encoding representation); real-refiner audit deferred. Entry points `restock3d_v3_{collect,run_all,train}.{py,sh}`; test `test_restock3d_v3_analytic_collect.py`. **v3 ADAPTIVITY REVIVED — coverage bug + `repeat` F3 certificate DONE 2026-08-21 (ADR/notebook `decisions/07`/`notebook/07` 2026-08-21; probe plan `docs/adaptivity_probe_plan_restock3d_v3.md`, fix-only pre-reg `docs/adaptivity_fix_only_prereg.md`):** the 2026-08-20 "adaptivity inert (adaptive≈static)" **and** "coverage/waste verified live+correct" claims were **both wrong** — a bug: `canonicalize._remap_refiner_metadata` coerced v3's F2/F4 `dev_added=None`/`dev_deleted=None`→`[]`, re-typing class-1 (culprits) records to class-2-empty-deviation so `blame()` dropped the culprits → pool `K` empty → **coverage/waste identically 0** on v3 (v3-only: DD2D omits the key, SB2D stores a real list; both byte-unchanged). Fix = `_rename_atoms` preserves `None` (+regression test; new invariant: **canonicalize must preserve `None` in `dev_added`/`dev_deleted`** — it is the class-1-vs-2 discriminator). Probes P0–P4 (no training): P0 census **F3=75% of failures & blameless**, F2 25%, F4~0%; **P2 oracle re-ranker ceiling** FP_static 11.05→**2.81** (75% headroom, 0 soundness violations), **P2b decomposition** F3-only `repeat`=**74%** / F2-only `regroup`=**1%** / F2-as-exact-step kills 263 successes (⇒ `blame==∅` gate load-bearing); blame-census ⇒ env-safe scope is `step_certificate ∧ provable ∧ blame==∅` (NOT "provable ∧ culprit-free" — DD2D 92% blameless-provable are means-failures). **New feature `repeat`** (deployed): F3 exact-step veto, gated by a **dedicated `QueryAxioms.step_certificate` flag** (NOT `proof_tier`, which also drives `dead`/token-holdout — reusing it would strip F2 from tokens; v3 `place_*` declare `step_certificate=True`, `proof_tier` stays False so `dead`/demotion/tokens byte-unchanged). Full plumbing (TrainConfig/argparse/`build_example`/`n_overlap_feats`×2/deploy path), backward-compat (old width-4 ckpts load `strict=True`), leakage-free (0 at |F|=0); tests `test_repeat_regroup.py`. **Result (test n=100, 3 seeds, adaptive; static ~12 every arm ⇒ purely adaptive):** knockout 11.11 (Δ+0.06 inert) → fix-only 12.18 (**Δ−0.09 still inert** — coverage speaks the ~1% F2/ordering channel) → **+repeat 3.13 (Δ−8.89 [−11.10,−6.80]; +repeat−fixonly −9.06 [−11.28,−6.95]), ~97% of the P2 ceiling**. **Deployed `--repeat-feats` alone** (`checkpoints_spectre_atoms_repeat`, `restock3d_v3_train.sh` + `_V3_ARM_OVERRIDES` repointed; SPECTRE now **3.13** ≫ LAZY 11.79 / PIGINet 38.11, was ≈LAZY). **`regroup` (F2 seating-chart, `grouping_certificate` flag) DEPRECATED/off** — adds nothing over repeat (+repeat+regroup 3.19≈3.13); the Stage-2 cross-env pre-check caught ungated regroup firing 42%/11.6%-wrong-polarity on DD2D (culprits there are blockers you *want* to stage) → `grouping_certificate` gate makes it inert on DD2D/SB2D. Both features gracefully degrade (inert where no schema declares them). **Standing:** the "inert adaptivity" was an evidence-*language* mismatch (v3's decision is grouping/assignment; load-bearing signal is the blameless F3 certificate, not ordering coverage/waste), not an adaptivity ceiling. **⚠️ still analytic-synthetic** — magnitudes upper-bound; real-refiner audit deferred. Deferred: `repeat` on DD2D proof-tier `retrieve`; DD2D/SB2D non-regression retrains; removing `regroup`. |
| VLMPlan baseline (zero-shot VLM planner) | `src/alphatamp/approaches/spectre/baselines/vlmplan/` — env-agnostic core; per-env `{dd2d,sb2d}_adapter.py` (prompt + grounding) and `sb2d_label.py` (off-pool labeler), dispatched by `registry.py`; entry points `experiments/spectre/vlmplan_{run,score}.py`. Protocol: `decisions/04` 2026-07-24; env-agnostic refactor: `decisions/07` 2026-08-01; prompt deviations: `baselines/vlmplan/prompts/PROVENANCE.md` |
| PIGINet baseline (low-level predictor) | `src/alphatamp/approaches/spectre/baselines/piginet/` — env-agnostic core behind a `PIGINetDomain` protocol; per-env `{dd2d,sb2d}_adapter.py`. `decisions/07` 2026-08-01 |
| LAZY baseline (learned adaptive re-ranker) | `src/alphatamp/approaches/spectre/baselines/lazy/` — Khodeir et al re-implemented over the fixed pool: prefix-tree GAT policy (`torch_geometric` GATv2Conv) + online feasibility ϕ. Trained via `experiments/spectre/lazy_train.py`; cached by `cache_lazy` in `precompute_dd2d_cache.py`; registered as `LAZY_FAMILIES` in `compare.py`. `decisions/07` 2026-08-09; `baselines/lazy/PROVENANCE.md` |
| **Method comparison** — one notebook, N environments | `experiments/spectre/compare_methods.py` (marimo) over `compare.py` (loaders, rollout sim, bootstrap) and `compare_envs.py` (**the env registry — a new environment is one `EnvSpec`**). `decisions/07` 2026-08-01 |
| Docs (living proposal, lit review, archived specs + dated writeup snapshots) | `src/alphatamp/approaches/spectre/docs/` |
| **ADR log** and **lab notebook** — chaptered by era, newest first | `docs/decisions/` and `docs/notebook/`, each with a **generated `README.md`** (chapter list, full entry ledger, by-track index, ID-resolution table, do-not-quote ledger, legacy date→entry map). Pre-split monoliths frozen in `docs/archive/*_monolithic.md`; `docs/decisions.md` / `docs/notebook.md` are stubs. Tooling: `doclog.py` + `experiments/spectre/decisions_index.py` |
| Hydra entry points + configs + SLURM launchers + analysis notebook | `experiments/spectre/` (configs under `experiments/spectre/conf/`) |
| Tests | `tests/approaches/spectre/` |
| Data (gitignored) | `data/spectre/{raw,derived,checkpoints,configs}/` — the `data_root: "data/spectre"` convention is relative to the repo root |
| SLURM logs | `experiments/slurm_outputs/` (shared scratch, gitignored) |

Spectre's Hydra configs are self-contained: `experiments/spectre/conf/`
holds `spectre_collect.yaml`, `spectre_build_vocab.yaml`, `spectre_train.yaml`,
`dd2d_convert.yaml`, the spectre-only env group
(`env/{dd2d_v2,dd2d_v3,dd2d_v4,stickbutton2d_v1,stickbutton2d_b1..b5}.yaml`; the
RT2D/ClutteredStorage2D env configs were archived in the 2026-08-12 refactor),
and spectre's own `hydra/launcher/slurm.yaml` (8 cpus / 32 GB). The shared
`experiments/conf/` tree belongs to other projects — never put spectre configs
there.

## Compute resources (dev workstation)

Primary dev/training box as of 2026-07-18 (replaces the earlier MacBook M3 Pro /
MPS setup; the SLURM launchers below remain for cluster runs):

- **GPU — NVIDIA RTX 5090, 32 GB VRAM, Blackwell (sm_120).** Driver 595.71,
  CUDA 13.2 runtime; driver-only, no `nvcc`/CUDA toolkit (fine for prebuilt
  wheels). Single GPU → one training run at a time; multi-seed sweeps run
  sequentially or share the card. Training goes on CUDA — unlike the old
  CPU/MPS box, so watch for code that hard-codes `cpu`/`mps` or mixes devices.
- **PyTorch must be the cu130 build** (`torch==2.13.0+cu130`), installed with
  `uv pip install torch --index-url https://download.pytorch.org/whl/cu130`
  **before** `uv pip install -e ".[develop,ttd]"`. cu130 is the
  actively-maintained line with native sm_120 support and matches the CUDA 13.2
  driver. `pyproject.toml` keeps `torch` unpinned (shared with SLURM / other
  machines), so the cu130 index is applied at install time, not baked in — if an
  editable reinstall ever pulls a PyPI-default torch, re-run the cu130 install
  and re-verify with a real device op (`(x@x)` on `cuda`), not just
  `torch.cuda.is_available()`.
- **CPU — AMD Ryzen 9 9950X, 16 cores / 32 threads** (~5.75 GHz boost). Local
  data collection / worker-parallel stages (`spectre_collect.py`, EDA) can use
  far more workers than the SLURM launcher's 8 cpus / 32 GB.
- **RAM ~64 GB** (59 GiB usable) + 14 GiB swap. **Disk ~1.2 TB free** on `/` for
  datasets/checkpoints under `data/spectre/`.
- **OS/toolchain:** Ubuntu 26.04 LTS, Python 3.11 venv (`.venv`, uv-managed),
  uv 0.11.29. Substrate dep pins were modernized on 2026-07-18
  (`decisions.md` that date) to resolve on a fresh machine — kindergarden 0.2.0,
  prpl-mono `e215d1fc`, kinder-baselines `4c731dc8`.

## Pipeline & how to run

Always `source .venv/bin/activate` first; run from the repo root. Stages in
order (details in @docs/proposal.md §4–5; respect the de-risking gates):

> **2026-08-12 refactor note.** RT2D and ClutteredStorage2D were archived and their env
> configs/scripts removed; the live env_variants are `dd2d_v4` and `stickbutton2d_v1`. The
> `env=routedtransport2d_n3_v1` examples below illustrate the *stage flow* — substitute a
> live variant (`env=dd2d_v4`, or the SB2D flow via `sb2d_finalize.sh`). Renamed/archived
> since: `spectre_score_v3.py`→`spectre_score.py`; the sanity-check
> (`spectre_check_pipeline.py`), the atom-sensitivity probe, the RT2D collect wrapper, and
> the `analyze_spectre.py` marimo notebook were archived — the analysis notebook is now
> `compare_methods.py` (stage 7).

1. **Collect** (500 train / 100 val / 100 test per env):
   `python experiments/spectre/spectre_collect.py env=routedtransport2d_n3_v1 split=train problem_seed_start=0 problem_seed_end=500`
   — or `bash experiments/spectre/collect_routedtransport2d_n3_v1.sh` (all
   three splits locally), or `sbatch experiments/spectre/spectre_collect.slurm …`
   / `bash experiments/spectre/submit_spectre_<env>.sh` on a cluster.
   **DD2D (`env=dd2d_v2`) skips this stage:** it has no native SPECTRE env — run
   `python experiments/spectre/dd2d_convert.py` instead to convert the migrated
   `envs/dd2d/data/dd2d/raw_v2` JSON dataset into `data/spectre/raw/dd2d_v2/…`
   episodes. To generate *fresh* DD2D data, run DD2D's own collector
   (`python -m alphatamp.approaches.spectre.envs.dd2d.drawer.collect --out-root …`,
   needs shapely + the planners) and re-run the converter pointed at its output.
   Stages 2–4 below then work unchanged with `env=dd2d_v2`.
2. **Vocab** (train split only, OOV-checks val/test):
   `python experiments/spectre/spectre_build_vocab.py env=routedtransport2d_n3_v1`
3. **Sanity-check** the collection + one collated batch — the `spectre_check_pipeline.py`
   script was archived in the 2026-08-12 refactor; the SB2D flow folds its checks into
   `sb2d_finalize.sh`.
4. **Train** (the deployed recipe; multi-seed):
   `python -m alphatamp.approaches.spectre.train --env dd2d_v4 --seed 0 --overlap-mode jaccard --coverage-feats --aggregate-records --evidence-attn --state-delta --select-window 5`
   — or, concurrently across seeds, `python experiments/spectre/spectre_sweep.py --preset v3final --seeds 0 1 2 --env dd2d_v4`; the SB2D flow is `bash experiments/spectre/sb2d_finalize.sh`. (The v1-era Hydra wrapper `spectre_train.py` / `spectre_train.slurm` + `conf/spectre_train.yaml` were archived in the 2026-08-12 refactor — the deployed training was always the argparse `train` module + `spectre_sweep.py`.)
5. **Analyze / experiments:** the analysis notebook is
   `experiments/spectre/compare_methods.py` (marimo — see stage 7); it drives `eda.py`
   (EDA gates, B1–B5 brackets, rollout simulation) and `compare.py`/`compare_envs.py`
   (the method-comparison table). The `analyze_spectre.py` notebook and the
   `spectre_probe_atom_sensitivity.py` diagnostic were archived in the 2026-08-12 refactor.
6. **VLMPlan baseline** (zero-shot VLM comparison row; two stages, only the first needs a
   model, so a re-collection re-runs just the second):
   ```bash
   lms server start   # or any OpenAI-compatible server (vLLM)
   export OPENAI_BASE_URL=http://localhost:1234/v1 OPENAI_API_KEY=lm-studio
   python experiments/spectre/vlmplan_run.py   env=dd2d_v3 split=train n_problems=5 run=pilot
   python experiments/spectre/vlmplan_score.py env=dd2d_v3 split=train n_problems=5 run=pilot
   # StickButton2D: a named config, because the 32B arm needs max_tokens=12288
   python experiments/spectre/vlmplan_run.py   --config-name vlmplan_sb2d_32b
   python experiments/spectre/vlmplan_score.py --config-name vlmplan_sb2d_32b
   ```
   **Frontier arm (`gpt-5.6-terra`, the headline VLMPlan row, 2026-08-08).** Named configs
   `vlmplan_{dd2d,sb2d}_terra` set `backend: openai_responses` (GPT-5 reasoning models need
   the Responses API — chat completions rejects `max_tokens`/`temperature`),
   `base_url: https://api.openai.com/v1` (billed) and `reasoning.effort: low`; export a real
   `OPENAI_API_KEY` first. **terra replaces the weaker `gpt-5.6-luna`** and is generated
   *with* the gripper-geometry disclosure (`prompts/PROVENANCE.md` deviation 9), which
   together roughly halve FP: **DD2D 62.98→35.23** (now ~tied with astar 34.52; was the worst
   method), **SB2D 11.85→6.42** (self-solves 39/40, 0 censored). A full-scale
   **medium-effort DD2D arm confirmed low** (33.5 vs 35.23, paired 95% CI [−18.6, +15.1]), so
   `effort: low` stands (also matching luna for a clean tier swap). They run **native** on
   `dd2d_v4` / `stickbutton2d_v1_kinder`, a **stratified 40** (`stratified_per_stratum: 10`,
   stride-not-truncate), and save the exact scene image sent to
   `…/vlmplan/<run>/images/<pid>.png`. SB2D uses `image_source: kinder_labeled` — kinder's
   real pixels with Set-of-Mark labels overlaid (unlabeled kinder discs are unusable by a
   VLM). **Pilot on ≥1 problem per stratum** (`stratified_per_stratum=1`, its own
   `run`/`cache_subdir`) and watch `n_truncated` — a 1-per-stratum DD2D pilot draws only
   easy-mode (trivially-graspable) problems and badly under-estimates FP (0 vs the true ~27).
   ```bash
   export OPENAI_API_KEY=sk-...   # a real key; this arm is billed (~$5-10 for both envs)
   python experiments/spectre/vlmplan_run.py   --config-name vlmplan_dd2d_terra
   python experiments/spectre/vlmplan_score.py --config-name vlmplan_dd2d_terra
   python experiments/spectre/vlmplan_run.py   --config-name vlmplan_sb2d_terra
   python experiments/spectre/vlmplan_score.py --config-name vlmplan_sb2d_terra
   ```
   The score record now carries wall-clock (`infer_s` VLM generation + `refine_s`), so
   VLMPlan appears in §2b; `VLMPlan-GPT5.6` is a `SEQUENCE_METHOD` in `compare.py`.
   Adapter *and* off-pool labeler are dispatched on `env_variant` by `baselines/vlmplan/registry.py`;
   a new environment is an `EnvAdapter` + a `Labeler`, both registered there.
   One `cache_subdir` is one method row — give a different `run` its own `cache_subdir`
   or the rows get averaged together (guarded, not silent). Check the printed
   **label-agreement gate** before trusting any number: below ~0.95 means the env code
   moved since that collection and in-pool vs off-pool labels disagree. (DD2D 0.982,
   SB2D 1.000.)
   **Pilot on explicit problem ids, not `n_problems=N`.** `n_problems` takes the first N in
   sorted problem-id order, and ids are banded by stratum — so on SB2D any `n_problems=N`
   subset is *all b1*, the 2-candidate stratum every method ties on. Same trap as *stride,
   never truncate*.
7. **Method comparison table:** `marimo edit experiments/spectre/compare_methods.py` —
   one notebook for every environment, picked from a dropdown. Adding an environment is
   adding an `EnvSpec` to `compare_envs.py`; no notebook edit. Rebuild its cache with
   `python experiments/spectre/precompute_dd2d_cache.py --env-variant <variant>`
   (`--force` if the arm moved — `_dir_complete` skips a full directory, which is how
   DD2D's v3 row went stale). Render headlessly for any entry with
   `SPECTRE_COMPARE_ENV=<key> python experiments/spectre/compare_methods.py`.
   A **wall-clock-to-first-success** section (§2b, DD2D + kinder SB2D, `EnvSpec.has_timing`)
   sits beside the FP table: plan-gen + inference + refinement seconds, reusing the stored
   per-candidate `refinement_wall_clock_s` (inference measured on GPU, plan-gen a per-stratum
   constant — an SB2D plan-gen times the acyclic pool draw via `collect.time_pool_generation`;
   all cached, so a `--force` rebuild is what refreshes them). SPECTRE's SB2D timing is grafted
   from the `stickbutton2d_v1` legacy cache (`compare.merge_time_records`, the timing analog of
   the FP `merge_collections`). It is reported under the **deployed per-candidate refinement
   cap** (`REFINE_CAP_S`, per env-variant: **2 s** on DD2D, **10 s** on SB2D): each skeleton is
   refined for at most C seconds before the next, so a slow near-feasible *trap* costs C, not
   the 20 s budget. The cap is a wall-clock deployment config applied to every method; the
   §1/§2 FP headline stays **uncapped**, and §2b prints the cap's tiny FP delta. Two findings,
   read together: (1) uncapped, **FP flatters the learned ranker** — v3-adaptive has 6× lower
   FP than astar but ~equal *uncapped* wall-clock, because its few failures are the *expensive*
   near-feasible candidates while astar's many are cheap dead-ends, so an FP margin is not a
   proportional uncapped wall-clock win; (2) **under the cap v3-adaptive is the *fastest***
   (1.79 s ALL vs astar 2.96), s1 collapsing 11.99 → 2.40, at an FP cost of +0.05 — because the
   cap targets exactly those expensive failures. The cap is a test-time accounting change (no
   retraining); the adaptive order is *re-run* under it (it diverges on 6/300 cells), never
   `min(t, C)`-accounted ([`notebook/07`](docs/notebook/07-stickbutton2d.md#2026-08-02-dd2d-s1-wall-clock-blow-up-diagnosed-per-candidate)
   / [`decisions/07`](docs/decisions/07-stickbutton2d.md#2026-08-02-per-candidate-refinement-cap-deployed-wall-clock-configuration)
   2026-08-02). **On SB2D the narrative inverts** (2026-08-03): its failures are *uniformly*
   expensive (all run to the 20 s budget), so FP and wall-clock align — v3-adaptive is fastest
   capped (11.2 s) *and* uncapped (14.0 s), the cap helps the *highest*-FP method (astar, −48 s)
   most, and because SB2D's 10 s cap sits *inside* the feasible distribution (a cap above the
   whole distribution cannot fit under the budget) it costs the learned methods a real **+0.3 FP**
   — not DD2D's near-free +0.05
   ([`notebook/07`](docs/notebook/07-stickbutton2d.md#2026-08-03-sb2d-2b-wall-clock-spectre-adaptive-fastest-per-env)
   / [`decisions/07`](docs/decisions/07-stickbutton2d.md#2026-08-03-sb2d-2b-wall-clock-breakdown-parity-dd2d)
   2026-08-03).

## SPECTRE — the unified method (was "v3")

The 2026-08-12 publication refactor unified v1/v2/v2.2/v3 into one SPECTRE and removed the
built-then-disabled features (proof-tier demotion, legacy-coverage, obj-evidence, sinusoidal
positions, `tail_max_f`, the unwired necessity head); EMA weight-averaging and the ablation
flags are kept, one flag away. The "v3" naming throughout the rest of this section is
historical — v3 *is* the current SPECTRE; the modules are now unsuffixed
(`model.py`/`dataset.py`/`train.py`/`inference.py`, shared primitives in
`layers.py`/`encoders.py`) and the as-built lives in [`docs/as_built.md`](docs/as_built.md).
The migration design intent is the archived
[`docs/archive/SPECTRE_v3_proposal.md`](docs/archive/SPECTRE_v3_proposal.md); it ran as gated
increments (G0–G11). **Current substrate is `dd2d_v4`** (grasp-fixed *and*
refiner-instrumented); dd2d_v2/v3 numbers predate the double-canonicalization fix and must
not be quoted without regenerating.

**Gate status.** Done: G0–G6b as before; **G7** (P-v3-3 falsified — `cand_overlap` is
load-bearing, −5.07 FP); **G8/P2–P9** the performance push (below). **G9 descoped** (encoder
built, experiment not run — its premise does not hold on DD2D: s0–s2 pools already contain
9-operator plans while s3 needs 7, so the position table is never OOV). **G10 not
attempted.** Remaining: **G11 consolidation** — `as_built_v3.md` and `porting_guide.md` are
written; `./run_ci_checks.sh` has residual pylint line-length debt.

**Where v3 stands (2026-07-31).** Deployed config =
`--overlap-mode jaccard --coverage-feats --aggregate-records --evidence-attn
--state-delta` (sweep preset `v3final`, checkpoints `checkpoints_v3_unified`), **with
proof-demotion off** and **coverage/waste on the unified definitions** — the latter is now
`TrainV3Config.unified_coverage=True`, so the preset needs no extra flag. Over **3 seeds
each**:

> **⚠️ CURRENT (2026-08-12).** The live `compare_methods.py` caches — retrained under the
> de-versioned code — read **DD2D SPECTRE-adaptive 6.29 ± 0.31** and **SB2D 1.75 ± 0.19**.
> **[`docs/as_built.md` §10](docs/as_built.md) is the authoritative current comparison** across
> both environments. The 5.78/5.92 (DD2D) and 1.69/1.84 (SB2D) figures in this section are the
> **frozen yardsticks** these tie within seed variance; they — and the `v3` / `TrainV3Config` /
> `unified_coverage` names below — are pre-refactor and retained here as history.

| | ALL | s0 | s1 | s2 | s3 |
|---|---|---|---|---|---|
| v3 deployed | **5.78 ± 0.10** | 0.00 | **3.44 ± 1.36** | **10.49 ± 0.77** | **9.19 ± 0.76** |
| v2.2 yardstick | 17.27 ± 3.02 | 0.00 | 13.67 ± 14.20 | 23.45 ± 2.76 | 31.95 ± 5.62 |

**⚠️ Update 2026-08-09 — the deployed model is now the *domain-agnostic* input surface + a wider
selection window; the 5.78 above is the frozen target-anchored yardstick.** The scene inputs were
narrowed to domain-agnostic columns (`obj_is_target`→`obj_is_goal`, `obj_rel` 8→3, `d_rel=3`;
[`decisions/07` 2026-08-08](docs/decisions/07-stickbutton2d.md#2026-08-08-domain-agnostic-scene-inputs-goal-replaces-target)),
which regressed the *mean* as optimization variance (not information loss — best seed ≈ baseline,
probe Δ0.00). The fix is a training-side lever, **`--select-window 5`** (now in the `v3final` preset
and `sb2d_finalize.sh`; `TrainV3Config` default stays 3), which **recovers parity: DD2D 5.92 ± 0.29
(s1 4.84), SB2D 1.84 ± 0.26** — both tie the frozen 5.78 / 1.69 (paired CI includes 0). EMA
weight-averaging (`--weight-avg ema`) was built and tested but is **inert on both envs** (kept, one
flag away).
[`decisions/07` 2026-08-09](docs/decisions/07-stickbutton2d.md#2026-08-09-narrowed-input-variance-selector-noise-fixed-wider).

**≈ −11.5 FP vs v2.2** (the paired CI for that pair is not computed — `spectre_score.py`
cannot take a v2 arm as `--baseline`). Against the *previous* v3 deployed definition the
paired margin is **−1.66 FP, CI [−2.71, −0.71]**, every seed beating every baseline seed
([`decisions.md`](docs/decisions/06-v3-performance.md) 2026-07-31). Reproduce:

```bash
python experiments/spectre/spectre_sweep.py --preset v3final --seeds 0 1 2
# demotion is OFF by default; `--with-demotion` is the ablation. The flag is GLOBAL, so
# `--v2-arm` would also lose its demotion -- the 17.27 yardstick comes from the compare
# cache, whose v2 path always demotes.
python experiments/spectre/spectre_score.py --env-variant dd2d_v4 \
    --arm "v3 deployed:checkpoints_v3_unified" --seeds 0 1 2
```

**Pre-2026-07-31 v3 numbers are not retracted** — 7.44 and everything before it were
measured on a consistent definition and remain valid for what they measured. The checkpoint
decides which definition is used at scoring time (`inference_v3._emit_kwargs` reads
`unified_coverage` from the saved cfg; absent ⇒ the old formula), so old checkpoints keep
scoring correctly. `--legacy-coverage` retrains the old definition into a `_legacycov`
directory. The pre-memoization replicate of the unified arm is
`checkpoints_v3_unified_prememo` (5.83 ± 0.11).

**v3 is a purely learned ranker as of 2026-07-30** (`decisions.md`). Proof-tier demotion
was **cut from the method**: nothing outside the network touches the ordering, and
`apply_demotion=False` is the default in `deployed_rollout_v3_traced`, `cache_spectre3`,
`train_v3.deployed_val_fp` and `spectre_score.py`. It cost **0.23 FP** (7.20 → 7.44);
it fired on only **6%** of deployed rollouts against **55%** on the stripped floor arm, and
the learned features absorbed ~79% of its value. The machinery is **kept and one flag away**
(`apply_demotion=True`) because the deduction is sound — on a domain whose proofs fire more
often the trade reverses. Two things bit during that change and will bite again:
**re-cache with `--force`** (`_dir_complete` keeps a stale dir), and **pin
`test_v3_equivalence` to `apply_demotion=True`** or it compares two different policies and
silently stops testing equivalence.

**Three numbers to quote together, not separately** (`as_built_v3.md` §7.1):
- **3 seeds is the count every *method* has** (v2.2 was trained at exactly 3), not the
  count v3 has — v3 has **6**, and over all six it reads **8.54 ± 1.43**.
- **The yardstick is v2.2's 3-seed mean (17.27), not its best seed (14.66).** With both
  sides at 3 seeds the like-for-like comparison is mean-to-mean; ~2.6 FP of the −9.83 is
  that change of basis, not a change in v3. v2.2's s1 spread is ±14.20 (seed 2 lands at
  30.04, `relrank` picking a bad epoch) — the miscalibration R8 replaced.
- **v2.2 keeps its demotion while v3 gives its up**, so the margin is measured against a
  stronger baseline than v3 gives itself. Deliberate: re-scoring the published baseline to
  match a v3 design choice would be moving the goalposts.

**s1 is where a small-seed report is least trustworthy.** The pre-delta arm's s1 read
3.79 ± 3.29 at 3 seeds and 5.60 ± 3.06 at 6, where four of six seeds were *worse* than the
v2.2 seed-0 figure. On a wide-spread stratum, check the margin against the seed sd, not the
sign.

**Both record consumptions matter** (6 seeds each): dropping the per-failure token stream
costs 1.28 FP (7.90 → 9.18), *entirely at s1* (5.60 → 10.78, worse than v2.2 there) while
s2/s3 tie, and it doubles the variance. Compact features carry s2/s3; tokens carry s1.

**What carries it: observed `coverage`/`waste`.** These are §5.1's necessity features with
per-object necessity **observed** (`FailureRecord.culprits`) instead of **predicted** — so no
head, no second loss, no geometry routine, and *more* C2-legal than the cut version since
nothing is inferred by us. Not `clears` (L2): that was a routine *we* ran. Both features are
exactly zero at |F|=0, so the first attempt stays static and the signal accrues as the
rollout observes. A leakage audit returned 0 violations.

**Retracted, do not quote:** G6's levels (18.59/19.15/20.95 — censored selector) and G6's
−3.37 "record increment", which was `cand_overlap`, not records (its bar removed both).

**Traps this push added** (details in `docs/autorun_decisions.md` A1–A13):
- **`dead` is a length proxy** — right at s3, wrong at s1 (corr(dead,|S|) = −0.284). Tuning
  it only trades strata; give the model the count it proxies for.
- **A token stream the model ignores is not free** — records cost −0.83 FP in training while
  `suppress_records` showed the deployed model barely reads them (16.17 → 16.40).
- **Evidence competed with geometry in one softmax** (~10 scene tokens vs up to 2045 record
  tokens), so discarding it was loss-minimizing.
- **Two runs sharing a checkpoint dir** silently interleave writes; `train_v3` now refuses
  via a `.owner` marker.

- **New modules** (v1/v2 are frozen — D-7): `domain.py` (the whole per-environment
  contract: per-query `QueryAxioms(monotone, local, exact)` + `min_calls_per_schema`),
  `failure_record.py`, `proof_demotion_v3.py`, `model_v3.py`, `dataset_v3.py`,
  `inference_v3.py`, `train_v3.py`, `necessity.py` (built, **unwired**).
- **D-8, exact-absence:** every v3 feature is config-gated and *off* reproduces v2.2's
  state dict byte-for-byte, so `test_v3_equivalence.py` keeps loading the v2.2 checkpoint
  and asserting identical decisions. That oracle is what makes data-path rewrites safe;
  it runs in `permissive` mode and retires when the position encoding changes (G9).
- **Necessity conditioning was CUT** ([`decisions.md` 2026-07-26](docs/decisions/README.md)): D2 showed the s2 deficit
  is *within-length*, which it does not address. P-v3-1 is withdrawn; s2 is reported as a
  characterized limitation.
- **Untaken lead:** enumeration order is a strong within-length signal (astar
  length-oracle 5.80 at s2) the deployed model cannot see, because R1 dropped the prior
  wholesale when only its short-first column was implicated. An index-only prior was never
  separately ablated.

Findings and numbers live in `docs/notebook.md` / `docs/decisions.md` under 2026-07-26 —
cite them rather than restating figures.

## DD2D generalization test — unseen count & unseen shapes (2026-08-01)

The dd2d_v4-trained v3 checkpoint is scored **train-old / test-new** on two held-out test-only
env_variants (each 40 problems, stratified s0–s3, seed bands disjoint from train/val/test):
**`dd2d_v4gen_count`** (14–16 items = 13–15 blockers vs the trained 9–12, old shapes) and
**`dd2d_v4gen_shape`** (same unseen count + two new *concave* families, a `tee` and a `cross`,
≥1 of each forced per scene). The grasp model is geometry-general (`grasps.py` reads only
`shape.polygon`), so the new shapes needed **no per-shape grasp code** — only `shapes.py`
(`_build` + `_CONCAVE_FAMILIES`, kept out of `_FAMILY_WEIGHTS`). Reproduce:

```bash
bash experiments/spectre/collect_dd2d_genset.sh 12   # collect A+B, convert, reuse dd2d_v4 vocab
python experiments/spectre/spectre_score.py --env-variant dd2d_v4 \
    --test-variant dd2d_v4gen_count --arm "v3:checkpoints_v3_unified" --astar-baseline --seeds 0 1 2
# then --test-variant dd2d_v4gen_shape
```

**Headline (v3 ALL FP, 3 seeds; paired vs astar-dist):** in-dist 5.78 → unseen-count
9.40 ± 2.62 → count+shape 11.26 ± 3.44. **v3 still beats astar overall on both (CI excludes
0)**, so the learned ranker's advantage survives OOD. Three things to quote together
([`notebook/07`](docs/notebook/07-stickbutton2d.md) 2026-08-01):
- absolute FP degrades ~1.6–1.9× — generalization is not free, shape is harder than count;
- **the ALL win is carried by s3** (astar's default order is pathological there, 108–167 FP),
  not a uniform advantage;
- **at s2 v3's edge collapses** under the shift (30.23 vs astar 28.30 count; 31.97 vs 22.00
  shape, within ±9 seed spread). **⚠️ The s2 column is a pool-composition artifact, not a clean
  model signal** (diagnosed 2026-08-02): s2 problems have only ~1.5 unique feasible solutions, and
  in-distribution the k=200 pool pads them with ~23 redundant feasible triples; at high blocker
  count C(14,2)=91 pairs flood the short-first cap and crowd the triples out (92→18 enumerated), so
  the feasible density collapses (26→2.9) and FP jumps. **Read the generalization at s3, not s2**
  — s3 was already feasible-scarce in training, so OOD s3 is in-regime and v3 improves there
  (9.19→4.87). A generator regen for s2 pair-diversity was explored and rejected as geometrically
  blocked (circular target + 18 diametric grasp axes cap feasible pairs at ~1.5). See
  [`notebook/07`](docs/notebook/07-stickbutton2d.md#2026-08-02-s2-ood-degradation-pool-composition-artifact-model)
  / [`decisions/07`](docs/decisions/07-stickbutton2d.md#2026-08-02-s2-generalization-degradation-characterized-pool-composition-artifact)
  2026-08-02.

Two invariants this exercised: **a scene truncating below the unseen floor falls back into the
seen range** — the collector rejects it (`min_items`), so cranking `fill_max` alone is not
enough; and **no OOV / no position-index error** OOD, because the vocab is over the fixed
op/pred/type set (a shape family is geometry metadata) and the dd2d_v4 vocab is reused.
Protocol ADR: [`decisions/07`](docs/decisions/07-stickbutton2d.md#2026-08-01-dd2d-generalization-test-unseen-count-unseen).

## StickButton2D — the second environment (2026-08-01)

DD2D was the only evaluation environment; SB2D is the second, so the generality claim in
[`docs/porting_guide.md`](docs/porting_guide.md) is finally being *tested* rather than
asserted. Everything wraps kinder's own env, operators and `BacktrackingRefiner`.
Chapter [`07-stickbutton2d`](docs/decisions/07-stickbutton2d.md) holds the ADRs.

**The dataset is `stickbutton2d_v1`**: b1/b2/b3/b5 pooled into one env_variant with
**button count as the stratum** (b10 dropped — 0/20 solvable, and the cause is pool prefix
homogeneity needing diverse plan *generation*, not a better heuristic). Problem ids encode
`split · 10⁶ + slot · 250000 + index` precisely so the existing `compare.stratum_of`
returns the slot, which is what lets ~15 call sites work unchanged. **Strata are
contiguous pid bands, so *stride, never truncate* is load-bearing here** — `paths[:N]` is
all b1.

**The two strata that matter are b3 and b5.** Pool sizes after the acyclic filter are
≈2 / 6–34 / 200 / 200, so b1 and b2 are anchors that every method ties on (b1 static
reads 0.08 FP) — the shape DD2D's `s0 = 0.00` already has. Do not read a pooled "ALL"
mean over unbalanced strata as a method comparison.

**Headline (3 seeds, uncensored, test n=100).** Mean failed attempts before first success. *(The
**1.69** here is the frozen target-anchored model; the deployed model since 2026-08-09 is the
domain-agnostic narrowed inputs + `--select-window 5`, which ties it at **1.84 ± 0.26**; the **live
cache now reads 1.75 ± 0.19** after the 2026-08-12 retrain ([`docs/as_built.md` §10](docs/as_built.md)
is the authoritative current table) — see the
"Where v3 stands" update above and [`decisions/07` 2026-08-09](docs/decisions/07-stickbutton2d.md#2026-08-09-narrowed-input-variance-selector-noise-fixed-wider).)*

| | ALL | b1 | b2 | b3 | b5 |
|---|---|---|---|---|---|
| **SPECTRE v3** | **1.69 ± 0.26** | 0.08 | 0.24 | **1.13 ± 0.12** | **5.29 ± 1.04** |
| B3 static-historical (best baseline) | 6.41 | 0.08 | 0.36 | 9.84 | 15.36 |
| B2 default A* order | 16.29 | 0.08 | 0.56 | 2.96 | 61.56 |
| B4 adaptive-historical | 22.56 | 0.08 | 0.24 | 26.88 | 63.04 |
| B1 random | 21.04 | 0.24 | 5.22 | 47.79 | 30.90 |

**The comparison row, and the result that matters most.** PIGINet is env-agnostic since
2026-08-01 (`baselines/piginet/` + per-env adapters), so SB2D has the representation contrast:

| method | ALL | b1 | b2 | b3 | b5 |
|---|---|---|---|---|---|
| astar-dist | 16.29 | 0.08 | 0.56 | 2.96 | 61.56 |
| VLMPlan-32B (local Qwen, n=40) | 13.18 | 0.70 | 1.30 | 6.20 | 44.50 |
| VLMPlan-GPT5.6 (terra, n=40) | 6.42 | 0.00 | 2.40 | 0.90 | 22.40 |
| LAZY-adaptive (Khodeir et al) | 1.85 ± 0.02 | 0.08 | 0.36 | 2.44 | 4.56 |
| PIGINet (low-level) | 2.02 ± 0.19 | 0.08 | 0.32 | 1.31 | 6.39 |
| SPECTREv3-static | 1.98 ± 0.28 | 0.08 | 0.32 | 1.52 | 5.99 |
| SPECTREv3-adaptive | **1.69 ± 0.26** | 0.08 | 0.24 | 1.13 | 5.29 |

**LAZY (`baselines/lazy/`, added 2026-08-09) is the learned *adaptive* competitor** (GAT
policy π + online feasibility ϕ, π̄=π·ϕ/Σ; PIGINet is static). On SB2D it **ties everything**
(paired vs LAZY: SPECTRE-adaptive −0.01 CI [−0.72,+0.72]; LAZY−PIGINet −0.44 CI [−1.18,+0.29]),
extending the SB2D non-separation finding to a third adaptive method. On **DD2D both learned
rankers beat it decisively** (LAZY 23.26 vs SPECTRE 5.92 / PIGINet 17.27, CIs exclude 0; still
beats astar 34.52 / VLMPlan 35.23, carried by s3). SB2D numbers rest on the small 17-episode b5
train split (b5=4.56 is largely a generalization number). `decisions/07` / `notebook/07`
2026-08-09; deviations from literal LAZY in `baselines/lazy/PROVENANCE.md`.

**VLMPlan-32B (the zero-data corner) is a genuine planner here** — 35/40 problems
self-solved, 0 censored, stratified n=40 (10/stratum, so the ALL is comparable). It sits
between astar and the learned methods: it beats the naive planner order overall (13.18 vs
16.29, but only via b5 where astar's default order is pathological) and is *worse* than
astar on b1/b2/b3, where its charged-but-failed off-pool proposals cost it. It trails
SPECTREv3/PIGINet by ~7×. The 2-problem train pilot badly mis-estimated it (drew the hard
tail — "loses to astar, censored on b5"); the stratified test sample overturns that. See
[`notebook/07`](docs/notebook/07-stickbutton2d.md) 2026-08-01. The full 100 was descoped
(b3/b5 tail problems run to the stall cap, ~16 h) — the ~7× gap and ordering are settled.

**The frontier arm `VLMPlan-GPT5.6` is now gpt-5.6-terra with gripper disclosure (2026-08-08),
replacing gpt-5.6-luna** (`prompts/PROVENANCE.md` deviation 9). On SB2D it reads **6.42 ALL**
(b1 0.00, b2 2.40, b3 0.90, b5 22.40), self-solving 39/40 with 0 censored — roughly half luna's
11.85 and clearing the local 32B (13.18). It now beats the naive order across the board (notably
b3 0.90 < astar 2.96, where luna had over-thought) but still trails the learned rankers by ~3–4×,
so the representation ordering is unchanged. Single generation seed → bare mean (like astar); use
the across-problem bootstrap for a spread.

**On SB2D the representation advantage does not reproduce.** Paired bootstrap over the 100
test problems: v3-static − PIGINet = −0.05, CI [−0.44, +0.35]; v3-adaptive − PIGINet =
−0.34, CI [−0.72, +0.05] — **neither separates**. What does is the adaptive increment
within SPECTRE (−0.29, CI [−0.52, −0.07]). That *inverts* DD2D's attribution, where the
static representation carried ~73% and adaptivity ~27%. Honest cross-environment statement:
**the abstract representation wins on DD2D and ties on SB2D; the adaptive increment is
positive on both.** Read it with the caveats in
[`notebook/07`](docs/notebook/07-stickbutton2d.md) — PIGINet's image channel is degenerate
here yet it still matches v3, and the `at-pose` literals it reads are synthesised by our
adapter, not stored. **These numbers are the *schematic*-crop PIGINet (2.02).** As of
2026-08-02 there is also a kinder-rendered variant **`stickbutton2d_v1_kinder`** (PIGINet's
crops re-sourced from kinder's own renderer via `sb2d_render_convert.py`; SPECTRE is
image-free and grafted from v1) — the validity fix. It **reinforced** this finding rather
than overturning it: kinder-crop PIGINet reads **2.28 ± 0.29**, if anything slightly worse
(all at b5), and still does not separate (v3-adaptive − PIGINet = −0.60, CI [−1.24, +0.08]).
The abstract crop context is positional, and unpressed buttons are identical discs in the
real env too, so it is net-neutral-to-mild-distractor. See
[`decisions/07`](docs/decisions/07-stickbutton2d.md#2026-08-02-kinder-rendered-piginet-crops-stickbutton2d-via-new)
/ [`notebook/07`](docs/notebook/07-stickbutton2d.md#2026-08-02-stickbutton2d-piginet-crops-re-sourced-kinder-s)
2026-08-02.

**Three things to quote together, not separately** — all in
[`notebook/07`](docs/notebook/07-stickbutton2d.md):

- **B4 is worse than random here** (22.56 vs 21.04). SPECTRE's headline comparison on
  RT2D/DD2D is the adaptivity premium over B4; on SB2D the bar is **B3, a *static*
  ranker**, and any cross-environment framing has to say so.
- **b5's train split has only 17 episodes** (collection was cut at a wall-clock budget),
  so 5.29 at b5 is substantially a *generalisation* number, not a like-for-like stratum
  result. Re-measure before quoting it as one.
- **`waste` helps the model and hurts a rule.** As a hand-coded tie-break it degrades b3
  (4.44 vs 2.88 for coverage alone); as a learned column it is worth +0.36 FP, CI
  [+0.08, +0.67]. A failed non-learned re-ranking probe is therefore **not** grounds for
  dropping a feature — it tests monotone usability in the guessed direction, not
  information content.

**Three things about SB2D that are not true of DD2D**, each of which changed code:

- **It has no class-1 evidence at all.** kinder's collision check returns a bool, so every
  failure is a class-2 *deviation* between predicted and achieved abstract state. Both
  channels are now always wired and an empty one is provably inert; see the ADR.
- **`jaccard` and `dead` are constant across the b5 pool.** Every b5 plan has length 6
  (5 presses + 1 stick pickup), so `manipulated = {robot, stick}` for every candidate.
  Coverage/waste and the token structure are the *only* discriminating features there.
- **Its pool generator pads plans with `PickStick`/`PlaceStick` cycles.** Filtered out;
  near-inert at b5, decisive at b1 (200 candidates → 2).

**`spectre_collect.py` is not the collector here** — `experiments/spectre/sb2d_collect.py`
is, because the collection pools four kinder env ids into one variant and **rejects and
resamples** problems with no feasible skeleton. Post-collection, `sb2d_finalize.sh` runs
vocab → B1–B5 bracket → training → scoring in the order the gates require (the standalone
sanity-check and re-ranking-gate stages were dropped in the 2026-08-12 refactor).

## Conventions and invariants

- **Loss:** listwise Plackett-Luce only. Pointwise BCE killed Attempt 2.
- **F-subsets:** `F ⊆ FAIL_e` strictly — never successes in F.
- **Vocab from train only;** id 0 = `<PAD>`/`<OOV>`; local-id 0 = pad.
- **Augmentation:** training only; per-type policy from `env_registry.py`
  (ordered/semantic RT2D types are non-augmentable).
- **Metrics:** model selection and early stopping are rollout-based —
  `val_rollout_attempts` (simulated sparse rollout on val, attempt budget 20;
  `checkpoint_metric` in `train.py`) — chosen to align with the rollout-based
  test-time objective. AUROC(3) is a secondary offline diagnostic (drives the
  during-training de-risking gates), never the selection criterion. The
  D.1/D.2 atom-sensitivity probes do NOT predict rollout performance —
  diagnostics only, never optimization targets.
- **Reporting:** paper numbers are mean ± std over ≥ 3 seeds. **Development runs
  1 seed** (2026-07-26 directive) — with 1 seed "within seed noise" is not
  measurable, so a gate is accepted by a **paired bootstrap over problems**
  (`spectre_score.py`, the instrument the P1/P4/P5 gates used); pairing removes
  the between-problem variance that dominates here.
- **Doc updates are part of development** — see "Documentation discipline"
  below. Archived specs and snapshots in `docs/archive/` are frozen — never
  edit them; annotations go in `docs/archive/README.md`.
- Tests: `pytest tests/approaches/spectre/`. Slow tests are skipped by default and
  **`-m ""` does NOT include them** — `tests/conftest.py` overrides an empty
  markexpr back to `not slow`. Use `-m slow` to run them.

## Working practices (hardware, long runs, traps)

**Use the hardware. Parallelise whenever tasks are independent.** Training here is
**CPU-bound, not GPU-bound** — measured 79% tensorization / 21% GPU, and three
concurrent arms occupy 3.5 GB of the 5090's 33.7 GB. Run arms, seeds, ablations and
data collection *concurrently* rather than in sequence whenever they do not depend on
each other; serial runs leave both the GPU and ~30 CPU threads idle.

- `python experiments/spectre/spectre_sweep.py --preset g6` — concurrent arms, one log
  each. `--arm "name:args"` for ad-hoc arms, `--seeds 0 1 2` for the paper runs.
- Keep `max_parallel × (1 + num_workers)` under the core count (32) or the runs contend
  and wall-clock stops improving. Measured: 38.9 s/epoch serial → ~33 s/epoch with three
  arms at once (~3.4× throughput).
- The DD2D collector already parallelises via `--workers`.

**Long runs must be interruptible and must expose an ETA.** Anything over a few
minutes goes to a named log with periodic heartbeats, so progress and remaining time can
be checked at any moment without disturbing the run — and so a run that has clearly gone
wrong can be stopped early instead of discovered at the end.

- Launch via `spectre_run.sh <name> <cmd...>` (or `spectre_sweep.py`), which logs to
  `data/spectre/logs/<name>.log`.
- Check with `python experiments/spectre/spectre_status.py` (`--watch` to follow):
  what is running, latest heartbeat + ETA per job, recently finished checkpoints.
- When adding a long-running script, emit a periodic heartbeat with elapsed, progress
  and ETA, and state the expected total up front.

**GPU contention:** LM Studio / `llama-server` (the VLMPlan backend) holds ~30 GB of
VRAM and will starve training into CUDA OOM warnings. Stop it before a sweep; the
VLMPlan results are already cached under `compare_cache/vlmplan_*`, so nothing is lost.

**Traps that have each cost real time:**
- **Stride, never truncate.** Episodes are stored in seed order and the collector fills
  strata in seed bands, so `paths[:N]` yields only the easy strata. Bit us twice (an
  equivalence test, then the val selector).
- **`canonicalize_episode` is not idempotent** — always tensorize from *raw* episodes.
  Double canonicalization silently changes the object→tag binding and skewed every cached
  comparison number before 2026-07-26.
- **Selection metrics must not be censored below the tail that separates models.** A val
  FP censored at 30 attempts rated v2.2 and v3 equal (11.12 vs 11.40) while they differed
  by 4 FP uncensored on test; uncensoring it (G6b) moved v3 from *significantly worse* than
  v2.2 to indistinguishable. The tell is **dynamic range, not jitter** — the censored
  curves were stable and picked sensible mid-training epochs, they just spanned ≈6 FP where
  the uncensored ones span ≈15. Stable curves are not evidence of a good selector.
- **DD2D generation is `PYTHONHASHSEED`-dependent**, so no collection is reproducible
  across processes; expect a fresh sample on any re-collection.

## Documentation discipline — keep the living docs alive

The living docs are the project's research memory: code records *what*, the
docs record *why*. Commit messages and checkpoint dirs are not a research
log. After any change that is more than mechanical, decide — **before
committing** — which of these needs an entry, and ship the entry **in the
same commit** as the change:

| The work... | Update | Format |
|---|---|---|
| produced any run/EDA/probe/ablation numbers — **including failed and negative runs** | `docs/notebook/` | dated What / Result / Takeaway-next entry (format in its README) |
| chose between alternatives with lasting consequences, killed an approach, or changed a convention / invariant / metric / protocol | `docs/decisions/` | ADR: context → decision → consequences, newest first |
| changed the method, loss, architecture, data pipeline, or evaluation protocol | `docs/proposal.md` | edit in place — it must always describe the *current* method; also reconcile §6 (add new unknowns, remove resolved ones) |

Exempt: mechanical refactors, formatting, typo fixes, CI appeasement —
anything that cannot affect results or future decisions.

Litmus test: *"In 3 months, will we know this happened and why?"* If the
change could alter a number in a future writeup/snapshot, or a future
contributor could plausibly re-litigate the choice, it needs an entry.

This rule exists because the passive version of it failed: `notebook.md`
stayed empty for the project's first ~2 months of training runs, and the
load-bearing rollout-metric decision (`b74b593`) got its ADR only
retroactively. Write entries at change time, not archaeology time.

### Mechanics — both logs are chaptered by era

Both logs are split into era chapters under `docs/decisions/` and
`docs/notebook/` (same boundaries in both), with a **generated `README.md`**
per log. `docs/decisions/README.md` is what this file `@`-imports, so the
indexes and the standing-invariants table are always in context while the
narratives are read on demand.

```bash
python experiments/spectre/decisions_index.py new --log decisions \
    --title "..." --tracks method,evaluation   # scaffold into the open chapter
python experiments/spectre/decisions_index.py index                # regenerate READMEs
python experiments/spectre/decisions_index.py check                # enforced by pytest
```

Four rules, all enforced by `tests/approaches/spectre/test_doclog.py`:

- **Historical entries are append-only.** To change what an old entry says, add
  a new one, set `supersedes` on it and `superseded_by` + a `status` + a banner
  on the old one. `check` compares every pre-split entry against the frozen
  monolith in `docs/archive/` and fails on a silent edit. It also enforces that
  `supersedes` / `superseded_by` point at each other.
- **Status is machine-readable.** Each entry carries a fenced metadata strip
  (`<!--strip-->`) with id, status, tracks and cross-references, so a chunk
  retrieved out of context still says whether it is quoting a retracted number.
  The README's **do-not-quote** table lists every one of them.
- **Cite by entry id**, e.g.
  `decisions/05-v3-migration.md#2026-07-26-selection-metric-never-censored`.
  Ids are permanent. Older `` `decisions.md` <date> `` citations survive — the
  stubs still resolve and each README carries a legacy date→entry table — but
  dates collide (2026-07-19 has six entries), so new citations use the id.
- **Close a chapter** at a named phase boundary, or when `check` warns that the
  open one passed ~650 lines / 12 entries; add it to `_ERAS` in `doclog.py`.

`autorun_decisions.md` and `autonomous_stickbutton_session.md` stay as
standalone session narratives — they were judged without a human in the loop, so
they are *unratified*. Promote one to an ADR only if it changed a convention,
architecture or invariant **and** no ADR states that decision; promotion means
writing the ADR with `ratifies:` set and adding a forward pointer, never moving
the text. Four were promoted on 2026-07-29 (A3, A8+A11, A10, A16).
