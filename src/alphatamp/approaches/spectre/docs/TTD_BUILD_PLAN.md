# TTD — Chunked Build Plan

Living build plan for the **Tote-and-Tray Decluttering (TTD)** benchmark
([`TTD_SPEC_v1.3.md`](TTD_SPEC_v1.3.md) is authoritative; `TTD_SPEC_v1.0.md` is
superseded — ignore it). This doc records *how* the spec is being built incrementally;
the spec records *what* and *why*. Update the "Status" column as chunks land.

## Why chunked

The spec is long and load-bearing in specific places, so we build in **verifiable
chunks**: implement one chunk, prove it works (tests + CI green), then start the next.
Two chunks are **go/no-go decision gates** the spec mandates *before* downstream
build-out — §10.0 pre-pilot (is the packing coupling geometrically viable at all?) and
§10.2 pilot (does the shared sampling refiner clear the C2 ceiling while cheap attacks
stay below the C5 floor?). The chunking is designed to reach each gate with minimal
wasted code, so a NO-GO triggers the pre-registered fallback before the expensive
generator / learning code exists.

## Ground rules

- **Location:** `src/alphatamp/approaches/spectre/ttd/` — a self-contained sub-project.
  It must NOT import spectre's model/dataset/collect/trajectory code, `bilevel_planning`,
  `relational_structs`, or `kinder`. TTD is the *environment*; the method eventually used
  to solve it may not be spectre, so the package stays decoupled.
- **Module map (spec §12.1):** `ttd_core/` (authoritative 2D geometry + streams + op
  counters), `ttd_gen/` (composition, planting, audits, JSON), `ttd_plan/` (proposer,
  staged refiner, evaluated variants), `ttd_learn/` (attacks, mid-tier, incumbents),
  `ttd_sim/` (PyBullet + realism tier), `ttd_eval/` (metrics, frontier plots), plus
  `calibrate.py` and `verify_labels.py`.
- **Tests:** mirror the source tree under `tests/approaches/spectre/ttd/`. Test-file
  basenames are prefixed `test_ttd_*` (pytest runs in prepend import mode with no
  `__init__.py`, so basenames must be unique repo-wide).
- **Deps:** the `[project.optional-dependencies] ttd` extra in the root `pyproject.toml`
  (shapely now; pybullet / scikit-learn / ffmpeg added by the chunk that needs them).
  Install with `uv pip install -e ".[ttd]"`.
- **CI hygiene:** black line length 88, isort profile black, docformatter wrap 88, mypy
  strict (`strict_equality`, `disallow_untyped_calls`, `warn_unreachable`), pylint via
  `pytest --pylint`. Module + public-symbol docstrings required. Slow tests marked
  `@pytest.mark.slow` (auto-skipped by `tests/conftest.py`).

## Chunk map

Ordering follows spec §12.3 (build order) and §12.5 (work packages: P0 = go/no-go,
P1 = the paper, P2 = post-submission).

| # | Chunk | Spec | WP | Status | Ends when (milestone) |
|---|---|---|---|---|---|
| 1 | ttd_core: geometry + shapes + params/dials | §2.8, §3, §4.2, §7.1 | P0 | ✅ done | Geometry primitives (inflation, NFP/IFP, antipodal), shape library, params dataclass, C7 dial-consistency all unit-tested; CI green. |
| 2 | ttd_core: nester + η + label rule | §7 | P0 | ✅ done | `N(S,r)`, `η` bisection, exact/anytime/intensified nester, feasible/infeasible/marginal label rule; hand-made nests classify correctly. |
| 3 | **§10.0 pre-pilot (GO/NO-GO #1)** + `calibrate.py` | §10.0, §5.3 | P0 | 🔬 machinery done; full run = cluster job | η landscape (ρ̂, spread, witness/decoy supply, band occupancy) measured across OP×band×c_v cells; op-cost model published. **Gate: viability condition decides the operating point or triggers the fallback — before any generator/learning code.** |
| 4 | ttd_core: streams + sampler + staged refiner | §4.3, §5.2, §5.4, §9.2 | P0 | ← next | Grasps/columns, compaction-biased sampler, staged refiner w/ revision tokens; **revision-required regression test** passes (§12.3 step 1 checkpoint); reproducible traces. |
| 5 | ttd_plan: proposer + skeletons + peel order | §6 | P1 | todo | Corridor/blocker-set proposer emits candidate list + published peel order on a hand scene; drowning-arithmetic appendix. |
| 6 | ttd_gen A: composition pre-check + planting + occlusion | §8.2–8.4 | P1 | todo | Planted witness refines ≥4/5 seeds pre-scramble (§12.3 step 3). |
| 7 | ttd_gen B: decoys/repair + audits + scramble/verify + JSON + `verify_labels.py` | §8.5–8.8 | P1 | todo | 50 instances pass audits A1–A7; instance JSON round-trips; label-escrow reproduces labels (§12.3 step 4). |
| 8 | **§10.2 pilot (GO/NO-GO #2)** + planner variants + metrics | §9.3, §10.2, §11 | P1 | todo | Uninformed/oracle/checker-in-loop/retrieval brackets; ceiling/floor/G1/G2 measured per cell; **operating point locked or fallback**. |
| 9 | ttd_learn: attacks + order-ablation + frontier + mid-tier + incumbents | §10.3, §11 | P1 | todo | Compute–accuracy frontier + order-ablation curve; mid-tier set-transformer selector; PIGINet/LAZY re-impls (G4). |
| 10 | ttd_sim: PyBullet + video + realism tier | §12.4, §13 | P1 | todo | Kinematic scene render, D2 trace video, realism-tier execution + ranking preservation. |

Cross-cutting deliverables land inside the chunk that first needs them: `calibrate.py`
(chunk 3), `verify_labels.py` (chunk 7), the ≥5k-instance **dataset build** (a cluster
run between chunks 8 and 9), `ttd_eval` metrics (chunk 8).

## Completed chunk scope

**Chunk 1 — `ttd/ttd_core/`:** `params.py` (§3 table, OP-A/OP-B, §2.8 μ / r_i / r_f),
`counters.py` (§5.3 op-counter scaffold), `geometry.py` (round inflation + Ã, star-fan
convex decomposition, Minkowski, NFP/IFP, arrangement-vertex + boundary-sampling helpers,
the antipodal-grasp primitive, descriptors), `shapes.py` (§4.2 seeded generator + library
+ JSON), `dials.py` (§2.8 C7 Brunn–Minkowski necessary condition + Φ occupancy).

**Chunk 2 — `ttd/ttd_core/nesting.py`:** the DFS bottom-left nester over the NFP/IFP
arrangement (exact / anytime / intensified modes, §7.2), `packs()` = `N(S, r)`,
`packing_margin_radius()` = `η` via bisection (§2.8), `verify_placements()`, and the §7.3
`label_candidate()` returning FEASIBLE / INFEASIBLE / MARGINAL / INDETERMINATE with the
feasible-side nest certificate. Built entirely on chunk 1's instrumented primitives (an
external heuristic packer cannot certify infeasibility or share the §5.3 op currency).
The nester caches NFPs by rotation pair and translates at use
(`NFP(translate(A,t),B)=translate(NFP(A,B),t)`) — the §5.3 caching win.

**Chunk 3 — `ttd/ttd_eval/`:** `calibrate.py` (the §5.3 op-cost table: µs/op per C-op
kind → calibrated geometric cost) and `pre_pilot.py` (the §10.0 landscape: matched-ΣA
subset sampling, η computed once per (OP, band) and re-thresholded per c_v, `go_criteria`
= witness ≥ 10% ∧ decoy ≥ 10% ∧ band ≤ 15%, a GO/NO-GO report, and a standalone CLI). The
machinery is unit-tested; the **full 200×36 pre-pilot is a cluster job** (a single η
bisection over concave k∈{4,5} subsets is seconds-to-minutes, per §7.2/§8.8), so it runs
off-box. A reduced local run gives a preliminary directional read only. The CLI has two
subcommands — `run` (shardable over the independent (OP, band) landscapes via
`--shard-index`/`--n-shards`) and `merge` (aggregates partial JSON reports, re-decides
GO/NO-GO). `experiments/spectre/ttd_pre_pilot.slurm` is a 6-way array job (2 OP × 3 bands)
that shards `run`, followed by one `merge`.

**Pinned conventions** (chunks 2–4 depend on them, documented at the top of
`geometry.py`): placement reference point = the shape's local origin (star center); NFP
orbit = A stationary, B orbits, `NFP = ⋃(A_i ⊕ reflect(B_j))`; authoritative
`Ã(s,r) = inflate(s,r).area` with a fixed `BUFFER_QUAD_SEGS`, shared between labeling and
Φ so they never disagree.

**Deferred out of chunk 1** (crisp boundary): the nester DFS, `N(S,r)`, `η`, intensified
mode → chunk 2. Streams / sampler / staged refiner → chunk 4. Out-of-family real-footprint
shapes (§4.2.1) and all splits beyond train/held-out → later. `ρ̂` is measured in §10.0
(chunk 3), so `dials.py` takes it as a caller argument.
