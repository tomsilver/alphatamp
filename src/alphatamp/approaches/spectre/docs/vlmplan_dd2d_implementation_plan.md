# VLMPlan Baseline for DD2D — Implementation Plan

> ## ⚠️ As-built delta (2026-07-24) — read this first
>
> **This document is now the *design* record, not the current state.** The baseline was
> built; where the two disagree, `decisions.md` 2026-07-24 and the code win.
>
> **Built:** `vlmplan/` (`template`/`parsing`/`adapter`/`dd2d_adapter`/`models`/`loop`/
> `score`/`runio`) + `envs/dd2d/spectre_render.py` + `experiments/spectre/vlmplan_{run,score}.py`
> + `conf/vlmplan.yaml` + `test_vlmplan.py`; wired into `compare_dd2d_methods.py` as the
> `VLMPlan` row via `dd2d_compare.SEQUENCE_METHODS`. Smoke numbers in `notebook.md` 2026-07-24.
>
> **Changed from this plan:**
> - **No `probe_blockers.py` / `probe_packing.py`** (§8) and **no LLMPlan ±image arm** (§4.1) —
>   descoped by the user. `DD2DAdapter(with_images=False)` still gives the text-only arm.
> - **Primary model is `qwen3-vl-8b-instruct` locally via LM Studio**, not vLLM; the 32B and
>   frontier-API arms are deferred config changes. §2.2's `OPENAI_BASE_URL` routing is exactly
>   how it works.
> - **The published-order fill lives in `score.py`, not `loop.py`** (§5 pseudocode). Generation
>   records what the model proposed; scoring turns that into an attempt sequence under a
>   budget. That split is what lets a re-collection re-run only the free local half.
> - **Two deviations this plan did not anticipate, both necessary and both disclosed** in
>   `prompts/PROVENANCE.md` §7 and the parser leniencies: per-skill semantics (`pick` vs
>   `retrieve` is not inferable from names — 28/28 plans failed on it) and accepting the
>   omitted `:type`/`[]` forms (31/31 blocks rejected on format alone otherwise).
> - **Temperature 1.0, not 0** (§2.2), plus a per-round seed: at temperature 0 a round that
>   yields nothing leaves the next round with a byte-identical prompt.
> - **`label_agreement` is a new first-class gate** this plan has no analogue for — mixing
>   stored in-pool with live off-pool labels is only valid if the env code has not moved since
>   the collection, and that has to be measured, not assumed.
> - **Dedup key is the full ordered step tuple**, which for DD2D is equivalent to §5's ordered
>   member tuple (pick/place must alternate), and strictly finer.
>
> **Still true and honoured:** the static hard line (§1.3), off-pool live-refinement and the
> filtered-for-free discount with all rates reported (§5), the vendored KinDER prompt + parser
> with per-block error semantics (§2.3, §3), the pilot-then-freeze order (§6), the
> trivial-mimicry null (§7), and the honesty gate.

**Status:** design document, pre-implementation. **Scope:** zero-shot LLMPlan/VLMPlan baselines
(KinDER convention, Huang et al. 2026) adapted to the DD2D skeleton-ranking protocol, built so the
DD2D-specific parts are confined to one adapter module and the harness extends unchanged to future
environments (block stacking/sorting, etc.).

**Honesty gate (inherited):** the primary metric depends on feasibility labels, and the
arrangement-complete negative certificate is still deferred. All numbers produced by this baseline
are **diagnostic only** until the certificate lands, same as every other label-dependent number in
the repo.

---

## 1. What we are building

A static, zero-shot plan-sequence generator evaluated on the shared attempts-to-first-feasible
metric:

1. Per test problem, prompt a model with the **KinDER LLMPlan template** (verbatim base; deviations
   enumerated in §4) filled with DD2D content. The VLMPlan arm additionally attaches the initial
   scene image; the LLMPlan arm is the identical prompt without the image (KinDER's controlled
   ±image pair).
2. Collect an **ordered sequence of up to N = k = 200 distinct skeletons** via batched generation
   rounds (protocol parity with the pyperplan pool budget; the enumerator's 40-candidate cap is
   oracle machinery and is deliberately **not** used as a budget anchor).
3. **Static constraint (hard line):** the model never observes refinement outcomes. Between rounds
   it sees only its own previously proposed plans (for dedup). Any outcome feedback makes this a
   different (adaptive) method and a different table row.
4. Score the sequence against memoised feasibility labels (stable refine seed); live-refine
   off-pool proposals; fall back to published order on exhaustion (disclosed censoring-to-B2).

**Framing in the paper:** the zero-training-data, generic-perception endpoint of the
data × perception grid. PIGINet = trained low-level predictor; SPECTRE = trained abstract-first
predictor; VLMPlan = zero-shot pretrained endpoint. Not a defeated rival; a corner of the grid.

---

## 2. Module architecture

Target layout (names indicative; adjust to repo conventions):

| Module | Responsibility | Env-specific? |
|---|---|---|
| `vlmplan/template.py` | KinDER LLMPlan template string, verbatim, with `{controllers}`, `{typed_objects}`, `{type_hierarchy}`, `{goal_str}`, `{init_state_str}` slots; the two extension blocks (§4) behind config flags | No |
| `vlmplan/adapter.py` | `EnvAdapter` abstract interface (below) | Interface: no |
| `vlmplan/dd2d_adapter.py` | DD2D implementation of `EnvAdapter` | **Yes — the only DD2D-aware module** |
| `vlmplan/models.py` | Thin factory over **`prpl_llm_utils`** (prpl-mono dependency, not new code): model construction + per-backend decode config | No |
| `vlmplan/parsing.py` | Output parser **vendored from `kinder-vlm-planning/utils.py`** with modified error semantics (§2.3) | No |
| `vlmplan/loop.py` | Batched generation loop, dedup context, exhaustion rule, static-constraint enforcement | No |
| `vlmplan/run.py` | CLI harness: split dir → per-problem JSONL transcripts + sequences | No |
| `vlmplan/score.py` | Sequence → attempts metric + diagnostics, stratified reporting | No (calls adapter's `label`) |
| `vlmplan/probe_blockers.py`, `vlmplan/probe_packing.py` | The two upstream probes (§8) | Yes (thin) |

### 2.1 `EnvAdapter` interface

The contract a new environment must implement — everything else is shared:

```python
class EnvAdapter(ABC):
    # --- prompt content ---
    def controllers_str(self, problem) -> str          # skills block, KinDER format
    def typed_objects_str(self, problem) -> str
    def type_hierarchy_str(self, problem) -> str
    def goal_str(self, problem) -> str                 # includes env semantics disclosure (§4.4)
    def init_state_str(self, problem) -> str
    def images(self, problem, labeled: bool) -> list   # [] for LLMPlan arm

    # --- output handling ---
    def parse_plan_lines(self, text) -> list[RawPlan]  # delegates to the shared vendored
                                                       #   parser (§2.3); env supplies the
                                                       #   skill/type/object tables
    def ground(self, raw: RawPlan, problem) -> Skeleton | Invalid
                                                       # symbolic applicability check
    def canonical_key(self, skeleton) -> Hashable      # DD2D: the ORDERED member tuple (§5)
    def published_order(self, problem) -> list[Skeleton]  # fallback source (B2)

    # --- labeling ---
    def label(self, skeleton, problem) -> Feasibility  # memoised refiner, stable seed;
                                                       #   live-refines off-pool skeletons
```

DD2D implementations are thin wrappers over existing code: `record_ext` / record JSONs for state
and images, `staging_skeleton` for grounding, the shared `DD2DRefiner` + memoisation (as in
`heuristic_experiment.py`) for `label`, the pyperplan pool for `published_order`.

### 2.2 Model backend — reuse `prpl_llm_utils` (do not write new backend code)

The backend layer specified in earlier drafts already exists as lab infrastructure:
`prpl_llm_utils` (in `prpl-mono`), the same library `kinder-vlm-planning` uses via
`create_vlm_by_name`. It provides the `PretrainedLargeModel` abstraction with `OpenAIModel`,
`OpenAIResponsesModel`, and `GeminiModel` subclasses; disk caching (file-based or SQLite) keyed by
query + model-id with hyperparameter-aware separation; PIL image attachment; seed-aware queries;
and multi-response support. **We depend on it directly** — lab-standard, tested, shared cache
format — and `vlmplan/models.py` is only a factory that constructs the right model + cache from
config.

**Local open-weight mode requires no new code:** `OpenAIModel` constructs a bare
`openai.OpenAI()`, and the OpenAI client honors the `OPENAI_BASE_URL` environment variable, so
pointing it at a local vLLM server (serving Qwen3-VL etc.) is a config/env change. If an explicit
`base_url` constructor argument is preferred over env plumbing, that is a five-line subclass —
or a small PR upstreamed to prpl-mono, which is the better long-term home for it.

Conventions on top of the library:

- **Caching:** `prpl_llm_utils`' cache is the response cache — resumable runs, offline re-scoring,
  and the released transcripts for the API arm's reproducibility story. Log the full
  request/response pair per round alongside it (JSONL).
- **Decoding (per-backend, all logged):** temperature 0 and fixed max tokens for the local model.
  Frontier reasoning models may reject or ignore temperature 0 (KinDER's own runs use
  `temperature=1` with GPT-5.2); for the frontier arm, match KinDER's published settings — itself
  a comparability argument — and record them. Hyperparameter-keyed caching keeps the arms'
  responses separated.
- **Failure handling:** transient API errors retry with backoff; a permanently failed round counts
  as a zero-yield round for the exhaustion rule (never silently skipped).

**Model choices** (verify exact checkpoint names at implementation time; this landscape moves):
primary = open-weight VLM local via vLLM — Qwen3-VL 8B (bf16) or 32B (AWQ) on the RTX 5090;
InternVL3 as the MIT-licensed alternative. Secondary = one frontier API model (KinDER used
GPT-5.2), version string pinned and recorded. Four cells total:
{LLMPlan, VLMPlan} × {open-local, frontier-API}.

### 2.3 Output parser — vendored from `kinder-vlm-planning`, one semantic change

`parse_model_output_into_option_plan` (`kinder-vlm-planning/src/kinder_vlm_planning/utils.py`,
~150 lines) already implements the template-format line parser with real validation: skill-name
lookup, `obj:type` splitting, type checking against the hierarchy, arity checking, and
continuous-param parsing against the declared `Box` shape. We **vendor it** (copy with
attribution) rather than import it, for two reasons:

1. **Dependency weight:** the original is typed against `relational_structs.Object/Type` and
   `bilevel_planning.LiftedParameterizedController` — heavy machinery for DD2D's one type, three
   skills, and empty parameter spaces. The vendored copy strips these to plain lookup tables
   supplied by the `EnvAdapter`, keeping the validation logic line-for-line otherwise.
2. **Error semantics must change:** the original **breaks and truncates the whole plan** at the
   first malformed line — correct for their single-plan open-loop setting, wrong for ours. The
   vendored copy invalidates **only the plan block containing the bad line** (drop-plan-on-error);
   other plans in the same response survive. Since the semantics change regardless, importing for
   "provably identical parsing" buys nothing.

Upstream of the per-line parser, `loop.py` splits each response into `Plan 1:` … `Plan B:` blocks
(the multi-plan extension, §4.1); the vendored parser then handles each block exactly as the
original handles its single `Plan:` section.

---

## 3. Prompt: template fidelity and slot fills

The base prompt is **loaded from the official prompt files** in the `kinder-baselines` release
(`kinder-vlm-planning/src/kinder_vlm_planning/prompts/llm_planning_prompt.txt`) rather than
transcribed from the paper's Fig. 7 — byte-level fidelity, and the strongest form of the "same
template" claim. Note: `llm_planning_prompt.txt` and `vlm_planning_prompt.txt` are **byte-identical**
in the release; VLMPlan is the same prompt with images attached at query time, which is exactly how
our ±image arms are constructed. Template structure: role preamble → skills
block (`ParameterizedController` format with `types` and `params_space: Box(...)`) → "only allowed
to use the provided skills" + permission to describe the scene and reason first → typed objects →
type hierarchy → goal expression → initial state → rigid output format
(`skill(obj:type, ...)[params]`, brackets mandatory even when empty, no numbering, no formatting,
reasoning above the `Plan:` heading only). Full prompt goes in the paper appendix, as KinDER does.

DD2D slot fills:

| Slot | Content |
|---|---|
| `{controllers}` | `pick(item)`, `place-buffer(item)`, `retrieve(item)`; discrete args only; `params_space = Box([], [], (0,), float)` with a note that low-level placement poses are chosen by a downstream sampler. Do **not** solicit continuous params — the VLM's job ends at the skeleton, and emitting poses would either be ignored (misleading) or change the refiner protocol (breaks comparability) |
| `{typed_objects}` | `mug_2: item`, …; target flagged (via a `target` type or a stated fact) |
| `{type_hierarchy}` | trivial (`item`, optionally `target - item`) |
| `{goal_str}` | `(extracted target_0)` + the semantics disclosure sentence (§4.4) |
| `{init_state_str}` | init literals (`in-drawer`, `handempty`) **plus** geometry text from `record_ext`: per-object `at-pose [x, y, θ]`, shape `{family, w, h, area, concave}`, drawer/buffer bounds. This is the DD2D analog of KinDER's object-centric state, and giving it to both arms is what makes LLMPlan-vs-VLMPlan the pixels-vs-coordinates comparison |
| images | VLMPlan arm only: top-down render with **item names overlaid** at centroids (Set-of-Mark-style; built from `render_scene`'s per-item segmentation + id→name) |

Output lines look like `place-buffer(mug_2:item)[]` — fully within the template's stated format.

---

## 4. Deviations from KinDER — complete list, each disclosed

Everything not listed here is verbatim.

1. **Multi-plan output block.** KinDER requests a single plan (their metric is open-loop SR); our
   metric needs an ordered sequence. The output section requests `Plan 1:` … `Plan B:` per round
   (B = plans per round, set by pilot; see §6), each block in the unmodified per-line format, with
   the added instruction that plans must be distinct and ordered most→least likely to be executable
   by the low-level refiner. Config `plans_per_round=1` reproduces the literal single-plan format
   (used for the KinDER-comparable secondary metric if we ever want it isolated).
2. **"Previously proposed plans" section** (rounds ≥ 2 only). Modeled on the LLMCon template's
   "Completed plans:" slot but renamed to `Previously proposed plans (do not repeat):` and
   containing **plans only, never outcomes** (static constraint). Round 1 omits the section, so
   round 1 is maximally template-faithful.
3. **Labeled scene image** (VLMPlan arm). KinDER attaches raw RGB; their scenes are legible and
   name↔pixel correspondence is inferable. DD2D item names are arbitrary and appear nowhere in a
   raw render, so without overlays the correspondence is unsolvable in principle and VLMPlan
   degenerates to LLMPlan + noise. Disclosed as a grounding aid.
4. **One-sentence semantics disclosure** in the task description: items may obstruct each other's
   grasps, and already-staged items may obstruct later buffer placements, so the **order** of
   staging can determine feasibility, and permutations of the same item set are distinct plans.
   Rationale: the PDDL is deliberately geometry-blind, so nothing in the formal inputs conveys
   this; the trained methods absorb it through labels. Natural-language disclosure of omitted
   domain semantics, not leakage — printed in the appendix for reviewers to judge.
5. **Parser error semantics** (harness-side, not prompt-side; details in §2.3). KinDER's parser
   truncates the entire plan at the first malformed line; our vendored copy drops only the
   offending plan block, because a multi-plan response must not lose valid later plans to one bad
   line. Validation logic is otherwise line-for-line identical to the release.

Zero-shot is preserved exactly: **no in-context examples** in any arm. (A VLMCon-style few-shot
arm is a separate, later decision, only if the zero-shot result demands the "you handicapped it"
defense.)

---

## 5. Generation loop, dedup, exhaustion

```
for each test problem:
    seq = []                       # ordered, deduped skeletons
    stall = 0
    for round in 1..MAX_ROUNDS:
        prompt = template(problem, prior_plans=seq)      # arm decides ±image
        text   = backend.generate(prompt)                # cached
        raws   = parse_plan_lines(text)                  # drop malformed (logged)
        new    = [ground(r) for r in raws if applicable] # drop invalid (logged)
        new    = dedup(new, key=canonical_key, against=seq)   # drop dupes (logged)
        seq   += new[: N - len(seq)]
        yield_rate = len(new) / plans_per_round
        stall = stall + 1 if yield_rate < TAU else 0
        if len(seq) >= N or stall >= R: break
    seq += published_order(problem, excluding=seq)       # censoring-to-B2 fill, flagged per-slot
```

**Dedup key = the ordered member tuple.** Same set in a different removal order is a distinct
skeleton: extraction order is load-bearing (blockers can block blockers — the enumerator's own
DFS exists to find a valid removal order) and insertion order gates the accessible-packing
certificate. Deduplicating on the unordered set would delete legitimate distinctions.
Canonicalization = normalized `(members…, retrieve(target))` tuple, nothing more.

**Exhaustion — operational definition (no enumerator-derived caps):** generation ends when the
budget N = 200 is reached, or when the **new-valid-unique yield rate falls below τ for R
consecutive rounds** (starting values τ = 0.2, R = 2; finalize in the pilot and freeze before the
main run), or at a hard safety cap MAX_ROUNDS (backstop only, sized in the pilot so it never binds
in practice — if it binds, report it). Remaining slots fill with published order and are flagged,
so an exhausted episode degrades to B2, not to a censored missing value. The **stall depth per
problem is itself a reported diagnostic** — where zero-shot proposal capacity runs out is part of
the result.

**Filtered-for-free:** malformed lines, symbolically inapplicable plans, and duplicates are dropped
without consuming attempt budget. Justification: other arms draw from a planner that guarantees
symbolic validity by construction, and symbolic checking is free relative to refinement, which is
the resource the metric counts. All three rates are reported per arm.

---

## 6. Pilot (before freezing anything)

~5 train-split problems, local model only, half a day:

- Choose **plans-per-round B**: how many distinct valid plans per response before quality or
  distinctness collapses (working guess 10–25 — a guess to be measured, not a spec).
- Confirm parse robustness on real model output; tune the (shared) format parser.
- Sanity-check the SoM render at model input resolution (~1024 px; overlay legibility).
- Set τ, R, MAX_ROUNDS from observed yield curves; freeze all loop constants before touching the
  test split.

---

## 7. Scoring and diagnostics (`score.py`)

**Primary:** attempts-to-first-feasible over the (VLM sequence + B2 fill), stratified by
`min_feasible_subset` and pooled — identical protocol to the heuristic-experiment arms (which
already establish the precedent of comparing heterogeneous orderings on the shared metric with
per-skeleton memoised labels). Off-pool proposals are labeled by a live memoised refiner call with
the stable seed; **off-pool rate is reported** (high off-pool rate = a finding about pool coverage,
worth a sentence).

**Secondary (free):** feasibility of Plan 1 of round 1 = the KinDER-style open-loop VLMPlan
success rate, computable from the same transcripts. Cite the KinDER convention for it.

**Diagnostics, all from logged data, no extra runs:**

- Per-round curves: validity rate, novelty rate, hit (feasible-found) rate vs depth — does
  proposal quality degrade at depth, and where.
- **Trivial-mimicry null (pre-registered):** per-problem rank correlation between the VLM
  sequence and published ascending-|S| order. Near-1 correlation ⇒ the number is uninformative
  relative to B2 regardless of where it lands (size-ascending enumeration, not geometric
  reasoning). This is a conjectured failure mode to check, not a predicted outcome.
- Invalid / duplicate / parse-failure rates per arm and model.
- Exhaustion depth distribution; fraction of episodes reaching B2 fill before first feasible.

**Pre-registered predictions** (write down before the main run): (1) confident — beats published
order on s1, where single-blocker identification is a canonical visual task; (2) the design
thesis under test — margin shrinks or inverts on s2/s3 where joint packing dominates;
(3) conjecture — gap to trained methods widens as λ tightens.

---

## 8. Upstream probes (run first — they carry the interpretation)

Both probes are interface-independent, reuse existing ground truth, and cost ~half a day total on
20–30 hand-checkable episodes. They are the only clean decomposition of *why* a generated sequence
is good or bad, since the sequence itself confounds PDDL fluency, enumeration bias, and geometric
judgment.

- **Blocker identification:** "which items prevent every grasp of the target?" vs `enumerate.py`
  ground-truth blocker sets. Include a stratum-0 case (unblocked target; correct answer = stage
  nothing) to catch an always-stage-something bias before it silently costs stratum 0.
- **Packing judgment:** "will this set of items jointly fit on the buffer with clearance?" —
  binary, vs the labeler's packing certificate. Isolates the global continuous statistic DD2D is
  built around.

If Probe 1 passes and Probe 2 fails, the headline VLMPlan number confirms the benchmark's design
thesis rather than merely "VLM bad." If both fail, check raw grounding (can it list the items?)
before concluding anything.

---

## 9. Build order (each stage a self-contained deliverable)

1. Probes 1–2 (half day; also a paper figure regardless of outcome).
2. SoM render function + `DD2DAdapter` prompt-content methods (day).
3. Install `prpl_llm_utils` + write the `models.py` factory config; stand up vLLM serving of the
   chosen open model on the 5090 and verify `OPENAI_BASE_URL` routing end-to-end (half day —
   backend code itself is a dependency, not a deliverable).
4. Generation loop + vendored parser (§2.3) + dedup + exhaustion; pilot on 5 train problems; freeze constants
   (1–2 days).
5. Main run, local model, both arms (LLMPlan/VLMPlan), full test split (~overnight; most
   refinements hit the memoisation cache since proposals will frequently coincide with
   already-labeled pool members).
6. Frontier-API arm(s) — same harness, different backend config (~hours + small API spend).
7. `score.py` reporting + diagnostics; λ-interaction analysis if/when the λ sweep exists.

Future environments: implement a new `EnvAdapter` (+ thin probes), point the CLI at it. Nothing in
`template.py`, `backends.py`, `loop.py`, or `score.py` changes.

---

## 10. Review risks and stated defenses

- **Zero-shot vs trained:** framed on the data axis, never as a defeated rival; the few-shot
  extension is the reserve defense, not a default arm.
- **Heterogeneous pools:** the VLM's effective pool differs from the pyperplan pool; defense = the
  repo's own 5-arm heuristic experiment already compares different orderings on the shared metric.
  State it in the paper before being asked.
- **Deviations from KinDER:** all four are enumerated (§4), each with rationale; full prompt in
  appendix.
- **API reproducibility:** cached transcripts released, model version pinned, temperature 0;
  acknowledged that hosted models are not bit-reproducible over time — which is why the open local
  model is the primary arm.
- **Contamination:** non-issue (procedural, novel benchmark); one sentence saying so.
- **Honesty gate:** no headline numbers until the negative certificate lands; interim outputs
  labeled diagnostic, same as the rest of the repo.
