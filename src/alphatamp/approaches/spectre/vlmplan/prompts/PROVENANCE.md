# Prompt provenance

`kinder_llm_planning_prompt.txt` is a **byte-identical** copy of

```
kinder-vlm-planning/src/kinder_vlm_planning/prompts/llm_planning_prompt.txt
```

from **kinder-baselines** (Princeton Robot Planning and Learning group),
commit `4c731dc81d68ee6888ef3a989034991cd0694630` — the same revision this repo pins
in `pyproject.toml` (`decisions.md` 2026-07-18). MIT licensed, © 2025 Yichao Liang.

Verify with `md5sum`: `67491e7476937f2bbda73bdf9ec6a389`.

## Why vendored rather than imported

`kinder_vlm_planning` is **not** an installed dependency of alphatamp (only
`kinder_bilevel_planning` and `kinder_models` are), and the local `kb/` checkout is
gitignored — so importing the file at runtime would make the baseline unreproducible on
any other machine. The copy is the only way the prompt travels with the repo.

Note: upstream's `llm_planning_prompt.txt` and `vlm_planning_prompt.txt` are **identical
files**. VLMPlan in KinDER is the same prompt with images attached at query time, not a
different template — which is exactly how our arms are constructed, so one copy suffices.

## Deviations from the verbatim template

Everything not listed here is verbatim. Rationale for each is in
`docs/vlmplan_dd2d_implementation_plan.md` §4 and the `docs/decisions.md` VLMPlan ADR;
the deviations are applied by `vlmplan/template.py` as **appended blocks**, never by
editing the file above.

1. **Multi-plan output block.** KinDER requests a single plan (their metric is open-loop
   success rate); our metric needs an ordered *sequence*, so the output section requests
   `Plan 1:` … `Plan B:`, each block in the unmodified per-line format, distinct and
   ordered most→least likely to refine. `plans_per_round=1` reproduces the literal
   single-plan format.
2. **"Previously proposed plans (do not repeat):" block** (rounds ≥ 2 only), modelled on
   the upstream `llmplanner_planning_prompt.txt`'s "Completed plans:" slot. It contains
   **plans only, never outcomes** — the static-method hard line. Round 1 omits the block
   entirely, so round 1 is maximally template-faithful.
3. **Labelled scene image** (Set-of-Mark). KinDER attaches a raw RGB render; DD2D item
   names are arbitrary (`item_7`) and appear nowhere in a raw render, so without number
   overlays the name↔pixel correspondence is unsolvable in principle. Disclosed as a
   grounding aid.
4. **One-sentence domain-semantics disclosure** in the goal block: items obstruct each
   other's grasps and already-staged items obstruct later buffer placements, so staging
   *order* can decide feasibility and permutations of the same item set are distinct
   plans. The DD2D PDDL is deliberately geometry-blind, so nothing in the formal inputs
   conveys this; the trained methods absorb it from labels.

7. **Per-skill semantics in the controllers block.** The template lists controller
   signatures only, relying on the names being self-descriptive. In DD2D they are not:
   `pick` and `retrieve` both plausibly mean "get that item out", and
   `qwen3-vl-8b-instruct` duly ended 28/28 otherwise-valid plans with `pick(target)`
   instead of `retrieve(target)`. So the block states each skill's preconditions and
   effects in words, plus the structural consequence (plans are pick/place-buffer pairs
   ending in one `retrieve`).

   This **removes a handicap rather than granting an advantage**: every other method in the
   comparison reads those preconditions and effects from the PDDL domain, so this is the
   domain in words. It says nothing about *which* items to stage or in what order, which is
   the actual decision under test.

8. **The StickButton2D chaining rule** (`sb2d_adapter._CONTROLLER_NOTE`). Same class as
   deviation 7, on the second environment, and measured the same way. StickButton2D's
   operators come in `...FromNothing` / `...FromButton` pairs whose difference is a
   precondition on where the robot is standing: pressing a button leaves the robot on it,
   so only the *first* press of a plan can be the `FromNothing` variant.

   Left to infer that from the names, `qwen3-vl-32b-instruct` wrote `FromNothing` for
   every press. Measured on train problem 500000 (b3): **11/11 parsed plans violated a
   precondition, every one of them this one** — a 100% invalidity rate carrying no
   information about the model's planning ability. The note now states the rule and gives
   one correct three-press example.

   As with deviation 7 this **removes a handicap rather than granting an advantage**: the
   rule is a precondition the PDDL domain already states, and every other method in the
   comparison reads it from that domain for free — SPECTRE and PIGINet never even see an
   inapplicable candidate, because their pools come from the planner. It says nothing
   about *which* buttons to press or in what order, which is the decision under test.

   The same block also discloses the reach limit and the sweep-on-approach rule, for the
   same reason the DD2D block discloses geometry: StickButton2D's symbolic model is
   reach-blind, so a prompt without it describes a problem where every plan is equally
   good.

## Decode settings (part of reproducing a run)

`temperature = 1.0` and `max_tokens = 8192`, recorded into every cache record via
`ModelConfig.describe()`. Neither is arbitrary:

- **Temperature 1.0, not 0.** The loop asks for *diverse* plans across rounds; at
  temperature 0 the only thing varying between rounds is the repeat-suppression block, so a
  round that yields nothing leaves the next round with a byte-identical prompt. KinDER's own
  runs also use temperature 1. `loop.py` additionally varies `seed` per round.
- **`max_tokens = 8192`, and serve the model at ≥ 32768 context.** At 4096 the 2026-07-24
  run truncated 16/104 completions (`completion_tokens` == the cap exactly) and hit the 8192
  server window exactly, cutting the last plan block mid-line so the parser dropped it.
  `RoundLog.truncated` now flags this exactly, and a nonzero `n_truncated` on a record means
  that run under-reports the model and should be re-run rather than reported.

## Harness-side deviations (parser, not prompt)

5. **Per-block error semantics.** The vendored parser drops only the offending **plan
   block** on a malformed line, where upstream truncates the entire plan at the first bad
   line. Required because one response carries many plans and a later valid plan must not
   be lost to an earlier bad one.

6. **Three format leniencies, each counted and reported.** Small open-weight models
   frequently deviate cosmetically from the output format. Rejecting those would make the
   headline number a measure of instruction-following rather than of planning, so the
   parser accepts them and `ParseStats` counts each one
   (`n_decoration_repaired`, `n_type_omitted`, `n_brackets_omitted`):

   - **markdown decoration** — `- **pick(item_0:item)[]**`, which the template explicitly
     forbids;
   - **omitted `:type`** — `pick(item_0)`. The type is then read from the object registry
     and checked against the skill's declared argument type exactly as before. This does
     not weaken validation: the registry is authoritative and a model restating the type
     adds no information. A type that is stated but *wrong* is still rejected;
   - **omitted `[]`** — accepted only for a skill whose `params_space` is empty, where the
     brackets carry zero information. A skill that does take parameters still requires
     them, and a parameter supplied against an empty box is still rejected.

   What is *not* lenient: unknown skills, unknown objects, wrong types, wrong arity,
   unbalanced parentheses, non-float parameters, and symbolic inapplicability (checked
   downstream by the adapter against the real STRIPS operators).

   Measured need for this on `qwen3-vl-8b-instruct`: with the strict parser, 31/31 plan
   blocks in a round were rejected solely for writing `pick(item_2)` instead of
   `pick(item_2:item)[]` — a 100% parse-failure rate carrying no information about the
   model's planning ability.
