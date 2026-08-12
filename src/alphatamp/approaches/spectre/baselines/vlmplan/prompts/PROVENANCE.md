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
   precondition on where the presser is standing: pressing a button leaves whatever
   pressed it standing on that button.

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

   **Corrected 2026-08-01, after the first version of the rule proved false.** It
   originally read "the first press is `...FromNothing`, every later press is
   `...FromButton`", which holds only *within* one uninterrupted run of presses by the
   same effector. `PlaceStick` and `PickStickFromButton` both re-establish
   `(AboveNoButton)`, so the press after either is `...FromNothing` again; and arm presses
   track `RobotAboveButton` while stick presses track `StickAboveButton`, so an arm run
   and a stick run can never chain into one another. A model following the old text
   produced mixed plans that could not ground — on b5 train problem 750000, round 0 was
   **19 parsed, 19 inapplicable**, and both b5 pilots returned zero usable plans.

   This matters for how the deviation should be read: a disclosure that *removes* a
   handicap only does so if it is correct. A wrong one is worse than silence, because the
   model obeys it. The corrected note states the effector-separation rule and the two
   reset actions, and gives a correct example of each of the two mixed strategies. Pinned
   by `test_vlmplan_sb2d.py::test_mixed_stick_then_arm_plan_grounds` and its converse.

9. **Quantitative gripper geometry in the geometry block** (`dd2d_adapter._geometry_str`,
   `sb2d_adapter._geometry_str`; added 2026-08-08 with the terra arm). Deviation 4 discloses
   the domain semantics *qualitatively* ("items obstruct the gripper's access"); an input
   audit found the prompt never gave the gripper's actual dimensions, even though DD2D
   feasibility is decided by exactly those dimensions — whether a two-finger, 2.5×2.0 cm,
   0.5–12 cm-aperture gripper can close on the target past its neighbours. The DD2D block now
   states the finger size, the aperture range and the number of approach angles; the SB2D
   block adds the arm-extension and gripper-jaw widths (its reach limit already carried the
   operative consequence). **The numbers are imported from the env** — `envs/dd2d/dd2d/grasps.py`
   (`FINGER_WIDTH`, `FINGER_THICK`, `MIN_APERTURE`, `MAX_APERTURE`, `N_DIRECTIONS`) and
   `StickButton2DEnvConfig` — so they can never drift from the model the refiner enforces.

   Same class as deviations 4/7/8: it **removes a handicap rather than granting an advantage**.
   The gripper is a fixed domain constant that every trained method absorbs from labels
   (SPECTRE from the feasibility labels, PIGINet from the same); a zero-shot VLM has no training
   to absorb it, so withholding it is a handicap unique to this baseline. It says nothing about
   *which* items to stage or in what order — the decision under test. Pinned by
   `test_vlmplan.py::test_dd2d_geometry_discloses_gripper_dimensions`.

   **Version note:** the **terra** arm (2026-08-08) is generated *with* this disclosure; the
   earlier **luna** and local-Qwen arms predate it. luna is dropped as the headline row, so the
   headline VLMPlan number is on the disclosed prompt; where the local `VLMPlan-32B` row still
   appears it is annotated as pre-disclosure.

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

## Frontier-VLM arm (gpt-5.6-luna, 2026-08-03)

The paper's headline VLMPlan row is **gpt-5.6-luna** over the OpenAI **Responses API**
(`model.backend: openai_responses`), replacing the local Qwen arms. The prompt is
byte-identical; only the model and decode change, plus the SB2D image source (below).

- **Responses API, not chat completions.** GPT-5 reasoning models reject `max_tokens` and a
  non-default `temperature` over chat completions. The Responses backend remaps
  `max_tokens → max_output_tokens` and drops `temperature`/`seed`, so **round-to-round
  diversity comes from the growing "previously proposed plans" block + the model's own
  sampling**, not from a per-round seed. This is still the static hard line — the model never
  sees an outcome.
- **`max_tokens = 16384`, `reasoning.effort = low`.** A reasoning model bills reasoning tokens
  against the output cap, so it is raised from 8192; `effort: low` keeps reasoning cost/latency
  down on problems that are easy for a human. Measured `n_truncated = 0` on both pilots and
  full runs; watch it on any re-collection. Recorded into every record via
  `ModelConfig.describe()`.
- **Wall-clock is dominated by generation, not refinement.** The Responses round-trip +
  reasoning is ~10–16 s/round, so VLMPlan's time-to-first-success (`infer_s`) is seconds–
  minutes; this is the honest cost of a zero-shot frontier planner and is reported in the §2b
  wall-clock section.

**Deviation 3, on the kinder-rendered SB2D variant.** For `stickbutton2d_v1_kinder`,
`image_source: kinder_labeled` attaches **kinder's own environment render** (the PIGINet-parity
pixels) with Set-of-Mark object labels overlaid — because kinder draws every unpressed button as
an identical unlabeled red disc, so the name↔disc correspondence is unsolvable without the
overlay, exactly the situation deviation 3 addresses on DD2D. Labels are the canonical object
names (`circle_N`/`rectangle_N`/`crv_robot_N`), drawn in data coordinates via kinder's
`ax_callback` so they sit exactly on the objects. The reach line is *not* drawn on this image
(the table band shows the base-exclusion zone and the numeric reach limit is in the text
prompt). The schematic renderer remains the default `image_source` for the other SB2D variant.

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
