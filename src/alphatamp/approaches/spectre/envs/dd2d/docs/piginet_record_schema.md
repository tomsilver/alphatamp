# PIGINet training-example record schema

A `blocks_tamp.record.PIGINetExample` is one training/eval datapoint for a plan
feasibility predictor `f(I, π, G) → [0, 1]` (PIGINet paper, Yang et al. 2023,
lines 67–76, 197, 223). One example = one *candidate task plan* for one problem,
plus the inputs PIGINet scores it from, plus a noisy feasibility label.

This document defines the schema and maps every field to the original (unreleased)
`fastamp` contract — the functions `get_facts_goals_visuals` / `get_plans_labels`
and the `PVT` checker in
`kitchen-worlds/pybullet_planning/pigi_tools/feasibility_checkers.py` — so a
re-implemented PIGINet can consume our records with a known correspondence.

## Fields

| Field | Type | Meaning | PIGINet paper | `fastamp` correspondence |
|---|---|---|---|---|
| `schema_version` | str | Schema version (`"1.0"`). | — | — |
| `problem_id` | str | Stable id of the source problem. | — | data-folder name |
| `objects` | list of dict | Object set 𝒪: `{name, category, color, size, is_blocker, start_table}`. | 𝒪 | object list passed to the visual encoder |
| `init_literals` | list of `[pred, *args]` | Initial state literals ℐ. | ℐ | `facts` from `get_facts_goals_visuals` |
| `goal_literals` | list of `[pred, *args]` | Goal literals 𝒢. | 𝒢 | `goals` from `get_facts_goals_visuals` |
| `task_plan` | list of `[operator, *args]` | The task plan π — the skeleton with **continuous args omitted** (paper Table II). | π | the plan tokenised by `get_plan_skeleton` |
| `images` | list of `ImageRef` | Per-object segmented-image references (see below). | rendered segmented images | `visuals` from `get_facts_goals_visuals` |
| `label` | bool | Feasibility: `true` iff refinement bound the skeleton within the attempt budget. | feasibility label | positive/negative from `get_plans_labels` |
| `label_source` | str | How the label was decided (`"refine_timeout"`). | "solved within timeout ⇒ positive" | same labelling rule |
| `refine` | dict | Diagnostics: `{status, steps_bound, plan_length, n_attempts, failure_action}`. | — (our addition) | — |
| `provenance` | dict | `{planner, seed, num_blocks, num_blockers, generator, …}`. | — | — |

### `ImageRef`

| Field | Type | Meaning |
|---|---|---|
| `object` | str | Object this image depicts. |
| `view` | str | Viewpoint (`"topdown"`, `"oblique"`, …). |
| `seg_id` | int \| null | Segmentation id of the object in the rendered frame. |
| `bbox` | `[row_min, col_min, row_max, col_max]` \| null | Object bbox from the segmentation mask (for cropping). |
| `path` | str \| null | File path once pixels are rendered; `null` = **deferred** (pixels not yet written). |

The PIGINet paper renders 6 viewpoints of segmented objects; an occluded object's
crop is all-background. Here, pixel rendering is **confirmed doable** (PyBullet /
numpy-2D backends produce real RGB + segmentation masks headlessly — see
`rendering.confirm_rendering`) but deferred: records carry the image *schema*
(object, view, seg id, bbox) with `path = null`. Writing crops later is mechanical
— fill `path` from the bbox already stored.

## Faithfulness notes / deliberate simplifications

- **Continuous args are intentionally absent** from `task_plan`, matching the
  PIGINet "task plan" (Table II): grasp/IK/placement values appear only during
  refinement, never in π.
- **Labels are noisy** by design (paper §"Deciding plan feasibility is NP-hard"):
  positive iff our sampling-based refinement found a binding within the attempt
  budget; a harder budget could flip borderline cases.
- **Feasibility comes from one of two refiners** (recorded in
  `provenance.refiner` and `label_source`): the default **`pybullet`** refiner
  (`refine_pybullet.py`) uses a *real* 7-DoF Panda — sampled top-down grasp +
  inverse kinematics + collision checks of the arm against the other objects
  (`label_source="refine_ik_collision"`); the **`analytic`** refiner (`refine.py`)
  uses a reach + top-down-clearance heuristic (`label_source="refine_clearance"`).
  Both are config-level (grasp + IK + collision, matching the released LAZY
  domain) — not a full motion planner with RRT. Labels remain noisy by design.
- **Symbolic vs geometric split:** geometric preconditions (`ik`, `colfree-block`,
  `table-support`) are dropped from the PDDL domain and handled in refinement —
  this is what turns obstruction into a *refinement* failure (the feasibility
  signal), exactly as in the LAZY/PIGINet pipeline.

## Example (abridged)

```json
{
  "schema_version": "1.0",
  "problem_id": "sorting_b3_k2_s5",
  "objects": [{"name": "red_block0", "category": "block", "color": "red",
               "size": [0.045, 0.045, 0.045], "is_blocker": false,
               "start_table": "purple_table"}, "..."],
  "init_literals": [["handempty"], ["on-table", "red_block0", "purple_table"],
                    ["clear", "red_block0"], "..."],
  "goal_literals": [["on-table", "red_block0", "red_table"], "..."],
  "task_plan": [["pick", "green_block0", "blue_table"],
                ["place", "green_block0", "green_table"], "..."],
  "label": true,
  "label_source": "refine_timeout",
  "refine": {"status": "feasible", "steps_bound": 6, "plan_length": 6,
             "n_attempts": 1, "failure_action": null},
  "images": [{"object": "red_block0", "view": "topdown", "seg_id": 4,
              "bbox": [197, 131, 204, 136], "path": null}, "..."],
  "provenance": {"planner": "symk-topk", "seed": 5, "num_blocks": 3,
                 "num_blockers": 2, "generator": "blocks_tamp.problem.generate_sorting_problem"}
}
```
