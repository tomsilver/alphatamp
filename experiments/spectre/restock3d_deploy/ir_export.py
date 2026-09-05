"""Export the winning plan's intermediate representation (IR) for skeleton injection.

This is the seam between alphatamp and the real-robot stack: instead of shipping a
joint trajectory (Level B), we ship only the planner's decisions — the skill skeleton
(which object to pick next, which shelf section it goes to) and the continuous
placement position on that section. The consumer is
``kinder_bilevel_planning.injection.run_injected_sesame``, whose CylinderShelf3D
skills refine the skeleton into motion for their own robot; execution-side parameters
(staging distances, grasp pitch/depth, base standoffs) are the consumer's calibration,
not part of the IR.

Vocabulary translation (alphatamp -> kinder):

- objects: ``scene.objects[i]`` -> ``cylinder{i}`` (by scene-file order, which the
  kinder scene config must share);
- ``pick(robot, o)`` -> ``MoveToPreGrasp(robot, cylinder_i)`` + ``Grasp(robot,
  cylinder_i)``;
- ``place_tall(robot, o)`` / ``place_short(robot, o)`` -> ``Place(robot, cylinder_i,
  shelf)`` with board layer 0 (bottom) / 1 (middle) recorded in ``placements``;
- placement position: the placed object's final pose, expressed as x/y offsets from
  the shelf centre so it is frame-independent (alphatamp plans in the robot-home
  frame, the consumer in its map frame).

x-offsets are clamped to ``X_OFFSET_LIMIT``: the consumer's OnFixture predicate uses
an x-tolerance of 0.20 m about the shelf centre, and an offset at the boundary makes
success a coin flip. Clamping distorts the packing (it shrinks the margin to the
neighbouring placement), so the limit sits above the planner's own band edge and a
clamp should be rare; any clamp is recorded in ``warnings``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

FORMAT = "restock3d-ir-v1"

#: Board layer (index into the consumer's bottom-up board list) per place operator.
_PLACE_LAYERS = {"place_tall": 0, "place_short": 1}

#: Largest exported |x - shelf_x|; see module docstring.
X_OFFSET_LIMIT = 0.19


def build_ir(plan, action_plan, scene, *, meta: Optional[dict] = None) -> dict:
    """Assemble the IR dict from the refined plan and the deploy scene."""
    cylinder_of = {o.name: f"cylinder{i}" for i, o in enumerate(scene.objects)}
    skeleton: list[list] = []
    placements: dict[str, dict] = {}
    final_state = plan.states[-1]
    sx, sy = scene.config.shelf_pose.position[0], scene.config.shelf_pose.position[1]
    warnings: list[str] = []
    for op in action_plan:
        target = next((p.name for p in op.parameters if p.name in cylinder_of), None)
        if target is None:
            raise ValueError(f"Operator {op.name} has no scene object argument")
        cyl = cylinder_of[target]
        if op.name == "pick":
            skeleton.append(["MoveToPreGrasp", ["robot", cyl]])
            skeleton.append(["Grasp", ["robot", cyl]])
        elif op.name in _PLACE_LAYERS:
            skeleton.append(["Place", ["robot", cyl, "shelf"]])
            pose = final_state.get_object_pose(target)
            x_off = float(pose.position[0]) - sx
            if abs(x_off) > X_OFFSET_LIMIT:
                warnings.append(
                    f"{target}: x_offset {x_off:+.3f} clamped to +-{X_OFFSET_LIMIT}"
                )
                x_off = max(-X_OFFSET_LIMIT, min(X_OFFSET_LIMIT, x_off))
            placements[cyl] = {
                "layer": _PLACE_LAYERS[op.name],
                "section": "tall" if op.name == "place_tall" else "short",
                "x_offset": round(x_off, 4),
                "y_offset": round(float(pose.position[1]) - sy, 4),
            }
        else:
            raise ValueError(f"Operator {op.name} is not supported by {FORMAT}")
    unplaced = sorted(set(cylinder_of.values()) - set(placements))
    if unplaced:
        raise ValueError(f"Plan leaves objects unplaced: {unplaced}")
    return {
        "format": FORMAT,
        "objects": [
            {
                "name": o.name,
                "cylinder": cylinder_of[o.name],
                "height": o.height,
                "radius": round(o.width / 2, 4),
            }
            for o in scene.objects
        ],
        "skeleton": skeleton,
        "placements": placements,
        "warnings": warnings,
        **({"meta": meta} if meta else {}),
    }


def export_ir(
    plan, action_plan, scene, out_dir: str | Path, *, meta: Optional[dict] = None
) -> Path:
    """Write ``plan_ir.json`` into ``out_dir`` and return its path."""
    ir = build_ir(plan, action_plan, scene, meta=meta)
    path = Path(out_dir) / "plan_ir.json"
    path.write_text(json.dumps(ir, indent=2))
    return path
