"""DD2D geometry sidecar for PIGINet records (Step 1 of docs/piginet_dd2d_plan.md).

The shared :class:`~blocks_tamp.record.PIGINetExample` schema is *symbolic*: for a DD2D
problem its ``init_literals`` are near-constant (``(handempty)``/``(target,X)``/
``(in-drawer,o)``) and ``objects[].size`` is only the axis-aligned bbox ``(w, h, 6.0)``.
But DD2D feasibility is a purely **geometric** property (which items straddle the target's
grasp corridors, and whether the chosen clearing subset packs the buffer), so a PIGINet
trained on the bare records would see constant input. This module adds the geometric
channel the full-multimodal model consumes, WITHOUT touching the shared schema (its
``objects``/``init_literals`` are free-form lists):

* :func:`write_crops` -- render the scene once and write one segmented crop PNG per object
  (shared across all of a problem's plans), returning :class:`~blocks_tamp.record.ImageRef`
  s with their ``path`` filled (the base ``build_image_refs`` leaves ``path=None``).
* :func:`build_dd2d_example` -- build the base record via ``record.build_example``, then
  augment it in place with per-object ``pose``/``shape``, ``at-pose`` init facts (explicit
  continuous-value tokens for the Init sequence), and the drawer frame in ``provenance``
  (so raw-cm poses are normalizable at train time from the record alone).

Poses appear both on ``objects[].pose`` (fused into ``g_obj = MLP([g_img; g_val(pose)])``)
and as ``at-pose`` init literals (explicit value tokens) -- intentional, mirroring PIGINet
§IV where poses are present implicitly in images and explicitly in initial literals.
"""

from __future__ import annotations

import os

from ..record import ImageRef, PIGINetExample, build_example, build_image_refs
from ..refine import RefineResult
from ..skeleton import Skeleton
from .problem import DD2DProblem
from .render import render_scene

_ROUND = 4  # decimal places for stored geometry (compact + stable JSON round-trip)


def write_crops(
    problem: DD2DProblem, images_dir: str, views: tuple[str, ...] = ("topdown",)
) -> list[ImageRef]:
    """Render ``problem.scene`` once and write one segmented crop PNG per object per
    view.

    Reuses :func:`blocks_tamp.record.build_image_refs` for the ``seg_id`` + pixel-space
    ``bbox`` of each object, then crops the rendered RGB by that bbox and writes it. The
    stored ``ImageRef.path`` is **relative** (``<basename(images_dir)>/<obj>__<view>.png``)
    so the dataset is portable; the Step-7 loader resolves it against the record's problem
    dir. Called once per problem (crops are identical across the problem's plans).
    """
    import imageio.v2 as imageio

    os.makedirs(images_dir, exist_ok=True)
    rel_prefix = os.path.basename(os.path.normpath(images_dir))
    problem_dir = os.path.dirname(os.path.normpath(images_dir))
    refs: list[ImageRef] = []
    for vi, view in enumerate(views):
        render = render_scene(problem.scene, view=view)
        # Save the full initial-state frame once (topdown/first view) next to the crops -- the
        # "initial state" overview the dataset-point visualizer shows. Reuses this render (no
        # extra draw).
        if vi == 0 and problem_dir:
            imageio.imwrite(os.path.join(problem_dir, "scene.png"), render.rgb)
        # build_image_refs iterates the same (object x view) order and fills seg_id/bbox;
        # restrict it to this single view so seg ids match this render.
        view_refs = build_image_refs(problem, render=render, views=(view,))
        for ref in view_refs:
            if ref.bbox is None or ref.seg_id is None:
                refs.append(
                    ref
                )  # no segment (defensive) -> keep path=None, write nothing
                continue
            r0, c0, r1, c1 = ref.bbox
            crop = render.rgb[r0 : r1 + 1, c0 : c1 + 1]
            fname = f"{ref.object}__{view}.png"
            imageio.imwrite(os.path.join(images_dir, fname), crop)
            ref.path = f"{rel_prefix}/{fname}"
            refs.append(ref)
    return refs


def build_dd2d_example(
    problem: DD2DProblem,
    skeleton: Skeleton,
    refine_result: RefineResult,
    planner_name: str,
    images: list[ImageRef] | None = None,
    label_source: str = "refine_buffer_stage",
    extra_provenance: dict | None = None,
) -> PIGINetExample:
    """Build a :class:`PIGINetExample` and augment it with DD2D geometry (in place).

    ``images`` should be the refs from :func:`write_crops` for this problem (shared across
    its plans). Adds ``pose``/``shape`` to each ``objects[]`` dict, one ``at-pose`` init
    fact per object, and the drawer frame (``drawer_wh``/``buffer_bounds``) to provenance.
    """
    ex = build_example(
        problem,
        skeleton,
        refine_result,
        planner_name,
        images=images,
        label_source=label_source,
        extra_provenance=extra_provenance,
    )

    scene = problem.scene
    # 1) per-object pose + shape from the live scene state
    for obj in ex.objects:
        st = scene.items[obj["name"]]
        x, y, theta = st.pose
        obj["pose"] = [round(x, _ROUND), round(y, _ROUND), round(theta, _ROUND)]
        w, h = st.shape.size
        obj["shape"] = {
            "family": st.shape.family,
            "w": round(w, _ROUND),
            "h": round(h, _ROUND),
            "area": round(st.shape.area, _ROUND),
            "concave": bool(st.shape.concave),
        }

    # 2) explicit pose-carrying init facts: ["at-pose", name, [x, y, theta]] (one per object)
    for obj in ex.objects:
        x, y, theta = scene.items[obj["name"]].pose
        ex.init_literals.append(
            [
                "at-pose",
                obj["name"],
                [round(x, _ROUND), round(y, _ROUND), round(theta, _ROUND)],
            ]
        )

    # 3) normalization reference (raw-cm poses -> [-1,1] at train time needs the drawer frame)
    ex.provenance["drawer_wh"] = [
        round(scene.dims["W"], _ROUND),
        round(scene.dims["D"], _ROUND),
    ]
    ex.provenance["buffer_bounds"] = [round(b, _ROUND) for b in scene.buffer.bounds]
    return ex
