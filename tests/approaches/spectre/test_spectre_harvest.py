"""Tests for the offline geometry-grounded post-mortem harvest (Step 11).

The harvest recovers typed facts for each failed skeleton from the record's *stored*
geometry + refiner metadata — never by re-refining or regenerating. These pin the fact
types, their proof/hint tiers, and the load-bearing soundness property (a proof fact
never fires on a feasible subset).
"""

from __future__ import annotations

import pytest

shapely = pytest.importorskip("shapely")

from alphatamp.approaches.spectre.envs.dd2d.spectre_harvest import (  # noqa: E402
    _metadata_hints,
    harvest_facts_from_geometry,
)
from alphatamp.approaches.spectre.schema import (  # noqa: E402
    ObjectGeometry,
    SceneGeometry,
)


def _scene_geometry_from_live(scene) -> SceneGeometry:
    from alphatamp.approaches.spectre.schema import ContainerGeometry

    objs = []
    for name, st in scene.items.items():
        ring = tuple(
            (float(px), float(py))
            for px, py in list(st.shape.polygon.exterior.coords)[:-1]
        )
        objs.append(
            ObjectGeometry(
                name=name,
                pose=tuple(float(v) for v in st.pose),
                boundary=ring,
                family=st.shape.family,
                area=float(st.shape.area),
                concave=bool(st.shape.concave),
                is_target=(name == scene.target),
            )
        )
    bx0, by0, bx1, by1 = scene.buffer.bounds
    return SceneGeometry(
        objects=tuple(objs),
        containers=(ContainerGeometry(kind="buffer", bounds=(bx0, by0, bx1, by1)),),
        frame={"drawer_w": scene.dims["W"], "drawer_d": scene.dims["D"]},
    )


def _live_scene(seed=4):
    from alphatamp.approaches.spectre.envs.dd2d.dd2d.scene import generate_scene

    return generate_scene(seed=seed, lam=0.8, crowd=5)


def test_metadata_hints_extraction_and_pack():
    facts = []

    def add(t, args, scalars=()):
        facts.append((t, tuple(args), tuple(scalars)))

    _metadata_hints(add, "pick(o11)", frozenset({"o11", "o3"}), 1.0)
    _metadata_hints(add, "place-buffer(o5, pose)", frozenset({"o5"}), 4.0)
    kinds = {f[0]: f for f in facts}
    assert kinds["extraction-failed"][1] == ("o11",)
    assert kinds["pack-exhausted"][1] == ("o5",)
    assert kinds["pack-exhausted"][2] == (("n_attempts", 4.0),)


def test_extraction_failed_is_hint_tier():
    sg = _scene_geometry_from_live(_live_scene())
    others = [o.name for o in sg.objects if not o.is_target]
    pm = harvest_facts_from_geometry(
        sg,
        frozenset(others),  # everything removed ⇒ target open ⇒ no blocked-at-contents
        skeleton_idx=3,
        refinement_seed=0,
        run_certificate=False,
        failure_action=f"pick({others[0]})",
    )
    ef = [f for f in pm.facts if f.fact_type == "extraction-failed"]
    assert ef and ef[0].tier == "hint" and ef[0].args == (others[0],)


def test_blocked_at_contents_is_proof_and_sound_on_feasible():
    """Blocked-at-contents is proof-tier and never fires on a subset that opens the
    target."""
    scene = _live_scene(seed=7)
    sg = _scene_geometry_from_live(scene)
    others = frozenset(o.name for o in sg.objects if not o.is_target)
    # removing everything opens the target -> the proof must NOT fire (soundness).
    pm = harvest_facts_from_geometry(
        sg, others, skeleton_idx=0, refinement_seed=0, run_certificate=False
    )
    assert not [f for f in pm.facts if f.fact_type == "blocked-at-contents"]
    # removing nothing (empty subset) at a crowded scene: if blocked, it is proof-tier.
    pm0 = harvest_facts_from_geometry(
        sg, frozenset(), skeleton_idx=0, refinement_seed=0, run_certificate=False
    )
    for f in pm0.facts:
        if f.fact_type == "blocked-at-contents":
            assert f.tier == "proof"
