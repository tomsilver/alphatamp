"""Offline, geometry-grounded post-mortem harvest for DD2D (Step 11 evidence pathway).

The definitive collection deliberately does **not** run the harvest in-loop
(``decisions.md`` 2026-07-19 decoupling). This module recovers the typed post-mortem
facts for each *failed* skeleton as a controlled offline pass — and, per the standing
**reconstruct-don't-regenerate** rule (``decisions.md`` 2026-07-19), it does so from the
record's *stored* ``scene_geometry``, never by regenerating or re-refining the scene.
Every fact here is a pure function of ``(geometry, staged subset)``:

| fact | tier (DD2D) | reconstruction | |---|---|---| | ``blocked-at-contents`` | proof
| target has no clear grasp with the staged subset removed
(``target_blocked_after_removing``) — removal-monotone | | ``grasp-witness`` | hint |
items blocking every still-open corridor after removal
(``grasp_witness_after_removing``) | | ``pack-impossible`` | proof | the §8.4
certificate proves the staged subset cannot pack |

The refiner-trace-only facts (``extracted-ok`` / ``packed-ok`` / ``pack-exhausted``)
need the bound prefix, which is not on the record and would require re-refinement (and
thus faithful scene regeneration); they are intentionally out of scope here so the whole
evidence pathway rides on the sound reconstruction basis. ``harvest_prefix`` /
``harvest_state_hash`` stay empty for the same reason (there is no replayed prefix).
"""

from __future__ import annotations

import time

from alphatamp.approaches.spectre.schema import (
    EpisodeRecord,
    Fact,
    PostMortemRecord,
    SceneGeometry,
)

from .drawer.certificate import certify_infeasible_by_packing
from .soundness import DD2D_REGISTRY, SoundnessRegistry
from .spectre_geometry import (
    grasp_witness_after_removing,
    reconstruct_scene,
    target_blocked_after_removing,
)


def _staged_subset(skeleton) -> frozenset[str]:
    """The clutter items a skeleton stages to the buffer (its ``place-buffer`` args)."""
    return frozenset(
        op.parameters[0].name
        for op in skeleton.operator_seq
        if op.name == "place-buffer"
    )


def _metadata_hints(add, failure_action: str, subset, n_attempts: float) -> None:
    """Typed hints read off the *stored* refiner metadata (no re-refinement).

    ``failure_action`` is the action the refiner stalled on: ``pick(item)`` — extraction
    stalled, a hint that ``item`` is buried given this subset — or ``place-buffer(...)``
    — staging exhausted its packing budget for the subset (P-F: these are observations
    of genuine attempts, not exact computations relabeled as features).
    """
    schema = failure_action.split("(")[0] if failure_action else ""
    if schema == "pick":
        inside = failure_action[
            failure_action.find("(") + 1 : failure_action.rfind(")")
        ]
        item = inside.split(",")[0].strip()
        if item:
            add("extraction-failed", (item,))
    elif schema == "place-buffer" and subset:
        add("pack-exhausted", subset, scalars=(("n_attempts", float(n_attempts)),))


def harvest_facts_from_geometry(
    scene_geometry: SceneGeometry,
    subset: frozenset[str],
    skeleton_idx: int,
    refinement_seed: int,
    registry: SoundnessRegistry = DD2D_REGISTRY,
    run_certificate: bool = True,
    scene=None,
    failure_action: str = "",
    n_attempts: float = 0.0,
) -> PostMortemRecord:
    """A geometry-grounded ``PostMortemRecord`` for one failed skeleton staging
    ``subset``.

    ``scene`` (a reconstructed ``DrawerScene``) may be passed to amortize reconstruction
    across an episode's failures; it is rebuilt from ``scene_geometry`` when omitted.
    ``failure_action`` / ``n_attempts`` come from the stored ``refiner_metadata`` and
    add the metadata-derived hints without any re-refinement.
    """
    t0 = time.perf_counter()
    facts: list[Fact] = []

    def add(fact_type: str, args, scalars=()) -> None:
        facts.append(
            Fact(
                fact_type=fact_type,
                args=tuple(sorted(args)),
                tier=registry.tier(fact_type),
                scalars=tuple(scalars),
            )
        )

    if scene is None:
        scene = reconstruct_scene(scene_geometry)

    # blocked-at-contents (proof) + its grasp-witness (hint), both after removing `subset`.
    if target_blocked_after_removing(scene_geometry, subset):
        present = [
            o.name
            for o in scene_geometry.objects
            if not o.is_target and o.name not in subset
        ]
        add("blocked-at-contents", present)
        witness = grasp_witness_after_removing(scene, subset)
        if witness:
            add("grasp-witness", witness, scalars=(("n_witness", float(len(witness))),))

    # pack-impossible (proof): the §8.4 certificate proves the staged subset cannot pack.
    if run_certificate and subset:
        if certify_infeasible_by_packing(scene, subset) is True:
            add("pack-impossible", subset)

    # metadata-derived hints (extraction-failed / pack-exhausted), no re-refinement.
    _metadata_hints(add, failure_action, subset, n_attempts)

    return PostMortemRecord(
        skeleton_idx=skeleton_idx,
        refinement_seed=refinement_seed,
        failed_schema=(failure_action.split("(")[0] if failure_action else None),
        facts=tuple(facts),
        harvest_cost_s=round(time.perf_counter() - t0, 6),
    )


def harvest_episode(
    episode: EpisodeRecord, run_certificate: bool = True
) -> EpisodeRecord:
    """Return a copy of ``episode`` with ``post_mortem`` populated on every ``fail``
    outcome.

    Reconstructs the scene once and harvests geometry-grounded facts per failed
    skeleton. Records without geometry are returned unchanged.
    """
    import dataclasses

    if episode.scene_geometry is None:
        return episode
    scene = reconstruct_scene(episode.scene_geometry)
    new_outcomes = []
    for skel, out in zip(episode.skeleton_pool, episode.outcomes):
        if out.outcome == "fail" and out.post_mortem is None:
            md = out.refiner_metadata or {}
            na = md.get("n_attempts", 0)
            pm = harvest_facts_from_geometry(
                episode.scene_geometry,
                _staged_subset(skel),
                skeleton_idx=out.skeleton_idx,
                refinement_seed=out.refinement_seed,
                run_certificate=run_certificate,
                scene=scene,
                failure_action=str(md.get("failure_action", "") or ""),
                n_attempts=float(na) if isinstance(na, (int, float)) else 0.0,
            )
            out = dataclasses.replace(out, post_mortem=pm)
        new_outcomes.append(out)
    return dataclasses.replace(episode, outcomes=tuple(new_outcomes))
