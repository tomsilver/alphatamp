"""Typed-local-id canonicalization for cross-problem skeleton equivariance.

Per ``docs/archive/SPECTRE_METHOD_SPEC.md`` §4.1.4: two skeletons that differ only by an
object renumbering must produce identical embeddings. We achieve this by
substituting every concrete object name with a within-type index
(``obstruction_0``, ``target_0``, ...).

Two modes:

- ``rng=None``: deterministic canonical ordering (objects sorted by original
  name, then enumerated within type). Used at evaluation / inference.
- ``rng=Generator``: random within-type permutation. Used at training as
  augmentation to prevent overfitting to any specific ordering convention.

Type identity is preserved from the input episode (not reconstructed) because
``relational_structs.Type`` uses full-field equality (including
``parent``) — hierarchical types must round-trip identically or downstream
``GroundOperator.__post_init__`` assertions will fail.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import numpy as np
from bilevel_planning.structs import RelationalAbstractState
from relational_structs import GroundAtom, GroundOperator, Object, Type

from alphatamp.approaches.spectre.schema import (
    AuxLabels,
    EpisodeRecord,
    PostMortemRecord,
    SceneGeometry,
    SkeletonRecord,
)


def _gather_types(episode: EpisodeRecord) -> dict[str, Type]:
    """Collect every ``Type`` actually referenced in the episode.

    We index by ``type.name`` and assert uniqueness so downstream code can look up the
    canonical ``Type`` instance that already carries the right parent-chain for the env.
    """
    found: dict[str, Type] = {}

    def _add(t: Type) -> None:
        existing = found.get(t.name)
        if existing is not None:
            # If equal, idempotent; if unequal, we have a conflict that would
            # break downstream is_instance checks.
            assert existing == t, (
                f"Multiple Type objects share name {t.name!r} but differ in"
                f" parent chain: {existing} vs {t}"
            )
            return
        found[t.name] = t

    for obj in episode.initial_abstract_state.objects:
        _add(obj.type)
    for atom in episode.initial_abstract_state.atoms:
        for e in atom.entities:
            _add(e.type)
    for atom in episode.goal_atoms:
        for e in atom.entities:
            _add(e.type)
    for skel in episode.skeleton_pool:
        for op in skel.operator_seq:
            for arg in op.parameters:
                _add(arg.type)
        for obj in skel.final_abstract_state.objects:
            _add(obj.type)
        for atom in skel.final_abstract_state.atoms:
            for e in atom.entities:
                _add(e.type)
    return found


def _renumber_mapping(
    episode: EpisodeRecord,
    rng: np.random.Generator | None,
    type_aug_policy: dict[str, bool] | None = None,
) -> dict[str, Object]:
    """Build a ``old_name -> new Object`` mapping with canonical-named objects.

    New object names are ``"{type_name}_{idx}"`` where ``idx`` is a within-type
    permutation index. ``Type`` instances are reused from the input episode so
    hierarchical-typing semantics are preserved.

    ``type_aug_policy`` is a ``{type_name: augmentable}`` dict (per
    ``docs/archive/SPECTRE_RT2D_METHOD_SPEC.md`` §4.6). Missing keys default to
    ``augmentable=True`` (backwards-compatible). When ``rng is None`` the policy is
    irrelevant — every type uses the deterministic alphabetical order.
    """
    type_table = _gather_types(episode)
    policy = type_aug_policy or {}

    by_type: dict[str, list[str]] = defaultdict(list)
    for obj_name, type_name in episode.object_registry.items():
        by_type[type_name].append(obj_name)

    mapping: dict[str, Object] = {}
    for type_name in sorted(by_type):
        names = sorted(by_type[type_name])
        augmentable = policy.get(type_name, True)
        if rng is None or not augmentable:
            permutation: list[int] = list(range(len(names)))
        else:
            permutation = list(rng.permutation(len(names)))
        if type_name not in type_table:
            # Registry mentions a type that appears only in object_registry but
            # nowhere inside a Type-carrying field. This can't happen in
            # episodes produced by collect.collect_episode (which derives
            # the registry from Object.type.name) but we guard anyway.
            raise KeyError(
                f"Type {type_name!r} referenced in object_registry but not"
                " found in any atom or operator. Cannot recover canonical Type."
            )
        type_obj = type_table[type_name]
        for original_name, new_idx in zip(names, permutation):
            new_name = f"{type_name}_{new_idx}"
            mapping[original_name] = Object(new_name, type_obj)
    return mapping


def _remap_atom(atom: GroundAtom, mapping: dict[str, Object]) -> GroundAtom:
    new_entities = [mapping[e.name] for e in atom.entities]
    return GroundAtom(atom.predicate, new_entities)


def _remap_state(
    state: RelationalAbstractState, mapping: dict[str, Object]
) -> RelationalAbstractState:
    new_atoms = {_remap_atom(a, mapping) for a in state.atoms}
    new_objects = {mapping[o.name] for o in state.objects}
    return RelationalAbstractState(atoms=new_atoms, objects=new_objects)


def _remap_operator(op: GroundOperator, mapping: dict[str, Object]) -> GroundOperator:
    """Re-ground through the parent ``LiftedOperator`` with the new objects."""
    assert op.parent is not None, "GroundOperator.parent required for remapping"
    new_objects = tuple(mapping[p.name] for p in op.parameters)
    return op.parent.ground(new_objects)


def _remap_scene_geometry(
    sg: SceneGeometry, mapping: dict[str, Object]
) -> SceneGeometry:
    """Rename each object's geometry to its canonical id (v2.2.1).

    Keeps geometry aligned with the canonicalized ``object_registry`` (invariant I5);
    containers / frame carry no object names.
    """
    objs = tuple(replace(o, name=mapping[o.name].name) for o in sg.objects)
    return replace(sg, objects=objs)


def _remap_aux_labels(aux: AuxLabels, mapping: dict[str, Object]) -> AuxLabels:
    return AuxLabels(
        necessary=frozenset(mapping[o].name for o in aux.necessary),
        relevant=frozenset(mapping[o].name for o in aux.relevant),
    )


def _remap_post_mortem(
    pm: PostMortemRecord, mapping: dict[str, Object]
) -> PostMortemRecord:
    """Rename the object-name arguments of every harvested fact to canonical ids
    (v2.2.1), so the typed-evidence tokens bind to the same tags as the scene/candidate
    tokens.

    Non-object scalars/prefix/hashes are untouched.
    """
    new_facts = tuple(
        replace(f, args=tuple(sorted(mapping[a].name for a in f.args)))
        for f in pm.facts
    )
    new_args = tuple(mapping[a].name for a in pm.failed_args)
    return replace(pm, facts=new_facts, failed_args=new_args)


def _remap_refiner_metadata(meta: dict, mapping: dict[str, Object]) -> dict:
    """Rename the object names inside v3 failure observations.

    ``refiner_metadata`` is a free-form dict, and its v3 ``failures`` entries carry object
    names in five roles: three flat lists (``args``, ``culprits``, ``unmoved``) and, for
    class-2 deviations, the argument lists *inside* ``dev_added`` / ``dev_deleted``. All of
    them must land in the same namespace as the scene and candidate tokens or the record
    tokens silently lose all object identity -- the tags simply fail to resolve and every
    record degenerates to "some failure of some schema". This is the same trap
    ``_remap_post_mortem`` exists for, and the nested case is the easier one to miss.

    Anything not a known object name (e.g. the ``__wall__`` sentinel, which is not an
    item) is passed through unchanged rather than dropped, so a non-object culprit stays
    visible as itself.
    """
    failures = meta.get("failures")
    if not isinstance(failures, (list, tuple)) or not failures:
        return meta

    def _rename(names) -> list[str]:
        return [
            mapping[n].name if n in mapping else n
            for n in (names or [])
            if isinstance(n, str)
        ]

    def _rename_atoms(atoms) -> list[list]:
        """``[[predicate, [arg, ...]], ...]`` with the arguments renamed in place."""
        return [
            [pair[0], _rename(pair[1])]
            for pair in (atoms or [])
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        ]

    out = dict(meta)
    out["failures"] = [
        (
            {
                **f,
                "args": _rename(f.get("args")),
                "culprits": _rename(f.get("culprits")),
                "unmoved": _rename(f.get("unmoved")),
                **(
                    {"dev_blame": _rename(f.get("dev_blame"))}
                    if "dev_blame" in f
                    else {}
                ),
                **(
                    {"dev_added": _rename_atoms(f.get("dev_added"))}
                    if "dev_added" in f
                    else {}
                ),
                **(
                    {"dev_deleted": _rename_atoms(f.get("dev_deleted"))}
                    if "dev_deleted" in f
                    else {}
                ),
            }
            if isinstance(f, dict)
            else f
        )
        for f in failures
    ]
    return out


def canonicalize_episode(
    episode: EpisodeRecord,
    rng: np.random.Generator | None = None,
    type_aug_policy: dict[str, bool] | None = None,
) -> EpisodeRecord:
    """Return a new ``EpisodeRecord`` with typed-local-id object names.

    Canonicalizes:
    - ``initial_abstract_state``
    - ``goal_atoms``
    - ``object_registry`` (names become ``"{type}_{idx}"``)
    - every skeleton's ``operator_seq`` and ``final_abstract_state``

    Provenance and summary are passed through unchanged. Outcomes are unchanged
    except for the v2.2.1 ``post_mortem`` fact arguments, which are renamed to the
    canonical object ids so the typed-evidence tokens share the scene tags.

    ``type_aug_policy`` (a ``{type_name: augmentable}`` dict per spec §4.6)
    suppresses the random within-type permutation for non-augmentable types
    even when ``rng`` is provided. Missing keys default to augmentable.
    """
    mapping = _renumber_mapping(episode, rng, type_aug_policy)

    new_s0 = _remap_state(episode.initial_abstract_state, mapping)
    new_goal = frozenset(_remap_atom(a, mapping) for a in episode.goal_atoms)
    new_registry = {obj.name: obj.type.name for obj in mapping.values()}

    new_skeletons: list[SkeletonRecord] = []
    for skel in episode.skeleton_pool:
        new_ops = tuple(_remap_operator(op, mapping) for op in skel.operator_seq)
        new_final = _remap_state(skel.final_abstract_state, mapping)
        new_skeletons.append(
            replace(
                skel,
                operator_seq=new_ops,
                final_abstract_state=new_final,
            )
        )

    # v2.2.1: keep the optional geometry / aux labels aligned with the canonical ids
    # (guarded — RT2D/kinder leave these None and are untouched).
    new_geometry = (
        _remap_scene_geometry(episode.scene_geometry, mapping)
        if episode.scene_geometry is not None
        else None
    )
    new_aux = (
        _remap_aux_labels(episode.aux_labels, mapping)
        if episode.aux_labels is not None
        else None
    )
    new_outcomes = tuple(
        replace(
            o,
            post_mortem=(
                _remap_post_mortem(o.post_mortem, mapping)
                if o.post_mortem is not None
                else None
            ),
            refiner_metadata=_remap_refiner_metadata(o.refiner_metadata or {}, mapping),
        )
        for o in episode.outcomes
    )

    return replace(
        episode,
        initial_abstract_state=new_s0,
        goal_atoms=new_goal,
        object_registry=new_registry,
        skeleton_pool=tuple(new_skeletons),
        outcomes=new_outcomes,
        scene_geometry=new_geometry,
        aux_labels=new_aux,
    )
