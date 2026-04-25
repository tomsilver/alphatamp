"""Vocabulary extraction for the skeleton encoder Φ.

Per ``SPECTRE_METHOD_SPEC.md`` §4.1 the encoder uses fixed-size embedding tables
keyed by lifted-operator name, predicate name, and type name. The vocab is
extracted from the training split *only* and then frozen for val/test use.

The ``<OOV>`` slot is reserved at index 0 even though v0.1 hard-fails on OOV
(``SPECTRE_METHOD_SPEC.md`` §8.5); this way, the graceful-fallback upgrade
path is a one-line change.

**Trajectory coverage.** Because we persist only ``s_0`` and the per-skeleton
``final_abstract_state`` (Substage A, §4.1.5), a naive scan of stored atoms
misses any predicate that only ever appears in intermediate states — e.g.
``Holding(?robot, ?block)`` in ClutteredStorage2D, which is added by a Pick
and deleted by the matching Place, so it never shows up in ``s_0`` or
``s_L`` of a goal-reaching plan. We close this hole by reconstructing the full
trajectory from ``(s_0, operator_seq)`` via STRIPS progression
(:func:`alphatamp.approaches.spectre.trajectory.reconstruct_trajectory`) and
scanning every atom in every reachable state. As a side benefit, this also
means switching to Substage B later needs no schema change — the intermediate
states are already available on demand.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bilevel_planning.structs import RelationalAbstractState

from alphatamp.approaches.spectre.io import list_episodes, load_episode
from alphatamp.approaches.spectre.trajectory import reconstruct_trajectory

OOV_TOKEN = "<OOV>"


@dataclass(frozen=True)
class Vocab:
    """Fixed-size symbol → index mapping plus observed maxima."""

    config_hash: str
    operators: dict[str, int]
    predicates: dict[str, dict[str, int]]  # name -> {"arity": int, "idx": int}
    types: dict[str, int]
    max_operator_arity: int
    max_predicate_arity: int
    max_skeleton_length: int
    max_atoms_per_state: int
    max_objects_per_state: int
    max_pool_size: int
    max_objects_per_type: dict[str, int] = field(default_factory=dict)
    # Per-type augmentability for object-renumbering augmentation
    # (SPECTRE_RT2D_METHOD_SPEC.md §4.6 / §10.1). Missing keys default to
    # augmentable=True so kinder vocabs round-trip unchanged.
    type_aug_policy: dict[str, bool] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        """Write a JSON serialization."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, path: Path) -> "Vocab":
        """Load a vocab JSON written by ``to_json``."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # Older vocab JSON files predate type_aug_policy; treat them as all-augmentable.
        raw.setdefault("type_aug_policy", {})
        return cls(**raw)

    def is_augmentable(self, type_name: str) -> bool:
        """Default-True policy lookup, per spec §10.1 backwards-compatibility note."""
        return self.type_aug_policy.get(type_name, True)

    def op_idx(self, name: str) -> int:
        """Map an operator name to its embedding index.

        Raises on OOV.
        """
        if name not in self.operators:
            raise KeyError(
                f"Unknown lifted operator '{name}'. OOV fallback not"
                " implemented; see SPECTRE_METHOD_SPEC.md §8.5."
            )
        return self.operators[name]

    def pred_idx(self, name: str) -> int:
        """Map a predicate name to its embedding index.

        Raises on OOV.
        """
        if name not in self.predicates:
            raise KeyError(f"Unknown predicate '{name}'")
        return self.predicates[name]["idx"]

    def type_idx(self, name: str) -> int:
        """Map a type name to its embedding index.

        Raises on OOV.
        """
        if name not in self.types:
            raise KeyError(f"Unknown type '{name}'")
        return self.types[name]


def extract_vocab(split_dir: Path, config_hash: str) -> Vocab:
    """Scan every episode in ``<split_dir>/episodes/`` and collect the vocab.

    For each skeleton we reconstruct ``[s_0, s_1, …, s_L]`` via STRIPS
    progression and scan every atom in every state — not just the stored
    ``s_0`` / ``final_abstract_state`` — so predicates that only live in
    intermediate states are not missed.

    Indices are assigned in lexicographic order for determinism; ``<OOV>`` is
    reserved at index 0 in each table.
    """
    operator_names: set[str] = set()
    predicate_arity: dict[str, int] = {}
    type_names: set[str] = set()

    max_op_arity = 0
    max_pred_arity = 0
    max_skel_len = 0
    max_atoms = 0
    max_objs = 0
    max_pool = 0
    max_per_type: dict[str, int] = {}

    def _record_state(state: RelationalAbstractState) -> None:
        nonlocal max_atoms, max_objs, max_pred_arity
        max_atoms = max(max_atoms, len(state.atoms))
        max_objs = max(max_objs, len(state.objects))
        type_counts: dict[str, int] = {}
        for atom in state.atoms:
            predicate_arity.setdefault(atom.predicate.name, atom.predicate.arity)
            max_pred_arity = max(max_pred_arity, atom.predicate.arity)
            for e in atom.entities:
                type_names.add(e.type.name)
        for obj in state.objects:
            type_names.add(obj.type.name)
            type_counts[obj.type.name] = type_counts.get(obj.type.name, 0) + 1
        for type_name, count in type_counts.items():
            max_per_type[type_name] = max(max_per_type.get(type_name, 0), count)

    for path in list_episodes(split_dir):
        ep = load_episode(path)
        max_pool = max(max_pool, len(ep.skeleton_pool))
        _record_state(ep.initial_abstract_state)
        for atom in ep.goal_atoms:
            predicate_arity.setdefault(atom.predicate.name, atom.predicate.arity)
            max_pred_arity = max(max_pred_arity, atom.predicate.arity)
            for e in atom.entities:
                type_names.add(e.type.name)
        for skel in ep.skeleton_pool:
            max_skel_len = max(max_skel_len, len(skel.operator_seq))
            # Reconstruct the full trajectory so predicates like Holding —
            # which appear only between a Pick's add-effect and its Place's
            # delete-effect — are captured in the vocab.
            trajectory = reconstruct_trajectory(
                ep.initial_abstract_state, skel.operator_seq
            )
            # trajectory[0] == ep.initial_abstract_state, already recorded above.
            for state in trajectory[1:]:
                _record_state(state)
            # Defense in depth: also scan the stored final; divergence from
            # trajectory[-1] would indicate schema corruption but does not
            # block vocab extraction.
            _record_state(skel.final_abstract_state)
            for op in skel.operator_seq:
                operator_names.add(op.name)
                max_op_arity = max(max_op_arity, len(op.parameters))
                for arg in op.parameters:
                    type_names.add(arg.type.name)

    def _indexed(names: list[str]) -> dict[str, int]:
        return {OOV_TOKEN: 0, **{name: i + 1 for i, name in enumerate(sorted(names))}}

    predicates = {}
    for i, name in enumerate([OOV_TOKEN] + sorted(predicate_arity)):
        predicates[name] = {
            "arity": 0 if name == OOV_TOKEN else predicate_arity[name],
            "idx": i,
        }

    return Vocab(
        config_hash=config_hash,
        operators=_indexed(sorted(operator_names)),
        predicates=predicates,
        types=_indexed(sorted(type_names)),
        max_operator_arity=max_op_arity,
        max_predicate_arity=max_pred_arity,
        max_skeleton_length=max_skel_len,
        max_atoms_per_state=max_atoms,
        max_objects_per_state=max_objs,
        max_pool_size=max_pool,
        max_objects_per_type=max_per_type,
    )


def validate_vocab(vocab: Vocab, split_dir: Path) -> list[str]:
    """Return a deduplicated list of OOV findings; empty means clean.

    Does not raise — the caller decides whether to hard-fail. Findings check both stored
    atom sets *and* the reconstructed trajectories (so an OOV predicate that only
    appears in intermediate states is still flagged).
    """
    findings_set: set[str] = set()

    def _check_atoms(atoms, where: str) -> None:
        for atom in atoms:
            if atom.predicate.name not in vocab.predicates:
                findings_set.add(f"predicate '{atom.predicate.name}' in {where}")
            for e in atom.entities:
                if e.type.name not in vocab.types:
                    findings_set.add(f"type '{e.type.name}' in {where}")

    for path in list_episodes(split_dir):
        ep = load_episode(path)
        pid = ep.provenance.problem_id
        _check_atoms(ep.goal_atoms, f"ep_{pid:05d} goal")
        _check_atoms(ep.initial_abstract_state.atoms, f"ep_{pid:05d} s_0")
        for skel in ep.skeleton_pool:
            loc = f"ep_{pid:05d} skel_{skel.skeleton_idx}"
            try:
                trajectory = reconstruct_trajectory(
                    ep.initial_abstract_state, skel.operator_seq
                )
            except AssertionError as exc:
                findings_set.add(f"invalid skeleton at {loc}: {exc}")
                continue
            for i, state in enumerate(trajectory[1:], start=1):
                _check_atoms(state.atoms, f"{loc} state[{i}]")
            _check_atoms(skel.final_abstract_state.atoms, f"{loc} stored_final")
            for op in skel.operator_seq:
                if op.name not in vocab.operators:
                    findings_set.add(f"operator '{op.name}' in {loc}")
                for arg in op.parameters:
                    if arg.type.name not in vocab.types:
                        findings_set.add(f"type '{arg.type.name}' in {loc}")
    return sorted(findings_set)
