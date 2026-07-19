"""Convert DD2D PIGINet-style JSON records into SPECTRE ``EpisodeRecord``s.

Each DD2D problem directory (``data/dd2d/raw_v2/<split>/dd2d_..._s<seed>/``)
holds one ``NNN.json`` per candidate skeleton — all sharing the same
``objects`` / ``init_literals`` / ``goal_literals`` but differing in
``task_plan`` and feasibility ``label``. That is exactly one SPECTRE episode: a
pool of candidate skeletons, each with a success/fail outcome.

This module maps a directory to a validated :class:`EpisodeRecord`:

- ``initial_abstract_state`` — the STRIPS atoms of ``init_literals``. The
  DD2D-only ``at-pose`` literals (continuous geometry) are dropped: SPECTRE is
  deliberately x0-free, so the abstract state carries only the six drawer
  predicates (see ``spectre_operators.py``).
- ``goal_atoms`` — ``{extracted(target)}`` from ``goal_literals``.
- ``skeleton_pool`` — one ``SkeletonRecord`` per JSON record: ``operator_seq``
  grounded from ``task_plan``; ``final_abstract_state`` recovered by STRIPS
  progression (``trajectory.reconstruct_trajectory``).
- ``outcomes`` — ``label == true`` → ``"success"``, else ``"fail"``. The DD2D
  ``label`` is a Day-1 labeler output: ``false`` means "not certified feasible /
  marginal", not proven-infeasible (see ``MIGRATION_DD2D.md`` §4). Fine for
  training; no label-dependent research number should be reported until DD2D's
  arrangement-complete negative certificate lands.

The **abstract state** stays x0-free (the ``at-pose`` literals are still dropped). Since
v2.2.1, ground-truth object-centric geometry (per-object pose + boundary ring + buffer)
IS carried alongside, on ``EpisodeRecord.scene_geometry`` (see ``_parse_scene_geometry``),
for the geometry-aware v2 model; a raw dir collected before the ``boundary`` field
existed yields ``scene_geometry=None`` and converts abstract-only as before.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from bilevel_planning.structs import RelationalAbstractState
from relational_structs import GroundAtom, GroundOperator, Object

from alphatamp.approaches.spectre.envs.dd2d.spectre_operators import (
    ALL_OPERATORS,
    ALL_PREDICATES,
    ALL_TYPES,
    OPERATOR_BY_NAME,
    PREDICATE_BY_NAME,
    ItemType,
)
from alphatamp.approaches.spectre.schema import (
    ContainerGeometry,
    EpisodeRecord,
    ObjectGeometry,
    OutcomeRecord,
    ProvenanceBlock,
    SceneGeometry,
    SkeletonRecord,
    SummaryBlock,
)
from alphatamp.approaches.spectre.trajectory import reconstruct_trajectory

# v2: also carries ground-truth SceneGeometry (per-object pose + boundary ring + buffer),
# so the config_hash changes and vocab/train re-read cleanly.
CONVERTER_VERSION = "dd2d_convert_v2"
DEFAULT_ENV_ID = "dd2d/DrawerDeclutter2D-v0"


def config_hash(env_variant: str) -> str:
    """Deterministic 12-hex hash stamped on every converted episode.

    SPECTRE's vocab build reads ``config_hash`` off the first train episode and
    ``train.py`` prints it; neither enforces a match, so any stable value is
    sufficient. Keying it on the converter version + env variant + the drawer
    vocabulary means a schema change to this converter yields a new hash.
    """
    payload = json.dumps(
        {
            "converter": CONVERTER_VERSION,
            "env_variant": env_variant,
            "operators": sorted(op.name for op in ALL_OPERATORS),
            "predicates": sorted(p.name for p in ALL_PREDICATES),
            "types": sorted(t.name for t in ALL_TYPES),
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ground_atoms(
    literals: list[list[Any]], objects: dict[str, Object]
) -> set[GroundAtom]:
    """Parse ``[pred, *args]`` literals into ``GroundAtom``s.

    Literals whose head is not one of the six drawer STRIPS predicates — i.e.
    the DD2D ``at-pose`` geometry literals, whose second arg is a ``[x,y,theta]``
    list rather than an object name — are silently skipped.
    """
    atoms: set[GroundAtom] = set()
    for lit in literals:
        head = lit[0]
        pred = PREDICATE_BY_NAME.get(head)
        if pred is None:
            continue
        args = [objects[a] for a in lit[1:]]
        atoms.add(GroundAtom(pred, args))
    return atoms


def _ground_task_plan(
    task_plan: list[list[str]], objects: dict[str, Object]
) -> tuple[GroundOperator, ...]:
    """Ground a ``[[op, *args], ...]`` skeleton into a ``GroundOperator`` tuple."""
    seq: list[GroundOperator] = []
    for step in task_plan:
        op = OPERATOR_BY_NAME[step[0]]
        args = tuple(objects[a] for a in step[1:])
        seq.append(op.ground(args))
    return tuple(seq)


def _record_paths(problem_dir: Path) -> list[Path]:
    """Sorted ``NNN.json`` skeleton records in a problem directory."""
    return sorted(problem_dir.glob("[0-9]*.json"))


def _parse_scene_geometry(first: dict[str, Any]) -> SceneGeometry | None:
    """Build ground-truth ``SceneGeometry`` from a DD2D record's per-object geometry.

    Requires the ``boundary`` ring (written by ``record_ext.build_dd2d_example``); a raw
    dir collected before that field existed yields ``None`` (episode stays abstract-only,
    ``scene_geometry=None``). The buffer is stored as an exact-bounds container; the
    drawer dimensions go in ``frame`` (its world origin is not in the record).
    """
    objs: list[ObjectGeometry] = []
    for o in first["objects"]:
        boundary = o.get("boundary")
        shape = o.get("shape")
        pose = o.get("pose")
        if boundary is None or shape is None or pose is None:
            return None
        objs.append(
            ObjectGeometry(
                name=o["name"],
                pose=tuple(float(v) for v in pose),  # type: ignore[arg-type]
                boundary=tuple((float(px), float(py)) for px, py in boundary),
                family=str(shape["family"]),
                area=float(shape["area"]),
                concave=bool(shape["concave"]),
                is_target=(o.get("category") == "target"),
            )
        )
    prov = first.get("provenance", {})
    containers: list[ContainerGeometry] = []
    bb = prov.get("buffer_bounds")
    if bb is not None:
        containers.append(
            ContainerGeometry(kind="buffer", bounds=tuple(float(v) for v in bb))  # type: ignore[arg-type]
        )
    dw = prov.get("drawer_wh")
    frame = {"drawer_w": float(dw[0]), "drawer_d": float(dw[1])} if dw else None
    return SceneGeometry(
        objects=tuple(objs), containers=tuple(containers), units="cm", frame=frame
    )


def convert_problem_dir(
    problem_dir: Path,
    env_variant: str = "dd2d_v2",
    split: str = "train",
    env_id: str = DEFAULT_ENV_ID,
) -> EpisodeRecord:
    """Convert one DD2D problem directory into a validated ``EpisodeRecord``.

    Raises ``AssertionError`` (via ``reconstruct_trajectory``) if any skeleton
    violates STRIPS preconditions — surfacing a malformed record rather than
    silently mislabeling it. The batch driver catches per-problem so one bad
    directory does not abort a whole split.
    """
    record_paths = _record_paths(problem_dir)
    if not record_paths:
        raise ValueError(f"No NNN.json skeleton records in {problem_dir}")

    first = _read_json(record_paths[0])
    objects: dict[str, Object] = {
        o["name"]: Object(o["name"], ItemType) for o in first["objects"]
    }
    object_registry = {name: ItemType.name for name in objects}

    s0 = RelationalAbstractState(
        atoms=_ground_atoms(first["init_literals"], objects),
        objects=set(objects.values()),
    )
    goal_atoms = frozenset(_ground_atoms(first["goal_literals"], objects))

    prov = first.get("provenance", {})
    problem_seed = int(prov.get("seed", 0))

    skeleton_records: list[SkeletonRecord] = []
    outcome_records: list[OutcomeRecord] = []
    first_success_idx: int | None = None

    for idx, path in enumerate(record_paths):
        rec = _read_json(path)
        operator_seq = _ground_task_plan(rec["task_plan"], objects)
        # verify_preconditions=True: a violation means the stored plan is
        # inconsistent with the drawer STRIPS model — treat as corruption.
        trajectory = reconstruct_trajectory(s0, operator_seq, verify_preconditions=True)

        skeleton_records.append(
            SkeletonRecord(
                skeleton_idx=idx,
                operator_seq=operator_seq,
                final_abstract_state=trajectory[-1],
            )
        )

        outcome = "success" if bool(rec["label"]) else "fail"
        if outcome == "success" and first_success_idx is None:
            first_success_idx = idx

        rprov = rec.get("provenance", {})
        refine_meta: dict[str, object] = {
            "label_source": rec.get("label_source"),
            "plan_idx": rprov.get("plan_idx", idx),
        }
        refine_meta.update(rec.get("refine", {}))

        outcome_records.append(
            OutcomeRecord(
                skeleton_idx=idx,
                outcome=outcome,  # type: ignore[arg-type]
                refinement_wall_clock_s=0.0,
                refinement_seed=int(rprov.get("refine_seed", 0)),
                refiner_metadata=refine_meta,
            )
        )

    summary = SummaryBlock(
        num_skeletons=len(skeleton_records),
        num_success=sum(1 for o in outcome_records if o.outcome == "success"),
        num_fail=sum(1 for o in outcome_records if o.outcome == "fail"),
        num_error=sum(1 for o in outcome_records if o.outcome == "error"),
        first_success_idx=first_success_idx,
        total_wall_clock_s=0.0,
        pool_truncated=False,
    )

    provenance = ProvenanceBlock(
        problem_id=problem_seed,
        env_id=env_id,
        env_variant=env_variant,
        split=split,
        config_hash=config_hash(env_variant),
        problem_seed=problem_seed,
        git_sha=CONVERTER_VERSION,
        collection_timestamp="",
        package_versions={},
        # scene_latent is reserved for envs with an externally-sampled
        # per-episode refinement latent (RT2D). DD2D has none; the source
        # directory is recoverable from (split, problem_seed).
        scene_latent=None,
    )

    return EpisodeRecord(
        provenance=provenance,
        initial_abstract_state=s0,
        goal_atoms=goal_atoms,
        object_registry=object_registry,
        skeleton_pool=tuple(skeleton_records),
        outcomes=tuple(outcome_records),
        summary=summary,
        scene_geometry=_parse_scene_geometry(first),
    )
