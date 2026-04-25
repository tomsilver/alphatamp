"""Per-problem instance sampling with rejection (spec §4.3).

A ``ProblemInstance`` bundles every per-problem decision needed to drive the
closed-form planner and the three-gate refiner: the same-side bipartite
choice, robot home, item sources/targets, per-passage widths, per-item sizes,
and the per-episode latent ``(blocked_color, blocked_grasp)``. The relational
abstract state ``s0`` and goal are also pre-built here so the gym env's
observation-to-state path is trivial — the "observation" carries the full
ProblemInstance through to env_models.observation_to_state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from bilevel_planning.structs import (
    RelationalAbstractGoal,
    RelationalAbstractState,
)
from relational_structs import GroundAtom, Object

from alphatamp.approaches.spectre.envs.routedtransport2d import operators as ops
from alphatamp.approaches.spectre.envs.routedtransport2d import topology as topo
from alphatamp.approaches.spectre.envs.routedtransport2d.tags import (
    SIZE_LEVELS,
    WIDTH_LEVELS,
    is_compatible,
    sample_size,
    sample_width,
)

# Default mode prior (spec §3.2). π_color × π_grasp = product of axis priors.
DEFAULT_PI_COLOR: Final[dict[str, float]] = {"A": 0.50, "B": 0.30, "C": 0.20}
DEFAULT_PI_GRASP: Final[dict[str, float]] = {"top": 0.60, "side": 0.40}

# Rejection sampling budget per spec §4.3.
_MAX_TAG_REJECTIONS: Final[int] = 20
_MAX_LATENT_REJECTIONS: Final[int] = 20


# ---- Public dataclass -----------------------------------------------------


@dataclass(frozen=True)
class ProblemInstance:
    """Complete per-problem state: layout, tags, latent, s0 and goal."""

    seed: int
    variant: str  # e.g. "n3-v1"
    num_items: int
    which_side: str  # "L" or "R" — all item sources/targets on this side
    robot_home: str  # zone id
    item_sources: dict[str, str]  # item_name -> source_zone
    item_targets: dict[str, str]  # item_name -> target_zone
    passage_widths: dict[str, str]  # passage_name -> width level
    item_sizes: dict[str, str]  # item_name -> size level
    blocked_color: str  # 'A', 'B', 'C'
    blocked_grasp: str  # 'top', 'side'

    initial_abstract_state: RelationalAbstractState
    goal: RelationalAbstractGoal
    objects: tuple[Object, ...]

    @property
    def scene_latent(self) -> dict[str, str]:
        """Schema-style dict for ``ProvenanceBlock.scene_latent``."""
        return {
            "blocked_color": self.blocked_color,
            "blocked_grasp": self.blocked_grasp,
        }


# ---- Sampling primitives --------------------------------------------------


def _sample_latent(
    rng: np.random.Generator,
    pi_color: dict[str, float] | None = None,
    pi_grasp: dict[str, float] | None = None,
) -> tuple[str, str]:
    pi_c = pi_color if pi_color is not None else DEFAULT_PI_COLOR
    pi_g = pi_grasp if pi_grasp is not None else DEFAULT_PI_GRASP
    colors = sorted(pi_c)
    grasps = sorted(pi_g)
    blocked_color = colors[int(rng.choice(len(colors), p=[pi_c[c] for c in colors]))]
    blocked_grasp = grasps[int(rng.choice(len(grasps), p=[pi_g[g] for g in grasps]))]
    return blocked_color, blocked_grasp


def _sample_layout(
    rng: np.random.Generator, num_items: int
) -> tuple[str, str, dict[str, str], dict[str, str]]:
    """Pick which_side, robot_home, and per-item (source, target) zones.

    Constraint (a): no two items share a source zone (spec §4.3 step 4(a)).
    Constraint (b): no two items share a target zone (spec §4.3 step 4(b)).
    For ``num_items > 3``, constraint (a) is relaxed (spec §9.4 default option 1).
    """
    which_side = "L" if rng.random() < 0.5 else "R"
    robot_home = topo.ALL_ZONES[int(rng.integers(0, len(topo.ALL_ZONES)))]
    side_zones = topo.L_ZONES if which_side == "L" else topo.R_ZONES

    item_names = [f"item_{i}" for i in range(num_items)]

    # Sources: prefer distinct, fall back to with-replacement when N > |side|.
    if num_items <= len(side_zones):
        src_choice = list(rng.permutation(side_zones))[:num_items]
    else:
        src_choice = [
            side_zones[int(rng.integers(0, len(side_zones)))] for _ in range(num_items)
        ]
    item_sources = dict(zip(item_names, src_choice))

    # Targets: distinct, distinct from per-item source. For num_items > |side|
    # we still require pairwise-distinct targets (constraint b held); for N ≤ 3
    # the pigeonhole still allows it. For N=4 with shared sources we still
    # require distinct targets, sampled from side_zones.
    item_targets: dict[str, str] = {}
    if num_items <= len(side_zones):
        # Random distinct-target permutation, retried if any item gets target ==
        # source. With 3 zones and constraint (a), there's always a feasible
        # permutation (count >= 1 by inspection).
        while True:
            shuffled = list(rng.permutation(side_zones))[:num_items]
            if all(shuffled[i] != src_choice[i] for i in range(num_items)):
                item_targets = dict(zip(item_names, shuffled))
                break
    else:
        # N > |side_zones|: targets must repeat too. Pick uniformly with the
        # constraint target != source per item.
        for name, src in item_sources.items():
            candidates = [z for z in side_zones if z != src]
            item_targets[name] = candidates[int(rng.integers(0, len(candidates)))]

    return which_side, robot_home, item_sources, item_targets


def _sample_tags(
    rng: np.random.Generator, num_items: int
) -> tuple[dict[str, str], dict[str, str]]:
    """Sample per-passage widths and per-item sizes from the default distributions."""
    widths: dict[str, str] = {p: sample_width(rng) for p in topo.all_passage_names()}
    sizes: dict[str, str] = {f"item_{i}": sample_size(rng) for i in range(num_items)}
    return widths, sizes


# ---- Family feasibility check ---------------------------------------------


def _loaded_passages_for_item(
    src_zone: str,
    dst_zone: str,
    color_pair: tuple[str, ...],
) -> tuple[str, ...]:
    """Return the canonical loaded-traversal passages for one item under ``color_pair``
    — same BFS the planner uses, so feasibility analysis here matches the planner's
    actual emitted skeletons."""
    return tuple(
        p_name
        for p_name, _src, _dst in topo.bfs_color_pair_path(
            src_zone, dst_zone, color_pair
        )
    )


def _family_has_feasible_skeleton(
    color_pair: frozenset[str],
    item_sources: dict[str, str],
    item_targets: dict[str, str],
    item_sizes: dict[str, str],
    passage_widths: dict[str, str],
) -> bool:
    """Check whether at least one skeleton in family F_{color_pair, *} is tag-feasible —
    i.e., every loaded traversal it makes uses a passage whose width fits the carried
    item's size.

    Since same-side loaded traversals follow the canonical BFS path (a fixed function of
    (src, dst, color_pair)), every skeleton in the family uses the SAME loaded passages
    per item — only differs in item ordering and grasp. So the family's tag-feasibility
    reduces to: do every item's loaded passages have width >= item size?
    """
    pair_sorted = tuple(sorted(color_pair))
    assert len(pair_sorted) == 2
    for item_name, src in item_sources.items():
        dst = item_targets[item_name]
        passages = _loaded_passages_for_item(src, dst, pair_sorted)
        size = item_sizes[item_name]
        for p_name in passages:
            if not is_compatible(size, passage_widths[p_name]):
                return False
    return True


def _family_for_mode(
    blocked_color: str, blocked_grasp: str
) -> tuple[frozenset[str], str]:
    """Family that succeeds in mode ``(blocked_color, blocked_grasp)``.

    The success-mode rule (spec §3.4 table): F_{ij,g} succeeds in mode (blocked_color =
    the missing color from {A,B,C}\\{i,j},  blocked_grasp = the OTHER grasp).
    """
    other_colors = frozenset({"A", "B", "C"}) - {blocked_color}
    other_grasp = "side" if blocked_grasp == "top" else "top"
    return other_colors, other_grasp


# ---- Abstract state assembly ----------------------------------------------


def _build_objects(
    num_items: int,
) -> tuple[
    Object,
    tuple[Object, ...],
    dict[str, Object],
    dict[str, Object],
    dict[str, Object],
    dict[str, Object],
]:
    """Build ``(robot, items, zones, passages, width_levels, size_levels)``.

    Tuples and dicts allow callers to look objects up by name. The robot and items come
    back as Object instances; everything else is name->Object dicts.
    """
    robot = Object("robot_0", ops.RobotType)
    items = tuple(Object(f"item_{i}", ops.ItemType) for i in range(num_items))
    zones = {z: Object(z, ops.ZoneType) for z in topo.ALL_ZONES}
    passages: dict[str, Object] = {}
    for color in topo.COLORS:
        for name in topo.PASSAGE_NAMES[color]:
            passages[name] = Object(name, ops.passage_subtype_for_color(color))
    width_levels: dict[str, Object] = {
        w: Object(f"width_{w}", ops.WidthLevelType) for w in WIDTH_LEVELS
    }
    size_levels: dict[str, Object] = {
        s: Object(f"size_{s}", ops.SizeLevelType) for s in SIZE_LEVELS
    }
    return robot, items, zones, passages, width_levels, size_levels


def _build_initial_state(
    robot: Object,
    items: tuple[Object, ...],
    zones: dict[str, Object],
    passages: dict[str, Object],
    width_levels: dict[str, Object],
    size_levels: dict[str, Object],
    robot_home: str,
    item_sources: dict[str, str],
    passage_widths: dict[str, str],
    item_sizes: dict[str, str],
) -> RelationalAbstractState:
    atoms: set[GroundAtom] = set()
    # Robot at home, hand empty.
    atoms.add(ops.At([robot, zones[robot_home]]))
    atoms.add(ops.HandEmpty([robot]))
    # Items at their sources.
    for it in items:
        atoms.add(ops.ItemAt([it, zones[item_sources[it.name]]]))
    # Static: Connects (both directions).
    for color in topo.COLORS:
        for i, (l, r) in enumerate(topo.COLOR_EDGES[color]):
            p_obj = passages[topo.PASSAGE_NAMES[color][i]]
            atoms.add(ops.Connects([p_obj, zones[l], zones[r]]))
            atoms.add(ops.Connects([p_obj, zones[r], zones[l]]))
    # Static: PassageWidth.
    for p_name, w in passage_widths.items():
        atoms.add(ops.PassageWidth([passages[p_name], width_levels[w]]))
    # Static: ItemSize.
    for it in items:
        atoms.add(ops.ItemSize([it, size_levels[item_sizes[it.name]]]))

    objects: set[Object] = (
        {robot}
        | set(items)
        | set(zones.values())
        | set(passages.values())
        | set(width_levels.values())
        | set(size_levels.values())
    )
    return RelationalAbstractState(atoms=atoms, objects=objects)


def _build_goal(
    items: tuple[Object, ...],
    zones: dict[str, Object],
    item_targets: dict[str, str],
) -> RelationalAbstractGoal:
    goal_atoms: set[GroundAtom] = {
        ops.ItemAt([it, zones[item_targets[it.name]]]) for it in items
    }
    # state_abstractor is set by env_models to a real callable; for the
    # purposes of construction here the goal only needs ``atoms``. Use a
    # passthrough lambda — RelationalAbstractGoal asserts isinstance only.
    return RelationalAbstractGoal(atoms=goal_atoms, state_abstractor=lambda x: x)


# ---- Top-level entry point ------------------------------------------------


def make_problem(
    seed: int,
    variant: str = "n3-v1",
) -> ProblemInstance:
    """Sample a :class:`ProblemInstance` deterministically from a seed.

    The ``variant`` string follows spec §4.2: ``"n3-v1"``, ``"n4-v1"``, ``"n2-v1"``.
    """
    num_items = _num_items_from_variant(variant)
    rng = np.random.default_rng(seed)

    # We want: layout (which_side, robot_home, sources, targets) is fixed once;
    # tags and latent may resample under rejection (spec §4.3).
    which_side, robot_home, item_sources, item_targets = _sample_layout(rng, num_items)

    accepted = False
    passage_widths: dict[str, str] = {}
    item_sizes: dict[str, str] = {}
    blocked_color = ""
    blocked_grasp = ""

    for _ in range(_MAX_TAG_REJECTIONS):
        passage_widths, item_sizes = _sample_tags(rng, num_items)

        # Inner loop: for these tags, try several latents to find a feasible
        # (latent, family) pair. Per spec §4.3, if no latent works we go back
        # and resample tags.
        for _ in range(_MAX_LATENT_REJECTIONS):
            blocked_color, blocked_grasp = _sample_latent(rng)
            family_colors, _family_grasp = _family_for_mode(
                blocked_color, blocked_grasp
            )
            if _family_has_feasible_skeleton(
                family_colors,
                item_sources,
                item_targets,
                item_sizes,
                passage_widths,
            ):
                accepted = True
                break
        if accepted:
            break

    if not accepted:
        # Hard restart on a new RNG offshoot — keeps the seed→problem map
        # deterministic but breaks pathological seeds with no feasible mode.
        # Practical experience says this is rare under default tag distributions.
        return make_problem(seed + 7919, variant)  # large prime offset

    (
        robot,
        items,
        zones,
        passages,
        width_levels,
        size_levels,
    ) = _build_objects(num_items)
    s0 = _build_initial_state(
        robot,
        items,
        zones,
        passages,
        width_levels,
        size_levels,
        robot_home,
        item_sources,
        passage_widths,
        item_sizes,
    )
    goal = _build_goal(items, zones, item_targets)

    objects: tuple[Object, ...] = tuple(
        sorted(s0.objects, key=lambda o: (o.type.name, o.name))
    )

    return ProblemInstance(
        seed=seed,
        variant=variant,
        num_items=num_items,
        which_side=which_side,
        robot_home=robot_home,
        item_sources=dict(item_sources),
        item_targets=dict(item_targets),
        passage_widths=dict(passage_widths),
        item_sizes=dict(item_sizes),
        blocked_color=blocked_color,
        blocked_grasp=blocked_grasp,
        initial_abstract_state=s0,
        goal=goal,
        objects=objects,
    )


def _num_items_from_variant(variant: str) -> int:
    """Parse num_items from a variant string like ``"n3-v1"``."""
    # Variant convention: "n<N>-<version>".
    head = variant.split("-")[0]
    assert head.startswith("n"), f"unrecognized variant {variant!r}"
    return int(head[1:])
