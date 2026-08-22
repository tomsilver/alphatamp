"""Bootstrap kinder environment ids that are not in the default registrations.

``kinder.register_all_environments()`` only registers a fixed set of variants (see
``kinder/__init__.py``). SPECTRE wants additional block/obstruction counts for dataset-
generation convenience. Registration is idempotent: calling with ids that already exist
is a no-op.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium
import kinder


@dataclass(frozen=True)
class ExtraVariant:
    """One extra gym id derived from a kinder env family."""

    family: str
    variant_char: str
    entry_point: str
    kwarg_name: str
    kwarg_value: int

    @property
    def gym_id(self) -> str:
        """The full ``kinder/...-v0`` id to register with gymnasium."""
        return f"kinder/{self.family}-{self.variant_char}{self.kwarg_value}-v0"


_CLUTTERED_STORAGE_ENTRY_POINT = (
    "kinder.envs.kinematic2d.clutteredstorage2d:ClutteredStorage2DEnv"
)
_CLUTTERED_RETRIEVAL_ENTRY_POINT = (
    "kinder.envs.kinematic2d.clutteredretrieval2d:ClutteredRetrieval2DEnv"
)
_STICK_BUTTON_ENTRY_POINT = "kinder.envs.kinematic2d.stickbutton2d:StickButton2DEnv"
_RESTOCK3D_ENTRY_POINT = (
    "alphatamp.approaches.spectre.envs.restock3d.kinematic_env:Restock3DEnv"
)
_RESTOCK3D_V2_ENTRY_POINT = (
    "alphatamp.approaches.spectre.envs.restock3d.kinematic_env:Restock3DV2Env"
)
_RESTOCK3D_V3_ENTRY_POINT = (
    "alphatamp.approaches.spectre.envs.restock3d.kinematic_env:Restock3DV3Env"
)
# v3 per-object-dims SPECTRE strata (banding stratum == recipe key, 0-3). One gym id per stratum;
# pooled into ``restock3d_v3`` by problem-id band (the shared 4-stratum band).
_RESTOCK3D_V3_STRATA = (0, 1, 2, 3)
# v2 continuous-packing SPECTRE strata (committed generator.STRATA keys 10-15). 10-13 =
# pilot 1x1..4x4; 14/15 = the full collection's asymmetric 3x4/4x3. One gym id per stratum;
# pooled into ``restock3d_v2_pilot`` (10-13) / ``restock3d_v2`` (11,12,14,15,13) by problem-id
# band.
_RESTOCK3D_V2_PILOT_STRATA = (10, 11, 12, 13, 14, 15)


# Per-type augmentation policy tables for SPECTRE Φ training-time
# object-renumbering augmentation (spec §4.6 / §10.1). Missing keys default to
# ``augmentable=True`` so kinder vocabs round-trip unchanged.

# DD2D has a single, fully-interchangeable object type (`item`): the target is
# marked by the `target` predicate, not by identity, so within-type permutation
# augmentation is correct and desirable. This entry is explicit for
# documentation — an absent entry defaults every type to augmentable anyway.
_DD2D_TYPE_AUG_POLICY: dict[str, bool] = {"item": True}

# StickButton2D has three types and all three are augmentable, but for different
# reasons worth writing down. `circle` (the buttons) are genuinely interchangeable --
# the goal names every one of them, so no button holds a privileged role that its
# local id has to encode. `crv_robot` and `rectangle` (the stick) are singletons in
# every scene, so permutation within those types is the identity and the flag is
# vacuous. An absent entry would default to the same thing; this is explicit so a
# future variant with two sticks or two robots has to make a deliberate choice.
_STICKBUTTON2D_TYPE_AUG_POLICY: dict[str, bool] = {
    "crv_robot": True,
    "rectangle": True,
    "circle": True,
}

# Restock3D (kinematic): the robot is a singleton (augmentation vacuous) and all movables share
# the `Kinematic3DCuboid` type -- cubes and blocks are interchangeable within it because each
# object's geometry (half-extents) is re-read after renumbering, so permutation is a valid symmetry.
# `region` is NON-augmentable: regions carry section-specific roles (a tall-section region and a
# short-section region are not interchangeable), so their ids must not be permuted.
_RESTOCK3D_TYPE_AUG_POLICY: dict[str, bool] = {
    "Kinematic3DRobot": True,
    "Kinematic3DCuboid": True,
    "region": False,
}

_TYPE_AUG_POLICIES: dict[str, dict[str, bool]] = {
    "dd2d_v2": _DD2D_TYPE_AUG_POLICY,
    # dd2d_v3: same domain and policy, re-collected after the 2026-07-24 grasp-model
    # changes (contact-run fix + internal grasps) invalidated v2's labels. A separate
    # variant, not an overwrite, so the stale v2 artifacts stay reproducible and a
    # v2-vs-v3 number can never be mixed by accident.
    "dd2d_v3": _DD2D_TYPE_AUG_POLICY,
    # dd2d_v4: same domain and policy again, re-collected with the v3 refiner
    # instrumentation so failures carry observed culprits / per-step effort /
    # exhausted-vs-budget. The instrumentation is observation-only, but DD2D's problem
    # *generator* is PYTHONHASHSEED-dependent, so v4 is not a byte-identical re-run of
    # v3: 86.9% of problems are fully identical and 0.08% of candidate labels differ.
    # Kept as a separate variant for the same reason as v3 -- so the two can never be
    # silently mixed in one number.
    "dd2d_v4": _DD2D_TYPE_AUG_POLICY,
    # stickbutton2d_v1: the collected dataset. b1/b2/b3/b5 pooled into one variant with
    # button count as the stratum axis (b10 dropped -- structurally infeasible, see
    # docs/autonomous_stickbutton_session.md D5). The per-button-count entries below are
    # the development variants the feasibility work used and stay for reproducing it.
    "stickbutton2d_v1": _STICKBUTTON2D_TYPE_AUG_POLICY,
    # stickbutton2d_v1_kinder: byte-identical records to v1 with kinder-rendered PIGINet
    # crops (SPECTRE is image-free, so its inputs are unchanged). Same augmentation
    # policy -- the tensorizer path is identical; only PIGINet's crop source differs.
    "stickbutton2d_v1_kinder": _STICKBUTTON2D_TYPE_AUG_POLICY,
    # StickButton2D, one variant per button count. See
    # docs/kinder_stickbutton2d_map.md for the substrate map and the measured
    # per-variant feasibility -- b10 does not yield positive labels.
    "stickbutton2d_b1": _STICKBUTTON2D_TYPE_AUG_POLICY,
    "stickbutton2d_b2": _STICKBUTTON2D_TYPE_AUG_POLICY,
    "stickbutton2d_b3": _STICKBUTTON2D_TYPE_AUG_POLICY,
    "stickbutton2d_b5": _STICKBUTTON2D_TYPE_AUG_POLICY,
    "stickbutton2d_b10": _STICKBUTTON2D_TYPE_AUG_POLICY,
    # restock3d_v1: the kinematic single-multi-section-shelf restock env (r0-r3 pooled by
    # problem-id band). One augmentation policy across strata (movables interchangeable, regions
    # section-semantic).
    "restock3d_v1": _RESTOCK3D_TYPE_AUG_POLICY,
    # restock3d_v2_pilot: the continuous-packing v2 SPECTRE pilot (1x1..4x4 configs). Same
    # movable interchangeability; v2 has no region state objects, so the `region` entry is
    # simply unused.
    "restock3d_v2_pilot": _RESTOCK3D_TYPE_AUG_POLICY,
    # restock3d_v2: the full continuous-packing v2 SPECTRE collection (2x2/3x3/3x4/4x3/4x4,
    # 250/75/75). Same policy as the pilot -- v2 has no region state objects.
    "restock3d_v2": _RESTOCK3D_TYPE_AUG_POLICY,
    # restock3d_v3: per-object widths + heights near the cutoff (4 strata). Same movable
    # interchangeability; no region state objects.
    "restock3d_v3": _RESTOCK3D_TYPE_AUG_POLICY,
}


# Per-env static-tag predicate registry for the F3-B-(1) dual-stream Φ_s
# pool. When ``use_static_tag_pool=True`` is set on TrainingConfig, atoms
# whose predicate-name is in this list are routed to a dedicated SAB+PMA
# stream so they don't compete for attention with the larger fluent atom
# population. Empty / missing entries leave the legacy single-pool path.
# No currently-registered environment declares static-tag predicates.
_STATIC_TAG_PREDICATES: dict[str, list[str]] = {}


def get_type_aug_policy(env_variant: str) -> dict[str, bool]:
    """Return the per-type augmentability policy for an env variant.

    Returns an empty dict for envs that have no policy entry; callers treat missing keys
    as ``augmentable=True`` (backwards-compatible with all kinder envs).
    """
    return dict(_TYPE_AUG_POLICIES.get(env_variant, {}))


def get_static_tag_predicates(env_variant: str) -> list[str]:
    """Return the predicate-name list for the static-tag pool stream.

    Empty list if the env has no registry entry; ``_StateTokenEncoder`` treats that as
    "single-pool legacy path". Callers may union with ``vocab.predicates`` to drop names
    the vocab has not seen.
    """
    return list(_STATIC_TAG_PREDICATES.get(env_variant, []))


def cluttered_storage_variants(block_counts: range | list[int]) -> list[ExtraVariant]:
    """Variants of ``ClutteredStorage2D`` parameterized by ``num_blocks``."""
    return [
        ExtraVariant(
            family="ClutteredStorage2D",
            variant_char="b",
            entry_point=_CLUTTERED_STORAGE_ENTRY_POINT,
            kwarg_name="num_blocks",
            kwarg_value=n,
        )
        for n in block_counts
    ]


def cluttered_retrieval_variants(
    obstruction_counts: range | list[int],
) -> list[ExtraVariant]:
    """Variants of ``ClutteredRetrieval2D`` parameterized by ``num_obstructions``."""
    return [
        ExtraVariant(
            family="ClutteredRetrieval2D",
            variant_char="o",
            entry_point=_CLUTTERED_RETRIEVAL_ENTRY_POINT,
            kwarg_name="num_obstructions",
            kwarg_value=n,
        )
        for n in obstruction_counts
    ]


def stick_button_variants(button_counts: range | list[int]) -> list[ExtraVariant]:
    """Variants of ``StickButton2D`` parameterized by ``num_buttons``.

    Kinder pre-registers ``b{1,2,3,5,10}`` natively; use this helper to bootstrap any
    additional counts (e.g. ``b4``, ``b6``, ``b8``) that an experiment may want. For the
    five pre-registered counts this helper is a no-op at registration time —
    ``register_extra_envs`` skips ids already present in ``gymnasium.registry``.
    """
    return [
        ExtraVariant(
            family="StickButton2D",
            variant_char="b",
            entry_point=_STICK_BUTTON_ENTRY_POINT,
            kwarg_name="num_buttons",
            kwarg_value=n,
        )
        for n in button_counts
    ]


def register_restock3d_envs() -> None:
    """Register ``spectre/Restock3D-r{0..3}-v0`` (our kinematic env, not a kinder
    family).

    A ``spectre/`` prefix, so it cannot go through :class:`ExtraVariant` (which forces
    ``kinder/``). One gym id per stratum (fixed object count within a stratum); pooled
    into ``restock3d_v1`` by problem-id banding, exactly as StickButton2D's ``b{N}``.
    """
    for r in range(4):
        gid = f"spectre/Restock3D-r{r}-v0"
        if gid not in gymnasium.registry:
            gymnasium.register(
                id=gid, entry_point=_RESTOCK3D_ENTRY_POINT, kwargs={"stratum": r}
            )
    # v2 continuous-packing pilot: one gym id per pilot stratum (10-13).
    for r in _RESTOCK3D_V2_PILOT_STRATA:
        gid = f"spectre/Restock3Dv2-r{r}-v0"
        if gid not in gymnasium.registry:
            gymnasium.register(
                id=gid, entry_point=_RESTOCK3D_V2_ENTRY_POINT, kwargs={"stratum": r}
            )
    # v3 per-object-dims: one gym id per banding stratum (0-3).
    for r in _RESTOCK3D_V3_STRATA:
        gid = f"spectre/Restock3Dv3-r{r}-v0"
        if gid not in gymnasium.registry:
            gymnasium.register(
                id=gid, entry_point=_RESTOCK3D_V3_ENTRY_POINT, kwargs={"stratum": r}
            )


def register_extra_envs(
    variants: list[ExtraVariant] | None = None,
) -> None:
    """Register kinder variants that are not registered by default.

    Safe to call multiple times. Defaults to ``ClutteredStorage2D-b{1..15}``; also
    always registers the ``spectre/Restock3D-r{0..3}`` ids.
    """
    kinder.register_all_environments()
    register_restock3d_envs()

    if variants is None:
        variants = list(cluttered_storage_variants(range(1, 16)))

    for v in variants:
        if v.gym_id in gymnasium.registry:
            continue
        gymnasium.register(
            id=v.gym_id,
            entry_point=v.entry_point,
            kwargs={v.kwarg_name: v.kwarg_value},
        )
