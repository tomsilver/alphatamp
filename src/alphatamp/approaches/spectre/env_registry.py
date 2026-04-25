"""Bootstrap kinder environment ids that are not in the default registrations.

``kinder.register_all_environments()`` only registers a fixed set of variants
(see ``kinder/__init__.py``). SPECTRE wants additional block/obstruction counts
for dataset-generation convenience. Registration is idempotent: calling with
ids that already exist is a no-op.
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


# Per-type augmentation policy tables for SPECTRE Φ training-time
# object-renumbering augmentation (spec §4.6 / §10.1). Missing keys default to
# ``augmentable=True`` so kinder vocabs round-trip unchanged.
#
# RT2D needs width_level / size_level / passage subtypes pinned to deterministic
# local ids: width and size are totally ordered, and the static Connects
# topology over zones / passages is fixed across all problems. Permuting any of
# those types destroys the relational signal RT2D is engineered to test.
_RT2D_TYPE_AUG_POLICY: dict[str, bool] = {
    "robot": True,
    "item": True,
    "zone": False,
    "passage": False,
    "passage_color_a": False,
    "passage_color_b": False,
    "passage_color_c": False,
    "width_level": False,
    "size_level": False,
}

_TYPE_AUG_POLICIES: dict[str, dict[str, bool]] = {
    "routedtransport2d_n2_v1": _RT2D_TYPE_AUG_POLICY,
    "routedtransport2d_n3_v1": _RT2D_TYPE_AUG_POLICY,
    "routedtransport2d_n4_v1": _RT2D_TYPE_AUG_POLICY,
}


# Per-env static-tag predicate registry for the F3-B-(1) dual-stream Φ_s
# pool. When ``use_static_tag_pool=True`` is set on TrainingConfig, atoms
# whose predicate-name is in this list are routed to a dedicated SAB+PMA
# stream so they don't compete for attention with the larger fluent atom
# population. Empty / missing entries leave the legacy single-pool path.
#
# RT2D rationale: PassageWidth (9 atoms) and ItemSize (3 atoms) are the
# load-bearing static tags whose values determine refinement-time
# feasibility. Connects (18 atoms) is also static (K₃,₃ topology, fixed
# across all problems) and is included so the static stream sees the
# whole "shape of the world" not just the tag values.
_RT2D_STATIC_TAG_PREDICATES: list[str] = [
    "PassageWidth",
    "ItemSize",
    "Connects",
]

_STATIC_TAG_PREDICATES: dict[str, list[str]] = {
    "routedtransport2d_n2_v1": _RT2D_STATIC_TAG_PREDICATES,
    "routedtransport2d_n3_v1": _RT2D_STATIC_TAG_PREDICATES,
    "routedtransport2d_n4_v1": _RT2D_STATIC_TAG_PREDICATES,
}


def get_type_aug_policy(env_variant: str) -> dict[str, bool]:
    """Return the per-type augmentability policy for an env variant.

    Returns an empty dict for envs that have no policy entry; callers treat
    missing keys as ``augmentable=True`` (backwards-compatible with all
    kinder envs).
    """
    return dict(_TYPE_AUG_POLICIES.get(env_variant, {}))


def get_static_tag_predicates(env_variant: str) -> list[str]:
    """Return the predicate-name list for the static-tag pool stream.

    Empty list if the env has no registry entry; ``_StateTokenEncoder``
    treats that as "single-pool legacy path". Callers may union with
    ``vocab.predicates`` to drop names the vocab has not seen.
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

    Kinder pre-registers ``b{1,2,3,5,10}`` natively; use this helper to
    bootstrap any additional counts (e.g. ``b4``, ``b6``, ``b8``) that an
    experiment may want. For the five pre-registered counts this helper is
    a no-op at registration time — ``register_extra_envs`` skips ids already
    present in ``gymnasium.registry``.
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


def register_extra_envs(
    variants: list[ExtraVariant] | None = None,
) -> None:
    """Register kinder variants that are not registered by default.

    Safe to call multiple times. Defaults to ``ClutteredStorage2D-b{1..15}``
    plus ``RoutedTransport2D-n{2,3,4}``. RoutedTransport2D is wired through
    the same registry but lives under
    ``alphatamp.approaches.spectre.envs.routedtransport2d`` (the import is
    deferred so consumers that never collect RT2D don't pay the import cost).
    """
    kinder.register_all_environments()

    if variants is None:
        # Local import — keeps RT2D module out of import-time hot path for
        # callers who only touch kinder envs.
        # pylint: disable=import-outside-toplevel
        from alphatamp.approaches.spectre.envs.routedtransport2d.gym_env import (
            routed_transport_variants,
        )

        variants = list(cluttered_storage_variants(range(1, 16)))
        variants.extend(routed_transport_variants([2, 3, 4]))

    for v in variants:
        if v.gym_id in gymnasium.registry:
            continue
        gymnasium.register(
            id=v.gym_id,
            entry_point=v.entry_point,
            kwargs={v.kwarg_name: v.kwarg_value},
        )
