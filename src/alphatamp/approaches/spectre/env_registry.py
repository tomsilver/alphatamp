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


def register_extra_envs(
    variants: list[ExtraVariant] | None = None,
) -> None:
    """Register kinder variants that are not registered by default.

    Safe to call multiple times. Defaults to ``ClutteredStorage2D-b{1..15}``.
    """
    kinder.register_all_environments()

    if variants is None:
        variants = cluttered_storage_variants(range(1, 16))

    for v in variants:
        if v.gym_id in gymnasium.registry:
            continue
        gymnasium.register(
            id=v.gym_id,
            entry_point=v.entry_point,
            kwargs={v.kwarg_name: v.kwarg_value},
        )
