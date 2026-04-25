"""Stub gym env for RoutedTransport2D.

Per the agreed integration design (env_registry-style + local env_models
factory), RT2D registers a gymnasium env that does not actually simulate
anything: ``reset(seed=N)`` builds a :class:`ProblemInstance` from seed N and
returns it inside the observation dict. Downstream
``env_models.observation_to_state`` (in :mod:`env_models`) unwraps it.

``step`` raises — RT2D's collection path bypasses gym stepping entirely
(``ThreeGateRefiner`` is closed-form). The env exists only so
``kinder.make(cfg.env_id)`` succeeds in the existing
:func:`alphatamp.approaches.spectre.collect.collect_episode` plumbing.
"""

from __future__ import annotations

from typing import Any, Final, Optional

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import Discrete

from alphatamp.approaches.spectre.env_registry import ExtraVariant
from alphatamp.approaches.spectre.envs.routedtransport2d.problem_generator import (
    ProblemInstance,
    make_problem,
)

_ENTRY_POINT: Final[str] = (
    "alphatamp.approaches.spectre.envs.routedtransport2d.gym_env:RoutedTransport2DEnv"
)


class RoutedTransport2DEnv(gym.Env):  # pylint: disable=abstract-method
    """Stub gym env carrying a ProblemInstance through reset()."""

    metadata: dict[str, list[str]] = {"render_modes": []}

    def __init__(self, num_items: int = 3, variant: str = "v1") -> None:
        super().__init__()
        self.num_items = num_items
        self.variant = variant
        # Trivial obs/action spaces. The live ProblemInstance ships through
        # ``info`` (free-form, not type-checked by gym) and the collector merges
        # it back into ``obs`` for the dispatcher; that keeps the obs space
        # consistent with what reset() actually returns and avoids triggering
        # gym's passive env checker.
        self.observation_space = DictSpace(
            {
                "problem_seed": Discrete(2**31),
                # Tiny non-degenerate Box keeps gym's passive env checker quiet.
                "_placeholder": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            }
        )
        self.action_space = Discrete(1)  # unused
        self._cached_problem: Optional[ProblemInstance] = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del options
        if seed is None:
            raise ValueError(
                "RoutedTransport2D requires an explicit seed; collect.py passes"
                " problem_id as seed."
            )
        variant_str = f"n{self.num_items}-{self.variant}"
        problem = make_problem(seed, variant=variant_str)
        self._cached_problem = problem
        observation: dict[str, Any] = {
            "problem_seed": seed,
            "_placeholder": np.zeros(1, dtype=np.float32),
        }
        info: dict[str, Any] = {"_problem": problem}
        return observation, info

    def step(self, action: object) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        del action
        raise NotImplementedError(
            "RoutedTransport2D does not support gym stepping — refinement is"
            " closed-form (see ThreeGateRefiner)."
        )

    def close(self) -> None:
        self._cached_problem = None


def routed_transport_variants(
    num_items_list: range | list[int],
) -> list[ExtraVariant]:
    """ExtraVariant rows for ``register_extra_envs`` (env_registry).

    Produces gym ids of the form ``kinder/RoutedTransport2D-n<N>-v0``. The
    SPECTRE-spec version (``v1``) is independent of gymnasium's ``-v0``
    suffix; the env constructor accepts ``variant="v1"`` separately, but
    only the ``num_items`` kwarg is plumbed through ExtraVariant.
    """
    return [
        ExtraVariant(
            family="RoutedTransport2D",
            variant_char="n",
            entry_point=_ENTRY_POINT,
            kwarg_name="num_items",
            kwarg_value=n,
        )
        for n in num_items_list
    ]
