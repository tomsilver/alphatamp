"""Tiny adapter for run_experiments.py"""

from __future__ import annotations

from dataclasses import dataclass

import prbench
from prbench_bilevel_planning.env_models import create_bilevel_planning_models


@dataclass
class ClutteredStorageBenchmark:
    """Hydra-instantiable benchmark for ClutteredStorage2D environment."""

    num_blocks: int = 3
    name: str = "cluttered_storage"

    def make_env_and_models(self, seed: int):
        """Create env, model set, and initial observation."""
        prbench.register_all_environments()
        env_id = f"prbench/ClutteredStorage2D-b{self.num_blocks}-v0"
        env = prbench.make(env_id)
        env.reset(seed=seed)

        env_models = create_bilevel_planning_models(
            "clutteredstorage2d",
            env.observation_space,
            env.action_space,
            num_blocks=int(self.num_blocks),
        )
        obs, _ = env.reset(seed=seed)
        return env, env_models, obs

    def check_success(self, env, plan) -> tuple[bool, int]:
        """Step the plan; return (success, num_actions)."""
        num = 0
        for act in plan.actions:
            _, _, done, _, _ = env.step(act)
            num += 1
            if done:
                return True, num
        return False, num
