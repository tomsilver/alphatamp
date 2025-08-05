"""Base class for AlphaTAMP approaches."""

import abc

import numpy as np
from bilevel_planning.structs import Plan, PlanningProblem, SesameModels


class BaseApproach(abc.ABC):
    """Base class for AlphaTAMP approaches."""

    def __init__(
        self,
        env_models: SesameModels,
        seed: int,
    ) -> None:
        self._env_models = env_models
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def train(self, problem: PlanningProblem) -> None:
        """Run training on the given training problem.

        Note that train() is called sequentially, one training problem at a time. If the
        approach uses batch training over multiple problems, it should store problems as
        they are received and then call batch training at some interval.
        """

    @abc.abstractmethod
    def run_planning(self, problem: PlanningProblem, timeout: float) -> Plan:
        """Run planning on a given (eval) problem.

        If no plan is found within the timeout, raises a TimeoutError.

        This should produce the best plan possible as quickly as possible. Some
        approaches may want to run other "slower" kinds of planning during training;
        that should happen in a different approach-specific method.
        """
