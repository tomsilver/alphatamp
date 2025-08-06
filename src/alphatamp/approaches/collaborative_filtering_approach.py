"""A learning approach for skeleton generation inspired by collaborative filtering."""

from itertools import islice
from typing import Callable, Hashable, Iterator, TypeVar

from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import Goal, Plan, PlanningProblem, SesameModels
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
)

from alphatamp.approaches.base_approach import BaseApproach

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
_S = TypeVar("_S", bound=Hashable)  # abstract state
_A = TypeVar("_A", bound=Hashable)  # abstract action


class CollaborativeFilteringApproach(BaseApproach[_O, _X, _U]):
    """A learning approach for skeleton gen inspired by collaborative filtering."""

    def __init__(
        self,
        env_models: SesameModels,
        seed: int,
        max_abstract_plans: int = 10,
        samples_per_step: int = 10,
        max_skill_horizon: int = 100,
        heuristic_name: str = "hff",
        skeleton_batch_size: int = 100,
    ):
        super().__init__(env_models, seed)
        self._max_abstract_plans = max_abstract_plans
        self._samples_per_step = samples_per_step
        self._max_skill_horizon = max_skill_horizon
        self._heuristic_name = heuristic_name
        self._skeleton_batch_size = skeleton_batch_size

    def _train(self, problem: PlanningProblem[_X, _U]) -> None:
        pass

    def _run_planning(
        self, problem: PlanningProblem[_X, _U], timeout: float
    ) -> Plan[_X, _U]:

        # Create the sampler.
        trajectory_sampler = ParameterizedControllerTrajectorySampler(
            controller_generator=RelationalControllerGenerator(self._env_models.skills),
            transition_function=self._env_models.transition_fn,
            state_abstractor=self._env_models.state_abstractor,
            max_trajectory_steps=self._max_skill_horizon,
        )

        # Create the abstract plan generator.
        abstract_plan_generator: AbstractPlanGenerator = (
            BatchRankingAbstractPlanGenerator(
                RelationalHeuristicSearchAbstractPlanGenerator(
                    self._env_models.types,
                    self._env_models.predicates,
                    self._env_models.operators,
                    self._heuristic_name,
                    seed=self._seed,
                ),
                score_fn=self._score_skeleton,
                batch_size=self._skeleton_batch_size,
                seed=self._seed,
            )
        )

        # Create the abstract successor function (not really used).
        abstract_successor_fn = RelationalAbstractSuccessorGenerator(
            self._env_models.operators
        )

        # Finish the planner.
        planner = SesamePlanner(
            abstract_plan_generator,
            trajectory_sampler,
            self._max_abstract_plans,
            self._samples_per_step,
            abstract_successor_fn,
            self._env_models.state_abstractor,
            seed=self._seed,
        )

        # Run the planner.
        plan, _ = planner.run(problem, timeout=timeout)
        if plan is None:
            raise TimeoutError("No plan found")

        return plan

    def _score_skeleton(self, skeleton: tuple[list[_S], list[_A]]) -> float:
        """Score skeletons.

        Higher is better.
        """
        # TODO
        return -len(skeleton[1])


class BatchRankingAbstractPlanGenerator(AbstractPlanGenerator[_X, _S, _A]):
    """Generates batches of abstract plans and then ranks them using a score function,
    where higher scores are considered better."""

    def __init__(
        self,
        base_generator: AbstractPlanGenerator[_X, _S, _A],
        score_fn: Callable[[tuple[list[_S], list[_A]]], float],
        batch_size: int,
        seed: int,
    ) -> None:
        self._base_generator = base_generator
        self._score_fn = score_fn
        self._batch_size = batch_size
        # In the future, make this public or find another workaround.
        abstract_successor_fn = (
            self._base_generator._abstract_successor_function
        )  # pylint: disable=protected-access
        super().__init__(abstract_successor_fn, seed)

    def __call__(
        self,
        x0: _X,
        s0: _S,
        goal: Goal,
        timeout: float,
        bpg: BilevelPlanningGraph[_X, _U, _S, _A],
    ) -> Iterator[tuple[list[_S], list[_A]]]:
        iterator = self._base_generator(x0, s0, goal, timeout, bpg)
        tiebreaking_score_fn = lambda x: (self._score_fn(x), self._rng.uniform())
        while batch := list(islice(iterator, self._batch_size)):
            for skeleton in sorted(batch, key=tiebreaking_score_fn, reverse=True):

                # TODO remove
                print("YIELDING")
                for a in skeleton[1]:
                    print(a.short_str)
                print()

                yield skeleton
