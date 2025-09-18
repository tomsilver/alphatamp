"""A baseline approach that runs pure planning and does not learn anything."""

from typing import TypeVar

from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    HeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.structs import Plan, PlanningProblem, SesameModels
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


class NonRelationalApproach(BaseApproach[_O, _X, _U]):
    """A baseline approach that runs pure planning and does not learn anything without
    using relational information."""

    def __init__(
        self,
        env_models: SesameModels,
        seed: int,
        max_abstract_plans: int = 10,
        samples_per_step: int = 10,
        max_skill_horizon: int = 100,
        heuristic_name: str = "hff",
    ):
        super().__init__(env_models, seed)
        self._max_abstract_plans = max_abstract_plans
        self._samples_per_step = samples_per_step
        self._max_skill_horizon = max_skill_horizon
        self._heuristic_name = heuristic_name

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

        # Create the abstract successor function (not really used).
        abstract_successor_fn = RelationalAbstractSuccessorGenerator(
            self._env_models.operators
        )

        # Create a trivial heuristic factory
        # Returns true (1) if the current abstract state is the goal state
        # False (0) otherwise
        def _trivial_heuristic_factory(init_s, goal):
            def _h(state):
                return goal.check_abstract_state(state) or goal.check_abstract_state(
                    init_s
                )

            return _h

        # Create the abstract plan generator. HeuristicSearchAbstractPlanGenerator
        abstract_plan_generator: AbstractPlanGenerator = (
            HeuristicSearchAbstractPlanGenerator(
                _trivial_heuristic_factory,
                abstract_successor_fn,
                seed=self._seed,
            )
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
