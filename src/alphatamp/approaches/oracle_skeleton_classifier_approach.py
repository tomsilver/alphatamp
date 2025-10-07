"""Uses an oracle skeleton generator policy for abstract planning."""

from typing import TypeVar

from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.structs import (
    Plan,
    PlanningProblem,
    SesameModels,
)
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
)

from alphatamp.approaches.base_approach import BaseApproach
from alphatamp.approaches.feasibility_classifiers.sesame_planner_with_classifier import (
    SesamePlannerWithClassifier,
)

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action


class OracleSkeletonClassifierApproach(BaseApproach[_O, _X, _U]):
    """Uses an oracle skeleton classifier policy for abstract planning."""

    def __init__(
        self,
        env_models: SesameModels,
        seed: int,
        max_abstract_plans: int = 10,
        samples_per_step: int = 10,
        max_skill_horizon: int = 100,
        heuristic_name: str = "hff",
        skeleton_batch_size: int = 100,
        num_training_skeletons_per_problem: int = 10,
        training_planning_timeout: float = 5,
    ):
        super().__init__(env_models, seed)
        self._max_abstract_plans = max_abstract_plans
        self._samples_per_step = samples_per_step
        self._max_skill_horizon = max_skill_horizon
        self._heuristic_name = heuristic_name
        self._skeleton_batch_size = skeleton_batch_size
        self._num_training_skeletons_per_problem = num_training_skeletons_per_problem
        self._training_planning_timeout = training_planning_timeout

        # Create the planning components.

        # Create the sampler.
        self._trajectory_sampler = ParameterizedControllerTrajectorySampler(
            controller_generator=RelationalControllerGenerator(self._env_models.skills),
            transition_function=self._env_models.transition_fn,
            state_abstractor=self._env_models.state_abstractor,
            max_trajectory_steps=self._max_skill_horizon,
        )

        # Create the abstract plan generator.
        self._base_abstract_plan_generator: AbstractPlanGenerator = (
            RelationalHeuristicSearchAbstractPlanGenerator(
                self._env_models.types,
                self._env_models.predicates,
                self._env_models.operators,
                self._heuristic_name,
                seed=self._seed,
            )
        )

        # Create the abstract successor function (not really used).
        self._abstract_successor_fn = RelationalAbstractSuccessorGenerator(
            self._env_models.operators
        )

        # Finish the planner.
        self._planner = SesamePlannerWithClassifier(
            self._base_abstract_plan_generator,
            self._trajectory_sampler,
            self._max_abstract_plans,
            self._samples_per_step,
            self._env_models,
            self._abstract_successor_fn,
            self._env_models.state_abstractor,
            seed=self._seed,
        )

    def _train(self, problem: PlanningProblem[_X, _U]) -> None:
        pass

    def _run_planning(
        self, problem: PlanningProblem[_X, _U], timeout: float
    ) -> Plan[_X, _U]:

        # Run the planner.
        plan, _ = self._planner.run(problem, timeout=timeout)
        if plan is None:
            raise TimeoutError("No plan found")

        return plan
