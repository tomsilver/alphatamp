"""Oracle policy"""

from typing import TypeVar

from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    HeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.structs import Plan, PlanningProblem, SesameModels

# samples trajectories that convert high level to low level
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


class BookshelfPolicy(BaseApproach[_O, _X, _U]):
    """Straightforward: use Sesame with a trivial goal-aware heuristic.
    no learning, just plan to achieve the env's goal."""

    def __init__(
        self,
        env_models: SesameModels,
        seed: int = 123,
        max_abstract_plans: int = 8,  # how many abstract plan candidates to evaluate overall
        max_skill_horizon: int = 80,  # cap for how long a single skill runs
        num_sampling_attempts_per_step: int = 8,  # how many controller samples to try for each operator step
    ):
        super().__init__(env_models, seed)
        self._max_abstract_plans = max_abstract_plans
        self._max_skill_horizon = max_skill_horizon
        self._num_sampling_attempts_per_step = num_sampling_attempts_per_step

    def _train(self, problem: PlanningProblem[_X, _U]) -> None:
        # No training
        return

    # input to _run_planning is problem, which contains the initial concrete state, goal (abstract), and spaces
    def _run_planning(
        self, problem: PlanningProblem[_X, _U], timeout: float
    ) -> Plan[_X, _U]:
        """Assemble the Sesame components and plan."""

        # 1) Given a abstract step, trajectory sampler executes parameterized controllers ("skills"). 'How to Move'
        trajectory_sampler = ParameterizedControllerTrajectorySampler(
            # controllers are skills (place book, pick up book) and parameterizations (different ways to pick it up)
            controller_generator=RelationalControllerGenerator(self._env_models.skills),
            # for each controller, simulate it using transition function
            transition_function=self._env_models.transition_fn,
            # then map concrete back to abstract so goal condition can be checked
            state_abstractor=self._env_models.state_abstractor,
            # when to stop the roll out
            max_trajectory_steps=self._max_skill_horizon,
        )

        # 2) Abstract successor generator over operators (symbolic layer)
        # operators are pick, place, move, etc
        # Output a list of next reachable abstract states by applying any operator
        abstract_successor_fn = RelationalAbstractSuccessorGenerator(
            self._env_models.operators
        )

        # 3) tiny heuristic:
        #    Return 1 if the goal holds (or we start in goal), else 0.
        def _trivial_goal_heuristic_factory(init_s, goal):
            def _h(state):
                return goal.check_abstract_state(state) or goal.check_abstract_state(
                    init_s
                )

            return _h

        # 4) Abstract plan generator
        # Search over abstract state space to find sequence of operators, using heuristic to guide exploration
        # Call the successor function repeatedly to get sequence of operators and use a heuristic to see which branch should be explored first
        # starts from abstract version of inital concrete state

        # Is it like a BFS like prof had mentioned before?
        abstract_plan_generator: AbstractPlanGenerator = (
            HeuristicSearchAbstractPlanGenerator(
                _trivial_goal_heuristic_factory,
                abstract_successor_fn,
                seed=self._seed,
            )
        )

        # This runs a heuristic search over abstract plans, and will be very slow, and

        # 5) Tie it together with Sesame and run
        planner = SesamePlanner(
            abstract_plan_generator=abstract_plan_generator,
            trajectory_sampler=trajectory_sampler,
            state_abstractor=self._env_models.state_abstractor,
            abstract_successor_function=abstract_successor_fn,
            max_abstract_plans=self._max_abstract_plans,
            num_sampling_attempts_per_step=self._num_sampling_attempts_per_step,
            seed=self._seed,
        )
        plan, _ = planner.run(problem, timeout=timeout)
        if plan is None:
            raise TimeoutError("BookshelfPolicy: no plan found within timeout.")
        return plan
