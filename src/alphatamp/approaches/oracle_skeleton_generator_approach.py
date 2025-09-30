"""Uses an oracle skeleton generator policy for abstract planning."""

from typing import Callable, Iterable, Iterator, TypeAlias, TypeVar

from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import (
    Goal,
    Plan,
    PlanningProblem,
    RelationalAbstractState,
    SesameModels,
)
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
)
from relational_structs import GroundOperator

from alphatamp.approaches.base_approach import BaseApproach

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
_S = TypeVar("_S")  # abstract state
_A = TypeVar("_A")  # abstract action
Skeleton: TypeAlias = tuple[list[RelationalAbstractState], list[GroundOperator]]
FrozenSkeleton: TypeAlias = tuple[
    tuple[RelationalAbstractState, ...], tuple[GroundOperator, ...]
]


class OracleAbstractPlanGenerator(AbstractPlanGenerator[_X, _S, _A]):
    """A generator that uses oracle knowledge to generate abstract plans."""

    def __init__(
        self,
        env_models: SesameModels,
        seed: int,
    ) -> None:
        # TODO
        # super().__init__(abstract_successor_function, seed)
        self._env_models = env_models

    def __call__(
        self,
        x0: _X,
        s0: _S,
        goal: Goal,
        timeout: float,
        bpg: BilevelPlanningGraph[_X, _U, _S, _A],
    ) -> Iterator[tuple[list[_S], list[_A]]]:
        """Generate abstract plans."""

        # Return the plan that picks and places block0, then picks and places block1,
        # then picks and places block2.
        operator_name_to_operator = {
            s.operator.name: s.operator for s in self._env_models.skills
        }

        # Lifted operators.
        PickBlockOnShelf = operator_name_to_operator["PickBlockOnShelf"]
        PlaceBlockOnShelf = operator_name_to_operator["PlaceBlockOnShelf"]
        PlaceBlockNotOnShelf = operator_name_to_operator["PlaceBlockNotOnShelf"]
        PickBlockNotOnShelf = operator_name_to_operator["PickBlockNotOnShelf"]

        # Objects.
        type_name_to_type = {t.name: t for t in self._env_models.types}
        block_type = type_name_to_type["target_block"]
        shelf_type = type_name_to_type["shelf"]
        robot_type = type_name_to_type["crv_robot"]
        block0, block1, block2 = sorted(x0.get_objects(block_type))
        # block0, = sorted(x0.get_objects(block_type))
        (robot,) = x0.get_objects(robot_type)
        (shelf,) = x0.get_objects(shelf_type)

        # Make abstract plan.
        abstract_actions = [
            PickBlockOnShelf.ground((robot, block0, shelf)),
            PlaceBlockNotOnShelf.ground((robot, block0, shelf)),
            PickBlockNotOnShelf.ground((robot, block0, shelf)),
            PlaceBlockOnShelf.ground((robot, block0, shelf)),
            PickBlockNotOnShelf.ground((robot, block1, shelf)),
            PlaceBlockOnShelf.ground((robot, block1, shelf)),
            PickBlockNotOnShelf.ground((robot, block2, shelf)),
            PlaceBlockOnShelf.ground((robot, block2, shelf)),
        ]

        # "Simulate" the execution of the abstract actions to get the abstract states.
        abstract_states = [s0]
        for abstract_action in abstract_actions:
            next_atoms = (
                abstract_states[-1].atoms - abstract_action.delete_effects
            ) | abstract_action.add_effects
            next_state = RelationalAbstractState(next_atoms, s0.objects)
            abstract_states.append(next_state)

        print()
        print("Abstract state:", sorted(abstract_states[0].atoms))
        for abstract_state, abstract_action in zip(
            abstract_states[1:], abstract_actions, strict=True
        ):
            print("Abstract action:", abstract_action.short_str)
            print("Abstract state:", sorted(abstract_state.atoms))

        print("Goal:", goal)

        return iter([(abstract_states, abstract_actions)])


class OracleSkeletonGeneratorApproach(BaseApproach[_O, _X, _U]):
    """Uses an oracle skeleton generator policy for abstract planning."""

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
        self._abstract_plan_generator: AbstractPlanGenerator = (
            OracleAbstractPlanGenerator(
                self._env_models,
                seed=self._seed,
            )
        )

        # Create the abstract successor function (not really used).
        self._abstract_successor_fn = RelationalAbstractSuccessorGenerator(
            self._env_models.operators
        )

        # Finish the planner.
        self._planner = SesamePlanner(
            self._abstract_plan_generator,
            self._trajectory_sampler,
            self._max_abstract_plans,
            self._samples_per_step,
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
