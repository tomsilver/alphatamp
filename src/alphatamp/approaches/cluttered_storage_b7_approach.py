"""Oracle approach specifically for ClutteredStorage2D-b7-v0."""

from typing import Iterable, Iterator, TypeAlias, TypeVar

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
from relational_structs import GroundOperator, ObjectCentricState

from alphatamp.approaches.base_approach import BaseApproach

_O = TypeVar("_O")  # observation
_U = TypeVar("_U")  # action
_X = TypeVar("_X", bound=ObjectCentricState)  # state
_S = TypeVar("_S", bound=RelationalAbstractState)  # abstract state
_A = TypeVar("_A", bound=GroundOperator)  # abstract action
Skeleton: TypeAlias = tuple[list[_S], list[_A]]
FrozenSkeleton: TypeAlias = tuple[tuple[_S, ...], tuple[_A, ...]]


def noop_successor_fn(_s: _S) -> Iterable[tuple[_A, _S]]:
    """Return no successors; placeholder to satisfy AbstractPlanGenerator.__init__."""
    return []


class b7OracleAbstractPlanGenerator(
    AbstractPlanGenerator[_X, RelationalAbstractState, GroundOperator]
):
    """A generator that uses oracle knowledge to generate abstract plans."""

    def __init__(
        self,
        env_models: SesameModels,
        seed: int,
    ) -> None:
        """Initialize with env models and seed."""
        super().__init__(noop_successor_fn, seed)
        self._env_models = env_models

    def __call__(
        self,
        x0: _X,
        s0: RelationalAbstractState,
        goal: Goal,
        timeout: float,
        bpg: BilevelPlanningGraph[_X, _U, RelationalAbstractState, GroundOperator],
    ) -> Iterator[tuple[list[RelationalAbstractState], list[GroundOperator]]]:
        """Generate abstract plans."""

        # Look at environment models to map operators to their names
        operator_name_to_operator = {
            s.operator.name: s.operator for s in self._env_models.skills
        }

        # Lifted operators that we will use for ClutteredStorage2D-b3-v0
        PickBlockOnShelf = operator_name_to_operator["PickBlockOnShelf"]
        PlaceBlockOnShelf = operator_name_to_operator["PlaceBlockOnShelf"]
        PlaceBlockNotOnShelf = operator_name_to_operator["PlaceBlockNotOnShelf"]
        PickBlockNotOnShelf = operator_name_to_operator["PickBlockNotOnShelf"]

        # Map object types to their names
        type_name_to_type = {t.name: t for t in self._env_models.types}

        # define 3 object types for ClutteredStorage2D-b3-v0
        block_type = type_name_to_type["target_block"]
        shelf_type = type_name_to_type["shelf"]
        robot_type = type_name_to_type["crv_robot"]

        # Get the objects from the initial state, sorting them so they
        # match their semantic names
        block0, block1, block2, block3, block4, block5, block6 = sorted(
            x0.get_objects(block_type)
        )
        (robot,) = x0.get_objects(robot_type)
        (shelf,) = x0.get_objects(shelf_type)

        # Creates the abstract plan by grounding the lifeted operators
        abstract_actions: list[GroundOperator] = [
            PickBlockOnShelf.ground((robot, block0, shelf)),
            PlaceBlockNotOnShelf.ground((robot, block0, shelf)),
            PickBlockNotOnShelf.ground((robot, block0, shelf)),
            PlaceBlockOnShelf.ground((robot, block0, shelf)),
            PickBlockOnShelf.ground((robot, block1, shelf)),
            PlaceBlockNotOnShelf.ground((robot, block1, shelf)),
            PickBlockNotOnShelf.ground((robot, block1, shelf)),
            PlaceBlockOnShelf.ground((robot, block1, shelf)),
            PickBlockOnShelf.ground((robot, block2, shelf)),
            PlaceBlockNotOnShelf.ground((robot, block2, shelf)),
            PickBlockNotOnShelf.ground((robot, block2, shelf)),
            PlaceBlockOnShelf.ground((robot, block2, shelf)),
            PickBlockNotOnShelf.ground((robot, block3, shelf)),
            PlaceBlockOnShelf.ground((robot, block3, shelf)),
            PickBlockNotOnShelf.ground((robot, block4, shelf)),
            PlaceBlockOnShelf.ground((robot, block4, shelf)),
            PickBlockNotOnShelf.ground((robot, block5, shelf)),
            PlaceBlockOnShelf.ground((robot, block5, shelf)),
            PickBlockNotOnShelf.ground((robot, block6, shelf)),
            PlaceBlockOnShelf.ground((robot, block6, shelf)),
        ]

        # "Simulate" the execution of the abstract actions to get the abstract states.
        # Starting from the initial abstract state s0, apply delete and
        # add effects of each action to the current set of atoms
        # to produce the next abstract state, and add them to abstract_states

        abstract_states: list[RelationalAbstractState] = [s0]
        for abstract_action in abstract_actions:
            next_atoms = (
                abstract_states[-1].atoms - abstract_action.delete_effects
            ) | abstract_action.add_effects
            next_state = RelationalAbstractState(next_atoms, s0.objects)
            abstract_states.append(next_state)

        return iter([(abstract_states, abstract_actions)])


class b7OracleGeneratorApproach(BaseApproach[_O, _X, _U]):
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
            b7OracleAbstractPlanGenerator(
                self._env_models,
                seed=self._seed,
            )
        )

        # Create the abstract successor function (not really used).
        self._abstract_successor_fn = RelationalAbstractSuccessorGenerator(
            self._env_models.operators
        )

        # Finish the planner.
        # Sesame planner uses operators to check symbolic feasbility and calls skills
        # using the sampler to attempt low-level roll-outs
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
