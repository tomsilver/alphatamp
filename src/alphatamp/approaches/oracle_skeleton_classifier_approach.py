"""Uses an oracle skeleton generator policy for abstract planning."""

import time
from typing import Hashable, TypeAlias, TypeVar

from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
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
_S = TypeVar("_S", bound=Hashable)  # abstract state
_A = TypeVar("_A", bound=Hashable)  # abstract action
Skeleton: TypeAlias = tuple[list[RelationalAbstractState], list[GroundOperator]]
FrozenSkeleton: TypeAlias = tuple[
    tuple[RelationalAbstractState, ...], tuple[GroundOperator, ...]
]

from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.bilevel_planner import BilevelPlanner
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.refiners.backtracking_refiner import BacktrackingRefiner
from bilevel_planning.refiners.refiner import Refiner
from bilevel_planning.structs import Plan, PlanningProblem
from bilevel_planning.trajectory_samplers.trajectory_sampler import (
    TrajectorySampler,
)
from prbench.envs.geom2d.object_types import CRVRobotType, RectangleType
from relational_structs import (
    GroundAtom,
    GroundOperator,
    Predicate,
)


class OracleAbstractPlanClassifier:
    """A classifier that uses oracle knowledge to classify abstract plans."""

    def __init__(
        self,
        env_models: SesameModels,
    ) -> None:
        # TODO
        # super().__init__(abstract_successor_function, seed)
        self._env_models = env_models

    def validate_plan(
        self,
        x0: _X,
        s_plan: list[_S],
        a_plan: list[_A],
    ) -> bool:
        """Classify abstract plans."""

        # Predicates.
        HoldingTgt = Predicate("HoldingTgt", [CRVRobotType, RectangleType])
        HandEmpty = Predicate("HandEmpty", [CRVRobotType])

        # Objects.
        type_name_to_type = {t.name: t for t in self._env_models.types}
        block_type = type_name_to_type["target_block"]
        obstruction_type = type_name_to_type["rectangle"]
        robot_type = type_name_to_type["crv_robot"]

        (target_block,) = x0.get_objects(block_type)
        (robot,) = x0.get_objects(robot_type)
        obstructions = x0.get_objects(obstruction_type)

        # remove target block from obstructions
        filtered_obstructions = set()
        for obstruction in obstructions:
            if obstruction.name != "target_block":
                filtered_obstructions.add(obstruction)

        # Oracle abstract plan
        empty_abstract_state_atoms: set[GroundAtom] = set()
        empty_abstract_state_atoms.add(GroundAtom(HandEmpty, [robot]))
        empty_abstract_state_objects = {robot, target_block} | filtered_obstructions
        empty_abstract_state = RelationalAbstractState(
            atoms=empty_abstract_state_atoms, objects=empty_abstract_state_objects
        )

        holding_abstract_state_atoms = set()
        holding_abstract_state_atoms.add(GroundAtom(HoldingTgt, [robot, target_block]))
        holding_abstract_state_objects = {robot, target_block} | filtered_obstructions
        holding_abstract_state = RelationalAbstractState(
            atoms=holding_abstract_state_atoms, objects=holding_abstract_state_objects
        )

        oracle_abstract_state = [empty_abstract_state, holding_abstract_state]

        oracle_abstract_state_ptr = 0
        # Classify plan only looking at state for now
        for plan_abstract_state in s_plan:
            plan_atoms, plan_objects = (
                plan_abstract_state.atoms,
                plan_abstract_state.objects,
            )
            oracle_atoms, oracle_objects = (
                oracle_abstract_state[oracle_abstract_state_ptr].atoms,
                oracle_abstract_state[oracle_abstract_state_ptr].objects,
            )

            # import ipdb; ipdb.set_trace()
            if plan_atoms != oracle_atoms or plan_objects != oracle_objects:
                return False
            oracle_abstract_state_ptr += 1

        return True


"""Multi-abstract plan + backtracking refinement planner."""


class SesamePlannerWithClassifier(BilevelPlanner[_X, _U, _S, _A]):
    """Multi-abstract plan + backtracking refinement planner that classifier that
    determines if a plan is feasible or not."""

    def __init__(
        self,
        abstract_plan_generator: AbstractPlanGenerator[_X, _S, _A],
        trajectory_sampler: TrajectorySampler[_X, _U, _S, _A],
        max_abstract_plans: int,
        num_sampling_attempts_per_step: int,
        env_model: SesameModels,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._abstract_plan_generator = abstract_plan_generator
        self._trajectory_sampler = trajectory_sampler
        self._max_abstract_plans = max_abstract_plans
        self._refiner: Refiner[_X, _U, _S, _A] = BacktrackingRefiner(
            self._trajectory_sampler, num_sampling_attempts_per_step, seed=self._seed
        )
        self._abstract_plan_classifier = OracleAbstractPlanClassifier(
            env_models=env_model
        )

    def run(
        self, problem: PlanningProblem[_X, _U], timeout: float
    ) -> tuple[Plan | None, BilevelPlanningGraph]:
        start_time = time.perf_counter()

        # Get the initial abstract state.
        x0 = problem.initial_state
        s0 = self._state_abstractor(x0)

        # Initialize the bilevel planning graph.
        bpg: BilevelPlanningGraph[_X, _U, _S, _A] = BilevelPlanningGraph()
        bpg.add_state_node(x0)
        bpg.add_abstract_state_node(s0)
        bpg.add_state_abstractor_edge(x0, s0)

        # Generate abstract plans and attempt to refine them.
        gen = self._abstract_plan_generator(
            x0,
            s0,
            problem.goal,
            timeout,
            bpg,
        )
        num_abstract_plans = 0

        while (
            num_abstract_plans < self._max_abstract_plans
            and time.perf_counter() - start_time < timeout
        ):
            # Get the next abstract plan.
            try:
                s_plan, a_plan = next(gen)
                num_abstract_plans += 1
            except StopIteration:
                break
            # Quit early if timeout.
            remaining_time = timeout - (time.perf_counter() - start_time)
            if remaining_time < 0:
                break

            # Try to classify whether or not this abstract plan is valid
            if self._abstract_plan_classifier.validate_plan(x0, s_plan, a_plan):
                # Try to refine this abstract plan.
                plan = self._refiner(x0, s_plan, a_plan, remaining_time, bpg)
                # Plan successfully found.
                if plan is not None:
                    return plan, bpg

        return None, bpg


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
