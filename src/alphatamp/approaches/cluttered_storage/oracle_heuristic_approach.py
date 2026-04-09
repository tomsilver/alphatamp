"""
Approach that loads a heuristic from alphatamp/generated_heuristic.py and runs the planner.
No LLM calls, no candidate selection.
"""

import importlib.util
from pathlib import Path
from typing import (
    Any,
    Callable,
    Iterator,
    TypeAlias,
    TypeVar,
)

from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.structs import (
    Goal,
    Plan,
    PlanningProblem,
    RefinementMetrics,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
    cached_all_ground_operators,
    create_pyperplan_heuristic_from_fn,
)
from pyperplan.heuristics.heuristic_base import Heuristic as PyperplanHeuristic
from relational_structs import (
    GroundOperator,
    LiftedOperator,
    ObjectCentricState,
    PDDLProblem,
    Predicate,
    Type,
)

from alphatamp.approaches.base_approach import BaseApproach

_O = TypeVar("_O")
_U = TypeVar("_U")
_X = TypeVar("_X", bound=ObjectCentricState)
_S = TypeVar("_S", bound=RelationalAbstractState)
_A = TypeVar("_A", bound=GroundOperator)
Skeleton: TypeAlias = tuple[list[_S], list[_A]]

_HEURISTIC_PATH = Path(__file__).parents[4] / "generated_heuristic.py"


class HeuristicGenerator(RelationalHeuristicSearchAbstractPlanGenerator):
    """Plans using a heuristic loaded from a file."""

    def __init__(
        self,
        types: set[Type],
        predicates: set[Predicate],
        operators: set[LiftedOperator],
        seed: int,
        heuristic_path: Path,
    ) -> None:
        super().__init__(types, predicates, operators, "hff", seed)
        self._heuristic_path = heuristic_path

    def _load_generate_heuristic_fn(self) -> Callable:
        spec = importlib.util.spec_from_file_location("generated_heuristic", self._heuristic_path)
        module = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(module)  # type: ignore
        return getattr(module, "generate_heuristic")

    def _relational_heuristic_factory(
        self,
        init_abstract_state: RelationalAbstractState,
        goal: Goal,
    ) -> Callable[[RelationalAbstractState], float]:
        assert isinstance(init_abstract_state, RelationalAbstractState)
        assert isinstance(goal, RelationalAbstractGoal)
        pddl_problem = PDDLProblem(
            "custom-domain",
            "custom-problem",
            init_abstract_state.objects,
            init_abstract_state.atoms,
            goal.atoms,
        )
        ground_operators = cached_all_ground_operators(
            self._pddl_domain.operators, init_abstract_state.objects
        )
        generate_heuristic = self._load_generate_heuristic_fn()
        pyperplan_heuristic = create_pyperplan_heuristic_from_fn(
            generate_heuristic, self._pddl_domain, pddl_problem, ground_operators
        )
        return lambda s: pyperplan_heuristic(s.atoms)

    def __call__(self, *args: Any, **kwargs: Any) -> Iterator:
        for s_plan, a_plan in super().__call__(*args, **kwargs):
            self._last_abstract_plan = a_plan
            readable = [
                {"operator_name": a.name, "arguments": [o.name for o in a.parameters]}
                for a in a_plan
            ]
            print("Trying abstract plan:", readable)
            yield s_plan, a_plan


class OracleHeuristicApproach(BaseApproach[_O, _X, _U]):
    """Runs the planner with the heuristic from alphatamp/generated_heuristic.py."""

    def __init__(
        self,
        env_models: SesameModels,
        seed: int,
        max_abstract_plans: int = 10,
        samples_per_step: int = 10,
        max_skill_horizon: int = 100,
        heuristic_path: Path = _HEURISTIC_PATH,
        training_planning_timeout: float = 5,
        use_stored_heuristic: bool = True,
    ):
        super().__init__(env_models, seed)
        self._max_abstract_plans = max_abstract_plans
        self._samples_per_step = samples_per_step
        self._heuristic_path = heuristic_path

        self._trajectory_sampler = ParameterizedControllerTrajectorySampler(
            controller_generator=RelationalControllerGenerator(self._env_models.skills),
            transition_function=self._env_models.transition_fn,
            state_abstractor=self._env_models.state_abstractor,
            max_trajectory_steps=max_skill_horizon,
        )
        self._abstract_successor_fn = RelationalAbstractSuccessorGenerator(
            self._env_models.operators
        )
        self.last_metrics: RefinementMetrics | None = None

    def _train(self, problem: PlanningProblem[_X, _U]) -> None:
        pass

    def _run_planning(self, problem: PlanningProblem[_X, _U], timeout: float) -> Plan[_X, _U]:
        print(f"\n=== Running oracle heuristic from {self._heuristic_path} ===")
        generator = HeuristicGenerator(
            types=self._env_models.types,
            predicates=self._env_models.predicates,
            operators=self._env_models.operators,
            seed=self._seed,
            heuristic_path=self._heuristic_path,
        )
        planner = SesamePlanner(
            generator,
            self._trajectory_sampler,
            self._max_abstract_plans,
            self._samples_per_step,
            self._abstract_successor_fn,
            self._env_models.state_abstractor,
            seed=self._seed,
        )
        plan, _ = planner.run(problem, timeout=timeout)
        self.last_metrics = planner.last_metrics

        if plan is None:
            raise TimeoutError("No plan found")
        print("Succeeded with abstract plan:", [
            {"operator_name": a.name, "arguments": [o.name for o in a.parameters]}
            for a in getattr(generator, "_last_abstract_plan", [])
        ])
        return plan
