"""
Approach that uses an LLM to generate an policy, given the oracle in the prompt
"""

from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
    cast,
)

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
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from alphatamp.approaches.cluttered_storage.prompt import HEURISTIC_PROMPT
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
    cached_all_ground_operators,
    create_pyperplan_heuristic_from_fn
)
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.code import (
    SyntaxRepromptCheck,
    synthesize_python_function_with_llm,
)
from prpl_llm_utils.models import OpenAIModel, PretrainedLargeModel
from prpl_llm_utils.structs import Query
from relational_structs import (LiftedOperator,
                                GroundOperator,
                                ObjectCentricState,
                                PDDLProblem,
                                Predicate,
                                Type)

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


class HeuristicGenerator(
    RelationalHeuristicSearchAbstractPlanGenerator):
    """A generator that uses an LLM to generate heuristic instead of hFF"""

    def __init__(
        self,
        types: set[Type],
        predicates: set[Predicate],
        operators: set[LiftedOperator],
        llm: Any,
        seed: int,
        prompt: str,
    ) -> None:
        super().__init__(types, predicates, operators, "hff", seed)
        query = Query(
            prompt=prompt,
            imgs=None,
            hyperparameters={"temperature": 1.0},
        )
        self._llm_fn = synthesize_python_function_with_llm(
            model=llm,
            function_name="generate_heuristic",
            query=query,
            reprompt_checks=[SyntaxRepromptCheck()],
        )
        Path("generated_heuristic.py").write_text(self._llm_fn.code_str)
    
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
        # Load the module directly to avoid pickling issues: SynthesizedPythonFunction.run()
        # passes results through mp.Manager which requires pickle, but locally-defined
        # classes (defined inside generate_heuristic) cannot be pickled.
        module = self._llm_fn._load_module()
        generate_heuristic = getattr(module, self._llm_fn.function_name)
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


class HeuristicLLMApproach(BaseApproach[_O, _X, _U]):
    """Uses an LLM-generated heuristic for abstract planning."""

    def __init__(
        self,
        env_models: SesameModels,
        seed: int,
        max_abstract_plans: int = 10,
        samples_per_step: int = 10,
        max_skill_horizon: int = 100,
        skeleton_batch_size: int = 100,
        num_training_skeletons_per_problem: int = 10,
        training_planning_timeout: float = 5,
    ):
        super().__init__(env_models, seed)
        self._max_abstract_plans = max_abstract_plans
        self._samples_per_step = samples_per_step
        self._max_skill_horizon = max_skill_horizon
        self._skeleton_batch_size = skeleton_batch_size
        self._num_training_skeletons_per_problem = num_training_skeletons_per_problem
        self._training_planning_timeout = training_planning_timeout

        # create the sampler
        self._trajectory_sampler = ParameterizedControllerTrajectorySampler(
            controller_generator=RelationalControllerGenerator(self._env_models.skills),
            transition_function=self._env_models.transition_fn,
            state_abstractor=self._env_models.state_abstractor,
            max_trajectory_steps=self._max_skill_horizon,
        )
        
        # create the llm
        cache = SQLite3PretrainedLargeModelCache(Path("llm_cache.db"))
        self._llm = OpenAIModel("gpt-4.1", cache)
        
        # heuristic plan generator
        # paste the prompt text directly here as a string
        prompt = HEURISTIC_PROMPT
        self._abstract_plan_generator: AbstractPlanGenerator = (
            HeuristicGenerator(
                types=self._env_models.types,
                predicates=self._env_models.predicates,
                operators=self._env_models.operators,
                llm=self._llm,
                seed=self._seed,
                prompt=prompt,
            )
        )
        # create the abstract successor function
        self._abstract_successor_fn = RelationalAbstractSuccessorGenerator(
            self._env_models.operators
        )

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
        plan, _ = self._planner.run(problem, timeout=timeout)
        if plan is None:
            raise TimeoutError("No plan found")
        last = self._abstract_plan_generator._last_abstract_plan
        print("Succeeded with abstract plan:", [
            {"operator_name": a.name, "arguments": [o.name for o in a.parameters]}
            for a in last
        ])
        return plan

