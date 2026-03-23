"""Approach that uses an LLM to generate an policy, given the oracle in the prompt."""

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
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
)
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.code import (
    SyntaxRepromptCheck,
    synthesize_python_function_with_llm,
)
from prpl_llm_utils.models import OpenAIModel, PretrainedLargeModel
from prpl_llm_utils.structs import Query
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


class LLMAbstractPlanGenerator(
    AbstractPlanGenerator[_X, RelationalAbstractState, GroundOperator]
):
    """A generator that uses an LLM to generate abstract plans."""

    def __init__(
        self, env_models: SesameModels, seed: int, llm: PretrainedLargeModel  # added
    ) -> None:
        """Initialize with env models and seed."""
        super().__init__(noop_successor_fn, seed)
        self._env_models = env_models
        self._llm = llm  # added
        self._plan_fn: Optional[
            Callable[
                [RelationalAbstractState, Goal, SesameModels], List[Dict[str, Any]]
            ]
        ] = None  # store llm call, so we don't recall
        self._seed = seed

    def __call__(
        self,
        x0: _X,
        s0: RelationalAbstractState,
        goal: Goal,
        timeout: float,
        bpg: BilevelPlanningGraph[_X, _U, RelationalAbstractState, GroundOperator],
    ) -> Iterator[tuple[list[RelationalAbstractState], list[GroundOperator]]]:
        """Generate abstract plans."""
        # Cast goal to RelationalAbstractGoal for accessing atoms
        assert isinstance(goal, RelationalAbstractGoal)
        relational_goal = cast(RelationalAbstractGoal, goal)

        if self._plan_fn is None:  # check to see if plan exists
            self._plan_fn = self._make_plan_fn(s0, relational_goal)

        llm_plan: List[Dict[str, Any]] = self._plan_fn(
            s0, relational_goal, self._env_models
        )
        print(llm_plan)
        abstract_actions = self._ground(llm_plan, s0)

        abstract_states: list[RelationalAbstractState] = [s0]

        for abstract_action in abstract_actions:
            next_atoms = (
                abstract_states[-1].atoms - abstract_action.delete_effects
            ) | abstract_action.add_effects
            next_state = RelationalAbstractState(next_atoms, s0.objects)
            abstract_states.append(next_state)

        return iter([(abstract_states, abstract_actions)])

    def _make_plan_fn(self, s0: RelationalAbstractState, goal: RelationalAbstractGoal):
        """Synthesize generate_oracle_plan once, using s0-based prompt."""
        prompt = self._build_prompt(s0, goal)

        reprompt_checks: Sequence[SyntaxRepromptCheck] = [
            SyntaxRepromptCheck(),
        ]
        # prpl-llm-utils expects a Query, not a raw string
        query = Query(
            prompt=prompt,
            imgs=None,
            hyperparameters={
                "temperature": 0.0,
            },
        )
        plan_fn = synthesize_python_function_with_llm(
            model=self._llm,
            function_name="generate_oracle_plan",
            query=query,
            reprompt_checks=list(reprompt_checks),
        )
        return plan_fn

    def _build_prompt(
        self, s0: RelationalAbstractState, goal: RelationalAbstractGoal
    ) -> str:
        # Include operator signatures with parameter types and order
        # We have to do this so that robot, block, and shelf are all handled correctly
        ops = "\n".join(
            f"- {op.name}({', '.join(f'{p.name}:{p.type.name}' for p in op.parameters)})"
            for op in self._env_models.operators
        )

        obs = "\n".join(
            f"- {obj.name} ({obj.type.name})"
            for obj in sorted(s0.objects, key=lambda o: o.name)
        )

        # Extract initial state predicates/atoms
        initial_atoms = "\n".join(f"- {atom}" for atom in sorted(s0.atoms, key=str))

        # Extract goal predicates/atoms
        goal_atoms = "\n".join(f"- {atom}" for atom in sorted(goal.atoms, key=str))

        prompt = f"""
You are an oracle high-level planner for a Sesame TAMP system.
-------------------------------------
Operators (with parameter order):
{ops}

Objects:
{obs}

Initial State (true predicates):
{initial_atoms}

Goal State (target predicates)
{goal_atoms}
-------------------------------------

Task
----
Write a Python function that returns a FIXED abstract plan (list of action dicts). Each action dict:
{{
  "operator_name": "<operator name>",
  "arguments": ["obj1", "obj2", ...]  # MUST match operator parameter order!
}}

Function Signature
------------------
from typing import List, Dict, Any

def generate_oracle_plan(abstract_state, goal, env_models) -> List[Dict[str, Any]]:
    \"\"\"
    Given the initial state above, return a plan that achieves the goal.
    Hint: Remove all the blocks before placing them back on the shelf.
    Use operator names exactly as in env_models.operators.
    \"\"\"

"""
        # Note! For earlier iterations of the prompt, telling the LLM
        # to ignore the goal helped with generation of end-to-end plans
        return prompt.strip()

    def _ground(
        self, llm_plan: List[Dict[str, Any]], s0: RelationalAbstractState
    ) -> list[GroundOperator]:
        name_to_op = {s.operator.name: s.operator for s in self._env_models.skills}
        name_to_obj = {obj.name: obj for obj in s0.objects}
        actions = []
        for step in llm_plan:
            op = name_to_op[step["operator_name"]]
            objs = [name_to_obj[n] for n in step["arguments"]]
            actions.append(op.ground(tuple(objs)))
        return actions


class BaseLLMApproach(BaseApproach[_O, _X, _U]):
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

        # Create the llm
        cache = SQLite3PretrainedLargeModelCache(Path("llm_cache.db"))
        self._llm = OpenAIModel("gpt-4.1", cache)  # use a better model
        # Create the abstract plan generator.
        self._abstract_plan_generator: AbstractPlanGenerator = LLMAbstractPlanGenerator(
            self._env_models,
            seed=self._seed,
            llm=self._llm,
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
