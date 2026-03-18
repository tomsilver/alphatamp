"""Approach that uses an LLM to generate an policy, given the oracle in the prompt."""

import time
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
)

from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.refiners.backtracking_refiner import BacktrackingRefiner
from bilevel_planning.structs import (
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

# The variables below are generic type variables
_O = TypeVar("_O")  # observation
_U = TypeVar("_U")  # action
_X = TypeVar("_X", bound=ObjectCentricState)  # state
_S = TypeVar(
    "_S", bound=RelationalAbstractState
)  # abstract state. Must be hashable for search algorithms
_A = TypeVar("_A", bound=GroundOperator)  # abstract action, must also be hashable
Skeleton: TypeAlias = tuple[list[_S], list[_A]]  # a set of abstract states and actions
FrozenSkeleton: TypeAlias = tuple[tuple[_S, ...], tuple[_A, ...]]  # immutable skeleton


# abstract plan generator expects a successor function in the constructor, but this LLM
# approach doesn't use a successor function, since the LLM does the reasoning
def noop_successor_fn(_s: _S) -> Iterable[tuple[_A, _S]]:
    """Return no successors; placeholder to satisfy AbstractPlanGenerator.__init__."""
    return []


class RepromptLLMAbstractPlanGenerator(
    AbstractPlanGenerator[_X, RelationalAbstractState, GroundOperator]
):
    """A generator that uses an LLM to generate abstract plans."""

    def __init__(
        self,
        env_models: SesameModels,
        seed: int,
        llm: PretrainedLargeModel,  # llm is the LLM instance that will generate plans
        failure_context: Optional[
            Dict[str, Any]
        ] = None,  # additional failure_context parameter
    ) -> None:
        """Initialize with env models and seed."""
        super().__init__(noop_successor_fn, seed)
        self._env_models = env_models
        self._llm = llm
        self._seed = seed
        self._plan_fn: Optional[
            Callable[
                [RelationalAbstractState, RelationalAbstractGoal, SesameModels],
                List[Dict[str, Any]],
            ]
        ] = None  # store llm call, so we don't recall
        self.failure_context = failure_context
        self._last_abstract_actions: list[GroundOperator] = []  # to store plan
        # to store failure context
        self._last_abstract_states: list[RelationalAbstractState] = []

    def __call__(
        self,
        x0: _X,
        s0: RelationalAbstractState,
        goal: Any,
        timeout: float,
        bpg: BilevelPlanningGraph[_X, _U, RelationalAbstractState, GroundOperator],
    ) -> Iterator[tuple[list[RelationalAbstractState], list[GroundOperator]]]:
        """Generate abstract plans."""

        if self._plan_fn is None:  # check to see if the planning function exists
            # no benefit to cacheing for my current testing, since I only do one test
            # cacheing helps if I want to use one plan across multiple problem instances
            self._plan_fn = self._make_plan_fn(s0, goal)

        # Generate plan with llm. Provide current state, goal, and env_models
        # This returns a list of dicts, representing an action like
        # "operator name": "pick", "arguments": ["robot, block1"]
        llm_plan: List[Dict[str, Any]] = self._plan_fn(s0, goal, self._env_models)

        print(llm_plan)

        # convert the dictionary representation into GroundOperator objects
        abstract_actions = self._ground(llm_plan, s0)
        self._last_abstract_actions = abstract_actions  # store for failure analysis
        abstract_states: list[RelationalAbstractState] = [s0]

        # calculate the sequence of abstract states by applying each actions affects
        # for each action compute the next state by removing atoms,
        # adding atoms, and appending state to the list
        for abstract_action in abstract_actions:
            next_atoms = (
                abstract_states[-1].atoms - abstract_action.delete_effects
            ) | abstract_action.add_effects
            next_state = RelationalAbstractState(next_atoms, s0.objects)
            abstract_states.append(next_state)

        self._last_abstract_states = abstract_states

        # final result is an iterator representing one plan
        return iter([(abstract_states, abstract_actions)])

    def _make_plan_fn(self, s0: RelationalAbstractState, goal: RelationalAbstractGoal):
        """Synthesize generate_oracle_plan once, using s0-based prompt."""
        prompt = self._build_prompt(
            s0, goal
        )  # build a text prompt describing the problem

        reprompt_checks: Sequence[SyntaxRepromptCheck] = [
            SyntaxRepromptCheck(),
        ]
        # prpl-llm-utils expects a Query, not a raw string
        # query is how I actually send prompt to llm.
        # if I want to add an image, add it here
        # 0.0 temperature makes it deterministic
        query = Query(
            prompt=prompt,
            imgs=None,
            hyperparameters={
                "temperature": 0.0,
            },
        )
        # this sends the prompt to the llm,
        # and the llm generates a function called "generate_oracle_plan"
        # function is parsed and checked, and returned through plan_fn
        # the function right now just returns the plan
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

        # list all objects and their type in the initial state
        obs = "\n".join(
            f"- {obj.name} ({obj.type.name})"
            for obj in sorted(s0.objects, key=lambda o: o.name)
        )

        # Extract initial state predicates/atoms
        initial_atoms = "\n".join(f"- {atom}" for atom in sorted(s0.atoms, key=str))

        # Extract goal predicates/atoms
        goal_atoms = "\n".join(f"- {atom}" for atom in sorted(goal.atoms, key=str))

        # failure section. injected only if failure context exists
        failure_section = ""
        if self.failure_context is not None:
            failed_plan_str = "\n".join(
                f"- {step['operator_name']}({', '.join(step['arguments'])})"
                for step in self.failure_context["failed_plan"]
            )
            coords_str = "\n".join(
                f"- {name}: ({x:.3f}, {y:.3f})"
                for name, (x, y) in sorted(self.failure_context["coordinates"].items())
            )

            failed_action = self.failure_context["failed_action"]
            idx = failed_action["index"]
            action = failed_action["action"]
            predicates = "\n".join(f"- {p}" for p in failed_action["predicates"])
            failure_section = (
                "Previous Attempt (FAILED)\n"
                "-------------------------\n"
                "The following plan was attempted but FAILED during low-level"
                "trajectory sampling. Reason about why this plan might have failed"
                "and generate a DIFFERENT plan.\n"
                f"Failed plan:\n{failed_plan_str}\n\n"
                f"The plan failed at step {idx} ({action})\n"
                "The abstract state before this action had these predicates:\n"
                f"{predicates}"
                f"Object positions of the failed state:\n{coords_str}\n"
                "Reason and replan"
                "-------------------------\n"
            )

        prompt = f"""
You are an oracle high-level planner for a Sesame TAMP system.
-------------------------------------
Operators (with parameter order):
{ops}

Objects:
{obs}

Initial State (true predicates):
{initial_atoms}

Goal State (target predicates):
{goal_atoms}
-------------------------------------
{failure_section}
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
    Given the initial and goal states above, return a plan that achieves the goal.
    Use operator names exactly as in env_models.operators.
    \"\"\"

"""
        return prompt.strip()

    # convert dicts to GroundOperator objects
    def _ground(
        self, llm_plan: List[Dict[str, Any]], s0: RelationalAbstractState
    ) -> list[GroundOperator]:
        # map operator names to operator objects
        name_to_op = {s.operator.name: s.operator for s in self._env_models.skills}
        # map object names to object instances
        name_to_obj = {obj.name: obj for obj in s0.objects}
        actions = []
        # for each action dictionary from the plan, call op.ground() to create a
        # GroundOperator object that contains operator + specific objects like block2
        for step in llm_plan:
            op = name_to_op[step["operator_name"]]
            objs = [name_to_obj[n] for n in step["arguments"]]
            actions.append(op.ground(tuple(objs)))
        return actions


class RepromptApproach(BaseApproach[_O, _X, _U]):
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
        self.last_metrics: RefinementMetrics | None = None
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
        self._llm = OpenAIModel("gpt-4.1", cache)

        # Create the abstract successor function (not really used).
        self._abstract_successor_fn = RelationalAbstractSuccessorGenerator(
            self._env_models.operators
        )

    def _train(self, problem: PlanningProblem[_X, _U]) -> None:
        pass

    def _run_planning(
        self, problem: PlanningProblem[_X, _U], timeout: float
    ) -> Plan[_X, _U]:
        """1. Run planner once. 2) If it fails, extract failure info and try again."""
        start_time = time.perf_counter()

        # Create the abstract plan generator.
        initial_plan_generator: RepromptLLMAbstractPlanGenerator
        initial_plan_generator = RepromptLLMAbstractPlanGenerator(
            self._env_models, seed=self._seed, llm=self._llm, failure_context=None
        )

        # Finish the planner.
        # Sesame planner uses operators to check symbolic feasbility and calls skills
        # using the sampler to attempt low-level roll-outs
        initial_planner = SesamePlanner(
            initial_plan_generator,
            self._trajectory_sampler,
            max_abstract_plans=1,
            num_sampling_attempts_per_step=self._samples_per_step,
            abstract_successor_function=self._abstract_successor_fn,
            state_abstractor=self._env_models.state_abstractor,
            seed=self._seed,
        )

        tracking_refiner = FailureTrackingBacktrackingRefiner(
            self._trajectory_sampler,
            self._samples_per_step,
            seed=self._seed,
        )
        initial_planner._refiner = tracking_refiner  # pylint: disable=protected-access

        # Run the planner. # Need to change the timeout param later on
        plan, _ = initial_planner.run(problem, timeout=min(50, timeout))
        # 1. when sesame planner.run is called, the generator returns (as, aa)
        # 2. Call backtracking refiner to refine abstract plan.
        # 2a. Sample low-level traj for each abstract action
        # 3. if successful, return (plan, bpg), if failed return (none, bpg)
        # bpg contains search tree of states and actions tried. Use it next for next step
        self.last_metrics = tracking_refiner.metrics
        if plan is not None:
            print("Initial plan succeeded.")
            return plan

        print("Initial plan failed")

        # extract failure context from failed plan
        remaining_time = timeout - (time.perf_counter() - start_time)
        if remaining_time <= 0:
            raise TimeoutError("init plan failed")

        failure_context = self._extract_failure_context(
            initial_plan_generator._last_abstract_actions,  # pylint: disable=protected-access
            tracking_refiner._failed_concrete_state,  # type: ignore[arg-type]  # pylint: disable=protected-access
            tracking_refiner._deepest_failed_index,  # pylint: disable=protected-access
            initial_plan_generator._last_abstract_states,  # pylint: disable=protected-access
        )

        replanned_generator: RepromptLLMAbstractPlanGenerator
        replanned_generator = RepromptLLMAbstractPlanGenerator(
            self._env_models,
            seed=self._seed,
            llm=self._llm,
            failure_context=failure_context,
        )

        replanner = SesamePlanner(
            replanned_generator,
            self._trajectory_sampler,
            max_abstract_plans=1,
            num_sampling_attempts_per_step=self._samples_per_step,
            abstract_successor_function=self._abstract_successor_fn,
            state_abstractor=self._env_models.state_abstractor,
            seed=self._seed,
        )

        plan, _ = replanner.run(problem, timeout=remaining_time)
        self.last_metrics = replanner.last_metrics

        if plan is None:
            raise TimeoutError("No plan found")
        return plan

    @staticmethod
    def _extract_failure_context(
        abstract_actions: list[GroundOperator],
        failed_state: ObjectCentricState,
        failed_index: int,
        abstract_states: list[RelationalAbstractState],
    ) -> Dict[str, Any]:
        """Extract failure context from the attempted plan and initial state."""
        failed_plan = [
            {"operator_name": a.name, "arguments": [o.name for o in a.parameters]}
            for a in abstract_actions
        ]

        # Extract (x, y) for every object using state.get(obj, feature).
        # Shelf uses "x1"/"y1" for its primary surface; "x"/"y" is the bookend.
        coordinates = {}
        for obj in failed_state.data:
            if "x1" in failed_state.type_features.get(obj.type, []):
                # Shelf (DoubleRectType) — use x1/y1 for primary surface
                x = float(failed_state.get(obj, "x1"))
                y = float(failed_state.get(obj, "y1"))
            else:
                # Robot, blocks — standard x/y
                x = float(failed_state.get(obj, "x"))
                y = float(failed_state.get(obj, "y"))
            coordinates[obj.name] = (x, y)

        # which action failed and the predicates at that point
        failed_action_info = {
            "index": failed_index,
            "action": str(abstract_actions[failed_index]),
            "predicates": sorted(str(a) for a in abstract_states[failed_index].atoms),
        }

        return {
            "failed_plan": failed_plan,
            "failed_action": failed_action_info,
            "coordinates": coordinates,
        }


class FailureTrackingBacktrackingRefiner(BacktrackingRefiner):
    """New backtracking refiner to track failure information."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # deepest failed index represents the furthest I get in the plan
        self._deepest_failed_index: int = -1
        self._failed_concrete_state: ObjectCentricState | None = None

    def __call__(self, x0, s_plan, a_plan, timeout, bpg) -> Plan | None:
        self._deepest_failed_index = -1  # reset each call
        self._failed_concrete_state = None
        return super().__call__(x0, s_plan, a_plan, timeout, bpg)

    def _refine_from_step(
        self, index, x, s_plan, a_plan, remaining_time, bpg
    ) -> tuple[bool, list | None]:
        success, plan = super()._refine_from_step(
            index, x, s_plan, a_plan, remaining_time, bpg
        )
        if not success and index > self._deepest_failed_index:
            self._deepest_failed_index = index
            self._failed_concrete_state = x  # the concrete state when action attempted
        return success, plan
