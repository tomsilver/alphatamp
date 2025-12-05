"""
Approach that uses an LLM to generate an policy, given the oracle in the prompt
"""

# -- from generalized oracle -- 
from typing import Any, Callable, Dict, List, Iterator, Generic, TypeVar

from typing import Any, Iterable, Iterator, TypeAlias, TypeVar

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
from prbench.envs.geom2d.utils import is_inside_shelf
from relational_structs import GroundOperator, Object, ObjectCentricState

from alphatamp.approaches.base_approach import BaseApproach
# ---------------------------------

# copied from llm_ppl_approach.py
from prpl_llm_utils.code import ( 
    FunctionOutputRepromptCheck,
    SyntaxRepromptCheck,
    synthesize_python_function_with_llm,
)
from prpl_llm_utils.models import PretrainedLargeModel
from pathlib import Path
from prpl_llm_utils.structs import Query
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel
# --------------------------------------------------------------


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

class LLMOracleAbstractPlanGenerator(
    AbstractPlanGenerator[_X, RelationalAbstractState, GroundOperator]
):
    """A generator that uses an LLM to generate abstract plans."""

    def __init__(
        self,
        env_models: SesameModels,
        seed: int,
        llm: PretrainedLargeModel # added
    ) -> None:
        """Initialize with env models and seed."""
        super().__init__(noop_successor_fn, seed)
        self._env_models = env_models
        self._llm = llm # added
        self._plan_fn = None # to store llm call, so we don't recall
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

        if self._plan_fn is None: # check to see if plan exists
            self._plan_fn = self._make_plan_fn(s0)

        llm_plan: List[Dict[str, Any]] = self._plan_fn(s0, goal, self._env_models)
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
    
    def _make_plan_fn(self, s0: RelationalAbstractState):
        """Synthesize generate_oracle_plan once, using s0-based prompt."""
        prompt = self._build_prompt(s0)

        reprompt_checks = [
            SyntaxRepromptCheck(),
            FunctionOutputRepromptCheck("generate_oracle_plan", [], []),
        ]
        # prpl-llm-utils expects a Query, not a raw string
        query = Query(
            prompt=prompt,
            imgs=None, # look into this 
            hyperparameters={"temperature": 0.0, },
        )
        # THIS ARG ORDER MATTERS:
        #   model, function_name, examples, prompt, reprompt_checks
        plan_fn = synthesize_python_function_with_llm(
            model=self._llm,
            function_name="generate_oracle_plan",
            query=query
            #reprompt_checks=reprompt_checks,
        )  
        return plan_fn
    
    def _build_prompt(self, s0: RelationalAbstractState) -> str:
        # Include operator signatures with parameter types and order
        ops = "\n".join(
            f"- {op.name}({', '.join(f'{p.name}:{p.type.name}' for p in op.parameters)})"
            for op in self._env_models.operators
        )
        obs = "\n".join(f"- {obj.name} ({obj.type.name})" for obj in sorted(s0.objects, key=lambda o: o.name))
        prompt = f"""
You are an oracle high-level planner for a Sesame TAMP system.

-------------------------------------
Operators:
{ops}

Objects:
{obs}
-------------------------------------

Object API
----------
Each obj has:
- obj.name  (e.g., "block0")
- obj.type.name  (e.g., "target_block")
Never use obj.type_name.                    # interesting that I need to add this, 

Task
----
Write a Python function that returns a FIXED abstract plan (list of action dicts),
ignoring the goal. Each action dict:

{{
  "operator_name": "<operator in env_models.operators>",
  "arguments": ["obj1", "obj2", ...]  # object names from abstract_state
}}

DEBUG BASELINE (ALWAYS THE SAME)
--------------------------------
1. Select objects exactly as:

    robot = next(o for o in abstract_state.objects if o.type.name=="crv_robot")
    shelf = next(o for o in abstract_state.objects if o.type.name=="shelf")
    blocks = sorted(
        [o for o in abstract_state.objects if o.type.name=="target_block"],
        key=lambda o: o.name
    )
    block0, block1, block2 = blocks[:3]

2. Return this fixed plan:

    [
      {{"operator_name":"PickBlockOnShelf",
       "arguments":[robot.name, block0.name, shelf.name]}},
      {{"operator_name":"PlaceBlockNotOnShelf",
       "arguments":[robot.name, block0.name, shelf.name]}},
      {{"operator_name":"PickBlockNotOnShelf",
       "arguments":[robot.name, block0.name, shelf.name]}},
      {{"operator_name":"PlaceBlockOnShelf",
       "arguments":[robot.name, block0.name, shelf.name]}},

      {{"operator_name":"PickBlockNotOnShelf",
       "arguments":[robot.name, block1.name, shelf.name]}},
      {{"operator_name":"PlaceBlockOnShelf",
       "arguments":[robot.name, block1.name, shelf.name]}},

      {{"operator_name":"PickBlockNotOnShelf",
       "arguments":[robot.name, block2.name, shelf.name]}},
      {{"operator_name":"PlaceBlockOnShelf",
       "arguments":[robot.name, block2.name, shelf.name]}},
    ]

Function Signature
------------------
from typing import List, Dict, Any

def generate_oracle_plan(abstract_state, goal, env_models) -> List[Dict[str, Any]]:
    \"\"\"
    Return the fixed debug plan above.
    Use obj.type.name.
    Use operator names exactly as in env_models.operators.
    Ignore the goal completely.
    \"\"\"
"""
#---------------------------------------------------------------------------------------------

        prompt2 = f"""
You are an oracle high-level planner for a Sesame TAMP system.
-------------------------------------
Operators (with parameter order):
{ops}

Objects:
{obs}
-------------------------------------
Task
----
Write a Python function that returns a FIXED abstract plan (list of action dicts),
ignoring the goal. Each action dict:

{{
  "operator_name": "<operator name>",
  "arguments": ["obj1", "obj2", ...]  # MUST match operator parameter order!
}}

Function Signature
------------------
from typing import List, Dict, Any

def generate_oracle_plan(abstract_state, goal, env_models) -> List[Dict[str, Any]]:
    \"\"\"
    Return a plan that places all the blocks on the shelf.
    There are some existing blocks in the shelf that you should remove first.
    Use operator names exactly as in env_models.operators.
    Ignore the goal completely.
    \"\"\"
    
"""
        return prompt2.strip()

    def _ground(self, llm_plan: List[Dict[str, Any]], 
                s0: RelationalAbstractState) -> list[GroundOperator]:
        name_to_op = {s.operator.name: s.operator for s in self._env_models.skills}
        name_to_obj = {obj.name: obj for obj in s0.objects}
        actions = []
        for step in llm_plan:
            op = name_to_op[step["operator_name"]]
            objs = [name_to_obj[n] for n in step["arguments"]]
            actions.append(op.ground(tuple(objs)))
        return actions

class GeneralizedLLMOracleApproach(BaseApproach[_O, _X, _U]):
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
        llm = OpenAIModel("gpt-4.1", cache) # use a better model
        # Create the abstract plan generator.
        self._abstract_plan_generator: AbstractPlanGenerator = (
            LLMOracleAbstractPlanGenerator(
                self._env_models,
                seed=self._seed,
                llm=llm,
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



""" High level next steps:
0. Sesame model is a base class that defines the interface between high level and low level
1. Env model is a SesameModel object, specified for the environment
2. I need to create a prompt that contains the skills, operators, parameters of the specific env model
3. Once I have this prompt, how is the planning being done?

With regular approach
Abstract successor function defines what is possible -> Generate plans --> Sample plans

With oracle approach
Abstract successor function (useless) --> plan is hardcoded

With LLM approach
Abstract successor function defines what is possible --> LLM needs to factor this in and generate a plan --> sample plan 
"""

"""
Goal: Instead of a handwritten oracle plan, use an LLM that sees the 
Sesame models + current state + goal and return an abstract plan that 
Sesame can execute

Steps:
1. Define an LLMORacleAbstractPlanGenerator that implements the same
    interface as my current oracle generator, but calls an LLM

2. Wrap that in an LLMOracleApproach that builds SesamePlanner with 
    this generator

3. Reuse the synthesize_python_function_with_llm pattern from 
    llm_ppl_approach.py

Data Flow:
1. PRBench gives me a PlanningProblem with the env, a SesameModels instance
a goal, and an initial relational abstract state

2. My approach builds a Plan generator, then a sesame planner

3. When sesamePlanner.run() is called, it asks the abstract plan generator
    for candidate plans. For each plan, I do feasabiliyy checks and rollouts

"""

"""
You are an oracle symbolic planner for a Sesame-based task-and-motion planning system.

-------------------------------------
Operators (name only):
{ops}

Objects in the current abstract state:
{obs}
-------------------------------------

Object / type API (IMPORTANT)
-----------------------------
Each object 'obj' in abstract_state.objects has:
- obj.name: a string like "block0", "shelf", "robot"
- obj.type: a Type with attribute 'name', e.g. "target_block", "shelf", "crv_robot"

There is NO attribute 'type_name'. You MUST use 'obj.type.name', never 'obj.type_name'.

Task
----
Write a Python function that, given (abstract_state, goal, env_models),
returns a HIGH-LEVEL ABSTRACT PLAN as a list of action dicts.

Each action dict must be:
    {{
        "operator_name": "<name of operator in env_models.operators>",
        "arguments": ["obj1", "obj2", ...]  # names from abstract_state.objects
    }}

Use ONLY:
- operator names that appear in env_models.operators
- object names that appear in abstract_state.objects

DEBUG BASELINE (FIXED PLAN)
---------------------------
For this debug version, ALWAYS return the SAME fixed sequence of actions,
constructed from the current abstract_state:

1. Select objects using EXACTLY this pattern:

    robot = next(obj for obj in abstract_state.objects
                 if obj.type.name == "crv_robot")
    shelf = next(obj for obj in abstract_state.objects
                 if obj.type.name == "shelf")
    blocks = sorted(
        [obj for obj in abstract_state.objects if obj.type.name == "target_block"],
        key=lambda o: obj.name,
    )
    block0, block1, block2 = blocks[:3]

2. Then build and return the following plan (as a list of dicts):

    plan = [
        # For block0: take from shelf, place off shelf, then back on shelf
        {{"operator_name": "PickBlockOnShelf",
          "arguments": [robot.name, block0.name, shelf.name]}},
        {{"operator_name": "PlaceBlockNotOnShelf",
          "arguments": [robot.name, block0.name, shelf.name]}},
        {{"operator_name": "PickBlockNotOnShelf",
          "arguments": [robot.name, block0.name, shelf.name]}},
        {{"operator_name": "PlaceBlockOnShelf",
          "arguments": [robot.name, block0.name, shelf.name]}},

        # For block1: pick not-on-shelf, place on shelf
        {{"operator_name": "PickBlockNotOnShelf",
          "arguments": [robot.name, block1.name, shelf.name]}},
        {{"operator_name": "PlaceBlockOnShelf",
          "arguments": [robot.name, block1.name, shelf.name]}},

        # For block2: pick not-on-shelf, place on shelf
        {{"operator_name": "PickBlockNotOnShelf",
          "arguments": [robot.name, block2.name, shelf.name]}},
        {{"operator_name": "PlaceBlockOnShelf",
          "arguments": [robot.name, block2.name, shelf.name]}},
    ]

3. Ignore 'goal' entirely. Do not perform any search or reasoning. Just construct and
   return this list.

Function to define
------------------

from typing import List, Dict, Any

def generate_oracle_plan(abstract_state, goal, env_models) -> List[Dict[str, Any]]:
    \"\"\"
    Debug baseline:
    - Ignore 'goal' and always return the same fixed plan described above.
    - Use object types and names from 'abstract_state.objects'.
    - Use 'obj.type.name', never 'obj.type_name'.
    - Use operator_name strings that exactly match env_models.operators.
    \"\"\"
    ...
"""