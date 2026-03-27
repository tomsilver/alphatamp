"""
Approach that uses an LLM to generate an policy, given the oracle in the prompt
"""

import ast
import importlib.util
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
    RefinementMetrics,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)
from alphatamp.approaches.cluttered_storage.prompt import build_heuristic_prompt
from bilevel_planning.refiners.backtracking_refiner import (
    BacktrackingRefiner,  # NEW: base class for FailureTrackingBacktrackingRefiner
)
from bilevel_planning.trajectory_samplers.parameterized_controller_sampler import (
    ParameterizedControllerTrajectorySampler,
)
from bilevel_planning.utils import (
    RelationalAbstractSuccessorGenerator,
    RelationalControllerGenerator,
    cached_all_ground_operators,
    create_pyperplan_heuristic_from_fn
)
from pyperplan.heuristics.heuristic_base import Heuristic as PyperplanHeuristic
from prpl_llm_utils.cache import SQLite3PretrainedLargeModelCache
from prpl_llm_utils.models import OpenAIModel
from prpl_llm_utils.reprompting import (
    RepromptCheck,
    create_reprompt_from_error_message,
    query_with_reprompts,
)
from prpl_llm_utils.structs import Query, Response
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


def _parse_all_python_code_blocks(text: str) -> list[str]:
    """Extract all ```python ... ``` blocks from text."""
    blocks = []
    prefix = "```python"
    suffix = "```"
    remaining = text
    while prefix in remaining:
        start = remaining.index(prefix) + len(prefix)
        remaining = remaining[start:]
        if suffix in remaining:
            end = remaining.index(suffix)
            blocks.append(remaining[:end])
            remaining = remaining[end + len(suffix):]
        else:
            blocks.append(remaining)
            break
    return blocks


class _MultiBlockSyntaxCheck(RepromptCheck):
    """Validate that the response contains N syntactically valid Python blocks."""

    def __init__(self, expected_count: int) -> None:
        self._expected_count = expected_count

    def get_reprompt(self, query: Query, response: Response) -> Query | None:
        blocks = _parse_all_python_code_blocks(response.text)
        if len(blocks) < self._expected_count:
            error_msg = (
                f"Expected {self._expected_count} ```python blocks but found "
                f"{len(blocks)}. Please provide exactly {self._expected_count} "
                f"separate ```python code blocks."
            )
            return create_reprompt_from_error_message(query, response, error_msg)
        for i, block in enumerate(blocks):
            try:
                ast.parse(block)
            except SyntaxError as e:
                error_msg = f"Heuristic {i} has a syntax error: {e}"
                return create_reprompt_from_error_message(query, response, error_msg)
        return None


def _load_heuristic_fn(code_str: str) -> Callable:
    """Load a generate_heuristic function from a code string."""
    namespace: dict[str, Any] = {"Heuristic": PyperplanHeuristic}
    exec(compile(code_str, "<generated_heuristic>", "exec"), namespace)  # noqa: S102
    return namespace["generate_heuristic"]


class HeuristicGenerator(
    RelationalHeuristicSearchAbstractPlanGenerator):
    """A generator that uses an LLM to generate heuristic instead of hFF"""

    def __init__(
        self,
        types: set[Type],
        predicates: set[Predicate],
        operators: set[LiftedOperator],
        seed: int,
        use_stored_heuristic: bool = False,
        stored_heuristic_path: Path = Path("generated_heuristic.py"),
        generate_heuristic_fn: Optional[Callable] = None,
    ) -> None:
        super().__init__(types, predicates, operators, "hff", seed)
        self._use_stored_heuristic = use_stored_heuristic
        self._stored_heuristic_path = stored_heuristic_path
        self._generate_heuristic_fn = generate_heuristic_fn

        self._last_abstract_states: list = []

        if use_stored_heuristic:
            print(f"Loading stored heuristic from {stored_heuristic_path}")

    def _load_generate_heuristic_fn(self) -> Callable:
        """Load generate_heuristic from the stored file via importlib."""
        spec = importlib.util.spec_from_file_location(
            "generated_heuristic", self._stored_heuristic_path
        )
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
        if self._generate_heuristic_fn is not None:
            generate_heuristic = self._generate_heuristic_fn
        else:
            generate_heuristic = self._load_generate_heuristic_fn()
        pyperplan_heuristic = create_pyperplan_heuristic_from_fn(
            generate_heuristic, self._pddl_domain, pddl_problem, ground_operators
        )
        return lambda s: pyperplan_heuristic(s.atoms)

    def __call__(self, *args: Any, **kwargs: Any) -> Iterator:
        for s_plan, a_plan in super().__call__(*args, **kwargs):
            self._last_abstract_plan = a_plan
            self._last_abstract_states = s_plan
            readable = [
                {"operator_name": a.name, "arguments": [o.name for o in a.parameters]}
                for a in a_plan
            ]
            print("Trying abstract plan:", readable)
            yield s_plan, a_plan


# NEW: copied from reprompt_approach.py so we can track how deep each candidate gets.
# _deepest_failed_index is the metric: higher = heuristic guided the search further.
class FailureTrackingBacktrackingRefiner(BacktrackingRefiner):
    """Backtracking refiner that records the furthest abstract-plan step reached."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._deepest_failed_index: int = -1
        self._failed_concrete_state: ObjectCentricState | None = None

    def __call__(self, x0, s_plan, a_plan, timeout, bpg) -> Plan | None:
        self._deepest_failed_index = -1  # reset for each new plan attempt
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
            self._failed_concrete_state = x
        return success, plan


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
        use_stored_heuristic: bool = False,
        stored_heuristic_path: Path = Path("generated_heuristic.py"),
        # NEW: number of candidate heuristics to generate and evaluate
        num_candidates: int = 3,
        # NEW: per-candidate evaluation timeout in seconds
        eval_timeout: float = 20.0,
    ):
        super().__init__(env_models, seed)
        self._max_abstract_plans = max_abstract_plans
        self._samples_per_step = samples_per_step
        self._max_skill_horizon = max_skill_horizon
        # NEW: store these for use in _run_planning
        self._use_stored_heuristic = use_stored_heuristic
        self._stored_heuristic_path = stored_heuristic_path
        self._num_candidates = num_candidates
        self._eval_timeout = eval_timeout

        # create the sampler (unchanged)
        self._trajectory_sampler = ParameterizedControllerTrajectorySampler(
            controller_generator=RelationalControllerGenerator(self._env_models.skills),
            transition_function=self._env_models.transition_fn,
            state_abstractor=self._env_models.state_abstractor,
            max_trajectory_steps=self._max_skill_horizon,
        )

        # create the llm (unchanged)
        cache = SQLite3PretrainedLargeModelCache(Path("llm_cache.db"))
        self._llm = OpenAIModel("gpt-4.1", cache)

        # create the abstract successor function (unchanged)
        self._abstract_successor_fn = RelationalAbstractSuccessorGenerator(
            self._env_models.operators
        )

        # REMOVED: no longer create _abstract_plan_generator or _planner here.
        # The pipeline in _run_planning creates fresh generators per candidate.

        # Refinement metrics from the most recent successful run_planning call.
        self.last_metrics: RefinementMetrics | None = None

    def _train(self, problem: PlanningProblem[_X, _U]) -> None:
        pass

    # ── NEW helpers ──────────────────────────────────────────────────────────

    def _make_generator(
        self,
        generate_heuristic_fn: Optional[Callable] = None,
        use_stored: bool = False,
        stored_path: Optional[Path] = None,
    ) -> HeuristicGenerator:
        """
        Factory for HeuristicGenerator.
        Called once per candidate during evaluation and once for final planning.
        """
        return HeuristicGenerator(
            types=self._env_models.types,
            predicates=self._env_models.predicates,
            operators=self._env_models.operators,
            seed=self._seed,
            use_stored_heuristic=use_stored,
            stored_heuristic_path=stored_path or Path("generated_heuristic.py"),
            generate_heuristic_fn=generate_heuristic_fn,
        )

    def _make_planner(self, generator: HeuristicGenerator) -> SesamePlanner:
        """
        Factory for SesamePlanner.
        Called once per candidate during evaluation and once for final planning.
        """
        return SesamePlanner(
            generator,
            self._trajectory_sampler,
            self._max_abstract_plans,
            self._samples_per_step,
            self._abstract_successor_fn,
            self._env_models.state_abstractor,
            seed=self._seed,
        )

    def _synthesize_all_heuristics(self, prompt: str) -> list[tuple[Callable, str]]:
        """Single LLM call → list of (generate_heuristic fn, code_str) for each candidate."""
        multi_prompt = (
            prompt
            + f"\n\nProvide exactly {self._num_candidates} diverse heuristic implementations, "
            f"each in its own separate ```python code block. "
            f"Each block must be self-contained and define a `def generate_heuristic(task):` "
            f"function that returns a heuristic instance."
        )
        query = Query(prompt=multi_prompt, imgs=None, hyperparameters={"temperature": 1.0})
        reprompt_checks: list[RepromptCheck] = [_MultiBlockSyntaxCheck(self._num_candidates)]
        response = query_with_reprompts(self._llm, query, reprompt_checks)
        blocks = _parse_all_python_code_blocks(response.text)[:self._num_candidates]

        results = []
        for i, code_str in enumerate(blocks):
            try:
                results.append((_load_heuristic_fn(code_str), code_str))
            except Exception as e:  # pylint: disable=broad-except
                print(f"  Skipping malformed heuristic block {i}: {e}")
        return results

    def _evaluate_heuristic(
        self,
        problem: PlanningProblem,
        generate_heuristic_fn: Callable,
        eval_timeout: float,
    ) -> tuple[int, bool]:
        """
        Run a short Sesame trial with one candidate heuristic.
        Uses FailureTrackingBacktrackingRefiner to record the furthest step reached.
        Returns (deepest_failed_index, plan_succeeded).
          - deepest_failed_index=-1 means the planner never made progress
          - plan_succeeded=True means this candidate already solved the problem
        """
        generator = self._make_generator(generate_heuristic_fn=generate_heuristic_fn)
        planner = self._make_planner(generator)

        # Swap in the tracking refiner so we get _deepest_failed_index
        tracking_refiner = FailureTrackingBacktrackingRefiner(
            self._trajectory_sampler,
            self._samples_per_step,
            seed=self._seed,
        )
        planner._refiner = tracking_refiner  # pylint: disable=protected-access

        plan, _ = planner.run(problem, timeout=eval_timeout)
        return (
            tracking_refiner._deepest_failed_index,  # pylint: disable=protected-access
            plan is not None,
        )

    def _select_best_heuristic(
        self,
        problem: PlanningProblem,
        candidates: list[tuple[Callable, str]],
        start_time: float,
        timeout: float,
    ) -> tuple[Callable, str, float]:
        """Evaluate candidates and return (best_fn, best_code, best_score)."""
        best_fn: Callable = candidates[0][0]
        best_code: str = candidates[0][1]
        best_score: float = float("-inf")

        for i, (fn, code_str) in enumerate(candidates):
            elapsed = time.perf_counter() - start_time
            remaining = timeout - elapsed
            if remaining < self._eval_timeout:
                print(f"  Skipping heuristic {i + 1}: only {remaining:.1f}s remaining")
                break

            print(f"\n  --- Evaluating heuristic {i + 1}/{len(candidates)} ---")
            try:
                deepest, succeeded = self._evaluate_heuristic(problem, fn, self._eval_timeout)
            except Exception as e:  # pylint: disable=broad-except
                print(f"  Heuristic {i + 1} raised exception: {e}")
                deepest, succeeded = -1, False

            score: float = float("inf") if succeeded else float(deepest)
            print(
                f"  Heuristic {i + 1}: deepest_step={deepest}, "
                f"succeeded={succeeded}, score={score}"
            )

            if score > best_score:
                best_score = score
                best_fn = fn
                best_code = code_str
                Path("llm_heuristic_best.py").write_text(best_code)

            if succeeded:
                print(f"  Heuristic {i + 1} solved the problem during eval — using it.")
                break

        return best_fn, best_code, best_score

    @staticmethod
    def _extract_failure_context(
        abstract_actions: list,
        failed_state: Any,
        failed_index: int,
        abstract_states: list,
    ) -> Dict[str, Any]:
        """Extract failure context from the attempted plan and concrete state."""
        failed_plan = [
            {"operator_name": a.name, "arguments": [o.name for o in a.parameters]}
            for a in abstract_actions
        ]
        coordinates: Dict[str, tuple] = {}
        for obj in failed_state.data:
            if "x1" in failed_state.type_features.get(obj.type, []):
                x = float(failed_state.get(obj, "x1"))
                y = float(failed_state.get(obj, "y1"))
            else:
                x = float(failed_state.get(obj, "x"))
                y = float(failed_state.get(obj, "y"))
            coordinates[obj.name] = (x, y)
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

    @staticmethod
    def _build_failure_section(failure_context: Dict[str, Any]) -> str:
        """Build a failure context string to append to the heuristic prompt."""
        failed_plan_str = "\n".join(
            f"- {step['operator_name']}({', '.join(step['arguments'])})"
            for step in failure_context["failed_plan"]
        )
        coords_str = "\n".join(
            f"- {name}: ({x:.3f}, {y:.3f})"
            for name, (x, y) in sorted(failure_context["coordinates"].items())
        )
        failed_action = failure_context["failed_action"]
        idx = failed_action["index"]
        action = failed_action["action"]
        predicates = "\n".join(f"- {p}" for p in failed_action["predicates"])
        return (
            "\n\nPrevious Attempt (FAILED)\n"
            "-------------------------\n"
            "The heuristics generated previously failed to guide the planner to a "
            "solution during low-level trajectory sampling. Use this failure "
            "information to generate BETTER heuristics that guide the planner past "
            "this failure point.\n"
            f"Attempted abstract plan:\n{failed_plan_str}\n\n"
            f"The plan failed at step {idx} ({action})\n"
            "The abstract state before this action had these predicates:\n"
            f"{predicates}\n"
            f"Object positions at the point of failure:\n{coords_str}\n"
            "Generate heuristics that better prioritize the actions needed to succeed "
            "at this step and beyond.\n"
            "-------------------------"
        )

    def _run_planning(
        self, problem: PlanningProblem[_X, _U], timeout: float
    ) -> Plan[_X, _U]:
        start_time = time.perf_counter()

        # ── Fast path: oracle / pre-written heuristic, skip LLM pipeline ─────
        if self._use_stored_heuristic:
            generator = self._make_generator(
                use_stored=True, stored_path=self._stored_heuristic_path
            )
            planner = self._make_planner(generator)
            plan, _ = planner.run(problem, timeout=timeout)
            if plan is None:
                raise TimeoutError("No plan found")
            self.last_metrics = planner.last_metrics
            print("Succeeded with abstract plan:", [
                {"operator_name": a.name, "arguments": [o.name for o in a.parameters]}
                for a in getattr(generator, "_last_abstract_plan", [])
            ])
            return plan

        # ── Step 1: build a problem-specific prompt ───────────────────────────
        # Extract s0 (abstract initial state) and goal atoms from the problem.
        # This is what gets injected into the prompt so the LLM knows the instance.
        s0 = self._env_models.state_abstractor(problem.initial_state)
        goal = problem.goal
        assert isinstance(goal, RelationalAbstractGoal)
        initial_atoms = "\n".join(f"- {atom}" for atom in sorted(s0.atoms, key=str))
        goal_atoms = "\n".join(f"- {atom}" for atom in sorted(goal.atoms, key=str))
        prompt_str = build_heuristic_prompt(initial_atoms, goal_atoms)

       # ── Step 2: one LLM call → 3 candidate heuristics ────────────────────
        print(f"\n=== Generating {self._num_candidates} candidates in one LLM call ===")
        candidates = self._synthesize_all_heuristics(prompt_str)
        for i, (_, code_str) in enumerate(candidates):
            Path(f"src/alphatamp/approaches/cluttered_storage/llm_heuristic_{i}.py").write_text(code_str)
        print(f"  Got {len(candidates)} valid heuristics from LLM response")
        if not candidates:
            raise RuntimeError("LLM returned no valid heuristic code blocks")


        # ── Step 3: evaluate each candidate, track best by deepest step ───────
        print(f"\n=== Evaluating candidates ({self._eval_timeout}s each) ===")
        best_fn, best_code, best_score = self._select_best_heuristic(
            problem, candidates, start_time, timeout
        )
        print(f"\n=== Best score: {best_score} — running full planning ===")

        # ── Step 4: full planning with the winning heuristic ─────────────────
        remaining_time = timeout - (time.perf_counter() - start_time)
        generator = self._make_generator(generate_heuristic_fn=best_fn)
        planner = self._make_planner(generator)
        tracking_refiner = FailureTrackingBacktrackingRefiner(
            self._trajectory_sampler, self._samples_per_step, seed=self._seed
        )
        planner._refiner = tracking_refiner  # pylint: disable=protected-access
        plan, _ = planner.run(problem, timeout=min(remaining_time, 50)) # mention this in my thesis, why I made this decision. It was bc i didn't actually get to the retry
        self.last_metrics = planner.last_metrics

        if plan is not None:
            print("Succeeded with abstract plan:", [
                {"operator_name": a.name, "arguments": [o.name for o in a.parameters]}
                for a in getattr(generator, "_last_abstract_plan", [])
            ])
            return plan

        # ── Step 5: retry with failure-informed heuristics ───────────────────
        print("\n=== Full planning failed — retrying with failure context ===")
        remaining_time = timeout - (time.perf_counter() - start_time)
        if remaining_time <= 0:
            raise TimeoutError("No plan found — no time remaining for retry")

        abstract_actions = getattr(generator, "_last_abstract_plan", [])
        abstract_states = getattr(generator, "_last_abstract_states", [])
        failed_state = tracking_refiner._failed_concrete_state  # pylint: disable=protected-access
        failed_index = tracking_refiner._deepest_failed_index  # pylint: disable=protected-access

        if (failed_state is not None and abstract_actions
                and abstract_states and 0 <= failed_index < len(abstract_actions)):
            failure_context = self._extract_failure_context(
                abstract_actions, failed_state, failed_index, abstract_states
            )
            retry_prompt = prompt_str + self._build_failure_section(failure_context)
        else:
            retry_prompt = prompt_str

        print("\n=== Generating 1 retry heuristic with failure context ===")
        retry_candidates = self._synthesize_all_heuristics(retry_prompt)
        if not retry_candidates:
            raise RuntimeError("LLM returned no valid heuristic code blocks on retry")

        retry_fn = retry_candidates[0][0]
        remaining_time = timeout - (time.perf_counter() - start_time)
        generator = self._make_generator(generate_heuristic_fn=retry_fn)
        planner = self._make_planner(generator)
        plan, _ = planner.run(problem, timeout=remaining_time)
        self.last_metrics = planner.last_metrics

        if plan is None:
            raise TimeoutError("No plan found")
        print("Succeeded with abstract plan:", [
            {"operator_name": a.name, "arguments": [o.name for o in a.parameters]}
            for a in getattr(generator, "_last_abstract_plan", [])
        ])
        return plan

