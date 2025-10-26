"""N-gram approach for learning operator sequence patterns in TAMP."""

from __future__ import annotations

from typing import TypeAlias, TypeVar

from bilevel_planning.abstract_plan_generators.abstract_plan_generator import (
    AbstractPlanGenerator,
)
from bilevel_planning.abstract_plan_generators.heuristic_search_plan_generator import (
    RelationalHeuristicSearchAbstractPlanGenerator,
)
from bilevel_planning.bilevel_planners.sesame_planner import SesamePlanner
from bilevel_planning.bilevel_planning_graph import BilevelPlanningGraph
from bilevel_planning.structs import (
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
from relational_structs import GroundOperator, LiftedOperator

from alphatamp.approaches.base_approach import BaseApproach
from alphatamp.scoring_utils.batch_ranking import BatchRankingAbstractPlanGenerator

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
Skeleton: TypeAlias = tuple[list[RelationalAbstractState], list[GroundOperator]]
FrozenSkeleton: TypeAlias = tuple[
    tuple[RelationalAbstractState, ...], tuple[GroundOperator, ...]
]
# Union type for operator sequences (both lifted and grounded modes)
OperatorSequence: TypeAlias = tuple[LiftedOperator, ...] | tuple[GroundOperator, ...]


class TrieNode:
    """Node in a trie data structure for tracking failed skeleton prefixes."""

    def __init__(self) -> None:
        """Initialize a trie node."""
        self.children: dict[GroundOperator, TrieNode] = {}
        self.is_failed: bool = False


class FailedSkeletonTrie:
    """Trie data structure for efficient failed skeleton prefix checking.

    Skeletons that failed to refine are added to the trie. The trie is then used to
    prune skeletons that extend any failed prefixes.
    """

    def __init__(self) -> None:
        self.root = TrieNode()

    def add_failed_skeleton(self, skeleton_ops: tuple[GroundOperator, ...]) -> None:
        """Add a failed skeleton to the trie."""
        # Ignore empty skeletons
        if not skeleton_ops:
            return

        current = self.root

        # Traverse or create path for this skeleton
        for op in skeleton_ops:
            if op not in current.children:
                current.children[op] = TrieNode()
            current = current.children[op]

        # Mark the final node as a failed complete skeleton
        current.is_failed = True

    def extends_failed_prefix(self, skeleton_ops: tuple[GroundOperator, ...]) -> bool:
        """Check if skeleton extends any failed skeleton prefix."""
        current = self.root

        for op in skeleton_ops:
            # Check if current position is a failed complete skeleton
            if current.is_failed:
                return True

            # Continue traversing if possible
            if op not in current.children:
                return False
            current = current.children[op]
        return current.is_failed


class NGramApproach(BaseApproach[_O, _X, _U]):
    """Learn operator sequence patterns using n-grams. Each n-gram is tracked with
    success/failure counts across training problems.

    Supports two modes for n-gram extraction:
    1. Lifted: Abstract away object identities (block0, block1)
       and learn which operator sequences succeed/fail.

    2. Grounded mode: Use ground operators with specific objects
    for instance-specific patterns.
    """

    def __init__(
        self,
        env_models: SesameModels,
        seed: int,
        max_abstract_plans: int = 10,
        samples_per_step: int = 10,
        max_skill_horizon: int = 100,
        heuristic_name: str = "hff",
        skeleton_batch_size: int = 100,
        max_ngram_size: int = 3,
        training_planning_timeout: float = 5,
        laplace_smoothing_k_success: float = 0.1,
        laplace_smoothing_k_failure: float = 0.9,
        score_unseen_ngrams: bool = True,
        use_grounded_ngrams: bool = True,
        failure_penalty_mode: bool = True,
    ):
        """Initialize the lifted n-gram approach.

        laplace_smoothing_k_success: Pseudo-count for successes in Laplace smoothing.
        laplace_smoothing_k_failure: Pseudo-count for failures in Laplace smoothing.
        score_unseen_ngrams: Whether to score unseen n-grams or ignore them.
        use_grounded_ngrams: Whether to use grounded operators for n-grams.
        failure_penalty_mode: Whether to use failure-penalty scoring.
            Default: score LOW for skeletons that AVOID failed n-grams.
            Alternative: score HIGH for skeletons that PURSUE successful n-grams.
        """
        super().__init__(env_models, seed)
        self._max_abstract_plans = max_abstract_plans
        self._samples_per_step = samples_per_step
        self._max_skill_horizon = max_skill_horizon
        self._heuristic_name = heuristic_name
        self._skeleton_batch_size = skeleton_batch_size
        self._max_ngram_size = max_ngram_size
        self._training_planning_timeout = training_planning_timeout
        self._laplace_k_success = laplace_smoothing_k_success
        self._laplace_k_failure = laplace_smoothing_k_failure
        self._score_unseen_ngrams = score_unseen_ngrams
        self._use_grounded_ngrams = use_grounded_ngrams
        self._failure_penalty_mode = failure_penalty_mode

        # N-gram statistics: n-gram -> (success_count, failure_count)
        self._ngram_stats: dict[OperatorSequence, tuple[int, int]] = {}
        # Trie for failed skeleton prefixes
        self._failed_skeleton_trie = FailedSkeletonTrie()

        # Counter for online learning: tracks how many failures we've already
        # Reset to 0 at start of each run_planning() call.
        self._num_failures_processed = 0

        # Create the trajectory sampler for refinement
        self._trajectory_sampler = ParameterizedControllerTrajectorySampler(
            controller_generator=RelationalControllerGenerator(self._env_models.skills),
            transition_function=self._env_models.transition_fn,
            state_abstractor=self._env_models.state_abstractor,
            max_trajectory_steps=self._max_skill_horizon,
        )

        # Create the base abstract plan generator (no scoring yet)
        self._base_abstract_plan_generator: AbstractPlanGenerator = (
            RelationalHeuristicSearchAbstractPlanGenerator(
                self._env_models.types,
                self._env_models.predicates,
                self._env_models.operators,
                self._heuristic_name,
                seed=self._seed,
            )
        )

        # Wrap with batch ranking and scoring
        self._batched_abstract_plan_generator: AbstractPlanGenerator = (
            BatchRankingAbstractPlanGenerator(
                self._base_abstract_plan_generator,
                score_fn=self._score_skeleton,
                batch_size=self._skeleton_batch_size,
                seed=self._seed,
            )
        )

        self._abstract_successor_fn = RelationalAbstractSuccessorGenerator(
            self._env_models.operators
        )

        # Finish the planner.
        self._planner = SesamePlanner(
            self._batched_abstract_plan_generator,
            self._trajectory_sampler,
            self._max_abstract_plans,
            self._samples_per_step,
            self._abstract_successor_fn,
            self._env_models.state_abstractor,
            seed=self._seed,
        )

        # Use the same refiner at training time that we will use at test time. Do this
        # by stealing the refiner from the planner.
        self._refiner = self._planner._refiner  # pylint: disable=protected-access

    def _extract_operator_sequence(self, skeleton: Skeleton) -> OperatorSequence:
        """Extract operator sequence from skeleton (lifted or grounded based on
        config)."""
        _, operators = skeleton

        # Grounded mode
        if self._use_grounded_ngrams:
            return tuple(operators)

        # Lifted mode: op.parent gets the lifted operator.
        return tuple(op.parent for op in operators if op.parent is not None)

    def _extract_ngrams(
        self, operator_sequence: OperatorSequence
    ) -> list[OperatorSequence]:
        """Extract all n-grams from an operator sequence.

        Works with both lifted and grounded operator sequences.
        """
        ngrams = []

        # Extract n-grams of each length
        for n in range(1, min(self._max_ngram_size + 1, len(operator_sequence) + 1)):
            # Slide window of size n over the sequence
            for i in range(len(operator_sequence) - n + 1):
                ngram = operator_sequence[i : i + n]
                ngrams.append(ngram)

        return ngrams

    def _update_ngram_stats(
        self, operator_sequence: OperatorSequence, success: bool
    ) -> None:
        """Update n-gram statistics with a new skeleton result."""
        ngrams = self._extract_ngrams(operator_sequence)

        for ngram in ngrams:
            if ngram not in self._ngram_stats:
                self._ngram_stats[ngram] = (0, 0)  # (success_count, fail_count)

            success_count, fail_count = self._ngram_stats[ngram]

            if success:
                self._ngram_stats[ngram] = (success_count + 1, fail_count)
            else:
                self._ngram_stats[ngram] = (success_count, fail_count + 1)

    def _compute_smoothed_success_rate(self, ngram: OperatorSequence) -> float:
        """Compute Laplace-smoothed success rate for an n-gram.

        Formula: (success_count + k_success) / (total_count + k_success + k_failure)
        """
        if ngram in self._ngram_stats:
            success_count, fail_count = self._ngram_stats[ngram]
        else:
            # Unseen n-gram: 0 successes, 0 failures
            success_count, fail_count = 0, 0

        k_success = self._laplace_k_success
        k_failure = self._laplace_k_failure
        total_count = success_count + fail_count

        # Asymmetric smoothing: different pseudo-counts for success vs failure
        smoothed_rate = (success_count + k_success) / (
            total_count + k_success + k_failure
        )

        return smoothed_rate

    def _train(self, problem: PlanningProblem[_X, _U]) -> None:
        """Train on a single problem to learn n-gram patterns."""
        x0 = problem.initial_state
        s0 = self._env_models.state_abstractor(x0)

        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_state_node(x0)
        bpg.add_abstract_state_node(s0)
        bpg.add_state_abstractor_edge(x0, s0)

        # For debugging purposes:
        skeleton_count = 0
        success_count = 0

        # Generate skeletons from base generator
        for skeleton in self._base_abstract_plan_generator(
            x0,
            s0,
            problem.goal,
            self._training_planning_timeout,
            bpg,
        ):
            skeleton_count += 1

            # Extract operator sequence (lifted or grounded based on config)
            operator_sequence = self._extract_operator_sequence(skeleton)

            # Try to refine the skeleton
            plan = self._refiner(
                x0, skeleton[0], skeleton[1], self._training_planning_timeout, bpg
            )
            success = plan is not None

            if success:
                success_count += 1
            else:
                # Add failed skeleton to prefix trie (ground operators)
                ground_prefix = tuple(skeleton[1])
                self._failed_skeleton_trie.add_failed_skeleton(ground_prefix)

            # Update n-gram statistics
            self._update_ngram_stats(operator_sequence, success)

    def _run_planning(
        self, problem: PlanningProblem[_X, _U], timeout: float
    ) -> Plan[_X, _U]:
        """Run planning using learned n-gram patterns to score skeletons."""
        # Reset online learning counter for new problem
        self._num_failures_processed = 0

        plan, _ = self._planner.run(problem, timeout=timeout)

        if plan is None:
            raise TimeoutError("No plan found")

        return plan

    def _score_skeleton(
        self, skeleton: Skeleton, failed_skeletons: list[Skeleton]
    ) -> float:
        """Score a skeleton based on learned n-gram patterns.

        Two scoring modes (based on config):
        1. SUCCESS-PATTERN MODE (failure_penalty_mode=False, DEFAULT):
            Score HIGH if skeleton contains high-success-rate n-grams
        2. FAILURE-PENALTY MODE (failure_penalty_mode=True):
            Score HIGH if skeleton AVOIDS high-failure-rate n-grams
        """
        # Check prefix trie for hard pruning
        ground_ops = tuple(skeleton[1])
        if self._failed_skeleton_trie.extends_failed_prefix(ground_ops):
            return float("-inf")

        # Online learning: Update n-gram stats with NEW failures only
        # Prevent O(n^2) duplicate counting during batch re-scoring
        if len(failed_skeletons) > self._num_failures_processed:
            # Extract only the new failures
            new_failures = failed_skeletons[self._num_failures_processed :]

            for failed_skel in new_failures:
                # Add to prefix trie (ground operators)
                ground_prefix = tuple(failed_skel[1])
                self._failed_skeleton_trie.add_failed_skeleton(ground_prefix)

                operator_seq = self._extract_operator_sequence(failed_skel)
                self._update_ngram_stats(operator_seq, success=False)

            # Update counter track that we have processed these failures
            self._num_failures_processed = len(failed_skeletons)

        # Extract n-grams
        operator_sequence = self._extract_operator_sequence(skeleton)
        ngrams = self._extract_ngrams(operator_sequence)

        # Score based on mode
        if self._failure_penalty_mode:
            return self._score_failure_mode(ngrams)
        return self._score_success_mode(ngrams)

    def _score_success_mode(self, ngrams: list[OperatorSequence]) -> float:
        """Score skeleton by pursuing successful n-gram patterns.

        Sum up success rates of all n-grams, weighted by length. Longer n-grams (more
        specific) get more weight.
        """
        total_score = 0.0
        total_weight = 0.0

        # Score each n-gram using smoothed success rate
        for ngram in ngrams:
            # Weight by n-gram length
            # Longer (more specific) patterns more influential
            weight = len(ngram)

            # Check if we should score this n-gram
            is_seen = ngram in self._ngram_stats
            should_score = is_seen or self._score_unseen_ngrams

            if should_score:
                # Accumulate weighted success rate and total weight
                success_rate = self._compute_smoothed_success_rate(ngram)
                total_score += weight * success_rate
                total_weight += weight

        # Return weighted average
        if total_weight > 0:
            return total_score / total_weight
        # Fallback: assign very low score if no n-grams to score
        return float("-inf")

    def _score_failure_mode(self, ngrams: list[OperatorSequence]) -> float:
        """
        Score skeleton by avoiding failure n-gram patterns.
        Stronger penalties for:
        Longer n-grams, higher failure rates/more observed failures

        Doesn't penalize n-grams without strong failure evidence.
        """
        total_penalty = 0.0

        for ngram in ngrams:
            if ngram not in self._ngram_stats:
                # Optimistic assumption for unseen n-grams
                # Unseen n-gram: No failure evidence, no penalty
                continue

            success_count, fail_count = self._ngram_stats[ngram]

            # Only penalize if we have STRONG evidence of failure
            if fail_count <= success_count:
                # This n-gram succeeds at least as often as it fails
                # No penalty (neutral or positive pattern)
                continue

            # Compute failure rate and observation confidence
            total_observations = success_count + fail_count
            failure_rate = fail_count / total_observations

            # Penalty formula: weight * failure_rate * observation_confidence

            # Give more weight to longer n-grams
            weight = len(ngram)
            # Ramp up confidence weight with more observations
            observation_confidence = min(1.0, total_observations / 10.0)

            penalty = weight * failure_rate * observation_confidence
            total_penalty += penalty

        # Return negative penalty as score
        return -total_penalty

    def get_ngram_summary(self) -> dict[tuple[str, ...], dict[str, float]]:
        """Get summary of learned n-gram statistics for analysis/debugging.

        Converts operator n-grams to string representations for readability. Works with
        both lifted and grounded n-grams.
        """
        summary = {}

        for ngram, (success_count, fail_count) in self._ngram_stats.items():
            total_count = success_count + fail_count
            success_rate = success_count / total_count if total_count > 0 else 0.0

            # Convert operator tuple to string tuple
            if self._use_grounded_ngrams:
                # Grounded mode: show full operator
                ngram_names = tuple(op.short_str for op in ngram)
            else:
                # Lifted mode: show just operator names
                ngram_names = tuple(op.name for op in ngram)

            summary[ngram_names] = {
                "success_count": success_count,
                "fail_count": fail_count,
                "success_rate": success_rate,
                "total_count": total_count,
            }

        return summary
