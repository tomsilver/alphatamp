from typing import TypeAlias, TypeVar, List, Dict, Tuple, Optional, Set
import numpy as np
import math
import time

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
from relational_structs import GroundOperator

from alphatamp.approaches.base_approach import BaseApproach
from alphatamp.scoring_utils.batch_ranking import BatchRankingAbstractPlanGenerator

_O = TypeVar("_O")  # observation
_X = TypeVar("_X")  # state
_U = TypeVar("_U")  # action
Skeleton: TypeAlias = tuple[list[RelationalAbstractState], list[GroundOperator]]
FrozenSkeleton: TypeAlias = tuple[
    tuple[RelationalAbstractState, ...], tuple[GroundOperator, ...]
]


class BoxApproach(BaseApproach[_O, _X, _U]):
    """An approach that implements the BOX algorithm for skeleton selection."""

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
        exploration_constant: float = math.sqrt(2),
    ):
        super().__init__(env_models, seed)
        self._max_abstract_plans = max_abstract_plans
        self._samples_per_step = samples_per_step
        self._max_skill_horizon = max_skill_horizon
        self._heuristic_name = heuristic_name
        self._skeleton_batch_size = skeleton_batch_size
        self._num_training_skeletons_per_problem = num_training_skeletons_per_problem
        self._training_planning_timeout = training_planning_timeout
        self._exploration_constant = exploration_constant # c parameter in UCB

        # Create the trajectory sampler for refinement
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

        self._batched_abstract_plan_generator: AbstractPlanGenerator = (
            BatchRankingAbstractPlanGenerator(
                self._base_abstract_plan_generator,
                score_fn=self._score_skeleton,
                batch_size=self._skeleton_batch_size,
                seed=self._seed,
            )
        )

        # Create the abstract successor function (not really used).
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

        # Use the same refiner at training time that we will use at test time.
        self._refiner = self._planner._refiner  # pylint: disable=protected-access

        # Store training data.
        # List of dicts, where each dict maps FrozenSkeleton -> success (bool)
        self._data: list[dict[FrozenSkeleton, bool]] = []
        
        # BOX Model parameters (will init once after training)
        self._skeletons_vocab: List[FrozenSkeleton] = []
        self._skeleton_to_idx: Dict[FrozenSkeleton, int] = {}
        self._prior_mu: Optional[np.ndarray] = None
        self._prior_sigma: Optional[np.ndarray] = None
        self._model_built = False # ensure we don't rebuild multiple times

    def _train(self, problem: PlanningProblem[_X, _U]) -> None:
        """Collect training data by generating skeletons and checking refinability."""
        x0 = problem.initial_state
        s0 = self._env_models.state_abstractor(x0)

        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_state_node(x0)
        bpg.add_abstract_state_node(s0)
        bpg.add_state_abstractor_edge(x0, s0)

        problem_data: dict[FrozenSkeleton, bool] = {}

        # Generate a fixed number of skeletons for this training problem
        gen = self._base_abstract_plan_generator(
            x0,
            s0,
            problem.goal,
            self._training_planning_timeout,
            bpg,
        )

        count = 0
        for skeleton in gen:
            # Only generate up to _num_training_skeletons_per_problem skeletons
            if count >= self._num_training_skeletons_per_problem:
                break
            
            # Attempt refinement
            plan = self._refiner(
                x0, skeleton[0], skeleton[1], self._training_planning_timeout, bpg
            )
            label = plan is not None
            frozen_skeleton = (tuple(skeleton[0]), tuple(skeleton[1]))
            # TODO: try other scores (e.g. refinement cost, ngram score, etc)
            problem_data[frozen_skeleton] = label # store success/failure
            count += 1

        self._data.append(problem_data)

    def _build_box_model(self) -> None:
        """Builds the prior mu and sigma from collected training data."""
        # Don't rebuild if already built
        if self._model_built:
            return

        # Identify fixed set of skeletons from training
        # Take the union of all skeletons seen during training as analogy to "constraints" from BOX paper
        all_skeletons: Set[FrozenSkeleton] = set()
        for problem_data in self._data:
            all_skeletons.update(problem_data.keys())
        
        self._skeletons_vocab = sorted(list(all_skeletons), key=lambda s: str(s))
        self._skeleton_to_idx = {s: i for i, s in enumerate(self._skeletons_vocab)}
        
        N = len(self._data)
        M = len(self._skeletons_vocab)
        
        if M == 0 or N == 0:
            # Fallback if no data
            self._prior_mu = np.zeros(M)
            self._prior_sigma = np.eye(M)
            self._model_built = True
            return

        # Construct Score Matrix D (N x M)
        # NOTE: we currently assume that if a problem was not generated for a problem it stays 0.
        D = np.zeros((N, M))
        for i, problem_data in enumerate(self._data):
            for skel, success in problem_data.items():
                j = self._skeleton_to_idx[skel]
                D[i, j] = 1.0 if success else 0.0
        
        # Mean vector
        self._prior_mu = np.mean(D, axis=0)
        
        # Covariance matrix
        self._prior_sigma = np.cov(D, rowvar=False)

        self._model_built = True

    def _run_planning(
        self, problem: PlanningProblem[_X, _U], timeout: float
    ) -> Plan[_X, _U]:
        
        start_time = time.time()  # Track start time

        # Ensure model is built
        self._build_box_model()
        
        assert self._prior_mu is not None
        assert self._prior_sigma is not None
        
        # Candidates are the fixed vocabulary
        num_candidates = len(self._skeletons_vocab)
        all_indices = np.arange(num_candidates)
        
        # Track observed data
        observed_indices: List[int] = []
        observed_scores: List[float] = []
        
        # Track tried skeletons to avoid repeating them (e.g. during fallback process)
        tried_skeletons: Set[FrozenSkeleton] = set()
        
        # Setup for refinement
        x0 = problem.initial_state
        s0 = self._env_models.state_abstractor(x0)
        bpg: BilevelPlanningGraph = BilevelPlanningGraph()
        bpg.add_state_node(x0)
        bpg.add_abstract_state_node(s0)
        bpg.add_state_abstractor_edge(x0, s0)

        # Candidates from fixed set still remain
        if num_candidates > 0:
            for _ in range(num_candidates):
                # Check global timeout
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    break
                
                # Find untried candidates
                untried_mask = np.ones(num_candidates, dtype=bool)
                untried_mask[observed_indices] = False
                untried_indices = all_indices[untried_mask]
                
                if len(untried_indices) == 0:
                    break

                # Compute mu and sigma for untried candidates
                if len(observed_indices) == 0:
                    mu_t = self._prior_mu[untried_indices]
                    sigma_t = self._prior_sigma[np.ix_(untried_indices, untried_indices)]
                else:
                    # Equation (3) from BOX paper
                    idx_1 = untried_indices
                    idx_2 = observed_indices
                    
                    mu_1 = self._prior_mu[idx_1]
                    mu_2 = self._prior_mu[idx_2]
                    
                    sigma_11 = self._prior_sigma[np.ix_(idx_1, idx_1)]
                    sigma_12 = self._prior_sigma[np.ix_(idx_1, idx_2)]
                    sigma_21 = self._prior_sigma[np.ix_(idx_2, idx_1)]
                    sigma_22 = self._prior_sigma[np.ix_(idx_2, idx_2)]
                    
                    # ensure sigma_22 is invertible
                    sigma_22_reg = sigma_22 + 1e-6 * np.eye(len(idx_2))
                    
                    # Compute term = Sigma_12 * Sigma_22^-1
                    # Can do by solve Sigma_22 * X = Sigma_21
                    try:
                        term_T = np.linalg.solve(sigma_22_reg, sigma_21)
                        term = term_T.T
                    except np.linalg.LinAlgError:
                        # Fallback to pseudo-inverse if cannot solve
                        sigma_22_inv = np.linalg.pinv(sigma_22_reg)
                        term = sigma_12 @ sigma_22_inv

                    # Updated Mean: mu_1 + term * (J_observed - mu_2)
                    J_observed = np.array(observed_scores)
                    mu_t = mu_1 + term @ (J_observed - mu_2)
                    
                    # Updated Covariance: Sigma_11 - term * Sigma_21
                    sigma_t = sigma_11 - term @ sigma_21

                #  UCB step
                sigma_diag = np.diag(sigma_t)
                sigma_diag = np.maximum(sigma_diag, 0.0) # safety: avoid negative due to numerical issues
                
                ucb_scores = mu_t + self._exploration_constant * np.sqrt(sigma_diag)
                
                best_idx = untried_indices[np.argmax(ucb_scores)]
                
                # Execute and Evaluate
                skeleton = self._skeletons_vocab[best_idx]
                tried_skeletons.add(skeleton)
                
                skel_states = list(skeleton[0])
                skel_ops = list(skeleton[1])
                
                # Recalculate remaining time
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    break
                remaining = timeout - elapsed

                plan = self._refiner(
                    x0, skel_states, skel_ops, remaining, bpg
                )
                
                if plan is not None:
                    return plan
                
                # Didn't find a plan with this skeleton
                # Update observations with failure (score 0.0)
                observed_indices.append(best_idx)
                observed_scores.append(0.0)

        # Fallback Loop
        # Handles zero-shot cases or when all candidates exhausted
        elapsed = time.time() - start_time
        if elapsed < timeout:
            # Recalculate remaining time
            remaining = timeout - elapsed
            gen = self._base_abstract_plan_generator(
                x0, s0, problem.goal, remaining, bpg
            )
            for skeleton in gen:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    break
                remaining = timeout - elapsed
                
                frozen = (tuple(skeleton[0]), tuple(skeleton[1]))
                if frozen in tried_skeletons:
                    continue
                
                tried_skeletons.add(frozen)
                
                plan = self._refiner(
                    x0, skeleton[0], skeleton[1], remaining, bpg
                )
                if plan is not None:
                    return plan

        return None
        # raise RuntimeError("Failed to find a plan with BoxApproach")

    def run_planning_filtered(
        self, init_obs: _O, timeout: float
    ) -> Plan[_X, _U]:
        """Use the base generator but filters out skeletons that
        have consistently failed in training."""
        
        start_time = time.time()
        problem = self._observation_to_planning_problem(init_obs)
        
        # Identify skeletons that have been tried but never succeeded
        tried_skeletons: Set[FrozenSkeleton] = set()
        successful_skeletons: Set[FrozenSkeleton] = set()
        
        for problem_data in self._data:
            for skel, success in problem_data.items():
                tried_skeletons.add(skel)
                if success:
                    successful_skeletons.add(skel)
        
        # Skeletons that were tried but never succeeded
        always_failed_skeletons = tried_skeletons - successful_skeletons
        
        x0 = problem.initial_state
        s0 = self._env_models.state_abstractor(x0)
        bpg = BilevelPlanningGraph()
        bpg.add_state_node(x0)
        bpg.add_abstract_state_node(s0)
        bpg.add_state_abstractor_edge(x0, s0)

        elapsed = time.time() - start_time
        if elapsed >= timeout:
            return None
        remaining = timeout - elapsed

        gen = self._base_abstract_plan_generator(
            x0, s0, problem.goal, remaining, bpg
        )
        
        for skeleton in gen:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                break
            remaining = timeout - elapsed

            frozen = (tuple(skeleton[0]), tuple(skeleton[1]))
            
            if frozen in always_failed_skeletons:
                continue
                
            plan = self._refiner(
                x0, skeleton[0], skeleton[1], remaining, bpg
            )
            
            if plan is not None:
                return plan
        
        return None

    def run_planning_successful_first(
        self, init_obs: _O, timeout: float
    ) -> Plan[_X, _U]:
        """Use previously successful skeletons first, then fall back to the generator."""
        
        start_time = time.time()
        problem = self._observation_to_planning_problem(init_obs)
        x0 = problem.initial_state
        s0 = self._env_models.state_abstractor(x0)
        bpg = BilevelPlanningGraph()
        bpg.add_state_node(x0)
        bpg.add_abstract_state_node(s0)
        bpg.add_state_abstractor_edge(x0, s0)

        # Collect successful skeletons from training
        successful_skeletons: List[FrozenSkeleton] = []
        seen_successful: Set[FrozenSkeleton] = set()
        
        for problem_data in self._data:
            for skel, success in problem_data.items():
                if success and skel not in seen_successful:
                    successful_skeletons.append(skel)
                    seen_successful.add(skel)
        
        tried_skeletons: Set[FrozenSkeleton] = set()

        # Try successful skeletons first
        for skeleton in successful_skeletons:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                break
            remaining = timeout - elapsed

            tried_skeletons.add(skeleton)
            
            # Convert FrozenSkeleton back to lists
            skel_states = list(skeleton[0])
            skel_ops = list(skeleton[1])
            
            plan = self._refiner(
                x0, skel_states, skel_ops, remaining, bpg
            )
            
            if plan is not None:
                return plan

        # 3. Fallback to generator
        print("[SuccessfulFirst] Fallback to generator.")
        
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            return None
        
        # recalculate remaining time
        remaining = timeout - elapsed

        gen = self._base_abstract_plan_generator(
            x0, s0, problem.goal, remaining, bpg
        )
        
        # TODO: could refactor to avoid code duplication
        for skeleton in gen:
            elapsed = time.time() - start_time
            if elapsed >= timeout:
                break
            remaining = timeout - elapsed

            frozen = (tuple(skeleton[0]), tuple(skeleton[1]))
            
            if frozen in tried_skeletons:
                continue
            
            tried_skeletons.add(frozen)
            
            plan = self._refiner(
                x0, skeleton[0], skeleton[1], remaining, bpg
            )
            
            if plan is not None:
                return plan
                
        return None

    def _score_skeleton(
        self, skeleton: Skeleton, failed_skeletons: list[Skeleton]
    ) -> float:
        """Score skeletons.
        
        In this BOX implementation, scoring is handled dynamically in _run_planning.
        """
        if not self._model_built:
            return 0.0
            
        frozen = (tuple(skeleton[0]), tuple(skeleton[1]))
        if frozen in self._skeleton_to_idx:
            idx = self._skeleton_to_idx[frozen]
            return float(self._prior_mu[idx])
        
        return 0.0
