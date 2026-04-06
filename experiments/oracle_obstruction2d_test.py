"""Oracle policy for the Obstruction2D environment.

Hardcodes a perfect abstract plan classifier (knows the correct operator
sequence) but uses the controllers' own ``sample_parameters`` for parameter
selection, with up to ``max_resamples`` retries per skill — matching the
resample budget that SimFreeParamPolicyApproach gets.

Plan:
  1. For each obstruction (regardless of location): Pick → PlaceOnTable.
  2. PickFromTable(target_block) → PlaceOnTarget(target_block).

This provides an upper-bound baseline: perfect plan selection + random
parameter sampling with the same retry budget as the learned approach.

Usage::

    python -m experiments.oracle_obstruction2d_test
    python -m experiments.oracle_obstruction2d_test --complexity 3 --num-seeds 50
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import gymnasium as gym
import kinder
import numpy as np
from gymnasium.wrappers import RecordVideo
from kinder.envs.kinematic2d.object_types import CRVRobotType, RectangleType
from kinder.envs.kinematic2d.obstruction2d import TargetBlockType, TargetSurfaceType
from kinder.envs.kinematic2d.utils import CRVRobotActionSpace, get_suctioned_objects
from kinder_bilevel_planning.env_models import create_bilevel_planning_models
from kinder_models.kinematic2d.envs.obstruction2d.parameterized_skills import (
    GroundPickController,
    GroundPlaceOnTableController,
    GroundPlaceOnTargetController,
)
from PIL import Image, ImageDraw, ImageFont
from relational_structs import Object, ObjectCentricState
from relational_structs.spaces import ObjectCentricBoxSpace

# ---------------------------------------------------------------------------
# Custom controllers for the oracle policy
# ---------------------------------------------------------------------------


class GroundPickFromCenterController(GroundPickController):
    """Pick controller that always grasps from the center of the block."""

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> float:
        block_width = x.get(self._block, "width")
        return block_width / 2.0


class GroundPlaceAwayFromTargetController(GroundPlaceOnTableController):
    """Place a held block on the table, avoiding the target surface and target block."""

    def sample_parameters(
        self, x: ObjectCentricState, rng: np.random.Generator
    ) -> float:
        margin = 0.1

        # Exclusion zone for the target surface.
        surface = x.get_objects(TargetSurfaceType)[0]
        surf_x = x.get(surface, "x")
        surf_w = x.get(surface, "width")
        exclude_zones: list[tuple[float, float]] = [
            (surf_x - margin, surf_x + surf_w + margin),
        ]

        # Exclusion zone for the target block.
        target_block = x.get_objects(TargetBlockType)[0]
        block_x = x.get(target_block, "x")
        block_w = x.get(target_block, "width")
        exclude_zones.append((block_x - margin, block_x + block_w + margin))

        # Merge overlapping exclusion zones.
        exclude_zones.sort()
        merged: list[tuple[float, float]] = []
        for lo, hi in exclude_zones:
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))

        # Build valid intervals from the gaps between exclusion zones.
        world_min_x = 0.0
        world_max_x = 1.0
        intervals: list[tuple[float, float]] = []
        cursor = world_min_x
        for lo, hi in merged:
            if cursor < lo:
                intervals.append((cursor, lo))
            cursor = max(cursor, hi)
        if cursor < world_max_x:
            intervals.append((cursor, world_max_x))

        if not intervals:
            return rng.uniform(world_min_x, world_max_x)

        # Weight intervals by length for uniform sampling over free space.
        lengths = [hi - lo for lo, hi in intervals]
        total = sum(lengths)
        probs = [l / total for l in lengths]
        idx = rng.choice(len(intervals), p=probs)
        lo, hi = intervals[idx]
        return rng.uniform(lo, hi)


# ---------------------------------------------------------------------------
# Predicate overlay wrapper
# ---------------------------------------------------------------------------


class PredicateOverlayWrapper(gym.Wrapper):  # type: ignore[type-arg]
    """Gymnasium wrapper that overlays active predicates and the current
    oracle action on rendered frames."""

    def __init__(self, env: gym.Env, state_abstractor) -> None:  # type: ignore[type-arg]
        super().__init__(env)
        self._state_abstractor = state_abstractor
        self._current_state: ObjectCentricState | None = None
        self._action_label: str = ""

    def set_state(self, state: ObjectCentricState) -> None:
        self._current_state = state

    def set_action_label(self, label: str) -> None:
        self._action_label = label

    def render(self) -> Any:
        frame = self.env.render()
        if frame is None:
            return frame
        img = Image.fromarray(np.asarray(frame, dtype=np.uint8))
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=14)

        lines: list[tuple[str, tuple[int, int, int]]] = []
        if self._action_label:
            lines.append((self._action_label, (255, 255, 0)))
        if self._current_state is not None:
            abstract_state = self._state_abstractor(self._current_state)
            atom_strs = sorted(str(a) for a in abstract_state.atoms)
            for atom_str in atom_strs:
                lines.append((atom_str, (100, 255, 100)))

        y = 8.0
        for text, color in lines:
            bbox = draw.textbbox((8, y), text, font=font)
            draw.rectangle(
                [bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2],
                fill=(0, 0, 0, 180),
            )
            draw.text((8, y), text, fill=color, font=font)
            y = bbox[3] + 6.0
        return np.array(img)


# ---------------------------------------------------------------------------
# Oracle helpers
# ---------------------------------------------------------------------------


def _obs_to_state(
    obs: np.ndarray, observation_space: ObjectCentricBoxSpace
) -> ObjectCentricState:
    return observation_space.devectorize(obs)


def _get_all_obstructions(num_obstructions: int) -> list[Object]:
    """Return all obstruction objects."""
    return [Object(f"obstruction{i}", RectangleType) for i in range(num_obstructions)]


# ---------------------------------------------------------------------------
# Run a single controller to completion, with resampling on failure
# ---------------------------------------------------------------------------


def _run_skill_with_resamples(
    gym_env,
    controller,
    state: ObjectCentricState,
    observation_space: ObjectCentricBoxSpace,
    rng: np.random.Generator,
    max_resamples: int,
    resamples_used: int,
) -> tuple[ObjectCentricState, bool, bool, int]:
    """Run a controller, resampling parameters on failure up to the budget.

    A pick "succeeds" when the robot is suctioning the target block after the
    controller terminates.  A place "succeeds" when done=True (env goal met).

    Returns (state, done, exhausted, resamples_used).
    """
    robot = Object("robot", CRVRobotType)

    while resamples_used < max_resamples:
        params = controller.sample_parameters(state, rng)
        controller.reset(state, params)

        while not controller.terminated():
            action = controller.step()
            obs, _, done, _, _ = gym_env.step(action)
            state = _obs_to_state(obs, observation_space)
            controller.observe(state)
            if done:
                return state, True, False, resamples_used

        # Check if the skill achieved its sub-goal.
        # For pick: robot should be holding something.
        # For place: the goal check is done=True (handled above).
        # If the controller terminated without done, check if it was a pick
        # (robot should now be suctioning).
        suctioned = get_suctioned_objects(state, robot)
        is_pick = isinstance(controller, GroundPickController)
        if is_pick and len(suctioned) > 0:
            return state, False, False, resamples_used
        if not is_pick:
            # Place controller terminated — block was released.
            # Not done means goal not met, but the skill itself completed.
            return state, False, False, resamples_used

        # Pick failed (nothing suctioned) — resample.
        resamples_used += 1

    return state, False, True, resamples_used


# ---------------------------------------------------------------------------
# Oracle episode
# ---------------------------------------------------------------------------


def _run_controller(
    gym_env,
    controller,
    state: ObjectCentricState,
    observation_space: ObjectCentricBoxSpace,
    rng: np.random.Generator,
    overlay: PredicateOverlayWrapper | None = None,
) -> tuple[ObjectCentricState, bool]:
    """Sample parameters, run one controller to completion.

    Returns (state, done).
    """
    params = controller.sample_parameters(state, rng)
    controller.reset(state, params)
    if overlay is not None:
        overlay.set_action_label(type(controller).__name__)
    while not controller.terminated():
        action = controller.step()
        obs, _, done, _, _ = gym_env.step(action)
        state = _obs_to_state(obs, observation_space)
        controller.observe(state)
        if overlay is not None:
            overlay.set_state(state)
        if done:
            return state, True
    return state, False


def run_oracle_episode(
    gym_env,
    observation_space: ObjectCentricBoxSpace,
    action_space: CRVRobotActionSpace,
    num_obstructions: int,
    seed: int,
    max_resamples: int = 10,
    overlay: PredicateOverlayWrapper | None = None,
) -> bool:
    """Run one episode with the oracle policy. Returns True if goal reached.

    The oracle knows the correct plan but samples random parameters, retrying
    the full pick→place cycle up to ``max_resamples`` times total.
    """
    obs, _ = gym_env.reset(seed=seed)
    state = _obs_to_state(obs, observation_space)
    if overlay is not None:
        overlay.set_state(state)
        overlay.set_action_label("")
    rng = np.random.default_rng(seed)
    robot = Object("robot", CRVRobotType)
    target_block = Object("target_block", TargetBlockType)

    all_obstructions = _get_all_obstructions(num_obstructions)
    obstructions_remaining = list(all_obstructions)
    phase = "clear"  # "clear" obstructions first, then "place" target block

    for _ in range(max_resamples):
        if phase == "clear" and obstructions_remaining:
            # Pick the next obstruction and place it on the table.
            obs_obj = obstructions_remaining[0]
            pick_ctrl = GroundPickFromCenterController([robot, obs_obj], action_space)
            state, done = _run_controller(
                gym_env, pick_ctrl, state, observation_space, rng, overlay
            )
            if done:
                return True

            suctioned = get_suctioned_objects(state, robot)
            if not suctioned:
                continue  # pick failed, retry

            place_ctrl = GroundPlaceAwayFromTargetController([robot, obs_obj], action_space)
            state, done = _run_controller(
                gym_env, place_ctrl, state, observation_space, rng, overlay
            )
            if done:
                return True

            # Obstruction successfully moved, advance to the next one.
            obstructions_remaining.pop(0)
        else:
            # All obstructions cleared — pick target block and place on target.
            phase = "place"
            pick_ctrl = GroundPickFromCenterController([robot, target_block], action_space)
            state, done = _run_controller(
                gym_env, pick_ctrl, state, observation_space, rng, overlay
            )
            if done:
                return True

            suctioned = get_suctioned_objects(state, robot)
            if not suctioned:
                continue

            place_ctrl = GroundPlaceOnTargetController(
                [robot, target_block], action_space
            )
            state, done = _run_controller(
                gym_env, place_ctrl, state, observation_space, rng, overlay
            )
            if done:
                return True

    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_ENV_REGISTRY: dict[str, tuple[str, str, str]] = {
    "clutteredretrieval2d": (
        "kinder/ClutteredRetrieval2D-o{n}-v0",
        "clutteredretrieval2d",
        "num_obstructions",
    ),
    "obstruction2d": (
        "kinder/Obstruction2D-o{n}-v0",
        "obstruction2d",
        "num_obstructions",
    ),
    "dynobstruction2d": (
        "kinder/DynObstruction2D-o{n}-v0",
        "dynobstruction2d",
        "num_obstructions",
    ),
    "clutteredstorage2d": (
        "kinder/ClutteredStorage2D-b{n}-v0",
        "clutteredstorage2d",
        "num_boxes",
    ),
}


def main(
    env_name: str = "obstruction2d",
    complexity: int = 1,
    num_seeds: int = 50,
    start_seed: int = 0,
    max_resamples: int = 10,
    save_video: bool = False,
) -> dict:
    """Run oracle policy on multiple seeds and report success rate."""
    env_id_template, _, _ = _ENV_REGISTRY[env_name]

    kinder.register_all_environments()
    env_id = env_id_template.format(n=complexity)
    gym_env = kinder.make(env_id, render_mode="rgb_array")

    # --- ADDED: Set up the Predicate Overlay ---
    # 1. Get the state abstractor from the bilevel planning models
    models = create_bilevel_planning_models(env_name, gym_env.observation_space, gym_env.action_space, num_obstructions=complexity)
    state_abstractor = models.state_abstractor  # Adjust if your API returns a tuple instead of an object
    
    # 2. Wrap the base environment and keep a reference to the overlay
    overlay_wrapper = PredicateOverlayWrapper(gym_env, state_abstractor)
    gym_env = overlay_wrapper
    # -------------------------------------------

    if save_video:
        video_dir = Path("experiments/oracle_outputs") / env_name / f"c{complexity}"
        video_dir.mkdir(parents=True, exist_ok=True)
        # Note: RecordVideo wraps the overlay so the text is baked into the video frames
        gym_env = RecordVideo(
            gym_env,
            str(video_dir),
            name_prefix="oracle",
            episode_trigger=lambda _: True,
        )

    assert isinstance(gym_env.observation_space, ObjectCentricBoxSpace)
    assert isinstance(gym_env.action_space, CRVRobotActionSpace)

    successes = 0
    failures: list[int] = []

    for seed in range(start_seed, start_seed + num_seeds):
        ok = run_oracle_episode(
            gym_env,
            gym_env.observation_space,
            gym_env.action_space,
            complexity,
            seed,
            max_resamples=max_resamples,
            overlay=overlay_wrapper,  # --- ADDED: Pass the overlay reference ---
        )
        if ok:
            successes += 1
        else:
            failures.append(seed)
        logging.info("Seed %d: %s", seed, "success" if ok else "FAIL")

    rate = successes / num_seeds
    print(f"\nOracle results: {env_name} (complexity={complexity})")
    print(f"  max_resamples={max_resamples}")
    print(f"  {successes}/{num_seeds} = {rate:.1%}")
    if failures:
        print(f"  Failed seeds: {failures}")

    gym_env.close()
    return {"successes": successes, "num_seeds": num_seeds, "rate": rate}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Oracle policy for Obstruction2D")
    parser.add_argument(
        "--env",
        default="obstruction2d",
        help="Environment name (default: obstruction2d)",
    )
    parser.add_argument("--complexity", type=int, default=1)
    parser.add_argument("--num-seeds", type=int, default=50)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-resamples", type=int, default=10)
    parser.add_argument(
        "--save-video", action="store_true", help="Save episode videos to experiments/oracle_outputs"
    )
    args = parser.parse_args()

    main(
        env_name=args.env,
        complexity=args.complexity,
        num_seeds=args.num_seeds,
        start_seed=args.start_seed,
        max_resamples=args.max_resamples,
        save_video=args.save_video,
    )
