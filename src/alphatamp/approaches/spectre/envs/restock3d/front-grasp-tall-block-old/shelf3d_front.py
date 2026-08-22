"""Bilevel-planning models for the Shelf3D front-grasp / tall-block variant.

Same task, predicates, operators, abstractor, and goal as the stock kinematic3D
``shelf3d`` env model, but:

* the default env config uses a **tall** block (full height = half a shelf
  section), and
* the skills use the **front-grasp pick + translate-only place** controllers
  (``create_front_lifted_controllers``) instead of the top-down ones.

Portability note: this is a **standalone builder** -- call
``create_bilevel_planning_models(...)`` from this module DIRECTLY. Do NOT route
through ``kinder_bilevel_planning.env_models.create_bilevel_planning_models(
"shelf3d_front", ...)``: that string dispatcher only finds files that live
*inside* the installed ``kinder_bilevel_planning`` package, not in your repo.

When you drop this into your package, change the ``front_grasp_skills`` import
below to your package path (e.g. ``from <your_pkg>.front_grasp_skills import
...``). All ``kinder_*`` / ``bilevel_planning`` / ``relational_structs``
imports stay as-is.
"""

import numpy as np
from bilevel_planning.structs import (
    LiftedSkill,
    RelationalAbstractGoal,
    RelationalAbstractState,
    SesameModels,
)

# --- Change this import to your package path when you vendor the files. ---
from front_grasp_skills import create_front_lifted_controllers
from gymnasium.spaces import Space
from kinder.envs.kinematic3d.object_types import (
    Kinematic3DCuboidType,
    Kinematic3DFixtureType,
)
from kinder.envs.kinematic3d.shelf3d import (
    Kinematic3DRobotType,
    ObjectCentricShelf3DEnv,
    Shelf3DEnvConfig,
    Shelf3DObjectCentricState,
)
from kinder.envs.kinematic3d.utils import Kinematic3DRobotActionSpace
from numpy.typing import NDArray
from relational_structs import (
    GroundAtom,
    LiftedAtom,
    LiftedOperator,
    ObjectCentricState,
    Predicate,
    Variable,
)
from relational_structs.spaces import ObjectCentricBoxSpace, ObjectCentricStateSpace

GRIPPER_OPEN_THRESHOLD = 0.01

# Tall block: 0.05 x 0.05 x 0.127 m (full height = half a shelf section of 0.254 m).
TALL_BLOCK_HALF_EXTENTS = (0.025, 0.025, 0.0635)


def _tall_block_config() -> Shelf3DEnvConfig:
    return Shelf3DEnvConfig(block_half_extents=TALL_BLOCK_HALF_EXTENTS)


def create_bilevel_planning_models(
    observation_space: Space,
    action_space: Space,
    num_objects: int = 1,
    config: Shelf3DEnvConfig | None = None,
) -> SesameModels:
    """Create the front-grasp / tall-block Shelf3D env models.

    ``config`` overrides the internal planning sim's config; defaults to a
    tall-block ``Shelf3DEnvConfig``. It must match the config used to build the
    executable env (so the planner's sim, abstractor, and transition function
    all see the same tall block).
    """
    assert isinstance(observation_space, ObjectCentricBoxSpace)
    assert isinstance(action_space, Kinematic3DRobotActionSpace)

    if config is None:
        config = _tall_block_config()
    sim = ObjectCentricShelf3DEnv(
        num_cubes=num_objects, config=config, allow_state_access=True
    )

    def observation_to_state(o: NDArray[np.float32]) -> ObjectCentricState:
        """Convert the vectors back into (hashable) object-centric states."""
        return observation_space.devectorize(o)

    def transition_fn(
        x: ObjectCentricState,
        u: NDArray[np.float32],
    ) -> ObjectCentricState:
        """Simulate the action."""
        state = x.copy()
        assert isinstance(state, Shelf3DObjectCentricState)
        sim.set_state(state)
        obs, _, _, _, _ = sim.step(u)
        return obs.copy()

    types = {Kinematic3DCuboidType, Kinematic3DFixtureType, Kinematic3DRobotType}
    state_space = ObjectCentricStateSpace(types)

    # Predicates (identical to the stock shelf3d model).
    OnFixture = Predicate("OnFixture", [Kinematic3DCuboidType, Kinematic3DFixtureType])
    OnGround = Predicate("OnGround", [Kinematic3DCuboidType])
    Holding = Predicate("Holding", [Kinematic3DRobotType, Kinematic3DCuboidType])
    HandEmpty = Predicate("HandEmpty", [Kinematic3DRobotType])
    predicates = {OnFixture, OnGround, Holding, HandEmpty}

    def state_abstractor(x: ObjectCentricState) -> RelationalAbstractState:
        """Get the abstract state for the current state."""
        robot = x.get_objects(Kinematic3DRobotType)[0]
        target_objects = x.get_objects(Kinematic3DCuboidType)
        target_fixtures = x.get_objects(Kinematic3DFixtureType)

        atoms: set[GroundAtom] = set()

        assert isinstance(x, Shelf3DObjectCentricState)
        sim.set_state(x)

        on_ground_tol = 0.01
        for target in target_objects:
            z = x.get(target, "pose_z")
            bb_z = x.get(target, "half_extent_z")
            if np.isclose(z, bb_z, atol=on_ground_tol):
                atoms.add(GroundAtom(OnGround, [target]))

        if x.grasped_object is None:
            if x.get(robot, "finger_state") < GRIPPER_OPEN_THRESHOLD:
                atoms.add(GroundAtom(HandEmpty, [robot]))

        for target in target_objects:
            if (
                x.get(target, "pose_z") > 0.3
                and x.get(robot, "finger_state") > GRIPPER_OPEN_THRESHOLD
            ):
                if target.name == x.grasped_object:
                    atoms.add(GroundAtom(Holding, [robot, target]))

        for target in target_objects:
            for fixture in target_fixtures:
                if (
                    np.isclose(
                        x.get(target, "pose_x") - x.get(fixture, "pose_x"),
                        0.0,
                        atol=0.15,
                    )
                    and np.isclose(
                        x.get(target, "pose_y") - x.get(fixture, "pose_y"),
                        0.0,
                        atol=0.25,
                    )
                    and x.get(target, "pose_z") > 0.3
                ):
                    atoms.add(GroundAtom(OnFixture, [target, fixture]))

        objects = {robot} | set(target_objects) | set(target_fixtures)
        return RelationalAbstractState(atoms, objects)

    def goal_deriver(x: ObjectCentricState) -> RelationalAbstractGoal:
        """The goal is to have every cube on the shelf and the hand empty."""
        robot = x.get_objects(Kinematic3DRobotType)[0]
        target_objects = x.get_objects(Kinematic3DCuboidType)
        target_shelf = x.get_objects(Kinematic3DFixtureType)[0]
        atoms: set[GroundAtom] = set()
        for target in target_objects:
            atoms.add(GroundAtom(OnFixture, [target, target_shelf]))
        atoms.add(GroundAtom(HandEmpty, [robot]))
        return RelationalAbstractGoal(atoms, state_abstractor)

    # Operators (identical to the stock shelf3d model).
    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)
    PickOperator = LiftedOperator(
        "Pick",
        [robot, target],
        preconditions={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnGround, [target])},
        add_effects={LiftedAtom(Holding, [robot, target])},
        delete_effects={LiftedAtom(HandEmpty, [robot]), LiftedAtom(OnGround, [target])},
    )

    # Front-grasp controllers.
    lifted_controllers = create_front_lifted_controllers(action_space, sim)
    PickController = lifted_controllers["front_pick"]

    robot = Variable("?robot", Kinematic3DRobotType)
    target = Variable("?target", Kinematic3DCuboidType)
    target_shelf = Variable("?target_shelf", Kinematic3DFixtureType)
    PlaceOperator = LiftedOperator(
        "Place",
        [robot, target, target_shelf],
        preconditions={LiftedAtom(Holding, [robot, target])},
        add_effects={
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(OnFixture, [target, target_shelf]),
        },
        delete_effects={LiftedAtom(Holding, [robot, target])},
    )
    PlaceController = lifted_controllers["front_place"]

    skills = {
        LiftedSkill(PickOperator, PickController),
        LiftedSkill(PlaceOperator, PlaceController),
    }

    return SesameModels(
        observation_space,
        state_space,
        action_space,
        transition_fn,
        types,
        predicates,
        observation_to_state,
        state_abstractor,
        goal_deriver,
        skills,
    )
