"""A classifier that uses oracle knowledge to classify abstract plans."""

from typing import Hashable, TypeVar

from bilevel_planning.structs import (
    RelationalAbstractState,
    SesameModels,
)
from kinder.envs.kinematic2d.object_types import CRVRobotType, RectangleType
from relational_structs import GroundOperator
from relational_structs.object_centric_state import ObjectCentricState
from relational_structs.pddl import (
    GroundAtom,
    Predicate,
)

from alphatamp.approaches.feasibility_classifiers.base_feasibility_classifier import (
    BaseFeasibilityClassifier,
)

_S = TypeVar("_S", bound=Hashable)  # abstract state
_A = TypeVar("_A", bound=Hashable)  # abstract action
_X = TypeVar("_X")


class OracleAbstractPlanClassifier(BaseFeasibilityClassifier):
    """A classifier that uses oracle knowledge to classify abstract plans."""

    def __init__(
        self,
        env_models: SesameModels,
    ) -> None:
        # super().__init__(abstract_successor_function, seed)
        self._env_models = env_models

    def validate_plan(
        self,
        x0: ObjectCentricState | _X,
        abstract_states: list[RelationalAbstractState] | list[_S],
        abstract_actions: list[GroundOperator] | list[_A],
    ) -> bool:
        """Classify abstract plans."""

        # Predicates.
        HoldingTgt = Predicate("HoldingTgt", [CRVRobotType, RectangleType])
        HandEmpty = Predicate("HandEmpty", [CRVRobotType])
        Inside = Predicate("Inside", [RectangleType, RectangleType])

        # Objects.
        type_name_to_type = {t.name: t for t in self._env_models.types}
        block_type = type_name_to_type["target_block"]
        obstruction_type = type_name_to_type["rectangle"]
        robot_type = type_name_to_type["crv_robot"]
        region_type = type_name_to_type["target_region"]

        assert isinstance(x0, ObjectCentricState)
        (target_block,) = x0.get_objects(block_type)
        (robot,) = x0.get_objects(robot_type)
        obstructions = x0.get_objects(obstruction_type)
        (target_region,) = x0.get_objects(region_type)

        # remove target block from obstructions
        filtered_obstructions = set()
        for obstruction in obstructions:
            if obstruction.name != "target_block":
                filtered_obstructions.add(obstruction)

        # Oracle abstract plan
        empty_abstract_state_atoms: set[GroundAtom] = set()
        empty_abstract_state_atoms.add(GroundAtom(HandEmpty, [robot]))
        empty_abstract_state_objects = {robot, target_block} | filtered_obstructions
        empty_abstract_state = RelationalAbstractState(
            atoms=empty_abstract_state_atoms, objects=empty_abstract_state_objects
        )

        holding_abstract_state_atoms = set()
        holding_abstract_state_atoms.add(GroundAtom(HoldingTgt, [robot, target_block]))
        holding_abstract_state_objects = {robot, target_block} | filtered_obstructions
        holding_abstract_state = RelationalAbstractState(
            atoms=holding_abstract_state_atoms, objects=holding_abstract_state_objects
        )

        inside_target_abstract_state_atoms = set()
        inside_target_abstract_state_atoms.add(GroundAtom(HandEmpty, [robot]))
        inside_target_abstract_state_atoms.add(
            GroundAtom(Inside, [target_block, target_region])
        )
        inside_target_abstract_state_objects = {
            robot,
            target_block,
        } | filtered_obstructions
        inside_target_abstract_state = RelationalAbstractState(
            atoms=inside_target_abstract_state_atoms,
            objects=inside_target_abstract_state_objects,
        )

        oracle_abstract_state = [
            empty_abstract_state,
            holding_abstract_state,
            inside_target_abstract_state,
        ]

        oracle_abstract_state_ptr = 0
        # Classify plan only looking at state for now
        for plan_abstract_state in abstract_states:
            assert isinstance(plan_abstract_state, RelationalAbstractState)
            plan_atoms, plan_objects = (
                plan_abstract_state.atoms,
                plan_abstract_state.objects,
            )
            oracle_atoms, oracle_objects = (
                oracle_abstract_state[oracle_abstract_state_ptr].atoms,
                oracle_abstract_state[oracle_abstract_state_ptr].objects,
            )

            if plan_atoms != oracle_atoms or plan_objects != oracle_objects:
                return False
            oracle_abstract_state_ptr += 1

        return True
