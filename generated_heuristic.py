
from pyperplan.heuristics.heuristic_base import Heuristic

def generate_heuristic(task):
    class ClutteredStorage2DHeuristic(Heuristic):
        """
        Summary:
        --------
        A domain-dependent heuristic for the ClutteredStorage2D domain that estimates
        the number of actions required to reach the goal. The heuristic penalizes states
        with unwanted ("clutter") blocks on the shelf and with goal blocks not yet properly
        placed, and strongly prefers to first clear non-goal blocks from the shelf before
        placing goal blocks.

        Assumptions:
        -----------
        - There is only one robot and one shelf per problem instance.
        - Each block can be on the shelf, not on the shelf, or held by the robot.
        - The robot can only hold one block at a time (enforced by domain).
        - The only meaningful 'clutter' in this domain arises from "OnShelf"
          facts for blocks that are not goal blocks for that shelf.

        Heuristic Initialization:
        ------------------------
        - Extract all (block, shelf) pairs that are goal conditions (i.e., (OnShelf block shelf) in task.goals).
        - Store the set of blocks that need to be on the shelf in the goal.

        Step-By-Step Thinking for Computing Heuristic:
        ---------------------------------------------
        1. Extract the current set of (block, shelf) pairs where blocks are on the shelf in the given state.
        2. Identify all blocks that are required to be on the shelf in the goal.
        3. Phase 1: For each block that is currently on the shelf but should NOT be (i.e., not a goal (block, shelf)),
           add a penalty for the need to first remove this non-goal block ("clutter") by picking it up and placing it not on the shelf (2 actions each).
        4. Phase 2: For each goal (block, shelf) that is NOT yet satisfied in the current state,
           add an action penalty for picking that block up and placing it on the shelf (2 actions each).
        5. Increase the weight of Phase 1 (by a multiplier, e.g., 3) so that the heuristic strongly
           prefers to first clear all non-goal blocks from the shelf before placing goal blocks.
        6. Heuristic value is zero if all "OnShelf" goal facts are true in the current state.

        This provides a non-admissible but highly effective heuristic that aggressively guides search to clear the shelf
        before attempting to satisfy goal placements.
        """

        def __init__(self, task):
            # Find all goal (block, shelf) pairs that must be OnShelf in the goal
            self.goal_onshelf = set()
            for fact in task.goals:
                parts = fact[1:-1].split()
                if parts[0] == "OnShelf":
                    self.goal_onshelf.add((parts[1], parts[2]))
            # For fast lookup, collect all blocks that must be on shelf
            self.goal_blocks = {block for (block, _) in self.goal_onshelf}

        def __call__(self, node) -> float:
            state = node.state

            # Extract all (block, shelf) that are currently OnShelf
            onshelf = set()
            for fact in state:
                parts = fact[1:-1].split()
                if parts[0] == "OnShelf":
                    onshelf.add((parts[1], parts[2]))

            # Goal test: if all goal OnShelf conditions are met, return 0
            if self.goal_onshelf <= onshelf:
                return 0.0

            # Phase 1: Non-goal blocks on shelf (clutter) that need to be removed
            non_goal_on_shelf = [(b, s) for (b, s) in onshelf if (b, s) not in self.goal_onshelf]
            phase1_cost = 2 * len(non_goal_on_shelf)  # each: Pick + PlaceNotOnShelf

            # Phase 2: Goal OnShelf that are not yet satisfied
            goal_not_placed = [(b, s) for (b, s) in self.goal_onshelf if (b, s) not in onshelf]
            phase2_cost = 2 * len(goal_not_placed)  # each: Pick + PlaceOnShelf

            # Strongly prefer states where the shelf is cleared first
            return float(3 * phase1_cost + phase2_cost)

    return ClutteredStorage2DHeuristic(task)
