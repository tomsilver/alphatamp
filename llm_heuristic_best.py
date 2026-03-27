
from pyperplan.heuristics.heuristic_base import Heuristic

def generate_heuristic(task):
    """
    Heuristic 1: Minimum Grasp-Place Pairs + Hand Occupancy

    This heuristic computes, for all blocks required to be on the shelf but not currently on it,
    the number of pick-place "moves" needed, with a penalty if the robot's hand is currently 
    not empty or is holding a block that's not immediately placeable (i.e., it's not a goal block 
    or not needed on the shelf).

    Provides goal-awareness about which blocks matter, and a bias toward finishing the current 
    held block's placement before initiating a new pick.
    """
    class ClutteredStorage2DHeuristic(Heuristic):
        """
        Summary:
            Counts how many blocks need to be moved onto the shelf (compared to the goal), 
            multiplied by two (pick+place), plus an extra step if the robot's hand is not 
            immediately able to continue.

        Assumptions:
            - Only one robot, one shelf, unary manipulation.
            - Each block not already on the shelf but required in the goal requires one pick and one place.
            - The Pick and Place actions are always available unless the robot is holding something.
            - The hardest point in execution occurs if the robot's hand is busy with a "wrong" block.

        Heuristic Initialization:
            - Parse facts to find the unique robot and shelf.
            - For each "OnShelf block shelf" in the goal, note required blocks.
            - No static facts are needed.

        Step-By-Step Thinking for Computing Heuristic:
            1. For each block required on the shelf in the goal, check if it is already there.
            2. For each block missing, add 2 (pick+place).
            3. If the robot is holding a block:
                - If it's a needed block and it's not on the shelf, just count an extra place (as it's partway through a move).
                - If it's not needed, count an additional penalty (+2), as that block must be dropped somewhere before continuing.
            4. If the robot is not holding a block, standard cost.
            5. If the state is a goal, heuristic is 0.
        """
        def __init__(self, task):
            self.goal_blocks = set()
            self.shelf = None
            self.robot = None
            for fact in task.facts:
                parts = fact[1:-1].split()
                if parts[0] == "OnShelf":
                    self.shelf = parts[2]
                elif parts[0] == "HandEmpty":
                    self.robot = parts[1]
            for goal in task.goals:
                parts = goal[1:-1].split()
                if parts[0] == "OnShelf":
                    self.goal_blocks.add(parts[1])

        def __call__(self, node) -> float:
            state = node.state
            # Check for goal:
            if all(goal in state for goal in task.goals):
                return 0
            on_shelf = set()
            holding = None
            hand_empty = False
            for fact in state:
                parts = fact[1:-1].split()
                if parts[0] == "OnShelf":
                    on_shelf.add(parts[1])
                elif parts[0] == "Holding":
                    holding = parts[2]
                elif parts[0] == "HandEmpty":
                    hand_empty = True

            needed = self.goal_blocks - on_shelf
            cost = 2 * len(needed)

            # Hand logic:
            if holding:
                # If holding a needed block not yet placed, only need Place
                if holding in needed:
                    # We'll save one pick (already holding), only need 'place' for that block
                    # So Deduct 1 for the pick that isn't needed
                    cost -= 1
                else:
                    # Holding something not needed: must drop or place it somewhere (at least 2 steps)
                    cost += 2
                # Encourage planner to finish with currently held goal block
            return cost

    return ClutteredStorage2DHeuristic(task)
