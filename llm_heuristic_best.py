
from pyperplan.heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch

def generate_heuristic(task):
    class ClutteredStorage2DHeuristic(Heuristic):
        """
        Summary:
            Estimates the total number of pick and place actions required to bring all required blocks onto the shelf,
            ignoring the order and robot hand content. This corresponds to a h_add/FF-style relaxed plan ignoring delete effects.

        Assumptions:
            - Each Pick or Place action costs 1 step.
            - Only one robot ("HandEmpty" or "Holding" holds at most one block at a time).
            - No mutual exclusion or stacking, i.e., any block not on shelf can be picked whenever the robot is hand-empty.

        Heuristic Initialization:
            - Extracts all (OnShelf block shelf) goals.
            - Caches block and shelf identifiers used in goals.

        Step-By-Step Thinking for Computing Heuristic:
            1. For each block that is required to be OnShelf but is not yet OnShelf,
               estimate the minimal actions needed:
                a. If the block is already held: only Place is needed.
                b. If the robot is already holding a different block: must first Place that block (not counted per block).
                c. If the robot is hand-empty: Pick is available.
            2. For each such block, sum up: Pick + Place = 2 (unless block is held).
            3. If the robot is already holding a block not needed for the goal (i.e., not one that should be OnShelf),
               add a bonus action since the robot must place it first.
            4. Return the total minimal action estimate.
        """

        def __init__(self, task):
            # List of (block, shelf) for each OnShelf goal
            self.onshelf_goals = []
            for goal in task.goals:
                parts = goal[1:-1].split()
                if parts[0] == "OnShelf":
                    self.onshelf_goals.append((parts[1], parts[2]))

        def __call__(self, node) -> float:
            state = node.state
            # Facts as set for quick lookup
            facts = set(state)

            # Find which blocks already satisfy their goal (OnShelf), and which do not
            # Also, for blocks not on shelf, check if held by the robot
            blocks_needed = []
            holding_block = None
            robot = None
            # Find which block, if any, is currently held
            for fact in state:
                parts = fact[1:-1].split()
                if parts[0] == "Holding":
                    holding_block = parts[2]
                    robot = parts[1]
                    break
            # If robot variable is not found, scan for HandEmpty to extract robot name
            if robot is None:
                for fact in state:
                    parts = fact[1:-1].split()
                    if parts[0] == "HandEmpty":
                        robot = parts[1]
                        break

            actions = 0
            blocks_remaining = []
            for block, shelf in self.onshelf_goals:
                if f"(OnShelf {block} {shelf})" in facts:
                    continue  # Already on shelf
                if holding_block == block:
                    # Only need Place to finish
                    actions += 1
                else:
                    # Need to Pick and Place
                    blocks_remaining.append(block)
                    actions += 2

            # If the robot is holding a block that is not needed in the goal (i.e., not among blocks_remaining)
            # we need an extra Place action before we can work on any desired block
            if holding_block is not None and holding_block not in blocks_remaining:
                actions += 1

            return float(actions)
    return ClutteredStorage2DHeuristic(task)
