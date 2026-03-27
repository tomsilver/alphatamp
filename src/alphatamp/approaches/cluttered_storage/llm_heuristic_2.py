
from pyperplan.heuristics.heuristic_base import Heuristic

def generate_heuristic(task):
    class ClutteredStorage2DHeuristic(Heuristic):
        """
        Summary:
            Provides a sequence-aware greedy heuristic: It simulates a minimal action sequence to bring all blocks
            to their goal shelf, accounting for the need to clear the hand if carrying wrong blocks and
            always choosing the "easiest" next block.

        Assumptions:
            - One robot and all blocks can be picked from either on-shelf/not-on-shelf.
            - The robot can only hold one block at a time.
            - At each step, the planner will always choose to put down a wrong block if necessary to pick a needed one.

        Heuristic Initialization:
            - Extracts the set of necessary (OnShelf block shelf) goals, and list of blocks involved.

        Step-By-Step Thinking for Computing Heuristic:
            1. List all the blocks that must be on-shelf according to goal and are not there yet.
            2. Determine if robot is holding any block; if so, check if it is one of the needed blocks.
               - If not a goal block, must drop it first (counts as an extra move).
            3. For every needed block, except possibly the one currently held, each requires:
               a. If hand empty: Pick (1), Place (1) = 2 actions.
               b. If holding the needed block: Only Place (1).
               c. If hand occupied by wrong block: Place wrong (1), Pick goal (1), Place goal (1) = 3 actions.
            4. Greedily assumes the planner will always pick to minimize extra "drop" actions.

        """
        def __init__(self, task):
            self.goals = task.goals
            # Find all goal blocks
            self.needed_blocks = set()
            for g in self.goals:
                parts = g[1:-1].split()
                if len(parts) == 3 and parts[0] == "OnShelf":
                    self.needed_blocks.add(parts[1])

        def __call__(self, node):
            state = node.state
            on_shelf_now = set()
            not_on_shelf_now = set()
            held_block = None
            hand_empty = False
            for f in state:
                parts = f[1:-1].split()
                if len(parts) == 3 and parts[0] == "OnShelf":
                    on_shelf_now.add(parts[1])
                elif len(parts) == 3 and parts[0] == "NotOnShelf":
                    not_on_shelf_now.add(parts[1])
                elif len(parts) == 3 and parts[0] == "Holding":
                    held_block = parts[2]
                elif parts[0] == "HandEmpty":
                    hand_empty = True
            # Determine missing blocks
            left = [b for b in self.needed_blocks if b not in on_shelf_now]
            if not left:
                return 0
            actions = 0
            # Is robot already holding one of these?
            if held_block is not None:
                if held_block in left:
                    # Place directly, then do the rest
                    actions += 1
                    left.remove(held_block)
                    hand_empty = True
                else:
                    # Must drop wrong block
                    actions += 1
                    hand_empty = True
            # Now for each remaining left block: Pick+Place if hand is empty, or if we greedily put down last placed block.
            actions += 2 * len(left)
            return actions

    return ClutteredStorage2DHeuristic(task)
