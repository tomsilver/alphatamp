
from pyperplan.heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch

def generate_heuristic(task):
    class ClutteredStorage2DHeuristic(Heuristic):
        """
        Summary:
            Estimates the number of actions required for all blocks to be placed on the shelf
            as specified in the goal, given the domain structure with pick/place and shelf/not-on-shelf predicates.
            This heuristic assumes one robot and accounts for whether the robot is holding a block.

        Assumptions:
            - There is a single robot and a single shelf.
            - Each block is handled one at a time by the robot.
            - The robot can only pick up or place one block at a time.
            - The only actions are pick (from on-shelf/not-on-shelf) and place (on-shelf/not-on-shelf).
            - Predicate strings are provided without nested structure, in consistent ordering.

        Heuristic Initialization:
            - The heuristic identifies which (OnShelf block shelf) facts are required in the goal.
            - For efficiency, it extracts all (OnShelf block shelf) goals into a set.
            - The static facts are not used as there are no static predicates in clutteredstorage2d.

        Step-By-Step Thinking for Computing Heuristic:
            1. Count the number of (OnShelf block shelf) goals not satisfied in the current state.
               These are blocks that still need to be placed on the shelf.
            2. For each unsatisfied (OnShelf block shelf), determine where the corresponding block is:
                a. If it's held: Only needs to be placed (1 action).
                b. If it's NotOnShelf: Robot must be hand-empty, pick the block (1), then place (1).
                c. If another block is already held: First needs to put down the block (if not the target), then proceed as above.
            3. If the robot is currently holding a block that is not needed on the shelf (i.e., not an unsatisfied goal block),
               it must first put it down before proceeding with the next pick.
            4. The heuristic thus estimates:
               - For each missing goal OnShelf, needs (pick + place) unless already held (just place).
               - If holding an "extra" block, add a "place not-on-shelf" action to clear the hand.
            5. Returns the sum as the heuristic estimate.

        """
        def __init__(self, task):
            # Collect relevant goal facts: all (OnShelf block shelf) entries
            self.on_shelf_goals = set()
            for fact in task.goals:
                parts = fact[1:-1].split()
                if len(parts) == 3 and parts[0] == "OnShelf":
                    self.on_shelf_goals.add((parts[1], parts[2]))  # (block, shelf)
            # Get blocks and shelf names
            self.all_blocks = {block for (block, shelf) in self.on_shelf_goals}
            self.shelves = {shelf for (block, shelf) in self.on_shelf_goals}

        def __call__(self, node):
            state = node.state
            # Set up current block locations/status
            on_shelf = set()
            not_on_shelf = set()
            held_blocks = set()
            hand_empty = False
            for f in state:
                parts = f[1:-1].split()
                if parts[0] == "OnShelf":
                    on_shelf.add((parts[1], parts[2]))
                elif parts[0] == "NotOnShelf":
                    not_on_shelf.add((parts[1], parts[2]))
                elif parts[0] == "Holding":
                    held_blocks.add(parts[2])
                elif parts[0] == "HandEmpty":
                    hand_empty = True
            # Early exit if all goals already reached
            if self.on_shelf_goals <= on_shelf:
                return 0

            cost = 0
            needed = [block for (block, shelf) in self.on_shelf_goals if (block, shelf) not in on_shelf]

            # If the robot is holding a block not needed as next, must put it down first (if not needed for next place)
            if held_blocks:
                held_block = next(iter(held_blocks))
                if held_block in needed:
                    # The robot is holding a needed block, one 'place' needed
                    cost += 1
                    needed.remove(held_block)
                else:
                    # The robot is holding a non-needed block, must put it down first
                    cost += 1  # Place this non-goal block somewhere not-on-shelf
            # Now, the hand is (or soon will be) empty; we must do pick+place for each remaining block
            cost += 2 * len(needed)
            return cost

    return ClutteredStorage2DHeuristic(task)
