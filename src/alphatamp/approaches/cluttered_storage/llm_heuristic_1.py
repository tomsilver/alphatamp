
from pyperplan.heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch

def generate_heuristic(task):
    class ClutteredStorage2DHeuristic(Heuristic):
        """
        Summary:
            Heuristic returns the number of blocks not on the shelf and not being held, plus an adjustment for whether the
            robot is ready to pick immediately. This is a "count unsatisfied goals" plus a "step for hand" style heuristic.

        Assumptions:
            - Each block goal requires two steps (pick+place), unless the robot is currently holding a required block.
            - If the robot is holding a wrong block, this adds a "correction cost".
            - Only the "HandEmpty" or "Holding" is possible, i.e., only one block held at once.

        Heuristic Initialization:
            - Caches the set of block/shelf goal pairs for OnShelf.
            - No static facts are relevant.

        Step-By-Step Thinking for Computing Heuristic:
            1. For each goal block:
               a. If OnShelf in state: 0 cost
               b. Else, if robot is holding that block: 1 (just place)
               c. Else: 2 (pick+place)
            2. If the robot is holding any block that is not needed, add penalty (must place before picking).
            3. Otherwise, sum the required actions.
        """
        def __init__(self, task):
            self.onshelf_goals = set()
            for goal in task.goals:
                parts = goal[1:-1].split()
                if parts[0] == "OnShelf":
                    self.onshelf_goals.add((parts[1], parts[2]))

        def __call__(self, node) -> float:
            state = node.state
            facts = set(state)
            holding = None
            robot = None
            for fact in state:
                parts = fact[1:-1].split()
                if parts[0] == "Holding":
                    holding = parts[2]
                    robot = parts[1]
                    break
            if robot is None:
                for fact in state:
                    parts = fact[1:-1].split()
                    if parts[0] == "HandEmpty":
                        robot = parts[1]
                        break
            cost = 0
            for block, shelf in self.onshelf_goals:
                if f"(OnShelf {block} {shelf})" in facts:
                    continue
                if holding == block:
                    cost += 1  # Just need Place
                else:
                    cost += 2  # Need Pick + Place

            # Correction: if robot is holding an "irrelevant" block, add forced placement
            if holding is not None:
                holding_relevant = any(
                    (holding, shelf) in self.onshelf_goals and f"(OnShelf {holding} {shelf})" not in facts
                    for shelf in [shelf for (_, shelf) in self.onshelf_goals]
                )
                if not holding_relevant:
                    cost += 1
            return float(cost)
    return ClutteredStorage2DHeuristic(task)
