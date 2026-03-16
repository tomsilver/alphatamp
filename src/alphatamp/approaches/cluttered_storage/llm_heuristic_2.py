
from pyperplan.heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch

def generate_heuristic(task):
    class ClutteredStorage2DHeuristic(Heuristic):
        """
        Summary:
            This heuristic uses a "max-goal-distance" strategy: it returns the
            maximum number of actions needed to move any single block from its current state to the goal,
            plus the number of other unsatisfied goals (to prioritize making progress on the most difficult block).
            This prioritizes the "critical path".

        Assumptions:
            - Each block must be picked and placed unless already on shelf or held.
            - Robot must be hand-empty to pick a new block, possibly requiring to place an unneeded held block.
            - Only one block may be held at a time.

        Heuristic Initialization:
            - Extracts all block/shelf pairs for the OnShelf goals.

        Step-By-Step Thinking for Computing Heuristic:
            1. For each block that is required on the shelf:
                a. If already on the shelf, cost is 0.
                b. If currently held, cost is 1.
                c. Otherwise, cost is 2 (pick+place).
            2. If the robot is holding an irrelevant block, the first pick will cost an extra action (for placing it).
            3. Take the max per-block cost (critical path), then add the number of other goals not yet satisfied to encourage overall progress.
        """
        def __init__(self, task):
            self.onshelf_goals = []
            for goal in task.goals:
                parts = goal[1:-1].split()
                if parts[0] == "OnShelf":
                    self.onshelf_goals.append((parts[1], parts[2]))

        def __call__(self, node) -> float:
            state = node.state
            facts = set(state)

            holding_block = None
            robot = None
            for fact in state:
                parts = fact[1:-1].split()
                if parts[0] == "Holding":
                    holding_block = parts[2]
                    robot = parts[1]
                    break
            if robot is None:
                for fact in state:
                    parts = fact[1:-1].split()
                    if parts[0] == "HandEmpty":
                        robot = parts[1]
                        break

            per_block_costs = []
            unsatisfied = 0
            for block, shelf in self.onshelf_goals:
                if f"(OnShelf {block} {shelf})" in facts:
                    per_block_costs.append(0)
                    continue
                unsatisfied += 1
                if holding_block == block:
                    per_block_costs.append(1)  # Only Place needed
                else:
                    per_block_costs.append(2)  # Pick + Place

            # If holding a block that is not among the currently needed blocks, extra Place needed
            needed_blocks = set(block for block, shelf in self.onshelf_goals if f"(OnShelf {block} {shelf})" not in facts)
            penalty = 0
            if holding_block is not None and holding_block not in needed_blocks:
                penalty = 1
            if per_block_costs:
                heuristic = max(per_block_costs) + unsatisfied - 1 + penalty
            else:
                heuristic = 0
            return float(heuristic)
    return ClutteredStorage2DHeuristic(task)
