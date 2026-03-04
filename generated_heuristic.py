
from pyperplan.heuristics.heuristic_base import Heuristic

def generate_heuristic(task):
    class ClutteredStorage2DHeuristic(Heuristic):
        """
        Two-Phase Heuristic for ClutteredStorage2D
        ==========================================

        Strategy: clear the shelf completely first, then place all goal blocks.

        Phase 1 — Clear the shelf:
            Penalise every block that started on the shelf and is still there
            (or is currently being held for disposal).
            Cost = 2 per block (Pick + PlaceNotOnShelf), weighted 10×.

        Phase 2 — Place goal blocks:
            Penalise every goal block not yet on the shelf.
            Cost = 2 per block (Pick + PlaceOnShelf).

        Key fix — 1-step lookahead via Holding:
            PickBlockOnShelf does NOT remove OnShelf from the abstract state.
            Without this fix, "picking up clutter" and "picking up a goal block"
            look identical to the heuristic (both still show the block OnShelf),
            so the random tiebreaker in A* determines which path gets explored.

            Fix: if the robot is currently HOLDING an initial-on-shelf block,
            treat that block as already cleared from the shelf. This drops the
            h-value of "just picked up clutter" from 24 to 4, giving the
            clearing path strictly lower priority than the direct path and
            ensuring it is explored first regardless of random tiebreaks.

        Phase detection:
            When all non-initial goal blocks are placed, we are in phase 2
            (placing initial blocks back). Initial blocks on the shelf in this
            phase are "correctly placed back", not "still to clear", so h = 0
            at the true goal state.
        """

        def __init__(self, task):
            # Goal: blocks that must end up OnShelf
            self.goal_onshelf = set()
            for fact in task.goals:
                parts = fact[1:-1].split()
                if parts[0] == "OnShelf":
                    self.goal_onshelf.add((parts[1], parts[2]))

            # Phase 1 targets: blocks that start on the shelf
            self.initial_onshelf = set()
            for fact in task.initial_state:
                parts = fact[1:-1].split()
                if parts[0] == "OnShelf":
                    self.initial_onshelf.add((parts[1], parts[2]))

            # Just the block names from initial_onshelf (for Holding lookup)
            self.initial_blocks = {b for (b, _) in self.initial_onshelf}

            # Non-initial goal blocks: used to detect when phase 1 is done
            self.non_initial_goals = self.goal_onshelf - self.initial_onshelf

        def __call__(self, node) -> float:
            state = node.state

            onshelf = set()
            holding_blocks = set()
            for fact in state:
                parts = fact[1:-1].split()
                if parts[0] == "OnShelf":
                    onshelf.add((parts[1], parts[2]))
                elif parts[0] == "Holding":
                    holding_blocks.add(parts[2])  # just the block name

            # Phase detection: non-initial goals all placed → we're in phase 2
            phase1_done = self.non_initial_goals.issubset(onshelf)

            if phase1_done:
                # Phase 2: count remaining goal blocks to place
                goal_not_placed = self.goal_onshelf - onshelf
                return float(2 * len(goal_not_placed))

            # 1-step lookahead: if the robot is holding an initial-on-shelf block,
            # treat it as already cleared (it's mid-removal, not stuck on shelf).
            holding_initial = self.initial_blocks & holding_blocks
            still_to_clear = {
                (b, s) for (b, s) in (self.initial_onshelf & onshelf)
                if b not in holding_initial
            }

            # Phase 1: heavily penalise initial blocks still on shelf
            phase1_cost = 2 * len(still_to_clear)

            # Phase 2: goal blocks not yet on the shelf
            goal_not_placed = self.goal_onshelf - onshelf
            phase2_cost = 2 * len(goal_not_placed)

            return float(10 * phase1_cost + phase2_cost)

    return ClutteredStorage2DHeuristic(task)
