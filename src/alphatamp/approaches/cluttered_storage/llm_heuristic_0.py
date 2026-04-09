
from fnmatch import fnmatch
from pyperplan.heuristics.heuristic_base import Heuristic


def generate_heuristic(task):
    class ClutteredStorage2DHeuristic(Heuristic):
        """
        Summary
        -------
        Greedy "count remaining placements" heuristic with a small correction for whether
        the robot is currently holding a block. In clutteredstorage2d, the only meaningful
        goal atoms are typically (OnShelf b s). A block that is not on its goal shelf must
        be picked (if not already held) and placed onto that shelf.

        Assumptions
        -----------
        - Each block has exactly one relevant goal shelf in the goal description.
        - If a block is not currently on its goal shelf, then achieving it requires:
          - 2 actions if the robot is HandEmpty: Pick + Place.
          - 1 action if the robot is already Holding that same block: Place.
        - If the robot holds a non-goal block (or a block whose goal is already satisfied),
          it will need at least 1 extra action to "get unstuck" (place it somewhere),
          because all Pick actions require HandEmpty.

        Heuristic Initialization
        ------------------------
        - Parse the goal set and build a mapping: block -> required shelf (from (OnShelf b s)).
        - No static facts are required in this domain.

        Step-By-Step Thinking for Computing Heuristic
        ---------------------------------------------
        For a given state:
        1) Extract whether the robot hand is empty and whether it holds some block.
        2) Count unsatisfied goal blocks: those where (OnShelf b goal_shelf) is not true.
        3) Base estimate:
           - If holding the (unique) unsatisfied goal block, we count 1 for that block (Place).
           - Otherwise, for each unsatisfied goal block, count 2 (Pick+Place).
        4) If holding some block that is NOT the currently-needed goal block to place next,
           add 1 as an "unload penalty" because we must place it somewhere to be able to pick.
        """

        def __init__(self, task):
            self.goals = task.goals
            # Map each block to its goal shelf (only for (OnShelf b s) goals).
            self.goal_shelf_of = {}
            for g in self.goals:
                parts = g[1:-1].split()
                if len(parts) == 3 and parts[0] == "OnShelf":
                    b, s = parts[1], parts[2]
                    self.goal_shelf_of[b] = s

        def __call__(self, node) -> float:
            state = node.state

            def match(fact, *pat):
                parts = fact[1:-1].split()
                return len(parts) == len(pat) and all(fnmatch(p, a) for p, a in zip(parts, pat))

            hand_empty = any(match(f, "HandEmpty", "*") for f in state)

            held_block = None
            for f in state:
                if match(f, "Holding", "*", "*"):
                    held_block = f[1:-1].split()[2]
                    break

            # Determine which goal blocks are unsatisfied.
            unsat = []
            for b, s in self.goal_shelf_of.items():
                if f"(OnShelf {b} {s})" not in state:
                    unsat.append(b)

            if not unsat:
                return 0.0

            # Base cost: 2 per unsatisfied goal (Pick+Place), except possible 1 if already holding that goal block.
            cost = 0
            for b in unsat:
                if held_block == b:
                    cost += 1  # only Place needed (assuming it can be placed to goal shelf)
                else:
                    cost += 2  # Pick + Place

            # If holding an unhelpful block, add an unload penalty (must place it somewhere first).
            if held_block is not None:
                # If holding a block that isn't an unsatisfied goal, or isn't the one we will place next,
                # we must place it away first to free the hand for picks.
                if held_block not in unsat:
                    cost += 1
                else:
                    # If there are multiple unsatisfied goals and we hold one of them, that's fine;
                    # no extra unload penalty.
                    pass
            else:
                # If not holding and not hand_empty, domain usually ensures consistency, but be robust:
                if not hand_empty:
                    cost += 1

            # Ensure heuristic is positive for non-goal states.
            return float(max(1, cost))

    return ClutteredStorage2DHeuristic(task)
