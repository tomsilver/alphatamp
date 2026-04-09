
from fnmatch import fnmatch
from pyperplan.heuristics.heuristic_base import Heuristic


def generate_heuristic(task):
    class ClutteredStorage2DHeuristic(Heuristic):
        """
        Summary
        -------
        "Hybrid sum + tie-break bias" heuristic: primarily estimate remaining actions
        by summing required actions for unsatisfied goals, but add a small penalty
        that prefers states where the robot is already holding a useful block (one that
        is unsatisfied in the goal). This provides more informed guidance than pure
        goal-counting while remaining O(|state| + |goals|).

        Assumptions
        -----------
        - Goal conditions are (OnShelf b s).
        - Each unsatisfied goal generally needs Pick+Place (2 actions), unless already holding
          that block (then only Place, 1 action).
        - If holding a "useless" block (already-goal-satisfied or not a goal block), we likely
          need an additional Place to free the hand (1 action).
        - Small fractional penalties are acceptable (non-admissible) to improve greedy ordering.

        Heuristic Initialization
        ------------------------
        - Extract goal mapping block -> shelf from task.goals.

        Step-By-Step Thinking for Computing Heuristic
        ---------------------------------------------
        For a given state:
        1) Identify the held block (if any).
        2) For each goal block b:
           - If (OnShelf b goalShelf) already true: add 0.
           - Else add:
               * 1 if currently holding b
               * 2 otherwise
        3) Add "hand management" penalty:
           - If holding some block that is not currently an unsatisfied goal block, add 1.
        4) Add a small bias (0.25) if holding a useful unsatisfied goal block, to prefer keeping it
           and placing it rather than swapping.
        5) Return 0 only if all goals satisfied.
        """

        def __init__(self, task):
            self.goals = task.goals
            self.goal_shelf_of = {}
            for g in self.goals:
                parts = g[1:-1].split()
                if len(parts) == 3 and parts[0] == "OnShelf":
                    self.goal_shelf_of[parts[1]] = parts[2]

        def __call__(self, node) -> float:
            state = node.state

            def match(fact, *pat):
                parts = fact[1:-1].split()
                return len(parts) == len(pat) and all(fnmatch(p, a) for p, a in zip(parts, pat))

            held_block = None
            for f in state:
                if match(f, "Holding", "*", "*"):
                    held_block = f[1:-1].split()[2]
                    break

            unsatisfied = set()
            cost = 0.0
            for b, s in self.goal_shelf_of.items():
                if f"(OnShelf {b} {s})" not in state:
                    unsatisfied.add(b)
                    cost += 1.0 if held_block == b else 2.0

            if not unsatisfied:
                return 0.0

            # If holding an irrelevant block, we must place it away.
            if held_block is not None and held_block not in unsatisfied:
                cost += 1.0

            # Bias: prefer states already holding a useful block (slightly lower).
            if held_block is not None and held_block in unsatisfied:
                cost -= 0.25

            # Keep strictly positive for non-goal.
            return float(max(1e-6, cost))

    return ClutteredStorage2DHeuristic(task)
