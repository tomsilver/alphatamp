
from fnmatch import fnmatch
from pyperplan.heuristics.heuristic_base import Heuristic


def generate_heuristic(task):
    class ClutteredStorage2DHeuristic(Heuristic):
        """
        Summary
        -------
        "Max-per-block" heuristic: estimate remaining plan length as the maximum remaining
        effort over all individual goal blocks (rather than the sum). This often guides
        greedy best-first search to focus on the most problematic unsatisfied goal first,
        reducing node expansions in domains where goals are largely independent and actions
        are reversible.

        Assumptions
        -----------
        - Goals are (OnShelf b s) atoms.
        - Achieving each unsatisfied (OnShelf b s) requires:
            * If Holding b already: 1 (PlaceBlockOnShelf).
            * Else: 2 (Pick + Place).
        - If robot is holding some other block, at least 1 action is needed to place it away
          before any Pick action can be performed.

        Heuristic Initialization
        ------------------------
        - Build mapping block -> goal shelf from the goal set.

        Step-By-Step Thinking for Computing Heuristic
        ---------------------------------------------
        For a given state:
        1) Identify held block (if any).
        2) For every goal (OnShelf b s):
           - If already true: effort 0.
           - Else effort is:
               a) 1 if currently holding b (just place).
               b) 2 if not holding b and hand is free.
               c) 3 if not holding b and holding some other block (place-away + pick + place).
        3) Return the maximum effort across all goal blocks (and 0 only at goal).
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

            # Compute per-goal effort and take maximum.
            max_effort = 0
            for b, s in self.goal_shelf_of.items():
                if f"(OnShelf {b} {s})" in state:
                    effort = 0
                else:
                    if held_block == b:
                        effort = 1
                    elif held_block is None:
                        effort = 2
                    else:
                        effort = 3  # must unload held block first, then pick+place
                if effort > max_effort:
                    max_effort = effort

            return float(max_effort)

    return ClutteredStorage2DHeuristic(task)
