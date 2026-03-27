
from pyperplan.heuristics.heuristic_base import Heuristic

def generate_heuristic(task):
    class ClutteredStorage2DHeuristic(Heuristic):
        """
        Summary:
            Computes a relaxed plan-length estimate for placing all required blocks on the shelf,
            ignoring delete effects and hand constraints, by simply counting the number of blocks not yet on the shelf.

        Assumptions:
            - The robot can in theory pick and place simultaneously in the relaxed plan.
            - Ignores effects of hand occupancy; relaxes the problem substantially.
            - Only actions that satisfy goals matter, and actions are independent per block.

        Heuristic Initialization:
            - Builds the list of (OnShelf block shelf) facts present in goals.
            - No static facts are used.

        Step-By-Step Thinking for Computing Heuristic:
            1. Count the number of goal facts (OnShelf block shelf) not true in the current state.
            2. For each such missing goal, one 'place' action suffices in the relaxed plan,
               since we ignore hand and precondition constraints.
            3. Thus the heuristic is simply the number of unsatisfied (OnShelf ... ...) goal facts.

        """
        def __init__(self, task):
            self.on_shelf_goals = set()
            for g in task.goals:
                parts = g[1:-1].split()
                if len(parts) == 3 and parts[0] == "OnShelf":
                    self.on_shelf_goals.add((parts[1], parts[2]))
        def __call__(self, node):
            state = node.state
            on_shelf_now = set()
            for f in state:
                parts = f[1:-1].split()
                if len(parts) == 3 and parts[0] == "OnShelf":
                    on_shelf_now.add((parts[1], parts[2]))
            unsatisfied = self.on_shelf_goals - on_shelf_now
            return 0 if not unsatisfied else len(unsatisfied)
    return ClutteredStorage2DHeuristic(task)
