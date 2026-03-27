
def generate_heuristic(task):
    class ClutteredStorage2DHeuristic(Heuristic):
        """
        Summary:
            Domain-dependent heuristic for the 'clutteredstorage2d' domain.
            This heuristic estimates the number of actions required for the robot
            to place all blocks onto the shelf as defined in the goal, given the current state.
            The robot can only hold one block at a time, and moving blocks is strictly sequential.

        Assumptions:
            - There is exactly one robot and one shelf (according to provided examples).
            - Each block must be OnShelf in the goal.
            - The robot can hold at most one block at a time (domain invariant).
            - Each Pick/Place action involves at most one block.

        Heuristic Initialization:
            - The constructor parses the goal to extract all (block, shelf) pairs that must be true in the goal state ("OnShelf block shelf").
            - No use is made of static facts since the domain does not provide any static predicates.

        Step-By-Step Thinking for Computing Heuristic:
            1. For every required ("OnShelf block shelf") goal fact, check if it is already true in the state. 
               If so, no further action is needed for that block.
            2. For each goal "OnShelf block shelf" that is not yet true:
                - The corresponding block is either:
                    (a) NotOnShelf (i.e., needs to be picked and placed), or
                    (b) Already being held by the robot (i.e., can be placed without pick).
            3. The robot can only hold one block at a time; actions for blocks must occur one after the other.
                - For each unsatisfied block:
                    - If it is currently being held: only a Place action is required for that block.
                    - Otherwise: both a Pick and Place action are required (2 actions per block).
            4. However, if the robot is holding a block that is NOT among the goal blocks, no shortcut is possible; all remaining require both Pick and Place actions.
            5. The heuristic is 0 if and only if all goal OnShelf predicates are satisfied in the state.

            This greedy (inadmissible) heuristic closely estimates actions-to-goal, providing much more guidance than a blind or h_max heuristic, for example.
        """

        def __init__(self, task):
            # Extract the set of (block, shelf) pairs required by the goal.
            self.goal_pairs = set()
            for fact in task.goals:
                parts = fact[1:-1].split()
                if parts[0] == "OnShelf":
                    self.goal_pairs.add((parts[1], parts[2]))

        def __call__(self, node) -> float:
            state = node.state

            # Identify all required OnShelf predicates not currently true.
            missing = set()
            for block, shelf in self.goal_pairs:
                if f"(OnShelf {block} {shelf})" not in state:
                    missing.add((block, shelf))

            if not missing:
                return 0.0

            # Identify if the robot is holding a block, and which one (only one can be held).
            held_block = None
            for fact in state:
                if fact.startswith("(Holding "):
                    parts = fact[1:-1].split()
                    if len(parts) == 3:
                        _, _, block = parts
                        held_block = block
                        break

            total = 0

            # If the robot is already holding a block that is among the missing blocks,
            # it can immediately place it (only 1 action needed), so do not count Pick for it.
            if held_block is not None and any(held_block == block for (block, shelf) in missing):
                total += 1
                missing = {(block, shelf) for (block, shelf) in missing if block != held_block}

            # For every other missing block: need Pick + Place
            total += 2 * len(missing)
            return float(total)

    return ClutteredStorage2DHeuristic(task)
