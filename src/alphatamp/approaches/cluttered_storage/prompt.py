HEURISTIC_PROMPT = '''
<problem-description>
You are a highly-skilled professor in AI planning and a proficient 
Python programmer creating a domain-dependent heuristic function for the 
PDDL domain <domain>clutteredstorage2d</domain>. The heuristic function you create 
will be used to guide a greedy best-first search to solve instances from this domain.
Therefore, the heuristic does not need to be admissible. For a given state, 
the heuristic function should estimate the required number of actions to reach a goal 
state as accurately as possible, while remaining efficiently computable. 
The name of the heuristic should be ClutteredStorage2DHeuristic. 
The heuristic should be efficiently computable, and it should minimize the number of 
expanded nodes during the search. Next, you will receive a sequence of file contents to 
help you with your task and to show you the definition of the clutteredstorage2d domain.
</problem-description>

This is the PDDL domain file of the clutteredstorage2d domain, for which you need to create a domain-dependent heuristic:
<domain-file>
(define (domain clutteredstorage2d)

(:requirements :strips)

(:predicates (Holding ?robot ?block)
             (OnShelf ?block ?shelf)
             (NotOnShelf ?block ?shelf)
             (HandEmpty ?robot))

(:action PickBlockNotOnShelf
  :parameters (?robot ?block ?shelf)
  :precondition (and (HandEmpty ?robot) (NotOnShelf ?block ?shelf))
  :effect (and (Holding ?robot ?block) (not (HandEmpty ?robot))))

(:action PickBlockOnShelf
  :parameters (?robot ?block ?shelf)
  :precondition (and (HandEmpty ?robot) (OnShelf ?block ?shelf))
  :effect (and (Holding ?robot ?block) (not (HandEmpty ?robot))))

(:action PlaceBlockOnShelf
  :parameters (?robot ?block ?shelf)
  :precondition (and (Holding ?robot ?block) (NotOnShelf ?block ?shelf))
  :effect (and (HandEmpty ?robot) (OnShelf ?block ?shelf)
               (not (Holding ?robot ?block)) (not (NotOnShelf ?block ?shelf))))

(:action PlaceBlockNotOnShelf
  :parameters (?robot ?block ?shelf)
  :precondition (and (Holding ?robot ?block) (OnShelf ?block ?shelf))
  :effect (and (HandEmpty ?robot) (NotOnShelf ?block ?shelf)
               (not (Holding ?robot ?block)) (not (OnShelf ?block ?shelf)))))

</domain-file>

This is an example of a PDDL instance file of the clutteredstorage2d domain:
<instance-file-example-1>
(define (problem clutteredstorage2d-b3-01)
 (:domain clutteredstorage2d)
 (:objects robot0 block0 block1 block2 shelf0)
 (:init
    (HandEmpty robot0)
    (OnShelf block0 shelf0)
    (NotOnShelf block1 shelf0)
    (NotOnShelf block2 shelf0)
)
 (:goal (and
    (OnShelf block0 shelf0)
    (OnShelf block1 shelf0)
    (OnShelf block2 shelf0)
)))
</instance-file-example-1>

This is a second example of a PDDL instance file of the clutteredstorage2d domain:
<instance-file-example-2>

(define (problem clutteredstorage2d-b15-01)
 (:domain clutteredstorage2d)
 (:objects robot0 block0 block1 block2 block3 block4 block5 block6
           block7 block8 block9 block10 block11 block12 block13 block14 shelf0)
 (:init
    (HandEmpty robot0)
    (OnShelf block0 shelf0)
    (OnShelf block1 shelf0)
    (OnShelf block2 shelf0)
    (OnShelf block3 shelf0)
    (OnShelf block4 shelf0)
    (OnShelf block5 shelf0)
    (OnShelf block6 shelf0)
    (NotOnShelf block7 shelf0)
    (NotOnShelf block8 shelf0)
    (NotOnShelf block9 shelf0)
    (NotOnShelf block10 shelf0)
    (NotOnShelf block11 shelf0)
    (NotOnShelf block12 shelf0)
    (NotOnShelf block13 shelf0)
    (NotOnShelf block14 shelf0)
)
 (:goal (and
    (OnShelf block0 shelf0)
    (OnShelf block1 shelf0)
    (OnShelf block2 shelf0)
    (OnShelf block3 shelf0)
    (OnShelf block4 shelf0)
    (OnShelf block5 shelf0)
    (OnShelf block6 shelf0)
    (OnShelf block7 shelf0)
    (OnShelf block8 shelf0)
    (OnShelf block9 shelf0)
    (OnShelf block10 shelf0)
    (OnShelf block11 shelf0)
    (OnShelf block12 shelf0)
    (OnShelf block13 shelf0)
    (OnShelf block14 shelf0)
)))

</instance-file-example-2>

This is the PDDL domain file of another domain, called Gripper, which serves as an example:
<gripper-domain-file>
(define (domain gripper-strips)
   (:predicates (room ?r)
		(ball ?b)
		(gripper ?g)
		(at-robby ?r)
		(at ?b ?r)
		(free ?g)
		(carry ?o ?g))

   (:action move
       :parameters  (?from ?to)
       :precondition (and  (room ?from) (room ?to) (at-robby ?from))
       :effect (and  (at-robby ?to)
		     (not (at-robby ?from))))

   (:action pick
       :parameters (?obj ?room ?gripper)
       :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper)
			    (at ?obj ?room) (at-robby ?room) (free ?gripper))
       :effect (and (carry ?obj ?gripper)
		    (not (at ?obj ?room))
		    (not (free ?gripper))))

   (:action drop
       :parameters  (?obj  ?room ?gripper)
       :precondition  (and  (ball ?obj) (room ?room) (gripper ?gripper)
			    (carry ?obj ?gripper) (at-robby ?room))
       :effect (and (at ?obj ?room)
		    (free ?gripper)
		    (not (carry ?obj ?gripper)))))
</gripper-domain-file>

This is an example of an instance file from the Gripper domain:
<gripper-instance-file-example>
(define (problem strips-gripper-x-20)
   (:domain gripper-strips)
   (:objects rooma roomb ball42 ball41 ball40 ball39 ball38 ball37
             ball36 ball35 ball34 ball33 ball32 ball31 ball30 ball29 ball28
             ball27 ball26 ball25 ball24 ball23 ball22 ball21 ball20 ball19
             ball18 ball17 ball16 ball15 ball14 ball13 ball12 ball11 ball10
             ball9 ball8 ball7 ball6 ball5 ball4 ball3 ball2 ball1 left right)
   (:init (room rooma)
          (room roomb)
          (ball ball42)
          (ball ball41)
          (ball ball40)
          (ball ball39)
          (ball ball38)
          (ball ball37)
          (ball ball36)
          (ball ball35)
          (ball ball34)
          (ball ball33)
          (ball ball32)
          (ball ball31)
          (ball ball30)
          (ball ball29)
          (ball ball28)
          (ball ball27)
          (ball ball26)
          (ball ball25)
          (ball ball24)
          (ball ball23)
          (ball ball22)
          (ball ball21)
          (ball ball20)
          (ball ball19)
          (ball ball18)
          (ball ball17)
          (ball ball16)
          (ball ball15)
          (ball ball14)
          (ball ball13)
          (ball ball12)
          (ball ball11)
          (ball ball10)
          (ball ball9)
          (ball ball8)
          (ball ball7)
          (ball ball6)
          (ball ball5)
          (ball ball4)
          (ball ball3)
          (ball ball2)
          (ball ball1)
          (at-robby rooma)
          (free left)
          (free right)
          (at ball42 rooma)
          (at ball41 rooma)
          (at ball40 rooma)
          (at ball39 rooma)
          (at ball38 rooma)
          (at ball37 rooma)
          (at ball36 rooma)
          (at ball35 rooma)
          (at ball34 rooma)
          (at ball33 rooma)
          (at ball32 rooma)
          (at ball31 rooma)
          (at ball30 rooma)
          (at ball29 rooma)
          (at ball28 rooma)
          (at ball27 rooma)
          (at ball26 rooma)
          (at ball25 rooma)
          (at ball24 rooma)
          (at ball23 rooma)
          (at ball22 rooma)
          (at ball21 rooma)
          (at ball20 rooma)
          (at ball19 rooma)
          (at ball18 rooma)
          (at ball17 rooma)
          (at ball16 rooma)
          (at ball15 rooma)
          (at ball14 rooma)
          (at ball13 rooma)
          (at ball12 rooma)
          (at ball11 rooma)
          (at ball10 rooma)
          (at ball9 rooma)
          (at ball8 rooma)
          (at ball7 rooma)
          (at ball6 rooma)
          (at ball5 rooma)
          (at ball4 rooma)
          (at ball3 rooma)
          (at ball2 rooma)
          (at ball1 rooma)
          (gripper left)
          (gripper right))
   (:goal (and (at ball42 roomb)
               (at ball41 roomb)
               (at ball40 roomb)
               (at ball39 roomb)
               (at ball38 roomb)
               (at ball37 roomb)
               (at ball36 roomb)
               (at ball35 roomb)
               (at ball34 roomb)
               (at ball33 roomb)
               (at ball32 roomb)
               (at ball31 roomb)
               (at ball30 roomb)
               (at ball29 roomb)
               (at ball28 roomb)
               (at ball27 roomb)
               (at ball26 roomb)
               (at ball25 roomb)
               (at ball24 roomb)
               (at ball23 roomb)
               (at ball22 roomb)
               (at ball21 roomb)
               (at ball20 roomb)
               (at ball19 roomb)
               (at ball18 roomb)
               (at ball17 roomb)
               (at ball16 roomb)
               (at ball15 roomb)
               (at ball14 roomb)
               (at ball13 roomb)
               (at ball12 roomb)
               (at ball11 roomb)
               (at ball10 roomb)
               (at ball9 roomb)
               (at ball8 roomb)
               (at ball7 roomb)
               (at ball6 roomb)
               (at ball5 roomb)
               (at ball4 roomb)
               (at ball3 roomb)
               (at ball2 roomb)
               (at ball1 roomb))))
</gripper-instance-file-example>

This is an example of a domain-dependent heuristic for Gripper:
<code-file-heuristic-1>
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class GripperHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Gripper domain.

    # Summary
    This heuristic estimates the number of actions needed to transport all balls
    from `rooma` to `roomb`.

    # Assumptions:
    - The robot has two grippers, allowing it to carry up to two balls per trip.
    - The robot must return to rooma after each trip, except for the final trip.
    - If the robot starts in roomb, it must move to rooma first.

    # Heuristic Initialization
    - Implicitly assume that all balls must be in `roomb` at the end.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the number of balls still in `rooma` that need to be transported.
    2. Determine if the robot is currently carrying balls (it may start with 1 or 2 already).
    3. Check whether the robot is in `rooma` or `roomb`:
       - If in room B, it may need to drop the carried balls first before moving to A.
       - If in room A, it can immediately begin planning the transport.
    4. Handle the case where the robot starts with balls in the grippers:
       - If carrying 2 balls, it should move to B, drop them, and return to A.
       - If carrying 1 ball and an odd number remains in `rooma`, it may pick up another ball before moving.
       - If carrying 1 ball and an even number remains, it transports the single ball first.
    5. Compute the number of full two-ball trips needed:
       - This is `balls_in_rooma // 2` (since up to 2 balls are moved per trip).
       - Each full two-ball trip costs 6 actions (except for the last trip).
    6. Handle the last remaining ball (if the total number of balls is odd):
       - If one ball is left, the robot moves to A, picks it up, moves to B, and drops it.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        # The set of facts that must hold in goal states. We assume that all balls must be in `roomb` at the end.
        self.goals = task.goals
        # Static facts are not needed for this heuristic.
        static_facts = task.static

    def __call__(self, node):
        """Estimate the minimum cost to transport all remaining balls from room A to room B."""
        state = node.state

        def match(fact, *args):
            """
            Utility function to check if a PDDL fact matches a given pattern.
            - `fact`: The fact as a string (e.g., "(at ball1 rooma)").
            - `args`: The pattern to match (e.g., "at", "*", "rooma").
            - Returns `True` if the fact matches the pattern, `False` otherwise.
            """
            parts = fact[1:-1].split()  # Remove parentheses and split into individual elements.
            return all(fnmatch(part, arg) for part, arg in zip(parts, args))

        # Count how many balls are currently in room A.
        balls_in_room_a = sum(1 for fact in state if match(fact, "at", "*", "rooma"))

        # Count the number of balls currently held by the robot.
        balls_in_grippers = sum(1 for fact in state if match(fact, "carry", "*", "*"))

        # Check if the robot is in room A.
        robot_in_room_a = "(at-robby rooma)" in state

        # Define the cost of each individual action for readability.
        move_cost = 1  # Moving between rooms.
        pick_cost = 1  # Picking up a ball.
        drop_cost = 1  # Dropping a ball.

        total_cost = 0  # Initialize the heuristic cost.

        # Handle cases where the robot is already carrying balls.
        if robot_in_room_a:
            if balls_in_grippers == 2:
                # Both grippers are full, so move to room B and drop both balls.
                total_cost += move_cost + 2 * drop_cost
            elif balls_in_grippers == 1 and balls_in_room_a % 2 == 1:
                # Pick one more ball to fill both grippers, then move and drop both.
                total_cost += pick_cost + move_cost + 2 * drop_cost
                balls_in_room_a -= 1  # Since we moved one extra ball.
            elif balls_in_grippers == 1 and balls_in_room_a % 2 == 0:
                # Move with one ball and drop it, leaving an even number of balls.
                total_cost += move_cost + drop_cost
        else:
            # If the robot is in room B, it must drop any carried balls.
            total_cost += balls_in_grippers * drop_cost

        if balls_in_room_a > 0:
            # Move back to room A to continue transporting balls.
            total_cost += move_cost

            # Compute the number of trips with two balls.
            num_two_ball_trips = balls_in_room_a // 2

            # Each trip includes: 2 picks, 1 move to B, 2 drops and 1 move back to A (except for the last trip).
            total_cost += num_two_ball_trips * (2 * pick_cost + move_cost + 2 * drop_cost + move_cost) - 1

            # If there's a single ball left after the two-ball trips, go back to A and move the ball by itself.
            if balls_in_room_a % 2 == 1:
                total_cost += move_cost + pick_cost + move_cost + drop_cost

        # Return the estimated cost to goal state.
        return total_cost

</code-file-heuristic-1>

This is the PDDL domain file of another domain, called Logistics, to serve as a second example:
<logistics-domain-file>
(define (domain logistics-strips)
  (:requirements :strips)
  (:predicates 	(OBJ ?obj)
	       	(TRUCK ?truck)
               	(LOCATION ?loc)
		(AIRPLANE ?airplane)
                (CITY ?city)
                (AIRPORT ?airport)
		(at ?obj ?loc)
		(in ?obj1 ?obj2)
		(in-city ?obj ?city))

(:action LOAD-TRUCK
  :parameters
   (?obj
    ?truck
    ?loc)
  :precondition
   (and (OBJ ?obj) (TRUCK ?truck) (LOCATION ?loc)
   (at ?truck ?loc) (at ?obj ?loc))
  :effect
   (and (not (at ?obj ?loc)) (in ?obj ?truck)))

(:action LOAD-AIRPLANE
  :parameters
   (?obj
    ?airplane
    ?loc)
  :precondition
   (and (OBJ ?obj) (AIRPLANE ?airplane) (LOCATION ?loc)
   (at ?obj ?loc) (at ?airplane ?loc))
  :effect
   (and (not (at ?obj ?loc)) (in ?obj ?airplane)))

(:action UNLOAD-TRUCK
  :parameters
   (?obj
    ?truck
    ?loc)
  :precondition
   (and (OBJ ?obj) (TRUCK ?truck) (LOCATION ?loc)
        (at ?truck ?loc) (in ?obj ?truck))
  :effect
   (and (not (in ?obj ?truck)) (at ?obj ?loc)))

(:action UNLOAD-AIRPLANE
  :parameters
   (?obj
    ?airplane
    ?loc)
  :precondition
   (and (OBJ ?obj) (AIRPLANE ?airplane) (LOCATION ?loc)
        (in ?obj ?airplane) (at ?airplane ?loc))
  :effect
   (and (not (in ?obj ?airplane)) (at ?obj ?loc)))

(:action DRIVE-TRUCK
  :parameters
   (?truck
    ?loc-from
    ?loc-to
    ?city)
  :precondition
   (and (TRUCK ?truck) (LOCATION ?loc-from) (LOCATION ?loc-to) (CITY ?city)
   (at ?truck ?loc-from)
   (in-city ?loc-from ?city)
   (in-city ?loc-to ?city))
  :effect
   (and (not (at ?truck ?loc-from)) (at ?truck ?loc-to)))

(:action FLY-AIRPLANE
  :parameters
   (?airplane
    ?loc-from
    ?loc-to)
  :precondition
   (and (AIRPLANE ?airplane) (AIRPORT ?loc-from) (AIRPORT ?loc-to)
	(at ?airplane ?loc-from))
  :effect
   (and (not (at ?airplane ?loc-from)) (at ?airplane ?loc-to)))
)
</logistics-domain-file>

This is an example of an instance file from the Logistics domain:
<logistics-instance-file-example>
(define (problem strips-log-y-5)
   (:domain logistics-strips)
   (:objects package5 package4 package3 package2 package1 city8
             city7 city6 city5 city4 city3 city2 city1 truck15 truck14
             truck13 truck12 truck11 truck10 truck9 truck8 truck7 truck6
             truck5 truck4 truck3 truck2 truck1 plane1 city8-2 city8-1
             city7-2 city7-1 city6-2 city6-1 city5-2 city5-1 city4-2
             city4-1 city3-2 city3-1 city2-2 city2-1 city1-2 city1-1
             city8-3 city7-3 city6-3 city5-3 city4-3 city3-3 city2-3
             city1-3)
   (:init (obj package5)
          (obj package4)
          (obj package3)
          (obj package2)
          (obj package1)
          (city city8)
          (city city7)
          (city city6)
          (city city5)
          (city city4)
          (city city3)
          (city city2)
          (city city1)
          (truck truck15)
          (truck truck14)
          (truck truck13)
          (truck truck12)
          (truck truck11)
          (truck truck10)
          (truck truck9)
          (truck truck8)
          (truck truck7)
          (truck truck6)
          (truck truck5)
          (truck truck4)
          (truck truck3)
          (truck truck2)
          (truck truck1)
          (airplane plane1)
          (location city8-2)
          (location city8-1)
          (location city7-2)
          (location city7-1)
          (location city6-2)
          (location city6-1)
          (location city5-2)
          (location city5-1)
          (location city4-2)
          (location city4-1)
          (location city3-2)
          (location city3-1)
          (location city2-2)
          (location city2-1)
          (location city1-2)
          (location city1-1)
          (airport city8-3)
          (location city8-3)
          (airport city7-3)
          (location city7-3)
          (airport city6-3)
          (location city6-3)
          (airport city5-3)
          (location city5-3)
          (airport city4-3)
          (location city4-3)
          (airport city3-3)
          (location city3-3)
          (airport city2-3)
          (location city2-3)
          (airport city1-3)
          (location city1-3)
          (in-city city8-3 city8)
          (in-city city8-2 city8)
          (in-city city8-1 city8)
          (in-city city7-3 city7)
          (in-city city7-2 city7)
          (in-city city7-1 city7)
          (in-city city6-3 city6)
          (in-city city6-2 city6)
          (in-city city6-1 city6)
          (in-city city5-3 city5)
          (in-city city5-2 city5)
          (in-city city5-1 city5)
          (in-city city4-3 city4)
          (in-city city4-2 city4)
          (in-city city4-1 city4)
          (in-city city3-3 city3)
          (in-city city3-2 city3)
          (in-city city3-1 city3)
          (in-city city2-3 city2)
          (in-city city2-2 city2)
          (in-city city2-1 city2)
          (in-city city1-3 city1)
          (in-city city1-2 city1)
          (in-city city1-1 city1)
          (at plane1 city3-3)
          (at truck15 city8-2)
          (at truck14 city7-2)
          (at truck13 city6-1)
          (at truck12 city5-2)
          (at truck11 city4-1)
          (at truck10 city3-2)
          (at truck9 city2-1)
          (at truck8 city1-2)
          (at truck7 city6-1)
          (at truck6 city3-2)
          (at truck5 city1-3)
          (at truck4 city4-3)
          (at truck3 city1-3)
          (at truck2 city7-1)
          (at truck1 city8-1)
          (at package5 city3-2)
          (at package4 city5-1)
          (at package3 city1-1)
          (at package2 city5-2)
          (at package1 city2-1))
   (:goal (and (at package5 city6-3)
               (at package4 city5-3)
               (at package3 city8-3)
               (at package2 city4-3)
               (at package1 city6-3))))
</logistics-instance-file-example>

This is an example of a domain-dependent heuristic for Logistics:
<code-file-heuristic-2>
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class LogisticsHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Logistics domain.

    # Summary
    The heuristic estimates the number of necessary actions (load, unload, and transport) in order to transport each package to its goal based on its current state.

    # Assumptions
    - Packages can be on the ground, in a truck, or in a plane.
    - Trucks move within a city, while planes move between cities.
    - If a package is already at the goal, no extra actions are needed.

    # Heuristic Initialization
    - Extract the goal locations for each package and the static facts (e.g., `in-city` relationships and airport locations) from the task.

    # Step-by-Step Thinking for Computing the Heuristic Value
    Below is the thought process for computing the heuristic for a given state:

    1. Extract Relevant Information:
    - Identify the current location of every package.
    - Identify whether a package is inside a vehicle (truck or plane), and if so, find the physical location of that vehicle.

    2. Distinguish Between Intra-city and Inter-city Transport:
    - Determine the current city and goal city for each package by checking its location-to-city mapping.
    - If the current city is the same as the goal city, follow the intra-city package movement rules.
    - If the current city is different from the goal city, follow the inter-city package movement rules.

    3. Handle Intra-city Transport:
    - If the package is already at its goal location, no action is required.
    - If the package is in a plane, it must be unloaded.
    - If the package is not in a truck and not already at its goal, it must be loaded into a truck.
    - If the package is in a truck or not yet at its final location, it must be unloaded from the truck at the goal.

    4. Handle Inter-city Transport:
    - Step 1: Move the package to the airport in the current city.
        - If the package is not inside a truck and not at an airport, it must be loaded into a truck.
        - If the package is not at an airport or inside a truck, it must be unloaded from the truck at the airport.
    - Step 2: Fly the package to the destination city.
        - If the package is not in a plane, it must be loaded into a plane.
        - The package must always be unloaded from the plane at the airport of the destination city.
    - Step 3: Move the package from the airport to its final location.
        - If the goal location is not an airport, the package must be loaded into a truck at the airport.
        - Finally, the package must be unloaded from the truck at the goal location.

    5. Summing the Actions:
    - The total heuristic value is the sum of all necessary actions.
    - Loading and unloading costs are counted exactly based on the required actions.
    - Transport movements (trucks or planes) are counted only when necessary.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each package.
        - Static facts (`in-city` relationships and airport locations).
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.

        # Map locations to their respective cities using "in-city" relationships.
        self.location_to_city = {
            get_parts(fact)[1]: get_parts(fact)[2]
            for fact in static_facts
            if match(fact, "in-city", "*", "*")
        }

        # Identify all airport locations.
        self.airports = {
            get_parts(fact)[1]
            for fact in static_facts
            if match(fact, "airport", "*")
        }

        # Store goal locations for each package.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Track where packages and vehicles are currently located.
        current_locations = {}
        for fact in state:
            predicate, *args = get_parts(fact)
            if predicate in ["at", "in"]:  # Track both direct location and inside vehicle.
                obj, location = args
                current_locations[obj] = location

        total_cost = 0  # Initialize action cost counter.

        for package, goal_location in self.goal_locations.items():
            # Get the current location of the package (could be a city location, truck or plane).
            current_location = current_locations[package]

            # Check if the package is inside a vehicle.
            in_vehicle = current_location not in self.location_to_city

            if in_vehicle:
                # Identify type of vehicle (truck or plane).
                in_plane = current_location.startswith("plane")
                in_truck = current_location.startswith("truck")
                assert in_plane ^ in_truck, f"Invalid state: {current_location}"

                # Retrieve the physical location of the vehicle.
                current_location = current_locations[current_location]
            else:
                in_plane = False
                in_truck = False

            # Get the city of the package's current location and goal.
            current_city = self.location_to_city[current_location]
            goal_city = self.location_to_city[goal_location]

            # Intra-city Transport (Same City)
            if current_city == goal_city:
                if in_plane:
                    total_cost += 1  # Unload from the plane.

                if current_location != goal_location and not in_truck:
                    total_cost += 1  # Load into a truck.

                if current_location != goal_location or in_truck:
                    total_cost += 1  # Unload from the truck.

            # Inter-city Transport (Different Cities)
            else:
                # Step 1: Move to the airport in the current city.
                if current_location not in self.airports and not in_truck:
                    total_cost += 1  # Load into a truck.

                if current_location not in self.airports or in_truck:
                    total_cost += 1  # Unload from the truck at the airport.

                # Step 2: Fly to the destination city.
                if not in_plane:
                    total_cost += 1  # Load into a plane.

                total_cost += 1  # Unload from the plane.

                # Step 3: Transport from airport to the goal (if required).
                if goal_location not in self.airports:
                    total_cost += 1  # Load into a truck at destination airport.
                    total_cost += 1  # Unload at the destination.

        return total_cost

</code-file-heuristic-2>

This is how an example state from the clutteredstorage2d domain is represented internally by the planner. Note that PDDL facts are represented as strings, for example '(predicate_name object1 object2)'.
<state>
frozenset({'(Holding robot0 block4)', '(OnShelf block0 shelf0)', '(OnShelf block1 shelf0)', '(OnShelf block2 shelf0)', '(NotOnShelf block3 shelf0)', '(NotOnShelf block4 shelf0)', '(NotOnShelf block5 shelf0)', '(NotOnShelf block6 shelf0)'})
</state>

This is an example for how the static information is represented internally by the planner:
<static>
frozenset()

</static>

This is the source code for representing operators and tasks in the planner:
<code-file-task>
"""
Classes for representing a STRIPS planning task
"""


class Operator:
    """
    The preconditions represent the facts that have to be true
    before the operator can be applied.
    add_effects are the facts that the operator makes true.
    delete_effects are the facts that the operator makes false.
    """

    def __init__(self, name, preconditions, add_effects, del_effects):
        self.name = name
        self.preconditions = frozenset(preconditions)
        self.add_effects = frozenset(add_effects)
        self.del_effects = frozenset(del_effects)

    def applicable(self, state):
        """
        Operators are applicable when their set of preconditions is a subset
        of the facts that are true in "state".

        @return True if the operator's preconditions is a subset of the state,
                False otherwise
        """
        return self.preconditions <= state

    def apply(self, state):
        """
        Applying an operator means removing the facts that are made false
        by the operator from the set of true facts in state and adding
        the facts made true.

        Note that therefore it is possible to have operands that make a
        fact both false and true. This results in the fact being true
        at the end.

        @param state The state that the operator should be applied to
        @return A new state (set of facts) after the application of the
                operator
        """
        assert self.applicable(state)
        assert type(state) in (frozenset, set)
        return (state - self.del_effects) | self.add_effects

    def __eq__(self, other):
        return (
            self.name == other.name
            and self.preconditions == other.preconditions
            and self.add_effects == other.add_effects
            and self.del_effects == other.del_effects
        )

    def __hash__(self):
        return hash((self.name, self.preconditions, self.add_effects, self.del_effects))

    def __str__(self):
        s = "%s\n" % self.name
        for group, facts in [
            ("PRE", self.preconditions),
            ("ADD", self.add_effects),
            ("DEL", self.del_effects),
        ]:
            for fact in facts:
                s += "  {}: {}\n".format(group, fact)
        return s

    def __repr__(self):
        return "<Op %s>" % self.name


class Task:
    """
    A STRIPS planning task
    """

    def __init__(self, name, facts, initial_state, goals, operators, static):
        """
        @param name The task's name
        @param facts A set of all the fact names that are valid in the domain
        @param initial_state A set of fact names that are true at the beginning
        @param goals A set of fact names that must be true to solve the problem
        @param operators A set of operator instances for the domain
        @param static_info A set of facts that are true in every state
        """
        self.name = name
        self.facts = facts
        self.initial_state = initial_state
        self.goals = goals
        self.operators = operators
        self.static = static

    def goal_reached(self, state):
        """
        The goal has been reached if all facts that are true in "goals"
        are true in "state".

        @return True if all the goals are reached, False otherwise
        """
        return self.goals <= state

    def get_successor_states(self, state):
        """
        @return A list with (op, new_state) pairs where "op" is the applicable
        operator and "new_state" the state that results when "op" is applied
        in state "state".
        """
        return [(op, op.apply(state)) for op in self.operators if op.applicable(state)]

    def __str__(self):
        s = "Task {0}\n  Vars:  {1}\n  Init:  {2}\n  Goals: {3}\n  Ops:   {4}"
        return s.format(
            self.name,
            ", ".join(self.facts),
            self.initial_state,
            self.goals,
            "\n".join(map(repr, self.operators)),
        )

    def __repr__(self):
        string = "<Task {0}, vars: {1}, operators: {2}>"
        return string.format(self.name, len(self.facts), len(self.operators))

</code-file-task>

Provide only the Python code of the domain-dependent heuristic for the clutteredstorage2d domain.
Write a Python function named `generate_heuristic` that takes a `task` argument and
returns an instance of the heuristic class. The function must follow this pattern:

def generate_heuristic(task):
    class ClutteredStorage2DHeuristic(Heuristic):
        def __init__(self, task): ...
        def __call__(self, node) -> float: ...
    return ClutteredStorage2DHeuristic(task)

Here is a checklist to help you with your code:
1) The code for extracting objects from facts remembers to ignore the surrounding brackets.
2) The heuristic is 0 only for goal states.
3) The heuristic value is finite for solvable states.
4) All used modules are imported. Use `from pyperplan.heuristics.heuristic_base import Heuristic` (not `from heuristics.heuristic_base import Heuristic`).
5) The information from static facts is extracted into suitable data structures in the constructor.
6) Provide a detailed docstring explaining the heuristic calculation.
For this, divide the docstring into sections "Summary", "Assumptions", "Heuristic Initialization" and "Step-By-Step Thinking for Computing Heuristic".

Important: This heuristic needs to consider
'''


def build_heuristic_prompt(initial_atoms: str, goal_atoms: str) -> str:
    return HEURISTIC_PROMPT + f"""
This is the specific problem instance:

Initial State (true predicates):
{initial_atoms}

Goal State (target predicates):
{goal_atoms}
"""
