"""A base class for a parameter policy wrapper over a ParameterizedController"""

import math
from typing import Any
from matplotlib.pylab import Generator
from prbench.envs.geom2d.structs import SE2Pose
from prbench_models.geom2d.utils import Geom2dRobotController
from bilevel_planning.structs import ParameterizedController
from relational_structs import ObjectCentricState


class ParameterPolicy(Geom2dRobotController):
    def __init__(self, controller: ParameterizedController, energy_function) -> None:
        self._controller = controller
        
        self._energy_function = energy_function
        self._param_sample_count = 100
        # need to get 
        pass

    def sample_parameters(self, x: Any, rng: Generator) -> Any:
        optimal_params = None
        optimal_score = -math.inf
        for _ in range(self._param_sample_count):
            params = self._controller.sample_parameters(x, rng)

            # Now we score the params based on the energy function
            energy_score = self._energy_function(x, params)

            if energy_score > optimal_score:
                optimal_score = energy_score
                optimal_params = params

        return optimal_params
    
    def update_distribution(self, data):
        # given some data, update the energy function
        # minimizing BCE loss
        pass


    def _generate_waypoints(self, state: ObjectCentricState) -> list[tuple[SE2Pose, float]]:
        assert isinstance(self._controller, Geom2dRobotController)
        return self._controller._generate_waypoints(state)
    
    def _get_vacuum_actions(self) -> tuple[float, float]:
        assert isinstance(self._controller, Geom2dRobotController)
        return self._controller._get_vacuum_actions()
    

    

