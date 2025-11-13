"""A base class for a parameter policy wrapper over a ParameterizedController"""

import math
from typing import Any
from matplotlib.pylab import Generator

from alphatamp.approaches.parameter_scorers.base_parameter_scorer import BaseParameterScorer
from bilevel_planning.structs import ParameterizedController


class ParameterPolicy:
    def __init__(self, controller: ParameterizedController, energy_function: BaseParameterScorer) -> None:
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
            energy_score = self._energy_function.score(x, params)

            if energy_score > optimal_score:
                optimal_score = energy_score
                optimal_params = params

        return optimal_params
    
    def update_distribution(self, data):
        # given some data, update the energy function
        # minimizing BCE loss
        self._energy_function.train(data)
    

    

