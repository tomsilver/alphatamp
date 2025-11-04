"""
llm_ppl_approach.py
"""

from typing import Any, Callable, Generic, TypeVar

# copied from llm_ppl_approach.py
from prpl_llm_utils.code import ( 
    FunctionOutputRepromptCheck,
    SyntaxRepromptCheck,
    synthesize_python_function_with_llm,
)
from prpl_llm_utils.models import PretrainedLargeModel
from prpl_llm_utils.structs import Query
# --------------------------------------------------------------

from bilevel_planning.structs import Plan, PlanningProblem, SesameModels
from alphatamp.approaches.base_approach import BaseApproach

_O = TypeVar("_O")  # observation
_U = TypeVar("_U")  # action
_X = TypeVar("_X")  # state


""" High level next steps:
0. Sesame model is a base class that defines the interface between high level and low level
1. Env model is a SesameModel object, specified for the environment
2. I need to create a prompt that contains the skills, operators, parameters of the specific env model
3. Once I have this prompt, how is the planning being done?

With regular approach
Abstract successor function defines what is possible -> Generate plans --> Sample plans

With oracle approach
Abstract successor function (useless) --> plan is hardcoded

With LLM approach
Abstract successor function defines what is possible --> LLM needs to factor this in and generate a plan --> sample plan 

"""
