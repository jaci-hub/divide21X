"""
Author: Jacinto Jeje Matamba Quimua
Date: 11/01/2025

Description:
------------
Grader module for Divide21X Phase 1: Action-State benchmark environment
for faithful strategic reasoning.

Grading is based on:
    (1) Action Fidelity — how closely the LLM's chosen action matches the ground-truth action.
    (2) State Fidelity  — how similar the resulting next-state is to the ground-truth next-state.

Both are combined to form an overall Divide21X Phase 1 score.
"""
import divide21env
from divide21env.envs.divide21_env import Divide21Env
import numpy as np
import math
from typing import Dict, Any, Tuple
from divide21x.evaluation.evaluator import Evaluator
from divide21x.utils.logger import EpisodeLogger


BASE_DIR='./divide21x/grading/logs'
# categories
ACTION = 'action'
STATE = 'state'
ACTION_STATE = 'action_state'
# types
CRITICAL = 'critical'
WARNING = 'warning'
NOTE = 'note'
SCORE = 'score'
EQUIVALENT = 'equivalent'
DEDUCTION_POINTS = 'deduction_points'


class Grader(Evaluator):
    """
    Evaluates action-state submissions (no explanations) from LLMs against
    the Divide21 ground-truth agent.

    Usage:
    -------
    >>> grader = Grader()
    >>> result = grader.grade_submission()
    >>> print(result)
    {'action_grade': 80, 'state_grade': 90, 'overall_grade': 85}
    """

    def __init__(self, action=None, state=None):
        super().__init__(action, state)
        
        self.evaluate()
        
        self.action_grade = 0
        self.state_grade = 0
        self.overall_grade = 0
        
        # Logging
        self.logger = EpisodeLogger(BASE_DIR)
        
    def grade_submission(self):
        """
        grade an LLM submission (action + state) against ground truth.

        Returns
        -------
        dict
            {
                "action_grade": float,
                "state_grade": float,
                "overall_grade": float
            }
        """

        return {
            "action_grade": round(float(self.action_grade), 2),
            "state_grade": round(float(self.state_grade), 2),
            "overall_grade": round(float(self.overall_grade), 2),
        }
