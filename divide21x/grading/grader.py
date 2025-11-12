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
GRADE = 'grade'


class Grader(Evaluator):
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
        
        # check inspection results
        #   (1) Action-State passed
        if self.action_passed() and self.state_passed():
            self.action_grade = max(0, self.ground_truth_action_score - self.points_to_deduct/2)
            self.state_grade = max(0, self.ground_truth_state_score - self.points_to_deduct/2)
            message = "Both action and state passed the inspection."
            self.logger.add_info(ACTION_STATE, NOTE, message)
            self.logger.add_info(ACTION, GRADE, self.action_grade)
            self.logger.add_info(STATE, GRADE, self.state_grade)
            
        #   (2) Only Action passed
        elif self.action_passed() and not self.state_passed():
            self.action_grade = self.ground_truth_action_score
            self.state_grade = 0
            message = "Only the action passed the inspection."
            self.logger.add_info(ACTION_STATE, WARNING, message)
            self.logger.add_info(ACTION, GRADE, self.action_grade)
            self.logger.add_info(STATE, GRADE, self.state_grade)
        
        #   (3) Only State passed
        elif not self.action_passed() and self.state_passed():
            self.action_grade = 0
            self.state_grade = self.ground_truth_state_score
            message = "Only the state passed the inspection."
            self.logger.add_info(ACTION_STATE, WARNING, message)
            self.logger.add_info(ACTION, GRADE, self.action_grade)
            self.logger.add_info(STATE, GRADE, self.state_grade)
        
        #   (4) Action-State did not pass
        elif not self.action_passed() and not self.state_passed():
            self.action_grade = 0
            self.state_grade = 0
            message = "Both the action and the state failed the inspection."
            self.logger.add_info(ACTION_STATE, CRITICAL, message)
            self.logger.add_info(ACTION, GRADE, self.action_grade)
            self.logger.add_info(STATE, GRADE, self.state_grade)

        # overall grade
        self.overall_grade = (self.action_grade + self.state_grade)/2
        self.logger.add_info(ACTION_STATE, GRADE, self.overall_grade)
        
        # log
        if self.logger.info not in self.logger.episode_log:
            self.logger.episode_log.append(self.logger.info)
        self.logger.save_episode()
        
        return {
            "action_grade": round(float(self.action_grade), 2),
            "state_grade": round(float(self.state_grade), 2),
            "overall_grade": round(float(self.overall_grade), 2),
        }
    
    def grade_submission2(self):
        """
        grade an LLM submission state against ground truth.

        Returns
        -------
        float
        """
        
        # check inspection results
        #   (1) State passed
        if self.state_passed():
            self.state_grade = max(0, self.ground_truth_state_score - self.points_to_deduct)
            message = "State passed the inspection."
            self.logger.add_info(STATE, NOTE, message)
            self.logger.add_info(STATE, GRADE, self.state_grade)
            
        #   (2) State failed
        else:
            self.state_grade = 0
            message = "State failed the inspection."
            self.logger.add_info(STATE, WARNING, message)
            self.logger.add_info(STATE, GRADE, self.state_grade)
            
        # log
        if self.logger.info not in self.logger.episode_log:
            self.logger.episode_log.append(self.logger.info)
        self.logger.save_episode()
        
        return round(float(self.state_grade), 2)
