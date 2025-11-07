import gymnasium as gym
from gymnasium import spaces
import divide21env
import json
import os
from divide21x.grading.grader import Grader
from divide21x.utils.logger import EpisodeLogger


BASE_DIR='./divide21x/envs/logs'
# categories
ENVIRONMENT = 'environment'
RUN = 'run'
RESULT = 'result'
# types
CRITICAL = 'critical'
WARNING = 'warning'
NOTE = 'note'
DONE = 'done'



class Divide21X(Grader):
    def __init__(self, action=None, state=None):
        super().__init__(action, state)
        
        self.result = 0
        self.model = None
        
        # Logging
        self.logger = EpisodeLogger(BASE_DIR)
    
    def start(self):
        self.result = self.grade_submission2()
        
        self.logger.add_info(ENVIRONMENT, DONE, True)
        self.logger.add_info(ENVIRONMENT, RESULT, self.result)
        # log
        if self.logger.info not in self.logger.episode_log:
            self.logger.episode_log.append(self.logger.info)
        
        self.logger.save_episode()
    
    def get_result(self):
        return self.result
    