import gymnasium as gym
from gymnasium import spaces
import divide21env
import json
import os
from divide21x.grading.grader import Grader


class Divide21X(Grader):
    def __init__(self, action=None, state=None):
        super().__init__(action, state)
        
    def start(self):
        self.grade_submission()