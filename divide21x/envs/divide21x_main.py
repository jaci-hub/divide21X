import gymnasium as gym
from gymnasium import spaces
import divide21env
import json
import os
from divide21x.grading.grader import Grader
from divide21x.utils.logger import EpisodeLogger
from divide21x.utils.util import get_utc_date, get_utc_day, get_utc_hour


BASE_DIR='./divide21x/envs/logs'
RESULTS_DIR = './divide21x/results'
# categories
ENVIRONMENT = 'environment'
RUN = 'run'
RESULT = 'result'
ANSWER = 'answer'
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
    


if __name__ == "__main__":
    # navigate the results dir
    date = str(get_utc_date())
    day = str(get_utc_day())
    
    # get the results file
    data = None
    file_name = day + '.json'
    file = os.path.join(RESULTS_DIR, file_name)
    
    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.load(f)
        
    if data:
        for key, value in data.items():
            divide21x = Divide21X(state=value[ANSWER])
            divide21x.start()
            # add result
            value[RESULT] = divide21x.get_result()
    
        # update the file
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)
    