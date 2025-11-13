import re
import gymnasium as gym
from gymnasium import spaces
import divide21env
import json
import os
from divide21x.grading.grader import Grader
from divide21x.utils.logger import EpisodeLogger
from divide21x.utils.util import get_utc_date, get_utc_hour


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
    hour = str(get_utc_hour())
    daily_results_dir = os.path.join(RESULTS_DIR, date)
    
    # get the results file for the hour
    data = None
    file_name = hour + '.json'
    file = os.path.join(daily_results_dir, file_name)
    
    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.load(f)
        
    if data:
        for key, value in data.items():
            # clean answer
            # remove the Markdown ```json fences
            value[ANSWER] = re.sub(r"^```json|```$", "", value[ANSWER].strip())
            # unescape backslashes
            value[ANSWER] = value[ANSWER].encode('utf-8').decode('unicode_escape')

            # load into a Python dict
            value[ANSWER] = json.loads(value[ANSWER])
            
            divide21x = Divide21X(state=value[ANSWER])
            divide21x.start()
            # add result
            value[RESULT] = divide21x.get_result()
    
        # update the file
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)
    