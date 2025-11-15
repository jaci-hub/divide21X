import csv
import gymnasium as gym
from gymnasium import spaces
import divide21env
import json
import os
from divide21x.grading.grader import Grader
from divide21x.utils.logger import EpisodeLogger
from divide21x.utils.util import get_llm_registry, get_utc_date, get_utc_day, get_utc_hour


BASE_DIR='./divide21x/envs/logs'
RESULTS_DIR = './divide21x/results'
LEADERBOARDS_DIR = './divide21x/leaderboards'
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
    results_path = os.path.join(RESULTS_DIR, date[:7])
    file = os.path.join(results_path, file_name)
    # for leaderboard
    leaderboard_file_name = day + '.csv'
    leaderboards_path = os.path.join(LEADERBOARDS_DIR, date[:7])
    os.makedirs(leaderboards_path, exist_ok=True)
    leaderboard_file = os.path.join(leaderboards_path, leaderboard_file_name)
    
    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.load(f)
        
    if data:
        leaderboard_data = []
        for key, value in data.items():
            divide21x = Divide21X(state=value[ANSWER])
            divide21x.start()
            # add result
            value[RESULT] = divide21x.get_result()
            # for leaderboard
            provider = None
            registry = get_llm_registry()
            for entry in registry:
                if entry['alias'] == key:
                    provider = entry['provider']
                    break
            leaderboard_data.append([key, provider, value[RESULT]])
        
        # update the file
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)
        
        # sort leaderboard data by result descending
        leaderboard_data.sort(key=lambda x: x[2],  reverse=True)
        # create leaderboard csv file
        with open(leaderboard_file, 'w') as f:
            header = ["Model", "Provider", "Score"]
            leaderboard_data.insert(0, header)
            writer = csv.writer(f)
            writer.writerows(leaderboard_data)