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
SCORE = 'score'
PROXIMITY = 'proximity'
ANSWER = 'answer'
# types
CRITICAL = 'critical'
WARNING = 'warning'
NOTE = 'note'
DONE = 'done'



class Divide21X(Grader):
    def __init__(self, action=None, state=None):
        super().__init__(action, state)
        
        self.proximity = 0
        self.model = None
        
        # Logging
        self.logger = EpisodeLogger(BASE_DIR)
    
    def start(self):
        self.proximity = self.grade_submission2()
        
        self.logger.add_info(ENVIRONMENT, DONE, True)
        self.logger.add_info(ENVIRONMENT, PROXIMITY, self.proximity)
        # log
        if self.logger.info not in self.logger.episode_log:
            self.logger.episode_log.append(self.logger.info)
        
        self.logger.save_episode()
    
    def get_proximity(self):
        return self.proximity
    

def handle_averages():
    for metric in [PROXIMITY, SCORE]:
        date = str(get_utc_date())
        day = str(get_utc_day())
        metric_data = {}
        for root, dirs, files in os.walk(RESULTS_DIR):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        for alias, value in data.items():
                            if alias not in metric_data:
                                metric_data[alias] = []
                            metric_data[alias].append(value[metric])
        
        # sort in descending order
        metric_data = sorted(metric_data.items(), key=lambda x: (sum(x[1]) / len(x[1])), reverse=True)
        metric_data = dict(metric_data)
        
        # write averages to csv
        average_metric_file = os.path.join(LEADERBOARDS_DIR, date[:7], f'average_{metric}.csv')
        with open(average_metric_file, mode="w", newline="") as f:
            # write header
            header = ["Model", "Provider", f"Average {metric.capitalize()} (%)"]
            writer = csv.writer(f)
            writer.writerow(header)
            for alias, metric_value in metric_data.items():
                average_metric = sum(metric_value) / len(metric_value)
                # make sure it's a percentage
                if metric == SCORE:
                    average_metric = average_metric * 100
                # round to 2 decimal places
                average_metric = round(average_metric, 2)
                # get provider
                provider = None
                registry = get_llm_registry()
                for entry in registry:
                    if entry['alias'] == alias:
                        provider = entry['provider']
                        break
                # write row
                writer.writerow([alias, provider, average_metric])

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
            # add proximity
            value[PROXIMITY] = divide21x.get_proximity()
            # add score
            #   because Divide21X is deterministic, only 100% proximity (meaning exact match to the correct answer) gets a score of 1
            value[SCORE] = 1 if value[PROXIMITY] == 100 else 0
            # for leaderboard
            provider = None
            registry = get_llm_registry()
            for entry in registry:
                if entry['alias'] == key:
                    provider = entry['provider']
                    break
            leaderboard_data.append([key, provider, value[PROXIMITY], value[SCORE]])
        
        # update the file
        with open(file, 'w') as f:
            json.dump(data, f, indent=4)
        
        # sort leaderboard data by proximity descending
        leaderboard_data.sort(key=lambda x: x[2],  reverse=True)
        # create leaderboard csv file
        with open(leaderboard_file, mode="w", newline="") as f:
            header = ["Model", "Provider", "Proximity (%)", "Score (0/1)"]
            leaderboard_data.insert(0, header)
            writer = csv.writer(f)
            writer.writerows(leaderboard_data)
        
        # handle averages
        handle_averages()