import json
import os
from datetime import datetime

SCORE = 'score'

class EpisodeLogger:
    def __init__(self, base_dir="./logs"):
        os.makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir
        self.info = {}
        self.episode = 0
        self.episode_log = []

    def add_info(self, category=None, type=None, message=None):
        if category not in self.info:
            self.info[category] = {}

        if type not in self.info[category]:
            if type != SCORE:
                self.info[category][type] = []
                self.info[category][type].append(message)
            else:
                self.info[category][type] = message
        else:
            if type != SCORE:
                self.info[category][type].append(message)
            else:
                self.info[category][type] = message
    
    def save_episode(self):
        if not self.episode_log:
            return
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.base_dir, f"episode_{self.episode}_{ts}.json")
        with open(path, "w") as f:
            json.dump(self.episode_log, f, indent=2)
        self.episode += 1
        return path
