import json
import os
from datetime import datetime

class EpisodeLogger:
    def __init__(self, base_dir="./logs"):
        os.makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir
        self.episode = 0
        self.episode_log = []

    def save_episode(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.base_dir, f"episode_{self.episode}_{ts}.json")
        with open(path, "w") as f:
            json.dump(self.episode_log, f, indent=2)
        self.episode += 1
        return path
