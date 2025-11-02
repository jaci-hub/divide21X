import gymnasium as gym
from gymnasium import spaces
import divide21env
import json
import os
from divide21x.utils.logger import EpisodeLogger


class Divide21XActionOnly(gym.Env):
    """
    Phase 1 Divide21X environment:
    Agents submit actions only — no explanations yet.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.base_env = gym.make("Divide21-v0")
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self.render_mode = render_mode

        # Logging
        self.logger = EpisodeLogger()

    def reset(self, *, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self.state = obs
        self.logger = EpisodeLogger()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.base_env.step(action)

        # Log transition
        self.logger.episode_log.append({
            "state before action": self.state,
            "action": action,
            "state after action": obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated
        })
        
        # Update the state
        self.state = obs

        # Append placeholder for future explanations
        info["explanation"] = "Not required in this version of Divide21X."
        
        self.logger.save_episode()

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

