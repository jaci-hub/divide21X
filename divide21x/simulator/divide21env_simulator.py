import gymnasium as gym
from gymnasium import spaces
import divide21env
from divide21env.envs.divide21_env import Divide21Env
import numpy as np
import math
from divide21x.utils.logger import EpisodeLogger


BASE_DIR='./divide21x/simulator/logs'


class Divide21EnvSimulator(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, digits=2, players=1, render_mode=None, auto_render=False, given_obs=None):
        super().__init__()
        self.base_env = gym.make("Divide21-v0", digits=digits, players=players, render_mode=render_mode, auto_render=auto_render)
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self.render_mode = render_mode
        self.auto_render = auto_render
        
        self.given_obs = given_obs

        # Logging
        self.logger = EpisodeLogger(BASE_DIR)
    
    def reset(self, *, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        self.state = obs
        self.logger = EpisodeLogger(BASE_DIR)
        return obs, info

    def _decode_dynamic_number(self, dynamic_number):
        decoded_dynamic_number = [str(d) for d in dynamic_number.tolist()]
        decoded_dynamic_number = ''.join(decoded_dynamic_number)
        decoded_dynamic_number = int(decoded_dynamic_number)
        return decoded_dynamic_number
    
    def _decode_available_digits(self, flat_mask, digits):
        '''
        Reconstruct the available_digits_per_rindex dictionary from the
        flattened mask produced by _encode_available_digits().

        Args:
            flat_mask (array-like): Flattened binary mask of shape (self.digits * 10,)

        Returns:
            dict[int, list[int]]: Dictionary mapping each rindex to its list of available digits.
        '''
        # Ensure it's a list of ints
        flat_list = list(flat_mask)
        # Rebuild mask as 2D matrix (digits x 10)
        mask_2d = [flat_list[i * 10 : (i + 1) * 10] for i in range(digits)]
        
        # Decode dictionary
        decoded = {}
        for rindex, row in enumerate(mask_2d):
            available_digits = [digit for digit, flag in enumerate(row) if flag == 1]
            decoded[rindex] = available_digits

        return decoded
    
    def _decode_players(self, flat_array):
        '''
        Reconstruct the list of player dictionaries from the flattened NumPy array
        produced by _encode_players().

        Args:
            flat_array (array-like): Flattened 1D list or NumPy array of shape (num_players * 3,).

        Returns:
            list[dict]: List of players, each as {"id": int, "score": int, "is_current_turn": int}.
        '''
        # Ensure it's a plain list of ints
        flat_list = list(flat_array)
        
        # Each player has 3 attributes: [id, score, is_current_turn]
        num_players = len(flat_list) // 3
        
        players = []
        for i in range(num_players):
            start = i * 3
            pid, score, turn_flag = flat_list[start:start + 3]
            player = {
                "id": int(pid),
                "score": int(score),
                "is_current_turn": int(turn_flag)
            }
            players.append(player)
        
        return players
    
    def _decode_player_turn(self, player_turn):
        return int(player_turn)
    
    def _decode_state(self, state):
        '''
        the observation space attributes from Divide21Env are in numpy variables which are not json compatible, 
        so make sure they are.
        '''
        decoded_static_number = self._decode_dynamic_number(state["static_number"])
        decoded_dynamic_number = self._decode_dynamic_number(state["dynamic_number"])
        decoded_state = {
            "static_number": decoded_static_number,
            "dynamic_number": decoded_dynamic_number,
            "available_digits_per_rindex": self._decode_available_digits(state["available_digits_per_rindex"], len(str(decoded_dynamic_number))),
            "players": self._decode_players(state["players"]),
            "player_turn": self._decode_player_turn(state["player_turn"])
        }
        
        return decoded_state
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.base_env.step(action)

        # Decode states
        #   (1)
        state_before_action_decoded = self._decode_state(self.state)
        #   (2)
        state_after_action_decoded = self._decode_state(obs)
        
        action_log = {
            "division": bool(action["division"]),
            "digit": int(action["digit"]),
            "rindex": int(action["rindex"]) if not bool(action["division"]) else None
        }
        
        # Log transition
        self.logger.episode_log.append({
            "state_before_action": state_before_action_decoded,
            "action": action_log,
            "state_after_action": state_after_action_decoded,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        })
        
        # Update the state
        self.state = obs
        
        # log episode
        self.logger.save_episode()

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

