import divide21env
import json
import os
import numpy as np
from divide21x.utils.logger import EpisodeLogger

# base dir
BASE_DIR = './divide21x/inspection/logs'
# categories
ACTION = 'action'
STATE = 'state'
# types
CRITICAL = 'critical'
WARNING = 'warning'
NOTE = 'note'
SCORE = 'score'

class Inspector():
    def __init__(self, action=None, state=None):
        self.action = action
        # action keys
        self.division = None
        self.digit = None
        self.rindex = None
        
        self.state = state
        # state keys
        self.static_number = None
        self.dynamic_number = None
        self.available_digits_per_rindex = None
        self.players = None
        self.player_turn = None
        
        # scores
        self.action_score = 10
        self.state_score = 40
        self.overall_score = self.action_score + self.state_score
        self.action_passing_score = 10
        self.state_passing_score = 40
        self.overall_passing_score = self.action_passing_score + self.state_passing_score
                
        # Logging
        self.logger = EpisodeLogger(BASE_DIR)
    
    def get_action(self):
        return self.action
    
    def get_state(self):
        return self.state
    
    def inspect_action(self):
        '''
        inspect the action, to ensure it follows the format of the game Divide21
        '''
        # check action
        expected_keys = {'division', 'digit', 'rindex'}
        if not isinstance(self.action, dict):
            self.action_score -= 10
            message = "Action must be a Python dictionary."
            self.logger.add_info(ACTION, CRITICAL, message)
        elif set(self.action.keys()) != expected_keys:
            self.action_score -= 9
            message = f"Action dictionary must have exactly these keys: {', '.join(expected_keys)}."
            self.logger.add_info(ACTION, CRITICAL, message)
        else:
            # get key values
            if self.action["division"] in [0, 1, True, False]:
                self.division = bool(self.action["division"])
            if self.action["digit"] in range(0, 10):
                self.digit = int(self.action["digit"])
            if (isinstance(self.action["rindex"], (int, np.integer)) and self.action["rindex"] >= 0):
                self.rindex = int(self.action["rindex"])

            # check division
            if self.division is None:
                self.action_score -= 7
                message = "The value for the division attribute must be either True or False, or 1 or 0."
                self.logger.add_info(ACTION, CRITICAL, message)
            # check digit
            elif self.digit is None:
                self.action_score -= 7
                message = "Digit must be between 0-9."
                self.logger.add_info(ACTION, CRITICAL, message)
            # check rindex
            elif self.rindex is None and self.division is None:
                self.action_score -= 7
                message = "*Rindex must be an integer greater than or equal to 0."
                self.logger.add_info(ACTION, CRITICAL, message)
                
            # Division attempt
            elif self.division:
                # deduct points if rindex is not None
                if self.rindex != None:
                    self.action_score -= 2
                    message = "Rindex should have not been provided!"
                    self.logger.add_info(ACTION, WARNING, message)
                    
        message = self.action_score
        self.logger.add_info(ACTION, SCORE, message)
        # log
        if self.logger.info not in self.logger.episode_log:
            self.logger.episode_log.append(self.logger.info)
    
    def inspect_state(self):
        '''
        inspect the state, to ensure it follows the format of the game Divide21
        '''
        # check state
        expected_keys = {'static_number', 'dynamic_number', 'available_digits_per_rindex', 'players', 'player_turn'}
        if not isinstance(self.state, dict):
            self.state_score -= 40
            message = "State must be a Python dictionary."
            self.logger.add_info(STATE, CRITICAL, message)
        elif set(self.state.keys()) != expected_keys:
            self.state_score -= 38
            message = f"State dictionary must have exactly these keys: {', '.join(expected_keys)}."
            self.logger.add_info(STATE, CRITICAL, message)
        else:
            # get attributes
            #   (0) check static_number
            if (isinstance(self.state["static_number"], (int, np.integer)) and self.state["static_number"] > 0):
                self.static_number = self.state["static_number"]
            else:
                self.state_score -= 7
                message = "The original number must be a non-negative integer."
                self.logger.add_info(STATE, CRITICAL, message)
            
            #   (1) check dynamic_number
            if (isinstance(self.state["dynamic_number"], (int, np.integer)) and self.state["dynamic_number"] > 0):
                self.dynamic_number = self.state["dynamic_number"]
            else:
                self.state_score -= 7
                message = "The number being manipulated must be a non-negative integer."
                self.logger.add_info(STATE, CRITICAL, message)
            
            #   (2) check available digits per rindex
            field = "available_digits_per_rindex"
            value = self.state.get(field, None)
            # (2.1) Must be a dictionary
            if not isinstance(value, dict):
                self.state_score -= 7
                message = f"'{field}' must be a Python dictionary."
                self.logger.add_info(STATE, CRITICAL, message)
                return
            # (2.2) Must not be empty
            if len(value) == 0:
                self.state_score -= 6
                message = f"'{field}' dictionary must not be empty."
                self.logger.add_info(STATE, CRITICAL, message)
                return
            # (2.3) Validate keys and values
            for k, v in value.items():
                # Key must be an integer ≥ 0
                if isinstance(k, (int, np.integer)) and 0 <= k < len(str(self.static_number)) == False:
                    self.state_score -= 5
                    message = f"Key '{k}' in '{field}' must be a non-negative integer."
                    self.logger.add_info(STATE, CRITICAL, message)
                    break
                # Value must be a list
                if not isinstance(v, list):
                    self.state_score -= 5
                    message = f"Value for key '{k}' in '{field}' must be a Python list."
                    self.logger.add_info(STATE, CRITICAL, message)
                    break
                # Each element must be a unique integer digit between 0–9
                if not all(isinstance(d, (int, np.integer)) and 0 <= d <= 9 for d in v):
                    self.state_score -= 4
                    message = f"All elements in '{field}[{k}]' must be digits between 0 and 9."
                    self.logger.add_info(STATE, CRITICAL, message)
                    break
                if len(v) != len(set(v)):
                    self.state_score -= 3
                    message = f"Duplicate digits found in '{field}[{k}]'."
                    self.logger.add_info(STATE, WARNING, message)
            # (2.4) Assign valid value
            self.available_digits_per_rindex = value
            
            #   (3) check players
            if isinstance(self.state["players"], list):
                # check the length
                if len(self.state["players"]) == 0:
                    self.state_score -= 6
                    message = "players list must have at least one player."
                    self.logger.add_info(STATE, CRITICAL, message)
                else:
                    # check the player keys
                    expected_players_key = {'id', 'score', 'is_current_turn'}
                    for player in self.state["players"]:
                        if set(player.keys()) != expected_players_key:
                            self.state_score -= 5
                            message = f"player must have exactly these keys: {', '.join(expected_players_key)}."
                            self.logger.add_info(STATE, CRITICAL, message)
                            break
                        else:
                            # check the value of the keys
                            #   id
                            if not (isinstance(player["id"], (int, np.integer)) and 0 <= player["id"] < len(self.state["players"])):
                                self.state_score -= 4
                                message = "The player id must be a non-negative integer less than the number of players."
                                self.logger.add_info(STATE, CRITICAL, message)
                                break
                            #   score
                            elif not (isinstance(player["score"], (int, np.integer)) and -9*len(str(self.static_number)) - 8 <= player["score"] <= 9*len(str(self.static_number)) + 8):
                                self.state_score -= 4
                                message = "The player score, s, must satisfy: -9*(the original number of digits) - 8 <= s <= 9*(the original number of digits) + 8."
                                self.logger.add_info(STATE, CRITICAL, message)
                                break
                            #   is_current_turn
                            elif not (isinstance(player["is_current_turn"], (int, np.integer)) and 0 <= player["is_current_turn"] <= 1):
                                self.state_score -= 4
                                message = "is_current_turn must be 1 or 0, which means that it is the player's turn or not, respectivelly."
                                self.logger.add_info(STATE, CRITICAL, message)
                                break
                            # assign valid value for players
                            self.players = self.state["players"]
            else:
                self.state_score -= 7
                message = "players must be a Python list."
                self.logger.add_info(STATE, CRITICAL, message)
            
            #   (4) check player_turn
            if self.players is not None:
                if (isinstance(self.state["player_turn"], (int, np.integer)) and 0 <= self.state["player_turn"] < len(self.players)):
                    self.player_turn = int(self.state["player_turn"])
            if self.player_turn is None:
                self.state_score -= 7
                message = "The player turn must be a non-negative integer less than the number of players."
                self.logger.add_info(STATE, CRITICAL, message)
                
        message = self.state_score
        self.logger.add_info(STATE, SCORE, message)
        # log
        if self.logger.info not in self.logger.episode_log:
            self.logger.episode_log.append(self.logger.info)
    
    def inspect_all(self):
        '''
        inspect both the action and the state, to ensure they follow the format of the game Divide21
        '''
        self.inspect_action()
        self.inspect_state()
        self.logger.save_episode()
    
    def get_action_score(self):
        return self.action_score
    
    def action_passed(self):
        return self.action_passing_score == self.action_score
    
    def get_state_score(self):
        return self.state_score
    
    def state_passed(self):
        return self.state_passing_score == self.state_score
    
    def get_overall_score(self):
        return self.overall_score
    
    def all_passed(self):
        return self.overall_score == self.overall_passing_score
    
    def get_action_passing_score(self):
        return self.action_passing_score
    
    def get_state_passing_score(self):
        return self.state_passing_score
    
    def get_overall_passing_score(self):
        return self.overall_passing_score
