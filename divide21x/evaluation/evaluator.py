"""
Author: Jacinto Jeje Matamba Quimua
Date: 11/01/2025

Description:
------------
Evaluation module for Divide21X Phase 1: Action-only benchmark environment
for faithful strategic reasoning.

This module checks the inspection result for the action and the state. 
    (1) If failed inspection, then they are redirected to the appropriate graders
        else, the action-state implication/generation is evaluated
            if implication/generation is False, then points are deducted
    (2) They are compared to the ground truth action and state.
        Then redirected to the appropriate graders
"""
import gymnasium as gym
from gymnasium import spaces
import divide21env
from divide21env.envs.divide21_env import Divide21Env
from divide21x.inspection.inspector import Inspector
from divide21x.simulator.divide21env_simulator import Divide21EnvSimulator
import numpy as np
import math
from divide21x.utils.logger import EpisodeLogger


BASE_DIR='./divide21x/evaluation/logs'
# categories
ACTION = 'action'
STATE = 'state'
STATE_COMPARISON = 'state_comparison'
# types
CRITICAL = 'critical'
WARNING = 'warning'
NOTE = 'note'
SCORE = 'score'
EQUIVALENT = 'equivalent'

class Evaluator(Inspector):
    """
    Evaluates action-only submissions (no explanations) from LLMs against
    the Divide21 ground-truth agent.

    Usage:
    -------
    >>> evaluator = Evaluator(action, state)
    >>> action_passed_inspection = evaluator.action_passed() # via Inspector
    >>> print(action_passed_inspection)
    True
    >>> state_inspection_score = evaluator.get_state_score() # via Inspector
    >>> print(state_inspection_score)
    17
    """
    def __init__(self, action=None, state=None):
        super().__init__(action, state)
        self.inspect_all()
        
        self.generated_state = None
                
        # Logging
        self.logger = EpisodeLogger(BASE_DIR)
        
    def evaluate(self):
        pass
    
    def compare_states(self, given_state1=None, given_state2=None):
        '''
        compares two states from Divide21 game, and checks if they are equivalent.
        Note: it cant just do a comparison with "==" because since both of them have lists (the keys of 'available_digits_per_rindex', and also the 'players' key), 
                the lists might not be in the same order but still have all the correct elements,
                which would result in a False negative!
        returns: 
            tuple(states_are_equivalent, similarity_score): 
                states_are_equivalent(bool): True if the states are equivalent, meaning that they have the exact same values, in the same amount, disregarding the order.
                similarity_score(float): how similar the states are on a scale of 0-100:
                    If 0, they are completely different
                    If 100, they are equivalent
        
        state format example:
            state = {
                "static_number": 19,
                "dynamic_number": 59,
                "available_digits_per_rindex": {0: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [2, 3, 4, 6, 7, 8, 9]},
                "players": [{'id': 0, 'score': -13, 'is_current_turn': 1}],
                "player_turn": 0
            }
        '''
        state1 = None
        state2 = None
        
        # setup states 1 and 2, that will be used for comparison
        if given_state1 is None:
            state1 = self.state
        else:
            state1 = given_state1
        if given_state2 is None:
            state2 = self.generated_state
        else:
            state2 = given_state2
            
        # Start comparison of states 1 and 2
        if state1 is None or state2 is None:
            message = "Only one state was provided."
            self.logger.add_info(STATE_COMPARISON, WARNING, message)
            # log
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
            return (False, 0.0)
        
        # --- Early shape check ---
        expected_keys = {'static_number', 'dynamic_number', 'available_digits_per_rindex', 'players', 'player_turn'}
        if set(state1.keys()) != expected_keys or set(state2.keys()) != expected_keys:
            message = f"A state dictionary must have exactly these keys: {', '.join(expected_keys)}."
            self.logger.add_info(STATE_COMPARISON, WARNING, message)
            # log
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
            return (False, 0.0)

        total_weight = 5  # number of components
        matches = 0

        # (1) static_number
        if state1["static_number"] == state2["static_number"]:
            matches += 1

        # (2) dynamic_number
        if state1["dynamic_number"] == state2["dynamic_number"]:
            matches += 1

        # (3) available_digits_per_rindex
        adpr1 = state1["available_digits_per_rindex"]
        adpr2 = state2["available_digits_per_rindex"]
        if isinstance(adpr1, dict) and isinstance(adpr2, dict):
            if set(adpr1.keys()) == set(adpr2.keys()):
                per_key_match = True
                for k in adpr1.keys():
                    v1 = sorted(adpr1[k]) if isinstance(adpr1[k], list) else []
                    v2 = sorted(adpr2[k]) if isinstance(adpr2[k], list) else []
                    if v1 != v2:
                        per_key_match = False
                        break
                if per_key_match:
                    matches += 1

        # (4) players
        players1 = state1["players"]
        players2 = state2["players"]
        if isinstance(players1, list) and isinstance(players2, list) and len(players1) == len(players2):
            # normalize players by sorting by id (or JSON string sort for safety)
            try:
                norm1 = sorted(players1, key=lambda x: (x["id"], x["score"], x["is_current_turn"]))
                norm2 = sorted(players2, key=lambda x: (x["id"], x["score"], x["is_current_turn"]))
                if all(p1 == p2 for p1, p2 in zip(norm1, norm2)):
                    matches += 1
            except Exception:
                pass

        # (5) player_turn
        if state1["player_turn"] == state2["player_turn"]:
            matches += 1

        # --- Compute result ---
        similarity_score = round((matches / total_weight) * 100, 2)
        equivalent = similarity_score == 100.0
        
        self.logger.add_info(STATE_COMPARISON, EQUIVALENT, equivalent)
        self.logger.add_info(STATE_COMPARISON, SCORE, similarity_score)

        # log
        if self.logger.info not in self.logger.episode_log:
            self.logger.episode_log.append(self.logger.info)
                
        return equivalent, similarity_score
        
    
    def action_generates_state(self):
        '''
        checks if the given action implies/generates the given state
        '''
        # get the generated state by applying the given action on the given state
        divide21env_simulator = Divide21EnvSimulator(given_obs=self.state)
        obs, reward, terminated, truncated, info = divide21env_simulator.step(self.action)
        self.generated_state = divide21env_simulator._decode_state(obs)
        
        # compare the given state to the generated state
        #   note: cant just do a comparison with "==" because since both of them have lists, 
        #         the lists might not be in the same order but still have all the correct elements
        #         which would result in a False negative! So, a new function specialized for this is needed.
        equivalent, similarity_score = self.compare_states()
    