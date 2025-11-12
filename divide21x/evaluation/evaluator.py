import json
import os
import gymnasium as gym
from gymnasium import spaces
import divide21env
from divide21env.envs.divide21_env import Divide21Env
from divide21x.challenge_maker.challenge_maker import ChallengeMaker
from divide21x.inspection.inspector import Inspector
from divide21x.simulator.divide21env_simulator import Divide21EnvSimulator
import numpy as np
import math
from divide21x.utils.logger import EpisodeLogger
from divide21x.ground_truth.ground_truth import GroundTruth
from divide21x.utils.util import get_utc_date, get_utc_datetime, get_utc_hour


BASE_DIR='./divide21x/evaluation/logs'
CHALLENGES_DIR = './divide21x/challenges'
# categories
ACTION = 'action'
STATE = 'state'
ACTION_STATE = 'action_state'
ACTION_COMPARISON = 'action_comparison'
STATE_COMPARISON = 'state_comparison'
CHALLENGE = 'challenge'
# types
CRITICAL = 'critical'
WARNING = 'warning'
NOTE = 'note'
SCORE = 'score'
EQUIVALENT = 'equivalent'
DEDUCTION_POINTS = 'deduction_points'

class Evaluator(Inspector):
    """
    Evaluates action-State submissions (no explanations) from LLMs against
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
        self.points_to_deduct = 0
        self.ground_truth_action_score = 0
        self.ground_truth_state_score = 0
        
        # Logging
        self.logger = EpisodeLogger(BASE_DIR)
        
    def evaluate(self):
        # check inspection results
        if self.action_passed() and self.state_passed():
            # check if action implies/generates state
            states_are_equivalent, states_similarity_score = self.action_generates_state()
            if not states_are_equivalent:
                message = 'Action does not generate state!'
                self.logger.add_info(WARNING, ACTION_STATE, message)
                # set points to deduct
                self.points_to_deduct = 100 - states_similarity_score
                self.logger.add_info(WARNING, DEDUCTION_POINTS, self.points_to_deduct)
            
            # compare to ground truth
            self.compare_to_ground_truth()
            
        elif self.state_passed():
            '''
            ---------- ONLY THIS PART WILL RUN BECAUSE ACTION=None ----------
                THIS IS FOR TESTING Problem design:
                    State_x + Action_y = State_z
                    State_a + Action_b = ?
            '''
            # compare to ground truth
            self.compare_to_ground_truth2()
        
        # save logs
        self.logger.save_episode()
    
    def compare_actions(self, given_action1=None, given_action2=None):
        '''
        compares two actions from Divide21 game, and checks if they are equivalent.
        Note: it cant just do a comparison with "==" because it does not consider closeness. 
            For example, if action1 and action2 only differ in one key-value, the similarity score should be higher than
            if they differed in all three.
        returns: 
            tuple(actions_are_equivalent, similarity_score): 
                actions_are_equivalent(bool): True if the actions are equivalent, meaning that they have the exact same value.
                similarity_score(float): how similar the actions are on a scale of 0-100:
                    If 0, they are completely different
                    If 100, they are equivalent
        
        action format example:
            action = {
                "v": 1, # it can be True
                "g": 2,
                "r": None
            }
        '''
        # --- Setup ---
        action1 = given_action1 if given_action1 is not None else None
        action2 = given_action2 if given_action2 is not None else None

        if action1 is None or action2 is None:
            message = "Two actions must be provided."
            self.logger.add_info(ACTION_COMPARISON, CRITICAL, message)
            # log
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
            return (False, 0.0)

        expected_keys = {"v", "g", "r"}
        if set(action1.keys()) != expected_keys or set(action2.keys()) != expected_keys:
            message = f"Action dictionary must have exactly these keys: {', '.join(expected_keys)}."
            self.logger.add_info(ACTION_COMPARISON, CRITICAL, message)
            # log
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
            return (False, 0.0)

        # --- Normalize ---
        def normalize(a):
            return {
                "v": bool(a.get("v", False)),
                "g": a.get("g", None),
                "r": a.get("r", None)
            }

        a1 = normalize(action1)
        a2 = normalize(action2)

        # --- Comparison ---
        total_keys = 3
        matches = 0

        # (1) division
        if a1["v"] == a2["v"]:
            matches += 1

        # (2) digit
        if a1["g"] == a2["g"]:
            matches += 1

        # (3) rindex
        if a1["r"] == a2["r"]:
            matches += 1

        # --- Compute final score ---
        similarity_score = round((matches / total_keys) * 100, 2)
        actions_are_equivalent = similarity_score == 100.0
        
        self.logger.add_info(ACTION_COMPARISON, EQUIVALENT, actions_are_equivalent)
        self.logger.add_info(ACTION_COMPARISON, SCORE, similarity_score)

        # log
        if self.logger.info not in self.logger.episode_log:
            self.logger.episode_log.append(self.logger.info)

        return actions_are_equivalent, similarity_score
    
    def compare_states(self, given_state1=None, given_state2=None):
        '''
        compares two states from Divide21 game, and checks if they are equivalent.
        Note: it cant just do a comparison with "==" because since both of them have lists (the keys of 'a', and also the 'p' key), 
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
                "s": 19,
                "d": 59,
                "a": {0: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [2, 3, 4, 6, 7, 8, 9]},
                "p": [{'i': 0, 'c': -13, 'm': 1}],
                "t": 0
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
            message = "Two states must be provided."
            self.logger.add_info(STATE_COMPARISON, CRITICAL, message)
            # log
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
            return (False, 0.0)
        
        # --- Early shape check ---
        expected_keys = {"s", "d", "a", "p", "t"}
        if set(state1.keys()) != expected_keys or set(state2.keys()) != expected_keys:
            message = f"State dictionary must have exactly these keys: {', '.join(expected_keys)}."
            self.logger.add_info(STATE_COMPARISON, CRITICAL, message)
            # log
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
            return (False, 0.0)

        total_weight = 5  # number of components
        matches = 0

        # (1) static_number
        if state1["s"] == state2["s"]:
            matches += 1

        # (2) dynamic_number
        if state1["d"] == state2["d"]:
            matches += 1

        # (3) available_digits_per_rindex
        adpr1 = state1["a"]
        adpr2 = state2["a"]
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
        players1 = state1["p"]
        players2 = state2["p"]
        if isinstance(players1, list) and isinstance(players2, list) and len(players1) == len(players2):
            # normalize players by sorting by id (or JSON string sort for safety)
            try:
                norm1 = sorted(players1, key=lambda x: (x["i"], x["c"], x["m"]))
                norm2 = sorted(players2, key=lambda x: (x["i"], x["c"], x["m"]))
                if all(p1 == p2 for p1, p2 in zip(norm1, norm2)):
                    matches += 1
            except Exception:
                pass

        # (5) player_turn
        if state1["t"] == state2["t"]:
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
        states_are_equivalent, states_similarity_score = self.compare_states()

        return states_are_equivalent, states_similarity_score
    
    def compare_to_ground_truth(self):
        '''
        checks if the action and state are optimal
        '''
        # get optimal action
        ground_truth = GroundTruth(self.state)
        ground_truth_action = ground_truth.get_action()
        # generate state from the optimal action
        divide21env_simulator = Divide21EnvSimulator(given_obs=self.state)
        obs, reward, terminated, truncated, info = divide21env_simulator.step(ground_truth_action)
        ground_truth_state = divide21env_simulator._decode_state(obs)
        
        # (1) action
        action_are_equivalent, action_similarity_score = self.compare_actions(self.action, ground_truth_action)
        # (2) state
        states_are_equivalent, states_similarity_score = self.compare_states(self.state, ground_truth_state)
        
        # set ground truth score
        self.ground_truth_action_score = action_similarity_score
        self.ground_truth_state_score = states_similarity_score
    
    def compare_to_ground_truth2(self):
        '''
        checks if the LLM given state is actually generated
        '''
        # get challenge state and action
        hour = get_utc_hour()
        date = str(get_utc_date())
        challenge_dir = os.path.join(CHALLENGES_DIR, date)
        challenge_name = str(hour) + '.json'
        challenge_file = os.path.join(challenge_dir, challenge_name)
        data = None
        if not os.path.exists(challenge_file):
            message = f"No challenge found!"
            self.logger.add_info(CHALLENGE, CRITICAL, message)
            # log
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
            return
        with open(challenge_file, 'r') as f:
            data = json.load(f)
        challenge_state = data["challenge"]["initial_state"]
        challenge_action = data["challenge"]["action"]
        
        # generate state from the action given in the challenge
        divide21env_simulator = Divide21EnvSimulator()
        options = {
            'obs': challenge_state
        }
        obs, info = divide21env_simulator.reset(options=options)
        obs, reward, terminated, truncated, info = divide21env_simulator.step(challenge_action)
        ground_truth_state = divide21env_simulator._decode_state(obs)
        
        # compare states
        states_are_equivalent, states_similarity_score = self.compare_states(self.state, ground_truth_state)
        
        self.ground_truth_state_score = states_similarity_score
        
        