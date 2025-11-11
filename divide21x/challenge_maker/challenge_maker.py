import json
import os
import datetime, hashlib, random
from divide21x.simulator.divide21env_simulator import Divide21EnvSimulator
from divide21x.utils.logger import EpisodeLogger
from divide21x.utils.util import get_utc_date, get_utc_datetime, get_utc_hour


BASE_DIR='./divide21x/challenge_maker/logs'
CHALLENGES_DIR = './divide21x/challenges'
# categories
CHALLENGE = 'challenge'
# types
ACTION = 'action'
STATE = 'state'
CRITICAL = 'critical'
WARNING = 'warning'
NOTE = 'note'
ID = 'id'
HASH = 'hash'


class ChallengeMaker():
    def __init__(self):
        # Challenge examples
        self.digit_change_example_state_1 = {
            "s": 43,
            "d": 45,
            "a": {0: [0, 1, 2, 4, 6, 7, 8, 9], 1: [1, 2, 3, 5, 6, 7, 8, 9]},
            "p": [{"pi": 0, "ps": -6, "pt": 0}, {"pi": 1, "ps": 0, "pt": 1}],
            "t": 1
        }
        self.digit_change_example_action = {
            "dv": False,
            "dg": 2,
            "ri": 0
        }
        self.digit_change_example_state_2 = {
            "s": 43,
            "d": 42,
            "a": {0: [0, 1, 4, 6, 7, 8, 9], 1: [1, 2, 3, 5, 6, 7, 8, 9]},
            "p": [{"pi": 0, "ps": -6, "pt": 1}, {"pi": 1, "ps": 0, "pt": 0}],
            "t": 0
        }
        
        self.division_example_state_1 = {
            "s": 523,
            "d": 195,
            "a": {0: [0, 1, 2, 4, 6, 7, 8, 9], 1: [0, 1, 3, 4, 5, 6, 7], 2: [2, 3, 4, 6, 7, 8, 9]},
            "p": [{"pi": 0, "ps": -27, "pt": 0}, {"pi": 1, "ps": -11, "pt": 0}, {"pi": 2, "ps": -16, "pt": 0}, {"pi": 3, "ps": 3, "pt": 1}],
            "t": 3
        }
        self.division_example_action = {
            "dv": True,
            "dg": 3,
            "ri": None
        }
        self.division_example_state_2 = {
            "s": 523,
            "d": 65,
            "a": {0: [0, 1, 2, 4, 6, 7, 8, 9], 1: [1, 3, 4, 5, 7]},
            "p": [{"pi": 0, "ps": -27, "pt": 0}, {"pi": 1, "ps": -11, "pt": 0}, {"pi": 2, "ps": -16, "pt": 0}, {"pi": 3, "ps": 6, "pt": 1}],
            "t": 3
        }
        
        # challenge state and action
        self.state = None
        self.action = None
        
        # challenge id and hash
        self.challenge_id = None
        self.challenge_hash = None
        
        # Logging
        self.logger = EpisodeLogger(BASE_DIR)
    
    def get_state(self):
        return self.state
    
    def get_action(self):
        return self.action

    def make_challenge(self):
        """
        Returns the Divide21x challenge for a specific hour.
        time: datetime or None (defaults to current UTC hour)
        """
        # Use timezone-aware UTC datetime
        utc_datetime = get_utc_datetime()
        hour = get_utc_hour()
        hour_2 = hour + 2
        date = str(get_utc_date())
        date_hour_str = utc_datetime[:13]  # e.g. "2025-11-04T15"
        
        # place challenge in the challenges dir
        challenge_dir = os.path.join(CHALLENGES_DIR, date)
        os.makedirs(challenge_dir, exist_ok=True)
        challenge_name = str(hour) + '.json'
        challenge_file = os.path.join(challenge_dir, challenge_name)
        challenge_name_tmp = str(hour) + '.json.tmp'
        challenge_file_tmp = os.path.join(challenge_dir, challenge_name_tmp)
        
        # check if challenge already exists
        if os.path.isfile(challenge_file):
            message = "Challenge was already created."
            self.logger.add_info(CHALLENGE, WARNING, message)
            # log
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
                
            self.logger.save_episode()
            return

        # Create a deterministic seed based on the string
        seed = int(hashlib.sha256(date_hour_str.encode()).hexdigest(), 16) % (10**8)
        random.seed(seed)

        #   get player number between 2 and hour+2
        players = random.randint(2, hour_2)
        #   set state
        divide21env_simulator = Divide21EnvSimulator(digits=hour_2, players=players)
        obs, info = divide21env_simulator.reset(seed=seed)
        
        #   play for at least 100 actions - this prevents the initial state from always being given
        state_collection = []
        for i in range(100):
            # create action
            division = bool(random.randint(0, 1))
            action = {
                "division": division,
                "digit": int(random.randint(0, 9)) if not division else int(random.randint(2, 9)),
                "rindex": int(random.randint(0, hour_2-1)) if not division else None
            }
            # apply action
            obs, reward, done, trunc, info = divide21env_simulator.step(action)
            obs = divide21env_simulator._decode_state(obs)
            state_collection.append(obs)
            # update hour_2
            hour_2 = len(str(obs["dynamic_number"]))
            if done or trunc:
                # do not append the final state
                state_collection.pop()
                break
            
        selected = random.randint(0, len(state_collection)-1)
        self.state = state_collection[selected]
        #   set action
        division = bool(random.randint(0, 1))
        rindex = int(random.randint(0, hour_2-1)) if not division else None
        digit = None
        if division:
            digit = int(random.randint(2, 9))
        else:
            available_digits_list = self.state["available_digits_per_rindex"][rindex]
            digit = available_digits_list[random.randint(0, len(available_digits_list)-1)]
        
        self.action = {
            "division": division,
            "digit": digit,
            "rindex": rindex
        }
        
        # make sure the state and action follow the schema
        # (1) state
        self.state["s"] = self.state.pop("static_number")
        self.state["d"] = self.state.pop("dynamic_number")
        self.state["a"] = self.state.pop("available_digits_per_rindex")
        self.state["p"] = self.state.pop("players")
        #   go through each player dict and update key name
        for player in self.state["p"]:
            player["pi"] = player.pop("id")
            player["ps"] = player.pop("score")
            player["pt"] = player.pop("is_current_turn")
        self.state["t"] = self.state.pop("player_turn")
        # (2) action
        self.action["dv"] = self.action.pop("division")
        self.action["dg"] = self.action.pop("digit")
        self.action["ri"] = self.action.pop("rindex")
        
        # build the challenge dict
        # (1) examples
        challenge = {}
        challenge["example_1"] = {
            "initial_state": self.digit_change_example_state_1,
            "action": self.digit_change_example_action,
            "final_state": self.digit_change_example_state_2,
        }
        challenge["example_2"] = {
            "initial_state": self.division_example_state_1,
            "action": self.division_example_action,
            "final_state": self.division_example_state_2,
        }
        # (2) challenge
        challenge["challenge"] = {
            "initial_state": self.state,
            "action": self.action
        }
        # make the challenge file
        with open(challenge_file_tmp, 'w') as tmp_file:
            json.dump(challenge, tmp_file, indent=4)
        os.rename(challenge_file_tmp, challenge_file)
        
        # log a unique challenge ID and hash
        self.challenge_id = date_hour_str
        to_hash = self.challenge_id + str(challenge)
        self.challenge_hash = hashlib.sha256(to_hash.encode()).hexdigest()
        
        message = "Challenge created."
        self.logger.add_info(CHALLENGE, NOTE, message)
        self.logger.add_info(CHALLENGE, ID, self.challenge_id)
        self.logger.add_info(CHALLENGE, HASH, self.challenge_hash)
        self.logger.add_info(CHALLENGE, STATE, self.state)
        self.logger.add_info(CHALLENGE, ACTION, self.action)
        
        # log
        if self.logger.info not in self.logger.episode_log:
            self.logger.episode_log.append(self.logger.info)
            
        self.logger.save_episode()

if __name__ == "__main__":
    challenge_maker = ChallengeMaker()
    challenge_maker.make_challenge()