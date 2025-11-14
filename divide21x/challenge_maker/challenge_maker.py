import json
import os
import datetime, hashlib, random
from divide21x.simulator.divide21env_simulator import Divide21EnvSimulator
from divide21x.utils.logger import EpisodeLogger
from divide21x.utils.util import get_utc_date, get_utc_datetime, get_utc_day, get_utc_hour


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
            "p": [{"i": 0, "c": -6, "m": 0}, {"i": 1, "c": 0, "m": 1}],
            "t": 1
        }
        self.digit_change_example_action = {
            "v": False,
            "g": 2,
            "r": 0
        }
        self.digit_change_example_state_2 = {
            "s": 43,
            "d": 42,
            "a": {0: [0, 1, 4, 6, 7, 8, 9], 1: [1, 2, 3, 5, 6, 7, 8, 9]},
            "p": [{"i": 0, "c": -6, "m": 1}, {"i": 1, "c": 0, "m": 0}],
            "t": 0
        }
        
        self.division_example_state_1 = {
            "s": 523,
            "d": 195,
            "a": {0: [0, 1, 2, 4, 6, 7, 8, 9], 1: [0, 1, 3, 4, 5, 6, 7], 2: [2, 3, 4, 6, 7, 8, 9]},
            "p": [{"i": 0, "c": -27, "m": 0}, {"i": 1, "c": -11, "m": 0}, {"i": 2, "c": -16, "m": 0}, {"i": 3, "c": 3, "m": 1}],
            "t": 3
        }
        self.division_example_action = {
            "v": True,
            "g": 3,
            "r": None
        }
        self.division_example_state_2 = {
            "s": 523,
            "d": 65,
            "a": {0: [0, 1, 2, 4, 6, 7, 8, 9], 1: [1, 3, 4, 5, 7]},
            "p": [{"i": 0, "c": -27, "m": 0}, {"i": 1, "c": -11, "m": 0}, {"i": 2, "c": -16, "m": 0}, {"i": 3, "c": 6, "m": 1}],
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
        Returns the Divide21x challenge of the day.
        """
        # Use timezone-aware UTC datetime
        date = str(get_utc_date())
        year_month = date[:7]
        day = get_utc_day()
        day_after = int(day) + 1 # i use day after becuase when i set it to digits it keeps it from being 1, for the first day of the month!
        
        # place challenge in the challenges dir
        challenge_path = os.path.join(CHALLENGES_DIR, year_month)
        os.makedirs(challenge_path, exist_ok=True)
        challenge_name = str(day) + '.json'
        challenge_file = os.path.join(challenge_path, challenge_name)
        challenge_name_tmp = challenge_name + '.tmp'
        challenge_file_tmp = os.path.join(challenge_path, challenge_name_tmp)
        
        # check if challenge already exists
        if os.path.isfile(challenge_file):
            message = "Challenge for today has already been created."
            self.logger.add_info(CHALLENGE, WARNING, message)
            # log
            if self.logger.info not in self.logger.episode_log:
                self.logger.episode_log.append(self.logger.info)
                
            self.logger.save_episode()
            return

        # Create a deterministic seed based on the string
        seed = int(hashlib.sha256(date.encode()).hexdigest(), 16) % (10**8)
        random.seed(seed)

        #   get player number between 2 and day_after
        players = random.randint(2, day_after)
        #   set state
        divide21env_simulator = Divide21EnvSimulator(digits=day_after, players=players)
        obs, info = divide21env_simulator.reset(seed=seed)
        
        #   play for at least 100 actions - this prevents the initial state from always being given
        state_collection = []
        for i in range(100):
            # create action
            division = bool(random.randint(0, 1))
            action = {
                "v": division,
                "g": int(random.randint(0, 9)) if not division else int(random.randint(2, 9)),
                "r": int(random.randint(0, day_after-1)) if not division else None
            }
            # apply action
            obs, reward, done, trunc, info = divide21env_simulator.step(action)
            obs = divide21env_simulator._decode_state(obs)
            state_collection.append(obs)
            # update day_after
            day_after = len(str(obs["d"]))
            if done or trunc:
                # do not append the final state
                state_collection.pop()
                break
            
        selected = random.randint(0, len(state_collection)-1)
        self.state = state_collection[selected]
        #   set action
        division = bool(random.randint(0, 1))
        rindex = int(random.randint(0, day_after-1)) if not division else None
        digit = None
        if division:
            digit = int(random.randint(2, 9))
        else:
            available_digits_list = self.state["a"][rindex]
            digit = available_digits_list[random.randint(0, len(available_digits_list)-1)]
        
        self.action = {
            "v": division,
            "g": digit,
            "r": rindex
        }
        
        # build the challenge dict
        # (1) examples
        challenge = {}
        challenge["example_1"] = {
            "z": self.digit_change_example_state_1,
            "a": self.digit_change_example_action,
            "o": self.digit_change_example_state_2,
        }
        challenge["example_2"] = {
            "z": self.division_example_state_1,
            "a": self.division_example_action,
            "o": self.division_example_state_2,
        }
        # (2) challenge
        challenge["challenge"] = {
            "z": self.state,
            "a": self.action
        }
        # make the challenge file
        with open(challenge_file_tmp, 'w') as tmp_file:
            json.dump(challenge, tmp_file, indent=4)
        os.rename(challenge_file_tmp, challenge_file)
        
        # log a unique challenge ID and hash
        self.challenge_id = date
        to_hash = self.challenge_id + str(challenge)
        self.challenge_hash = hashlib.sha256(to_hash.encode()).hexdigest()
        
        message = f"Challenge for today [{date}] has been created."
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