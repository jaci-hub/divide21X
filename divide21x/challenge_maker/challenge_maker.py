import datetime, hashlib, random
from divide21x.simulator.divide21env_simulator import Divide21EnvSimulator
from divide21x.utils.logger import EpisodeLogger


BASE_DIR='./divide21x/challenge_maker/logs'
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
        self.digit_change_example_state_1 = None
        self.digit_change_example_action = None
        self.digit_change_example_state_2 = None
        
        self.division_example_state_1 = None
        self.division_example_action = None
        self.division_example_state_2 = None
        
        self.state = None
        self.action = None
        
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
        utc_datetime = datetime.datetime.now(datetime.timezone.utc)
        utc_datetime = utc_datetime.isoformat()
        
        # get string representing the year-month-day-hour for deterministic seeding
        date_hour_str = utc_datetime[:13]  # e.g. "2025-11-04T15"

        # Create a deterministic seed based on the string
        challenge_hash = hashlib.sha256(date_hour_str.encode()).hexdigest()
        seed = int(hashlib.sha256(date_hour_str.encode()).hexdigest(), 16) % (10**8)
        random.seed(seed)

        # Generate the deterministic challenge
        #   get hour to help model digits
        hour = int(date_hour_str[date_hour_str.index('T') + 1:])
        hour_2 = hour + 2 # to avoid digits < 2
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
        
        # Attach a unique challenge ID and hash
        self.challenge_id = date_hour_str
        self.challenge_hash = challenge_hash

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

