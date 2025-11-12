import random
from divide21x.simulator.divide21env_simulator import Divide21EnvSimulator


if __name__ == "__main__":
    # given state
    state = {
        "s": 19,
        "d": 59,
        "a": {0: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [2, 3, 4, 6, 7, 8, 9]},
        "p": [{'i': 0, 'c': -13, 'm': 1}],
        "t": 0
    }
    options = {
        'obs': state
    }
    #   set state
    divide21env_simulator = Divide21EnvSimulator()
    obs, info = divide21env_simulator.reset(options=options)
    
    # create action
    division = bool(random.randint(0, 1))
    action = {
        "v": division,
        "g": int(random.randint(0, 9)) if not division else int(random.randint(2, 9)),
        "r": int(random.randint(0, 1)) if not division else None
    }
    # apply action
    obs, reward, done, trunc, info = divide21env_simulator.step(action)
    obs = divide21env_simulator._decode_state(obs)

    