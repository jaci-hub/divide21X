from divide21x.envs.divide21x_main import Divide21X


if __name__ == "__main__":
    # LLM input action
    action = {
        "v": 0,
        "g": 7,
        "r": 0
    }
    
    # LLM input state
    state = {
        "s": 19,
        "d": 59,
        "a": {0: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [2, 3, 4, 6, 7, 8, 9]},
        "p": [{'i': 0, 'c': -13, 'm': 1}],
        "t": 0
    }
    
    env = Divide21X(state=state)
    env.start()
        
    print(env.get_result())
