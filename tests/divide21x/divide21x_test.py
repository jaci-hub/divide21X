from divide21x.envs.divide21x import Divide21X


if __name__ == "__main__":
    # LLM input action
    action = {
        "division": 0,
        "digit": 7,
        "rindex": 0
    }
    
    # LLM input state
    state = {
        "static_number": 19,
        "dynamic_number": 59,
        "available_digits_per_rindex": {0: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [2, 3, 4, 6, 7, 8, 9]},
        "players": [{'id': 0, 'score': -13, 'is_current_turn': 1}],
        "player_turn": 0
    }
    
    env = Divide21X(state=state)
    env.start()
        
    print(env.get_result())
