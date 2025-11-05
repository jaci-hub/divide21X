from divide21x.evaluation.evaluator import Evaluator


if __name__ == "__main__":
    # given state
    state = {
        "static_number": 19,
        "dynamic_number": 59,
        "available_digits_per_rindex": {0: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [2, 3, 4, 6, 7, 8, 9]},
        "players": [{'id': 0, 'score': -13, 'is_current_turn': 1}],
        "player_turn": 0
    }
    
    state1 = {
        "static_number": 10,
        "dynamic_number": 59,
        "available_digits_per_rindex": {0: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [2, 3, 4, 6, 7, 8, 9]},
        "players": [{'id': 0, 'score': -13, 'is_current_turn': 1}],
        "player_turn": 1
    }
    
    state2 = {
        "static_number": 11,
        "dynamic_number": 59,
        "available_digits_per_rindex": {0: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [2, 3, 4, 6, 7, 8, 9]},
        "players": [{'id': 0, 'score': -13, 'is_current_turn': 1}],
        "player_turn": 1
    }
    
    evaluator = Evaluator(state=state)
    evaluator.compare_states(state1, state2)
    # evaluator.action_generates_state()
    
    evaluator.logger.save_episode()