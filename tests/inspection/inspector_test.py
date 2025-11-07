from divide21x.inspection.inspector import Inspector


if __name__ == "__main__":
    # action
    action = {
        "division": True,
        "digit": 4,
        "rindex": None
    }
    
    # state
    state = {
        "static_number": 19,
        "dynamic_number": 59,
        "available_digits_per_rindex": {0: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [2, 3, 4, 6, 7, 8, 9]},
        "players": [{'id': 0, 'score': -13, 'is_current_turn': 1}],
        "player_turn": 0
    }
    
    # initialize inspector
    inspector = Inspector(action=action, state=state)
    inspector.inspect_all()
    