from divide21x.inspection.inspector import Inspector


if __name__ == "__main__":
    # action
    action = {
        "division": 1,
        "digit": 7,
        "rindex": 0
    }
    
    # state
    state = {
        "static_number": 899,
        "dynamic_number": 17,
        "available_digits_per_rindex": {0: [9, 9], 1: []},
        "players": [],
        "player_turn": 0
    }
    
    # initialize inspector
    inspector = Inspector(action, state)
    inspector.inspect_all()
    inspector.logger.save_episode()
    