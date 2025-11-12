from divide21x.inspection.inspector import Inspector


if __name__ == "__main__":
    # action
    action = {
        "v": True,
        "g": 4,
        "r": None
    }
    
    # state
    state = {
        "s": 19,
        "d": 59,
        "a": {0: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [2, 3, 4, 6, 7, 8, 9]},
        "p": [{'i': 0, 'c': -13, 'm': 1}],
        "t": 0
    }
    
    # initialize inspector
    inspector = Inspector(action=action, state=state)
    inspector.inspect_all()
    