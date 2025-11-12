from divide21x.evaluation.evaluator import Evaluator



def test_action():
    # given action
    action = {
        "v": 1, # it can be True
        "g": 2,
        "r": None
    }
    
    action1 = {
        "v": 1, # it can be True
        "g": 2,
        "r": 2
    }
    
    action2 = {
        "v": 1, # it can be True
        "g": 2,
        "r": None
    }
    
    evaluator = Evaluator(action=action)
    evaluator.compare_actions(action1, action2)
    # evaluator.action_generates_state()
    

def test_state():
    # given state
    state = {
        "s": 19,
        "d": 59,
        "a": {0: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [2, 3, 4, 6, 7, 8, 9]},
        "p": [{'i': 0, 'c': -13, 'm': 1}],
        "t": 0
    }
    
    state1 = {
        "s": 10,
        "d": 59,
        "a": {0: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [2, 3, 4, 6, 7, 8, 9]},
        "p": [{'i': 0, 'c': -13, 'm': 1}],
        "t": 1
    }
    
    state2 = {
        "s": 11,
        "d": 59,
        "a": {0: [0, 1, 2, 3, 4, 5, 6, 7, 8], 1: [2, 3, 4, 6, 7, 8, 9]},
        "p": [{'i': 0, 'c': -13, 'm': 1}],
        "t": 1
    }
    
    evaluator = Evaluator(state=state)
    evaluator.compare_states(state1, state2)
    # evaluator.action_generates_state()
        

if __name__ == "__main__":    
    test_action()
    
    # test_state()
    