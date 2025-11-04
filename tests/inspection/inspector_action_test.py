from divide21x.inspection.inspector_action import InsperctorAction

env = InsperctorAction()


if __name__ == "__main__":
    # user/model input action
    action = {
        "division": 0,
        "digit": 7,
        "rindex": 0
    }
    
    obs, info = env.reset()
    # execute action
    obs, reward, terminated, truncated, info = env.step(action)
    