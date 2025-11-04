from divide21x.envs.divide21x_action_only import Divide21XActionOnly

env = Divide21XActionOnly()


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
