from divide21x.envs.divide21x_action_only import Divide21XActionOnly

env = Divide21XActionOnly()
obs, info = env.reset()


if __name__ == "__main__":
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    env.render()
