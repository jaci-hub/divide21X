from divide21x.envs.divide21x import Divide21X

env = Divide21X()


if __name__ == "__main__":
    # LLM input action
    action = {
        "division": 0,
        "digit": 7,
        "rindex": 0
    }
    
    # LLM input state
    
