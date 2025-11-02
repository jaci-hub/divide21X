from gymnasium.envs.registration import register

register(
    id="Divide21X-v0",
    entry_point="divide21x.envs.divide21x_action_only:Divide21XActionOnly",
)