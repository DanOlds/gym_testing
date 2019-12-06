from gym.envs.registration import register

register(
    id='change1-v0',
    entry_point='gym_change.envs:Change1',
)