import gym
import numpy as np


def test_make_env():
    env = gym.make('change1-v0', c=10, L=500)
    assert env is not None


def test_step():
    env = gym.make('change1-v0', c=10, L=500)
    env.reset()
    assert all(env.state.shape == np.array([10, ]))
    ##assert env.step(1) == np.array([1, 2, 3])


def test_action_space():
    env = gym.make('change1-v0', c=10, L=500)
    assert env.action_space.n == 10


def test_observation_space():
    env = gym.make('change1-v0', c=10, L=500)
    assert env.observation_space.shape == (10, )
