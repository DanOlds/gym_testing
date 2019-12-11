import datetime
import random

import gym
import keras
import numpy as np
import tensorflow as tf

from gym_change.envs.change1_env import Change1


STORE_PATH = './log/change'
MAX_EPSILON = 1
MIN_EPSILON = 0.01
LAMBDA = 0.0005
GAMMA = 0.95
BATCH_SIZE = 32
TAU = 0.08
RANDOM_REWARD_STD = 1.0

#env = gym.make("change1-v0", c=10, L=500)
env = Change1(c=10, L=500)
state_size = env.observation_space.shape[0]  # usually 10
num_actions = env.action_space.n  # also 10

_primary_network = keras.Sequential([
    keras.layers.Dense(env.observation_space.shape[0], activation='relu', kernel_initializer=keras.initializers.he_normal()),
    keras.layers.Dense(20, activation='relu', kernel_initializer=keras.initializers.he_normal()),
    keras.layers.Dense(10, activation='relu', kernel_initializer=keras.initializers.he_normal()),
    keras.layers.Dense(num_actions)
])
_target_network = keras.Sequential([
    keras.layers.Dense(env.observation_space.shape[0], activation='relu', kernel_initializer=keras.initializers.he_normal()),
    keras.layers.Dense(20, activation='relu', kernel_initializer=keras.initializers.he_normal()),
    keras.layers.Dense(10, activation='relu', kernel_initializer=keras.initializers.he_normal()),
    keras.layers.Dense(num_actions)
])
_primary_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')


class Memory:
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []

    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)

    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)

    @property
    def num_samples(self):
        return len(self._samples)


_memory = Memory(5000)


def choose_action(state, primary_network, eps):
    if random.random() < eps:
        action = random.randint(0, num_actions - 1)
    else:
        action = np.argmax(primary_network.predict(state.reshape(1, -1)))
    return action


def train(primary_network, memory, target_network=None):
    if memory.num_samples < BATCH_SIZE * 3:
        return 0
    batch = memory.sample(BATCH_SIZE)
    states = np.array([val[0] for val in batch])
    actions = np.array([val[1] for val in batch])
    rewards = np.array([val[2] for val in batch])
    next_states = np.array([(np.zeros(state_size)
                             if val[3] is None else val[3]) for val in batch])
    # predict Q(s,a) given the batch of states
    prim_qt = primary_network.predict(states)
    # predict Q(s',a') from the evaluation network
    prim_qtp1 = primary_network.predict(next_states)
    # copy the prim_qt into the target_q tensor - we then will update one index corresponding to the max action
    target_q = prim_qt
    updates = rewards
    valid_idxs = np.array(next_states).sum(axis=1) != 0
    batch_idxs = np.arange(BATCH_SIZE)
    if target_network is None:
        updates[valid_idxs] += GAMMA * np.amax(prim_qtp1[valid_idxs, :], axis=1)
    else:
        prim_action_tp1 = np.argmax(prim_qtp1, axis=1)
        q_from_target = target_network.predict(next_states)
        updates[valid_idxs] += GAMMA * q_from_target[batch_idxs[valid_idxs], prim_action_tp1[valid_idxs]]
    target_q[batch_idxs, actions] = updates
    loss = primary_network.train_on_batch(states, target_q)
    if target_network is not None:
        # update target network parameters slowly from primary network
        for t, e in zip(target_network.trainable_weights, primary_network.trainable_weights):
            t.assign(t * (1 - TAU) + e * TAU)
    return loss


num_episodes = 500
_eps = MAX_EPSILON
render = False
train_writer = tf.summary.create_file_writer(STORE_PATH + f"/DoubleQ_{datetime.datetime.now().strftime('%d%m%Y%H%M')}")
double_q = True
steps = 0
for i in range(num_episodes):
    _state = env.reset()
    #print(f"after env.reset(): state: {_state}")
    cnt = 0
    avg_loss = 0
    while True:
        if render:
            env.render()
        _action = choose_action(_state, _primary_network, _eps)
        next_state, reward, done, info = env.step(_action)
        if done:
            next_state = None
        # store in memory
        _memory.add_sample((_state, _action, reward, next_state))
        loss = train(_primary_network, _memory, _target_network if double_q else None)
        avg_loss += loss
        _state = next_state
        # exponentially decay the eps value
        steps += 1
        _eps = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-LAMBDA * steps)
        if done:
            avg_loss /= cnt
            print(f"Episode: {i}, Reward: {cnt}, avg loss: {avg_loss:.3f}, eps: {_eps:.3f}")
            with train_writer.as_default():
                tf.summary.scalar('reward', cnt, step=i)
                tf.summary.scalar('avg loss', avg_loss, step=i)
            break
        cnt += 1

# save the trained network
_primary_network.save('trained_network.h5')
