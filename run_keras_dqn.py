import gym
#from gym import wrappers # for output display
import numpy as np
from learn_change2 import DDQNAction
import matplotlib.pyplot as plt
from gym_change.envs.change1_env import Change1, Change2
import tensorflow as tf
import datetime



if __name__ == '__main__':
    #env = gym.make("CartPole-v0")
    #env = Change2(lookback=10,power=.25,num_flips=3)
    env = Change1(lookback=10,power=.50)

    ddqn_agent = DDQNAction(alpha = 0.0005, gamma = 0.99, n_actions = 10, epsilon = 1.0,
                    batch_size = 128, input_dims=22)

    n_games = 2000
    #ddqn_agent.load_model() # if load a saved model
    ddqn_scores = []
    eps_history = []
    reward_history = []
    action_history = []

    best_avg_score = -10.0
    best_single_score = -10.0
    
    STORE_PATH = './log/change'
    train_writer = tf.summary.create_file_writer(STORE_PATH + \
             f"/DoubleQ_{datetime.datetime.now().strftime('%d%m%Y%H%M')}")

    for i in range(n_games):
        done = False
        score = 0
        action_counter = []
        reward_counter = []
        observation = env.reset()
        while not done:
            action = ddqn_agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            action_counter.append(action)
            reward_counter.append(reward)
            ddqn_agent.remember(observation, action, reward, observation_, done)
            observation = observation_
            ddqn_agent.learn()

        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(score)
        

        #reward_history.append(reward_counter)
        #action_history.append(action_counter)

        avg_score = np.mean(ddqn_scores[max(0, i-100):(i+1)])
        if avg_score > best_avg_score and i > 100:
            print ("new avg highscore!")
            ddqn_agent.save_model('best_avgscore_dqn.h5')
            best_avg_score = avg_score

        if score > best_single_score and i > 10:
            print ("new single highscore!")
            ddqn_agent.save_model('best_singlescore_dqn.h5')
            best_single_score = score

        #print to screen stats
        print ('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)

        #tensorboard output
        with train_writer.as_default():
            tf.summary.scalar('score', score, step=i)
            tf.summary.scalar('avg_score', avg_score, step=i)
            tf.summary.histogram('action distributions', action_counter, step=i)
            tf.summary.histogram('reward distributions', reward_counter, step=i)
        #periodically save agent
        if i % 10 == 0 and i > 0:
            ddqn_agent.save_model()

    
    #x = [i+1 for i in range(n_games)]
    #plt.figure()
    #plt.plot(x, ddqn_scores)
    #plt.plot(x, eps_history)
    #plt.show()
