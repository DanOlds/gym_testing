import gym
#from gym import wrappers # for output display
import numpy as np
from learn_change2 import DDQNAction
import matplotlib.pyplot as plt
from gym_change.envs.change1_env import Change1



if __name__ == '__main__':
    #env = gym.make("CartPole-v0")
    env = Change1(c=100, L=300,lookback=10,power=.25)

    ddqn_agent = DDQNAction(alpha = 0.0005, gamma = 0.99, n_actions = 10, epsilon = 1.0,
                    batch_size = 64, input_dims=22)

    n_games = 1000
    ddqn_agent.load_model() # if load a saved model
    ddqn_scores = []
    eps_history = []
    best_avg_score = -9999999.
    best_single_score = -9999999.
    
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset(cmin=50,cmax=250,wmin=4,wmax=6,power=.25)
        while not done:
            action = ddqn_agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            ddqn_agent.remember(observation, action, reward, observation_, done)
            observation = observation_
            ddqn_agent.learn()

        eps_history.append(ddqn_agent.epsilon)
        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[max(0, i-100):(i+1)])
        if avg_score > best_avg_score and i > 100:
            print ("new avg highscore!")
            ddqn_agent.save_model('best_avgscore_dqn.h5')
            best_avg_score = avg_score

        if score > best_single_score and i > 10:
            print ("new single highscore!")
            ddqn_agent.save_model('best_singlescore_dqn.h5')
            best_single_score = score


        print ('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score)

        if i % 10 == 0 and i > 0:
            ddqn_agent.save_model()

    
    #x = [i+1 for i in range(n_games)]
    #plt.figure()
    #plt.plot(x, ddqn_scores)
    #plt.plot(x, eps_history)
    #plt.show()
