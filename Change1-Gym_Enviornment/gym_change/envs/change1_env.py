import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x,c=0,w=1):
    return 1.0/(1.0+np.exp(-(x-c)/w))

def d1_sigmoid(x,c=0,w=1):
    return (1.0/w)*sigmoid(x,c,w)*(1.0-sigmoid(x,c,w))

class Change1(gym.Env):
    """Model Change environment
    This game presents moves along a linear chain of states, 
    where each action is the number of states to take (from 1 to 10),
    which moves along the chain.  The state will be returned as both the 
    current integer-index, and the value of a scaler from 0 to 1.  This
    scaler will vary according to a sigmoid function, defined at the time of 
    enviornment instantiation (define width and center position).
    
    The reward from each point will be the derivative of this sigmoid function,
    offset and rescaled by certain criteria passed to the env at creation.
    
    The goal is thus to take large jumps along the path when nothing is changing,
    and small steps when changing dramatically.  The game completes when a move 
    takes the state beyond end endpoint.

    The observed state is the current state in the chain (0 to n-1) and value of scaler.
    """    
    def __init__(self, L=500, c=300, w = 10, sinkscore = 0.2, power=.50, lookback=4):
        self.L = L
        self.c = c  # center of sigmoid
        self.w = w  # width of sigmoid
        self.power = power
        self.sinkscore = sinkscore #fraction to drop score floor
        self.lookback = lookback #how many previous entries to remember
        
        self.action_space = spaces.Discrete(10) #assume we can take steps 1 to 10
        
        #setup observation_space
        high = np.array((self.lookback+1)*[
            int(self.L),
            np.finfo(np.float32).max])
        low = np.array((self.lookback+1)*[
            int(0),
            -1.0*np.finfo(np.float32).max])
            
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        ##############

        self.seed()
        
        #setup value-map
        self.x = np.arange(0,self.L,dtype=int)
        self.value_map = sigmoid(self.x,c=self.c,w=self.w)
        #setup score-map
        self.score_map =d1_sigmoid(self.x,c=self.c,w=self.w)**self.power-self.sinkscore*max(d1_sigmoid(self.x,c=self.c,w=self.w)**self.power) 

        #state needs to begin with a bunch of nothing
        self.state = np.array((self.lookback+1)*[0.0, 0.0])
        self.state[0] = 0
        self.state[1] = self.value_map[0] #first value
        #####################
        #fill out first N=lookback spaces by taking single-steps
        for i in range(self.lookback):
            self.step(0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        #take number of steps based on action-1
        #self.state += action+1 #action = 0 to 9
        #new state update
        newstate = np.roll(self.state,2)
        newstate[0] = int(self.state[0]) + (action+1)
        #newstate[1] = self.value_map[newstate[0]]
        
        
        if newstate[0] >= self.L:
            done = True
            #reward = 0
            newstate[0] = self.L-1
            
        else:
            #reward = self.score_map[self.state] 
            done = False
        
        newstate[1] = self.value_map[int(newstate[0])]
        self.state = newstate
        
        reward = self.score_map[int(self.state[0])]
        
        return self.state, reward, done, {}

    def reset(self):
        return self.random_reset()
        # # jl self.state = 0
        # self.state = np.array(5 * [0.0, 0.0])
        # self.x = np.arange(0,self.L,dtype=int)
        # self.score_map =d1_sigmoid(self.x,c=self.c,w=self.w)-self.sinkscore*max(d1_sigmoid(self.x,c=self.c,w=self.w))
        # self.value_map = sigmoid(self.x,c=self.c,w=self.w)
        # return self.state
        
    def random_reset(self,cmin = 10, cmax=490, wmin = 1, wmax = 10):
        self.c = np.random.random()*(cmax-cmin)+cmin
        self.w = np.random.random()*(wmax-wmin)+wmin
        
        self.x = np.arange(0,self.L,dtype=int)
        self.score_map =d1_sigmoid(self.x,c=self.c,w=self.w)**self.power-self.sinkscore*max(d1_sigmoid(self.x,c=self.c,w=self.w)**self.power) 
        self.value_map = sigmoid(self.x,c=self.c,w=self.w)
        
        #state needs to begin with a bunch of nothing
        self.state = np.array(5*[0.0, 0.0])
        self.state[0] = 0
        self.state[1] = self.value_map[0] #first value
        return self.state
