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

        #state needs to begin with a bunch of nothing
        self.state = np.array((self.lookback+1)*[0.0, 0.0])
        self.state[0] = 0
        #self.state[1] = self.value_map[0] #first value
        self.state[1] = self.value_map_func(0) #first value

        #####################
        #fill out first N=lookback spaces by taking single-steps
        for _ in range(self.lookback):
            self.step(0)

    def value_map_func(self,x):
        return sigmoid(x,c=self.c, w=self.w)

    def score_map_func(self,x):
        return d1_sigmoid(x,c=self.c,w=self.w)**self.power-self.sinkscore

    def report(self):
        print ("at site "+str((int(self.state[0]))) +" with value "+str(self.state[1]))
        print ("back\tvalue")
        for i in np.arange(2,2*self.lookback+1,2):
            print (str(int(self.state[i]))+'\t'+str(self.state[i+1]))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,action):
        assert self.action_space.contains(action)
        newstate = np.roll(self.state,2)
        newstate[2] = 0
        newstate[0] = 0
        
        for i in np.arange(2,int(self.lookback*2+1),2):
            newstate[i] = newstate[i]+(action+1)
            
        newstate[0] = int(self.state[0])+(action+1)
        
        
        if newstate[0] >= self.L:
            done = True
            newstate[0] = self.L-1
        else:
            done = False
            
        newstate[1] = self.value_map_func(newstate[0])
        self.state = newstate
        
        reward = self.score_map_func(self.state[0])
        
        return self.state, reward, done, {}

    def reset(self, cmin= 10, cmax = 490, wmin = 1, wmax = 10, power = .5):
        return self.random_reset(cmin=cmin,cmax=cmax,wmin=wmin,wmax=wmax, power=power)

    def random_reset(self,cmin = 10, cmax=490, wmin = 1, wmax = 10, power=.5):
        self.c = np.random.random()*(cmax-cmin)+cmin
        self.w = np.random.random()*(wmax-wmin)+wmin
        
        self.power = power
        
        #state needs to begin with a bunch of nothing
        self.state = np.array((self.lookback+1)*[0.0, 0.0])
        self.state[0] = 0
        self.state[1] = self.value_map_func(self.state[0]) #first value
        
        #fill out first N=lookback spaces by taking single-steps
        for _ in range(self.lookback):
            self.step(0)
        
        return self.state

class Change2(gym.Env):
    """Model Change environment
    This game presents moves along a linear chain of states, 
    where each action is the number of states to take (from 1 to 10),
    which moves along the chain.  The state will be returned as both the 
    current integer-index, and the value of a scaler.  This
    scaler will vary according to a sequence of sigmoid function, defined at the time of 
    enviornment instantiation or reset (define width, center position, and amplitude).
    
    The reward from each point will be the derivative of this sigmoid function,
    offset and rescaled by certain criteria passed to the env at creation.
    
    The goal is thus to take large jumps along the path when nothing is changing,
    and small steps when changing dramatically.  The game completes when a move 
    takes the state beyond end endpoint.

    The observed state is the current state in the chain (0 to n-1) and value of scaler.
    """    
    def __init__(self, L=500, wmin=1, wmax=10, cmin=100, cmax=400, amin=-1,amax=1,
                    sinkscore = 0.2, power=.50, lookback=4, num_flips = 3):
        
        self.num_flips = num_flips
        self.L = L

        self.cmin = cmin
        self.cmax = cmax
        self.wmin = wmin
        self.wmax = wmax
        self.amin = amin
        self.amax = amax

        self.clist = np.random.random(self.num_flips)*(self.cmax-self.cmin)+self.cmin
        self.wlist = np.random.random(self.num_flips)*(self.wmax-self.wmin)+self.wmin
        self.alist = np.random.random(self.num_flips)*(self.amax-self.amin)+self.amin
        
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
        
        
        #state needs to begin with a bunch of nothing
        self.state = np.array((self.lookback+1)*[0.0, 0.0])
        self.state[0] = 0
        self.state[1] = self.value_map_func(0) #first value

        #####################
        #fill out first N=lookback spaces by taking single-steps
        for _ in range(self.lookback):
            self.step(0)

    def value_map_func(self,x):
        my_func = 0*x
            
        for i in range(self.num_flips):
            my_func += sigmoid(x,self.clist[i],self.wlist[i])*self.alist[i]
        return my_func
            
    def score_map_func(self,x):
        my_score = 0*x - self.sinkscore
        for i in range(self.num_flips):
            my_score += d1_sigmoid(x,self.clist[i],self.wlist[i])**self.power
        return my_score
    

    def report(self):
        print ("at site "+str((int(self.state[0]))) +" with value "+str(self.state[1]))
        print ("back\tvalue")
        for i in np.arange(2,2*self.lookback+1,2):
            print (str(int(self.state[i]))+'\t'+str(self.state[i+1]))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,action):
        assert self.action_space.contains(action)
        newstate = np.roll(self.state,2)
        newstate[2] = 0
        newstate[0] = 0
        
        for i in np.arange(2,int(self.lookback*2+1),2):
            newstate[i] = newstate[i]+(action+1)
            
        newstate[0] = int(self.state[0])+(action+1)
        
        
        if newstate[0] >= self.L:
            done = True
            newstate[0] = self.L-1
        else:
            done = False
            
        newstate[1] = self.value_map_func(newstate[0])
        self.state = newstate
        
        reward = self.score_map_func(self.state[0])
        
        return self.state, reward, done, {}
    
    def reset(self):
        
        self.clist = np.random.random(self.num_flips)*(self.cmax-self.cmin)+self.cmin
        self.wlist = np.random.random(self.num_flips)*(self.wmax-self.wmin)+self.wmin
        self.alist = np.random.random(self.num_flips)*(self.amax-self.amin)+self.amin                
        
        #state needs to begin with a bunch of nothing
        self.state = np.array((self.lookback+1)*[0.0, 0.0])
        self.state[0] = 0
        self.state[1] = self.value_map_func(self.state[0]) #first value
        
        #fill out first N=lookback spaces by taking single-steps
        for _ in range(self.lookback):
            self.step(0)
        
        return self.state