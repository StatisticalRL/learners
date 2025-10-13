
from statisticalrl_learners.MABs import BanditAgent
import numpy as np

class UE(BanditAgent):
    """Uniform Exploration"""
    def __init__(self,nA):
        #self.env=env
        #self.nA = env.action_space.n
        BanditAgent.__init__(self, nA, name="UE")

    def reset(self):
        self.nbDraws = np.zeros(self.nA)
        self.cumRewards = np.zeros(self.nA)

    def play(self):
        return np.random.randint(self.nA)#self.env.action_space.sample()

    def update(self, action, reward):
        self.cumRewards[action] = self.cumRewards[action] + reward
        self.nbDraws[action] = self.nbDraws[action] + 1