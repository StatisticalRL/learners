from src.MABs.utils import *

from src.MABs import Agent

class FTL(Agent):
    """Follow The Leader (a.k.a. greedy strategy)"""
    def __init__(self,nbArms):
        self.nbArms = nbArms
        Agent.__init__(self,self.nbArms,name="FTL")

    def reset(self, initstate=0):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)

    def play(self,state=0):
        if (min(self.nbDraws)==0):
            return randmax(-self.nbDraws)
        else:
            return randmax(self.cumRewards/self.nbDraws)


    def update(self, state, arm, reward, observation):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1
