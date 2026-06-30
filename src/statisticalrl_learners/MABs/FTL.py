
from statisticalrl_learners.MABs import BanditAgent
from statisticalrl_learners.Generic.utils import *

class FTL(BanditAgent):
    """Follow The Leader (a.k.a. greedy strategy)"""
    def __init__(self,nbArms):
        BanditAgent.__init__(self,nbArms, name="FTL")

    def reset(self):
        self.nbDraws = np.zeros(self.nA)
        self.cumRewards = np.zeros(self.nA)

    def play(self):
        if (min(self.nbDraws)==0):
            return randmax(-self.nbDraws)
        else:
            return randmax(self.cumRewards/self.nbDraws)


    def update(self,  arm, reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1
