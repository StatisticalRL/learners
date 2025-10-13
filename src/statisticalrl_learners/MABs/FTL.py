
from statisticalrl_learners.MABs import BanditAgent
from statisticalrl_learners.MABs.utils import *

class FTL(BanditAgent):
    """Follow The Leader (a.k.a. greedy strategy)"""
    def __init__(self,nbArms):
        self.nbArms = nbArms
        BanditAgent.__init__(self, self.nbArms, name="FTL")

    def reset(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)

    def play(self):
        if (min(self.nbDraws)==0):
            return randmax(-self.nbDraws)
        else:
            return randmax(self.cumRewards/self.nbDraws)


    def update(self,  arm, reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1
