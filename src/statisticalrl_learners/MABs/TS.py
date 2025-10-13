
from statisticalrl_learners.MABs import BanditAgent
from statisticalrl_learners.MABs.utils import *

'''
Bernoulli distributions
'''
class TS(BanditAgent):
    """Thomson Sampling"""
    def __init__(self,nbArms):
        self.nbArms = nbArms
        BanditAgent.__init__(self, self.nbArms, name="TS")

    def reset(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.theta = np.zeros(self.nbArms)

    def play(self):
        return randmax(self.theta)

    def update(self, arm, reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1

        self.theta = [np.random.beta(max(self.cumRewards[a],0) + 1, max(self.nbDraws[a] - self.cumRewards[a],0) + 1) for a in range(self.nbArms)]


