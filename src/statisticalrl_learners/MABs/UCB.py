
from statisticalrl_learners.MABs import BanditAgent
from statisticalrl_learners.MABs.utils import *

class UCB(BanditAgent):
    """Upper Confidence Bound"""
    def __init__(self,nbArms,delta):
        self.nbArms = nbArms
        self.delta = delta
        BanditAgent.__init__(self, self.nbArms, name="UCB")

    def reset(self):
        self.time = 0
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)
        self.indexes = np.zeros(self.nbArms)

    def play(self):
        return randmax(self.indexes)

    def update(self, arm, reward):
        self.time = self.time + 1
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] /self.nbDraws[arm]


        self.indexes = [self.means[a] + sqrt(log(1/self.delta(self.time))/(2*self.nbDraws[a])) if self.nbDraws[a] > 0 else np.Inf for a in range(self.nbArms)]
