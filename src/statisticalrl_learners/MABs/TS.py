
from statisticalrl_learners.MABs import BanditAgent
from statisticalrl_learners.MABs import BatchBanditAgent
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



class TS(BatchBanditAgent):
    """Thomson Sampling"""
    def __init__(self,nbArms):
        self.nbArms = nbArms
        BatchBanditAgent.__init__(self, self.nbArms, name="Batch-TS")

    def reset(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.theta = np.zeros(self.nbArms)

    def play(self):
        self.theta = [np.random.beta(max(self.cumRewards[a],0) + 1, max(self.nbDraws[a] - self.cumRewards[a],0) + 1) for a in range(self.nbArms)]
        return randmax(self.theta)

    def update(self, arm, reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1


    def batchplay(self,batchsize):
        return [self.play() for b in range(batchsize)]

    def batchupdate(self,batcharm,batchreward):
        for arm,reward in zip(batcharm,batchreward):
            self.update(arm,reward)

