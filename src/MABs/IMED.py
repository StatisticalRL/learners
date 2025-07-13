from src.MABs.utils import *
from src.MABs import Agent

'''
Bernoulli distributions
'''
class IMED(Agent):
    """Indexed Minimum Empirical Divergence"""
    def __init__(self,nbArms,kullback):
        self.nbArms = nbArms
        self.kl = kullback
        Agent.__init__(self,self.nbArms,name="IMED")

    def reset(self,initstate=0):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.means = np.zeros(self.nbArms)
        self.maxMeans = 0
        self.indexes = np.zeros(self.nbArms)

    def play(self,state=0):
        return randmin(self.indexes)

    def update(self, state, arm, reward, observation):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.means[arm] = self.cumRewards[arm] /self.nbDraws[arm]
        self.maxMeans = max(self.means)

        self.indexes = [self.nbDraws[a]*self.kl(self.means[a],self.maxMeans)+log(self.nbDraws[a]) if self.nbDraws[a] > 0 else 0 for a in range(self.nbArms)]
