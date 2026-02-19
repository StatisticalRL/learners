
from statisticalrl_learners.MABs import BanditAgent
from statisticalrl_learners.Generic.utils import *

class NPTS(BanditAgent):
    """Non-parametric Thompson Sampling (aka Bounded Dirichlet Sampling)"""
    def __init__(self,nbArms,bound=1.):
        self.nbArms = nbArms
        BanditAgent.__init__(self, self.nbArms, name="BDS")
        self.bound = bound

    def reset(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.meanRewards = np.zeros(self.nbArms, dtype=float)
        self.rewardHistory = [[self.bound] for _ in range(self.nbArms)]
        self.playbuffer= list(range(self.nbArms))

    def play(self):
        if (len(self.playbuffer)==0):
            leader = randmax(self.nbDraws)
            muleader = self.meanRewards[leader]
            for a in range(self.nbArms):
                if self.nbDraws[a] < self.nbDraws[leader] and self.nbDraws[a] > 0:
                    tmua = self._dirichletmean(self.rewardHistory[a])
                    if max(self.meanRewards[a],tmua) >= muleader:
                        self.playbuffer.append(a)
        return self.playbuffer.pop()

    def _dirichletmean(self, rewards):
        w = np.random.dirichlet(np.ones(len(rewards)))
        return np.dot(w, rewards)

    def update(self, arm, reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1
        self.meanRewards[arm] = self.cumRewards[arm] / self.nbDraws[arm]
        self.rewardHistory[arm].append(reward)
