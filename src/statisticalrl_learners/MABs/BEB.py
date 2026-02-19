
from statisticalrl_learners.MABs import BanditAgent, BatchBanditAgent
from statisticalrl_learners.MABs.utils import *

class BEB(BatchBanditAgent):
    """Bounded-CVaR-Thompson-Sampling Batch for CVaR=Expectation"""
    def __init__(self,nbArms,bound=1.):
        self.nbArms = nbArms
        BatchBanditAgent.__init__(self, self.nbArms, name="BEB"+str(int(bound)))
        self.bound = bound

    def reset(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.meanRewards = np.zeros(self.nbArms, dtype=float)
        self.rewardHistory = [[self.bound] for _ in range(self.nbArms)]
        self.playbuffer= list(range(self.nbArms))

    def play(self):
        return randmax([self._dirichletmean(self.rewardHistory[a]) for a in range(self.nbArms)])

    def _dirichletmean(self, rewards):
        w = np.random.dirichlet(np.ones(len(rewards)))
        return np.dot(w, rewards)

    def update(self, arm, reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1
        self.meanRewards[arm] = self.cumRewards[arm] / self.nbDraws[arm]
        self.rewardHistory[arm].append(reward)

    def batchplay(self,batchsize):
        batcharms=[]
        for b in range(batchsize):
            a = self.play()
            batcharms.append(a)
            self.nbDraws[a]= self.nbDraws[a] +1
        return batcharms

    def batchupdate(self,batcharm,batchreward):
         for arm,reward in zip(batcharm,batchreward):
             self.cumRewards[arm] = self.cumRewards[arm] + reward
             #[Already done] self.nbDraws[arm] = self.nbDraws[arm] + 1
             self.meanRewards[arm] = self.cumRewards[arm] / self.nbDraws[arm]
             self.rewardHistory[arm].append(reward)



class BEBnaive(BatchBanditAgent):
    """Bounded-CVaR-Thompson-Sampling Batch for CVaR=Expectation"""
    def __init__(self,nbArms,bound=1.):
        self.nbArms = nbArms
        BatchBanditAgent.__init__(self, self.nbArms, name="BEBnaive"+str(int(bound)))
        self.bound = bound

    def reset(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.meanRewards = np.zeros(self.nbArms, dtype=float)
        self.rewardHistory = [[self.bound] for _ in range(self.nbArms)]
        self.playbuffer= list(range(self.nbArms))

    def play(self):
        return randmax([self._dirichletmean(self.rewardHistory[a]) for a in range(self.nbArms)])

    def _dirichletmean(self, rewards):
        w = np.random.dirichlet(np.ones(len(rewards)))
        return np.dot(w, rewards)

    def update(self, arm, reward):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] +1
        self.meanRewards[arm] = self.cumRewards[arm] / self.nbDraws[arm]
        self.rewardHistory[arm].append(reward)
