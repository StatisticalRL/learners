

from statisticalrl_learners.MABs import BanditAgent, BatchBanditAgent
from statisticalrl_learners.MABs.utils import *
'''
Bernoulli distributions
'''
class IMEDDS(BatchBanditAgent):
    """Indexed Minimum Empirical Divergence"""
    def __init__(self,nbArms,kullback,bound=1.):
        self.nbArms = nbArms
        self.kl = kullback
        BatchBanditAgent.__init__(self, self.nbArms, name="B-IMED-DS"+str(int(bound)))
        self.bound = bound

    def reset(self):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.meanRewards = np.zeros(self.nbArms, dtype=float)
        self.maxMeans = 0
        self.indexes = np.zeros(self.nbArms)
        self.rewardHistory = [[self.bound] for _ in range(self.nbArms)]

    def play(self,state=0):
        return randmin(self.indexes)

    def _dirichletmean(self, rewards):
        w = np.random.dirichlet(np.ones(len(rewards)))
        return np.dot(w, rewards)

    def update(self, arm, reward):
        self.rewardHistory[arm].append(reward)
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1        
        self.meanRewards[arm] = self.cumRewards[arm] /self.nbDraws[arm]
        self.maxMeans = max(self.meanRewards)

        self._update_index()

    def _update_index(self):
        self.indexes = [self.nbDraws[a] * self.kl(self._dirichletmean(self.rewardHistory[a]), self.maxMeans) + log(
            self.nbDraws[a]) if self.nbDraws[a] > 0 else 0 for a in range(self.nbArms)]

    def batchplay(self,batchsize):
        batcharms=[]
        for b in range(batchsize):
            a = self.play()
            batcharms.append(a)
            self.nbDraws[a]= self.nbDraws[a] +1
            self._update_index()
        return batcharms

    def batchupdate(self,batcharm,batchreward):
         for arm,reward in zip(batcharm,batchreward):
             self.cumRewards[arm] = self.cumRewards[arm] + reward
             #[Already done] self.nbDraws[arm] = self.nbDraws[arm] + 1
             self.meanRewards[arm] = self.cumRewards[arm] / self.nbDraws[arm]
             self.rewardHistory[arm].append(reward)
         self._update_index()