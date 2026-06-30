
from statisticalrl_learners.MABs import BanditAgent
from statisticalrl_learners.Generic.utils import *

class BESA(BanditAgent):
    """ Best Empirical Sampled Average (2 arms) """
    def __init__(self,env):
        assert env.action_space.n == 2
        BanditAgent.__init__(self, 2, name="BESA")

    def reset(self,inistate=0):
        self.nbDraws = np.zeros(self.nA)
        self.rewards = [[] for a in range(self.nA)]
        self.sampleSize = 0
        self.samples = [[] for a in range(self.nA)]
        self.means = np.zeros(self.nA)

    def play(self):
        if self.sampleSize==0:
            return randmin(self.nbDraws)
        else:
            return randmax(self.means)

    def update(self, arm, reward):
        self.rewards[arm] = self.rewards[arm]+[reward]
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.sampleSize = int(min(self.nbDraws))

        self.samples = [ np.random.choice(self.rewards[a], size=self.sampleSize, replace=False) if self.sampleSize>0 else 0 for a in range(self.nA)]

        self.means = [ sum(self.samples[a])/self.sampleSize  if self.sampleSize>0 else 0 for a in range(self.nA) ]





