from utils import *

from src import Agent

class BESA(Agent):
    """ Best Empirical Sampled Average (2 arms) """
    def __init__(self,env):
        assert env.action_space.n == 2
        self.nbArms = 2
        Agent.__init__(self,self.nbArms,name="BESA")

    def reset(self,inistate=0):
        self.nbDraws = np.zeros(self.nbArms)
        self.rewards = [[] for a in range(self.nbArms)]
        self.sampleSize = 0
        self.sample = [[] for a in range(self.nbArms)]
        self.means = np.zeros(self.nbArms)

    def play(self,state):
        if self.sampleSize==0:
            return randmin(self.nbDraws)
        else:
            return randmax(self.means)

    def update(self, state, arm, reward, observation):
        self.rewards[arm] = self.rewards[arm]+[reward]
        self.nbDraws[arm] = self.nbDraws[arm] + 1
        self.sampleSize = int(min(self.nbDraws))

        self.samples = [ np.random.choice(self.rewards[a], size=self.sampleSize, replace=False) if self.sampleSize>0 else 0 for a in range(self.nbArms)]

        self.means = [ sum(self.samples[a])/self.sampleSize  if self.sampleSize>0 else 0 for a in range(self.nbArms) ]





