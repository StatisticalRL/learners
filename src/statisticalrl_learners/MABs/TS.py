from src import Agent

'''
Bernoulli distributions
'''
class TS(Agent):
    """Thomson Sampling"""
    def __init__(self,nbArms):
        self.nbArms = nbArms
        Agent.__init__(self,self.nbArms,name="TS")

    def reset(self, initstate=0):
        self.nbDraws = np.zeros(self.nbArms)
        self.cumRewards = np.zeros(self.nbArms)
        self.theta = np.zeros(self.nbArms)

    def play(self,state=0):
        return randmax(self.theta)

    def update(self, state, arm, reward, observation):
        self.cumRewards[arm] = self.cumRewards[arm]+reward
        self.nbDraws[arm] = self.nbDraws[arm] + 1

        self.theta = [np.random.beta(max(self.cumRewards[a],0) + 1, max(self.nbDraws[a] - self.cumRewards[a],0) + 1) for a in range(self.nbArms)]


