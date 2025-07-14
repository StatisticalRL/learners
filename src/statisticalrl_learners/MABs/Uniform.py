from src import Agent

class UE(Agent):
    """Uniform Exploration"""
    def __init__(self,nA):
        #self.env=env
        #self.nA = env.action_space.n
        Agent.__init__(self,nA,name="UE")

    def reset(self,initstate=0):
        self.nbDraws = np.zeros(self.nA)
        self.cumRewards = np.zeros(self.nA)

    def play(self,state=0):
        return np.random.randint(self.nA)#self.env.action_space.sample()

    def update(self, state, action, reward, observation):
        self.cumRewards[action] = self.cumRewards[action] + reward
        self.nbDraws[action] = self.nbDraws[action] + 1