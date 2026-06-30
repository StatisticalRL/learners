import numpy as np

class MDPAgent:
    def __init__(self, nS, nA, name="Agent"):
        self.nS = nS
        self.nA = nA
        self.agentname= name

    def name(self):
        return self.agentname

    def reset(self,inistate):
        ()

    def play(self,state):
        return np.random.randint(self.nA)

    def update(self, state, action, reward, observation):
        ()