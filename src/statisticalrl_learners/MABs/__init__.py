import numpy as np
class BanditAgent:
    def __init__(self, nA, name="BanditAgent"):
        self.nA = nA
        self.agentname= name

    def name(self):
        return self.agentname

    def reset(self):
        ()

    def play(self):
        return np.random.randint(self.nA)

    def update(self, action, reward):
        ()
