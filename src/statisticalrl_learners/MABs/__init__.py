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


class BatchBanditAgent:
    def __init__(self, nA, name="BatchBanditAgent"):
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


    def batchplay(self,batchsize):
        return [self.play() for b in range(batchsize)]

    def batchupdate(self, batcharm, batchreward):
        for arm, reward in zip(batcharm, batchreward):
            self.update(arm, reward)