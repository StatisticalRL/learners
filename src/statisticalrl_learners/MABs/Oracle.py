
from statisticalrl_learners.MABs import BanditAgent
from statisticalrl_learners.MABs.utils import *
class Oracle(BanditAgent):
    """Oracle"""
    def __init__(self,env):
        self.env=env
        nA = env.action_space.n
        BanditAgent.__init__(self, nA, name="Oracle")
        self.policy = [self.env.bestarm]

    def name(self):
        return "Oracle"
    def reset(self):
        ()

    def play(self):
        return self.env.bestarm

    def update(self, action, reward):
       ()