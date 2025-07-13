
from src.MABs.utils import *
from src.MABs import Agent

class Oracle(Agent):
    """Oracle"""
    def __init__(self,env):
        self.env=env
        nA = env.action_space.n
        Agent.__init__(self,nA,name="Oracle")
        self.policy = [self.env.bestarm]

    def reset(self,initstate=0):
        ()

    def play(self,state=0):
        return self.env.bestarm

    def update(self, state, action, reward, observation):
       ()