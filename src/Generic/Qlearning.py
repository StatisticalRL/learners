
import numpy as np
from src.MDPs_discrete.utils import *

class Qlearning:
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA
        self.t = 1
        self.Q=np.zeros((self.nS, self.nA))
        self.policy = np.zeros((self.nS, self.nA))
        self.alpha=np.zeros((self.nS, self.nA))
        self.gamma=0.99 #Fairly close to 1

    def name(self):
        return "Q-learning"

    def reset(self,inistate):
        self.t = 1
        for s in range(self.nS):
            for a in range(self.nA):
                self.Q[s, a] = 1./(1-self.gamma) # Optimistic initialization: Crucial for good performances.
                self.policy[s, a] = 1./self.nA
                self.alpha[s,a] = 1.

    def play(self,state):
        action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        return action

    def update(self, state, action, reward, observation):

        self.Q[state,action] = self.Q[state,action] + self.alpha[state,action]*(reward + self.gamma*max(self.Q[observation]) - self.Q[state,action])

        self.alpha[state,action] = 1./(1./self.alpha[state,action] +1.)

        (u, arg) = allmax(self.Q[state])
        self.policy[state] = [1. / len(arg) if x in arg else 0 for x in range(self.nA)]



class AdaQlearning:
    def __init__(self, nS, nA):
        self.nS = nS
        self.nA = nA
        self.t = 1
        self.Q={}
        self.policy = np.zeros((self.nS, self.nA))
        self.alpha=np.zeros((self.nS, self.nA))
        self.gamma=0.99# Fairly close to 1
        self.optValue = 1./(1-self.gamma)
        self.factor=1.01

    def name(self):
        return "AdaQ-learning"

    def reset(self,inistate):
        self.t = 1
        for s in range(self.nS):
            for a in range(self.nA):
                self.policy[s, a] = 1./self.nA
                self.alpha[s,a] = 1.

    def play(self,state):
        action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        return action

    def update(self, state, action, reward, observation):
        try:
            mQ = max([self.Q[(observation,a)] for a in range(self.nA) ])
        except  KeyError:
            for a in range(self.nA):
                self.Q[(observation, a)] = self.optValue
            mQ = self.optValue
            self.optValue = self.optValue*self.factor
            self.gamma = (self.optValue-1.)/self.optValue

        try:
            self.Q[(state,action)] = self.Q[(state,action)] + self.alpha[state,action]*(reward + self.gamma*mQ - self.Q[(state,action)])

        except  KeyError:
            for a in range(self.nA):
                self.Q[state,a]= self.optValue
            self.Q[(state, action)] = self.Q[(state, action)] + self.alpha[state, action] * (
                        reward + self.gamma * mQ - self.Q[(state, action)])

            self.optValue = self.optValue*self.factor
            self.gamma = (self.optValue-1.)/self.optValue

        self.alpha[state,action] = 1./(1./self.alpha[state,action] +1.)

        (u, arg) = allmax([self.Q[(state,a)]for a in range(self.nA)])
        self.policy[state] = [1. / len(arg) if x in arg else 0 for x in range(self.nA)]