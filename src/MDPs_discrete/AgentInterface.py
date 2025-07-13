import numpy as np
import src.MDPs_discrete.utils as utils


class Agent:
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



class SequentialAgent(Agent):
    def __init__(self, nbr_states, nbr_actions, name="Sequential Agent"):
        Agent.__init__(self,nbr_states,nbr_actions,name)
        self.nS = nbr_states
        self.nA = nbr_actions
        self.dirac = np.eye(self.nS, dtype=int)

        # Empirical estimates
        self.state_action_pulls = np.zeros((self.nS, self.nA), dtype=int)
        self.state_visits = np.zeros(self.nS, dtype=int)
        self.rewards = np.zeros((self.nS, self.nA)) + 0.5
        self.transitions = np.ones((self.nS, self.nA, self.nS)) / self.nS
        self.all_selected = np.zeros(self.nS, dtype=bool)
        self.u = np.zeros(self.nS)

        self.s = None

    def reset(self, state):
        # print(self.state_visits)
        # print(np.sort(self.state_visits))
        # print(self.state_action_pulls)
        # print(self.u)
        self.state_action_pulls = np.zeros((self.nS, self.nA), dtype=int)
        self.state_visits = np.zeros(self.nS, dtype=int)
        self.rewards = np.zeros((self.nS, self.nA)) + 0.5
        self.transitions = np.ones((self.nS, self.nA, self.nS)) / self.nS
        self.all_selected = np.zeros(self.nS, dtype=bool)
        self.u = np.zeros(self.nS)

        self.s = state


    def play(self, state):
        if self.all_selected[state]:
            action = utils.randamin(self.state_action_pulls[state])
        else:
            action = utils.randamin(self.state_action_pulls[state])
        return action

    def update(self, old_state, action, reward, current_state):
        na = self.state_action_pulls[old_state, action]
        ns = self.state_visits[old_state]
        r = self.rewards[old_state, action]
        p = self.transitions[old_state, action]

        self.state_action_pulls[old_state, action] = na + 1
        self.state_visits[old_state] = ns + 1
        self.rewards[old_state, action] = ((na+1)*r + reward) / (na + 2)
        self.transitions[old_state, action] = ((na + 1)*p + self.dirac[current_state]) / (na + 2)

        self.s = current_state

        if not self.all_selected[old_state]:
            self.all_selected[old_state] = np.all(self.state_action_pulls[old_state] > 0)