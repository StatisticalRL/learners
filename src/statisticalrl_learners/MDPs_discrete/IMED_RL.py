import numpy as np
from scipy.optimize import minimize_scalar
from src import Agent


def randamax(v, t=None, i=None):
    """
    V: array of values
    T: array used to break ties
    I: array of indices from which we should return an amax
    """
    if i is None:
        idxs = np.where(v == np.amax(v))[0]
        if t is None:
            idx = np.random.choice(idxs)
        else:
            assert len(v) == len(t), f"Lengths should match: len(v)={len(v)} - len(t)={len(t)}"
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = idxs[t_idxs]
    else:
        idxs = np.where(v[i] == np.amax(v[i]))[0]
        if t is None:
            idx = i[np.random.choice(idxs)]
        else:
            assert len(v) == len(t), f"Lengths should match: len(v)={len(v)} - len(t)={len(t)}"
            t = t[i]
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = i[idxs[t_idxs]]
    return idx


def randamin(v, t=None, i=None):
    """
    v: array of values
    t: array used to break ties
    i: array of indices from which we should return an amin
    """
    if i is None:
        idxs = np.where(v == np.amin(v))[0]
        if t is None:
            idx = np.random.choice(idxs)
        else:
            assert len(v) == len(t), f"Lengths should match: len(v)={len(v)} - len(t)={len(t)}"
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = idxs[t_idxs]
    else:
        idxs = np.where(v[i] == np.amin(v[i]))[0]
        if t is None:
            idx = i[np.random.choice(idxs)]
        else:
            assert len(v) == len(t), f"Lengths should match: len(v)={len(v)} - len(t)={len(t)}"
            t = t[i]
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = i[idxs[t_idxs]]
    return idx


class IMEDRL(Agent):
    def __init__(self, nbr_states, nbr_actions, name="IMED-RL",
                 max_iter=3000, epsilon=1e-3, max_reward=1):
        Agent.__init__(self, nbr_states, nbr_actions, name=name)
        self.nS = nbr_states
        self.nA = nbr_actions
        self.dirac = np.eye(self.nS, dtype=int)
        self.actions = np.arange(self.nA, dtype=int)
        self.max_iteration = max_iter
        self.epsilon = epsilon
        self.max_reward = max_reward

        # Empirical estimates
        self.state_action_pulls = np.zeros((self.nS, self.nA), dtype=int)
        self.state_visits = np.zeros(self.nS, dtype=int)
        self.rewards = np.zeros((self.nS, self.nA)) + 0.5
        self.transitions = np.ones((self.nS, self.nA, self.nS)) / self.nS
        self.all_selected = np.zeros(self.nS, dtype=bool)
        self.phi = np.zeros(self.nS)
        self.skeleton = {s: np.arange(self.nA, dtype=int) for s in range(self.nS)}
        self.index = np.zeros(self.nA)
        self.rewards_distributions = {s: {a: {} for a in range(self.nA)} for s in range(self.nS)}

        self.s = None

    def reset(self, state):
        self.state_action_pulls = np.zeros((self.nS, self.nA), dtype=int)
        self.state_visits = np.zeros(self.nS, dtype=int)
        self.rewards = np.zeros((self.nS, self.nA)) + 0.5
        self.transitions = np.ones((self.nS, self.nA, self.nS)) / self.nS
        self.all_selected = np.zeros(self.nS, dtype=bool)
        self.phi = np.zeros(self.nS)
        self.skeleton = {s: np.arange(self.nA, dtype=int) for s in range(self.nS)}
        self.index = np.zeros(self.nA)
        self.rewards_distributions = {s: {a: {1: 0, 0.5: 1} for a in range(self.nA)} for s in range(self.nS)}

        self.s = state

    def value_iteration(self):
        ctr = 0
        stop = False
        phi = np.copy(self.phi)
        phip = np.copy(self.phi)
        while not stop:
            ctr += 1
            for state in range(self.nS):
                u = - np.inf
                for action in self.skeleton[state]:
                    psa = self.transitions[state, action]
                    rsa = self.rewards[state, action]
                    u = max(u, rsa + psa @ phi)
                phip[state] = u
            phip = phip - np.min(phip)
            delta = np.max(np.abs(phi - phip))
            phi = np.copy(phip)
            stop = (delta < self.epsilon) or (ctr >= self.max_iteration)
        self.phi = np.copy(phi)

    def update(self, state, action, reward, observation):
        na = self.state_action_pulls[state, action]
        ns = self.state_visits[state]
        r = self.rewards[state, action]
        p = self.transitions[state, action]

        self.state_action_pulls[state, action] = na + 1
        self.state_visits[state] = ns + 1
        self.rewards[state, action] = ((na + 1) * r + reward) / (na + 2)
        self.transitions[state, action] = ((na + 1) * p + self.dirac[observation]) / (na + 2)

        if reward in self.rewards_distributions[state][action].keys():
            self.rewards_distributions[state][action][reward] += 1
        else:
            self.rewards_distributions[state][action][reward] = 1

        max_na = np.max(self.state_action_pulls[state])
        mask = self.state_action_pulls[state] >= np.log(max_na)**2
        self.skeleton[state] = self.actions[mask]

        self.s = observation

        if not self.all_selected[state]:
            self.all_selected[state] = np.all(self.state_action_pulls[state] > 0)

    def multinomial_imed(self, state):
        upper_bound = self.max_reward + np.max(self.phi)
        q = self.rewards[state] + self.transitions[state] @ self.phi
        mu = np.max(q)
        u = upper_bound / (upper_bound - mu) - 1e-2

        for a in range(self.nA):
            if q[a] >= mu:
                self.index[a] = np.log(self.state_action_pulls[state, a])
            else:
                r_d = self.rewards_distributions[state][a]
                vr = np.fromiter(r_d.keys(), dtype=float)
                pr = np.fromiter(r_d.values(), dtype=float)
                pr = pr / pr.sum()

                pt = self.transitions[state][a]

                p = np.zeros(len(pr)*self.nS)
                v = np.zeros(len(pr)*self.nS)
                k = 0
                for i in range(self.nS):
                    for j in range(len(pr)):
                        p[k] = pt[i]*pr[j]
                        v[k] = self.phi[i] + vr[j]
                        k += 1

                delta = v - mu

                h = lambda x: - np.sum(p * np.log(upper_bound - delta*x))

                res = minimize_scalar(h, bounds=(0, u), method='bounded')
                x = - res.fun
                n = self.state_action_pulls[state, a]
                self.index[a] = n * x + np.log(n)

    def play(self, state):
        if not self.all_selected[state]:
            action = randamin(self.state_action_pulls[state])
        else:
            self.value_iteration()
            self.multinomial_imed(state)
            action = randamin(self.index)
        return action
