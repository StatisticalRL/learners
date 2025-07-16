import copy as cp
from statisticalrl_learners.MDPs_discrete.AgentInterface import Agent
from statisticalrl_learners.MDPs_discrete.utils import *

class UCRL2(Agent):
    def __init__(self, nS, nA, delta):
        Agent.__init__(self, nS, nA, name="UCRL2")
        """
        Vanilla UCRL2 based on "Jaksch, Thomas, Ronald Ortner, and Peter Auer. "Near-optimal regret bounds for reinforcement learning." Journal of Machine Learning Research 11.Apr (2010): 1563-1600."
        :param nS: the number of states
        :param nA: the number of actions
        :param delta:  confidence level in (0,1)
        """
        self.nS = nS
        self.nA = nA
        self.t = 1
        self.delta = delta

        self.observations = [[], [], []] # list of the observed (states, actions, rewards) ordered by time
        self.vk = np.zeros((self.nS, self.nA)) #the state-action count for the current episode k
        self.Nk = np.zeros((self.nS, self.nA)) #the state-action count prior to episode k

        self.r_distances = np.zeros((self.nS, self.nA))
        self.p_distances = np.zeros((self.nS, self.nA))
        self.Pk = np.zeros((self.nS, self.nA, self.nS))
        self.Rk = np.zeros((self.nS, self.nA))

        self.u = np.zeros(self.nS)
        self.span = []
        self.policy = np.zeros((self.nS, self.nA)) # policy, seen as a stochastic policy here.
        for s in range(self.nS):
            for a in range(self.nA):
                self.policy[s, a] = 1. / self.nA

 #   def name(self):
 #       return "UCRL2"

    # Auxiliary function to update N the current state-action count.
    def updateN(self):
        for s in range(self.nS):
            for a in range(self.nA):
                self.Nk[s, a] += self.vk[s, a]

    # Auxiliary function to update R the accumulated reward.
    def updateR(self):
        self.Rk[self.observations[0][-2], self.observations[1][-1]] += self.observations[2][-1]

    # Auxiliary function to update P the transitions count.
    def updateP(self):
        self.Pk[self.observations[0][-2], self.observations[1][-1], self.observations[0][-1]] += 1

    # Auxiliary function updating the values of r_distances and p_distances (i.e. the confidence bounds used to build the set of plausible MDPs).
    def distances(self):
        for s in range(self.nS):
            for a in range(self.nA):
                self.r_distances[s, a] = np.sqrt((7 * np.log(2 * self.nS * self.nA * self.t / self.delta))
                                                 / (2 * max([1, self.Nk[s, a]])))
                self.p_distances[s, a] = np.sqrt((14 * self.nS * np.log(2 * self.nA * self.t / self.delta))
                                                 / (max([1, self.Nk[s, a]])))

    # Computing the maximum proba in the Extended Value Iteration for given state s and action a.
    def max_proba(self, p_estimate, sorted_indices, s, a):
        min1 = min([1, p_estimate[s, a, sorted_indices[-1]] + (self.p_distances[s, a] / 2)])
        max_p = np.zeros(self.nS)
        if min1 == 1:
            max_p[sorted_indices[-1]] = 1
        else:
            max_p = cp.deepcopy(p_estimate[s, a])
            max_p[sorted_indices[-1]] += self.p_distances[s, a] / 2
            l = 0
            while sum(max_p) > 1:
                max_p[sorted_indices[l]] = max([0, 1 - sum(max_p) + max_p[sorted_indices[l]]])  # Error?
                l += 1
        return max_p

    # The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
    def EVI(self, r_estimate, p_estimate, epsilon=0.01, max_iter=1000):
        u0 = self.u - min(self.u)  #sligthly boost the computation and doesn't seems to change the results
        u1 = np.zeros(self.nS)
        sorted_indices = np.arange(self.nS)
        niter = 0
        while True:
            niter += 1
            for s in range(self.nS):

                temp = np.zeros(self.nA)
                for a in range(self.nA):
                    max_p = self.max_proba(p_estimate, sorted_indices, s, a)
                    temp[a] = min((1, r_estimate[s, a] + self.r_distances[s, a])) + sum(
                        [u * p for (u, p) in zip(u0, max_p)])
                # This implements a tie-breaking rule by choosing:  Uniform(Argmmin(Nk))
                (u1[s], arg) = allmax(temp)
                nn = [-self.Nk[s, a] for a in arg]
                (nmax, arg2) = allmax(nn)
                choice = [arg[a] for a in arg2]
                self.policy[s] = [1. / len(choice) if x in choice else 0 for x in range(self.nA)]

            diff = [abs(x - y) for (x, y) in zip(u1, u0)]
            if (max(diff) - min(diff)) < epsilon:
                self.u = u1 - min(u1)
                break
            else:
                u0 = u1 - min(u1)
                u1 = np.zeros(self.nS)
                sorted_indices = np.argsort(u0)
            if niter > max_iter:
                self.u = u1 - min(u1)
                print("No convergence in EVI")
                break


    # To start a new episode (init var, computes estmates and run EVI).
    def new_episode(self):
        self.updateN()
        self.vk = np.zeros((self.nS, self.nA))
        r_estimate = np.zeros((self.nS, self.nA))
        p_estimate = np.zeros((self.nS, self.nA, self.nS))
        for s in range(self.nS):
            for a in range(self.nA):
                div = max([1, self.Nk[s, a]])
                r_estimate[s, a] = self.Rk[s, a] / div
                for next_s in range(self.nS):
                    p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
        self.distances()
        self.EVI(r_estimate, p_estimate, epsilon=1. / max(1, self.t))

    # To reinitialize the learner with a given initial state inistate.
    def reset(self, inistate):
        self.t = 1
        self.observations = [[inistate], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.u = np.zeros(self.nS)
        self.Pk = np.zeros((self.nS, self.nA, self.nS))
        self.Rk = np.zeros((self.nS, self.nA))
        self.span = [0]
        for s in range(self.nS):
            for a in range(self.nA):
                self.policy[s, a] = 1. / self.nA
        self.new_episode()

    # To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
    def play(self, state):
        action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        if self.vk[state, action] >= max([1, self.Nk[state, action]]):  # Stoppping criterion
            self.new_episode()
            action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        return action

    # To update the learner after one step of the current policy.
    def update(self, state, action, reward, observation):
        self.vk[state, action] += 1
        self.observations[0].append(observation)
        self.observations[1].append(action)
        self.observations[2].append(reward)
        self.updateP()
        self.updateR()
        self.t += 1

# This "UCRL2" algorithm is a slight modfication of UCRL2. the idea is to add some forced exploration trying all the unknown action in every state befor starting the optimism phase,
# and to run a random policy in unknown states.
class UCRL2_bis(UCRL2):
    def __init__(self, nS, nA, delta):
        UCRL2.__init__(self, nS, nA,delta)
        self.nS = nS
        self.nA = nA
        self.agentname = "UCRL2_bis"
        self.t = 1
        self.delta = delta
        self.observations = [[], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.policy = np.zeros((self.nS,), dtype=int)
        self.r_distances = np.zeros((self.nS, self.nA))
        self.p_distances = np.zeros((self.nS, self.nA))
        self.Pk = np.zeros((self.nS, self.nA, self.nS))
        self.Rk = np.zeros((self.nS, self.nA))
        self.u = np.zeros(self.nS)
        self.span = []
        self.visited = np.zeros((self.nS,
                                 self.nA + 1))  # +1 to register that the state is known, the rest to make sure that every action had been tried

    # at least one time


    # To reinitialize the learner with a given initial state inistate.
    def reset(self, inistate):
        self.t = 1
        self.visited = np.zeros((self.nS, self.nA + 1))
        self.observations = [[inistate], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.new_episode()
        self.u = np.zeros(self.nS)
        self.Pk = np.zeros((self.nS, self.nA, self.nS))
        self.Rk = np.zeros((self.nS, self.nA))
        self.span = [0]

    # To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
    def play(self, state):
        if self.visited[state, -1] == 0:
            self.visited[state, -1] = 1
            self.visited[state, 0] = 1
            return 0
        else:
            for a in range(self.nA):
                if self.visited[state, a] == 0:
                    self.visited[state, a] = 1
                    return a
        action = self.policy[state]
        if self.vk[state, action] >= max([1, self.Nk[state, action]]):  # Stoppping criterion
            self.new_episode()
        action = self.policy[state]
        return action

    # To update the learner after one step of the current policy.
    def update(self, state, action, reward, observation):
        self.vk[state, action] += 1
        self.observations[0].append(observation)
        self.observations[1].append(action)
        self.observations[2].append(reward)
        self.updateP()
        self.updateR()
        self.t += 1
