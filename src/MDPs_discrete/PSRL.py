import scipy.stats as stat
from src.MDPs_discrete.AgentInterface import Agent

from src.MDPs_discrete.utils import *


class PSRL(Agent):
    def __init__(self, nS, nA, delta):
        Agent.__init__(self, nS, nA,name="PSRL")
        self.nS = nS
        self.nA = nA
        self.t = 1
        self.delta = delta
        self.observations = [[], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.Nkmax = 0
        self.policy = np.zeros((self.nS, self.nA))
        self.u = np.zeros(self.nS)

        self.r_successCounts = np.ones((self.nS, self.nA))
        self.r_failureCounts = np.ones((self.nS, self.nA))
        self.p_pseudoCounts = np.ones((self.nS, self.nA, self.nS))

        self.r_sampled = np.zeros((self.nS, self.nA))
        self.p_sampled = np.zeros((self.nS, self.nA, self.nS))

    #def name(self):
    #    return "PSRL-AvR"

    # To reinitialize the learner with a given initial state inistate.
    def reset(self, inistate):
        self.t = 1
        self.observations = [[inistate], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.Nkmax = 0
        self.u = np.zeros(self.nS)
        self.policy = np.zeros((self.nS, self.nA))

        self.r_successCounts = np.ones((self.nS, self.nA))
        self.r_failureCounts = np.ones((self.nS, self.nA))
        self.p_pseudoCounts = np.ones((self.nS, self.nA, self.nS))


        self.r_sampled = np.zeros((self.nS, self.nA))
        self.p_sampled = np.zeros((self.nS, self.nA, self.nS))

        self.new_episode()


    # The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
    def VI(self, epsilon=0.01, max_iter=1000):

        u0 = self.u - min(self.u)
        u1 = np.zeros(self.nS)
        itera = 0

        while True:
            for s in range(self.nS):
                temp = np.zeros(self.nA)
                for a in range(self.nA):
                    # print("Support of ", s,a," : ", self.supports[s, a], ", ", support)
                    p = self.p_sampled[s, a]  # Allowed to sum  to <=1
                    # print("Max_p of ",s,a, " : ", max_p)
                    temp[a] = self.r_sampled[s, a] + sum([u0[ns] * p[ns] for ns in range(self.nS)])

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
            elif itera > max_iter:
                self.u = u1 - min(u1)
                print("[PSRL] No convergence in the VI at time ", self.t, " before ", max_iter, " iterations.")
                break
            else:
                u0 = u1 - min(u1)
                u1 = np.zeros(self.nS)
                itera += 1

    def new_episode(self):
        self.sumratios = 0.
        self.updateN()

        for s in range(self.nS):
            for a in range(self.nA):
                self.r_sampled[s,a] = stat.beta.rvs(self.r_successCounts[s,a],self.r_failureCounts[s,a])
                p = stat.dirichlet.rvs(alpha = self.p_pseudoCounts[s,a])
                p=p[0]
                self.p_sampled[s,a] = [p[ns] for ns in range(self.nS)]


        self.VI(epsilon=1. / max(1, self.t))

    ###### Steps and updates functions ######

    # Auxiliary function to update N the current state-action count.
    def updateN(self):
        self.Nkmax = 0.
        for s in range(self.nS):
            for a in range(self.nA):
                self.Nk[s, a] += self.vk[s, a]
                self.Nkmax = max(self.Nkmax, self.Nk[s, a])
                self.vk[s, a] = 0

    # To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
    def play(self, state):
        action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        # if self.sumratios >= 1.:  # Stoppping criterion
        if self.vk[state, action] >= max([1, self.Nk[state, action]]):  # Stopping criterion
            self.new_episode()
            action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        return action

    # To update the learner after one step of the current policy.
    def update(self, state, action, reward, observation):
        self.vk[state, action] += 1
        self.observations[0].append(observation)
        self.observations[1].append(action)
        self.observations[2].append(reward)

        self.r_successCounts[state,action] += reward
        self.r_failureCounts[state,action] += 1.-reward

        #print(state,action,observation)
        self.p_pseudoCounts[state,action,observation] +=1

        self.t += 1




def test():
    nS = 5
    nA = 2
    r_successCounts = np.ones((nS, nA))
    r_failureCounts = np.ones((nS, nA))
    p_pseudoCounts = np.ones((nS, nA, nS))

    r_sampled = np.zeros((nS, nA))
    p_sampled = np.zeros((nS, nA, nS))


    for i in range(1000):
        state = np.random.randint(nS)
        action = np.random.randint(nA)
        nstate = np.random.randint(nS)
        reward = np.random.rand()
        r_successCounts[state, action] += reward
        r_failureCounts[state, action] += 1. - reward
        p_pseudoCounts[state,action,nstate] +=1

    print("Counts:")
    for s in range(nS):
        for a in range(nA):
            print(r_successCounts[s,a], r_failureCounts[s,a], p_pseudoCounts[s,a])

    for i in range(2):
        print("Samples:")
        for s in range(nS):
            for a in range(nA):
                r_sampled[s, a] = stat.beta.rvs(r_successCounts[s, a], r_failureCounts[s, a])
                p = stat.dirichlet.rvs(alpha=p_pseudoCounts[s, a])
                p = p[0]
                p_sampled[s, a] = [p[ns] for ns in range(nS)]
                print(r_sampled[s, a], p_sampled[s, a])


