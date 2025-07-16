from statisticalrl_learners.MDPs_discrete.AgentInterface import Agent
from statisticalrl_learners.MDPs_discrete.utils import *

class UCRL3_KD(Agent):
    def __init__(self, nS, nA, env, delta):
        Agent.__init__(self, nS, nA,name="UCRL3-KD")
        self.nS = nS
        self.nA = nA
        self.t = 1
        self.delta = delta
        K=nS
        self.deltaSA = delta / (nS * nA * (3 + 3 * K))  # 3 for rewards (1 hoeffding, 2 empBernstein), nS for peeling, nS for Berend up, nS for Berend down
        self.observations = [[], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.Nkmax = 0
        self.policy = np.zeros((self.nS, self.nA))
        self.u = np.zeros(self.nS)

        self.r_meanestimate = np.zeros((self.nS, self.nA))
        self.r_varestimate = np.zeros((self.nS, self.nA))
        self.r_m2 = np.zeros((self.nS, self.nA))  # For Welford's algorithm to sequentially update the variance.
        self.supports = np.empty((self.nS, self.nA), dtype=object)
        self.r_upper = np.zeros((self.nS, self.nA))


        self.p = np.zeros((self.nS, self.nA, self.nS))

        for s in range(self.nS):
            for a in range(self.nA):
                self.p[s,a] = env.getTransition(s, a)

        self.sumratios = 0.

 #   def name(self):
 #       return "UCRL3"

    # To reinitialize the learner with a given initial state inistate.
    def reset(self, inistate):
        self.t = 1
        self.observations = [[inistate], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.Nkmax = 0
        self.u = np.zeros(self.nS)
        self.policy = np.zeros((self.nS, self.nA))

        self.r_meanestimate = np.zeros((self.nS, self.nA))
        self.r_varestimate = np.zeros((self.nS, self.nA))
        self.r_m2 = np.zeros((self.nS, self.nA))  # For Welford's algorithm
        self.r_upper = np.zeros((self.nS, self.nA))
        #self.p_estimate = np.empty((self.nS, self.nA), dtype=object)
        #self.p_upper = np.empty((self.nS, self.nA), dtype=object)
        #self.p_lower = np.empty((self.nS, self.nA), dtype=object)

        for s in range(self.nS):
            for a in range(self.nA):
                self.policy[s, a] = 1. / self.nA

        self.sumratios = 0.
        self.new_episode()

    ###### Computation of confidences intervals (named distances in implementation) ######

    def elln(self, n, delta):
        if (n <= 0):
            return np.infty
        else:
            eta = 1.12
            ell = eta * np.log(np.log(n * eta) * np.log(n * eta * eta) / (np.square(np.log(eta)) * delta))
            return ell / n

    def elln_DannEtAl(self, n, delta):
        if (n <= 0):
            return np.infty
        else:
            return (2 * np.log(np.log(max((np.exp(1), n)))) + np.log(3 / delta)) / n

    def confbound_HoeffdingLaplace(self, r, n, delta):
        if (n == 0):
            return np.infty
        return np.sqrt((1. + 1. / n) * np.log(np.sqrt(n + 1) / delta) / (2. * n))

    def confbound_EmpBersnteinPeeling(self, r, empvar, n, delta):
        if (n == 0):
            return np.infty
        elln = self.elln(n, delta)
        return np.sqrt(2. * empvar * elln) + 7. * elln / 3.


    def m_upper(self, rest, vest, n, delta):
        return min(1, rest + self.confbound_EmpBersnteinPeeling(1., vest, n, delta),
                   rest + self.confbound_HoeffdingLaplace(1., n, delta))

    def compute_ranges(self):
        delta = self.deltaSA
        for s in range(self.nS):
            for a in range(self.nA):
                n = self.Nk[s, a]
                self.r_upper[s, a] = self.m_upper(self.r_meanestimate[s, a], self.r_varestimate[s, a], n, delta)




    # The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
    def EVI(self, r_estimate, p_estimate, epsilon=0.01, max_iter=1000):

        u0 = self.u - min(self.u)
        u1 = np.zeros(self.nS)
        itera = 0

        while True:
            sorted_indices = np.argsort(u0)  # sorted in ascending orders
            kappa0 = 10 * (max(u0) - min(u0)) / (self.Nkmax ** (2. / 3.))
            for s in range(self.nS):
                temp = np.zeros(self.nA)
                for a in range(self.nA):
                    #support = self.computeSupport(s, a, u0, sorted_indices, kappa0 * len(self.supports[s, a]))
                    # print("Support of ", s,a," : ", self.supports[s, a], ", ", support)
                    #p = self.p[s,a]# self.max_proba(sorted_indices, s, a, support)  # Allowed to sum  to <=1
                    # print("Max_p of ",s,a, " : ", max_p)
                    temp[a] = self.r_upper[s, a] + sum([u0[ns] * self.p[s,a][ns] for ns in range(self.nS)])

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
                print("[UCRL3-KTP] No convergence of EVI at time ", self.t, " before ", max_iter, " iterations.")
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
                div = self.Nk[s, a]
                if (div == 0):
                    self.r_varestimate[s, a] = np.infty
                else:
                    self.r_varestimate[s, a] = self.r_m2[s, a] / div
                #self.supports[s, a] = self.p_estimate[s, a].keys()

        self.compute_ranges()
        self.EVI(self.r_meanestimate, self.p, epsilon=1. / max(1, self.t))

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
        self.sumratios = self.sumratios + 1. / max([1, self.Nk[state, action]])
        self.vk[state, action] += 1
        self.observations[0].append(observation)
        self.observations[1].append(action)
        self.observations[2].append(reward)

        n = max(1, self.Nk[state, action] + self.vk[state, action])
        Delta = reward - self.r_meanestimate[state, action]
        self.r_meanestimate[state, action] += Delta / n
        Delta2 = reward - self.r_meanestimate[state, action]
        self.r_m2[state, action] += Delta * Delta2

        # for next_s in self.p_estimate[state, action].keys():
        #     self.p_estimate[state, action][next_s] = self.p_estimate[state, action][next_s] * (n - 1.) / n
        # if (observation in self.p_estimate[state, action].keys()):
        #     self.p_estimate[state, action][observation] = self.p_estimate[state, action][observation] + 1. / n
        # else:
        #     self.p_estimate[state, action][observation] = 1. / n

        self.t += 1


# UCRL3 with nested loops in the EVI
class UCRL3_lazy(UCRL3_KD):
    def name(self):
        return "UCRL3-KTP"

    # EVI with nested loops
    def EVI(self, r_estimate, p_estimate, epsilon=0.01, max_iter=1000, nup_steps=5):
        u0 = self.u - min(self.u)
        u1 = np.zeros(self.nS)

        itera = 0

        while True:
            nup = nup_steps
            if (itera < nup_steps):  # Force checking criterion at all steps before nup_steps
                nup = 1
            for _ in range(nup):

                for s in range(self.nS):
                    temp = np.zeros(self.nA)
                    for a in range(self.nA):

                        temp[a] = self.r_upper[s, a] + sum([u0[ns] * self.p[s, a][ns] for ns in range(self.nS)])

                    # This implements a tie-breaking rule by choosing:  Uniform(Argmin(Nk))
                    (u1[s], arg) = allmax(temp)
                    nn = [self.Nk[s, a] for a in arg]
                    (nmax, arg2) = allmax(nn)
                    choice = [arg[a] for a in arg2]
                    self.policy[s] = [1. / len(choice) if x in choice else 0 for x in range(self.nA)]

                diff = [abs(x - y) for (x, y) in zip(u1, u0)]

                u0 = u1 - min(u1)
                u1 = np.zeros(self.nS)
                itera += 1

            if (max(diff) - min(diff)) < epsilon:
                self.u = u1 - min(u1)
                return None
            elif itera > max_iter:
                self.u = u1 - min(u1)
                print("[UCRL3-KTP] No convergence in the EVI at time ", self.t, " before ", max_iter, " iterations.")
                return None
