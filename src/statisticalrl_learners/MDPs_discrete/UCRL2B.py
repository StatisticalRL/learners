from src import Agent

class UCRL2B(Agent):
    def __init__(self, nS, nA, delta, K=-1):
        Agent.__init__(self, nS, nA, name="UCRL2B")
        self.nS = nS
        self.nA = nA
        self.t = 1
        self.delta = delta
        self.deltaSA = delta / (nS * nA *6*3)
        self.observations = [[], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.Nkmax = 0
        self.u = np.zeros(self.nS)

        self.r_meanestimate = np.zeros((self.nS, self.nA))
        self.r_varestimate = np.zeros((self.nS, self.nA))
        self.r_m2 = np.zeros((self.nS, self.nA))  # For Welford's algorithm to sequentially update the variance
        self.r_upper = np.zeros((self.nS, self.nA))
        self.p_estimate = np.zeros((self.nS, self.nA,self.nS))
        self.p_upper = np.zeros((self.nS, self.nA,self.nS))
        self.p_lower = np.zeros((self.nS, self.nA,self.nS))

        self.policy = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                self.policy[s, a] = 1. / self.nA


 #   def name(self):
 #       return "UCRL2B"

    # To reinitialize the learner with a given initial state inistate.
    def reset(self, inistate):
        self.t = 1
        self.observations = [[inistate], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.Nkmax = 0
        self.u = np.zeros(self.nS)

        self.r_meanestimate = np.zeros((self.nS, self.nA))
        self.r_varestimate = np.zeros((self.nS, self.nA))
        self.r_m2 = np.zeros((self.nS, self.nA))  # For Welford's algorithm
        self.r_upper = np.zeros((self.nS, self.nA))
        self.p_estimate = np.zeros((self.nS, self.nA,self.nS))
        self.p_upper = np.zeros((self.nS, self.nA,self.nS))
        self.p_lower = np.zeros((self.nS, self.nA,self.nS))

        for s in range(self.nS):
            for a in range(self.nA):
                self.policy[s, a] = 1. / self.nA
        self.new_episode()



    def q_upper(self, pest, n, delta):
        ll = np.log(n / delta)
        qup = pest + 2 * np.sqrt(pest*(1-pest) * ll / n) + 6 * ll / n
        return min(qup,1.)

    def q_lower(self, pest, n, delta):
        ll = np.log(n / delta)
        qlow = pest - (2 * np.sqrt(pest*(1-pest) * ll / n) + 6 * ll / n)
        return max(qlow,0.)

    def m_upper(self, rest, vest, n, delta):
        ll = np.log(n/delta)
        rup = rest + 2*np.sqrt(vest* ll / n) + 6*ll/n
        return min(rup,1.)


    def compute_ranges(self):
        delta = self.delta/(self.nS*self.nA*6*3)
        for s in range(self.nS):
            for a in range(self.nA):
                n = max(self.Nk[s, a],1)
                self.r_upper[s, a] = self.m_upper(self.r_meanestimate[s, a], self.r_varestimate[s, a], n, delta)
                for next_s in range(self.nS):  # self.p_estimate[s, a].keys():
                    p = self.p_estimate[s, a,next_s]
                    self.p_upper[s, a,next_s] = self.q_upper(p, n, delta)
                    self.p_lower[s, a,next_s] = self.q_lower(p, n, delta)

    # Inner maximization of the Extended Value Iteration
    def max_proba(self, sorted_indices, s, a, epsilon=10 ** (-8)):
        max_p = np.zeros(self.nS)
        delta = 1.
        for next_s in range(self.nS):
            max_p[next_s] = self.p_lower[s, a, next_s]
            delta += - max_p[next_s]

        next_s = sorted_indices[self.nS - 1]
        max_p[next_s] = self.p_lower[s, a, next_s]

        l = 0
        while (delta > epsilon) and (l <= self.nS - 1):
            idx = sorted_indices[self.nS - 1 - l]
            p_u = self.p_upper[s, a,idx]
            new_delta = min((delta, p_u - max_p[idx]))
            max_p[idx] += new_delta
            delta += - new_delta
            l += 1
        return max_p



    # The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
    def EVI(self, epsilon=0.01, max_iter = 1000):
        u0 = self.u - min(self.u)
        u1 = np.zeros(self.nS)
        sorted_indices = np.arange(self.nS)
        while True:
            for s in range(self.nS):
                temp = np.zeros(self.nA)
                for a in range(self.nA):
                    max_p = self.max_proba(sorted_indices, s, a)
                    temp[a] = self.r_upper[s, a] + sum([u * p for (u, p) in zip(u0, max_p)])

                # This implements a tie-breaking rule by choosing:  Uniform(Argmmin(Nk))
                (u1[s], arg) = allmax(temp)
                nn = [-self.Nk[s, a] for a in arg]
                (nmax, arg2) = allmax(nn)
                choice = [arg[a] for a in arg2]
                self.policy[s] = [1. / len(choice) if x in choice else 0 for x in range(self.nA)]

            diff = [abs(x - y) for (x, y) in zip(u1, u0)]
            if (max(diff) - min(diff)) < epsilon:
                self.u = u1-min(u1)
                break
            else:
                u0 = u1-min(u1)
                u1 = np.zeros(self.nS)
                sorted_indices = np.argsort(u0)

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
        self.compute_ranges()
        self.EVI(epsilon = 1./max(1,self.t))

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

        for next_s in range(self.nS):
            self.p_estimate[state, action,next_s] = self.p_estimate[state, action,next_s] * (n - 1.) / n
        self.p_estimate[state, action,observation] = self.p_estimate[state, action,observation] + 1. / n

        self.t += 1


class UCRL2_Bernstein_detRewards(Agent):
    def __init__(self, nS, nA, delta, K=-1):
        Agent.__init__(self, nS, nA, name="UCRL2B_det")
        self.nS = nS
        self.nA = nA
        self.t = 1
        self.delta = delta
        self.deltaSA = delta / (nS * nA *6*3)
        self.observations = [[], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.Nkmax = 0
        #self.policy = np.zeros((self.nS,), dtype=int)
        self.u = np.zeros(self.nS)
        # self.span = 0.
        self.r_meanestimate = np.ones((self.nS, self.nA))
        self.r_upper = np.ones((self.nS, self.nA))
        self.p_estimate = np.zeros((self.nS, self.nA,self.nS))
        self.p_upper = np.zeros((self.nS, self.nA,self.nS))
        self.p_lower = np.zeros((self.nS, self.nA,self.nS))
        self.policy = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                self.policy[s, a] = 1. / self.nA


   # def name(self):
   #     return "UCRL2B_det"

    # To reinitialize the learner with a given initial state inistate.
    def reset(self, inistate):
        self.t = 1
        self.observations = [[inistate], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.Nkmax = 0
        self.u = np.zeros(self.nS)
        # self.Rk = np.zeros((self.nS, self.nA))
        self.r_meanestimate = np.ones((self.nS, self.nA))

        self.r_upper = np.ones((self.nS, self.nA))
        self.p_estimate = np.zeros((self.nS, self.nA,self.nS))
        self.p_upper = np.zeros((self.nS, self.nA,self.nS))
        self.p_lower = np.zeros((self.nS, self.nA,self.nS))

        for s in range(self.nS):
            for a in range(self.nA):
                self.policy[s, a] = 1. / self.nA
        self.new_episode()



    def q_upper(self, pest, n, delta):
        ll = np.log(n / delta)
        qup = pest + 2 * np.sqrt(pest*(1-pest) * ll / n) + 6 * ll / n
        return min(qup,1.)

    def q_lower(self, pest, n, delta):
        ll = np.log(n / delta)
        qlow = pest - (2 * np.sqrt(pest*(1-pest) * ll / n) + 6 * ll / n)
        return max(qlow,0.)

    def m_upper(self, rest, vest, n, delta):
        ll = np.log(n/delta)
        rup = rest + 2*np.sqrt(vest* ll / n) + 6*ll/n
        return min(rup,1.)


    def compute_ranges(self):
        # delta = self.delta/(2.*self.nS*self.nA)
        delta = self.delta/(self.nS*self.nA*6*3)
        for s in range(self.nS):
            for a in range(self.nA):
                n = max(self.Nk[s, a],1)
                self.r_upper[s, a] = self.r_meanestimate[s, a]
                for next_s in range(self.nS):  # self.p_estimate[s, a].keys():
                    p = self.p_estimate[s, a,next_s]
                    self.p_upper[s, a,next_s] = self.q_upper(p, n, delta)
                    self.p_lower[s, a,next_s] = self.q_lower(p, n, delta)

    # Inner maximization of the Extended Value Iteration
    def max_proba(self, sorted_indices, s, a, epsilon=10 ** (-8)):
        max_p = np.zeros(self.nS)
        delta = 1.
        for next_s in range(self.nS):
            max_p[next_s] = self.p_lower[s, a, next_s]
            delta += - max_p[next_s]

        next_s = sorted_indices[self.nS - 1]
        max_p[next_s] = self.p_lower[s, a, next_s]

        l = 0
        while (delta > epsilon) and (l <= self.nS - 1):
            idx = sorted_indices[self.nS - 1 - l]
            p_u = self.p_upper[s, a,idx]
            new_delta = min((delta, p_u - max_p[idx]))
            max_p[idx] += new_delta
            delta += - new_delta
            l += 1
        return max_p



    # The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
    def EVI(self, r_estimate, p_estimate, epsilon=0.01, max_iter = 1000):
        u0 = self.u - min(self.u)
        u1 = np.zeros(self.nS)
        sorted_indices = np.arange(self.nS)
        while True:
            for s in range(self.nS):
                temp = np.zeros(self.nA)
                for a in range(self.nA):
                    max_p = self.max_proba(sorted_indices, s, a)
                    temp[a] = self.r_upper[s, a] + sum([u * p for (u, p) in zip(u0, max_p)])
                (u1[s], arg) = allmax(temp)
                nn = [-self.Nk[s, a] for a in arg]
                (nmax, arg2) = allmax(nn)
                choice = [arg[a] for a in arg2]
                self.policy[s] = [1. / len(choice) if x in choice else 0 for x in range(self.nA)]
                #self.policy[s] = np.random.choice(choice, 1, False, p=np.ones(len(choice)) / len(choice))
            diff = [abs(x - y) for (x, y) in zip(u1, u0)]
            if (max(diff) - min(diff)) < epsilon:
                self.u = u1-min(u1)
                break
            else:
                u0 = u1-min(u1)
                u1 = np.zeros(self.nS)
                sorted_indices = np.argsort(u0)

    def new_episode(self):
        self.sumratios = 0.
        self.updateN()
        # self.supports = self.computeEmpiricalSupports()
        self.compute_ranges()
        self.EVI(self.r_meanestimate, self.p_estimate, epsilon = 1./max(1,self.t))

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
        self.r_meanestimate[state, action] =reward

        for next_s in range(self.nS):
            self.p_estimate[state, action,next_s] = self.p_estimate[state, action,next_s] * (n - 1.) / n
        self.p_estimate[state, action,observation] = self.p_estimate[state, action,observation] + 1. / n

        self.t += 1