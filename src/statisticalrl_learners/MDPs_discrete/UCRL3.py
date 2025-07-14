from src import Agent

class UCRL3(Agent):
    def __init__(self, nS, nA, delta, K=-1):
        Agent.__init__(self, nS, nA,name="UCRL3")
        self.nS = nS
        self.nA = nA
        self.t = 1
        self.delta = delta
        if (K == -1):
            K = nS
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
        self.p_estimate = np.empty((self.nS, self.nA), dtype=object)
        self.p_upper = np.empty((self.nS, self.nA), dtype=object)
        self.p_lower = np.empty((self.nS, self.nA), dtype=object)

        for s in range(self.nS):
            for a in range(self.nA):
                self.p_estimate[s, a] = {}
                self.p_upper[s, a] = {}
                self.p_lower[s, a] = {}
                self.supports[s, a] = []

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
        self.p_estimate = np.empty((self.nS, self.nA), dtype=object)
        self.p_upper = np.empty((self.nS, self.nA), dtype=object)
        self.p_lower = np.empty((self.nS, self.nA), dtype=object)

        self.supports = np.empty((self.nS, self.nA), dtype=object)
        for s in range(self.nS):
            for a in range(self.nA):
                self.p_estimate[s, a] = {}
                self.p_upper[s, a] = {}
                self.p_lower[s, a] = {}
                self.supports[s, a] = []
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

    def confbound_BernoulliBernsteinPeeling(self, q, n, delta):  # q- pest <=
        if (n == 0):
            return np.infty
        elln = self.elln(n, delta)
        return np.sqrt(2. * q * (1. - q) * elln) + elln / 3.

    def confbound_BernoulliSubGaussianLaplace_bar(self, q, n, delta):  # q- pest <=
        if (n == 0):
            return np.infty
        if (q < 0.5):
            if (q == 0):
                gb = 0.
            else:
                gb = (1. / 2. - q) / np.log(1. / q - 1)
        else:
            gb = q * (1. - q)

        return np.sqrt(2. * gb * (1. + 1. / n) * np.log(2. * np.sqrt(n + 1) / delta) / n)

    def confbound_BernoullisubGaussianLaplace(self, q, n, delta):  # pest - q <=
        if (n == 0):
            return np.infty
        g = 0.
        if (q > 0) and (q < 1):
            if (q == 0.5):
                g = 0.25
            else:
                g = (1. / 2. - q) / np.log(1. / q - 1)
        return np.sqrt(2. * g * (1. + 1. / n) * np.log(2. * np.sqrt(n + 1) / delta) / n)

    def q_upper(self, pest, n, delta):
        q = search_up(lambda x: x - pest <= min(self.confbound_BernoulliBernsteinPeeling(x, n, delta),
                                                self.confbound_BernoulliSubGaussianLaplace_bar(x, n, delta)), 1., pest,
                      epsilon=0.0001)
        return q
    # In case one wants to use local KL optimization instead; (self.deltaSA needs to be modified accordingly)
    # elln = self.elln(n, delta)
    # q2 = search_up(lambda x: kl(pest, x) <= elln, 1., pest, epsilon=0.0001)
    # return min(q,q2)

    def q_lower(self, pest, n, delta):
        q = search_down(lambda x: pest - x <= min(self.confbound_BernoulliBernsteinPeeling(x, n, delta),
                                                  self.confbound_BernoullisubGaussianLaplace(x, n, delta)), pest, 0.,
                        epsilon=0.0001)
        return q

    # In case one wants to use local KL optimization instead; (self.deltaSA needs to be modified accordingly)
    # elln = self.elln(n,delta)
    # q2 = search_down(lambda x: kl(pest, x) <= elln, pest, 0., epsilon=0.0001)
    # return max(q,q2)

    def m_upper(self, rest, vest, n, delta):
        return min(1, rest + self.confbound_EmpBersnteinPeeling(1., vest, n, delta),
                   rest + self.confbound_HoeffdingLaplace(1., n, delta))

    def compute_ranges(self):
        delta = self.deltaSA
        for s in range(self.nS):
            for a in range(self.nA):
                n = self.Nk[s, a]
                self.r_upper[s, a] = self.m_upper(self.r_meanestimate[s, a], self.r_varestimate[s, a], n, delta)
                self.p_upper[s, a]['out'] = self.q_upper(0., n, delta)  # Outside support
                for next_s in self.supports[s, a]:  # self.p_estimate[s, a].keys():
                    p = self.p_estimate[s, a][next_s]
                    self.p_upper[s, a][next_s] = self.q_upper(p, n, delta)
                    self.p_lower[s, a][next_s] = self.q_lower(p, n, delta)
                    if (p >= 1.) and (n >= 1):
                        # Confidence bounds are not tight for such atypical sequences of observations.
                        # TODO: PLEASE CHANGE THIS LINE IN CASE YOU WANT TO HANDLE THIS SPECIAL CASE DIFFERENTLY
                        # self.p_lower[s, a][next_s] = 1.
                        self.p_lower[s, a][next_s] = max((0.5) ** (1. / n), self.p_lower[s, a][next_s])

    ###### Functions used to initialize an episode ######
    # Auxilliary function used to compute the Fplus (inside the extended support)
    # (compute the inner maximization in order to compute Fplus)
    def aux_compute_Fplus_in(self, s, a, sorted_indices, p, idx, support):
        sum_p = sum(p)
        while sum_p > 1 and idx < self.nS:
            if sorted_indices[idx] in support:
                p_l = 0.
                if sorted_indices[idx] in self.p_lower[s, a].keys():
                    p_l = self.p_lower[s, a][sorted_indices[idx]]
                temp = max((0, p[sorted_indices[idx]] - sum_p + 1, p_l))
                sum_p -= p[sorted_indices[idx]] - temp
                p[sorted_indices[idx]] = temp
                # if p[sorted_indices[idx]] - temp <= 0: # ????
                idx += 1
            else:
                idx += 1
        return p, idx

    # Auxilliary function used to compute the Fplus (outside of the extended support)
    # (compute the inner maximization in order to compute Fplus)
    def aux_compute_Fplus_out(self, s, a, sorted_indices, p, idx, support):
        sum_p = sum(p)
        while sum_p < 1 and idx >= 0:
            if sorted_indices[idx] not in support:
                if sorted_indices[idx] not in self.p_upper[s, a].keys():
                    p_u = self.p_upper[s, a]['out']
                else:
                    p_u = self.p_upper[s, a][sorted_indices[idx]]
                temp = min((p_u, max(0, 1 - sum_p)))
                sum_p += temp
                p[sorted_indices[idx]] = temp
                # if self.p_upper[s, a, sorted_indices[idx]] >= 1 - sum_p:
                idx -= 1
            else:
                idx -= 1
        return p, idx

    # Auxiliary function for the inner maximization of the EVI, dealing with the Near Optimistic Optimization based on the support
    # for transition function used in the inner maximization
    def computeSupport(self, s, a, u, sorted_indices, kappa=np.infty):
        support = []
        for next_s in self.supports[s, a]:
            support.append(next_s)
        if (sorted_indices[self.nS - 1] not in support):
            support.append(sorted_indices[self.nS - 1])
        min_u = min(u)


        p_in = {}
        for next_s in support:
            if next_s in self.p_upper[s, a].keys():
                p_in[next_s] = self.p_upper[s, a][next_s]
            else:
                p_in[next_s] = self.p_upper[s, a]['out']
        p_out = {}

        idx_in = 0
        idx_out = self.nS - 1
        p_out[idx_out] = 0.


        p_in, idx_in = self.aux_compute_Fplus_in(s, a, sorted_indices, p_in, idx_in, support)
        p_out, idx_out = self.aux_compute_Fplus_out(s, a, sorted_indices, p_out, idx_out, support)

        temp = sum([p_in[next_s] * (u[next_s] - min_u) for next_s in p_in.keys()])
        temp2 = sum([p_out[next_s] * (u[next_s] - min_u) for next_s in p_out.keys()])

        ll = 1

        while temp2 > min(kappa, temp) and ll < self.nS - 1:
            while sorted_indices[
                self.nS - 1 - ll] in support and ll < self.nS - 1:  # To add something that is actually not in the empirical support
                ll += 1
            next_s = sorted_indices[self.nS - 1 - ll]
            if (next_s not in support):
                support.append(next_s)


            if self.nS - 1 - ll > idx_in:
                if next_s in self.p_upper[s, a].keys():
                    p_in[next_s] = self.p_upper[s, a][next_s]
                else:
                    p_in[next_s] = self.p_upper[s, a]['out']
                p_in, idx_in = self.aux_compute_Fplus_in(s, a, sorted_indices, p_in, idx_in, support)
            p_out[next_s] = 0
            p_out, idx_out = self.aux_compute_Fplus_out(s, a, sorted_indices, p_out, idx_out, support)
            temp = sum([p_in[i] * (u[i] - min_u) for i in p_in.keys()])
            temp2 = sum([p_out[i] * (u[i] - min_u) for i in p_out.keys()])

        return support

    # Inner maximization of the Extended Value Iteration
    def max_proba(self, sorted_indices, s, a, support, epsilon=10 ** (-8)):
        max_p = {}
        delta = 1.
        for next_s in support:
            if next_s in self.p_lower[s, a].keys():
                max_p[next_s] = self.p_lower[s, a][next_s]
            else:
                max_p[next_s] = 0.
            delta += - max_p[next_s]
        next_s = sorted_indices[self.nS - 1]
        if next_s in self.p_lower[s, a].keys():
            max_p[next_s] = self.p_lower[s, a][next_s]
        else:
            max_p[next_s] = 0.
        l = 0
        while (delta > epsilon) and (l <= self.nS - 1):
            idx = sorted_indices[self.nS - 1 - l]
            if (l == 0) or (idx in support):
                if (idx in self.p_upper[s, a].keys()):
                    p_u = self.p_upper[s, a][idx]
                else:
                    p_u = self.p_upper[s, a]['out']
                new_delta = min((delta, p_u - max_p[idx]))
                max_p[idx] += new_delta
                delta += - new_delta
            l += 1
        return max_p

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
                    support = self.computeSupport(s, a, u0, sorted_indices, kappa0 * len(self.supports[s, a]))
                    # print("Support of ", s,a," : ", self.supports[s, a], ", ", support)
                    max_p = self.max_proba(sorted_indices, s, a, support)  # Allowed to sum  to <=1
                    # print("Max_p of ",s,a, " : ", max_p)
                    temp[a] = self.r_upper[s, a] + sum([u0[ns] * max_p[ns] for ns in max_p.keys()])

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
                print("[UCRL3] No convergence of EVI at time ", self.t, " before ", max_iter, " iterations.")
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
                self.supports[s, a] = self.p_estimate[s, a].keys()

        self.compute_ranges()
        self.EVI(self.r_meanestimate, self.p_estimate, epsilon=1. / max(1, self.t))

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

        for next_s in self.p_estimate[state, action].keys():
            self.p_estimate[state, action][next_s] = self.p_estimate[state, action][next_s] * (n - 1.) / n
        if (observation in self.p_estimate[state, action].keys()):
            self.p_estimate[state, action][observation] = self.p_estimate[state, action][observation] + 1. / n
        else:
            self.p_estimate[state, action][observation] = 1. / n

        self.t += 1


# UCRL3 with nested loops in the EVI
class UCRL3_lazy(UCRL3):
    def name(self):
        return "UCRL3"

    # EVI with nested loops
    def EVI(self, r_estimate, p_estimate, epsilon=0.01, max_iter=1000, nup_steps=5):
        u0 = self.u - min(self.u)
        u1 = np.zeros(self.nS)

        itera = 0
        max_p = np.empty((self.nS, self.nA), dtype=object)
        for s in range(self.nS):
            for a in range(self.nA):
                max_p[s, a] = {}

        while True:
            sorted_indices = np.argsort(u0)
            for s in range(self.nS):
                for a in range(self.nA):
                    support = self.computeSupport(s, a, u0, sorted_indices)
                    max_p[s, a] = self.max_proba(sorted_indices, s, a, support)
            nup = nup_steps
            if (itera < nup_steps):  # Force checking criterion at all steps before nup_steps
                nup = 1
            for _ in range(nup):

                for s in range(self.nS):
                    temp = np.zeros(self.nA)
                    for a in range(self.nA):

                        temp[a] = self.r_upper[s, a] + sum([u0[ns] * max_p[s, a][ns] for ns in max_p[s, a].keys()])

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
                print("No convergence in the EVI at time ", self.t, " before ", max_iter, " iterations.")
                return None
