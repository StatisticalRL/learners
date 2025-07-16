import numpy as np
from statisticalrl_learners.MDPs_discrete.AgentInterface import Agent

from statisticalrl_learners.MDPs_discrete.utils import *

# KL-UCRL is an improvement of UCRL2 introduced by Filippi et al. 2011
# This class proposes an implementation of this algorithm, it seems usefull to know that the algorithm proposed in the paper cannot be implemented as
# proposed. Some modifications have to be done (and are done here) in order to prevent some problems as: division by 0 or log(0) in function f and
# newton optimization on constant function.
class KLUCRL(Agent):
    def __init__(self, nS, nA, delta):
        Agent.__init__(self, nS, nA, name="KL-UCRL")
        """
        KL-UCRL is an improvement of UCRL2 introduced in "Filippi, Sarah, Olivier Cappé, and Aurélien Garivier. "Optimism in reinforcement learning and Kullback-Leibler divergence." 2010 48th Annual Allerton Conference on Communication, Control, and Computing (Allerton). IEEE, 2010."
        :param nS: the number of states
        :param nA: the number of actions
        :param delta:  confidence level in (0,1)
        """
        self.nS = nS
        self.nA = nA
        self.t = 1
        self.delta = delta
        self.observations = [[], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.r_distances = np.zeros((self.nS, self.nA))
        self.p_distances = np.zeros((self.nS, self.nA))
        self.Pk = np.zeros((self.nS, self.nA, self.nS))
        self.Rk = np.zeros((self.nS, self.nA))
        self.u = np.zeros(self.nS)
        self.span = []
        self.policy = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                self.policy[s, a] = 1. / self.nA

 #   def name(self):
 #       return "KL-UCRL"

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
    # KL-UCRL variant (Cp and Cr are difined as constrained constants in the paper of Filippi et al 2011, here we use the one used on the proofs
    # provided by the paper (with 2 instead of T at the initialization to prevent div by 0).
    def distances(self):
        B = np.log((2 * np.exp(1) * (self.nS) ** 2 * self.nA * np.log(max([2, self.t]))) / self.delta)
        Cp = self.nS * (B + np.log(B + 1 / np.log(max([2, self.t]))) * (1 + 1 / (B + 1 / np.log(max([2, self.t])))))
        Cr = np.sqrt((np.log(4 * self.nS * self.nA * np.log(max([2, self.t])) / self.delta)) / 1.99)
        for s in range(self.nS):
            for a in range(self.nA):
                self.r_distances[s, a] = Cr / np.sqrt(max([1, self.Nk[s, a]]))
                self.p_distances[s, a] = Cp / (max([1, self.Nk[s, a]]))

    # Key function of the problem -> solving the maximization problem is essentially based on finding roots of this function.
    def f(self, nu, p, V, Z_):  # notations of the paper
        sum1 = 0
        sum2 = 0
        for i in Z_:
            if nu == V[i]:
                return - 10 ** 10
            sum1 += p[i] * np.log(nu - V[i])
            sum2 += p[i] / (nu - V[i])
        if sum2 <= 0:
            return - 10 ** 10
        return sum1 + np.log(sum2)

    # Derivative of f, used in newton optimization.
    def diff_f(self, nu, p, V, Z_, epsilon=0):
        sum1 = 0
        sum2 = 0
        for i in range(len(p)):
            if i in Z_:
                sum1 += p[i] / (nu - V[i])
                sum2 += p[i] / (nu - V[i]) ** 2
        return sum1 - sum2 / sum1

    # The maximization algorithm proposed by Filippi et al. 2011.
    # Inputs:
    #	tau our approximation of 0
    #	tol precision required in newton optimization
    #	max_iter maximmum number of iterations on newton optimization
    def MaxKL(self, p_estimate, u0, s, a, tau=10 ** (-8), tol=10 ** (-5), max_iter=10):
        degenerate = False  # used to catch some errors
        Z, Z_, argmax = [], [], []
        maxV = max(u0)
        q = np.zeros(self.nS)
        for i in range(self.nS):
            if u0[i] == maxV:
                argmax.append(i)
            if p_estimate[s, a, i] > tau:
                Z_.append(i)
            else:
                Z.append(i)
        I = []
        test0 = False
        for i in argmax:
            if i in Z:
                I.append(i)
                test0 = True
        if test0:
            test = [(self.f(u0[i], p_estimate[s, a], u0, Z_) < self.p_distances[s, a]) for i in I]
        else:
            test = [False]
        if (True in test) and (maxV > 0):  # List I must not and cannot be empty if this is true.
            for i in range(len(test)):
                if test[i]:  # it has to happen because of previous if
                    nu = u0[I[i]]
                    break
            r = 1 - np.exp(self.f(nu, p_estimate[s, a], u0, Z_) - self.p_distances[s, a])
            for i in I:  # We want sum(q[i]) for i in I = r.
                q[i] = r / len(I)
        else:
            vzmax = max([u0[i] for i in Z_])
            # The following replaces Newton's steps by a simple Dichotomic search.
            # The later requires more steps to get same precision, but is more numericlaly stable, plus each iteration is O(1) versus O(S) for Newton's steps.
            nu = search_up( lambda x: self.f(x, p_estimate[s, a], u0, Z_) <= self.p_distances[s, a], vzmax+10**7,vzmax+10**(-7))
            r=0.
            # if len(     Z) >= self.nS - 1:  # To prevent the algorithm from running the Newton optimization on a constant or undefined function.
            #     degenerate = True
            #     q = p_estimate[s, a]
            # else:
            #     VZ_ = []
            #     for i in range(len(u0)):
            #         if p_estimate[s, a, i] > tau:
            #             VZ_.append(u0[i])
            #     nu0 = 1.1 * max(
            #         VZ_)  # This choice of initialization is based on the Matlab Code provided by Mohammad Sadegh Talebi, the one
            #     # provided by the paper leads to many errors while T is small.
            #     # about the following (unused) definition of nu0 see apendix B of Filippi et al 2011
            #     # nu0 = np.sqrt((sum([p_estimate[s, a, i] * u0[i]**2 for i in range(self.nS)]) -
            #     #			  (sum([p_estimate[s, a, i] * u0[i] for i in range(self.nS)]))**2) / (2 * self.p_distances[s, a]))
            #     #r = 0.
            #     #nu1 = 0
            #     err_nu = 10 ** 10
            #     k = 1
            #     while (err_nu >= tol) and (k < max_iter):
            #         nu1 = nu0 - (self.f(nu0, p_estimate[s, a], u0, Z_) - self.p_distances[s, a]) / (
            #             self.diff_f(nu0, p_estimate[s, a], u0, Z_))
            #         if nu1 < max(
            #                 VZ_):  # f defined on ]max(VZ_); +inf[ we have to prevent newton optimization from going out from the definition interval
            #             nu1 = max(VZ_) + tol
            #             nu0 = nu1
            #             k += 1
            #             break
            #         else:
            #             err_nu = np.abs(nu1 - nu0)
            #             k += 1
            #             nu0 = nu1
            #     nu = nu0
        if not degenerate:
            q_tilde = np.zeros(self.nS)
            for i in Z_:
                if np.abs(nu-u0[i])<tol:
                    q_tilde[i] = p_estimate[s, a, i] * 10 ** 10
                else:
                    q_tilde[i] = p_estimate[s, a, i] / (nu - u0[i])
            sum_q_tilde = sum(q_tilde)
            for i in Z_:
                q[i] = ((1 - r) * q_tilde[i]) / sum_q_tilde
        return q

    def EVI(self, r_estimate, p_estimate, epsilon=0.1):
        u0 = self.u - min(self.u)
        u1 = np.zeros(self.nS)
        tau = 10 ** (-6)
        maxiter = 1000
        niter = 0
        while True:
            for s in range(self.nS):
                temp = np.zeros(self.nA)
                test0 = (False in [tau > u for u in u0])  # Test u0 != [0,..., 0]
                for a in range(self.nA):
                    if not test0:  # MaxKL cannot run with V = [0, 0,..., 0, 0] because function f is undifined in this case.
                        max_p = p_estimate[s, a]
                    else:
                        max_p = self.MaxKL(p_estimate, u0, s, a)
                    temp[a] = r_estimate[s, a] + self.r_distances[s, a] + sum([u * p for (u, p) in zip(u0, max_p)])

                # This implements a tie-breaking rule by choosing:  Uniform(Argmmin(Nk))
                (u1[s], arg) = allmax(temp)
                nn = [-self.Nk[s, a] for a in arg]
                (nmax, arg2) = allmax(nn)
                choice = [arg[a] for a in arg2]
                self.policy[s] = [1. / len(choice) if x in choice else 0 for x in range(self.nA)]

            diff = [x - y for (x, y) in zip(u1, u0)]
            if (max(diff) - min(diff)) < epsilon:
                self.u = u1 - min(u1)
                break
            else:
                u0 = u1 - min(u1)
                u1 = np.zeros(self.nS)
            if niter > maxiter:
                self.u = u1- min(u1)
                break
            else:
                niter += 1

    # To start a new episode (init var, computes estimates and run EVI).
    def new_episode(self):
        self.updateN()  # Don't run it after the reinitialization of self.vk
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
