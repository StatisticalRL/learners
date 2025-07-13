import numpy as np
import copy as cp
import time
from src.utils import *
# from src.permutation_coherence import  *




def indic(x, y):
    if (x == y):
        return 1.
    return 0.

def intersects(c1, c2):
    l1, u1 = c1
    l2, u2 = c2
    return (l2 <= u1) and (u2 >= l1)

def buildIntersects(X, Cs):  # (C1,...,Ck) where C1 = [(Lx,Ux) x in X]
    # TODO : Make it computationally more efficient when K, S are large
    K = len(Cs)
    #print("K:",K)
    I = []
    for k in range(K):
        Ik = []
        for kp in range(K):
            Ikkp = []
            for i in range(len(X)):
                #Ikkpx =
                Ikkp.append([X[ip] for ip in range(len(X)) if intersects(Cs[k][i], Cs[kp][ip])])
            Ik.append(Ikkp)
        I.append(Ik)
    return I


def intersection(c1, c2):
    l1, u1 = c1
    l2, u2 = c2
    if (max(l1, l2) <= min(u1, u2)):
        return (max(l1, l2), min(u1, u2))
    return []

def union(c1, c2):
    l1, u1 = c1
    l2, u2 = c2
    return (min(l1, l2), max(u1, u2))

def unions(cs):
    c0 = cs[0]
    for i in range(len(cs) - 1):
        c0 = union(c0, cs[i + 1])
    return c0

def unionset(ss):
    samples = []
    [[samples.append(j) for j in s] for s in ss]
    return samples

def argminLsubset(I, subset):
    argmin = (-1, -1, -1)
    lmin = np.infty
    for k in range(len(I)):
        for kp in range(len(I[k])):
            for i in range(len(I[k][kp])):
                if ((k, kp, i) in subset):
                    v = len(I[k][kp][i])
                    if (v < lmin):
                        lmin = v
                        argmin = (k, kp, i, I[k][kp][i])
    return lmin, argmin

def pruneMatchings(X, I, maxL):
    # TODO : Make it computationally more efficient when K, S are large
    subset = []
    for k in range(len(I)):
        for kp in range(len(I[k])):
            if (k != kp):
                for i in range(len(I[k][kp])):
                    subset.append((k, kp, i))
    #print("Pruning")
    while (len(subset) > 0):
        #print("Subset:", subset)
        (ell, kkx) = argminLsubset(I, subset)
        k0, k1, i0, x0 = kkx
        #print("argmin", kkx, "length", ell)
        #print("Ik0k1:", I[k0][k1])
        if (ell <= maxL):  # Look for a pruning
            index0 = [i for i in range(len(X)) if I[k0][k1][i0] == I[k0][k1][i] and (k0, k1, i) in subset]
            #print("Index0", index0, [I[k0][k1][i] for i in index0])
            if (len(index0) == ell):
                for i in range(len(X)):
                    if (k0, k1, i) in subset:
                        if i not in index0:
                            Ii = [x for x in I[k0][k1][i] if (x not in I[k0][k1][i0])]
                            I[k0][k1][i] = Ii
                subset = [x for x in subset if not ((x[0], x[1]) == (k0, k1) and x[2] in index0)]
                #print("Update Ik0k1:", I[k0][k1])
            else:
                subset = [x for x in subset if not (x == (k0, k1, i0))]
        else:
            subset = []
        #print("\n")
    return I



def refinedConfidenceBounds(I, S, X, delta, computeCI):
    # TODO : Make it computationally more efficient when K, S are large
    rCs = []
    for k in range(len(I)):
        rCsk = []
        for i in range(len(X)):
            K_ki = []
            nK_ki = []
            for kp in range(len(I[k])):
                if (kp != k):
                    if (len(I[k][kp][i]) == 1):
                        K_ki.append(kp)
                    else:
                        nK_ki.append(kp)
            # print("K:",k,X[i],K_ki)
            sets = []
            for kp in K_ki:
                x = I[k][kp][i][0]
                ix = next(i for i in range(len(X)) if X[i] == x)
                sets.append(S[kp][ix])

            sets.append(S[k][i])
            #print("Sets:", len(sets))
            # c = computeCI(S[k][i],delta)
            c = computeCI(unionset(sets), delta)
            # c = computeCI(unionset([S[kp][ I[k][kp][0] ] for kp in K_ki]),delta)

            for kp in nK_ki:
                #print("C",c)
                #print("U", unions([computeCI(S[kp][j], delta) for j in range(len(X)) if X[j] in I[k][kp][i]]) )
                c = intersection(c, unions([computeCI(S[kp][j], delta) for j in range(len(X)) if X[j] in I[k][kp][i]]))

            rCsk.append(c)
        rCs.append(rCsk)
    return rCs



class CUCRL2B():
    # TODO:  avoid SxAxS tables and loops to make it scalable.
    def __init__(self, nS, nA, C, delta, K=-1):
        self.nS = nS
        self.nA = nA
        self.t = 1
        self.delta = delta
        self.deltaSA = delta / (nS * nA *6*3)
        self.observations = [[], [], []]
        self.nexts_observations = np.empty((self.nS, self.nA), dtype=object)
        self.r_observations = np.empty((self.nS, self.nA), dtype=object)
        self.Nk = np.zeros((self.nS, self.nA))
        self.vk = np.zeros((self.nS, self.nA))
        self.Nkmax = 0
        #self.policy = np.zeros((self.nS,), dtype=int)
        self.u = np.zeros(self.nS)
        self.r_meanestimate = np.zeros((self.nS, self.nA))
        self.r_varestimate = np.zeros((self.nS, self.nA))
        self.r_m2 = np.zeros((self.nS, self.nA))  # For Welford's algorithm
        self.r_upper = np.zeros((self.nS, self.nA))
        self.p_estimate = np.zeros((self.nS, self.nA,self.nS))
        self.p_upper = np.zeros((self.nS, self.nA,self.nS))
        self.p_lower = np.zeros((self.nS, self.nA,self.nS))
        self.policy = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                self.nexts_observations[s,a]= []
                self.r_observations[s,a]= []
                self.policy[s, a] = 1. / self.nA
        self.sumratios = 0
        self.sumratios_c = 0
        self.C = C #classes are numbered from 0,1,2...
        self.nC = np.max(C)+1
        self.Nk_c = np.zeros(self.nC)
        self.vk_c = np.zeros(self.nC)
        self.profile_mapping = np.zeros((self.nS, self.nA, self.nS))
        self.p_upper_refined = np.zeros((self.nS, self.nA,self.nS))
        self.p_lower_refined = np.zeros((self.nS, self.nA,self.nS))
        self.r_upper_refined = np.zeros((self.nS, self.nA))
        self.index_c = np.empty((self.nS, self.nA), dtype=int)
        nb_c = np.zeros(self.nC)
        for s in range(self.nS):
            for a in range(self.nA):
                c = self.C[s, a]
                self.index_c[s, a] = nb_c[c]
                nb_c[c] = nb_c[c] + 1


    def name(self):
        return "C_UCRL2B"

    # To reinitialize the learner with a given initial state inistate.
    def reset(self, inistate):
        self.t = 1
        self.observations = [[inistate], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.Nk_c = np.zeros(self.nC)
        self.vk_c = np.zeros(self.nC)
        self.sumratios = 0
        self.sumratios_c = 0
        self.Nkmax = 0
        self.u = np.zeros(self.nS)
        # self.Rk = np.zeros((self.nS, self.nA))
        self.r_meanestimate = np.zeros((self.nS, self.nA))
        self.r_varestimate = np.zeros((self.nS, self.nA))
        self.r_m2 = np.zeros((self.nS, self.nA))  # For Welford's algorithm

        self.r_upper = np.zeros((self.nS, self.nA))
        self.p_estimate = np.zeros((self.nS, self.nA,self.nS))
        self.p_upper = np.zeros((self.nS, self.nA,self.nS))
        self.p_lower = np.zeros((self.nS, self.nA,self.nS))

        for s in range(self.nS):
            for a in range(self.nA):
                self.nexts_observations[s,a]= []
                self.r_observations[s,a]= []
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

    def compute_CI(self,sample,delta):
        n = max(len(sample),1)
        pest = sum(sample)/n
        return (self.q_lower(pest,n,delta),self.q_upper(pest,n,delta))

    def compute_refined_bounds(self):
        inittime = time.time()
        print("I:",inittime)
        confidences = np.empty((self.nC), dtype=object)
        samples = np.empty((self.nC), dtype=object)
        for c in range(self.nC):
            confidences[c] = []
            samples[c] = []
        for s in range(self.nS):
            for a in range(self.nA):
                c = self.C[s,a]
                confidences[c].append([(self.p_lower[s,a,next_s],self.p_upper[s,a,next_s]) for next_s in range(self.nS)])
                samples[c].append([ [indic(next_s,ns) for ns in self.nexts_observations[s,a]]  for next_s in range(self.nS)] )

        print("M:",time.time()-inittime)
        X = range(self.nS)
        ref_cb = []
        for c in range(self.nC):
            print("M_",c,":", time.time() - inittime)
            I = buildIntersects(X, confidences[c])
            print("M_",c,"A:", time.time() - inittime)
            Ip = pruneMatchings(X, I, 3)
            print("M_",c,"B:", time.time() - inittime)
            ref_cb.append(refinedConfidenceBounds(Ip, samples[c], X, self.delta, self.compute_CI))

        for s in range(self.nS):
            for a in range(self.nA):
                c = (int) (self.C[s,a])
                for next_s in range(self.nS):
                    cr= ref_cb[c][self.index_c[s,a]][next_s]
                    self.p_upper_refined[s, a, next_s]=cr[1]
                    self.p_lower_refined[s, a, next_s]=cr[0]
                self.r_upper_refined[s, a] = self.r_upper[s, a]

        print("E:",time.time()-inittime)

        # print("U",self.p_upper)
        # print("Ur",self.p_upper_refined)
        # print("L",self.p_lower)
        # print("Lr",self.p_lower_refined)
        # print("R",self.r_upper)
        # print("Rr",self.r_upper_refined)




    # Inner maximization of the Extended Value Iteration
    def max_proba(self, sorted_indices, s, a, epsilon=10 ** (-8)):
        max_p = np.zeros(self.nS)
        delta = 1.
        for next_s in range(self.nS):
            max_p[next_s] = self.p_lower_refined[s, a, next_s]
            delta += - max_p[next_s]

        next_s = sorted_indices[self.nS - 1]
        max_p[next_s] = self.p_lower_refined[s, a, next_s]

        l = 0
        while (delta > epsilon) and (l <= self.nS - 1):
            idx = sorted_indices[self.nS - 1 - l]
            p_u = self.p_upper_refined[s, a,idx]
            new_delta = min((delta, p_u - max_p[idx]))
            max_p[idx] += new_delta
            delta += - new_delta
            l += 1
        return max_p



    # The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
    def EVI(self, epsilon=0.01,max_iter=1000):
        u0 = self.u - min(self.u)
        u1 = np.zeros(self.nS)
        itera=0
        while True:
            sorted_indices = np.argsort(u0)  # sorted in ascending orders
            for s in range(self.nS):
                temp = np.zeros(self.nA)
                for a in range(self.nA):
                    max_p = self.max_proba(sorted_indices, s, a)
                    temp[a] = self.r_upper_refined[s, a] + sum([u * p for (u, p) in zip(u0, max_p)])

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
                print("No convergence in the EVI at time ", self.t, " before ", max_iter, " iterations.")
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
        self.compute_ranges()
        self.compute_refined_bounds()
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
        for c in range(self.nC):
            self.Nk_c[c]+= self.vk_c[c]
            self.vk_c[c] = 0


    # To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
    def play(self, state):
        action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        #c = self.C[state,action]
        if self.sumratios_c >= 1.:  # Stoppping criterion
        #if self.vk[state, action] >= max([1, self.Nk[state, action]]):  # Stopping criterion
        #if self.vk_c[c] >= max([1, self.Nk_c[c]]):  # Stopping criterion

            self.new_episode()
            action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        return action

    # To update the learner after one step of the current policy.
    def update(self, state, action, reward, observation):
        self.vk[state, action] += 1
        self.vk_c[self.C[state, action]] += 1
        self.sumratios = self.sumratios + 1. / max([1, self.Nk[state, action]])
        self.sumratios_c = self.sumratios + 1. / max([1, self.Nk_c[self.C[state, action]]])
        self.observations[0].append(observation)
        self.observations[1].append(action)
        self.observations[2].append(reward)
        self.nexts_observations[state, action].append(observation)
        self.r_observations[state, action].append(reward)

        n = max(1, self.Nk[state, action] + self.vk[state, action])
        Delta = reward - self.r_meanestimate[state, action]
        self.r_meanestimate[state, action] += Delta / n
        Delta2 = reward - self.r_meanestimate[state, action]
        self.r_m2[state, action] += Delta * Delta2

        for next_s in range(self.nS):
            self.p_estimate[state, action,next_s] = self.p_estimate[state, action,next_s] * (n - 1.) / n
        self.p_estimate[state, action,observation] = self.p_estimate[state, action,observation] + 1. / n

        self.t += 1




class CUCRL2_Bernstein_detRewards():
    def __init__(self, nS, nA, C, delta, K=-1):
        self.nS = nS
        self.nA = nA
        self.t = 1
        self.delta = delta
        self.deltaSA = delta / (nS * nA *6*3*nS)
        self.observations = [[], [], []]
        self.nexts_observations = np.empty((self.nS, self.nA), dtype=object)
        self.Nk = np.zeros((self.nS, self.nA))
        self.vk = np.zeros((self.nS, self.nA))
        self.Nkmax = 0
        #self.policy = np.zeros((self.nS,), dtype=int)
        self.u = np.zeros(self.nS)
        self.r_meanestimate = np.ones((self.nS, self.nA))
        self.r_upper = np.ones((self.nS, self.nA))
        self.p_estimate = np.zeros((self.nS, self.nA,self.nS))
        self.p_upper = np.zeros((self.nS, self.nA,self.nS))
        self.p_lower = np.zeros((self.nS, self.nA,self.nS))
        self.policy = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                self.nexts_observations[s,a]= []
                self.policy[s, a] = 1. / self.nA
        self.sumratios = 0
        self.sumratios_c = 0
        self.C = C #classes are numbered from 0,1,2...
        self.nC = np.max(C)+1
        self.Nk_c = np.zeros(self.nC)
        self.vk_c = np.zeros(self.nC)
        self.profile_mapping = np.zeros((self.nS, self.nA, self.nS))
        self.p_upper_refined = np.zeros((self.nS, self.nA,self.nS))
        self.p_lower_refined = np.zeros((self.nS, self.nA,self.nS))
        self.r_upper_refined = np.zeros((self.nS, self.nA))
        self.index_c = np.empty((self.nS, self.nA), dtype=int)
        nb_c = np.zeros(self.nC)
        for s in range(self.nS):
            for a in range(self.nA):
                c = self.C[s, a]
                self.index_c[s, a] = nb_c[c]
                nb_c[c] = nb_c[c] + 1


    def name(self):
        return "C_UCRL2B"

    # To reinitialize the learner with a given initial state inistate.
    def reset(self, inistate):
        self.t = 1
        self.observations = [[inistate], [], []]
        self.vk = np.zeros((self.nS, self.nA))
        self.Nk = np.zeros((self.nS, self.nA))
        self.Nk_c = np.zeros(self.nC)
        self.vk_c = np.zeros(self.nC)
        self.sumratios = 0
        self.sumratios_c = 0
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
                self.nexts_observations[s,a]= []
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

    def compute_CI(self,sample,delta):
        n = max(len(sample),1)
        pest = sum(sample)/n
        return (self.q_lower(pest,n,delta),self.q_upper(pest,n,delta))

    def compute_refined_bounds(self):
        confidences = np.empty((self.nC), dtype=object)
        samples = np.empty((self.nC), dtype=object)
        for c in range(self.nC):
            confidences[c] = []
            samples[c] = []
        for s in range(self.nS):
            for a in range(self.nA):
                c = self.C[s,a]
                confidences[c].append([(self.p_lower[s,a,next_s],self.p_upper[s,a,next_s]) for next_s in range(self.nS)])
                samples[c].append([ [indic(next_s,ns) for ns in self.nexts_observations[s,a]]  for next_s in range(self.nS)] )
        X = range(self.nS)
        ref_cb = []
        for c in range(self.nC):
            I = buildIntersects(X, confidences[c])
            Ip = pruneMatchings(X, I, 5)
            ref_cb.append(refinedConfidenceBounds(Ip, samples[c], X, self.delta, self.compute_CI))
            #ref_cb[c][k][s] = (low,up)

        for s in range(self.nS):
            for a in range(self.nA):
                c = (int) (self.C[s,a])
                for next_s in range(self.nS):
                    cr= ref_cb[c][self.index_c[s,a]][next_s]
                    self.p_upper_refined[s, a, next_s]=cr[1]
                    self.p_lower_refined[s, a, next_s]=cr[0]
                self.r_upper_refined[s, a] = self.r_upper[s, a]
        print("Refined confidence bounds computed at time ", self.t)

        # print("U",self.p_upper)
        # print("Ur",self.p_upper_refined)
        # print("L",self.p_lower)
        # print("Lr",self.p_lower_refined)
        # print("R",self.r_upper)
        # print("Rr",self.r_upper_refined)




    # Inner maximization of the Extended Value Iteration
    def max_proba(self, sorted_indices, s, a, epsilon=10 ** (-8)):
        max_p = np.zeros(self.nS)
        delta = 1.
        for next_s in range(self.nS):
            max_p[next_s] = self.p_lower_refined[s, a, next_s]
            delta += - max_p[next_s]

        next_s = sorted_indices[self.nS - 1]
        max_p[next_s] = self.p_lower_refined[s, a, next_s]

        l = 0
        while (delta > epsilon) and (l <= self.nS - 1):
            idx = sorted_indices[self.nS - 1 - l]
            p_u = self.p_upper_refined[s, a,idx]
            new_delta = min((delta, p_u - max_p[idx]))
            max_p[idx] += new_delta
            delta += - new_delta
            l += 1
        return max_p



    # The Extend Value Iteration algorithm (approximated with precision epsilon), in parallel policy updated with the greedy one.
    def EVI(self, epsilon=0.01, max_iter = 1000):
        u0 = self.u - min(self.u)
        u1 = np.zeros(self.nS)
        itera = 0
        while True:
            sorted_indices = np.argsort(u0)  # sorted in ascending orders
            for s in range(self.nS):
                temp = np.zeros(self.nA)
                for a in range(self.nA):
                    max_p = self.max_proba(sorted_indices, s, a)
                    temp[a] = self.r_upper_refined[s, a] + sum([u * p for (u, p) in zip(u0, max_p)])

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
                print("No convergence in the EVI at time ", self.t, " before ", max_iter, " iterations.")
                break
            else:
                u0 = u1 - min(u1)
                u1 = np.zeros(self.nS)
                itera += 1

    def new_episode(self):
        self.sumratios = 0.
        self.updateN()
        self.compute_ranges()
        self.compute_refined_bounds()
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
        for c in range(self.nC):
            self.Nk_c[c]+= self.vk_c[c]
            self.vk_c[c] = 0


    # To chose an action for a given state (and start a new episode if necessary -> stopping criterion defined here).
    def play(self, state):
        action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        #c = self.C[state,action]
        if self.sumratios_c >= 1.:  # Stoppping criterion
        #if self.vk[state, action] >= max([1, self.Nk[state, action]]):  # Stopping criterion
        #if self.vk_c[c] >= max([1, self.Nk_c[c]]):  # Stopping criterion

            self.new_episode()
            action = categorical_sample([self.policy[state, a] for a in range(self.nA)], np.random)
        return action

    # To update the learner after one step of the current policy.
    def update(self, state, action, reward, observation):
        self.vk[state, action] += 1
        self.vk_c[self.C[state, action]] += 1
        self.sumratios = self.sumratios + 1. / max([1, self.Nk[state, action]])
        self.sumratios_c = self.sumratios + 1. / max([1, self.Nk_c[self.C[state, action]]])
        self.observations[0].append(observation)
        self.observations[1].append(action)
        self.observations[2].append(reward)
        self.nexts_observations[state, action].append(observation)

        n = max(1, self.Nk[state, action] + self.vk[state, action])
        self.r_meanestimate[state, action] =reward

        for next_s in range(self.nS):
            self.p_estimate[state, action,next_s] = self.p_estimate[state, action,next_s] * (n - 1.) / n
        self.p_estimate[state, action,observation] = self.p_estimate[state, action,observation] + 1. / n

        self.t += 1
