def build_opti(name, env, nS, nA):
        return Opti_controller(env, nS, nA)



#TODO: TO BE TESTED, CHECKED, COMPLETED

class Opti_controller:
    def __init__(self, env, nS, nA, epsilon=0.001, max_iter=100):
        """

        :param env:
        :param nS:
        :param nA:
        :param epsilon: precision of VI stoping criterion
        :param max_iter:
        """
        self.env = env
        self.nS = nS
        self.nA = nA
        self.u = np.zeros(self.nS)
        self.g = np.zeros(self.nS)
        self.epsilon = epsilon
        self.max_iter = max_iter

        self.not_converged = True
        self.transitions = np.zeros((self.nS, self.nA, self.nS))
        self.meanrewards = np.zeros((self.nS, self.nA))
        self.policy = np.zeros((self.nS, self.nA))

        try:
            for s in range(self.nS):
                for a in range(self.nA):
                    self.transitions[s, a] = self.env.getTransition(s, a)
                    self.meanrewards[s, a] = self.env.getMeanReward(s, a)
                    self.policy[s,a] = 1. / self.nA
        except AttributeError:
            for s in range(self.nS):
                for a in range(self.nA):
                    self.transitions[s, a], self.meanrewards[s, a] = self.extractRewardsAndTransitions(s, a)
                    self.policy[s, a] = 1. / self.nA

        #self.VI(epsilon=0.0000001, max_iter=100000)
        #self.GI0(epsilon=0.0001, max_iter=100000)
        self.BO_VI(epsilon=0.001, max_iter=10000)


    def extractRewardsAndTransitions(self,s,a):
        transition = np.zeros(self.nS)
        reward = 0.
        for c in self.env.P[s][a]: #c= proba, nexstate, reward, done
            transition[c[1]]=c[0]
            reward = c[2]
        return transition, reward

    def name(self):
        return "Blackwell-optimal_controller"

    def reset(self, inistate):
        ()

    def play(self, state):
        a = categorical_sample([self.policy[state,a] for a in range(self.nA)], np.random)
        return a

    def update(self, state, action, reward, observation):
        ()

    def VI(self, epsilon=0.01, max_iter=1000):
        u0 = self.u - min(self.u)  # np.zeros(self.nS)
        u1 = np.zeros(self.nS)
        itera = 0
        while True:
            sorted_indices = np.argsort(u0)  # sorted in ascending orders
            for s in range(self.nS):
                temp = np.zeros(self.nA)
                for a in range(self.nA):
                    temp[a] = self.meanrewards[s, a] + 0.999 * sum([u0[ns] * self.transitions[s, a, ns] for ns in range(self.nS)])
                (u1[s], choice) = allmax(temp)
                self.policy[s]= [ 1./len(choice) if x in choice else 0 for x in range(self.nA) ]
            diff = [abs(x - y) for (x, y) in zip(u1, u0)]
            if (max(diff) - min(diff)) < epsilon:
                self.u = u1-min(u1)
                break
            elif itera > max_iter:
                self.u = u1-min(u1)
                print("No convergence in VI at time ", self.t, " before ", max_iter, " iterations.")
                break
            else:
                u0 = u1- min(u1)
                u1 = np.zeros(self.nS)
                itera += 1

    def GI0(self, epsilon=0.01, max_iter=1000):
        g0 = self.g
        g1 = np.zeros(self.nS)
        itera = 0
        while True:
            for s in range(self.nS):
                temp = np.zeros(self.nA)
                for a in range(self.nA):
                    temp[a] = (self.meanrewards[s, a] + itera*sum(
                        [g0[ns] * self.transitions[s, a, ns] for ns in range(self.nS)]))/(itera+1)
                (g1[s], A1) = allmax(temp)
            diff = [abs(x - y) for (x, y) in zip(g1, g0)]
            if (max(diff) < epsilon) or (itera > max_iter):
                self.g = g1
                self.policy[s] = [1. / len(A1) if x in A1 else 0 for x in range(self.nA)]
                if (itera > max_iter):
                    print("No convergence in GI before ", max_iter, " iterations.")
                else:
                    print("Convergence in GI after ", itera, " iterations.")
                break
            else:
                g0 = g1
                g1 = np.zeros(self.nS)
                itera += 1

    def GI0_fast(self, epsilon=0.01, max_iter=1000):
        g0 = self.g
        #PRE-PROCESSING
        g1 = np.zeros(self.nS) #gain
        A1 = np.empty(self.nS,dtype=object) #optimalactions
        unstable = np.empty(self.nS,dtype=bool)
        for s in range(self.nS):
            unstable[s]=True
            A1[s] = []
        #PROCESSING
        itera = 0
        while (np.any(unstable)) and   (itera < max_iter):
            for s in range(self.nS):
                if (unstable[s]):
                    temp = np.zeros(self.nA)
                    for a in range(self.nA):
                        temp[a] = (self.meanrewards[s, a] + itera*sum([g0[ns] * self.transitions[s, a, ns] for ns in range(self.nS)]))/(itera+1)
                    #Compute max and Argmax:
                    (g1[s], A1[s]) = allmax(temp)

                    unstable[s] = abs(g1[s]-g0[s]) >= epsilon
                    g0[s]=g1[s]
            itera = itera+1
        #POST-PROCESSING
        self.g=g1
        for s in range(self.nS):
            self.policy[s] = [1. / len(A1[s]) if x in A1[s] else 0 for x in range(self.nA)]
        if (itera > max_iter):
            print("No convergence in GI at precision ", epsilon, " before ", max_iter, " iterations.")
        else:
            print("Convergence in GI at precision ", epsilon, " after ", itera, " iterations.")



    def BO_VI(self, epsilon=0.01, max_iter=1000):
        g_k = {}
        A_k = {}
        g_kp1 = {}
        nmax=2#self.nS
        # Ideal fixed point.
        #for n in range(self.nS)+1:
        #    for s in range(self.nS):
        #        temp = np.zeros(self.nA)
        #        for a in A_k[n][s]:
        #            temp[a] = sum(g_k[n][ns] * self.transitions[s, a, ns] for ns in range(self.nS))
        #            if (n == 1):
        #                temp[a] += self.meanrewards[s, a]
        #        (g, A) = allmax(temp)
        #        g_k[n][s] = g  - g_k[n-1][s]
        #        A_k[n+1][s] = A
        #
        for n in range(nmax+2):
            g_k[n] = np.zeros(self.nS)
            g_kp1[n] = np.zeros(self.nS)
            A_k[n] = np.empty(self.nS, dtype=object)

        g_k[-1] = np.zeros(self.nS)
        g_kp1[-1] = np.zeros(self.nS)
        for s in range(self.nS):
            A_k[0][s] = range(self.nA)
        conti = True
        itera = 0
        while conti:
            n=0
            while (n<=nmax):
                for s in range(self.nS):
                    temp = np.zeros(self.nA)
                    for a in A_k[n][s]:
                        temp[a] = sum(g_k[n][ns] * self.transitions[s, a, ns] for ns in range(self.nS))
                        if (n == 1):
                            temp[a] += self.meanrewards[s, a]
                    (g, A) = allmax(temp)
                    g_kp1[n][s] = g  - g_k[n-1][s]
                    A_k[n+1][s] = A

                n=n+1

            conti = False
            for n in range(nmax+1):
                diff = max([abs(x - y) for (x, y) in zip(g_kp1[n], g_k[n])])
                print(n,": ", g_k[n],"->", g_kp1[n], "d:", diff)
                conti = (conti) or (diff>epsilon)
                g_k[n] = g_kp1[n]
                #for s in range(self.nS):
                #    g_k[n][s]=g_k[n][s]-g_k[n][0]
                g_kp1[n] = np.zeros(self.nS)
            itera = itera + 1
            if (itera > max_iter):
                conti = False;
                print("No convergence in GI at precision ", epsilon, " before ", max_iter, " iterations.")

        for s in range(self.nS):
            self.policy[s] = [1. / len(A_k[n][s]) if x in A_k[n][s] else 0 for x in range(self.nA)]
        if (itera <=max_iter):
            print("Convergence in GI at precision ", epsilon, " after ", itera, " iterations.")
            print("Policy:\n", self.policy)






