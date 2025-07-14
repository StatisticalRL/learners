def build_opti(name, env, nS, nA):
    #if ("2-room" in name):
    #    return  Opti_911_2room(env)
    #elif ("4-room" in name):
    #    return  Opti_77_4room(env)
    #elif ("RiverSwim" in name):
    #    return Opti_swimmer(env)
    #else:
        return Opti_controller(env, nS, nA)



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

        self.VI(epsilon=0.0000001, max_iter=100000)


    def extractRewardsAndTransitions(self,s,a):

        transition  = self.env.getTransition(s,a)
        reward = self.env.getMeanReward(s,a)
        #transition = np.zeros(self.nS)
        #reward = 0.
        #for c in self.env.P[s][a]: #c= proba, nexstate, reward, done
        #    transition[c[1]]=c[0]
        #    reward = c[2]
        return transition, reward

    def name(self):
        return "Optimal_controller"

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
            #print("[Opt]",itera)
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
                print("[Opt] No convergence in VI at time ", self.t, " before ", max_iter, " iterations.")
                break
            else:
                u0 = u1- min(u1)
                u1 = np.zeros(self.nS)
                itera += 1







class Opti_swimmer:
    def __init__(self, env):
        self.env = env
        self.policy = np.zeros(self.env.nS)

    def name(self):
        return "Opti_swimmer"

    def reset(self, inistate):
        ()

    def play(self, state):
        return 0

    def update(self, state, action, reward, observation):
        ()


class Opti_77_4room:
    def __init__(self, env):
        self.env = env
        pol = (
            [[0, 0, 0, 0, 0, 0, 0],
             [0, 1, 3, 3, 1, 1, 0],
             [0, 1, 2, 0, 1, 2, 0],
             [0, 1, 0, 0, 1, 0, 0],
             [0, 3, 3, 3, 3, 1, 0],
             [0, 3, 0, 0, 3, 1, 0],
             [0, 0, 0, 0, 0, 0, 0]]
        )
        self.policy = np.zeros(49)
        for x in range(7):
            for y in range(7):
                self.policy[x * 7 + y] = pol[x][y]
        self.mapping = env.mapping

    def name(self):
        return "Opti_77_4room"

    def reset(self, inistate):
        ()

    def play(self, state):
        s = self.mapping[state]
        return self.policy[s]

    def update(self, state, action, reward, observation):
        ()


class Opti_911_2room:
    def __init__(self, env):
        self.env = env
        pol = (
             [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 3, 3, 3, 1, 1, 1, 2, 2, 2, 0],
             [0, 3, 3, 3, 1, 1, 1, 2, 2, 2, 0],
             [0, 3, 3, 3, 3, 1, 2, 2, 2, 2, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0],
             [0, 3, 3, 3, 3, 3, 3, 3, 1, 1, 0],
             [0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        )
        self.policy = np.zeros(9 * 11)
        for x in range(9):
            for y in range(11):
                self.policy[x * 11 + y] = pol[x][y]
        self.mapping = env.mapping

    def name(self):
        return "Opti_911_2room"

    def reset(self, inistate):
        ()

    def play(self, state):
        s = self.mapping[state]
        return self.policy[s]

    def update(self, state, action, reward, observation):
        ()
