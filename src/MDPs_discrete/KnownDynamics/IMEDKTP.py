import numpy as np
import scipy
from src.MDPs_discrete.utils import *
from src.MDPs_discrete.AgentInterface import Agent

# Official version from article "Logarithmic-regret-in-communicating-MDPs-Leveraging-known-dynamics-with-bandits"

def optimal_policy(p, r, max_iter, eps, gamma=None):
    ns, na, _ = p.shape
    ctr = 0
    stop = False
    phi0 = np.zeros(ns)
    phi1 = np.zeros(ns)
    policy = np.zeros(ns, dtype=int)
    # print(r, "r")

    while not stop:
        for state in range(ns):
            temporary = np.zeros(na)
            for action in range(na):
                if gamma is None:
                    temporary[action] = r[state, action] + sum(
                        [phi0[next_state] * p[state, action, next_state] for next_state in range(ns)])
                else:
                    temporary[action] = r[state, action] + gamma * sum(
                        [phi0[next_state] * p[state, action, next_state] for next_state in range(ns)])
            phi1[state] = np.max(temporary)
            policy[state] = np.argmax(temporary)

        delta = np.abs(phi1 - phi0)
        stop = (np.max(delta) - np.min(delta) < eps) or (ctr > max_iter)

        # print(phi1)
        if gamma is None:
            phi0 = np.copy(phi1 - np.min(phi1))
        else:
            phi0 = np.copy(phi1)
        phi1 = np.zeros(ns)
        ctr += 1
    # print("ctr", ctr, phi0)
    return policy


def gth_solve(p, overwrite=False):
    p1 = np.array(p, dtype=float, copy=not overwrite, order='C')

    if len(p1.shape) != 2 or p1.shape[0] != p1.shape[1]:
        raise ValueError('matrix must be square')

    n = p1.shape[0]
    x = np.zeros(n)

    # === Reduction === #
    for k in range(n - 1):
        scale = np.sum(p1[k, k + 1:n])
        if scale <= 0:
            # There is one (and only one) recurrent class contained in
            # {0, ..., k};
            # compute the solution associated with that recurrent class.
            n = k + 1
            break
        p1[k + 1:n, k] /= scale

        p1[k + 1:n, k + 1:n] += np.dot(p1[k + 1:n, k:k + 1], p1[k:k + 1, k + 1:n])

    # === Backward substitution === #
    x[n - 1] = 1
    for k in range(n - 2, -1, -1):
        x[k] = np.dot(x[k + 1:n], p1[k + 1:n, k])

    # === Normalization === #
    x /= np.sum(x)

    return x


def _csr_matrix_indices(p):
    m, n = p.shape

    for i in range(m):
        for j in range(p.indptr[i], p.indptr[i + 1]):
            row_index, col_index = i, p.indices[j]
            yield row_index, col_index


def compute_stationary(p):
    g = scipy.sparse.csr_matrix(np.ceil(p))
    nbr_scc, proj_scc = scipy.sparse.csgraph.connected_components(g, connection='strong', directed=True)

    if nbr_scc == 1:
        stationary_dists = np.zeros((nbr_scc, p.shape[0]))
        stationary_dists[0] = gth_solve(p)
    else:
        condensation_lil = scipy.sparse.lil_matrix((nbr_scc, nbr_scc), dtype=bool)
        for node_from, node_to in _csr_matrix_indices(g):
            scc_from, scc_to = proj_scc[node_from], proj_scc[node_to]
            if scc_from != scc_to:
                condensation_lil[scc_from, scc_to] = True
        sink_scc_labels = np.where(np.logical_not(condensation_lil.rows))[0]

        nbr_scc = len(sink_scc_labels)
        stationary_dists = np.zeros((nbr_scc, p.shape[0]))
        rec_classes = [np.where(proj_scc == k)[0] for k in range(nbr_scc)]
        for i, rec in enumerate(rec_classes):
            stationary_dists[i][rec] = gth_solve(p[np.ix_(rec, rec)])
    return stationary_dists


def gain_and_density(p, r):
    stationary_dists = compute_stationary(p)
    gains = (stationary_dists @ r.reshape(stationary_dists.shape[1], 1)).reshape(-1)

    return stationary_dists, gains


def analyze_policy(
        p,
        r,
        n,
        pi,
        threshold=None
):
    pp = []
    gp = []
    sp = []
    nbp = []
    tsp = []
    densities, gains = gain_and_density(p, r)
    for s, g in zip(densities, gains):
        pp.append(pi)
        gp.append(g)
        tsp.append(np.copy(s))
        if threshold is not None:
            s[s < threshold] = 0.
            s = s / s.sum()
        sp.append(s)
        nbp.append(np.min(n[s > 0.]))
    return pp, gp, sp, nbp, tsp


def neighbor_policies(
        transition,
        reward,
        nbr_pull,
        pi,
        threshold=None
):
    nbr_state, nbr_action, _ = transition.shape
    policy_pool = []
    gain_pool = []
    stationary_pool = []
    precision_pool = []
    true_stat_pool = []

    for state in range(nbr_state):
        for action in range(nbr_action):
            if action != pi[state]:
                policy = np.copy(pi)
                policy[state] = action
                p = transition[range(nbr_state), policy]
                r = reward[range(nbr_state), policy]
                n = nbr_pull[range(nbr_state), policy]

                pp, gp, sp, nbp, tsp = analyze_policy(p, r, n, policy, threshold)
                policy_pool.extend(pp)
                gain_pool.extend(gp)
                stationary_pool.extend(sp)
                precision_pool.extend(nbp)
                true_stat_pool.extend(tsp)

    return policy_pool, gain_pool, stationary_pool, precision_pool, true_stat_pool


def random_policies(
        transition,
        reward,
        nbr_pull,
        nbr_pi=10,
        threshold=None
):
    nbr_state, nbr_action, _ = transition.shape
    policy_pool = []
    gain_pool = []
    stationary_pool = []
    precision_pool = []
    true_stat_pool = []

    for _ in range(nbr_pi):
        policy = np.random.randint(0, nbr_action, nbr_state)
        p = transition[range(nbr_state), policy]
        r = reward[range(nbr_state), policy]
        n = nbr_pull[range(nbr_state), policy]

        pp, gp, sp, nbp, tsp = analyze_policy(p, r, n, policy, threshold)
        policy_pool.extend(pp)
        gain_pool.extend(gp)
        stationary_pool.extend(sp)
        precision_pool.extend(nbp)
        true_stat_pool.extend(tsp)

    return policy_pool, gain_pool, stationary_pool, precision_pool, true_stat_pool


class IMEDKTP(Agent):
    def __init__(
            self,
            env,
            name="IMED-KD",
            max_iter=12000,
            epsilon=1e-7,
            init_reward=0.,
            nbr_rn_pi=10,
            ada=False
    ):
        Agent.__init__(self, env.observation_space.n, env.action_space.n, name=name)
        self.nS = env.observation_space.n
        self.nA = env.action_space.n
        self.env = env
        self.transitions = np.zeros((self.nS, self.nA, self.nS))
        self.true_rewards = np.zeros((self.nS, self.nA))
        for s in range(self.nS):
            for a in range(self.nA):
                self.transitions[s, a] = env.getTransition(s, a)
                self.true_rewards[s, a] = env.getMeanReward(s, a)
        self.min_nbr_visit = 2
        self.max_iteration = max_iter
        self.epsilon = epsilon
        self.init_reward = init_reward
        self.nbr_rn_pi = nbr_rn_pi

        self.rewards = np.zeros((self.nS, self.nA)) + self.init_reward
        self.state_action_pulls = np.zeros((self.nS, self.nA), dtype=int)
        self.all_visited = False
        self.pi = np.zeros(self.nS, dtype=int)
        self.imed_pi = np.zeros(self.nS, dtype=int)
        self.empirical_optimal = None  # np.zeros(self.nS, dtype=int)
        self.new_episode = True
        self.playing_optimal = None
        self.target_state_reached = False
        self.states_to_visit = set()
        self.target_state = None
        self.policy_pool = None
        self.stationary_pool = None
        self.rvar = 0.25
        self.elapsed = 0

        self.ada = ada
        self.threshold = 1. / (2 * self.nS)
        #print("ada th")

        # debug
        self.t = 0
        self.g = 0
        self.new = 0
        self.opt = 0
        # print(self.transitions)
        # print(gain_and_density(
        #     self.transitions[range(self.nS), [0]*self.nS],
        #     self.true_rewards[range(self.nS), [0]*self.nS]))
        # print(self.transitions[range(self.nS), [0]*self.nS])

    def kl(self, a, b):
        return ((a - b) ** 2) / (2 * self.rvar)

    def update(self, state, action, reward, observation):
        # print(state, action, reward, self.state_action_pulls)
        self.elapsed += 1
        if self.ada:
            # print("here")
            self.threshold = min(self.rvar / np.sqrt(20*self.elapsed), 1./self.nS)  # 1. / (15*self.nS))
        else:
            self.threshold = 1. / (self.nS)
        na = self.state_action_pulls[state, action]
        r = self.rewards[state, action]

        self.state_action_pulls[state, action] = na + 1
        self.rewards[state, action] = ((na + 1) * r + reward) / (na + 2)

        self.t += 1
        # if self.t % 1 == 0:
        #     print(self.t, self.new, self.opt, self.opt / max(self.new, 1))  # self.state_action_pulls)
        #     print(self.rewards)

    def compute_indexes_and_info(self):
        policy_pool = []
        gain_pool = []
        stationary_pool = []
        precision_pool = []
        true_stat_pool = []

        policy = optimal_policy(self.transitions, self.rewards, self.max_iteration, self.epsilon)

        if self.empirical_optimal is not None:
            if np.all(policy == self.empirical_optimal):
                self.g += 1
                policy_pool = self.policy_pool.copy()
                gain_pool = []
                stationary_pool = []
                precision_pool = []
                true_stat_pool = [np.copy(s) for s in self.stationary_pool]

                for pi, s in zip(policy_pool, true_stat_pool):
                    r = self.rewards[range(self.nS), pi]
                    n = self.state_action_pulls[range(self.nS), pi]
                    s[s < self.threshold] = 0.
                    s = s / s.sum()
                    stationary_pool.append(s)
                    gain_pool.append(np.sum(s * r))
                    precision_pool.append(np.min(n[s > 0.]))

                pp, gp, sp, nbp, tsp = random_policies(self.transitions,
                                                       self.rewards,
                                                       self.state_action_pulls,
                                                       self.nbr_rn_pi,
                                                       self.threshold)
                policy_pool.extend(pp)
                gain_pool.extend(gp)
                stationary_pool.extend(sp)
                precision_pool.extend(nbp)

                gain_pool = np.array(gain_pool)
                precision_pool = np.array(precision_pool)
                delta = self.kl(gain_pool[0], gain_pool)
                imed = precision_pool * delta + np.log(precision_pool)
                delta = np.isclose(delta, 0)
                return imed, policy_pool, precision_pool, stationary_pool, delta

        p = self.transitions[range(self.nS), policy]
        r = self.rewards[range(self.nS), policy]
        n = self.state_action_pulls[range(self.nS), policy]
        pp, gp, sp, nbp, tsp = analyze_policy(p, r, n, policy, self.threshold)
        policy_pool.extend(pp)
        gain_pool.extend(gp)
        stationary_pool.extend(sp)
        precision_pool.extend(nbp)
        true_stat_pool.extend(tsp)

        pp, gp, sp, nbp, tsp = neighbor_policies(self.transitions,
                                                 self.rewards,
                                                 self.state_action_pulls,
                                                 policy,
                                                 self.threshold)
        policy_pool.extend(pp)
        gain_pool.extend(gp)
        stationary_pool.extend(sp)
        precision_pool.extend(nbp)
        true_stat_pool.extend(tsp)

        self.policy_pool = policy_pool.copy()
        self.stationary_pool = [np.copy(dis) for dis in true_stat_pool]

        pp, gp, sp, nbp, tsp = random_policies(self.transitions,
                                               self.rewards,
                                               self.state_action_pulls,
                                               self.nbr_rn_pi,
                                               self.threshold)
        policy_pool.extend(pp)
        gain_pool.extend(gp)
        stationary_pool.extend(sp)
        precision_pool.extend(nbp)
        true_stat_pool.extend(tsp)

        gain_pool = np.array(gain_pool)
        precision_pool = np.array(precision_pool)

        delta = self.kl(gain_pool[0], gain_pool)
        imed = precision_pool * delta + np.log(precision_pool)
        delta = np.isclose(delta, 0)
        return imed, policy_pool, precision_pool, stationary_pool, delta

    def play(self, state):
        if not self.all_visited:
            self.all_visited = np.all(self.state_action_pulls >= self.min_nbr_visit)
            idx_s, idx_a = np.unravel_index(self.state_action_pulls.argmin(), self.state_action_pulls.shape)
            r = np.zeros(self.state_action_pulls.shape)
            r[idx_s, idx_a] = 1.
            pi = optimal_policy(self.transitions, r, self.max_iteration, self.epsilon, gamma=0.99)
            self.pi = pi
            action = self.pi[state]
        else:
            # print(self.t, self.new_episode, self.playing_optimal)
            if self.new_episode:
                self.new += 1
                self.target_state_reached = False
                self.new_episode = False
                self.playing_optimal = False
                self.states_to_visit = set()

                info = self.compute_indexes_and_info()
                imed, policies, pulls, distributions, delta = info

                imed_idx = np.argmin(imed)
                policy = policies[imed_idx]
                # if imed_idx != 0:
                #     print(self.t, imed_idx)
                #     print(policies[0], policies[imed_idx])
                #     print(distributions[0], distributions[imed_idx])
                #     print(delta[0], delta[imed_idx])
                #     print(pulls[0], pulls[imed_idx])
                #     print(imed[0], imed[imed_idx])
                #     print()
                self.imed_pi = policy
                self.empirical_optimal = policies[0]
                n = pulls[imed_idx]
                rho = distributions[imed_idx]
                r = np.zeros(self.state_action_pulls.shape)
                if delta[imed_idx] or True:
                    self.opt += 1
                    # print(imed_idx, rho, self.empirical_optimal)
                    self.playing_optimal = True
                    self.states_to_visit = set(np.where(rho > 0)[0])
                    # print(self.states_to_visit)
                    r[np.argmax(rho)] = 1.
                    # for state in self.states_to_visit:
                    #     r[state] = 1.
                else:
                    to_pull = self.state_action_pulls[range(self.nS), policy] <= n + self.min_nbr_visit
                    self.states_to_visit = set(np.where((rho > 0) * to_pull)[0])
                    self.target_state = self.states_to_visit.pop()
                    r[self.target_state, policy[self.target_state]] = 1.

                pi = optimal_policy(self.transitions, r, self.max_iteration, self.epsilon)
                self.pi = pi
                action = self.pi[state]
            else:
                if self.playing_optimal:
                    if not self.target_state_reached:
                        self.target_state_reached = (state in self.states_to_visit)
                    if self.target_state_reached:
                        self.pi = self.empirical_optimal
                        if self.states_to_visit:
                            if state in self.states_to_visit:
                                self.states_to_visit.remove(state)
                        if not self.states_to_visit:
                            self.new_episode = True
                    action = self.pi[state]
                else:
                    action = self.pi[state]
                    self.target_state_reached = (state == self.target_state)
                    if self.target_state_reached:
                        if self.states_to_visit:
                            self.target_state = self.states_to_visit.pop()
                            r = np.zeros(self.state_action_pulls.shape)
                            r[self.target_state, self.imed_pi[self.target_state]] = 1.
                            pi = optimal_policy(self.transitions, r, self.max_iteration, self.epsilon)
                            self.pi = pi
                        else:
                            self.new_episode = True
        return action