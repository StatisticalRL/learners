from math import log, sqrt, exp
import numpy as np


## A function that returns an argmax at random in case of multiple maximizers

def randmax(A):
    maxValue = max(A)
    index = [i for i in range(len(A)) if A[i] == maxValue]
    return np.random.choice(index)


## A function that returns an argmin at random in case of multiple maximizers

def randmin(A):
    minValue = min(A)
    index = [i for i in range(len(A)) if A[i] == minValue]
    return np.random.choice(index)


## Kullback-Leibler divergence in exponential families

eps = 1e-15


def klBern(x, y):
    """Kullback-Leibler divergence for Bernoulli distributions."""
    x = min(max(x, eps), 1 - eps)
    y = min(max(y, eps), 1 - eps)
    return x * log(x / y) + (1 - x) * log((1 - x) / (1 - y))


def klGauss(x, y, sig2=1.):
    """Kullback-Leibler divergence for Gaussian distributions."""
    return (x - y) * (x - y) / (2 * sig2)


def klPoisson(x, y):
    """Kullback-Leibler divergence for Poison distributions."""
    x = max(x, eps)
    y = max(y, eps)
    return y - x + x * log(x / y)


def klExp(x, y):
    """Kullback-Leibler divergence for Exponential distributions."""
    x = max(x, eps)
    y = max(y, eps)
    return (x / y - 1 - log(x / y))


## computing the KL-UCB indices

def klucb(x, level, div, upperbound, lowerbound=-float('inf'), precision=1e-6):
    """Generic klUCB index computation using binary search:
    returns u>x such that div(x,u)=level where div is the KL divergence to be used.
    """
    l = max(x, lowerbound)
    u = upperbound
    while u - l > precision:
        m = (l + u) / 2
        if div(x, m) > level:
            u = m
        else:
            l = m
    return (l + u) / 2


def klucbBern(x, level, precision=1e-6):
    """returns u such that kl(x,u)=level for the Bernoulli kl-divergence."""
    upperbound = min(1., x + sqrt(level / 2))
    return klucb(x, level, klBern, upperbound, precision)


def klucbGauss(x, level, sig2=1., precision=0.):
    """returns u such that kl(x,u)=level for the Gaussian kl-divergence (can be done in closed form).
    """
    return x + sqrt(2 * sig2 * level)


def klucbPoisson(x, level, precision=1e-6):
    """returns u such that kl(x,u)=level for the Poisson kl-divergence."""
    upperbound = x + level + sqrt(level * level + 2 * x * level)
    return klucb(x, level, klPoisson, upperbound, precision)


def klucbExp(x, d, precision=1e-6):
    """returns u such that kl(x,u)=d for the exponential kl divergence."""
    if d < 0.77:
        upperbound = x / (
                    1 + 2. / 3 * d - sqrt(4. / 9 * d * d + 2 * d))  # safe, klexp(x,y) >= e^2/(2*(1-2e/3)) if x=y(1-e)
    else:
        upperbound = x * exp(d + 1)
    if d > 1.61:
        lowerbound = x * exp(d)
    else:
        lowerbound = x / (1 + d - sqrt(d * d + 2 * d))
    return klucb(x, d, klExp, upperbound, lowerbound, precision)


# Computing the complexity of a bandit instance
def complexity(bandit):
    """ computes the complexity of a Bernoulli or Gaussian unstructured bandit instance """
    meanMax = max(bandit.means)

    if bandit.distribution == 'Gaussian':
        return sum([(meanMax - bandit.means[a]) / klGauss(bandit.means[a], meanMax) for a in range(bandit.nbArms) if
                    a != bandit.bestarm])

    else:
        return sum([(meanMax - bandit.means[a]) / klBern(bandit.means[a], meanMax) for a in range(bandit.nbArms) if
                    a != bandit.bestarm])



from scipy.optimize import minimize_scalar, root_scalar
def KLinf_threshold(reward_history, mean_threshold,upper_bound=1.0, custom_optim=True):
    # Kinf is caculated via its concave dual problem
    #         max_{0<=lambda<=1/(B-mu^*)} E[log(1-(X-mu^*)*lambda)],
    #         where E is taken w.r.t the empirical measure hat{F}_k(t).
    X = np.array(reward_history)
    # Faster optimization: many times, the maximum of the concave
    # dual objective is attained on the boundary 0 or 1/(B-mu).
    # ~x2 speedup on some bandit instances.
    # If problem, fall back to standard minimize_scalar.
    fallback = False
    if custom_optim:
        def f(l):
            return np.mean(np.log(1 - (X - mean_threshold) * l))

        def jac(l):
            return -np.mean((X - mean_threshold) / (1 - (X - mean_threshold) * l))

        l_plus = 1e12 if mean_threshold == upper_bound else 1 / (upper_bound - mean_threshold)

        if jac(0) * jac(l_plus) >= 0:
            kinf = np.maximum(f(0), f(l_plus))
        else:
            ret = root_scalar(
                jac, method='brentq', bracket=[0, l_plus]
            )
            if ret.converged:
                kinf = np.max([f(ret.root), f(0), f(l_plus)])
            else:
                fallback = True
    if not custom_optim or fallback:
        # minimize -E[log(1-(X-mu^*)*lambda)]
        def f(l):
            return -np.mean(np.log(1 - (X - mean_threshold) * l))

        ret = minimize_scalar(
            f, method='bounded', bounds=(0, 1 / (upper_bound - mean_threshold))
        )
        if ret.success:
            kinf = -ret.fun
        else:
            # if error, just make this arm not eligible this turn
            kinf = np.inf
    return kinf

