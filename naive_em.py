"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K = mixture.mu.shape[0]

    post = np.zeros((n, K))

    for j in range(K):
        diff = X - mixture.mu[j]
        squared_distance = np.sum(diff ** 2, axis=1)

        gaussian = (
            1.0 / ((2 * np.pi * mixture.var[j]) ** (d / 2))
            * np.exp(-squared_distance / (2 * mixture.var[j]))
        )

        post[:, j] = mixture.p[j] * gaussian

    # p(x_i) = sum_j pi_j N(x_i | mu_j, var_j I)
    total = np.sum(post, axis=1)

    # log likelihood
    ll = np.sum(np.log(total))

    # normalize to obtain posterior probabilities
    post = post / total[:, np.newaxis]

    return post, ll


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, d = X.shape
    K = post.shape[1]

    # Effective number of points assigned to each cluster
    n_hat = np.sum(post, axis=0)

    # Mixing proportions
    p = n_hat / n

    # Means
    mu = (post.T @ X) / n_hat[:, np.newaxis]

    # Variances
    var = np.zeros(K)

    for j in range(K):
        diff = X - mu[j]
        squared_distance = np.sum(diff ** 2, axis=1)

        var[j] = np.sum(post[:, j] * squared_distance) / (d * n_hat[j])

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_ll = -np.inf

    while True:
        # E-step
        post, ll = estep(X, mixture)

        # Check convergence
        if ll - old_ll <= 1e-6 * abs(ll):
            break

        old_ll = ll

        # M-step
        mixture = mstep(X, post)

    return mixture, post, ll
