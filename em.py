"""Mixture model for matrix completion"""

from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component.

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """

    n, d = X.shape
    K = mixture.mu.shape[0]

    log_post = np.zeros((n, K))

    for i in range(n):

        # Only use observed entries
        mask = X[i] != 0
        d_obs = np.sum(mask)

        for j in range(K):

            # Squared distance using observed entries only
            diff = X[i, mask] - mixture.mu[j, mask]
            sq_dist = np.sum(diff ** 2)

            # Log Gaussian density
            log_gaussian = (
                -0.5 * d_obs * np.log(2 * np.pi * mixture.var[j])
                - sq_dist / (2 * mixture.var[j])
            )

            # log(pi_j * N(...))
            log_post[i, j] = (
                np.log(mixture.p[j] + 1e-16)
                + log_gaussian
            )

    # log p(x_i) = log sum_j exp(log_post[i,j])
    log_norm = logsumexp(log_post, axis=1)

    # Total log-likelihood
    ll = np.sum(log_norm)

    # Normalize in log domain and convert back to probabilities
    post = np.exp(log_post - log_norm[:, None])

    return post, ll


def mstep(X: np.ndarray, post: np.ndarray,
          mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset.

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """

    n, d = X.shape
    K = post.shape[1]

    # Effective number of points assigned to each component
    n_hat = np.sum(post, axis=0)

    # Mixing proportions
    p = n_hat / n

    # Start from old means because some coordinates may not have
    # enough observed support to be updated
    mu = mixture.mu.copy()

    # Update means
    for j in range(K):

        for l in range(d):

            observed = X[:, l] != 0

            weight = np.sum(post[observed, j])

            if weight >= 1:
                mu[j, l] = (
                    np.sum(
                        post[observed, j] * X[observed, l]
                    )
                    / weight
                )

    # Update variances
    var = np.zeros(K)

    for j in range(K):

        numerator = 0.0
        denominator = 0.0

        for i in range(n):

            observed = X[i] != 0
            d_i = np.sum(observed)

            if d_i == 0:
                continue

            diff = X[i, observed] - mu[j, observed]
            squared_distance = np.sum(diff ** 2)

            numerator += post[i, j] * squared_distance
            denominator += post[i, j] * d_i

        if denominator > 0:
            var[j] = numerator / denominator
        else:
            var[j] = mixture.var[j]

        # Prevent variance from becoming too small
        var[j] = max(var[j], min_variance)

    return GaussianMixture(mu, var, p)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model using the EM algorithm.

    Args:
        X: (n, d) array holding the data, with incomplete entries set to 0
        mixture: the current gaussian mixture
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the updated gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """

    prev_ll = None
    ll = None

    while prev_ll is None or ll - prev_ll > 1e-6 * abs(ll):
        prev_ll = ll

        # E-step
        post, ll = estep(X, mixture)

        # M-step
        mixture = mstep(X, post, mixture)

    return mixture, post, ll


def fill_matrix(X: np.ndarray,
                mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model.

    Args:
        X: (n, d) array of incomplete data (incomplete entries = 0)
        mixture: a mixture of gaussians

    Returns:
        np.ndarray: a (n, d) array with completed data
    """

    X_pred = X.copy()

    mu, _, _ = mixture

    # Posterior cluster probabilities for each datapoint
    post, _ = estep(X, mixture)

    # Missing entries
    miss_indices = np.where(X == 0)

    # Expected value for each missing entry:
    # sum_j post[i,j] * mu[j,l]
    X_pred[miss_indices] = (post @ mu)[miss_indices]

    return X_pred