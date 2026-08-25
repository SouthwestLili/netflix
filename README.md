# Gaussian Mixture Models and Collaborative Filtering

## Project Overview

This project implements clustering and matrix completion using K-Means and Gaussian Mixture Models (GMMs).

The project has two main goals:

1. Compare K-Means clustering with Gaussian Mixture Models trained using the Expectation-Maximization (EM) algorithm.
2. Apply a Gaussian mixture model to collaborative filtering and predict missing entries in an incomplete Netflix rating matrix.

The project contains five main experiments:

1. K-Means clustering on a 2D toy dataset
2. Gaussian Mixture Model clustering using EM
3. Model selection using the Bayesian Information Criterion (BIC)
4. Collaborative filtering on incomplete Netflix rating data
5. Matrix completion and evaluation using RMSE

For each value of $K$, multiple random initializations are tested to reduce the effect of initialization.

---

# Methods

## 1. K-Means Clustering

K-Means performs hard clustering. Each data point is assigned to the cluster whose mean is closest to that point.

The algorithm minimizes the distortion cost:

$$
J = \sum_i \|x_i - \mu_{z_i}\|^2
$$

where:

- $x_i$ is data point $i$
- $z_i$ is the cluster assigned to point $i$
- $\mu_{z_i}$ is the mean of that cluster

The experiment tests:

- $K = 1, 2, 3, 4$
- Seeds = 0, 1, 2, 3, 4

For each value of $K$, the model is initialized five times.

The initialization with the lowest distortion cost is selected.

---

## 2. Gaussian Mixture Model with EM

Unlike K-Means, a Gaussian Mixture Model performs soft clustering.

Instead of assigning each point to exactly one cluster, each data point has a probability of belonging to each Gaussian component.

Each Gaussian component contains three main parameters:

- Mean (`mu`)
- Variance (`var`)
- Mixing probability (`p`)

The model is trained using the Expectation-Maximization algorithm.

### E-Step

The E-step computes the posterior probability that each data point belongs to each Gaussian component.

For component $j$, the posterior probability represents:

$$
p(z_i = j \mid x_i)
$$

These posterior probabilities are also called soft assignments or responsibilities.

### M-Step

The M-step updates the Gaussian mixture parameters using the posterior probabilities obtained from the E-step.

The parameters updated are:

- Component means
- Component variances
- Mixing probabilities

The E-step and M-step are repeated until the log-likelihood converges.

The experiment tests:

- $K = 1, 2, 3, 4$
- Seeds = 0, 1, 2, 3, 4

For each $K$, the run with the highest log-likelihood is selected.

---

## 3. Model Selection with BIC

Increasing the number of mixture components generally makes the model more flexible and can improve the log-likelihood.

However, increasing $K$ also increases the number of model parameters.

Therefore, log-likelihood alone is not sufficient for choosing the number of Gaussian components.

The Bayesian Information Criterion (BIC) is used to balance model fit and model complexity.

The implementation uses:

$$
BIC = \log L - \frac{p}{2}\log n
$$

where:

- $L$ is the likelihood of the data
- $p$ is the number of free model parameters
- $n$ is the number of observations

For the spherical Gaussian mixture model used in this project, the number of free parameters is:

$$
p = K(d + 2) - 1
$$

where:

- $K$ is the number of Gaussian components
- $d$ is the dimensionality of the data

A higher BIC value is preferred.

---

## 4. EM for Incomplete Data

The standard EM implementation assumes that all entries in the dataset are observed.

For the Netflix collaborative filtering problem, some ratings are missing.

Missing entries are represented by `0`.

The EM algorithm is therefore modified so that missing values are ignored when computing Gaussian probabilities and updating model parameters.

For user $i$, define the set of observed entries as:

$$
O_i = \{l : X_{i,l} \neq 0\}
$$

During the E-step, only the observed ratings are used when calculating the probability that a user belongs to a Gaussian component.

During the M-step, each component mean is updated using only users for whom that particular rating is observed.

The component variances are also estimated using only observed entries.

A minimum variance of:

    0.25

is used to prevent Gaussian components from collapsing to extremely small variances.

---

## 5. Matrix Completion

After the Gaussian mixture model has been trained, it can be used to predict missing ratings.

For each user, the E-step calculates the posterior probability of belonging to each Gaussian component.

A missing rating is predicted using the posterior-weighted component means:

$$
\hat{x}_{i,l} = \sum_{j=1}^{K} p(z_i = j \mid x_i)\mu_{j,l}
$$

where:

- $\hat{x}_{i,l}$ is the predicted rating for user $i$ and item $l$
- $p(z_i=j \mid x_i)$ is the posterior probability that user $i$ belongs to component $j$
- $\mu_{j,l}$ is the mean rating of component $j$ for item $l$

Observed entries are kept unchanged.

Only entries represented by `0` in the incomplete matrix are replaced by model predictions.

---

# Results and Analysis

## 1. K-Means Results

The K-Means experiments produced the following results:

| K | Best Seed | Lowest Distortion Cost |
|---:|---:|---:|
| 1 | 0 | 5462.2975 |
| 2 | 0 | 1684.9080 |
| 3 | 3 | 1329.5949 |
| 4 | 4 | 1035.4998 |

The distortion cost decreases as $K$ increases.

The largest decrease occurs between $K=1$ and $K=2$:

    K = 1: 5462.2975
    K = 2: 1684.9080

This is expected because increasing $K$ provides more cluster centers and allows the model to represent the data more closely.

However, a lower K-Means distortion for a larger $K$ does not necessarily mean that the larger model is the most appropriate model. Increasing the number of clusters naturally gives K-Means more flexibility.

Different random seeds can also produce different solutions because K-Means is sensitive to the initial cluster centers.

For this reason, five random initializations are tested for each $K$, and the solution with the lowest distortion is selected.

---

## 2. Gaussian Mixture Model Results

The Gaussian Mixture Model was trained using EM for $K=1,2,3,4$.

Five different random initializations were tested for each value of $K$.

The best results are:

| K | Best Seed | Maximum Log-Likelihood |
|---:|---:|---:|
| 1 | 0 | -1307.2234 |
| 2 | 2 | -1175.7146 |
| 3 | 0 | -1138.8909 |
| 4 | 4 | -1138.6012 |

The log-likelihood improves as $K$ increases:

    K = 1: -1307.2234
    K = 2: -1175.7146
    K = 3: -1138.8909
    K = 4: -1138.6012

Since a higher log-likelihood is preferred, $K=4$ gives the highest likelihood among the tested models.

However, the improvement from $K=3$ to $K=4$ is very small.

The difference is approximately:

$$
-1138.6012 - (-1138.8909) = 0.2897
$$

This is much smaller than the improvements obtained when moving from $K=1$ to $K=2$ or from $K=2$ to $K=3$.

### K-Means vs. EM

K-Means and Gaussian Mixture Models use different approaches to clustering.

K-Means performs hard clustering:

    Each point → one cluster

EM with a Gaussian Mixture Model performs soft clustering:

    Each point → probability distribution over clusters

The Gaussian mixture model also estimates component means, variances, and mixing probabilities.

Therefore, it can represent uncertainty in cluster membership and differences in the statistical properties of the clusters.

---

## 3. BIC Model Selection

The Bayesian Information Criterion was used to determine whether the increase in log-likelihood from a larger $K$ justifies the additional model complexity.

The results are:

| K | Best Seed | Best Log-Likelihood | BIC |
|---:|---:|---:|---:|
| 1 | 0 | -1307.2234 | -1315.5056 |
| 2 | 2 | -1175.7146 | -1195.0397 |
| 3 | 0 | -1138.8909 | **-1169.2589** |
| 4 | 4 | -1138.6012 | -1180.0121 |

The highest BIC value is:

    BIC = -1169.2589
    K = 3

Therefore, BIC selects:

**K = 3**

This result is important because $K=4$ has a slightly higher log-likelihood than $K=3$:

    K = 3: -1138.8909
    K = 4: -1138.6012

However, the improvement is only approximately:

    0.2897

The improvement in likelihood is not large enough to compensate for the additional parameters introduced by the $K=4$ model.

Therefore, BIC prefers the simpler $K=3$ model.

This demonstrates the difference between maximizing model fit and selecting an appropriate model complexity.

---

## 4. Netflix Collaborative Filtering

The incomplete-data Gaussian mixture model was then applied to the Netflix rating dataset.

Two values of $K$ were compared:

- $K=1$
- $K=12$

Each model was tested using five random initializations.

### K = 1

All five random seeds converged to the same log-likelihood:

| Seed | Log-Likelihood |
|---:|---:|
| 0 | -1521060.9540 |
| 1 | -1521060.9540 |
| 2 | -1521060.9540 |
| 3 | -1521060.9540 |
| 4 | -1521060.9540 |

The selected result is:

    K = 1
    Best seed = 0
    Best log-likelihood = -1521060.9539852478

With only one Gaussian component, the model converges to the same solution for all five initializations.

### K = 12

The results for $K=12$ are:

| Seed | Log-Likelihood |
|---:|---:|
| 0 | -1399803.0467 |
| 1 | **-1390234.4223** |
| 2 | -1416862.4012 |
| 3 | -1393521.3930 |
| 4 | -1416733.8084 |

The best initialization is:

    K = 12
    Best seed = 1
    Best log-likelihood = -1390234.4223469393

The $K=12$ model achieves a substantially higher training log-likelihood than the $K=1$ model:

    K = 1:  -1521060.9540
    K = 12: -1390234.4223

This indicates that the more flexible $K=12$ mixture model fits the observed Netflix ratings substantially better than the single-component model.

Multiple Gaussian components allow the model to represent heterogeneous rating patterns that cannot be represented by a single Gaussian component.

The variation among the $K=12$ results also demonstrates that EM is sensitive to initialization and can converge to different local optima.

For example:

    seed = 1: -1390234.4223
    seed = 2: -1416862.4012

Therefore, testing multiple random initializations is important.

The solution with the highest log-likelihood, seed 1, is selected for matrix completion.

---

## 5. Matrix Completion and RMSE

The best $K=12$ model was used to predict the missing entries in the Netflix rating matrix.

The final model is:

    K = 12
    Best seed = 1
    Best log-likelihood = -1390234.4223469393

The missing entries are predicted using:

$$
\hat{x}_{i,l}=\sum_{j=1}^{K}p(z_i=j \mid x_i)\mu_{j,l}
$$

The completed matrix is then compared with the provided complete rating matrix.

The final prediction error is:

    RMSE = 0.4804908505400684

Therefore:

**RMSE ≈ 0.4805**

This means that the predicted ratings differ from the true ratings by approximately 0.48 rating points in terms of root mean squared error.

The RMSE provides a direct evaluation of the matrix completion predictions, while the log-likelihood measures how well the probabilistic mixture model fits the observed data.

These two metrics therefore evaluate different aspects of the model.

---

# Key Results

| Experiment | Result |
|---|---:|
| K-Means K=1 Cost | 5462.2975 |
| K-Means K=2 Cost | 1684.9080 |
| K-Means K=3 Cost | 1329.5949 |
| K-Means K=4 Cost | 1035.4998 |
| Best Toy EM Log-Likelihood | -1138.6012 |
| BIC-Selected Number of Components | **K = 3** |
| Netflix K=1 Log-Likelihood | -1521060.9540 |
| Netflix K=12 Best Seed | **1** |
| Netflix K=12 Best Log-Likelihood | **-1390234.4223** |
| Matrix Completion RMSE | **0.48049** |

---

# Project Structure

```text
.
├── main.py
├── common.py
├── kmeans.py
├── naive_em.py
├── em.py
├── test.py
├── toy_data.txt
├── netflix_incomplete.txt
├── netflix_complete.txt
├── test_incomplete.txt
└── test_complete.txt
```

## `common.py`

Contains shared utilities used throughout the project, including:

- `GaussianMixture`
- Mixture initialization
- Plotting
- RMSE calculation
- BIC calculation

## `kmeans.py`

Implements K-Means clustering.

The E-step performs hard assignment by assigning each point to the nearest cluster mean.

The M-step updates:

- Cluster means
- Variances
- Mixing proportions

The implementation also calculates the distortion cost.

## `naive_em.py`

Implements the standard Expectation-Maximization algorithm for complete data.

It performs soft clustering and estimates:

- Gaussian component means
- Variances
- Mixing probabilities
- Posterior cluster probabilities
- Log-likelihood

This implementation is used for the complete 2D toy dataset.

## `em.py`

Implements the modified EM algorithm for incomplete data.

The main difference from `naive_em.py` is that entries equal to `0` are treated as missing and are excluded from parameter estimation.

The file contains:

- `estep()`
- `mstep()`
- `run()`
- `fill_matrix()`

The `fill_matrix()` function predicts missing values using the posterior-weighted Gaussian component means.

## `main.py`

Runs the complete experiment pipeline:

1. K-Means clustering
2. Gaussian Mixture Model with EM
3. BIC model selection
4. Netflix collaborative filtering
5. Matrix completion
6. RMSE evaluation

## `test.py`

Tests the incomplete-data EM implementation on a small dataset before applying it to the full Netflix dataset.

The test checks:

- Initial mixture parameters
- First E-step
- First M-step
- Full EM convergence
- Posterior probabilities
- Matrix completion
- RMSE

---

# Requirements

The project requires Python 3 and the following packages:

```text
numpy
scipy
matplotlib
```

Install the dependencies using:

```bash
pip install numpy scipy matplotlib
```

---

# Running the Project

To run the complete experiment:

```bash
python main.py
```

This runs:

```text
K-Means
    ↓
Gaussian Mixture EM
    ↓
BIC Model Selection
    ↓
Netflix Collaborative Filtering
    ↓
Matrix Completion
    ↓
RMSE Evaluation
```

To run the smaller EM test:

```bash
python test.py
```

---

# Conclusion

This project demonstrates the differences between K-Means clustering and probabilistic clustering using Gaussian Mixture Models.

On the toy dataset, increasing $K$ decreases the K-Means distortion and increases the Gaussian mixture model log-likelihood. However, these improvements alone do not determine the most appropriate number of components.

Although $K=4$ achieves the highest log-likelihood among the tested Gaussian mixture models, the improvement over $K=3$ is small. After accounting for model complexity, BIC selects **K = 3**.

The project then extends the Gaussian mixture model to incomplete Netflix rating data. Missing ratings are excluded from the EM parameter updates, allowing the model to learn from only the observed entries.

For the Netflix experiment, the $K=12$ model achieves a substantially higher training log-likelihood than the $K=1$ model. Among the five $K=12$ initializations, seed 1 produces the highest log-likelihood:

    Best log-likelihood = -1390234.4223

This model is used to complete the missing entries in the rating matrix.

The final matrix completion result is:

**RMSE = 0.48049**

Overall, the project demonstrates how Gaussian mixture models can be extended from standard clustering to incomplete-data problems and used to make predictions through probabilistic matrix completion.