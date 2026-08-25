import numpy as np
import kmeans
import common
import naive_em
import em


##############################################################################
# 1. K-MEANS ON TOY DATA
##############################################################################

print("\n" + "=" * 60)
print("PART 1: K-MEANS")
print("=" * 60)

X = np.loadtxt("toy_data.txt")

for K in [1, 2, 3, 4]:

    best_cost = float("inf")
    best_mixture = None
    best_post = None
    best_seed = None

    for seed in [0, 1, 2, 3, 4]:

        mixture, post = common.init(X, K, seed)

        mixture, post, cost = kmeans.run(
            X, mixture, post
        )

        if cost < best_cost:
            best_cost = cost
            best_mixture = mixture
            best_post = post
            best_seed = seed

    print(f"\nK = {K}")
    print("Best seed =", best_seed)
    print("Lowest cost =", best_cost)

    common.plot(
        X,
        best_mixture,
        best_post,
        f"K-means K={K}"
    )


##############################################################################
# 2. EM / GAUSSIAN MIXTURE ON TOY DATA
##############################################################################

print("\n" + "=" * 60)
print("PART 2: EM / GAUSSIAN MIXTURE")
print("=" * 60)

X = np.loadtxt("toy_data.txt")

for K in [1, 2, 3, 4]:

    best_ll = -np.inf
    best_mixture = None
    best_post = None
    best_seed = None

    for seed in [0, 1, 2, 3, 4]:

        mixture, post = common.init(X, K, seed)

        mixture, post, ll = naive_em.run(
            X, mixture, post
        )

        print(
            f"K={K}, seed={seed}, "
            f"log-likelihood={ll}"
        )

        if ll > best_ll:
            best_ll = ll
            best_mixture = mixture
            best_post = post
            best_seed = seed

    print(f"\nK = {K}")
    print("Best seed =", best_seed)
    print("Maximum log-likelihood =", best_ll)

    common.plot(
        X,
        best_mixture,
        best_post,
        f"EM K={K}"
    )


##############################################################################
# 3. BIC MODEL SELECTION
##############################################################################

print("\n" + "=" * 60)
print("PART 3: BIC MODEL SELECTION")
print("=" * 60)

X = np.loadtxt("toy_data.txt")

best_bic = -np.inf
best_K = None

for K in [1, 2, 3, 4]:

    best_ll = -np.inf
    best_mixture = None
    best_seed = None

    for seed in [0, 1, 2, 3, 4]:

        mixture, post = common.init(X, K, seed)

        mixture, post, ll = naive_em.run(
            X, mixture, post
        )

        if ll > best_ll:
            best_ll = ll
            best_mixture = mixture
            best_seed = seed

    bic_value = common.bic(
        X,
        best_mixture,
        best_ll
    )

    print(f"\nK = {K}")
    print("Best seed =", best_seed)
    print("Best log-likelihood =", best_ll)
    print("BIC =", bic_value)

    if bic_value > best_bic:
        best_bic = bic_value
        best_K = K

print("\nBest K according to BIC =", best_K)
print("Best BIC =", best_bic)


##############################################################################
# 4. NETFLIX COLLABORATIVE FILTERING
##############################################################################

print("\n" + "=" * 60)
print("PART 4: NETFLIX COLLABORATIVE FILTERING")
print("=" * 60)

X = np.loadtxt("netflix_incomplete.txt")

for K in [1, 12]:

    best_ll = -np.inf
    best_seed = None
    best_mixture = None

    for seed in [0, 1, 2, 3, 4]:

        mixture, post = common.init(
            X, K, seed
        )

        mixture, post, ll = em.run(
            X, mixture, post
        )

        print(
            f"K={K}, seed={seed}, "
            f"log-likelihood={ll}"
        )

        if ll > best_ll:
            best_ll = ll
            best_seed = seed
            best_mixture = mixture

    print(f"\nK = {K}")
    print("Best seed =", best_seed)
    print("Best log-likelihood =", best_ll)


##############################################################################
# 5. MATRIX COMPLETION + RMSE
##############################################################################

print("\n" + "=" * 60)
print("PART 5: MATRIX COMPLETION AND RMSE")
print("=" * 60)

X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")

K = 12

best_ll = -np.inf
best_mixture = None
best_seed = None

for seed in [0, 1, 2, 3, 4]:

    mixture, post = common.init(
        X, K, seed
    )

    mixture, post, ll = em.run(
        X, mixture, post
    )

    print(
        f"seed={seed}, "
        f"log-likelihood={ll}"
    )

    if ll > best_ll:
        best_ll = ll
        best_mixture = mixture
        best_seed = seed


# Fill missing entries
X_pred = em.fill_matrix(
    X,
    best_mixture
)


# Compute RMSE
rmse_value = common.rmse(
    X_gold,
    X_pred
)


print("\nFinal Netflix Result")
print("----------------------")
print("K =", K)
print("Best seed =", best_seed)
print("Best log-likelihood =", best_ll)
print("RMSE =", rmse_value)