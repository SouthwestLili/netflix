import numpy as np
import em
import common

print("Using em.py from:")
print(em.__file__)

###################################################
# For debugging: check which run() is being used
###################################################
# import inspect
# print("\nActual run() being used:")
# print(inspect.getsource(em.run))

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
seed = 0

mixture, post = common.init(X, K, seed)

print("Initial Mu:")
print(mixture.mu)

print("Initial Var:")
print(mixture.var)

print("Initial P:")
print(mixture.p)

# FIRST E-STEP
post1, ll1 = em.estep(X, mixture)

print("\nAfter first E-step:")
print(post1)
print("LL:", ll1)

# FIRST M-STEP
mixture1 = em.mstep(X, post1, mixture)

print("\nAfter first M-step:")
print("Mu:")
print(mixture1.mu)

print("Var:")
print(mixture1.var)

print("P:")
print(mixture1.p)

# COMPLETE RUN
mixture, post, ll = em.run(X, mixture, post)

print("\nAfter run")
print("Mu:")
print(mixture.mu)

print("Var:")
print(mixture.var)

print("P:")
print(mixture.p)

print("post:")
print(post)

print("LL:", ll)

# MATRIX COMPLETION
X_pred = em.fill_matrix(X, mixture)

print("\nX_gold:")
print(X_gold)

print("\nX_pred:")
print(X_pred)

print("\nRMSE:", common.rmse(X_gold, X_pred))