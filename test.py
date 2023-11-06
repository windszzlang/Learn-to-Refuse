import numpy as np



# doc = [1,1,0,0,0]
# lls = [1,1,0,0,0]
# # lls = [1,0,1,0,0]


# def mc2(lls):
#     # Split on the first `0` as everything before it is true (`1`).
#     split_idx = list(doc).index(0)
#     # Compute the normalized probability mass for the correct answer.
#     ll_true, ll_false = lls[:split_idx], lls[split_idx:]

#     print(np.exp(np.array(ll_true)))
#     print(np.exp(np.array(ll_false)))
#     print(ll_true, ll_false)

#     p_true, p_false = np.exp(np.array(ll_true)), np.exp(np.array(ll_false))
#     p_true = p_true / (sum(p_true) + sum(p_false))
#     return sum(p_true)

# print(mc2(lls))

p = [1,2,3]
g = [1,2,4]

print(set(p) - set(g))