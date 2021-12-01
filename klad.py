# %%
import random
import numpy as np
import matplotlib.pyplot as plt

# %%

pts = np.array([15,12,10,9,8,7,6,5,4,3,2,1])
players = np.arange(len(pts))

scores_agg = []

for i in range(100000):
    scores = np.zeros_like(players)

    for ii in range(4):
        np.random.shuffle(players)
        scores += pts[players]

    scores_agg.extend(scores)

# %%

plt.hist(scores_agg, bins=60)

# %%
from scipy.stats import norm, beta

mu, sigma = np.mean(scores_agg), np.std(scores_agg)
alpha = np.pi / 8

r = 12
n = 12

s = 30

mu + norm.ppf((r-alpha)/(n-2*alpha+1)) * sigma

(n-1) - norm.cdf((s - mu) / sigma) * (n-2*alpha+1) + alpha

# %%
sigma

# %%
r = 1
n = 12

m, v = beta.stats(r, n+1-r, moments='mv')

m * 60

# %%
np.sum(np.repeat(pts, 4))
# %%
