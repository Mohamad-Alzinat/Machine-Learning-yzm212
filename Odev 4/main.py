import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
from utils import log_probability

true_mu = 150.0
true_sigma = 10.0
n_obs = 50

np.random.seed(42)
data = true_mu + true_sigma * np.random.randn(n_obs)

initial = [140, 5]
n_walkers = 32
pos = initial + 1e-4 * np.random.randn(n_walkers, 2)

sampler = emcee.EnsembleSampler(n_walkers, 2, log_probability, args=(data,))
sampler.run_mcmc(pos, 2000, progress=True)

flat_samples = sampler.get_chain(discard=500, thin=15, flat=True)

fig = corner.corner(flat_samples,
                    labels=["mu", "sigma"],
                    truths=[true_mu, true_sigma])

os.makedirs("results", exist_ok=True)
plt.savefig("results/corner_plot.png")
plt.show()

mu_median = np.median(flat_samples[:, 0])
sigma_median = np.median(flat_samples[:, 1])

with open("results/summary.txt", "w") as f:
    f.write(f"Mu Median: {mu_median}\n")
    f.write(f"Sigma Median: {sigma_median}\n")
