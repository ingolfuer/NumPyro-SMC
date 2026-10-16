# What is NumPyro SMC?
This is an implementation of Sequential Monte Carlo sampling for [NumPyro](https://github.com/pyro-ppl/numpyro) probabilistic models. It is based on the [PyMC implementation](https://www.pymc.io/projects/docs/en/latest/api/generated/pymc.smc.sample_smc.html) of SMC with Independent Metropolis-Hastings kernel, but with some changes and additions.

## Features
- Sequential Monte Carlo sampling with an automated tempering schedule based on effective sample size
- kernels:
    - Gaussian kernel
    - Langevin kernel
- proposal adaption:
    - based on particle covariance
    - based on acceptance probability
- equivalence mapping to account for symmetries in sampling space
- optimized for JIT compilation *through the entire sampling process* (this was used to do nested SMC, i.e. one complete SMC run for each outer SMC step)

## Changes compared to PyMC
- fixed number of MCMC mutation steps instead of autocorrelation threshold
- no equivalent to their [MH kernel](https://www.pymc.io/projects/docs/en/latest/api/generated/pymc.smc.kernels.MH.html#pymc.smc.kernels.MH)
- addition of Langevin kernel


# Notable deviations from NumPyro
- no equivalent to MCMC class, SMC sampling is only done by a function
- no implemention of sample function caching, only JAX's own caching available

# Known Issues
- site support depending on other site's value silently produces false results (akin to NumPyro issue #241)
- no support for 64-bit precision

# Note
We are not aware of any theoretical or systematic empirical analysis of
- combining covariance-based proposal adaption with acceptance-probability-based step size adjustment
- taking either of these measures from the previous SMC step and applying the adaption to the current one
- adapting Langeving step size in proportion to acceptance probability logit
These elements have simply been collected from different sources, adapted and combined in a way that seems to work reasonably well.

# References:
- J. Ching, Y. Chen: "Transitional Markov chain Monte Carlo method for Bayesian model updating, model class selection and model averaging" (SMC algorithm, covariance adaption in Gaussian kernel)
- Minson, Simons, Beck: "Bayesian inversion for finite fault earthquake source models I—theory and algorithm" (Effective-Sample-Size-based tempering schedule)
- T.L. Graves: "Automatic Step Size Selection in Random Walk Metropolis Algorithms" (Step size adaption based on acceptance probability)
- T. Xifara et al.: "Langevin diffusions and the Metropolis-adjusted Langevin algorithm" (Langevin kernel)
- Roberts, Gelman, Gilks: "Weak convergence and optimal scaling of random walk Metropolis algorithms" (Target acceptance probability Gaussian kernel)
- Pillai, Stuart, Thiéry: "Optimal scaling and diffusion limits for the Langevin algorithm in high dimensions" (Target acceptance probability Gaussian kernel Langevin kernel)