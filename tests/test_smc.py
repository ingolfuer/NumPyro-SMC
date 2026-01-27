# %%
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, ESS
import matplotlib.pyplot as plt
import arviz as az
import numpy as np

from numpyro_smc.smc import sample_SMC
from numpyro_smc.kernels import GaussianRandomWalkMetropolisSMC, LangevinMetropolisSMC


# %%
def test_smc():
    def test_model_1():
        x = numpyro.sample("x", dist.Uniform(low=-1, high=1))

    def test_model_2():
        x = numpyro.sample("x", dist.Normal(loc=0, scale=1))
        x_ = numpyro.deterministic("x_", 2 * x)
        y = numpyro.sample("y", dist.Normal(loc=x_, scale=0.01), obs=3)

    def test_model_3():
        x = numpyro.sample("x", dist.Dirichlet(concentration=jnp.ones(4)))

    def test_model_4():
        x = numpyro.sample("x", dist.Normal(loc=0, scale=0.1))

        numpyro.factor(
            "mixture",
            jax.scipy.special.logsumexp(
                jnp.array([-20 * (x - 5) ** 2, -20 * (x + 5) ** 2])
            ),
        )

    n_steps = 20
    rng_key = jax.random.key(6)

    for model in [test_model_1, test_model_2, test_model_3, test_model_4]:
        for kernel_type in [GaussianRandomWalkMetropolisSMC, LangevinMetropolisSMC]:
            kernel = kernel_type(model, step_size=1, likelihood_chunk_size=3200)
            samples, extra = sample_SMC(
                kernel,
                num_chains=4,
                num_particles=500,
                num_steps=n_steps,
                num_mcmc_steps=25,
                ess_threshold=0.5,
                rng_key=rng_key,
                chain_method="vectorized",
                progress_bar=True,
                extra_fields=("beta", "accept_prob"),
            )

            idata = az.from_dict(posterior=samples)
            print(idata.posterior.x.mean().values)
            az.plot_trace(idata)
            plt.tight_layout()
            plt.show()
            # print(extra["beta"])
            # plt.plot(np.log(extra["beta"]).T)
            plt.plot(extra["accept_prob"].mean(axis=-1).T)
            plt.ylim(0, 1)

        mcmc = MCMC(
            ESS(model),
            num_chains=16,
            num_samples=400,
            num_warmup=200,
            progress_bar=False,
            chain_method="vectorized",
        )
        mcmc.run(rng_key)
        idata = az.from_dict(posterior=mcmc.get_samples(group_by_chain=True))
        print(idata.posterior.x.mean().values)
        az.plot_trace(idata)
        plt.tight_layout()
        plt.show()
