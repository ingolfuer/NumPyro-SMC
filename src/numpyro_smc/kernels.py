import jax
import jax.numpy as jnp
from jax import vmap
from functools import partial
from typing import Callable
from abc import ABC, abstractmethod
from collections import namedtuple
from numpyro.infer import Predictive
from numpyro.infer.util import transform_fn, constrain_fn as replay_constrain_fn

from .util import ravel_batched_pytree, value_and_jac, parse_model, SMCModelInfo


def _compute_mass_matrix_sqrt(raveled_particles):
    """Compute Cholesky factor of regularized empirical covariance."""
    cov = jnp.cov(raveled_particles, rowvar=False, bias=True)
    cov = jnp.atleast_2d(cov)
    cov = cov + 1e-6 * jnp.eye(cov.shape[0], dtype=jnp.float32)

    return jnp.linalg.cholesky(cov)


class SMCKernel(ABC):
    """
    Abstract base class for SMC particle mutation kernels.

    Defines the interface for kernels used in Sequential Monte Carlo inference.
    """

    @abstractmethod
    def init(self, rng_key, n_particles, jitted, model_args, model_kwargs):
        """
        Initialize particles from the prior and return initial state.

        Args:
            rng_key: JAX random key
            n_particles: Number of particles to initialize
            jitted: Whether this function is being jit compiled
            model_args: Positional arguments for the model
            model_kwargs: Keyword arguments for the model

        Returns:
            Initial SMCState (kernel-specific namedtuple).
        """
        raise NotImplementedError

    @abstractmethod
    def _mutate(self, state, model_info):
        """
        Mutate all particles using the kernel's transition operator.

        Args:
            state: Current SMCState
            model_args: Positional arguments for the model
            model_kwargs: Keyword arguments for the model

        Returns:
            New SMCState after mutation
        """
        raise NotImplementedError

    def _parse_model(self, model_args: tuple, model_kwargs: dict) -> SMCModelInfo:
        """
        Lets the kernel modify the default outputs of parse_model, e.g. applying chunked vmap or value_and_grad

        Args:
            model_args (tuple): Positional arguments for the model
            model_kwargs (dict): Keyword arguments for the model

        Returns:
            SMCModelInfo: Modified model info tuple
        """
        model_info = parse_model(self._model, model_args, model_kwargs)
        return model_info._replace(
            equivalence_mapping_fn=vmap(model_info.equivalence_mapping_fn),
            split_log_probs_fn=vmap(model_info.split_log_probs_fn),
        )

    def postprocess_fn(self, model_args: tuple, model_kwargs: dict):
        """
        Get function that transforms unconstrained particles to constrained space.

        Args:
            model_args: Positional arguments for the model
            model_kwargs: Keyword arguments for the model

        Returns:
            Function that constrains particles
        """
        return vmap(
            partial(
                replay_constrain_fn,
                self._model,
                model_args,
                model_kwargs,
                return_deterministic=True,
            )
        )

    def multi_step_mutate(
        self,
        state,
        beta: float,
        num_mcmc_steps: int,
        jitted: bool,
        model_args: tuple,
        model_kwargs: dict,
    ):
        """
        Apply multiple mutation steps. Default implementation calls mutate repeatedly.
        Kernels can overwrite for optimizations (e.g., reusing mass matrix).

        Args:
            state: Current SMCState
            beta: Temperature parameter
            num_mcmc_steps: Number of MCMC steps to perform
            jitted: Whether this function is being jit compiled
            model_args: Positional arguments for the model
            model_kwargs: Keyword arguments for the model

        Returns:
            New SMCState after num_mcmc_steps mutations
        """
        # fetch model info
        if jitted:
            model_info = self._parse_model(model_args, model_kwargs)
        else:
            model_info = self._model_info

        if self.adapt_step:
            # compute mass matrix
            raveled_particles, _ = ravel_batched_pytree(state.z)
            mass_matrix_sqrt = _compute_mass_matrix_sqrt(raveled_particles)

            # adapt step size
            intercept = jnp.log(
                state.step_size
            ) / self.adaption_gain + jax.scipy.special.logit(state.mean_accept_prob)
            step_size = jnp.exp(
                self.adaption_gain
                * (intercept - jax.scipy.special.logit(self.target_accept_prob))
            )
        else:
            mass_matrix_sqrt = state.mass_matrix_sqrt
            step_size = state.step_size

        # mutate and apply equivalence mappings
        def mutate_step(i, state):
            state_mutated = self._mutate(state, model_info)
            return state_mutated._replace(
                z=model_info.equivalence_mapping_fn(state_mutated.z)
            )

        return jax.lax.fori_loop(
            0,
            num_mcmc_steps,
            mutate_step,
            state._replace(
                i=jnp.array(0, dtype=jnp.int32),
                beta=beta,
                mass_matrix_sqrt=mass_matrix_sqrt,
                step_size=step_size,
                mean_accept_prob=jnp.array(0.0, dtype=jnp.float32),
            ),
        )

    @property
    def target_accept_prob(self):
        return self._target_accept_prob

    @property
    def adapt_step(self):
        return self._adapt_step

    @property
    def adaption_gain(self):
        return self._adaption_gain

    @property
    @abstractmethod
    def sample_field(self):
        """The field in the state that contains the particle samples."""
        raise NotImplementedError

    @property
    @abstractmethod
    def resample_fields(self):
        """The fields in the state that correspond to particles (e.g. log density)
        and, thus, need to be resampled alongside the particles. Should also include
        the sample field itself."""
        raise NotImplementedError


# State for Gaussian Random Walk Metropolis kernel
GaussianRandomWalkMetropolisSMCState = namedtuple(
    "GaussianRandomWalkMetropolisSMCState",
    [
        "i",
        "i_smc",
        "z",
        "beta",
        "split_log_probs",
        "accept_prob",
        "mean_accept_prob",
        "mass_matrix_sqrt",
        "step_size",
        "rng_key",
    ],
)
"""
State for GaussianRandomWalkMetropolis SMC kernel.

Fields:
    - i: Counts the total number of MCMC steps (over all SMC steps).
    - i_smc: Counts the number of SMC steps.
    - z: Dict of unconstrained particle values
    - beta: Current temperature parameter.
    - split_log_probs: Log likelihood values, Log prior densities, Log determinants of Jacobian transformations (n_particles,3)
    - accept_prob: Acceptance probability for current step (n_particles,)
    - mean_accept_prob: Running mean acceptance probability (scalar)
    - mass_matrix_sqrt: Cholesky factor of the proposal covariance matrix, of shape ``(dim, dim)``.
    - step_size: Scalar step size for the proposal.
    - rng_key: random number generator seed used for generating proposals, etc.
"""


class GaussianRandomWalkMetropolisSMC(SMCKernel):
    """
    SMC kernel using Gaussian random walk Metropolis-Hastings proposals.

    Proposes new particles by adding Gaussian noise in unconstrained space,
    then accepts/rejects using Metropolis-Hastings criterion.

    Args:
        model: NumPyro model function
        init_step_size: Initial step size for proposals.
        target_accept_prob: Target acceptance probability for step size adaption. Has no effect if ``adapt_step=False``.
        likelihood_chunk_size: Chunk size for batched likelihood computation.
        adapt_step: Whether to adapt the proposal step size and mass matrix. Mass matrix is updated from
            the empirical covariance of the current particle cloud. Step size is updated based on target
            acceptance probability and mean acceptance probability in the last SMC step. Update is done once
            per SMC step.

    """

    def __init__(
        self,
        model: Callable,
        init_step_size: float = 1.0,
        target_accept_prob: float = 0.3,
        likelihood_chunk_size: int = 1024,
        adapt_step: bool = True,
        adaption_gain: float = 0.89,
    ):
        self._model = model
        self._init_step_size = init_step_size
        self._target_accept_prob = target_accept_prob
        self._likelihood_chunk_size = likelihood_chunk_size
        self._adapt_step = adapt_step
        self._adaption_gain = adaption_gain

        # If not jitted, this will be set during init
        self._model_info = None

    @property
    def sample_field(self) -> str:
        return "z"

    @property
    def resample_fields(self) -> list[str]:
        return ["z", "split_log_probs", "accept_prob"]

    def _parse_model(self, model_args, model_kwargs):
        model_info = parse_model(self._model, model_args, model_kwargs)

        def split_log_probs_fn(samples_unconstrained):
            return jax.lax.map(
                model_info.split_log_probs_fn,
                xs=samples_unconstrained,
                batch_size=self._likelihood_chunk_size,
            )

        return model_info._replace(
            equivalence_mapping_fn=vmap(model_info.equivalence_mapping_fn),
            split_log_probs_fn=split_log_probs_fn,
        )

    def init(
        self, rng_key, n_particles, jitted, model_args, model_kwargs
    ) -> GaussianRandomWalkMetropolisSMCState:
        model_info = self._parse_model(model_args, model_kwargs)

        # if not jitted, cache functions
        if not jitted:
            self._model_info = model_info

        rng_key, init_key = jax.random.split(rng_key)
        # Sample from prior (constrained space)
        _particles_constrained = Predictive(
            self._model,
            num_samples=n_particles,
            return_sites=model_info.latent_sites,
            parallel=True,
        )(init_key, *model_args, **model_kwargs)

        # Convert to unconstrained space
        particles_unconstrained = vmap(
            partial(transform_fn, model_info.inv_transforms, invert=True)
        )(_particles_constrained)
        # Apply equivalence mappings
        particles_unconstrained = model_info.equivalence_mapping_fn(
            particles_unconstrained
        )

        # Compute initial density factors
        # split_log_prob_fn returns [log_likelihood, log_prior, log_det]
        split_log_probs = model_info.split_log_probs_fn(particles_unconstrained)

        # Initialize mass matrix as identity; will be ovewritten by mutate if adapt_step is True
        raveled_particles, _ = ravel_batched_pytree(particles_unconstrained)
        mass_matrix_sqrt = jnp.eye(raveled_particles.shape[1], dtype=jnp.float32)

        return GaussianRandomWalkMetropolisSMCState(
            i=jnp.array(0, dtype=jnp.int32),
            i_smc=jnp.array(0, dtype=jnp.int32),
            z=particles_unconstrained,
            beta=jnp.array(0.0, dtype=jnp.float32),
            split_log_probs=split_log_probs,
            accept_prob=jnp.zeros(n_particles, dtype=jnp.float32),
            mean_accept_prob=jnp.array(self.target_accept_prob, dtype=jnp.float32),
            mass_matrix_sqrt=mass_matrix_sqrt,
            step_size=jnp.array(self._init_step_size, dtype=jnp.float32),
            rng_key=rng_key,
        )

    def _mutate(
        self, state: GaussianRandomWalkMetropolisSMCState, model_info: SMCModelInfo
    ) -> GaussianRandomWalkMetropolisSMCState:
        """
        Mutate particles using Gaussian random walk Metropolis-Hastings.

        Args:
            state: Current GaussianRandomWalkMetropolisState
            model_info: Parsed model info containing the split log probs function.

        Returns:
            Updated GaussianRandomWalkMetropolisState
        """
        # unpack state
        step = state.i
        particles_unconstrained = state.z
        split_log_probs = state.split_log_probs
        rng_key = state.rng_key

        # Compute current potential energies
        potential_energies = -(
            split_log_probs[:, 2]
            + split_log_probs[:, 1]
            + state.beta * split_log_probs[:, 0]
        )

        # Split rng key
        rng_key, acceptance_key, mutation_key = jax.random.split(rng_key, 3)

        # Propose new particles via Gaussian random walk with mass matrix
        raveled_particles, unravel_fn = ravel_batched_pytree(particles_unconstrained)
        n_particles, dim = raveled_particles.shape

        # Sample noise and apply mass matrix
        noise = jax.random.normal(mutation_key, (n_particles, dim))
        raveled_prop = raveled_particles + state.step_size * (
            noise @ state.mass_matrix_sqrt.T
        )
        particles_unconstrained_prop = unravel_fn(raveled_prop)

        # Compute log densities for proposed particles
        # split_log_prob_fn returns [log_likelihood, log_prior, log_det]
        split_log_probs_prop = model_info.split_log_probs_fn(
            particles_unconstrained_prop
        )

        potential_energies_prop = -(
            split_log_probs_prop[:, 2]
            + split_log_probs_prop[:, 1]
            + state.beta * split_log_probs_prop[:, 0]
        )

        # Metropolis-Hastings accept/reject
        log_accept_ratio = potential_energies - potential_energies_prop
        accept_prob = jnp.minimum(
            jnp.array(1.0, dtype=jnp.float32), jnp.exp(log_accept_ratio)
        )

        accept = jax.random.bernoulli(acceptance_key, p=accept_prob)

        # Return new or old particles based on acceptance
        raveled_new = jnp.where(accept[:, None], raveled_prop, raveled_particles)
        particles_new = unravel_fn(raveled_new)

        split_log_probs_new = jnp.where(
            accept[:, None], split_log_probs_prop, split_log_probs
        )

        # Update mean acceptance probability
        mean_accept_prob = state.mean_accept_prob + (
            jnp.mean(accept_prob) - state.mean_accept_prob
        ) / (step + 1)

        return state._replace(
            i=step + 1,
            z=particles_new,
            split_log_probs=split_log_probs_new,
            accept_prob=accept_prob,
            mean_accept_prob=mean_accept_prob,
            rng_key=rng_key,
        )


# State for Langevin Metropolis kernel
LangevinMetropolisSMCState = namedtuple(
    "LangevinMetropolisSMCState",
    [
        "i",
        "i_smc",
        "z",
        "beta",
        "split_log_probs",
        "grad_split_log_probs",
        "accept_prob",
        "mean_accept_prob",
        "mass_matrix_sqrt",
        "step_size",
        "rng_key",
    ],
)
"""
State for LangevinMetropolis SMC kernel.

Fields:
    - i: Counts the total number of MCMC steps (over all SMC steps).
    - i_smc: Counts the number of SMC steps.
    - z: Dict of unconstrained particle values
    - beta: Current temperature parameter.
    - split_log_probs: Log likelihood values, log prior densities, log determinants of Jacobian transformations (n_particles,3)
    - grad_split_log_probs: Gradients of log likelihood values, log prior densities, log determinants w.r.t. unconstrained params
    - accept_prob: Acceptance probability for current step (n_particles,)
    - mean_accept_prob: Running mean acceptance probability (scalar)
    - mass_matrix_sqrt: Cholesky factor of the proposal covariance matrix, of shape ``(dim, dim)``.
    - step_size: Scalar step size for the proposal.
    - rng_key: random number generator seed used for generating proposals, etc.
"""


class LangevinMetropolisSMC(SMCKernel):
    """
    SMC kernel using Langevin (gradient-based) Metropolis-Hastings proposals.

    Proposes new particles using Langevin dynamics in unconstrained space,
    then accepts/rejects using Metropolis-Hastings criterion with proposal correction.

    Args:

        model: NumPyro model function
        init_step_size: Initial step size for proposals.
        target_accept_prob: Target acceptance probability for step size adaption. Has no effect if ``adapt_step=False``.
        likelihood_chunk_size: Chunk size for batched likelihood computation.
        adapt_step: Whether to adapt the proposal step size and mass matrix. Mass matrix is updated from
            the empirical covariance of the current particle cloud. Step size is updated based on target
            acceptance probability and mean acceptance probability in the last SMC step. Update is done once
            per SMC step.
    """

    def __init__(
        self,
        model: Callable,
        init_step_size: float = 1.0,
        target_accept_prob: float = 0.574,
        likelihood_chunk_size: int = 1024,
        adapt_step: bool = True,
        adaption_gain: float = 0.3,
    ):
        self._model = model
        self._init_step_size = init_step_size
        self._target_accept_prob = target_accept_prob
        self._likelihood_chunk_size = likelihood_chunk_size
        self._adapt_step = adapt_step
        self._adaption_gain = adaption_gain

        # If not jitted, these will be set during init
        self._model_info = None

    @property
    def sample_field(self) -> str:
        return "z"

    @property
    def resample_fields(self) -> list[str]:
        return ["z", "split_log_probs", "grad_split_log_probs", "accept_prob"]

    def _parse_model(self, model_args: tuple, model_kwargs: dict) -> SMCModelInfo:
        """split_log_probs_fn returns value and jacobian."""
        model_info = parse_model(self._model, model_args, model_kwargs)

        def split_log_probs_and_grad_fn(
            samples_unconstrained: dict,
        ) -> tuple[jax.Array, jax.Array]:
            return jax.lax.map(
                value_and_jac(model_info.split_log_probs_fn),
                xs=samples_unconstrained,
                batch_size=self._likelihood_chunk_size,
            )

        return model_info._replace(
            equivalence_mapping_fn=vmap(model_info.equivalence_mapping_fn),
            split_log_probs_fn=split_log_probs_and_grad_fn,
        )

    def init(
        self,
        rng_key,
        n_particles: int,
        jitted: bool,
        model_args: tuple,
        model_kwargs: dict,
    ) -> LangevinMetropolisSMCState:
        model_info = self._parse_model(model_args, model_kwargs)

        # if not jitted, cache functions
        if not jitted:
            self._model_info = model_info

        rng_key, init_key = jax.random.split(rng_key)

        # Sample from prior (constrained space)
        _particles_constrained = Predictive(
            self._model,
            num_samples=n_particles,
            return_sites=model_info.latent_sites,
            parallel=True,
        )(init_key, *model_args, **model_kwargs)

        # Convert to unconstrained space
        particles_unconstrained = vmap(
            partial(transform_fn, model_info.inv_transforms, invert=True)
        )(_particles_constrained)
        # Apply equivalence mappings
        particles_unconstrained = model_info.equivalence_mapping_fn(
            particles_unconstrained
        )

        # Compute initial density factors and gradients
        split_log_probs, grad_split_log_probs = model_info.split_log_probs_fn(
            particles_unconstrained
        )

        # Initialize mass matrix as identity; will be ovewritten by mutate if adapt_step is True
        raveled_particles, _ = ravel_batched_pytree(particles_unconstrained)
        mass_matrix_sqrt = jnp.eye(raveled_particles.shape[1], dtype=jnp.float32)

        return LangevinMetropolisSMCState(
            i=jnp.array(0, dtype=jnp.int32),
            i_smc=jnp.array(0, dtype=jnp.int32),
            z=particles_unconstrained,
            beta=jnp.array(0.0, dtype=jnp.float32),
            split_log_probs=split_log_probs,
            grad_split_log_probs=grad_split_log_probs,
            accept_prob=jnp.zeros(n_particles, dtype=jnp.float32),
            mean_accept_prob=jnp.array(self.target_accept_prob, dtype=jnp.float32),
            mass_matrix_sqrt=mass_matrix_sqrt,
            step_size=jnp.array(self._init_step_size, dtype=jnp.float32),
            rng_key=rng_key,
        )

    def _mutate(
        self,
        state: LangevinMetropolisSMCState,
        model_info: SMCModelInfo,
    ) -> LangevinMetropolisSMCState:
        """
        Mutate particles using Langevin Metropolis-Hastings.

        Args:
            state: Current LangevinMetropolisSMCState
            model_info: Parsed model info containing the split log probs function.

        Returns:
            Updated LangevinMetropolisSMCState
        """
        step = state.i
        particles_unconstrained = state.z
        split_log_probs = state.split_log_probs
        grad_split_log_probs = state.grad_split_log_probs
        rng_key = state.rng_key

        mass_matrix = state.mass_matrix_sqrt @ state.mass_matrix_sqrt.T
        step_size = state.step_size

        # Ravel particles and gradients
        raveled_particles, unravel_fn = ravel_batched_pytree(particles_unconstrained)
        raveled_grad_split_log_probs, _ = ravel_batched_pytree(
            grad_split_log_probs, batch_ndims=2
        )
        n_particles, dim = raveled_particles.shape

        # Compute current gradients using saved components and current beta
        # Possible since tempered likelihood gradient scales linearly with beta
        raveled_grads = (
            raveled_grad_split_log_probs[:, 2]
            + raveled_grad_split_log_probs[:, 1]
            + state.beta * raveled_grad_split_log_probs[:, 0]
        )

        # Compute current potential energies
        potential_energies = -(
            split_log_probs[:, 2]
            + split_log_probs[:, 1]
            + state.beta * split_log_probs[:, 0]
        )

        rng_key, acceptance_key, mutation_key = jax.random.split(rng_key, 3)

        # Propose new particles via Langevin dynamics with mass matrix
        mean_prop = raveled_particles + (step_size**2 / 2) * (
            raveled_grads @ mass_matrix
        )
        noise = (
            step_size
            * jax.random.normal(mutation_key, (n_particles, dim))
            @ state.mass_matrix_sqrt.T
        )
        raveled_prop = mean_prop + noise
        particles_unconstrained_prop = unravel_fn(raveled_prop)

        # Compute log densities and gradients for proposed particles
        split_log_probs_prop, grad_split_log_probs_prop = model_info.split_log_probs_fn(
            particles_unconstrained_prop
        )

        # Ravel proposed gradients
        raveled_grad_split_log_probs_prop, _ = ravel_batched_pytree(
            grad_split_log_probs_prop, batch_ndims=2
        )
        raveled_grads_prop = (
            raveled_grad_split_log_probs_prop[:, 2]
            + raveled_grad_split_log_probs_prop[:, 1]
            + state.beta * raveled_grad_split_log_probs_prop[:, 0]
        )

        potential_energies_prop = -(
            split_log_probs_prop[:, 2]
            + split_log_probs_prop[:, 1]
            + state.beta * split_log_probs_prop[:, 0]
        )

        # Asymmetric MH correction for the Langevin proposal
        # Forward:  q(x' | x)     = N(x'  | x   + (h²/2) M ∇log p_β(x),     h² M)
        # Reverse:  q(x  | x')    = N(x   | x'  + (h²/2) M ∇log p_β(x'),    h² M)
        proposal_log_prob = vmap(
            partial(
                jax.scipy.stats.multivariate_normal.logpdf,
                cov=step_size**2 * mass_matrix,
            )
        )
        # Forward proposal: mean = x + (step_size^2/2) * M * grad
        log_forward_prop_density = proposal_log_prob(raveled_prop, mean=mean_prop)

        # Reverse proposal: mean = x_prop + (step_size^2/2) * M * grad_prop
        reverse_mean = raveled_prop + (step_size**2 / 2) * (
            raveled_grads_prop @ mass_matrix
        )
        log_reverse_prop_density = proposal_log_prob(raveled_particles, reverse_mean)

        log_accept_ratio = (
            potential_energies
            - potential_energies_prop
            + log_reverse_prop_density
            - log_forward_prop_density
        )
        accept_prob = jnp.minimum(
            jnp.array(1.0, dtype=jnp.float32), jnp.exp(log_accept_ratio)
        )

        accept = jax.random.bernoulli(acceptance_key, p=accept_prob)

        # Return new or old particles based on acceptance
        raveled_new = jnp.where(accept[:, None], raveled_prop, raveled_particles)
        particles_new = unravel_fn(raveled_new)

        split_log_probs_new = jnp.where(
            accept[:, None], split_log_probs_prop, split_log_probs
        )

        # Return new or old gradients based on acceptance
        def select_accepted_for_dict(new_val, old_val, accept):
            """Select new or old value based on acceptance, broadcasting accept properly."""
            return jax.tree.map(
                lambda x, y: jnp.where(
                    accept.reshape(accept.shape + (1,) * (x.ndim - 1)), x, y
                ),
                new_val,
                old_val,
            )

        grad_split_log_probs_new = select_accepted_for_dict(
            grad_split_log_probs_prop, grad_split_log_probs, accept
        )

        # Update mean acceptance probability
        mean_accept_prob = state.mean_accept_prob + (
            jnp.mean(accept_prob) - state.mean_accept_prob
        ) / (step + 1)

        return state._replace(
            i=step + 1,
            z=particles_new,
            split_log_probs=split_log_probs_new,
            grad_split_log_probs=grad_split_log_probs_new,
            accept_prob=accept_prob,
            mean_accept_prob=mean_accept_prob,
            rng_key=rng_key,
        )
