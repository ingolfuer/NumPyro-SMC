import jax
import jax.numpy as jnp
from numpyro.util import soft_vmap, identity
from numpyro.infer import Predictive
from numpyro.infer.util import get_transforms, transform_fn, initialize_model
from numpyro.handlers import seed
from functools import partial
from abc import ABC, abstractmethod
from collections import namedtuple
import operator

from .util import ravel_batched_pytree, compute_split_log_probs, value_and_jac


def _get_model_functions(model, model_args, model_kwargs):
    """
    Get transformation and density functions for the model. Transformation functions are vectorized, density functions are not.

    Args:
        model: NumPyro model function
        model_args: Positional arguments for the model
        model_kwargs: Keyword arguments for the model

    Returns:
        Tuple of (constrain_fn, unconstrain_fn, split_log_probs_fn)
    """
    if model_kwargs is None:
        model_kwargs = {}

    # get inverse transforms (constrained -> unconstrained)
    inv_transforms = get_transforms(
        seed(model, jax.random.key(0)), model_args, model_kwargs, {}
    )
    # Create forward transforms (unconstrained -> constrained)
    forward_transforms = {k: v.inv for k, v in inv_transforms.items()}

    # Create function to transform full dict
    constrain_fn = partial(transform_fn, inv_transforms)
    unconstrain_fn = partial(transform_fn, forward_transforms)

    def split_log_probs_fn(sample):
        log_probs_per_site = compute_split_log_probs(
            model,
            model_args=model_args,
            model_kwargs=model_kwargs,
            params_unconstrained=sample,
            model_inv_transforms=inv_transforms,
            sum_log_prob=True,
        )[0]
        # sum over sites
        return jax.tree.reduce(operator.add, log_probs_per_site)

    return (constrain_fn, unconstrain_fn, split_log_probs_fn)


def _compute_mass_matrix_sqrt(raveled_particles):
    """Compute Cholesky factor of regularized empirical covariance."""
    cov = jnp.cov(raveled_particles, rowvar=False, bias=True)
    cov = jnp.atleast_2d(cov)
    cov = cov + 1e-6 * jnp.eye(cov.shape[0])

    return jnp.linalg.cholesky(cov)


class SMCKernel(ABC):
    """
    Abstract base class for SMC particle mutation kernels.

    Defines the interface for kernels used in Sequential Monte Carlo inference.
    """

    @abstractmethod
    def init(self, rng_key, n_particles, model_args, model_kwargs):
        """
        Initialize particles from the prior and return initial state.

        Args:
            rng_key: JAX random key.
            n_particles: Number of particles to initialize.
            model_args: Positional arguments for the model.
            model_kwargs: Keyword arguments for the model.

        Returns:
            Initial SMCState (kernel-specific namedtuple).
        """
        raise NotImplementedError

    @abstractmethod
    def mutate(self, state, beta, model_args, model_kwargs):
        """
        Mutate all particles using the kernel's transition operator.

        Args:
            state: Current SMCState
            beta: Current temperature parameter
            model_args: Positional arguments for the model
            model_kwargs: Keyword arguments for the model

        Returns:
            New SMCState after mutation
        """
        raise NotImplementedError

    def postprocess_fn(self, model_args=(), model_kwargs=None):
        """
        Get function that transforms unconstrained particles to constrained space.

        Args:
            model_args: Positional arguments for the model
            model_kwargs: Keyword arguments for the model

        Returns:
            Function that constrains particles
        """
        if self._postprocess_fn is None:
            return identity
        return jax.vmap(self._postprocess_fn(*model_args, **model_kwargs))

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
        "z",
        "beta",
        "split_log_probs",
        "accept_prob",
        "mean_accept_prob",
        "mass_matrix_sqrt",
        "rng_key",
    ],
)
"""
State for GaussianRandomWalkMetropolis SMC kernel.

Fields:
    - i: iteration
    - z: Dict of unconstrained particle values
    - split_log_probs: Log likelihood values, Log prior densities, Log determinants of Jacobian transformations (n_particles,3)
    - accept_prob: Acceptance probability for current step (n_particles,)
    - mean_accept_prob: Running mean acceptance probability (scalar)
    - rng_key: random number generator seed used for generating proposals, etc.
"""


class GaussianRandomWalkMetropolisSMC(SMCKernel):
    """
    SMC kernel using Gaussian random walk Metropolis-Hastings proposals.

    Proposes new particles by adding Gaussian noise in unconstrained space,
    then accepts/rejects using Metropolis-Hastings criterion.

    Args:
        model: NumPyro model function
        step_size: Step size for proposals.
        likelihood_chunk_size: Chunk size for batched likelihood computation
        adapt_mass_matrix: Whether to adapt a mass matrix for proposal adaption.

    """

    def __init__(
        self,
        model,
        step_size: float = 1.0,
        likelihood_chunk_size: int = 1000,
        adapt_mass_matrix: bool = True,
    ):
        self._model = model
        self._step_size = step_size
        self._likelihood_chunk_size = likelihood_chunk_size
        self._adapt_mass_matrix = adapt_mass_matrix

        # These will be set during init
        self._constrain_fn = None
        self._unconstrain_fn = None
        self._split_log_probs_fn = None
        self._postprocess_fn = None

    @property
    def sample_field(self):
        return "z"

    @property
    def resample_fields(self):
        return ["z", "split_log_probs", "accept_prob"]

    def init(self, rng_key, n_particles, model_args, model_kwargs):
        """Initialize particles from the prior."""
        rng_key, init_key = jax.random.split(rng_key)

        # Get model functions
        self._constrain_fn, self._unconstrain_fn, split_log_probs_fn_ = (
            _get_model_functions(
                self._model, model_args=model_args, model_kwargs=model_kwargs
            )
        )

        def split_log_probs_fn(samples_unconstrained):
            return soft_vmap(
                split_log_probs_fn_,
                xs=samples_unconstrained,
                batch_ndims=1,
                chunk_size=self._likelihood_chunk_size,
            )

        self._split_log_probs_fn = split_log_probs_fn

        _, _, self._postprocess_fn, model_trace = initialize_model(
            jax.random.key(0),
            self._model,
            dynamic_args=True,
            model_args=model_args,
            model_kwargs=model_kwargs,
        )

        # Find unobserved (latent) sites
        latent_sites = [
            name
            for name, site in model_trace.items()
            if site["type"] == "sample" and not site["is_observed"]
        ]

        # Sample from prior (constrained space)
        _particles_constrained = Predictive(
            self._model,
            num_samples=n_particles,
            return_sites=latent_sites,
            parallel=True,
        )(init_key, *model_args, **model_kwargs)

        # Convert to unconstrained space
        particles_unconstrained = self._unconstrain_fn(_particles_constrained)

        # Compute initial density factors
        # split_log_prob_fn returns [log_likelihood, log_prior, log_det]
        split_log_probs = self._split_log_probs_fn(particles_unconstrained)

        # Initialize mass matrix
        raveled_particles, _ = ravel_batched_pytree(particles_unconstrained)
        mass_matrix_sqrt = jnp.eye(raveled_particles.shape[1])

        return GaussianRandomWalkMetropolisSMCState(
            i=0,
            z=particles_unconstrained,
            beta=0.0,
            split_log_probs=split_log_probs,
            accept_prob=jnp.zeros(n_particles),
            mean_accept_prob=0.0,
            mass_matrix_sqrt=mass_matrix_sqrt,
            rng_key=rng_key,
        )

    def mutate(self, state, beta, model_args, model_kwargs):
        """
        Mutate particles using Gaussian random walk Metropolis-Hastings.

        Args:
            state: Current GaussianRandomWalkMetropolisState
            beta: Temperature parameter
            model_args: Positional arguments for the model
            model_kwargs: Keyword arguments for the model

        Returns:
            New GaussianRandomWalkMetropolisState
        """
        step = state.i
        particles_unconstrained = state.z
        split_log_probs = state.split_log_probs
        rng_key = state.rng_key

        # Compute current potential energies
        potential_energies = -(
            split_log_probs[:, 2] + split_log_probs[:, 1] + beta * split_log_probs[:, 0]
        )

        # Split rng key
        rng_key, acceptance_key, mutation_key = jax.random.split(rng_key, 3)

        # Propose new particles via Gaussian random walk with mass matrix
        raveled_particles, unravel_fn = ravel_batched_pytree(particles_unconstrained)
        n_particles, dim = raveled_particles.shape

        if self._adapt_mass_matrix:
            mass_matrix_sqrt = _compute_mass_matrix_sqrt(raveled_particles)
        else:
            mass_matrix_sqrt = state.mass_matrix_sqrt

        # Sample noise and apply mass matrix
        noise = jax.random.normal(mutation_key, (n_particles, dim))
        raveled_prop = raveled_particles + self._step_size * (
            noise @ mass_matrix_sqrt.T
        )
        particles_unconstrained_prop = unravel_fn(raveled_prop)
        particles_constrained_prop = self._constrain_fn(particles_unconstrained_prop)

        # Compute log densities for proposed particles
        # split_log_prob_fn returns [log_likelihood, log_prior, log_det]
        split_log_probs_prop = self._split_log_probs_fn(particles_unconstrained_prop)

        potential_energies_prop = -(
            split_log_probs_prop[:, 2]
            + split_log_probs_prop[:, 1]
            + beta * split_log_probs_prop[:, 0]
        )

        # Metropolis-Hastings accept/reject
        log_accept_ratio = potential_energies - potential_energies_prop
        accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_ratio))

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

        return GaussianRandomWalkMetropolisSMCState(
            i=step + 1,
            z=particles_new,
            beta=beta,
            split_log_probs=split_log_probs_new,
            accept_prob=accept_prob,
            mean_accept_prob=mean_accept_prob,
            mass_matrix_sqrt=mass_matrix_sqrt,
            rng_key=rng_key,
        )


# State for Langevin Metropolis kernel
LangevinMetropolisSMCState = namedtuple(
    "LangevinMetropolisSMCState",
    [
        "i",
        "z",
        "beta",
        "split_log_probs",
        "grad_split_log_probs",
        "accept_prob",
        "mean_accept_prob",
        "mass_matrix_sqrt",
        "rng_key",
    ],
)
"""
State for LangevinMetropolis SMC kernel.

Fields:
    - i: iteration
    - z: Dict of unconstrained particle values
    - split_log_probs: Log determinants of Jacobian transformations (n_particles,)
    - grad_split_log_probs: Gradients of log determinants w.r.t. unconstrained params
    - accept_prob: Acceptance probability for current step (n_particles,)
    - mean_accept_prob: Running mean acceptance probability (scalar)
    - rng_key: random number generator seed used for generating proposals, etc.
"""


class LangevinMetropolisSMC(SMCKernel):
    """
    SMC kernel using Langevin (gradient-based) Metropolis-Hastings proposals.

    Proposes new particles using Langevin dynamics in unconstrained space,
    then accepts/rejects using Metropolis-Hastings criterion with proposal correction.

    Args:
        model: NumPyro model function
        step_size: Step size for proposals
        likelihood_chunk_size: Chunk size for batched likelihood computation
        adapt_mass_matrix: Whether to adapt a mass matrix for proposal adaption
    """

    def __init__(
        self,
        model,
        step_size: float = 1.0,
        likelihood_chunk_size: int = 1000,
        adapt_mass_matrix: bool = True,
    ):
        self._model = model
        self._step_size = step_size
        self._likelihood_chunk_size = likelihood_chunk_size
        self._adapt_mass_matrix = adapt_mass_matrix

        # These will be set during init
        self._constrain_fn = None
        self._unconstrain_fn = None
        self._split_log_prob_and_grad_fn = None
        self._postprocess_fn = None

    @property
    def sample_field(self):
        return "z"

    @property
    def resample_fields(self):
        return [
            "z",
            "split_log_probs",
            "grad_split_log_probs",
            "accept_prob",
        ]

    def init(self, rng_key, n_particles, model_args, model_kwargs):
        """Initialize particles from the prior."""
        rng_key, init_key = jax.random.split(rng_key)

        # Get model functions
        self._constrain_fn, self._unconstrain_fn, split_log_prob_fn_ = (
            _get_model_functions(
                self._model, model_args=model_args, model_kwargs=model_kwargs
            )
        )

        def split_log_prob_and_grad_fn(samples_unconstrained):
            return soft_vmap(
                value_and_jac(split_log_prob_fn_),
                xs=samples_unconstrained,
                batch_ndims=1,
                chunk_size=self._likelihood_chunk_size,
            )

        self._split_log_prob_and_grad_fn = split_log_prob_and_grad_fn

        _, _, self._postprocess_fn, model_trace = initialize_model(
            jax.random.key(0),
            self._model,
            dynamic_args=True,
            model_args=model_args,
            model_kwargs=model_kwargs,
        )

        # Find unobserved (latent) sites
        latent_sites = [
            name
            for name, site in model_trace.items()
            if site["type"] == "sample" and not site["is_observed"]
        ]

        # Sample from prior (constrained space)
        particles_constrained = Predictive(
            self._model,
            num_samples=n_particles,
            return_sites=latent_sites,
            parallel=True,
        )(init_key, *model_args, **model_kwargs)

        # Convert to unconstrained space
        particles_unconstrained = self._unconstrain_fn(particles_constrained)

        # Compute initial density factors and gradients
        # split_log_prob_and_grad_fn returns (log_probs, grad_log_probs)
        # where log_probs = [log_likelihood, log_prior, log_det]
        split_log_probs, grad_split_log_probs = self._split_log_prob_and_grad_fn(
            particles_unconstrained
        )

        # Initialize mass matrix
        raveled_particles, _ = ravel_batched_pytree(particles_unconstrained)
        mass_matrix_sqrt = jnp.eye(raveled_particles.shape[1])

        return LangevinMetropolisSMCState(
            i=0,
            z=particles_unconstrained,
            beta=0.0,
            split_log_probs=split_log_probs,
            grad_split_log_probs=grad_split_log_probs,
            accept_prob=jnp.zeros(n_particles),
            mean_accept_prob=0.0,
            mass_matrix_sqrt=mass_matrix_sqrt,
            rng_key=rng_key,
        )

    def mutate(self, state, beta, model_args, model_kwargs):
        """
        Mutate particles using Langevin Metropolis-Hastings with optional mass matrix.

        Args:
            state: Current LangevinMetropolisState
            beta: Temperature parameter
            model_args: Positional arguments for the model
            model_kwargs: Keyword arguments for the model

        Returns:
            New LangevinMetropolisState
        """
        step = state.i
        particles_unconstrained = state.z
        split_log_probs = state.split_log_probs
        grad_split_log_probs = state.grad_split_log_probs
        rng_key = state.rng_key

        step_size = self._step_size

        # Ravel particles and gradients
        raveled_particles, unravel_fn = ravel_batched_pytree(particles_unconstrained)
        raveled_grad_split_log_probs, _ = ravel_batched_pytree(
            grad_split_log_probs, batch_ndims=2
        )

        n_particles, dim = raveled_particles.shape

        # Compute mass matrix
        if self._adapt_mass_matrix:
            mass_matrix_sqrt = _compute_mass_matrix_sqrt(raveled_particles)
        else:
            mass_matrix_sqrt = state.mass_matrix_sqrt
        mass_matrix = mass_matrix_sqrt @ mass_matrix_sqrt.T

        # Compute current gradients using saved components and new beta
        raveled_grads = (
            raveled_grad_split_log_probs[:, 2]
            + raveled_grad_split_log_probs[:, 1]
            + beta * raveled_grad_split_log_probs[:, 0]
        )

        # Compute current potential energies
        potential_energies = -(
            split_log_probs[:, 2] + split_log_probs[:, 1] + beta * split_log_probs[:, 0]
        )

        # Split rng key
        rng_key, acceptance_key, mutation_key = jax.random.split(rng_key, 3)

        # Propose new particles via Langevin dynamics with mass matrix
        mean_prop = raveled_particles + (step_size**2 / 2) * (
            raveled_grads @ mass_matrix
        )
        noise = (
            step_size
            * jax.random.normal(mutation_key, (n_particles, dim))
            @ mass_matrix_sqrt.T
        )
        raveled_prop = mean_prop + noise
        particles_unconstrained_prop = unravel_fn(raveled_prop)

        # Compute log densities and gradients for proposed particles
        split_log_probs_prop, grad_split_log_probs_prop = (
            self._split_log_prob_and_grad_fn(particles_unconstrained_prop)
        )

        # Ravel proposed gradients
        raveled_grad_log_probs_prop, _ = ravel_batched_pytree(
            grad_split_log_probs_prop, batch_ndims=2
        )

        raveled_grads_prop = (
            raveled_grad_log_probs_prop[:, 2]
            + raveled_grad_log_probs_prop[:, 1]
            + beta * raveled_grad_log_probs_prop[:, 0]
        )

        potential_energies_prop = -(
            split_log_probs_prop[:, 2]
            + split_log_probs_prop[:, 1]
            + beta * split_log_probs_prop[:, 0]
        )

        # Compute proposal log probabilities (Gaussian with drift and mass matrix)
        # Forward: q(x_prop | x) = N(x_prop | x + (step_size^2/2)*M*grad, step_size^2*M)
        # Reverse: q(x | x_prop) = N(x | x_prop + (step_size^2/2)*M*grad_prop, step_size^2*M)
        proposal_log_prob = jax.vmap(
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

        # Metropolis-Hastings accept/reject with proposal correction
        log_accept_ratio = (
            potential_energies
            - potential_energies_prop
            + log_reverse_prop_density
            - log_forward_prop_density
        )
        accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_ratio))

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

        return LangevinMetropolisSMCState(
            i=step + 1,
            z=particles_new,
            beta=beta,
            split_log_probs=split_log_probs_new,
            grad_split_log_probs=grad_split_log_probs_new,
            accept_prob=accept_prob,
            mean_accept_prob=mean_accept_prob,
            mass_matrix_sqrt=mass_matrix_sqrt,
            rng_key=rng_key,
        )
