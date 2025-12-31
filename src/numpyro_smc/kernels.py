import jax
import jax.numpy as jnp
from numpyro.util import soft_vmap, identity
from numpyro.infer import Predictive
from numpyro.infer.util import get_transforms, transform_fn, initialize_model
from numpyro.handlers import seed, trace, substitute
from functools import partial
from abc import ABC, abstractmethod
from collections import namedtuple
import operator

from .util import ravel_batched_pytree


# TODO: handle intermediates and scale factors (see numpyro.infer.uitil.log_density)
def _get_model_functions(model, model_args, model_kwargs):
    """
    Get transformation and density functions for the model. Transformation functions are vectorized, density functions are not.

    Args:
        model: NumPyro model function
        model_args: Positional arguments for the model
        model_kwargs: Keyword arguments for the model

    Returns:
        Tuple of (constrain_fn, unconstrain_fn, log_det_fn, log_prior_fn,
                    log_likelihood_fn)
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

    # Define density functions that take unconstrained parameters
    def log_det_fn(unconstrained_sample):
        assert isinstance(unconstrained_sample, dict)
        constrained_sample = constrain_fn(unconstrained_sample)
        log_det = 0.0
        for k in unconstrained_sample.keys():
            log_det += inv_transforms[k].log_abs_det_jacobian(
                unconstrained_sample[k], constrained_sample[k]
            )
        return log_det

    def log_prior_fn(unconstrained_sample):
        assert isinstance(unconstrained_sample, dict)
        constrained_sample = constrain_fn(unconstrained_sample)
        substituted_model = substitute(model, data=constrained_sample)
        model_trace = trace(substituted_model).get_trace(*model_args, **model_kwargs)
        return jax.tree.reduce(
            operator.add,
            [
                site["fn"].log_prob(site["value"])
                for site in model_trace.values()
                if site["type"] == "sample" and not site["is_observed"]
            ],
            initializer=0.0,
        )

    def log_likelihood_fn(unconstrained_sample):
        assert isinstance(unconstrained_sample, dict)
        constrained_sample = constrain_fn(unconstrained_sample)
        substituted_model = substitute(model, data=constrained_sample)
        model_trace = trace(substituted_model).get_trace(*model_args, **model_kwargs)
        return jax.tree.reduce(
            operator.add,
            [
                site["fn"].log_prob(site["value"])
                for site in model_trace.values()
                if site["type"] == "sample" and site["is_observed"]
            ],
            initializer=0.0,
        )

    return (
        constrain_fn,
        unconstrain_fn,
        log_det_fn,
        log_prior_fn,
        log_likelihood_fn,
    )


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
        "log_dets",
        "log_priors",
        "log_likelihoods",
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
    - log_dets: Log determinants of Jacobian transformations (n_particles,)
    - log_priors: Log prior densities (n_particles,)
    - log_likelihoods: Log likelihood values (n_particles,)
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
        step_size,
        likelihood_chunk_size=1000,
        adapt_mass_matrix=True,
    ):
        self._model = model
        self._step_size = step_size
        self._likelihood_chunk_size = likelihood_chunk_size
        self._adapt_mass_matrix = adapt_mass_matrix

        # These will be set during init
        self._constrain_fn = None
        self._unconstrain_fn = None
        self._log_det_fn = None
        self._log_prior_fn = None
        self._log_likelihood_fn = None
        self._postprocess_fn = None

    @property
    def sample_field(self):
        return "z"

    @property
    def resample_fields(self):
        return ["z", "log_dets", "log_priors", "log_likelihoods", "accept_prob"]

    def get_model_functions(self, model_args, model_kwargs, chunk_size):
        """
        Get transformation and density functions for self._model.

        Args:
            model_args: Positional arguments for the model
            model_kwargs: Keyword arguments for the model
            chunk_size: Chunk size for likelihood computation

        Returns:
            Tuple of (constrain_fn, unconstrain_fn, log_det_fn, log_prior_fn,
                     log_likelihood_fn)
        """
        constrain_fn, unconstrain_fn, log_det_fn, log_prior_fn, log_likelihood_fn_ = (
            _get_model_functions(
                self._model, model_args=model_args, model_kwargs=model_kwargs
            )
        )

        def log_likelihood_fn(unconstrained_samples):
            return soft_vmap(
                log_likelihood_fn_,
                xs=unconstrained_samples,
                batch_ndims=1,
                chunk_size=chunk_size,
            )

        return (
            constrain_fn,
            unconstrain_fn,
            jax.vmap(log_det_fn),
            jax.vmap(log_prior_fn),
            log_likelihood_fn,
        )

    def init(self, rng_key, n_particles, model_args, model_kwargs):
        """Initialize particles from the prior."""
        rng_key, init_key = jax.random.split(rng_key)

        # Get model functions
        (
            self._constrain_fn,
            self._unconstrain_fn,
            self._log_det_fn,
            self._log_prior_fn,
            self._log_likelihood_fn,
        ) = self.get_model_functions(
            model_args=model_args,
            model_kwargs=model_kwargs,
            chunk_size=self._likelihood_chunk_size,
        )

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
        log_dets = self._log_det_fn(particles_unconstrained)
        log_priors = self._log_prior_fn(particles_unconstrained)
        log_likelihoods = self._log_likelihood_fn(particles_unconstrained)

        # Initialize mass matrix
        raveled_particles, _ = ravel_batched_pytree(particles_unconstrained)
        mass_matrix_sqrt = jnp.eye(raveled_particles.shape[1])

        return GaussianRandomWalkMetropolisSMCState(
            i=0,
            z=particles_unconstrained,
            beta=0.0,
            log_dets=log_dets,
            log_priors=log_priors,
            log_likelihoods=log_likelihoods,
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
            rng_key: JAX random key
            step: Current step number (for indexing step_sizes)

        Returns:
            New GaussianRandomWalkMetropolisState
        """
        step = state.i
        particles_unconstrained = state.z
        log_dets = state.log_dets
        log_priors = state.log_priors
        log_likelihoods = state.log_likelihoods
        rng_key = state.rng_key

        # Compute current potential energies
        potential_energies = -(log_dets + log_priors + beta * log_likelihoods)

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

        # Compute log Jacobian determinant, log prior and log likelihood for proposed particles
        log_det_prop = self._log_det_fn(particles_unconstrained_prop)
        log_prior_prop = self._log_prior_fn(particles_unconstrained_prop)
        log_likelihoods_prop = self._log_likelihood_fn(particles_unconstrained_prop)

        potential_energies_prop = -(
            log_det_prop + log_prior_prop + beta * log_likelihoods_prop
        )

        # Metropolis-Hastings accept/reject
        log_accept_ratio = potential_energies - potential_energies_prop
        accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_ratio))

        accept = jax.random.bernoulli(acceptance_key, p=accept_prob)

        # Return new or old particles based on acceptance
        raveled_new = jnp.where(accept[:, None], raveled_prop, raveled_particles)
        particles_new = unravel_fn(raveled_new)

        log_det_new = jnp.where(accept, log_det_prop, log_dets)
        log_prior_new = jnp.where(accept, log_prior_prop, log_priors)
        log_lik_new = jnp.where(accept, log_likelihoods_prop, log_likelihoods)

        # Update mean acceptance probability
        mean_accept_prob = state.mean_accept_prob + (
            jnp.mean(accept_prob) - state.mean_accept_prob
        ) / (step + 1)

        return GaussianRandomWalkMetropolisSMCState(
            i=step + 1,
            z=particles_new,
            beta=beta,
            log_dets=log_det_new,
            log_priors=log_prior_new,
            log_likelihoods=log_lik_new,
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
        "log_dets",
        "log_priors",
        "log_likelihoods",
        "grad_log_dets",
        "grad_log_priors",
        "grad_log_likelihoods",
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
    - log_dets: Log determinants of Jacobian transformations (n_particles,)
    - log_priors: Log prior densities (n_particles,)
    - log_likelihoods: Log likelihood values (n_particles,)
    - grad_log_dets: Gradients of log determinants w.r.t. unconstrained params
    - grad_log_priors: Gradients of log priors w.r.t. unconstrained params
    - grad_log_likelihoods: Gradients of log likelihoods w.r.t. unconstrained params
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
        step_size,
        likelihood_chunk_size=1000,
        adapt_mass_matrix=True,
    ):
        self._model = model
        self._step_size = step_size
        self._likelihood_chunk_size = likelihood_chunk_size
        self._adapt_mass_matrix = adapt_mass_matrix

        # These will be set during init
        self._constrain_fn = None
        self._unconstrain_fn = None
        self._log_det_and_grad_fn = None
        self._log_prior_and_grad_fn = None
        self._log_likelihood_and_grad_fn = None
        self._postprocess_fn = None

    @property
    def sample_field(self):
        return "z"

    @property
    def resample_fields(self):
        return [
            "z",
            "log_dets",
            "log_priors",
            "log_likelihoods",
            "grad_log_dets",
            "grad_log_priors",
            "grad_log_likelihoods",
            "accept_prob",
        ]

    def get_model_functions(self, model_args, model_kwargs, chunk_size):
        """
        Get transformation and density functions for self._model.
        Returns functions that compute values and gradients simultaneously.

        Args:
            model_args: Positional arguments for the model
            model_kwargs: Keyword arguments for the model
            chunk_size: Chunk size for likelihood computation

        Returns:
            Tuple of (constrain_fn, unconstrain_fn, log_det_and_grad_fn, log_prior_and_grad_fn,
                     log_likelihood_and_grad_fn)
        """
        constrain_fn, unconstrain_fn, log_det_fn, log_prior_fn, log_likelihood_fn_ = (
            _get_model_functions(
                self._model, model_args=model_args, model_kwargs=model_kwargs
            )
        )

        def log_likelihood_fn(unconstrained_samples):
            return soft_vmap(
                jax.value_and_grad(log_likelihood_fn_),
                xs=unconstrained_samples,
                batch_ndims=1,
                chunk_size=chunk_size,
            )

        return (
            constrain_fn,
            unconstrain_fn,
            jax.vmap(jax.value_and_grad(log_det_fn)),
            jax.vmap(jax.value_and_grad(log_prior_fn)),
            log_likelihood_fn,
        )

    def init(self, rng_key, n_particles, model_args, model_kwargs):
        """Initialize particles from the prior."""
        rng_key, init_key = jax.random.split(rng_key)

        # Get model functions
        (
            self._constrain_fn,
            self._unconstrain_fn,
            self._log_det_and_grad_fn,
            self._log_prior_and_grad_fn,
            self._log_likelihood_and_grad_fn,
        ) = self.get_model_functions(
            model_args=model_args,
            model_kwargs=model_kwargs,
            chunk_size=self._likelihood_chunk_size,
        )

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
        log_dets, grad_log_dets = self._log_det_and_grad_fn(particles_unconstrained)
        log_priors, grad_log_priors = self._log_prior_and_grad_fn(
            particles_unconstrained
        )
        log_likelihoods, grad_log_likelihoods = self._log_likelihood_and_grad_fn(
            particles_unconstrained
        )

        # Initialize mass matrix
        raveled_particles, _ = ravel_batched_pytree(particles_unconstrained)
        mass_matrix_sqrt = jnp.eye(raveled_particles.shape[1])

        return LangevinMetropolisSMCState(
            i=0,
            z=particles_unconstrained,
            beta=0.0,
            log_dets=log_dets,
            log_priors=log_priors,
            log_likelihoods=log_likelihoods,
            grad_log_dets=grad_log_dets,
            grad_log_priors=grad_log_priors,
            grad_log_likelihoods=grad_log_likelihoods,
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
        log_dets = state.log_dets
        log_priors = state.log_priors
        log_likelihoods = state.log_likelihoods
        grad_log_dets = state.grad_log_dets
        grad_log_priors = state.grad_log_priors
        grad_log_likelihoods = state.grad_log_likelihoods
        rng_key = state.rng_key

        step_size = self._step_size

        # Ravel particles and gradients
        raveled_particles, unravel_fn = ravel_batched_pytree(particles_unconstrained)
        raveled_grad_log_dets, _ = ravel_batched_pytree(grad_log_dets)
        raveled_grad_log_priors, _ = ravel_batched_pytree(grad_log_priors)
        raveled_grad_log_likelihoods, _ = ravel_batched_pytree(grad_log_likelihoods)

        n_particles, dim = raveled_particles.shape

        # Compute mass matrix
        if self._adapt_mass_matrix:
            mass_matrix_sqrt = _compute_mass_matrix_sqrt(raveled_particles)
        else:
            mass_matrix_sqrt = state.mass_matrix_sqrt
        mass_matrix = mass_matrix_sqrt @ mass_matrix_sqrt.T

        # Compute current gradients using saved components and new beta
        raveled_grads = (
            raveled_grad_log_dets
            + raveled_grad_log_priors
            + beta * raveled_grad_log_likelihoods
        )

        # Compute current potential energies
        potential_energies = -(log_dets + log_priors + beta * log_likelihoods)

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
        log_det_prop, grad_log_dets_prop = self._log_det_and_grad_fn(
            particles_unconstrained_prop
        )
        log_prior_prop, grad_log_priors_prop = self._log_prior_and_grad_fn(
            particles_unconstrained_prop
        )
        log_likelihoods_prop, grad_log_likelihoods_prop = (
            self._log_likelihood_and_grad_fn(particles_unconstrained_prop)
        )

        # Ravel proposed gradients
        raveled_grad_log_dets_prop, _ = ravel_batched_pytree(grad_log_dets_prop)
        raveled_grad_log_priors_prop, _ = ravel_batched_pytree(grad_log_priors_prop)
        raveled_grad_log_likelihoods_prop, _ = ravel_batched_pytree(
            grad_log_likelihoods_prop
        )

        raveled_grads_prop = (
            raveled_grad_log_dets_prop
            + raveled_grad_log_priors_prop
            + beta * raveled_grad_log_likelihoods_prop
        )

        potential_energies_prop = -(
            log_det_prop + log_prior_prop + beta * log_likelihoods_prop
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

        log_det_new = jnp.where(accept, log_det_prop, log_dets)
        log_prior_new = jnp.where(accept, log_prior_prop, log_priors)
        log_lik_new = jnp.where(accept, log_likelihoods_prop, log_likelihoods)

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

        grad_log_dets_new = select_accepted_for_dict(
            grad_log_dets_prop, grad_log_dets, accept
        )
        grad_log_priors_new = select_accepted_for_dict(
            grad_log_priors_prop, grad_log_priors, accept
        )
        grad_log_likelihoods_new = select_accepted_for_dict(
            grad_log_likelihoods_prop, grad_log_likelihoods, accept
        )

        # Update mean acceptance probability
        mean_accept_prob = state.mean_accept_prob + (
            jnp.mean(accept_prob) - state.mean_accept_prob
        ) / (step + 1)

        return LangevinMetropolisSMCState(
            i=step + 1,
            z=particles_new,
            beta=beta,
            log_dets=log_det_new,
            log_priors=log_prior_new,
            log_likelihoods=log_lik_new,
            grad_log_dets=grad_log_dets_new,
            grad_log_priors=grad_log_priors_new,
            grad_log_likelihoods=grad_log_likelihoods_new,
            accept_prob=accept_prob,
            mean_accept_prob=mean_accept_prob,
            mass_matrix_sqrt=mass_matrix_sqrt,
            rng_key=rng_key,
        )
