from functools import partial
import jax
from jax import random
import jax.numpy as jnp
from numpyro.util import fori_collect, nested_attrgetter, control_flow_prims_disabled
from numpyro.infer.mcmc import _laxmap
from typing import Literal
from time import perf_counter

from .kernels import SMCKernel


def effective_sample_size(log_weights):
    """Compute effective sample size from log weights."""
    normalized_weights = jnp.exp(log_weights - jax.scipy.special.logsumexp(log_weights))
    return 1.0 / jnp.sum(normalized_weights**2)


def _adaptive_tempering_step(log_likelihoods, beta_current, ess_threshold):
    """Find next beta that maintains ESS above threshold."""

    def ess_at_beta(beta_new):
        delta_beta = beta_new - beta_current
        log_weights = delta_beta * log_likelihoods
        return effective_sample_size(log_weights)

    num_samples = log_likelihoods.shape[0]

    # Binary search for beta that gives ESS approx ess_threshold * num_samples
    target_ess = num_samples * ess_threshold

    def binary_search_step(i, bounds):
        beta_min, beta_max = bounds
        beta_mid = (beta_min + beta_max) / 2
        ess_mid = ess_at_beta(beta_mid)

        # Update bounds based on ESS comparison
        beta_min = jnp.where(ess_mid > target_ess, beta_mid, beta_min)
        beta_max = jnp.where(ess_mid > target_ess, beta_max, beta_mid)

        return (beta_min, beta_max)

    # Run binary search for 20 iterations
    beta_min, beta_max = jax.lax.fori_loop(
        0, 20, binary_search_step, (beta_current, jnp.array(1.0, dtype=jnp.float32))
    )

    return beta_min


def _resample_particles(state, log_weights, resample_fields):
    """
    Resample particles and associated metadata using categorical sampling.

    Args:
        state: Current SMCState namedtuple.
        resample_fields: List of fields that represent per-particle data and need resampling.

    Returns:
        A new SMCState with particles and scores resampled.
    """
    num_particles = log_weights.shape[0]
    rng_key, resample_key = jax.random.split(state.rng_key)
    indices = jax.random.categorical(
        resample_key, logits=log_weights, shape=(num_particles,)
    )

    new_fields = {}
    for field in state._fields:
        value = getattr(state, field)
        if field in resample_fields:
            new_fields[field] = jax.tree.map(lambda x: x[indices], value)
        else:
            new_fields[field] = value
    new_fields["rng_key"] = rng_key

    return type(state)(**new_fields)


def _init_smc(
    kernel: SMCKernel,
    num_particles: int,
    jitted: bool,
    model_args: tuple,
    model_kwargs: dict,
    rng_key,
):
    """Initialize SMC state using kernel.init."""
    return kernel.init(rng_key, num_particles, jitted, model_args, model_kwargs)


_init_smc_vmapped = jax.vmap(_init_smc, in_axes=(None, None, None, None, None, 0))


def _smc_step(
    kernel: SMCKernel,
    num_mcmc_steps: int,
    ess_threshold: float,
    jitted: bool,
    model_args: tuple,
    model_kwargs: dict,
    state,
):
    """Single SMC step - can be scanned over."""

    def resample_and_mutate():
        log_likelihoods = state.split_log_probs[:, 0]

        # Adaptive tempering: find next beta
        beta_next = _adaptive_tempering_step(
            log_likelihoods=log_likelihoods,
            beta_current=state.beta,
            ess_threshold=ess_threshold,
        )
        beta_next = jnp.minimum(beta_next, 1.0)

        delta_beta = beta_next - state.beta

        # Compute weights
        log_weights = delta_beta * log_likelihoods
        log_weights = log_weights - jax.scipy.special.logsumexp(log_weights)

        # Resample
        resampled_state = _resample_particles(
            state=state,
            log_weights=log_weights,
            resample_fields=kernel.resample_fields,
        )

        # Mutate particles
        mutated_state = kernel.multi_step_mutate(
            resampled_state,
            beta_next,
            num_mcmc_steps=num_mcmc_steps,
            jitted=jitted,
            model_args=model_args,
            model_kwargs=model_kwargs,
        )

        return mutated_state

    def no_op():
        return state

    return jax.lax.cond(state.beta != 1.0, resample_and_mutate, no_op)


_smc_step_vmapped = jax.vmap(_smc_step, in_axes=(None, None, None, None, None, None, 0))


def _smc_loop(
    init_fn,
    step_fn,
    num_chains,
    num_steps,
    rng_key,
    progress_bar,
    extra_fields,
):
    # Initialize particles from prior
    init_state = init_fn(rng_key)

    # Prepare collection fields
    assert isinstance(extra_fields, (tuple, list))
    collect_fields = tuple(extra_fields) if extra_fields else ()

    # Transform function for fori_collect
    def extract_fn(state):
        if collect_fields:
            # Extract requested fields from state
            fields = nested_attrgetter(*collect_fields)(state)
            return fields if len(collect_fields) > 1 else (fields,)
        else:
            return ()

    # Use fori_collect to run the SMC loop with progress bar support
    collected, final_state = fori_collect(
        0,
        num_steps,
        step_fn,
        init_state,
        transform=extract_fn,
        progbar=progress_bar,
        return_last_val=True,
        progbar_desc=lambda i: f"SMC step {i+1}/{num_steps}",
        num_chains=num_chains,
    )

    # Package collected fields into a dict
    extra = dict(zip(collect_fields, collected))

    if extra_fields:
        return final_state, extra
    else:
        return final_state


def sample_SMC(
    kernel: SMCKernel,
    num_chains: int = 4,
    num_particles: int = 500,
    num_steps: int = 10,
    num_mcmc_steps: int = 1,
    ess_threshold: float = 0.5,
    model_args=(),
    model_kwargs=None,
    rng_key=jax.random.key(0),
    chain_method: Literal["sequential", "vectorized", "parallel"] = "vectorized",
    progress_bar=True,
    extra_fields=(),
    jit_sample_fn=True,
):
    """
    Estimate posterior using Sequential Monte Carlo with adaptive tempering.

    Args:
        kernel: SMCKernel instance for particle mutation
        num_chains: Number of independent chains to run
        num_particles: Number of particles (per chain)
        num_steps: Maximum number of SMC steps
        num_mcmc_steps: Number of MCMC steps per SMC step
        ess_threshold: ESS threshold as fraction of num_particles for adaptive tempering
        model_args: Tuple of model arguments (must be JAX-compatible)
        model_kwargs: Dict of model keyword arguments (must be JAX-compatible)
        rng_key: JAX random key
        chain_method: Method for running chains ('sequential', 'vectorized', or 'parallel')
        progress_bar: Whether to display progress bar
        extra_fields: Extra fields from the SMCState to collect at each iteration.
            Similar to MCMC.run, specify field names like "log_likelihoods", "accept_prob", "beta" etc.
        jit_sample_fn: Whether to JIT compile the sample function.
            Setting to False leads to very poor performance - use for debugging only.

    Returns:
        particles_constrained: Final particles in constrained space (shape: [num_chains, num_particles, ...])
        extra: Dict of collected extra_fields (if extra_fields is non-empty) (shape: [num_steps, num_chains, num_particles, ...])
    """
    if model_kwargs is None:
        model_kwargs = {}
    rng_keys = random.split(rng_key, num_chains)

    # Validate extra_fields
    assert isinstance(extra_fields, (tuple))

    if num_chains == 1:
        chain_method = "sequential"

    # Select base functions (use vmapped versions if vectorized)
    if chain_method == "vectorized":
        base_init = _init_smc_vmapped
        base_step = _smc_step_vmapped
    else:
        base_init = _init_smc
        base_step = _smc_step

    # Apply jit if requested (before partial to avoid cache misses)
    if jit_sample_fn:
        base_init = jax.jit(
            base_init, static_argnames=("kernel", "num_particles", "jitted")
        )
        base_step = jax.jit(
            base_step, static_argnames=("kernel", "num_mcmc_steps", "jitted")
        )

        # Pre-compile functions to measure performance
        if progress_bar:
            print("Compiling... ", end="")
            start = perf_counter()
        key = rng_keys if chain_method == "vectorized" else rng_key
        base_init.lower(
            kernel, num_particles, jit_sample_fn, model_args, model_kwargs, key
        ).compile()
        abstract_state = jax.eval_shape(
            base_init,
            kernel,
            num_particles,
            jit_sample_fn,
            model_args,
            model_kwargs,
            key,
        )
        base_step.lower(
            kernel,
            num_mcmc_steps,
            ess_threshold,
            jit_sample_fn,
            model_args,
            model_kwargs,
            abstract_state,
        ).compile()
        if progress_bar:
            print(f"done (took {perf_counter()-start}s)")

    # Create init_fn and step_fn with parameters bound via partial
    init_fn = partial(
        base_init, kernel, num_particles, jit_sample_fn, model_args, model_kwargs
    )
    step_fn = partial(
        base_step,
        kernel,
        num_mcmc_steps,
        ess_threshold,
        jit_sample_fn,
        model_args,
        model_kwargs,
    )

    # Create the main function to run
    partial_map_fn = partial(
        _smc_loop,
        init_fn,
        step_fn,
        num_chains if chain_method == "parallel" else 1,
        num_steps,
        progress_bar=progress_bar,
        extra_fields=extra_fields,
    )

    if chain_method == "sequential":
        ret = _laxmap(partial_map_fn, rng_keys)
    elif chain_method == "parallel":
        ret = jax.pmap(partial_map_fn)(rng_keys)
    else:  # vectorized
        assert chain_method == "vectorized"
        with control_flow_prims_disabled():  # don't do any jitting
            ret = partial_map_fn(rng_keys)

    postprocess_fn = jax.vmap(kernel.postprocess_fn(model_args, model_kwargs))
    if extra_fields:
        states, extra = ret
        particles_constrained = postprocess_fn(states.z)
        if chain_method != "vectorized":
            # Swap step and chain axes such that step is first
            extra = jax.tree.map(lambda x: x.swapaxes(0, 1), extra)
        return particles_constrained, extra
    else:
        states = ret
        particles_constrained = postprocess_fn(states.z)
        return particles_constrained
