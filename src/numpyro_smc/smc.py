import warnings
from functools import partial
import jax
from jax import random
import jax.numpy as jnp
from numpyro.diagnostics import print_summary
from numpyro.util import (
    find_stack_level,
    fori_collect,
    is_prng_key,
    nested_attrgetter,
)
from numpyro.infer.mcmc import _laxmap
from typing import Literal

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
        0, 20, binary_search_step, (beta_current, 1.0)
    )

    return beta_min


def _resample_particles(state, log_weights, resample_fields, rng_key):
    """
    Resample particles and associated metadata using categorical sampling.

    Args:
        state: Current SMCState namedtuple.
        resample_fields: List of fields that represent per-particle data and need resampling.
        rng_key: JAX random key.

    Returns:
        A new SMCState with particles and scores resampled.
    """
    num_particles = log_weights.shape[0]
    indices = jax.random.categorical(
        rng_key, logits=log_weights, shape=(num_particles,)
    )

    new_fields = {}
    for field in state._fields:
        value = getattr(state, field)
        if field in resample_fields:
            new_fields[field] = jax.tree.map(lambda x: x[indices], value)
        else:
            new_fields[field] = value

    return type(state)(**new_fields)


def _smc_step(
    carry, num_mcmc_steps, model_args, model_kwargs, kernel: SMCKernel, ess_threshold
):
    """Single SMC step - can be scanned over."""
    state, rng_key = carry

    # Conditional resampling using jax.lax.cond
    def resample_and_mutate(args):
        state, rng_key = args
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
        rng_key, resample_key = jax.random.split(rng_key)
        state = _resample_particles(
            state=state,
            log_weights=log_weights,
            resample_fields=kernel.resample_fields,
            rng_key=resample_key,
        )

        # Mutate particles
        state = kernel.multi_step_mutate(
            state,
            beta_next,
            model_args=model_args,
            model_kwargs=model_kwargs,
            num_mcmc_steps=num_mcmc_steps,
        )

        return state, rng_key

    def no_op(args):
        return args

    state, next_rng_key = jax.lax.cond(
        state.beta != 1.0, resample_and_mutate, no_op, (state, rng_key)
    )

    return (state, next_rng_key)


def _single_chain_smc(
    kernel: SMCKernel,
    num_chains: int,
    num_particles: int,
    num_steps: int,
    num_mcmc_steps: int,
    ess_threshold: float,
    model_args: tuple,
    model_kwargs: dict,
    rng_key,
    progress_bar: bool,
    extra_fields: tuple,
):
    if model_kwargs is None:
        model_kwargs = {}

    # Initialize particles from prior
    rng_key, init_key = jax.random.split(rng_key)
    state = kernel.init(init_key, num_particles, model_args, model_kwargs)

    # Prepare collection fields
    assert isinstance(extra_fields, (tuple, list))
    collect_fields = tuple(extra_fields) if extra_fields else ()

    # Create the step function with fixed arguments
    step_fn = partial(
        _smc_step,
        num_mcmc_steps=num_mcmc_steps,
        model_args=model_args,
        model_kwargs=model_kwargs,
        kernel=kernel,
        ess_threshold=ess_threshold,
    )

    # Transform function for fori_collect
    def extract_fn(carry):
        state, rng_key = carry
        if collect_fields:
            # Extract requested fields from state
            fields = nested_attrgetter(*collect_fields)(state)
            return fields if len(collect_fields) > 1 else (fields,)
        else:
            return ()

    # Use fori_collect to run the SMC loop with progress bar support
    # TODO: only pass rng_key to kernel - require it to be part of the state
    init_carry = (state, rng_key)

    collected, final_carry = fori_collect(
        0,
        num_steps,
        step_fn,
        init_carry,
        transform=extract_fn,
        progbar=progress_bar,
        return_last_val=True,
        progbar_desc=lambda i: f"SMC step {i+1}/{num_steps}",
        num_chains=num_chains,
    )
    state, _ = final_carry

    # Package collected fields into a dict
    extra = dict(zip(collect_fields, collected))

    # Post-processing - transform to constrained space and add deterministic sites
    postprocess_fn = kernel.postprocess_fn(model_args, model_kwargs)
    particles_constrained = postprocess_fn(state.z)

    if extra_fields:
        return particles_constrained, extra
    else:
        return particles_constrained


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
):
    """
    Estimate posterior using Sequential Monte Carlo with adaptive tempering.

    Args:
        kernel: SMCKernel instance for particle mutation
        num_chains: Number of independent chains to run
        num_samples: Number of particles (per chain)
        num_steps: Maximum number of SMC steps
        ess_threshold: ESS threshold as fraction of n_samples for adaptive tempering
        model_args: Tuple of model arguments
        model_kwargs: Dict of model keyword arguments
        rng_key: JAX random key
        progress_bar: Whether to display progress bar. Only partially working.
        extra_fields: Extra fields from the SMCState to collect at each iteration.
            Similar to MCMC.run, specify field names like "log_likelihoods", "accept_prob", "beta" etc.

    Returns:
        particles_constrained: Final particles in constrained space
        extra: Dict of collected collected_extra_fields fields (if extra_fields is non-empty)
    """
    rng_key = random.split(rng_key, num_chains)

    # Validate extra_fields
    assert isinstance(extra_fields, (tuple))

    partial_map_fn = partial(
        _single_chain_smc,
        kernel,
        num_chains if chain_method == "parallel" else 1,
        num_particles,
        num_steps,
        num_mcmc_steps,
        ess_threshold,
        model_args,
        model_kwargs,
        progress_bar=progress_bar,
        extra_fields=extra_fields,
    )

    if num_chains == 1:
        chain_method = "sequential"
        
    if chain_method == "sequential":
        ret = _laxmap(partial_map_fn, rng_key)
    elif chain_method == "parallel":
        ret = jax.pmap(partial_map_fn)(rng_key)
    else:  # vectorized
        assert chain_method == "vectorized"
        ret = jax.jit(jax.vmap(partial_map_fn))(rng_key)

    return ret


class SMC:
    def __init__(
        self,
        kernel: SMCKernel,
        *,
        num_particles: int,
        num_steps: int,
        ess_threshold: float = 0.5,
        num_chains: int = 1,
        chain_method: Literal["sequential", "vectorized", "parallel"] = "vectorized",
        progress_bar: bool = True,
    ):
        raise NotImplementedError
        from jax import local_device_count

        self.kernel = kernel
        self.num_particles = num_particles
        self.num_steps = num_steps
        self.ess_threshold = ess_threshold
        self.num_chains = num_chains

        if chain_method not in ["parallel", "vectorized", "sequential"]:
            raise ValueError(
                "Only supporting the following methods to draw chains: "
                '"sequential", "parallel", or "vectorized"'
            )

        if chain_method == "parallel" and local_device_count() < self.num_chains:
            chain_method = "sequential"
            warnings.warn(
                "There are not enough devices to run parallel chains: expected {} but got {}."
                " Chains will be drawn sequentially. If you are running SMC in CPU,"
                " consider using `numpyro.set_host_device_count({})` at the beginning"
                " of your program. You can double-check how many devices are available in"
                " your system using `jax.local_device_count()`.".format(
                    self.num_chains, local_device_count(), self.num_chains
                ),
                stacklevel=find_stack_level(),
            )

        self.chain_method = chain_method
        self.progress_bar = progress_bar

        # Internal state
        self._extra_fields_dict = None  # Extra fields collected at each step
        self._extra_fields_dict_flat = None  # Flattened version
        self._last_state = None  # Final SMCState from run
        self._args = None
        self._kwargs = None

    def _single_chain_smc(self, rng_key, args, kwargs, collect_fields):
        """Run SMC for a single chain."""
        # Initialize particles from prior
        rng_key, init_key = random.split(rng_key)
        init_state = self.kernel.init(init_key, self.num_particles, args, kwargs)

        # Create step function with fixed arguments
        step_fn = partial(
            _smc_step,
            model_args=args,
            model_kwargs=kwargs,
            kernel=self.kernel,
            ess_threshold=self.ess_threshold,
        )

        # Function extracting relevant information from carry of fori_collect
        def extract_fn(carry):
            state, rng_key = carry
            if collect_fields:
                # Extract requested fields from state
                fields = nested_attrgetter(*collect_fields)(state)
                return fields if len(collect_fields) > 1 else (fields,)
            else:
                return ()

        # Use fori_collect to run the SMC loop
        init_carry = (init_state, rng_key)

        collected, final_carry = fori_collect(
            0,
            self.num_steps,
            step_fn,
            init_carry,
            transform=extract_fn,
            progbar=self.progress_bar,
            return_last_val=True,
            progbar_desc=lambda i: f"SMC step {i+1}/{self.num_steps}",
            num_chains=self.num_chains if self.chain_method == "parallel" else 1,
        )

        state, _ = final_carry

        # Package collected fields
        if collect_fields:
            extra_fields_dict = dict(zip(collect_fields, collected))
        else:
            extra_fields_dict = {}

        return extra_fields_dict, state

    def run(self, rng_key, *args, extra_fields=(), **kwargs):
        """
        Run the SMC sampler and collect samples.

        :param random.PRNGKey rng_key: Random number generator key to be used for sampling.
            For multi-chains, a batch of `num_chains` keys can be supplied. If `rng_key`
            does not have batch_size, it will be split into a batch of `num_chains` keys.
        :param args: Arguments to be provided to the model.
        :param extra_fields: Extra fields from the SMCState to collect at each SMC step.
            Can include fields like "beta", "log_weights", "log_likelihoods", "accept_prob", etc.
            Note that subfields can be accessed using dots, e.g. "accept_prob" from the state.
        :type extra_fields: tuple or list of str
        :param kwargs: Keyword arguments to be provided to the model.
        """
        self._args = args
        self._kwargs = kwargs

        # Split rng_key for multiple chains if needed
        if self.num_chains > 1 and is_prng_key(rng_key):
            rng_key = random.split(rng_key, self.num_chains)

        # Validate extra_fields
        assert isinstance(extra_fields, (tuple, list))
        collect_fields = tuple(extra_fields) if extra_fields else ()

        # Prepare mapping function
        partial_map_fn = partial(
            self._single_chain_smc,
            args=args,
            kwargs=kwargs,
            collect_fields=collect_fields,
        )

        # Execute based on chain method
        if self.num_chains == 1:
            collected_extra_fields_flat, last_state = partial_map_fn(rng_key)
            # Add chain dimension
            last_state = jax.tree.map(lambda x: x[jnp.newaxis, ...], last_state)
            collected_extra_fields = jax.tree.map(
                lambda x: x[jnp.newaxis, ...], collected_extra_fields_flat
            )
        else:
            if self.chain_method == "sequential":
                collected_extra_fields, last_state = _laxmap(partial_map_fn, rng_key)
            elif self.chain_method == "parallel":
                collected_extra_fields, last_state = jax.pmap(partial_map_fn)(rng_key)
            else:  # vectorized
                collected_extra_fields, last_state = jax.vmap(partial_map_fn)(rng_key)
                # For vectorized, states are already batched over chains

        self._last_state = last_state
        self._extra_fields_dict = collected_extra_fields
        self._extra_fields_dict_flat = None

    def _get_extra_fields_flat(self):
        """Get flattened states across chains."""
        if self._extra_fields_dict_flat is None:
            self._extra_fields_dict_flat = jax.tree.map(
                lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:]),
                self._extra_fields_dict,
            )
        return self._extra_fields_dict_flat

    def get_samples(self, group_by_chain=False):
        """
        Get particle samples from the final SMC step.

        :param bool group_by_chain: Whether to preserve the chain dimension. If True,
            samples will have shape [num_chains, num_particles, ...]. If False,
            samples will have shape [num_chains * num_particles, ...].
        :return: Dictionary of samples keyed by site names, containing particles
            from the final SMC step in constrained space.
        """
        if self._last_state is None:
            raise RuntimeError("SMC has not been run yet. Call run() first.")

        # Post-process particles to constrained space
        postprocess_fn = self.kernel.postprocess_fn(self._args, self._kwargs)

        # TODO: make it already vmapped by kernel
        samples = jax.vmap(postprocess_fn)(self._last_state.z)

        if not group_by_chain:
            # Flatten chain and particle dimensions
            samples = jax.tree.map(
                lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1],) + x.shape[2:]),
                samples,
            )

        return samples

    def get_extra_fields(self, group_by_chain=False):
        """
        Get extra fields collected during the SMC run.

        :param bool group_by_chain: Whether to preserve the chain dimension. If True,
            fields will have shape [num_chains, num_steps, ...]. If False,
            fields will have shape [num_chains * num_steps, ...].
        :return: Dictionary of extra fields keyed by field names which were specified
            in the `extra_fields` keyword of :meth:`run`.
        """
        if self._extra_fields_dict is None:
            raise RuntimeError("SMC has not been run yet. Call run() first.")

        return (
            self._extra_fields_dict if group_by_chain else self._get_extra_fields_flat()
        )

    def print_summary(self, prob=0.9, exclude_deterministic=True):
        """
        Print the statistics of particle samples from the final SMC step.

        :param float prob: The probability mass of samples within the credible interval.
        :param bool exclude_deterministic: Whether to exclude deterministic sites
            from the printed summary.
        """
        samples = self.get_samples(group_by_chain=False)

        if isinstance(samples, dict) and exclude_deterministic:
            # Filter out deterministic sites if needed
            # (This logic might need adjustment based on how deterministic sites are marked)
            pass

        print_summary(samples, prob=prob)

        # Print SMC-specific diagnostics
        if self._last_state is not None:
            final_beta = self._last_state.beta
            if self.num_chains == 1:
                print(f"\nFinal beta: {final_beta}")
            else:
                if self.chain_method == "vectorized" or isinstance(
                    final_beta, jnp.ndarray
                ):
                    if final_beta.ndim > 0:
                        print(f"\nFinal beta (per chain): {final_beta}")
                    else:
                        print(f"\nFinal beta: {final_beta}")
                else:
                    print(f"\nFinal beta: {final_beta}")

            # Print ESS if available
            if hasattr(self._last_state, "log_weights"):
                log_weights = self._last_state.log_weights
                if self.num_chains == 1:
                    ess = effective_sample_size(log_weights)
                    print(
                        f"Final ESS: {ess:.2f} ({ess/self.num_particles*100:.1f}% of {self.num_particles} particles)"
                    )
                else:
                    if self.chain_method == "vectorized":
                        ess_per_chain = jax.vmap(effective_sample_size)(log_weights)
                        print(f"Final ESS (per chain): {ess_per_chain}")
                    else:
                        # For parallel/sequential, compute for each chain
                        print(f"Final ESS: {effective_sample_size(log_weights)}")

    @property
    def last_state(self):
        """
        The final SMC state at the end of the sampling process.
        """
        return self._last_state
