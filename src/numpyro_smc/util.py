import jax
import jax.numpy as jnp
from collections.abc import Iterable
import numpy as np
from typing import Any, Callable, TypeAlias

from jax._src.lax import lax
from jax._src import dtypes
from jax._src.tree_util import tree_flatten, tree_unflatten, PyTreeDef, Leaf
from jax._src.util import safe_zip as zip, unzip2, HashablePartial
from jax._src.typing import Array

from numpyro.distributions.util import is_identically_one, sum_rightmost
from numpyro._typing import TraceT
from numpyro.distributions import constraints
from numpyro.handlers import seed, trace, substitute
from numpyro.infer.util import transform_fn, get_transforms

Sizes: TypeAlias = tuple[int, ...]
Shapes: TypeAlias = tuple[tuple[int, ...], ...]


def value_and_jac(f):
    """Only use when f maps to vectors."""

    def wrapper(x):
        y, vjp_fn = jax.vjp(f, x)
        jac = jax.vmap(vjp_fn)(jnp.eye(y.shape[0], dtype=y.dtype))
        return y, jac[0]

    return wrapper


def compute_split_log_probs(
    model,
    model_args: tuple,
    model_kwargs: dict,
    params_unconstrained: dict,
    sum_log_prob: bool = True,
    model_inv_transforms=None,
) -> tuple[dict[str, jax.Array], TraceT]:
    if model_inv_transforms is None:
        model_inv_transforms = get_transforms(
            seed(model, jax.random.key(0)), model_args, model_kwargs, {}
        )
    params_constrained = transform_fn(model_inv_transforms, params_unconstrained)

    model = substitute(model, data=params_constrained)
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)

    split_log_joint = {}
    for site in model_trace.values():
        if site["type"] == "sample":
            name = site["name"]
            value = site["value"]
            intermediates = site["intermediates"]
            scale = site["scale"]
            is_observed = site["is_observed"]
            fn = site["fn"]

            if intermediates:
                log_prob = fn.log_prob(value, intermediates)
            else:
                log_prob = fn.log_prob(value)

            if (scale is not None) and (not is_identically_one(scale)):
                log_prob = scale * log_prob

            # log_prob represents likelihood if observed, prior otherwise
            log_lik = log_prob if is_observed else jnp.zeros_like(log_prob)
            log_prior = jnp.zeros_like(log_prob) if is_observed else log_prob

            # compute jacobian determinant
            log_det = jnp.zeros_like(log_prob)
            # if (name in params) and (not is_observed):
            if not is_observed:
                support = fn.support
                if not (
                    support is constraints.real
                    or (
                        isinstance(support, constraints.independent)
                        and support.base_constraint is constraints.real
                    )
                ):
                    t = model_inv_transforms[name]
                    p = params_unconstrained[name]

                    # TODO: is this needed?
                    # i = site["infer"].get("_scan_current_index", None)
                    # if i is not None:
                    #     event_dim_shift = t.codomain.event_dim - t.domain.event_dim
                    #     expected_unconstrained_dim = len(fn.shape()) - event_dim_shift
                    #     # check if p has additional time dimension
                    #     if jnp.ndim(p) > expected_unconstrained_dim:
                    #         p = p[i]
                    log_det = t.log_abs_det_jacobian(p, value)
                    log_det = sum_rightmost(
                        log_det,
                        jnp.ndim(log_det) - jnp.ndim(value) + len(fn.event_shape),
                    )

            if sum_log_prob:
                log_lik = jnp.sum(log_lik)
                log_prior = jnp.sum(log_prior)
                log_det = jnp.sum(log_det)

            split_log_joint[name] = jnp.stack([log_lik, log_prior, log_det], axis=0)

    return split_log_joint, model_trace


def ravel_batched_pytree(
    pytree: Any, batch_ndims: int = 1
) -> tuple[Array, Callable[[Array], Any]]:
    leaves, treedef = tree_flatten(pytree)
    flat_matrix, unravel_list = _ravel_batched_list(leaves, batch_ndims=batch_ndims)
    return flat_matrix, HashablePartial(_unravel_batched_pytree, treedef, unravel_list)


def _unravel_batched_pytree(
    treedef: PyTreeDef,
    unravel_list: Callable[[Array], Iterable[Leaf]],
    flat_matrix: Array,
) -> Any:
    return tree_unflatten(treedef, unravel_list(flat_matrix))


def _ravel_batched_list(
    lst: list[Any], batch_ndims: int
) -> tuple[Array, Callable[[Array], list[Any]]]:
    if not lst:
        return lax.full((0, 0), 0, "float32"), lambda _: []
    from_dtypes = tuple(dtypes.dtype(l) for l in lst)
    to_dtype = dtypes.result_type(*from_dtypes)
    leading_shape = lst[0].shape[:batch_ndims]

    trailing_sizes, shapes = unzip2(
        (int(np.prod(x.shape[batch_ndims:])), x.shape) for x in lst
    )

    if all(dt == to_dtype for dt in from_dtypes):
        # Skip conversion to allow the unravel function to work with any input dtype
        del from_dtypes, to_dtype
        ravel = lambda e, m_i: lax.reshape(e, (*leading_shape, m_i))
        raveled = lax.concatenate(
            [ravel(e, m_i) for e, m_i in zip(lst, trailing_sizes)],
            dimension=batch_ndims,
        )
        return raveled, HashablePartial(
            _unravel_batched_list_single_dtype, trailing_sizes, shapes, batch_ndims
        )

    # When there is more than one distinct input dtype, we perform type
    # conversions and produce a dtype-specific unravel function.
    ravel = lambda e, m_i: lax.convert_element_type(
        lax.reshape(e, (*leading_shape, m_i)), to_dtype
    )
    raveled = lax.concatenate(
        [ravel(x, m_i) for x, m_i in zip(lst, trailing_sizes)], dimension=1
    )
    unrav = HashablePartial(
        _unravel_batched_list,
        trailing_sizes,
        shapes,
        batch_ndims,
        from_dtypes,
        to_dtype,
    )
    return raveled, unrav


def _unravel_batched_list_single_dtype(
    sizes: Sizes, shapes: Shapes, batch_ndims: int, matrix: Array
) -> list[Array]:
    chunks = lax.split(matrix, sizes, axis=batch_ndims)
    return [chunk.reshape(shape) for chunk, shape in zip(chunks, shapes)]


def _unravel_batched_list(
    sizes: Sizes,
    shapes: Shapes,
    batch_ndims: int,
    from_dtypes: tuple[np.dtype, ...],
    to_dtype: np.dtype,
    matrix,
) -> list[Array]:
    arr_dtype = dtypes.dtype(matrix)
    if arr_dtype != to_dtype:
        raise TypeError(
            f"unravel function given array of dtype {arr_dtype}, "
            f"but expected dtype {to_dtype}"
        )
    chunks = lax.split(matrix, sizes, axis=batch_ndims)
    return [
        lax.convert_element_type(chunk.reshape(shape), dtype)
        for chunk, shape, dtype in zip(chunks, shapes, from_dtypes)
    ]
