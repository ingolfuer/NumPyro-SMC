# Adapted from JAX source

from collections.abc import Iterable
import numpy as np
from typing import Any, Callable, TypeAlias

from jax._src.lax import lax
from jax._src import dtypes
from jax._src.tree_util import tree_flatten, tree_unflatten, PyTreeDef, Leaf
from jax._src.util import safe_zip as zip, unzip2, HashablePartial
from jax._src.typing import Array

Sizes: TypeAlias = tuple[int, ...]
Shapes: TypeAlias = tuple[tuple[int, ...], ...]


def ravel_batched_pytree(pytree: Any) -> tuple[Array, Callable[[Array], Any]]:
    """Ravel (flatten) a pytree of arrays down to a 2D array.
    All pytree leaves are expected to have ndim >= 1 and equally
    sized first dimensions.

    Args:
      pytree: a pytree of arrays and scalars to ravel.

    Returns:
      A pair where the first element is a 2D array representing the flattened and
      concatenated leaf values, with dtype determined by promoting the dtypes of
      leaf values, and the second element is a callable for unflattening a 2D
      array of the same length back to a pytree of the same structure as the
      input ``pytree``.

    """
    leaves, treedef = tree_flatten(pytree)
    flat_matrix, unravel_list = _ravel_batched_list(leaves)
    return flat_matrix, HashablePartial(_unravel_batched_pytree, treedef, unravel_list)


def _unravel_batched_pytree(
    treedef: PyTreeDef,
    unravel_list: Callable[[Array], Iterable[Leaf]],
    flat_matrix: Array,
) -> Any:
    return tree_unflatten(treedef, unravel_list(flat_matrix))


def _ravel_batched_list(
    lst: list[Any], /
) -> tuple[Array, Callable[[Array], list[Any]]]:
    if not lst:
        return lax.full((0, 0), 0, "float32"), lambda _: []
    from_dtypes = tuple(dtypes.dtype(l) for l in lst)
    to_dtype = dtypes.result_type(*from_dtypes)
    n = lst[0].shape[0]

    trailing_sizes, shapes = unzip2((int(np.prod(x.shape[1:])), x.shape) for x in lst)

    if all(dt == to_dtype for dt in from_dtypes):
        # Skip conversion to allow the unravel function to work with any input dtype
        del from_dtypes, to_dtype
        ravel = lambda e, m_i: lax.reshape(e, (n, m_i))
        raveled = lax.concatenate(
            [ravel(e, m_i) for e, m_i in zip(lst, trailing_sizes)], dimension=1
        )
        return raveled, HashablePartial(
            _unravel_batched_list_single_dtype, trailing_sizes, shapes
        )

    # When there is more than one distinct input dtype, we perform type
    # conversions and produce a dtype-specific unravel function.
    ravel = lambda e, m_i: lax.convert_element_type(lax.reshape(e, (n, m_i)), to_dtype)
    raveled = lax.concatenate(
        [ravel(x, m_i) for x, m_i in zip(lst, trailing_sizes)], dimension=1
    )
    unrav = HashablePartial(
        _unravel_batched_list, trailing_sizes, shapes, from_dtypes, to_dtype
    )
    return raveled, unrav


def _unravel_batched_list_single_dtype(
    sizes: Sizes, shapes: Shapes, matrix: Array
) -> list[Array]:
    chunks = lax.split(matrix, sizes, axis=1)
    return [chunk.reshape(shape) for chunk, shape in zip(chunks, shapes)]


def _unravel_batched_list(
    sizes: Sizes,
    shapes: Shapes,
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
    chunks = lax.split(matrix, sizes, axis=1)
    return [
        lax.convert_element_type(chunk.reshape(shape), dtype)
        for chunk, shape, dtype in zip(chunks, shapes, from_dtypes)
    ]
