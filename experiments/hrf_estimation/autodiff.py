"""Optional JAX gradients for the unchanged, profiled continuous HRF objective."""

from collections import OrderedDict
from hashlib import sha256

import numpy as np
import jax
import jax.numpy as jnp

from experiments.hrf_estimation.autodiff_design import design
from experiments.hrf_estimation.continuous import _decode
from experiments.hrf_estimation.prepared import valid_shape

_CACHE = OrderedDict()
_STATIC_KEYS = ("sampling","weights","duration","tr","mode","q","c","c_norm")


def _fingerprint(value, digest):
    if isinstance(value,dict):
        for key in sorted(value):
            digest.update(key.encode())
            _fingerprint(value[key],digest)
    elif isinstance(value,np.ndarray):
        digest.update(str((value.shape,value.dtype.str)).encode())
        digest.update(value.tobytes())
    else:
        digest.update(repr(value).encode())


def _error(unit, item, projected_y, raw_y):
    z = design(unit,item)
    q,c = jnp.asarray(item["q"]),jnp.asarray(item["c"])
    projected = z-q@(q.T@z)
    threshold = (max(c.shape[0],c.shape[1]+z.shape[1])*jnp.finfo(z.dtype).eps
                 *jnp.maximum(item["c_norm"],jnp.linalg.norm(z,ord=2)))
    rank = jnp.sum(jnp.linalg.svd(projected,compute_uv=False)>threshold)
    fallback = rank<jnp.linalg.matrix_rank(z)
    def full(_):
        matrix = jnp.column_stack([c,z])
        beta = jax.lax.stop_gradient(jnp.linalg.lstsq(
            matrix,raw_y,rcond=jnp.finfo(z.dtype).eps*max(matrix.shape))[0])
        return raw_y-matrix@beta
    def reduced(_):
        beta = jax.lax.stop_gradient(jnp.linalg.lstsq(
            projected,projected_y,rcond=jnp.finfo(z.dtype).eps*max(projected.shape))[0])
        return projected_y-projected@beta
    return jax.lax.cond(fallback,full,reduced,None)


def compiled_objective(prepared):
    static = tuple({key:item[key] for key in _STATIC_KEYS} for item in prepared)
    digest = sha256()
    for item in static:
        _fingerprint(item,digest)
    key = digest.hexdigest()
    if key not in _CACHE:
        def objective(unit,signals,energy):
            return sum(jnp.sum(_error(unit,item,*ys)**2)
                       for item,ys in zip(static,signals))/energy
        _CACHE[key] = jax.jit(jax.value_and_grad(objective))
        if len(_CACHE)>32:
            _CACHE.popitem(last=False)
    _CACHE.move_to_end(key)
    return _CACHE[key]


def make_value_gradient(prepared,feature):
    compiled = compiled_objective(prepared)
    signals = tuple((jnp.asarray(item["y"][:,feature]),jnp.asarray(item["signals"][:,feature]))
                    for item in prepared)
    energy = sum(item["energy"][feature] for item in prepared)
    raw = sum(item["raw"][feature] for item in prepared)
    def evaluate(unit):
        if energy<=64*np.finfo(float).eps*raw:
            return np.nan,np.zeros(5)
        if not valid_shape(_decode(unit)):
            return np.inf,np.zeros(5)
        value,gradient = compiled(jnp.asarray(unit),signals,energy)
        return float(value),np.asarray(gradient)
    return evaluate


def value_gradient(prepared,feature,unit):
    return make_value_gradient(prepared,feature)(unit)


def design_value(unit,item):
    return np.asarray(design(jnp.asarray(unit),item))
