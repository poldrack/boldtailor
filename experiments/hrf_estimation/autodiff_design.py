"""JAX version of the existing double-gamma sampling and normalization."""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

from boldtailor._hrf_design import OVERSAMPLING
from boldtailor.hrf_library import _kernel_samples

jax.config.update("jax_enable_x64", True)


def decode(unit):
    low = jnp.array([2., 8., np.log(.3), np.log(.5), np.log(2.)])
    high = jnp.array([10., 24., np.log(2.5), np.log(4.), np.log(20.)])
    values = low + unit*(high-low)
    return jnp.concatenate([values[:2], jnp.exp(values[2:])])


def hrf(unit, tr):
    a,b,c,d,ratio = decode(unit)
    dt = tr/OVERSAMPLING
    times = jnp.arange(_kernel_samples(36.,dt))*dt
    positive = jnp.where(times>0,times,1.)
    def gamma_pdf(delay, dispersion):
        shape = delay/dispersion
        logpdf = ((shape-1)*jnp.log(positive)-positive/dispersion
                  -gammaln(shape)-shape*jnp.log(dispersion))
        return jnp.where(times>0,jnp.exp(logpdf),0.)
    values = gamma_pdf(a,c)-gamma_pdf(b,d)/ratio
    return values/jnp.max(values)


def _event_scales(kernel, counts):
    unique,inverse = np.unique(np.maximum(1,counts),return_inverse=True)
    prefix = jnp.concatenate([jnp.zeros(1),jnp.cumsum(kernel)])
    peaks = []
    for count in unique:
        indices = np.arange(len(kernel)+int(count)-1)
        upper = np.minimum(indices+1,len(kernel))
        lower = np.maximum(0,indices-int(count)+1)
        peaks.append(jnp.max(prefix[upper]-prefix[lower]))
    return 1/jnp.stack(peaks)[inverse]


def _sample(sampling, kernel, normalize):
    prefix = jnp.concatenate([jnp.zeros(1),jnp.cumsum(kernel)])
    def at(indices):
        a = np.clip(indices[:,None]-sampling["starts"]+1,0,len(kernel))
        b = np.clip(indices[:,None]-sampling["stops"]+1,0,len(kernel))
        return prefix[a]-prefix[b]
    fraction = jnp.asarray(sampling["fraction"][:,None])
    result = (1-fraction)*at(sampling["lower"])+fraction*at(sampling["upper"])
    if normalize:
        result *= _event_scales(kernel,sampling["stops"]-sampling["starts"])
    return result


def design(unit, item):
    kernel = hrf(unit,item["tr"])
    z = _sample(item["sampling"],kernel,True)@jnp.asarray(item["weights"])
    if item["mode"] == "duration":
        rt = (jnp.zeros(z.shape[0]) if item["duration"] is None else
              _sample(item["duration"],kernel,False).sum(axis=1))
        z = z.at[:,4].set(rt)
    return z
