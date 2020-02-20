# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import singledispatch

import torch

import pyro.distributions as dist
from pyro.poutine.messenger import Messenger
from pyro.poutine.util import site_is_subsample


class PrefixReplayMessenger(Messenger):
    """
    EXPERIMENTAL Given a trace of training data, replay a model with batched
    sites extended to include both training and forecast time, using the guide
    trace for the training prefix and samples from the prior on the forecast
    postfix.

    :param trace: a guide trace.
    :type trace: ~pyro.poutine.trace_struct.Trace
    """
    def __init__(self, trace):
        super().__init__()
        self.trace = trace

    def _pyro_post_sample(self, msg):
        if site_is_subsample(msg):
            return

        name = msg["name"]
        if name not in self.trace:
            return

        model_value = msg["value"]
        guide_value = self.trace.nodes[name]["value"]
        if model_value.shape == guide_value.shape:
            msg["value"] = guide_value
            return

        # Search for a single dim with mismatched size.
        assert model_value.dim() == guide_value.dim()
        for dim in range(model_value.dim()):
            if model_value.size(dim) != guide_value.size(dim):
                break
        assert model_value.size(dim) > guide_value.size(dim)
        assert model_value.shape[dim + 1:] == guide_value.shape[dim + 1:]
        split = guide_value.size(dim)
        index = (slice(None),) * dim + (slice(split, None),)
        msg["value"] = torch.cat([guide_value, model_value[index]], dim=dim)


class PrefixConditionMessenger(Messenger):
    """
    EXPERIMENTAL Given a prefix of t-many observations, condition a (t+f)-long
    distribution on the observations, converting it to an f-long distribution.

    :param dict data: A dict mapping site name to tensors of observations.
    """
    def __init__(self, data):
        super().__init__()
        self.data = data

    def _pyro_sample(self, msg):
        if msg["name"] not in self.data:
            return

        assert msg["value"] is None
        data = self.data[msg["name"]]
        msg["fn"] = prefix_condition(msg["fn"], data)


# ----------------------------------------------------------------------------
# The pattern-matching code in the remainder of this file could be eventually
# replace by much simpler Funsor logic.

UNIVARIATE_DISTS = {
    dist.Cauchy: ("loc", "scale"),
    dist.Laplace: ("loc", "scale"),
    dist.Normal: ("loc", "scale"),
    dist.Stable: ("stability", "skew", "scale", "loc"),
    dist.StudentT: ("df", "loc", "scale"),
}


@singledispatch
def prefix_condition(d, data):
    """
    EXPERIMENTAL Given a distribution ``d`` of shape ``batch_shape + (t+f, d)``
    and data ``x`` of shape ``batch_shape + (t, d)``, compute a conditional
    distribution of shape ``batch_shape + (f, d)``. Typically ``t`` is the
    number of training time steps, ``f`` is the number of forecast time steps,
    and ``d`` is the data dimension.

    :param d: a distribution with ``len(d.shape()) >= 2``
    :type d: ~pyro.distributions.Distribution
    :param data: data of dimension at least 2.
    :type data: ~torch.Tensor
    """
    try:
        return d.prefix_condition(data)
    except AttributeError:
        raise NotImplementedError("prefix_condition() does not suport {}".format(type(d)))


@prefix_condition.register(dist.Independent)
def _(d, data):
    base_dist = prefix_condition(d.base_dist, data)
    return base_dist.to_event(d.reinterpreted_batch_ndims)


def _prefix_condition_univariate(d, data):
    t = data.size(-2)
    params = [getattr(d, name)[..., t:, :]
              for name in UNIVARIATE_DISTS[type(d)]]
    return type(d)(*params)


for _type in UNIVARIATE_DISTS:
    prefix_condition.register(_type)(_prefix_condition_univariate)


@prefix_condition.register(dist.MultivariateNormal)
def _(d, data):
    t = data.size(-2)
    loc = d.loc[..., t:, :]
    scale_tril = d.scale_tril[..., t:, :, :]
    return dist.MultivariateNormal(loc, scale_tril=scale_tril)


@singledispatch
def reshape_batch(d, batch_shape):
    """
    EXPERIMENTAL Given a distribution ``d``, reshape to different batch shape
    of same number of elements.

    This is typically used to move the the rightmost batch dimension "time" to
    an event dimension, while preserving the positions of other batch
    dimensions.

    :param d: A distribution.
    :type d: ~pyro.distributions.Distribution
    :param tuple batch_shape: A new batch shape.
    :returns: A distribution with the same type but given batch shape.
    :rtype: ~pyro.distributions.Distribution
    """
    raise NotImplementedError


@reshape_batch.register(dist.Independent)
def _(d, batch_shape):
    base_shape = batch_shape + d.event_shape[:d.reinterpreted_batch_ndims]
    base_dist = reshape_batch(d.base_dist, base_shape)
    return base_dist.to_event(d.reinterpreted_batch_ndims)


def _reshape_batch_univariate(d, batch_shape):
    params = [getattr(d, name).reshape(batch_shape)
              for name in UNIVARIATE_DISTS[type(d)]]
    return type(d)(*params)


for _type in UNIVARIATE_DISTS:
    reshape_batch.register(_type)(_reshape_batch_univariate)


@reshape_batch.register(dist.MultivariateNormal)
def _(d, batch_shape):
    dim = d.event_shape[0]
    loc = d.loc.reshape(batch_shape + (dim,))
    scale_tril = d.scale_tril.reshape(batch_shape + (dim, dim))
    return dist.MultivariateNormal(loc, scale_tril=scale_tril)


@reshape_batch.register(dist.GaussianHMM)
def _(d, batch_shape):
    init = d._init
    if init.batch_shape:
        init = init.expand(d.batch_shape)
        init = init.reshape(batch_shape)

    trans = d._trans
    if len(trans.batch_shape) > 1:
        trans = trans.expand(d.batch_shape + (-1,))
        trans = trans.reshape(batch_shape + (-1,))

    obs = d._obs
    if len(obs.batch_shape) > 1:
        obs = obs.expand(d.batch_shape + (-1,))
        obs = obs.reshape(batch_shape + (-1,))

    new = d._get_checked_instance(dist.GaussianHMM)
    new.hidden_dim = d.hidden_dim
    new.obs_dim = d.obs_dim
    new._init = init
    new._trans = trans
    new._obs = obs
    super(dist.GaussianHMM, new).__init__(d.duration, batch_shape, d.event_shape,
                                          validate_args=d._validate_args)
    return new


@reshape_batch.register(dist.LinearHMM)
def _(d, batch_shape):
    init_dist = d.initial_dist
    if init_dist.batch_shape:
        init_dist = init_dist.expand(d.batch_shape)
        init_dist = reshape_batch(init_dist, batch_shape)

    trans_mat = d.transition_matrix
    if trans_mat.dim() > 3:
        trans_mat = trans_mat.expand(d.batch_shape + (-1, d.hidden_dim, d.hidden_dim))
        trans_mat = trans_mat.reshape(batch_shape + (-1, d.hidden_dim, d.hidden_dim))

    trans_dist = d.transition_dist
    if len(trans_dist.batch_shape) > 1:
        trans_dist = trans_dist.expand(d.batch_shape + (-1,))
        trans_dist = reshape_batch(trans_dist, batch_shape + (-1,))

    obs_mat = d.observation_matrix
    if obs_mat.dim() > 3:
        obs_mat = obs_mat.expand(d.batch_shape + (-1, d.hidden_dim, d.hidden_dim))
        obs_mat = obs_mat.reshape(batch_shape + (-1, d.hidden_dim, d.hidden_dim))

    obs_dist = d.observation_dist
    if len(obs_dist.batch_shape) > 1:
        obs_dist = obs_dist.expand(d.batch_shape + (-1,))
        obs_dist = reshape_batch(obs_dist, batch_shape + (-1,))

    return dist.LinearHMM(init_dist, trans_mat, trans_dist, obs_mat, obs_dist,
                          duration=d.duration)
