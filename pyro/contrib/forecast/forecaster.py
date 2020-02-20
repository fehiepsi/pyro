# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoNormal, init_to_sample
from pyro.nn.module import PyroModule
from pyro.optim import ClippedAdam

from .util import PrefixConditionMessenger, PrefixReplayMessenger, reshape_batch

logger = logging.getLogger(__name__)


class _ForecastingModelMeta(type(PyroModule), ABCMeta):
    pass


class ForecastingModel(PyroModule, metaclass=_ForecastingModelMeta):
    """
    Abstract base class for forecasting models.

    Derived classes must implement the :meth:`model` method.
    """
    def __init__(self):
        super().__init__()
        self._prefix_condition_data = {}

    @abstractmethod
    def model(self, zero_data, covariates):
        """
        Generative model definition.

        Implementations must call the :meth:`predict` method exactly once.

        Implementations must draw all time-dependent noise inside the
        :meth:`time_plate`. The prediction passed to :meth:`predict` must be a
        deterministic function of noise tensors that are independent over time.
        This requirement is slightly more general than state space models.

        :param zero_data: A zero tensor like the input data, but extended to
            the duration of the :meth:`time_plate`. This allows models to
            depend on the shape and device of data but not its value.
        :type zero_data: ~torch.Tensor
        :param covariates: A tensor of covariates with time dimension -2.
        :type covariates: ~torch.Tensor
        :returns: Return value is ignored.
        """
        raise NotImplementedError

    @property
    def time_plate(self):
        """
        :returns: A plate named "time" with size ``covariates.size(-2)`` and
            ``dim=-1``. This is available only during model execution.
        :rtype: :class:`~pyro.plate`
        """
        assert self._time_plate is not None, ".time_plate accessed outside of .model()"
        return self._time_plate

    def predict(self, noise_dist, prediction):
        """
        Prediction function, to be called by :meth:`model` implementations.

        This should be called outside of the  :meth:`time_plate`.

        This is similar to an observe statement in Pyro::

            pyro.sample("residual", noise_dist,
                        obs=(data - prediction))

        but with (1) additional reshaping logic to allow time-dependent
        ``noise_dist`` (most often a :class:`~pyro.distributions.GaussianHMM`
        or variant); and (2) additional logic to allow only a partial
        observation and forecast the remaining data.

        :param noise_dist: A noise distribution with ``.event_dim in {0,1,2}``.
            ``noise_dist`` is typically zero-mean or zero-median or zero-mode
            or somehow centered.
        :type noise_dist: ~pyro.distributions.Distribution
        :param prediction: A prediction for the data. This should have the same
            shape as ``data``, but extended to full duration of the
            ``covariates``.
        :type prediction: ~torch.Tensor
        """
        assert self._data is not None, ".predict() called outside .model()"
        assert self._forecast is None, ".predict() called twice"
        assert isinstance(noise_dist, dist.Distribution)
        assert isinstance(prediction, torch.Tensor)
        assert len(noise_dist.shape()) >= 2
        assert noise_dist.shape()[-2:] == prediction.shape[-2:]
        assert noise_dist.event_dim in {0, 1, 2}
        noise_dist = noise_dist.to_event(2 - noise_dist.event_dim)

        # The following reshaping logic is required to reconcile batch and
        # event shapes. This would be unnecessary if Pyro used name dimensions
        # internally, e.g. using Funsor.
        #
        #     batch_shape                    | event_shape
        #     -------------------------------+----------------
        #  1. sample_shape + shape + (time,) | (obs_dim,)
        #  2.           sample_shape + shape | (time, obs_dim)
        #  3.    sample_shape + shape + (1,) | (time, obs_dim)
        #
        # Parameters like noise_dist.loc typically have shape as in 1.  However
        # calling .to_event(1) will shift the shapes resulting in 2., where
        # sample_shape+shape will be misaligned with other batch shapes in the
        # trace. To fix this the following logic "unsqueezes" the distribution,
        # resulting in correctly aligned shapes 3. Note the "time" dimension is
        # effectively moved from a batch dimension to an event dimension.
        noise_dist = reshape_batch(noise_dist, noise_dist.batch_shape + (1,))
        data = self._data.unsqueeze(-3)
        prediction = prediction.unsqueeze(-3)

        # Create a sample site.
        t_obs = data.size(-2)
        t_cov = prediction.size(-2)
        if t_obs == t_cov:  # training
            pyro.sample("residual", noise_dist, obs=data - prediction)
            self._forecast = data.new_zeros(data.shape[:-2] + (0,) + data.shape[-1:])
        else:  # forecasting
            left_pred = prediction[..., :t_obs, :]
            right_pred = prediction[..., t_obs:, :]

            # This prefix_condition indirection is needed to ensure that
            # PrefixConditionMessenger is handled outside of the .model() call.
            self._prefix_condition_data["residual"] = data - left_pred
            noise = pyro.sample("residual", noise_dist)
            del self._prefix_condition_data["residual"]

            assert noise.shape[-data.dim():] == right_pred.shape[-data.dim():]
            self._forecast = right_pred + noise

        # Move the "time" batch dim back to its original place.
        assert self._forecast.size(-3) == 1
        self._forecast = self._forecast.squeeze(-3)

    def forward(self, data, covariates):
        assert data.dim() >= 2
        assert covariates.dim() >= 2
        t_obs = data.size(-2)
        t_cov = covariates.size(-2)
        assert t_obs <= t_cov

        try:
            self._data = data
            self._time_plate = pyro.plate("time", t_cov, dim=-1)
            if t_obs == t_cov:  # training
                zero_data = data.new_zeros(()).expand(data.shape)
            else:  # forecasting
                zero_data = data.new_zeros(()).expand(
                    data.shape[:-2] + covariates.shape[-2:-1] + data.shape[-1:])
            self._forecast = None

            self.model(zero_data, covariates)

            assert self._forecast is not None, ".predict() was not called by .model()"
            return self._forecast
        finally:
            self._data = None
            self._time_plate = None
            self._forecast = None


class Forecaster(nn.Module):
    """
    Forecaster for a :class:`ForecastingModel`.

    On initialization, this fits a distribution using variational inference
    over latent variables and exact inference over the noise distribution,
    typically a :class:`~pyro.distributions.GaussianHMM` or variant.

    After construction this can be called to generate sample forecasts.

    :ivar list losses: A list of losses recorded during training, typically
        used to debug convergence. Defined by ``loss = -elbo / data.numel()``.
    :param ForecastingModel model: A forecasting model subclass instance.
    :param data: A tensor dataset with time dimension -2.
    :type data: ~torch.Tensor
    :param covariates: A tensor of covariates with time dimension -2.
        For models not using covariates, pass a shaped empty tensor
        ``torch.empty(duration, 0)``.
    :type covariates: ~torch.Tensor
    """
    def __init__(self, model, data, covariates, *,
                 learning_rate=0.01,
                 betas=(0.9, 0.99),
                 learning_rate_decay=0.1,
                 num_steps=1001,
                 log_every=100,
                 init_scale=0.1,
                 num_particles=1,
                 vectorize_particles=True):
        assert data.size(-2) == covariates.size(-2)
        super().__init__()
        self.model = model
        self.guide = AutoNormal(self.model, init_loc_fn=init_to_sample, init_scale=init_scale)
        optim = ClippedAdam({"lr": learning_rate, "betas": betas,
                             "lrd": learning_rate_decay ** (1 / num_steps)})
        elbo = Trace_ELBO(num_particles=num_particles,
                          vectorize_particles=vectorize_particles)
        elbo._guess_max_plate_nesting(self.model, self.guide, (data, covariates), {})
        elbo.max_plate_nesting = max(elbo.max_plate_nesting, 1)  # force a time plate

        svi = SVI(self.model, self.guide, optim, elbo)

        losses = []
        for step in range(num_steps):
            loss = svi.step(data, covariates) / data.numel()
            if log_every and step % log_every == 0:
                logger.info("step {: >4d} loss = {:0.6g}".format(step, loss))
            losses.append(loss)

        self.max_plate_nesting = elbo.max_plate_nesting
        self.losses = losses

    @torch.no_grad()
    def forward(self, data, covariates, num_samples):
        assert data.size(-2) < covariates.size(-2)
        assert self.max_plate_nesting >= 1
        dim = -1 - self.max_plate_nesting

        with poutine.trace() as tr:
            with pyro.plate("particles", num_samples, dim=dim):
                self.guide()
        with PrefixReplayMessenger(tr.trace):
            with PrefixConditionMessenger(self.model._prefix_condition_data):
                with pyro.plate("particles", num_samples, dim=dim):
                    return self.model(data, covariates)
