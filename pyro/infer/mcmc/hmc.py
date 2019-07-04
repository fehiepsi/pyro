from __future__ import absolute_import, division, print_function

import math
from collections import OrderedDict

import torch

import pyro
import pyro.distributions as dist
from pyro.distributions.util import eye_like, scalar_like

from pyro.infer.mcmc.adaptation import WarmupAdapter
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
from pyro.infer.mcmc.util import initialize_model
from pyro.ops.integrator import velocity_verlet
from pyro.util import optional, torch_isnan


class HMC(MCMCKernel):
    r"""
    Simple Hamiltonian Monte Carlo kernel, where ``step_size`` and ``num_steps``
    need to be explicitly specified by the user.

    **References**

    [1] `MCMC Using Hamiltonian Dynamics`,
    Radford M. Neal

    :param model: Python callable containing Pyro primitives.
    :param potential_fn: Python callable calculating potential energy with input
        is a dict of real support parameters.
    :param float step_size: Determines the size of a single step taken by the
        verlet integrator while computing the trajectory using Hamiltonian
        dynamics. If not specified, it will be set to 1.
    :param float trajectory_length: Length of a MCMC trajectory. If not
        specified, it will be set to ``step_size x num_steps``. In case
        ``num_steps`` is not specified, it will be set to :math:`2\pi`.
    :param int num_steps: The number of discrete steps over which to simulate
        Hamiltonian dynamics. The state at the end of the trajectory is
        returned as the proposal. This value is always equal to
        ``int(trajectory_length / step_size)``.
    :param bool adapt_step_size: A flag to decide if we want to adapt step_size
        during warm-up phase using Dual Averaging scheme.
    :param bool adapt_mass_matrix: A flag to decide if we want to adapt mass
        matrix during warm-up phase using Welford scheme.
    :param bool full_mass: A flag to decide if mass matrix is dense or diagonal.
    :param dict transforms: Optional dictionary that specifies a transform
        for a sample site with constrained support to unconstrained space. The
        transform should be invertible, and implement `log_abs_det_jacobian`.
        If not specified and the model has sites with constrained support,
        automatic transformations will be applied, as specified in
        :mod:`torch.distributions.constraint_registry`.
    :param int max_plate_nesting: Optional bound on max number of nested
        :func:`pyro.plate` contexts. This is required if model contains
        discrete sample sites that can be enumerated over in parallel.
    :param bool jit_compile: Optional parameter denoting whether to use
        the PyTorch JIT to trace the log density computation, and use this
        optimized executable trace in the integrator.
    :param dict jit_options: A dictionary contains optional arguments for
        :func:`torch.jit.trace` function.
    :param bool ignore_jit_warnings: Flag to ignore warnings from the JIT
        tracer when ``jit_compile=True``. Default is False.
    :param float target_accept_prob: Increasing this value will lead to a smaller
        step size, hence the sampling will be slower and more robust. Default to 0.8.

    .. note:: Internally, the mass matrix will be ordered according to the order
        of the names of latent variables, not the order of their appearance in
        the model.

    Example:

        >>> true_coefs = torch.tensor([1., 2., 3.])
        >>> data = torch.randn(2000, 3)
        >>> dim = 3
        >>> labels = dist.Bernoulli(logits=(true_coefs * data).sum(-1)).sample()
        >>>
        >>> def model(data):
        ...     coefs_mean = torch.zeros(dim)
        ...     coefs = pyro.sample('beta', dist.Normal(coefs_mean, torch.ones(3)))
        ...     y = pyro.sample('y', dist.Bernoulli(logits=(coefs * data).sum(-1)), obs=labels)
        ...     return y
        >>>
        >>> hmc_kernel = HMC(model, step_size=0.0855, num_steps=4)
        >>> mcmc_run = MCMC(hmc_kernel, num_samples=500, warmup_steps=100).run(data)
        >>> posterior = mcmc_run.marginal('beta').empirical['beta']
        >>> posterior.mean  # doctest: +SKIP
        tensor([ 0.9819,  1.9258,  2.9737])
    """

    def __init__(self,
                 model=None,
                 potential_fn=None,
                 step_size=1,
                 trajectory_length=None,
                 num_steps=None,
                 adapt_step_size=True,
                 adapt_mass_matrix=True,
                 full_mass=False,
                 transforms=None,
                 max_plate_nesting=None,
                 jit_compile=False,
                 jit_options=None,
                 ignore_jit_warnings=False,
                 target_accept_prob=0.8):
        if not ((model is None) ^ (potential_fn is None)):
            raise ValueError("Only one of `model` or `potential_fn` must be specified.")
        # NB: deprecating args - model, transforms
        self.model = model
        self.transforms = transforms
        self._max_plate_nesting = max_plate_nesting
        self._jit_compile = jit_compile
        self._jit_options = jit_options
        self._ignore_jit_warnings = ignore_jit_warnings

        self.potential_fn = potential_fn
        if trajectory_length is not None:
            self.trajectory_length = trajectory_length
        elif num_steps is not None:
            self.trajectory_length = step_size * num_steps
        else:
            self.trajectory_length = 2 * math.pi  # from Stan
        # The following parameter is used in find_reasonable_step_size method.
        # In NUTS paper, this threshold is set to a fixed log(0.5).
        # After https://github.com/stan-dev/stan/pull/356, it is set to a fixed log(0.8).
        self._direction_threshold = math.log(0.8)  # from Stan
        self._max_sliced_energy = 1000
        self._reset()
        self._adapter = WarmupAdapter(step_size,
                                      adapt_step_size=adapt_step_size,
                                      adapt_mass_matrix=adapt_mass_matrix,
                                      target_accept_prob=target_accept_prob,
                                      is_diag_mass=not full_mass)
        super(HMC, self).__init__()

    def _kinetic_energy(self, r):
        r_flat = torch.cat([r[site_name].reshape(-1) for site_name in sorted(r)])
        if self.inverse_mass_matrix.dim() == 2:
            return 0.5 * self.inverse_mass_matrix.matmul(r_flat).dot(r_flat)
        else:
            return 0.5 * self.inverse_mass_matrix.dot(r_flat ** 2)

    def _energy(self, z, r):
        return self._kinetic_energy(r) + self.potential_fn(z)

    def _reset(self):
        self._t = 0
        self._accept_cnt = 0
        self._num_diverging = 0
        self._prototype_trace = None
        self._initial_params = None
        self._z_last = None
        self._potential_energy_last = None
        self._z_grads_last = None
        self._warmup_steps = None

    def _find_reasonable_step_size(self, z):
        step_size = self.step_size

        # We are going to find a step_size which make accept_prob (Metropolis correction)
        # near the target_accept_prob. If accept_prob:=exp(-delta_energy) is small,
        # then we have to decrease step_size; otherwise, increase step_size.
        potential_energy = self.potential_fn(z)
        r, _ = self._sample_r(name="r_presample_0")
        energy_current = self._kinetic_energy(r) + potential_energy
        z_new, r_new, z_grads_new, potential_energy_new = velocity_verlet(
            z, r, self.potential_fn, self.inverse_mass_matrix, step_size)
        energy_new = self._kinetic_energy(r_new) + potential_energy_new
        delta_energy = energy_new - energy_current
        # direction=1 means keep increasing step_size, otherwise decreasing step_size.
        # Note that the direction is -1 if delta_energy is `NaN` which may be the
        # case for a diverging trajectory (e.g. in the case of evaluating log prob
        # of a value simulated using a large step size for a constrained sample site).
        direction = 1 if self._direction_threshold < -delta_energy else -1

        # define scale for step_size: 2 for increasing, 1/2 for decreasing
        step_size_scale = 2 ** direction
        direction_new = direction
        # keep scale step_size until accept_prob crosses its target
        # TODO: make thresholds for too small step_size or too large step_size
        t = 0
        while direction_new == direction:
            t += 1
            step_size = step_size_scale * step_size
            r, _ = self._sample_r(name="r_presample_{}".format(t))
            energy_current = self._kinetic_energy(r) + potential_energy
            z_new, r_new, z_grads_new, potential_energy_new = velocity_verlet(
                z, r, self.potential_fn, self.inverse_mass_matrix, step_size)
            energy_new = self._kinetic_energy(r_new) + potential_energy_new
            delta_energy = energy_new - energy_current
            direction_new = 1 if self._direction_threshold < -delta_energy else -1
        return step_size

    def _sample_r(self, name):
        r_dist = self._adapter.r_dist
        r_flat = pyro.sample(name, r_dist)
        r = {}
        pos = 0
        for name, param in sorted(self.initial_params.items()):
            next_pos = pos + param.numel()
            r[name] = r_flat[pos:next_pos].reshape(param.shape)
            pos = next_pos
        assert pos == r_flat.size(0)
        return r, r_flat

    @property
    def inverse_mass_matrix(self):
        return self._adapter.inverse_mass_matrix

    @property
    def step_size(self):
        return self._adapter.step_size

    @property
    def num_steps(self):
        return max(1, int(self.trajectory_length / self.step_size))

    @property
    def initial_params(self):
        return self._initial_params

    @initial_params.setter
    def initial_params(self, params):
        self._initial_params = params

    def _initialize_model_properties(self, model_args, model_kwargs):
        init_params, potential_fn, transforms, trace = initialize_model(
            self.model,
            model_args,
            model_kwargs,
            transforms=self.transforms,
            max_plate_nesting=self._max_plate_nesting,
            jit_compile=self._jit_compile,
            jit_options=self._jit_options,
            skip_jit_warnings=self._ignore_jit_warnings,
        )
        self.potential_fn = potential_fn
        self.transforms = transforms
        if self._initial_params is None:
            self._initial_params = init_params
        self._prototype_trace = trace

    def _initialize_adapter(self):
        mass_matrix_size = sum([p.numel() for p in self.initial_params.values()])
        site_value = list(self.initial_params.values())[0]
        if self._adapter.is_diag_mass:
            initial_mass_matrix = torch.ones(mass_matrix_size,
                                             dtype=site_value.dtype,
                                             device=site_value.device)
        else:
            initial_mass_matrix = eye_like(site_value, mass_matrix_size)
        self._adapter.configure(self._warmup_steps,
                                inv_mass_matrix=initial_mass_matrix,
                                find_reasonable_step_size_fn=self._find_reasonable_step_size)

        if self._adapter.adapt_step_size:
            self._adapter.reset_step_size_adaptation(self._initial_params)

    def setup(self, warmup_steps, *args, **kwargs):
        self._warmup_steps = warmup_steps
        if self.model is not None:
            self._initialize_model_properties(args, kwargs)
        potential_energy = self.potential_fn(self.initial_params)
        self._cache(self.initial_params, potential_energy, None)
        if self.initial_params:
            self._initialize_adapter()

    def cleanup(self):
        self._reset()

    def _cache(self, z, potential_energy, z_grads=None):
        self._z_last = z
        self._potential_energy_last = potential_energy
        self._z_grads_last = z_grads

    def clear_cache(self):
        self._z_last = None
        self._potential_energy_last = None
        self._z_grads_last = None

    def _fetch_from_cache(self):
        return self._z_last, self._potential_energy_last, self._z_grads_last

    def sample(self, params):
        z, potential_energy, z_grads = self._fetch_from_cache()
        # recompute PE when cache is cleared
        if z is None:
            z = params
            potential_energy = self.potential_fn(z)
            self._cache(z, potential_energy)
        # return early if no sample sites
        elif len(z) == 0:
            self._accept_cnt += 1
            self._t += 1
            return params
        r, _ = self._sample_r(name="r_t={}".format(self._t))
        energy_current = self._kinetic_energy(r) + potential_energy

        # Temporarily disable distributions args checking as
        # NaNs are expected during step size adaptation
        with optional(pyro.validation_enabled(False), self._t < self._warmup_steps):
            z_new, r_new, z_grads_new, potential_energy_new = velocity_verlet(z, r, self.potential_fn,
                                                                              self.inverse_mass_matrix,
                                                                              self.step_size,
                                                                              self.num_steps,
                                                                              z_grads=z_grads)
            # apply Metropolis correction.
            energy_proposal = self._kinetic_energy(r_new) + potential_energy_new
        delta_energy = energy_proposal - energy_current
        # handle the NaN case which may be the case for a diverging trajectory
        # when using a large step size.
        delta_energy = scalar_like(delta_energy, float("inf")) if torch_isnan(delta_energy) else delta_energy
        if delta_energy > self._max_sliced_energy and self._t >= self._warmup_steps:
            self._num_diverging += 1

        accept_prob = (-delta_energy).exp().clamp(max=1.)
        rand = pyro.sample("rand_t={}".format(self._t), dist.Uniform(scalar_like(accept_prob, 0.),
                                                                     scalar_like(accept_prob, 1.)))
        if rand < accept_prob:
            self._accept_cnt += 1
            z = z_new
            self._cache(z, potential_energy_new, z_grads_new)

        if self._t < self._warmup_steps:
            self._adapter.step(self._t, z, accept_prob)

        self._t += 1

        return z.copy()

    def diagnostics(self):
        return OrderedDict([
            ("step size", "{:.2e}".format(self.step_size)),
            ("acc. rate", "{:.3f}".format(self._accept_cnt / self._t)),
            ("diverging", "{}".format(self._num_diverging)),
        ])
