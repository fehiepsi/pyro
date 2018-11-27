from __future__ import absolute_import, division, print_function

from torch.nn import Parameter

import pyro.distributions as dist
from pyro.contrib.gp.util import Parameterized


class GPLVM(Parameterized):
    """
    Gaussian Process Latent Variable Model (GPLVM) model.

    GPLVM is a Gaussian Process model with its train input data is a latent variable.
    This model is useful for dimensional reduction of high dimensional data. Assume the
    mapping from low dimensional latent variable to is a Gaussian Process instance.
    Then the high dimensional data will play the role of train output ``y`` and our
    target is to learn latent inputs which best explain ``y``. For the purpose of
    dimensional reduction, latent inputs should have lower dimensions than ``y``.

    We follows reference [1] to put a unit Gaussian prior to the input and approximate
    its posterior by a multivariate normal distribution with two variational
    parameters: ``X_loc`` and ``X_scale_tril``.

    For example, we can do dimensional reduction on Iris dataset as follows:

        >>> # With y as the 2D Iris data of shape 150x4 and we want to reduce its dimension
        >>> # to a tensor X of shape 150x2, we will use GPLVM.

        .. doctest::
           :hide:

            >>> # Simulating iris data.
            >>> y = torch.stack([dist.Normal(4.8, 0.1).sample((150,)),
            ...                 dist.Normal(3.2, 0.3).sample((150,)),
            ...                 dist.Normal(1.5, 0.4).sample((150,)),
            ...                 dist.Exponential(0.5).sample((150,))])

        >>> # First, define the initial values for X parameter:
        >>> X_init = torch.zeros(150, 2)
        >>> # Then, define a Gaussian Process model with input X_init and output y:
        >>> kernel = gp.kernels.RBF(input_dim=2, lengthscale=torch.ones(2))
        >>> Xu = torch.zeros(20, 2)  # initial inducing inputs of sparse model
        >>> gpmodel = gp.models.SparseGPRegression(X_init, y, kernel, Xu)
        >>> # Finally, wrap gpmodel by GPLVM, optimize, and get a "learned" X
        >>> gplvm = gp.models.GPLVM(gpmodel)
        >>> gp.util.train(gplvm)  # doctest: +SKIP
        >>> X = gplvm.X

    Reference:

    [1] Bayesian Gaussian Process Latent Variable Model
    Michalis K. Titsias, Neil D. Lawrence

    :param ~pyro.contrib.gp.models.model.GPModel base_model: A Pyro Gaussian Process
        model object. Note that ``base_model.X`` will be the initial value for the
        variational parameter ``X_loc``.
    :param str name: Name of this model.
    """
    def __init__(self, base_model):
        super(GPLVM, self).__init__()
        if base_model.X.dim() != 2:
            raise ValueError("GPLVM model only works with 2D latent X, but got "
                             "X.dim() = {}.".format(base_model.X.dim()))
        self.base_model = base_model
        self.X = Parameter(self.base_model.X)
        self.set_prior("X", dist.Normal(0, 1).expand(self.X.shape).independent(self.X.dim()))
        self.autoguide("X", dist.Normal)

    def model(self):
        self.mode = "model"
        self.base_model.set_data(self.X, self.base_model.y)
        self.base_model.model()

    def guide(self):
        self.mode = "guide"
        self.base_model.set_data(self.X, self.base_model.y)
        self.base_model.guide()

    def forward(self, **kwargs):
        """
        Forward method has the same signal as its ``base_model``.
        """
        self.mode = "guide"
        self.base_model.set_data(self.X, self.base_model.y)
        return self.base_model(**kwargs)
