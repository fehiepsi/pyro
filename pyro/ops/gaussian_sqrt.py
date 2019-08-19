import math

import torch
from torch.distributions.utils import lazy_property
from torch.nn.functional import pad

from pyro.distributions.util import broadcast_shape


def triangularize(A):
    """
    Transforms a matrix to a lower triangular matrix such that A @ A.T = f(A) @ f(A).T
    """
    return A.transpose(-1, -2).qr(some=True).R.transpose(-1, -2)


class GaussianS:
    """
    Non-normalized Gaussian distribution.

    This represents an arbitrary semidefinite quadratic function, which can be
    interpreted as a rank-deficient scaled Gaussian distribution. The precision
    matrix may have zero eigenvalues, thus it may be impossible to work
    directly with the covariance matrix.

    :param torch.Tensor log_normalizer: a normalization constant, which is mainly used to keep
        track of normalization terms during contractions.
    :param torch.Tensor info_vec: information vector, which is a scaled version of the mean
        ``info_vec = precision @ mean``. We use this represention to make gaussian contraction
        fast and stable.
    :param torch.Tensor prec_sqrt: square root of precision matrix of this gaussian. We use this
        representation to preserve symmetric and reduce condition numbers.
    """
    def __init__(self, log_normalizer, info_vec, prec_sqrt):
        # NB: using info_vec instead of mean to deal with rank-deficient problem
        assert info_vec.dim() >= 1
        assert prec_sqrt.dim() >= 2
        assert prec_sqrt.shape[-2:] == info_vec.shape[-1:] * 2
        self.log_normalizer = log_normalizer
        self.info_vec = info_vec
        self.prec_sqrt = prec_sqrt

    def dim(self):
        return self.info_vec.size(-1)

    @lazy_property
    def batch_shape(self):
        return broadcast_shape(self.log_normalizer.shape,
                               self.info_vec.shape[:-1],
                               self.prec_sqrt.shape[:-2])

    @lazy_property
    def precision(self):
        return self.prec_sqrt.matmul(self.prec_sqrt.transpose(-2, -1))

    def expand(self, batch_shape):
        n = self.dim()
        log_normalizer = self.log_normalizer.expand(batch_shape)
        info_vec = self.info_vec.expand(batch_shape + (n,))
        prec_sqrt = self.prec_sqrt.expand(batch_shape + (n, n))
        return GaussianS(log_normalizer, info_vec, prec_sqrt)

    def reshape(self, batch_shape):
        n = self.dim()
        log_normalizer = self.log_normalizer.reshape(batch_shape)
        info_vec = self.info_vec.reshape(batch_shape + (n,))
        prec_sqrt = self.prec_sqrt.reshape(batch_shape + (n, n))
        return GaussianS(log_normalizer, info_vec, prec_sqrt)

    def __getitem__(self, index):
        """
        Index into the batch_shape of a Gaussian.
        """
        assert isinstance(index, tuple)
        log_normalizer = self.log_normalizer[index]
        info_vec = self.info_vec[index + (slice(None),)]
        prec_sqrt = self.prec_sqrt[index + (slice(None), slice(None))]
        return GaussianS(log_normalizer, info_vec, prec_sqrt)

    @staticmethod
    def cat(parts, dim=0):
        """
        Concatenate a list of Gaussians along a given batch dimension.
        """
        if dim < 0:
            dim += len(parts[0].batch_shape)
        args = [torch.cat([getattr(g, attr) for g in parts], dim=dim)
                for attr in ["log_normalizer", "info_vec", "prec_sqrt"]]
        return GaussianS(*args)

    def event_pad(self, left=0, right=0):
        """
        Pad along event dimension.
        """
        lr = (left, right)
        log_normalizer = self.log_normalizer
        info_vec = pad(self.info_vec, lr)
        prec_sqrt = pad(self.prec_sqrt, lr + (0, 0))
        return GaussianS(log_normalizer, info_vec, prec_sqrt)

    def event_permute(self, perm):
        """
        Permute along event dimension.
        """
        raise NotImplementedError
        assert isinstance(perm, torch.Tensor)
        assert perm.shape == (self.dim(),)
        info_vec = self.info_vec[..., perm]
        prec_sqrt = self.prec_sqrt[..., perm, :]
        return GaussianS(self.log_normalizer, info_vec, prec_sqrt)

    def __add__(self, other):
        """
        Adds two Gaussians in log-density space.
        """
        assert isinstance(other, GaussianS)
        assert self.dim() == other.dim()
        return GaussianS(self.log_normalizer + other.log_normalizer,
                         self.info_vec + other.info_vec,
                         pad(self.prec_sqrt, (0, other.dim())) + pad(other.prec_sqrt, (self.dim(), 0)))

    def log_density(self, value):
        """
        Evaluate the log density of this Gaussian at a point value::

            -0.5 * value.T @ precision @ value + value.T @ info_vec + log_normalizer

        This is mainly used for testing.
        """
        if value.size(-1) == 0:
            batch_shape = broadcast_shape(value.shape[:-1], self.batch_shape)
            return self.log_normalizer.expand(batch_shape)
        result = value.matmul(self.prec_sqrt)
        result = (-0.5) * result.pow(2).sum(-1)
        result = result + self.info_vec
        result = (value * result).sum(-1)
        return result + self.log_normalizer

    def condition(self, value):
        """
        Condition this Gaussian on a trailing subset of its state.
        This should satisfy::

            g.condition(y).dim() == g.dim() - y.size(-1)

        Note that since this is a non-normalized Gaussian, we include the
        density of ``y`` in the result. Thus :meth:`condition` is similar to a
        ``functools.partial`` binding of arguments::

            left = x[..., :n]
            right = x[..., n:]
            g.log_density(x) == g.condition(right).log_density(left)
        """
        assert isinstance(value, torch.Tensor)
        assert value.size(-1) <= self.info_vec.size(-1)

        n = self.dim() - value.size(-1)
        info_a = self.info_vec[..., :n]
        info_b = self.info_vec[..., n:]
        Psqrt_a = self.prec_sqrt[..., :n, :]
        Psqrt_b = self.prec_sqrt[..., n:, :]
        b = value

        Psqrt_b_b = b.matmul(Psqrt_b)
        info_vec = info_a - Psqrt_a.matmul(Psqrt_b_b)
        prec_sqrt = Psqrt_a
        log_normalizer = (self.log_normalizer +
                          -0.5 * Psqrt_b_b.pow(2).sum(-1) +
                          b.mul(info_b).sum(-1))
        return GaussianS(log_normalizer, info_vec, prec_sqrt)

    def marginalize(self, left=0, right=0):
        """
        Marginalizing out variables on either side of the event dimension::

            g.marginalize(left=n).event_logsumexp() = g.logsumexp()
            g.marginalize(right=n).event_logsumexp() = g.logsumexp()

        and for data ``x``:

            g.condition(x).event_logsumexp()
              = g.marginalize(left=g.dim() - x.size(-1)).log_density(x)
        """
        if left == 0 and right == 0:
            return self
        if left > 0 and right > 0:
            raise NotImplementedError
        n = self.dim()
        n_b = left + right
        a = slice(left, n - right)  # preserved
        b = slice(None, left) if left else slice(n - right, None)

        Psqrt_a = self.prec_sqrt[..., a, :]
        Psqrt_b = self.prec_sqrt[..., b, :]
        # precision = P_aa - Pab @ inv(Pbb) @ Pba
        #           = Psqrt_a @ Psqrt_at - Psqrt_a @ Psqrt_bt @ inv(Psqrt_b @ Psqrt_bt) @ Psqrt_b @ Psqrt_at
        #           = Psqrt_a @ [I - Psqrt_bt @ inv(Psqrt_b @ Psqrt_bt) @ Psqrt_b] @ Psqrt_at
        #           = Psqrt_a @ (I - B) @ Psqrt_at
        #           = Psqrt_a @ (I - B) @ (I - Bt) @ Psqrt_at  (NB: (I - B) @ (I - Bt) = I - B !!)
        # Hence, prec_sqrt = Psqrt_a - Psqrt_a @ B

        # compute Psqrt_bt @ inv(Psqrt_b @ Psqrt_bt) @ Psqrt_b
        # XXX: we can use cholesky of (Psqrt_b @ Psqrt_bt) as in Gaussian implementation
        # but using QR here seems more stable
        Psqrt_b_tril = triangularize(Psqrt_b)
        B_sqrt = Psqrt_b.triangular_solve(Psqrt_b_tril, upper=False).solution
        B = B_sqrt.transpose(-1, -2).matmul(B_sqrt)
        Psqrt_a_B = Psqrt_a.matmul(B)
        prec_sqrt = Psqrt_a - Psqrt_a_B

        info_a = self.info_vec[..., a]
        info_b = self.info_vec[..., b]
        b_tmp = info_b.unsqueeze(-1).triangular_solve(Psqrt_b_tril, upper=False).solution
        info_vec = info_a - Psqrt_a_B.matmul(b_tmp).squeeze(-1)

        log_normalizer = (self.log_normalizer +
                          0.5 * n_b * math.log(2 * math.pi) -
                          0.5 * Psqrt_b_tril.diagonal(dim1=-2, dim2=-1).pow(2).log().sum(-1) +
                          0.5 * b_tmp.squeeze(-1).pow(2).sum(-1))
        return GaussianS(log_normalizer, info_vec, prec_sqrt)

    def event_logsumexp(self):
        """
        Integrates out all latent state (i.e. operating on event dimensions).
        """
        n = self.dim()
        P_sqrt = triangularize(self.prec_sqrt)
        P_sqrt_u = self.info_vec.unsqueeze(-1).triangular_solve(P_sqrt, upper=False).solution.squeeze(-1)
        u_P_u = P_sqrt_u.pow(2).sum(-1)
        return (self.log_normalizer + 0.5 * n * math.log(2 * math.pi) + 0.5 * u_P_u -
                0.5 * P_sqrt.diagonal(dim1=-2, dim2=-1).pow(2).log().sum(-1))


class AffineNormalS:
    """
    Represents a conditional diagonal normal distribution over a random
    variable ``Y`` whose mean is an affine function of a random variable ``X``.
    The likelihood of ``X`` is thus::

        AffineNormal(matrix, loc, scale).condition(y).log_density(x)

    which is equivalent to::

        Normal(x @ matrix + loc, scale).to_event(1).log_prob(y)

    :param torch.Tensor matrix: A transformation from ``X`` to ``Y``.
        Should have rightmost shape ``(x_dim, y_dim)``.
    :param torch.Tensor loc: A constant offset for ``Y``'s mean.
        Should have rightmost shape ``(y_dim,)``.
    :param torch.Tensor scale: Standard deviation for ``Y``.
        Should have rightmost shape ``(y_dim,)``.
    """
    def __init__(self, matrix, loc, scale):
        assert loc.shape == scale.shape
        x_dim, y_dim = matrix.shape[-2:]
        self.matrix = matrix
        self.loc = loc
        self.scale = scale

    def condition(self, value):
        """
        Condition on a ``Y`` value.

        :param torch.Tensor value: A value of ``Y``.
        :return Gaussian: A gaussian likelihood over ``X``.
        """
        assert value.size(-1) == self.loc.size(-1)
        prec_sqrt = self.matrix / self.scale.unsqueeze(-2)
        delta = (value - self.loc) / self.scale
        info_vec = prec_sqrt.matmul(delta.unsqueeze(-1)).squeeze(-1)
        log_normalizer = (-0.5 * self.loc.size(-1) * math.log(2 * math.pi)
                          - 0.5 * delta.pow(2).sum(-1) - self.scale.log().sum(-1))
        return GaussianS(log_normalizer, info_vec, prec_sqrt)


def _scale_tril_to_prec_sqrt(L):
    # NB: prec_sqrt here is a upper triangular matrix. We can use torch.flip
    # to create a lower triangular matrix, but it is not necessary.
    identity = torch.eye(L.size(-1), device=L.device, dtype=L.dtype)
    return identity.triangular_solve(L, upper=False).solution.tranpose(-1, -2)


def mvn_to_gaussianS(mvn):
    """
    Convert a MultivaiateNormal distribution to a Gaussian.

    :param ~torch.distributions.MultivariateNormal mvn: A multivariate normal distribution.
    :return: An equivalent Gaussian object.
    :rtype: ~pyro.ops.gaussian.Gaussian
    """
    assert isinstance(mvn, torch.distributions.MultivariateNormal)
    n = mvn.loc.size(-1)
    prec_sqrt = _scale_tril_to_prec_sqrt(mvn.scale_tril)
    info_vec = prec_sqrt.matmul(prec_sqrt.transpose(-2, -1)).matmul(mvn.loc.unsqueeze(-1)).squeeze(-1)
    log_normalizer = (-0.5 * n * math.log(2 * math.pi) +
                      -0.5 * (info_vec * mvn.loc).sum(-1) -
                      mvn.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1))
    return GaussianS(log_normalizer, info_vec, prec_sqrt)


def matrix_and_mvn_to_gaussianS(matrix, mvn):
    """
    Convert a noisy affine function to a Gaussian. The noisy affine function is defined as::

        y = x @ matrix + mvn.sample()

    :param ~torch.Tensor matrix: A matrix with rightmost shape ``(x_dim, y_dim)``.
    :param ~torch.distributions.MultivariateNormal mvn: A multivariate normal distribution.
    :return: A Gaussian with broadcasted batch shape and ``.dim() == x_dim + y_dim``.
    :rtype: ~pyro.ops.gaussian.Gaussian
    """
    assert (isinstance(mvn, torch.distributions.MultivariateNormal) or
            (isinstance(mvn, torch.distributions.Independent) and
             isinstance(mvn.base_dist, torch.distributions.Normal)))
    assert isinstance(matrix, torch.Tensor)
    x_dim, y_dim = matrix.shape[-2:]
    assert mvn.event_shape == (y_dim,)
    batch_shape = broadcast_shape(matrix.shape[:-2], mvn.batch_shape)
    matrix = matrix.expand(batch_shape + (x_dim, y_dim))
    mvn = mvn.expand(batch_shape)

    # Handle diagonal normal distributions as an efficient special case.
    if isinstance(mvn, torch.distributions.Independent):
        return AffineNormalS(matrix, mvn.base_dist.loc, mvn.base_dist.scale)

    y_gaussian = mvn_to_gaussianS(mvn)
    prec_sqrt = torch.cat([-matrix.matmul(y_gaussian.prec_sqrt), y_gaussian.prec_sqrt], -2)
    info_y = y_gaussian.info_vec
    info_x = -matrix.matmul(info_y.unsqueeze(-1)).squeeze(-1)
    info_vec = torch.cat([info_x, info_y], -1)
    log_normalizer = y_gaussian.log_normalizer

    result = GaussianS(log_normalizer, info_vec, prec_sqrt)
    assert result.batch_shape == batch_shape
    assert result.dim() == x_dim + y_dim
    return result


def gaussian_tensordotS(x, y, dims=0):
    """
    Computes the integral over two gaussians:

        `(x @ y)(a,c) = log(integral(exp(x(a,b) + y(b,c)), b))`,

    where `x` is a gaussian over variables (a,b), `y` is a gaussian over variables
    (b,c), (a,b,c) can each be sets of zero or more variables, and `dims` is the size of b.

    :param x: a Gaussian instance
    :param y: a Gaussian instance
    :param dims: number of variables to contract
    """
    assert isinstance(x, GaussianS)
    assert isinstance(y, GaussianS)
    na = x.dim() - dims
    nb = dims
    nc = y.dim() - dims
    assert na >= 0
    assert nb >= 0
    assert nc >= 0

    # TODO: avoid broadcasting
    device = x.info_vec.device
    perm = torch.cat([
        torch.arange(na, device=device),
        torch.arange(x.dim(), x.dim() + nc, device=device),
        torch.arange(na, x.dim(), device=device)])
    return (x.event_pad(right=nc) + y.event_pad(left=na)).event_permute(perm).marginalize(right=nb)
    """
    Paa, Pba, Pbb = x.precision[..., :na, :na], x.precision[..., na:, :na], x.precision[..., na:, na:]
    Qbb, Qbc, Qcc = y.precision[..., :nb, :nb], y.precision[..., :nb, nb:], y.precision[..., nb:, nb:]
    xa, xb = x.info_vec[..., :na], x.info_vec[..., na:]  # x.precision @ x.mean
    yb, yc = y.info_vec[..., :nb], y.info_vec[..., nb:]  # y.precision @ y.mean

    precision = pad(Paa, (0, nc, 0, nc)) + pad(Qcc, (na, 0, na, 0))
    info_vec = pad(xa, (0, nc)) + pad(yc, (na, 0))
    log_normalizer = x.log_normalizer + y.log_normalizer
    if nb > 0:
        B = pad(Pba, (0, nc)) + pad(Qbc, (na, 0))
        b = xb + yb

        # Pbb + Qbb needs to be positive definite, so that we can malginalize out `b` (to have a finite integral)
        L = torch.cholesky(Pbb + Qbb)
        LinvB = torch.triangular_solve(B, L, upper=False)[0]
        LinvBt = LinvB.transpose(-2, -1)
        Linvb = torch.triangular_solve(b.unsqueeze(-1), L, upper=False)[0]

        precision = precision - torch.matmul(LinvBt, LinvB)
        # NB: precision might not be invertible for getting mean = precision^-1 @ info_vec
        if na + nc > 0:
            info_vec = info_vec - torch.matmul(LinvBt, Linvb).squeeze(-1)
        logdet = torch.diagonal(L, dim1=-2, dim2=-1).log().sum(-1)
        diff = 0.5 * nb * math.log(2 * math.pi) + 0.5 * Linvb.squeeze(-1).pow(2).sum(-1) - logdet
        log_normalizer = log_normalizer + diff

    return Gaussian(log_normalizer, info_vec, precision)
    """
