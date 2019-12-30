import math

import torch
from torch.distributions.utils import lazy_property
from torch.nn.functional import pad

from pyro.distributions.util import broadcast_shape
from pyro.ops.gamma_gaussian import Gamma


def absolute_central_moment_matching(st, new_df):
    """
    Approximates a StudentT by another one with different degree of freedom.
    """
    


class StudentT:
    """
    Non-normalized StudentT distribution:

        StudentT(x) ~ Integral(p(x, s), s)

    where

        p(x, s) = (0.5 * df + 0.5 * rank - 1) * log(s) - 0.5 * df * s
                  - s * 0.5 * info_vec.T @ inv(precision) @ info_vec
                  - s * 0.5 * x.T @ precision @ x + s * x.T @ info_vec.

    The `s` variable plays the role of a mixing variable such that

        p(x | s) ~ Gaussian(s * info_vec, s * precision).

    Conditioned on `s`, this represents an arbitrary semidefinite quadratic function,
    which can be interpreted as a rank-deficient Gaussian distribution.
    The precision matrix may have zero eigenvalues, thus it may be impossible
    to work directly with the covariance matrix.

    :param torch.Tensor log_normalizer: a normalization constant, which is mainly used to keep
        track of normalization terms during contractions.
    :param torch.Tensor info_vec: information vector, which is a scaled version of the mean
        ``info_vec = precision @ mean``. We use this represention to make gaussian contraction
        fast and stable.
    :param torch.Tensor precision: precision matrix of this gaussian.
    :param torch.Tensor df: degree of freedom.
    :param torch.Tensor rank: rank of the precision matrix.
    """
    def __init__(self, log_normalizer, info_vec, precision, df, rank):
        # NB: using info_vec instead of mean to deal with rank-deficient problem
        assert info_vec.dim() >= 1
        assert precision.dim() >= 2
        assert precision.shape[-2:] == info_vec.shape[-1:] * 2
        self.log_normalizer = log_normalizer
        self.info_vec = info_vec
        self.precision = precision
        self.df = df
        self.rank = rank

    def dim(self):
        return self.info_vec.size(-1)

    @lazy_property
    def batch_shape(self):
        return broadcast_shape(self.log_normalizer.shape,
                               self.info_vec.shape[:-1],
                               self.precision.shape[:-2],
                               self.df.shape,
                               self.d_eff.shape)

    def expand(self, batch_shape):
        n = self.dim()
        log_normalizer = self.log_normalizer.expand(batch_shape)
        info_vec = self.info_vec.expand(batch_shape + (n,))
        precision = self.precision.expand(batch_shape + (n, n))
        df = self.df.expand(batch_shape)
        rank = self.rank.expand(batch_shape)
        return StudentT(log_normalizer, info_vec, precision, df, rank)

    def reshape(self, batch_shape):
        n = self.dim()
        log_normalizer = self.log_normalizer.reshape(batch_shape)
        info_vec = self.info_vec.reshape(batch_shape + (n,))
        precision = self.precision.reshape(batch_shape + (n, n))
        df = self.df.reshape(batch_shape)
        rank = self.d_eff.reshape(batch_shape)
        return StudentT(log_normalizer, info_vec, precision, df, rank)

    def __getitem__(self, index):
        """
        Index into the batch_shape of a GammaGaussian.
        """
        assert isinstance(index, tuple)
        log_normalizer = self.log_normalizer[index]
        info_vec = self.info_vec[index + (slice(None),)]
        precision = self.precision[index + (slice(None), slice(None))]
        df = self.df[index]
        rank = self.df[index]
        return StudentT(log_normalizer, info_vec, precision, df, rank)

    @staticmethod
    def cat(parts, dim=0):
        """
        Concatenate a list of GammaGaussians along a given batch dimension.
        """
        if dim < 0:
            dim += len(parts[0].batch_shape)
        args = [torch.cat([getattr(g, attr) for g in parts], dim=dim)
                for attr in ["log_normalizer", "info_vec", "precision", "df", "rank"]]
        return StudentT(*args)

    def event_pad(self, left=0, right=0):
        """
        Pad along event dimension.
        """
        lr = (left, right)
        info_vec = pad(self.info_vec, lr)
        precision = pad(self.precision, lr + lr)
        # by keeping the same rank, we obtain the same StudentT instance
        return StudentT(self.log_normalizer, info_vec, precision, self.df, self.rank)

    def event_permute(self, perm):
        """
        Permute along event dimension.
        """
        assert isinstance(perm, torch.Tensor)
        assert perm.shape == (self.dim(),)
        info_vec = self.info_vec[..., perm]
        precision = self.precision[..., perm][..., perm, :]
        return StudentT(self.log_normalizer, info_vec, precision, self.df, self.rank)

    def nonexact_add(self, other):
        """
        Approximates the sum of two StudentT in log-density space.
        """
        assert isinstance(other, StudentT)
        assert self.dim() == other.dim()
        assert self.rank == other.rank
        df = torch.min(self.df, other.df)
        self_st = absolute_central_moment_matching(self, df)
        other_st = absolute_central_moment_matching(other, df)
        # TODO: deal with the rank???
        rank = torch.max(self.rank, other.rank)
        return StudentT(self_st.log_normalizer + other_st.log_normalizer,
                        self_st.info_vec + other_st.info_vec,
                        self_st.precision + other_st.precision,
                        df,
                        rank)

    def log_density(self, value):
        """
        Evaluate the log density of this StudentT at a point value.

        This is mainly used for testing.
        """
        batch_shape = broadcast_shape(value.shape[:-1], self.batch_shape)
        log_normalizer = self.log_normalizer.expand(batch_shape)
        # compute posterior of mixing variable, then marginalize it
        rate = 0.5 * self.df
        concentration = rate + 0.5 * self.rank
        if value.size(-1) > 0:  # nondegenerate case
            chol_P = self.precision.cholesky()
            chol_P_solve_u = self.info_vec.unsqueeze(-1).triangular_solve(chol_P, upper=False).solution.squeeze(-1)
            u_Pinv_u = chol_P_solve_u.pow(2).sum(-1)
            u_x = (value * self.info_vec).sum(-1)
            P_x = self.precision.matmul(value.unsqueeze(-1)).squeeze(-1)
            rate = rate + 0.5 * u_Pinv_u + ((0.5 * P_x - self.info_vec) * value).sum(-1)
        return Gamma(log_normalizer, concentration, rate).logsumexp()

    def condition(self, value):
        """
        Condition on a trailing subset of its state.
        This should satisfy::

            g.condition(y).dim() == g.dim() - y.size(-1)

        Note that since this is a non-normalized StudentT, we include the
        density of ``y`` in the result. Thus :meth:`condition` is similar to a
        ``functools.partial`` binding of arguments::

            left = x[..., :n]
            right = x[..., n:]
            g.log_density(x, s) == g.condition(right).log_density(left, s)
        """
        assert isinstance(value, torch.Tensor)
        assert value.size(-1) <= self.info_vec.size(-1)

        n = self.dim() - value.size(-1)
        info_a = self.info_vec[..., :n]
        info_b = self.info_vec[..., n:]
        P_aa = self.precision[..., :n, :n]
        P_ab = self.precision[..., :n, n:]
        P_bb = self.precision[..., n:, n:]
        b = value

        info_vec = info_a - P_ab.matmul(b.unsqueeze(-1)).squeeze(-1)
        precision = P_aa

        log_normalizer = self.log_normalizer
        df = self.df + value.size(-1)
        rank = self.rank - value.size(-1)
        # TODO: rescale info_vec, precision
        # beta = self.beta + 0.5 * P_bb.matmul(b.unsqueeze(-1)).squeeze(-1).mul(b).sum(-1) - b.mul(info_b).sum(-1)
        return GammaGaussian(log_normalizer, info_vec, precision, df, rank)

    def marginalize(self, left=0, right=0):
        """
        Marginalizing out variables on either side of the event dimension::

            g.marginalize(left=n).event_logsumexp() = g.event_logsumexp()
            g.marginalize(right=n).event_logsumexp() = g.event_logsumexp()

        and for data ``x``:

            g.condition(x).event_logsumexp().log_density(s)
              = g.marginalize(left=g.dim() - x.size(-1)).log_density(x, s)
        """
        # NB: the easiest way to think about this process is to consider GammaGaussian
        # as a Gaussian with precision and info_vec scaled by `s`.
        if left == 0 and right == 0:
            return self
        if left > 0 and right > 0:
            raise NotImplementedError
        n = self.dim()
        n_b = left + right
        a = slice(left, n - right)  # preserved
        b = slice(None, left) if left else slice(n - right, None)

        P_aa = self.precision[..., a, a]
        P_ba = self.precision[..., b, a]
        P_bb = self.precision[..., b, b]
        P_b = P_bb.cholesky()
        P_a = P_ba.triangular_solve(P_b, upper=False).solution
        P_at = P_a.transpose(-1, -2)
        precision = P_aa - P_at.matmul(P_a)

        info_a = self.info_vec[..., a]
        info_b = self.info_vec[..., b]
        b_tmp = info_b.unsqueeze(-1).triangular_solve(P_b, upper=False).solution
        info_vec = info_a
        if n_b < n:
            info_vec = info_vec - P_at.matmul(b_tmp).squeeze(-1)

        df = self.df
        rank = self.rank - n_b
        # TODO: rescale precision, info_vec
        # beta = self.beta - 0.5 * b_tmp.squeeze(-1).pow(2).sum(-1)
        log_normalizer = (self.log_normalizer +
                          0.5 * n_b * math.log(2 * math.pi) -
                          P_b.diagonal(dim1=-2, dim2=-1).log().sum(-1))
        return GammaGaussian(log_normalizer, info_vec, precision, self.df, beta)

    def event_logsumexp(self):
        """
        Integrates out all latent state (i.e. operating on event dimensions).
        """
        # It is easier to work with GammaGaussian representation, where we can integrate
        # x variable first, then integrate s variable later.
        # Consider GammaGaussian as a Gaussian with precision = s * precision, info_vec = s * info_vec,
        # marginalize x variable, we get
        #   logsumexp(s) = (0.5 * df + 0.5 * rank - 1) * log(s) - 0.5 * df * s + 0.5 n * log(2 pi) + \
        #       0.5 s * uPu - 0.5 * log|P| - 0.5 n * log(s)
        chol_P = self.precision.cholesky()
        chol_P_u = self.info_vec.unsqueeze(-1).triangular_solve(chol_P, upper=False).solution.squeeze(-1)
        u_P_u = chol_P_u.pow(2).sum(-1)
        concentration = self.alpha + 0.5 * (self.rank - self.dim())
        rate = 0.5 * df - 0.5 * u_P_u
        log_normalizer = self.log_normalizer + 0.5 * n * math.log(2 * math.pi)
        log_normalizer = log_normalizer - chol_P.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return Gamma(log_normalizer, concentration, rate).logsumexp()


def mvt_to_studentt(mvt):
    pass


def matrix_and_mvt_to_studentt(matrix, mvt):
    """
    Convert a noisy affine function to a GammaGaussian, where the noise precision
    is scaled by an auxiliary variable `s`. The noisy affine function (conditioned
    on `s`) is defined as::

        y = x @ matrix + MVN(mvn.loc, precision=s * mvn.precision).sample()

    :param ~torch.Tensor matrix: A matrix with rightmost shape ``(x_dim, y_dim)``.
    :param ~pyro.distributions.MultivariateNormal mvn: A multivariate normal distribution.
    :return: A GammaGaussian with broadcasted batch shape and ``.dim() == x_dim + y_dim``.
    :rtype: ~pyro.ops.gaussian_gamma.GammaGaussian
    """
    # TODO: ...
    assert isinstance(mvn, torch.distributions.MultivariateNormal)
    assert isinstance(matrix, torch.Tensor)
    x_dim, y_dim = matrix.shape[-2:]
    assert mvn.event_shape == (y_dim,)
    batch_shape = broadcast_shape(matrix.shape[:-2], mvn.batch_shape)
    matrix = matrix.expand(batch_shape + (x_dim, y_dim))
    mvn = mvn.expand(batch_shape)

    P_yy = mvn.precision_matrix
    neg_P_xy = matrix.matmul(P_yy)
    P_xy = -neg_P_xy
    P_yx = P_xy.transpose(-1, -2)
    P_xx = neg_P_xy.matmul(matrix.transpose(-1, -2))
    precision = torch.cat([torch.cat([P_xx, P_xy], -1),
                           torch.cat([P_yx, P_yy], -1)], -2)
    info_y = P_yy.matmul(mvn.loc.unsqueeze(-1)).squeeze(-1)
    info_x = -matrix.matmul(info_y.unsqueeze(-1)).squeeze(-1)
    info_vec = torch.cat([info_x, info_y], -1)
    log_normalizer = -0.5 * y_dim * math.log(2 * math.pi) - mvn.scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    alpha = matrix.new_tensor(0.5 * y_dim)
    beta = 0.5 * (info_y * mvn.loc).sum(-1)

    result = GammaGaussian(log_normalizer, info_vec, precision, alpha, beta)
    assert result.batch_shape == batch_shape
    assert result.dim() == x_dim + y_dim
    return result


def studentt_tensordot(x, y, dims=0):
    """
    Computes the integral over two GammaGaussians:

        `(x @ y)((a,c),s) = log(integral(exp(x((a,b),s) + y((b,c),s)), b))`,

    where `x` is a gaussian over variables (a,b), `y` is a gaussian over variables
    (b,c), (a,b,c) can each be sets of zero or more variables, and `dims` is the size of b.

    :param x: a GammaGaussian instance
    :param y: a GammaGaussian instance
    :param dims: number of variables to contract
    """
    assert isinstance(x, GammaGaussian)
    assert isinstance(y, GammaGaussian)
    na = x.dim() - dims
    nb = dims
    nc = y.dim() - dims
    assert na >= 0
    assert nb >= 0
    assert nc >= 0

    device = x.info_vec.device
    perm = torch.cat([
        torch.arange(na, device=device),
        torch.arange(x.dim(), x.dim() + nc, device=device),
        torch.arange(na, x.dim(), device=device)])
    return (x.event_pad(right=nc).nonexact_add(y.event_pad(left=na))).event_permute(perm).marginalize(right=nb)
