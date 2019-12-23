""" Slow Feature Analysis

Reference: Wiskott, L. and Sejnowski, T.J., Slow Feature Analysis: Unsupervised
           Learning of Invariances, Neural Computation, 14(4):715-770 (2002)
"""

# Author: Alberto N Escalante B <alberto.escalante@ini.rub.de>
#         xxx  <xxx@ini.rub.de>
#         xxx  <xxx@ini.rub.de>
#         xxx  <xxx@ini.rub.de>
#         xxx  <xxx@ini.rub.de>
# This algorithm has been adapted Modular toolkit for Data Processing (MDP),
# see: http://mdp-toolkit.sourceforge.net/
# License: xxx

from builtins import str
from builtins import range

import warnings
from .sfa_symeig_semidefinite import (symeig_semidefinite_pca,
                                     symeig_semidefinite_reg,
                                     symeig_semidefinite_svd,
                                     symeig_semidefinite_ldl)
from ..utils import check_array
from ..base import BaseEstimator, TransformerMixin
from ..utils.validation import check_is_fitted
from ..utils import check_random_state, as_float_array
import numpy as np

import inspect


def get_symeig(linalg):
    # if we have scipy, check if the version of
    # scipy.linalg.eigh supports the rich interface
    args = inspect.getargspec(linalg.eigh)[0]
    if len(args) > 4:
        # if yes, just wrap it
        from ._symeig import wrap_eigh as symeig
        # config.ExternalDepFound('symeig', 'scipy.linalg.eigh')
    else:
        # either we have numpy, or we have an old scipy
        # we need to use our own rich wrapper
        from ._symeig import _symeig_fake as symeig
        # config.ExternalDepFound('symeig', 'symeig_fake')
    return symeig
symeig = get_symeig(np.linalg)


def refcast(array, dtype):
    """
    Cast the array to dtype only if necessary, otherwise return a reference.
    """
    dtype = np.dtype(dtype)
    if array.dtype == dtype:
        return array
    return array.astype(dtype)


def svd(x, compute_uv = True):
    """Wrap the numpy SVD routine, so that it returns arrays of the correct
    dtype and a SymeigException in case of failures."""
    tc = x.dtype
    try:
        if compute_uv:
            u, s, v = np.linalg.svd(x)
            return refcast(u, tc), refcast(s, tc), refcast(v, tc)
        else:
            s = np.linalg.svd(x, compute_uv=False)
            return refcast(s, tc)
    except np.linalg.LinAlgError as exc:
        raise Exception(str(exc))  # TODO: use specific object type


def _check_roundoff(t, dtype):
    """Check if t is so large that t+1 == t up to 2 precision digits"""
    # limit precision
    limit = 10.**(np.finfo(dtype).precision-2)
    if int(t) >= limit:
        wr = ('You have summed %e entries in the covariance matrix.'
              '\nAs you are using dtype \'%s\', you are '
              'probably getting severe round off'
              '\nerrors. See CovarianceMatrix docstring for more'
              ' information.' % (t, dtype.name))
        warnings.warn(wr, UserWarning)  # Verify, was mdp.MDPWarning


class CovarianceMatrix(object):
    """This class stores an empirical covariance matrix that can be updated
    incrementally. A call to the 'fix' method returns the current state of
    the covariance matrix, the average and the number of observations, and
    resets the internal data.

    Note that the internal sum is a standard __add__ operation. We are not
    using any of the fancy sum algorithms to avoid round off errors when
    adding many numbers.
    """
    def __init__(self, dtype=None, bias=False):
        """If dtype is not defined, it will be inherited from the first
        data bunch received by 'update'.
        All the matrices in this class are set up with the given dtype and
        no upcast is possible.
        If bias is True, the covariance matrix is normalized by dividing
        by T instead of the usual T-1.
        """
        if dtype is None:
            self._dtype = None
        else:
            self._dtype = np.dtype(dtype)
        self._input_dim = None  # will be set in _init_internals
        # covariance matrix, updated during the training phase
        self._cov_mtx = None
        # average, updated during the training phase
        self._avg = None
        # number of observation so far during the training phase
        self._tlen = 0

        self.bias = bias

    def _init_internals(self, x):
        """Init the internal structures.

        The reason this is not done in the constructor is that we want to be
        able to derive the input dimension and the dtype directly from the
        data this class receives.
        """
        # init dtype
        if self._dtype is None:
            self._dtype = x.dtype
        dim = x.shape[1]
        self._input_dim = dim
        type_ = self._dtype
        # init covariance matrix
        self._cov_mtx = np.zeros((dim, dim), type_)
        # init average
        self._avg = np.zeros(dim, type_)

    def update(self, x):
        """Update internal structures.

        Note that no consistency checks are performed on the data (this is
        typically done in the enclosing node).
        """
        if self._cov_mtx is None:
            self._init_internals(x)
        # cast input
        x = refcast(x, self._dtype)
        # update the covariance matrix, the average and the number of
        # observations (try to do everything inplace)
        self._cov_mtx += np.dot(x.T, x)
        self._avg += x.sum(axis=0)
        self._tlen += x.shape[0]

    def fix(self, center=True):
        """Returns a triple containing the covariance matrix, the average and
        the number of observations. The covariance matrix is then reset to
        a zero-state.

        If center is false, the returned matrix is the matrix of the second moments,
        i.e. the covariance matrix of the data without subtracting the mean."""
        # local variables
        type_ = self._dtype
        tlen = self._tlen
        _check_roundoff(tlen, type_)
        avg = self._avg
        cov_mtx = self._cov_mtx

        ##### fix the training variables
        # fix the covariance matrix (try to do everything inplace)
        if self.bias:
            cov_mtx /= tlen
        else:
            cov_mtx /= tlen - 1

        if center:
            avg_mtx = np.outer(avg, avg)
            if self.bias:
                avg_mtx /= tlen*(tlen)
            else:
                avg_mtx /= tlen*(tlen - 1)
            cov_mtx -= avg_mtx

        # fix the average
        avg /= tlen

        ##### clean up
        # covariance matrix, updated during the training phase
        self._cov_mtx = None
        # average, updated during the training phase
        self._avg = None
        # number of observation so far during the training phase
        self._tlen = 0

        return cov_mtx, avg, tlen


SINGULAR_VALUE_MSG = '''
This usually happens if there are redundancies in the (expanded) training data.
There are several ways to deal with this issue:

  - Use more data.

  - Use another solver for the generalized eigenvalue problem.
    The default solver requires the covariance matrix to be strictly positive
    definite. Construct your node with e.g. rank_deficit_method='auto' to use
    a more robust solver that allows positive semidefinite covariance matrix.
    Available values for rank_deficit_method: none, auto, pca, reg, svd, ldl
    See mdp.utils.symeig_semidefinite for details on the available methods.

  - Add noise to the data. This can be done by chaining an additional NoiseNode
    in front of a troublesome SFANode. Noise levels do not have to be high.
    Note:
    You will get a somewhat similar effect by rank_deficit_method='reg'.
    This will be more efficient in execution phase.

  - Run training data through PCA. This can be done by chaining an additional
    PCA node in front of the troublesome SFANode. Use the PCA node to discard
    dimensions within your data with lower variance.
    Note:
    You will get the same result by rank_deficit_method='pca'.
    This will be more efficient in execution phase.
'''

class SFA(BaseEstimator):
    """Extract the slowly varying components from the input data.
    More information about Slow Feature Analysis can be found in
    Wiskott, L. and Sejnowski, T.J., Slow Feature Analysis: Unsupervised
    Learning of Invariances, Neural Computation, 14(4):715-770 (2002).

    **Instance variables of interest**

      ``self.avg``
          Mean of the input data (available after training)

      ``self.sf``
          Matrix of the SFA filters (available after training)

      ``self.d``
          Delta values corresponding to the SFA components (generalized
          eigenvalues). [See the docs of the ``get_eta_values`` method for
          more information]

      ``self.rank_deficit``
          If an SFA solver detects rank deficit in the covariance matrix,
          it stores the count of close-to-zero-eigenvalues as ``rank_deficit``.

    **Special arguments for constructor**

      ``include_last_sample``
          If ``False`` the `train` method discards the last sample in every
          chunk during training when calculating the covariance matrix.
          The last sample is in this case only used for calculating the
          covariance matrix of the derivatives. The switch should be set
          to ``False`` if you plan to train with several small chunks. For
          example we can split a sequence (index is time)::

            x_1 x_2 x_3 x_4

          in smaller parts like this::

            x_1 x_2
            x_2 x_3
            x_3 x_4

          The SFANode will see 3 derivatives for the temporal covariance
          matrix, and the first 3 points for the spatial covariance matrix.
          Of course you will need to use a generator that *connects* the
          small chunks (the last sample needs to be sent again in the next
          chunk). If ``include_last_sample`` was True, depending on the
          generator you use, you would either get::

             x_1 x_2
             x_2 x_3
             x_3 x_4

          in which case the last sample of every chunk would be used twice
          when calculating the covariance matrix, or::

             x_1 x_2
             x_3 x_4

          in which case you loose the derivative between ``x_3`` and ``x_2``.

          If you plan to train with a single big chunk leave
          ``include_last_sample`` to the default value, i.e. ``True``.

          You can even change this behaviour during training. Just set the
          corresponding switch in the `train` method.


      ``rank_deficit_method``
          Possible values: 'none' (default), 'reg', 'pca', 'svd', 'auto'
          If not 'none', the ``stop_train`` method solves the SFA eigenvalue
          problem in a way that is robust against linear redundancies in
          the input data. This would otherwise lead to rank deficit in the
          covariance matrix, which usually yields a
          SymeigException ('Covariance matrices may be singular').
          There are several solving methods implemented:

          reg  - works by regularization
          pca  - works by PCA
          svd  - works by SVD
          ldl  - works by LDL decomposition (requires SciPy >= 1.0)

          auto - (Will be: selects the best-benchmarked method of the above)
                 Currently it simply selects pca.

          Note: If you already received an exception
          SymeigException ('Covariance matrices may be singular')
          you can manually set the solving method for an existing node::

             sfa.set_rank_deficit_method('pca')

          That means,::

             sfa = SFANode(rank_deficit='pca')

          is equivalent to::

             sfa = SFANode()
             sfa.set_rank_deficit_method('pca')

          After such an adjustment you can run ``stop_training()`` again,
          which would save a potentially time-consuming rerun of all
          ``train()`` calls.
    """

    def __init__(self, copy=True, input_dim=None, output_dim=None, dtype=None,
                 include_last_sample=True, rank_deficit_method='none',
                 random_state=None):
        """
        For the ``include_last_sample`` switch have a look at the
        SFANode class docstring.
        """

        # super(SFANode, self).__init__(input_dim, output_dim, dtype)
        # initialize basic attributes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dtype = dtype
        # this var stores at which point in the training sequence we are
        self._train_phase = 0
        # this var is False if the training of the current phase hasn't
        #  started yet, True otherwise
        self._train_phase_started = False
        # this var is False if the complete training is finished
        self._training = True

        self.copy = copy
        self._include_last_sample = include_last_sample
        self.random_state = random_state

        # init two covariance matrices
        # one for the input data
        self._cov_mtx = CovarianceMatrix(dtype)
        # one for the derivatives
        self._dcov_mtx = CovarianceMatrix(dtype)

        # set routine for eigenproblem
        self.set_rank_deficit_method(rank_deficit_method)
        self.rank_threshold = 1e-12
        self.rank_deficit = 0

        # SFA eigenvalues and eigenvectors, will be set after training
        self.d = None
        self.sf = None  # second index for outputs
        self.avg = None
        self._bias = None  # avg multiplied with sf
        self.tlen = None

    def set_rank_deficit_method(self, rank_deficit_method):
        if rank_deficit_method == 'pca':
            self._symeig = symeig_semidefinite_pca
        elif rank_deficit_method == 'reg':
            self._symeig = symeig_semidefinite_reg
        elif rank_deficit_method == 'svd':
            self._symeig = symeig_semidefinite_svd
        elif rank_deficit_method == 'ldl':
            try:
                from scipy.linalg.lapack import dsytrf
            except ImportError:
                err_msg = ("ldl method for solving SFA with rank deficit covariance "
                           "requires at least SciPy 1.0.")
                raise NodeException(err_msg)
            self._symeig = symeig_semidefinite_ldl
        elif rank_deficit_method == 'auto':
            self._symeig = symeig_semidefinite_pca
        elif rank_deficit_method == 'none':
            self._symeig = symeig
        else:
            raise ValueError("Invalid value for rank_deficit_method: %s" \
                    %str(rank_deficit_method))

    def time_derivative(self, x):
        """Compute the linear approximation of the time derivative."""
        # this is faster than a linear_filter or a weave-inline solution
        return x[1:, :]-x[:-1, :]

    def _set_range(self):
        if self.output_dim is not None and (self.input_dim is None or self.output_dim <= self.input_dim):
            # (eigenvalues sorted in ascending order)
            rng = (1, self.output_dim)
        else:
            # otherwise, keep all output components
            rng = None
            self.output_dim = self.input_dim
        return rng

    def _check_train_args(self, x, *args, **kwargs):
        # check that we have at least 2 time samples to
        # compute the update for the derivative covariance matrix
        s = x.shape[0]
        if  s < 2:
            raise TrainingException('Need at least 2 time samples to '
                                    'compute time derivative (%d given)'%s)

    def _train(self, x, include_last_sample=None):
        """
        For the ``include_last_sample`` switch have a look at the
        SFANode class docstring.
        """
        if include_last_sample is None:
            include_last_sample = self._include_last_sample
        # works because x[:None] == x[:]
        last_sample_index = None if include_last_sample else -1

        # update the covariance matrices
        self._cov_mtx.update(x[:last_sample_index, :])
        self._dcov_mtx.update(self.time_derivative(x))

    def _stop_training(self, debug=False):
        ##### request the covariance matrices and clean up
        if hasattr(self, '_dcov_mtx'):
            self.cov_mtx, self.avg, self.tlen = self._cov_mtx.fix()
            del self._cov_mtx
        # do not center around the mean:
        # we want the second moment matrix (centered about 0) and
        # not the second central moment matrix (centered about the mean), i.e.
        # the covariance matrix
        if hasattr(self, '_dcov_mtx'):
            self.dcov_mtx, self.davg, self.dtlen = self._dcov_mtx.fix(center=False)
            del self._dcov_mtx

        rng = self._set_range()

        #### solve the generalized eigenvalue problem
        # the eigenvalues are already ordered in ascending order
        try:
            try:
                # We first try to fulfill the extended signature described
                # in mdp.utils.symeig_semidefinite
                self.d, self.sf = self._symeig(
                        self.dcov_mtx, self.cov_mtx, True, "on", rng,
                        overwrite=(not debug),
                        rank_threshold=self.rank_threshold, dfc_out=self)
            except TypeError:
                self.d, self.sf = self._symeig(
                        self.dcov_mtx, self.cov_mtx, True, "on", rng,
                        overwrite=(not debug))
            d = self.d
            # check that we get only *positive* eigenvalues
            if d.min() < 0:
                err_msg = ("Got negative eigenvalues: %s.\n"
                           "You may either set output_dim to be smaller,\n"
                           "or prepend the SFANode with a PCANode(reduce=True)\n"
                           "or PCANode(svd=True)\n"
                           "or set a rank deficit method, e.g.\n"
                           "create the SFA node with rank_deficit_method='auto'\n"
                           "and try higher values for rank_threshold, e.g. try\n"
                           "your_node.rank_threshold = 1e-10, 1e-8, 1e-6, ..."%str(d))
                raise NodeException(err_msg)
        except SymeigException as exception:
            errstr = (str(exception)+"\n Covariance matrices may be singular.\n"
                    +SINGULAR_VALUE_MSG)
            raise NodeException(errstr)

        if not debug:
            # delete covariance matrix if no exception occurred
            del self.cov_mtx
            del self.dcov_mtx

        # store bias
        self._bias = np.dot(self.avg, self.sf)

    def _execute(self, x, n=None):
        """Compute the output of the slowest functions.
        If 'n' is an integer, then use the first 'n' slowest components."""
        if n:
            sf = self.sf[:, :n]
            bias = self._bias[:n]
        else:
            sf = self.sf
            bias = self._bias
        return np.dot(x, sf) - bias



    def get_eta_values(self, t=1):
        """Return the eta values of the slow components learned during
        the training phase. If the training phase has not been completed
        yet, call `stop_training`.

        The delta value of a signal is a measure of its temporal
        variation, and is defined as the mean of the derivative squared,
        i.e. delta(x) = mean(dx/dt(t)^2).  delta(x) is zero if
        x is a constant signal, and increases if the temporal variation
        of the signal is bigger.

        The eta value is a more intuitive measure of temporal variation,
        defined as
        eta(x) = t/(2*pi) * sqrt(delta(x))
        If x is a signal of length 't' which consists of a sine function
        that accomplishes exactly N oscillations, then eta(x)=N.

        :Parameters:
           t
             Sampling frequency in Hz.

             The original definition in (Wiskott and Sejnowski, 2002)
             is obtained for t = number of training data points, while
             for t=1 (default), this corresponds to the beta-value defined in
             (Berkes and Wiskott, 2005).
        """
        if self.is_training():
            self.stop_training()
        return self._refcast(t / (2 * np.pi) * np.sqrt(self.d))

    # From PCA
    def fit(self, X, y=None):
        """Fit the model with X by extracting the slowest features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples in the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fit(check_array(X))
        self._stop_training()
        return self

    def _fit(self, X):
        """Fit the model to the data X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        Returns
        -------
        X : ndarray, shape (n_samples, n_features)
            The input data, copied, and centered when requested.
        """
        random_state = check_random_state(self.random_state)
        X = np.atleast_2d(as_float_array(X, copy=self.copy))

        # Center data
        #self.mean_ = np.mean(X, axis=0)
        #X -= self.mean_

        self._train(X)


        return X

    def transform(self, X):
        """Apply dimensionality reduction on X.

        X is projected on the first principal components previous extracted
        from a training set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        # check_is_fitted(self, 'mean_')

        X = check_array(X)
        X = self._execute(X)
        return X

    def fit_transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data, where n_samples in the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)

        """
        X = check_array(X)
        X = self._fit(X)
        return np.dot(X, self.components_.T)

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Returns an array X_original whose transform would be X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_components)
            New data, where n_samples in the number of samples
            and n_components is the number of components.

        Returns
        -------
        X_original array-like, shape (n_samples, n_features)

        Notes
        -----
        None yet.
        """
        #check_is_fitted(self, 'mean_')

        X_original = np.dot(y, pinv(self.sf)) + self.avg
        return X_original

if __name__ == "__main__":
    sfa = SFA(input_dim=None, output_dim=3, dtype=None,
              include_last_sample=True, rank_deficit_method='none')
    X = np.random.normal(size=(20, 5))
    sfa.fit(X)
    y = sfa.transform(X)
    print(y)
    print(sfa.d)
    print(sfa.sf[0:3, 0:3])
    print(sfa.avg)
    print(sfa._bias)
    try:
        import mdp
        sfa_node = mdp.nodes.SFANode(output_dim=3,
                                     include_last_sample=True,
                                     rank_deficit_method='none')
        sfa_node.train(X)
        sfa_node.stop_training()
        y_mdp = sfa_node.execute(X)

        # Adjustment of the global feature sign
        y_n = y * np.sign(y[0, :])
        y_mpd_n = y_mdp * np.sign(y_mdp[0, :])
        print(y_mpd_n)
        print(sfa_node.d)
        print(sfa_node.sf[0:3, 0:3])
        print(sfa.avg)
        print(sfa_node._bias)
    except ImportError as ex:
        print('Could not import mdp:', ex)

# Consider the following conde after SFANode has been ported to scikits
# class SFA2Node(SFANode):
#     """Get an input signal, expand it in the space of
#     inhomogeneous polynomials of degree 2 and extract its slowly varying
#     components. The ``get_quadratic_form`` method returns the input-output
#     function of one of the learned unit as a ``QuadraticForm`` object.
#     See the documentation of ``mdp.utils.QuadraticForm`` for additional
#     information.
#
#     More information about Slow Feature Analysis can be found in
#     Wiskott, L. and Sejnowski, T.J., Slow Feature Analysis: Unsupervised
#     Learning of Invariances, Neural Computation, 14(4):715-770 (2002)."""
#
#     def __init__(self, input_dim=None, output_dim=None, dtype=None,
#                  include_last_sample=True, rank_deficit_method='none'):
#         self._expnode = mdp.nodes.QuadraticExpansionNode(input_dim=input_dim,
#                                                          dtype=dtype)
#         super(SFA2Node, self).__init__(input_dim, output_dim, dtype,
#                                        include_last_sample, rank_deficit_method)
#
#     @staticmethod
#     def is_invertible():
#         """Return True if the node can be inverted, False otherwise."""
#         return False
#
#     def _set_input_dim(self, n):
#         self._expnode.input_dim = n
#         self._input_dim = n
#
#     def _train(self, x, include_last_sample=None):
#         # expand in the space of polynomials of degree 2
#         super(SFA2Node, self)._train(self._expnode(x), include_last_sample)
#
#     def _set_range(self):
#         if (self.output_dim is not None) and (
#             self.output_dim <= self._expnode.output_dim):
#             # (eigenvalues sorted in ascending order)
#             rng = (1, self.output_dim)
#         else:
#             # otherwise, keep all output components
#             rng = None
#         return rng
#
#     def _stop_training(self, debug=False):
#         super(SFA2Node, self)._stop_training(debug)
#
#         # set the output dimension if necessary
#         if self.output_dim is None:
#             self.output_dim = self._expnode.output_dim
#
#     def _execute(self, x, n=None):
#         """Compute the output of the slowest functions.
#         If 'n' is an integer, then use the first 'n' slowest components."""
#         return super(SFA2Node, self)._execute(self._expnode(x), n)
#
#     def get_quadratic_form(self, nr):
#         """
#         Return the matrix H, the vector f and the constant c of the
#         quadratic form 1/2 x'Hx + f'x + c that defines the output
#         of the component 'nr' of the SFA node.
#         """
#         if self.sf is None:
#             self._if_training_stop_training()
#
#         sf = self.sf[:, nr]
#         c = -np.dot(self.avg, sf)
#         n = self.input_dim
#         f = sf[:n]
#         h = np.zeros((n, n), dtype=self.dtype)
#         k = n
#         for i in range(n):
#             for j in range(n):
#                 if j > i:
#                     h[i, j] = sf[k]
#                     k = k+1
#                 elif j == i:
#                     h[i, j] = 2*sf[k]
#                     k = k+1
#                 else:
#                     h[i, j] = h[j, i]
#
#         return QuadraticForm(h, f, c, dtype=self.dtype)
