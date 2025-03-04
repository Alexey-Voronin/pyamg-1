"""Method to create pre and post-smoothers on the levels of a MultilevelSolver.

The setup_smoother_name functions are helper functions for
parsing user input and assigning each level the appropriate smoother for
the functions in 'change_smoothers'.

The standard interface is

Parameters
----------
lvl : multilevel level
    the level in the hierarchy for which to assign a smoother
iterations : int
    how many smoother iterations
optional_params : dict
    optional params specific for each method such as omega or sweep

Returns
-------
Function pointer for the appropriate relaxation method for level=lvl

Examples
--------
See change_smoothers above
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator
from ..util.utils import scale_rows, get_block_diag, get_diagonal
from ..util.linalg import approximate_spectral_radius
from ..krylov import gmres, cgne, cgnr, cg

from . import relaxation
from .chebyshev import chebyshev_polynomial_coefficients


# Default relaxation parameters and list of by-definition
# symmetric relaxation schemes, e.g. Jacobi.
DEFAULT_SWEEP = 'forward'
DEFAULT_NITER = 1
SYMMETRIC_RELAXATION = ['jacobi', 'richardson', 'block_jacobi',
                        'jacobi_ne', 'chebyshev', 'jacobi_krylov', None]


def _unpack_arg(v):
    if isinstance(v, tuple):
        return v[0], v[1]
    return v, {}


def change_smoothers(ml, presmoother, postsmoother):
    """Initialize pre and post smoothers.

    Initialize pre- and post- smoothers throughout a MultilevelSolver, with
    the option of having different smoothers at different levels

    For each level of the MultilevelSolver 'ml' (except the coarsest level),
    initialize the .presmoother() and .postsmoother() methods used in the
    multigrid cycle.

    Parameters
    ----------
    ml : pyamg multilevel hierarchy
        Data structure that stores the multigrid hierarchy.
    presmoother : None, string, tuple, list
        presmoother can be (1) the name of a supported smoother, e.g.
        "gauss_seidel", (2) a tuple of the form ('method','opts') where
        'method' is the name of a supported smoother and 'opts' a dict of
        keyword arguments to the smoother, or (3) a list of instances of
        options 1 or 2.  See the Examples section for illustrations of the
        format.

        If presmoother is a list, presmoother[i] determines the smoothing
        strategy for level i.  Else, presmoother defines the same strategy
        for all levels.

        If len(presmoother) < len(ml.levels), then
        presmoother[-1] is used for all remaining levels

        If len(presmoother) > len(ml.levels), then
        the remaining smoothing strategies are ignored

    postsmoother : string, tuple, list
        Defines postsmoother in identical fashion to presmoother

    Returns
    -------
    ml changed in place
    ml.levels[i].presmoother   <===  presmoother[i]
    ml.levels[i].postsmoother  <===  postsmoother[i]
    ml.symmetric_smoothing is marked True/False depending on whether
        the smoothing scheme is symmetric.

    Notes
    -----
    - Parameter 'omega' of the Jacobi, Richardson, and jacobi_ne
      methods is scaled by the spectral radius of the matrix on
      each level.  Therefore 'omega' should be in the interval (0,2).
    - Parameter 'withrho' (default: True) controls whether the omega is
      rescaled by the spectral radius in jacobi, block_jacobi, and jacobi_ne
    - By initializing the smoothers after the hierarchy has been setup, allows
      for "algebraically" directed relaxation, such as strength_based_schwarz,
      which uses only the strong connections of a degree-of-freedom to define
      overlapping regions
    - Available smoother methods::

        gauss_seidel
        block_gauss_seidel
        jacobi
        block_jacobi
        richardson
        sor
        chebyshev
        gauss_seidel_nr
        gauss_seidel_ne
        jacobi_ne
        cg
        gmres
        cgne
        cgnr
        schwarz
        strength_based_schwarz
        None

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.aggregation import smoothed_aggregation_solver
    >>> from pyamg.relaxation.smoothing import change_smoothers
    >>> from pyamg.util.linalg import norm
    >>> import numpy as np
    >>> A = poisson((10,10), format='csr')
    >>> b = np.random.rand(A.shape[0],)
    >>> ml = smoothed_aggregation_solver(A, max_coarse=10)
    >>> # Set all levels to use gauss_seidel's defaults
    >>> smoothers = 'gauss_seidel'
    >>> change_smoothers(ml, presmoother=smoothers, postsmoother=smoothers)
    >>> residuals=[]
    >>> x = ml.solve(b, tol=1e-8, residuals=residuals)
    >>> # Set all levels to use three iterations of gauss_seidel's defaults
    >>> smoothers = ('gauss_seidel', {'iterations' : 3})
    >>> change_smoothers(ml, presmoother=smoothers, postsmoother=None)
    >>> residuals=[]
    >>> x = ml.solve(b, tol=1e-8, residuals=residuals)
    >>> # Set level 0 to use gauss_seidel's defaults, and all
    >>> # subsequent levels to use 5 iterations of cgnr
    >>> smoothers = ['gauss_seidel', ('cgnr', {'maxiter' : 5})]
    >>> change_smoothers(ml, presmoother=smoothers, postsmoother=smoothers)
    >>> residuals=[]
    >>> x = ml.solve(b, tol=1e-8, residuals=residuals)

    """
    ml.symmetric_smoothing = True

    # interpret arguments into list
    if isinstance(presmoother, (str, tuple)) or (presmoother is None):
        presmoother = [presmoother]
    elif not isinstance(presmoother, list):
        raise ValueError('Unrecognized presmoother')

    if isinstance(postsmoother, (str, tuple)) or (postsmoother is None):
        postsmoother = [postsmoother]
    elif not isinstance(postsmoother, list):
        raise ValueError('Unrecognized postsmoother')

    # set ml.levels[i].presmoother = presmoother[i],
    #     ml.levels[i].postsmoother = postsmoother[i]
    fn1 = None      # Predefine to keep scope beyond first loop
    fn2 = None
    kwargs1 = {}
    kwargs2 = {}
    min_len = min(len(presmoother), len(postsmoother), len(ml.levels[:-1]))
    # same = (len(presmoother) == len(postsmoother))
    for i in range(0, min_len):
        # unpack presmoother[i]
        fn1, kwargs1 = _unpack_arg(presmoother[i])
        # get function handle
        try:
            setup_presmoother = _setup_call(str(fn1).lower())
        except NameError as e:
            raise NameError(f'Invalid presmoother method: {fn1}') from e

        ml.levels[i].presmoother = setup_presmoother(ml.levels[i], **kwargs1)

        # unpack postsmoother[i]
        fn2, kwargs2 = _unpack_arg(postsmoother[i])
        # get function handle
        try:
            setup_postsmoother = _setup_call(str(fn2).lower())
        except NameError as e:
            raise NameError(f'Invalid postsmoother method: {fn2}') from e
        ml.levels[i].postsmoother = setup_postsmoother(ml.levels[i], **kwargs2)

        # Check if symmetric smoothing scheme
        if 'iterations' in kwargs1:
            it1 = kwargs1['iterations']
        else:
            it1 = DEFAULT_NITER

        if 'iterations' in kwargs2:
            it2 = kwargs2['iterations']
        else:
            it2 = DEFAULT_NITER

        if (fn1 != fn2) or (it1 != it2):
            ml.symmetric_smoothing = False
        elif fn1 not in SYMMETRIC_RELAXATION:
            if 'sweep' in kwargs1:
                sweep1 = kwargs1['sweep']
            else:
                sweep1 = DEFAULT_SWEEP

            if 'sweep' in kwargs2:
                sweep2 = kwargs2['sweep']
            else:
                sweep2 = DEFAULT_SWEEP

            if (sweep1, sweep2) not in (('forward', 'backward'),
                                        ('backward', 'forward'),
                                        ('symmetric', 'symmetric')):
                ml.symmetric_smoothing = False

    if len(presmoother) < len(postsmoother):
        mid_len = min(len(postsmoother), len(ml.levels[:-1]))
        for i in range(min_len, mid_len):
            # Set up presmoother
            ml.levels[i].presmoother = setup_presmoother(ml.levels[i], **kwargs1)

            # unpack postsmoother[i]
            fn2, kwargs2 = _unpack_arg(postsmoother[i])
            # get function handle
            try:
                setup_postsmoother = _setup_call(str(fn2).lower())
            except NameError as e:
                raise NameError(f'Invalid postsmoother method: {fn2}') from e

            ml.levels[i].postsmoother = setup_postsmoother(ml.levels[i], **kwargs2)

            # Check if symmetric smoothing scheme
            if 'iterations' in kwargs1:
                it1 = kwargs1['iterations']
            else:
                it1 = DEFAULT_NITER

            if 'iterations' in kwargs2:
                it2 = kwargs2['iterations']
            else:
                it2 = DEFAULT_NITER

            if (fn1 != fn2) or (it1 != it2):
                ml.symmetric_smoothing = False
            elif fn1 not in SYMMETRIC_RELAXATION:
                if 'sweep' in kwargs1:
                    sweep1 = kwargs1['sweep']
                else:
                    sweep1 = DEFAULT_SWEEP

                if 'sweep' in kwargs2:
                    sweep2 = kwargs2['sweep']
                else:
                    sweep2 = DEFAULT_SWEEP

                if (sweep1, sweep2) not in (('forward', 'backward'),
                                            ('backward', 'forward'),
                                            ('symmetric', 'symmetric')):
                    ml.symmetric_smoothing = False

    elif len(presmoother) > len(postsmoother):
        mid_len = min(len(presmoother), len(ml.levels[:-1]))
        for i in range(min_len, mid_len):
            # unpack presmoother[i]
            fn1, kwargs1 = _unpack_arg(presmoother[i])
            # get function handle
            try:
                setup_presmoother = _setup_call(str(fn1).lower())
            except NameError as e:
                raise NameError(f'Invalid presmoother method: {fn1}') from e

            ml.levels[i].presmoother = setup_presmoother(ml.levels[i], **kwargs1)

            # Set up postsmoother
            ml.levels[i].postsmoother = setup_postsmoother(ml.levels[i], **kwargs2)

            # Check if symmetric smoothing scheme
            if 'iterations' in kwargs1:
                it1 = kwargs1['iterations']
            else:
                it1 = DEFAULT_NITER

            if 'iterations' in kwargs2:
                it2 = kwargs2['iterations']
            else:
                it2 = DEFAULT_NITER

            if (fn1 != fn2) or (it1 != it2):
                ml.symmetric_smoothing = False
            elif fn1 not in SYMMETRIC_RELAXATION:
                if 'sweep' in kwargs1:
                    sweep1 = kwargs1['sweep']
                else:
                    sweep1 = DEFAULT_SWEEP

                if 'sweep' in kwargs2:
                    sweep2 = kwargs2['sweep']
                else:
                    sweep2 = DEFAULT_SWEEP

                if (sweep1, sweep2) not in (('forward', 'backward'),
                                            ('backward', 'forward'),
                                            ('symmetric', 'symmetric')):
                    ml.symmetric_smoothing = False

    else:
        mid_len = min_len

    # Fill in remaining levels
    for i in range(mid_len, len(ml.levels[:-1])):
        ml.levels[i].presmoother = setup_presmoother(ml.levels[i], **kwargs1)
        ml.levels[i].postsmoother = setup_postsmoother(ml.levels[i], **kwargs2)


def rho_D_inv_A(A):
    """Return the (approx.) spectral radius of D^-1 * A.

    Parameters
    ----------
    A : sparse-matrix

    Returns
    -------
    approximate spectral radius of diag(A)^{-1} A

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.relaxation.smoothing import rho_D_inv_A
    >>> from scipy.sparse import csr_matrix
    >>> import numpy as np
    >>> A = csr_matrix(np.array([[1.0,0,0],[0,2.0,0],[0,0,3.0]]))
    >>> print(f'{rho_D_inv_A(A):2.2}')
    1.0

    """
    if not hasattr(A, 'rho_D_inv'):
        D_inv = get_diagonal(A, inv=True)
        D_inv_A = scale_rows(A, D_inv, copy=True)
        A.rho_D_inv = approximate_spectral_radius(D_inv_A)

    return A.rho_D_inv


def rho_block_D_inv_A(A, Dinv):
    """Return the (approx.) spectral radius of block D^-1 * A.

    Parameters
    ----------
    A : sparse-matrix
        size NxN
    Dinv : array
        Inverse of diagonal blocks of A
        size (N/blocksize, blocksize, blocksize)

    Returns
    -------
    approximate spectral radius of (Dinv A)

    Examples
    --------
    >>> from pyamg.gallery import poisson
    >>> from pyamg.relaxation.smoothing import rho_block_D_inv_A
    >>> from pyamg.util.utils import get_block_diag
    >>> A = poisson((10,10), format='csr')
    >>> Dinv = get_block_diag(A, blocksize=4, inv_flag=True)

    """
    if not hasattr(A, 'rho_block_D_inv'):

        blocksize = Dinv.shape[1]
        if Dinv.shape[1] != Dinv.shape[2]:
            raise ValueError('Dinv has incorrect dimensions')

        if Dinv.shape[0] != int(A.shape[0]/blocksize):
            raise ValueError('Dinv and A have incompatible dimensions')

        Dinv = sparse.bsr_matrix((Dinv,
                                  np.arange(Dinv.shape[0]),
                                  np.arange(Dinv.shape[0]+1)),
                                 shape=A.shape)

        # Don't explicitly form Dinv*A
        def matvec(x):
            return Dinv*(A*x)
        D_inv_A = LinearOperator(A.shape, matvec, dtype=A.dtype)

        A.rho_block_D_inv = approximate_spectral_radius(D_inv_A)

    return A.rho_block_D_inv


# pylint: disable=redefined-builtin
def matrix_asformat(lvl, name, format, blocksize=None):
    """Set a matrix to a specific format.

    This routine looks for the matrix "name" in the specified format as a
    member of the level instance, lvl.  For example, if name='A', format='bsr'
    and blocksize=(4,4), and if lvl.Absr44 exists with the correct blocksize,
    then lvl.Absr is returned.  If the matrix doesn't already exist, lvl.name
    is converted to the desired format, and made a member of lvl.

    Only create such persistent copies of a matrix for routines such as
    presmoothing and postsmoothing, where the matrix conversion is done every
    cycle.

    Calling this function can _dramatically_ increase your memory costs.
    Be careful with it's usage.

    """
    desired_matrix = name + format
    M = getattr(lvl, name)

    if format == 'bsr':
        desired_matrix += str(blocksize[0])+str(blocksize[1])

    if hasattr(lvl, desired_matrix):
        # if lvl already contains lvl.name+format
        pass
    elif M.format == format and format != 'bsr':
        # is base_matrix already in the correct format?
        setattr(lvl, desired_matrix, M)
    elif M.format == format and format == 'bsr':
        # convert to bsr with the right blocksize
        # tobsr() will not do anything extra if this is uneeded
        setattr(lvl, desired_matrix, M.tobsr(blocksize=blocksize))
    else:
        # convert
        newM = getattr(M, 'to' + format)()
        setattr(lvl, desired_matrix, newM)

    return getattr(lvl, desired_matrix)


# pylint: disable=unused-argument
def setup_gauss_seidel(lvl, iterations=DEFAULT_NITER, sweep=DEFAULT_SWEEP):
    """Set up Gauss-Seidel."""
    def smoother(A, x, b):
        relaxation.gauss_seidel(A, x, b, iterations=iterations, sweep=sweep)
    return smoother


def setup_jacobi(lvl, iterations=DEFAULT_NITER, omega=1.0, withrho=True):
    """Set up weighted-Jacobi."""
    if withrho:
        omega = omega/rho_D_inv_A(lvl.A)

    def smoother(A, x, b):
        relaxation.jacobi(A, x, b, iterations=iterations, omega=omega)
    return smoother

# TODO: generalize
def setup_jacobi_krylov(lvl, iterations=DEFAULT_NITER, omega=1.0, withrho=True):
    """Set up weighted-Jacobi."""
    if withrho:
        omega = omega/rho_D_inv_A(lvl.A)

    def smoother(A, x, b):
        import scipy.sparse as sp
        from scipy.sparse.linalg import LinearOperator
        def mv(r):
            dx = np.zeros_like(r)
            relaxation.jacobi(A, dx, r, iterations=1, omega=omega)

            return dx
        M = sp.linalg.LinearOperator(A.shape, matvec=mv)

        from pyamg.krylov import fgmres
        #relaxation.jacobi(A, x, b, iterations=1, omega=omega)
        x[:] = fgmres(A, b, x0=x,
                            tol=1e-20,
                            maxiter=1,  # = max_outer
                            restrt=iterations,  # = max_inner
                            M=M)[0]

    return smoother



def setup_schwarz(lvl, iterations=DEFAULT_NITER, subdomain=None,
                  subdomain_ptr=None, inv_subblock=None, inv_subblock_ptr=None,
                  sweep=DEFAULT_SWEEP):
    """Set up Schwarz."""
    matrix_asformat(lvl, 'A', 'csr')
    lvl.Acsr.sort_indices()
    subdomain, subdomain_ptr, inv_subblock, inv_subblock_ptr = \
        relaxation.schwarz_parameters(lvl.Acsr, subdomain, subdomain_ptr,
                                      inv_subblock, inv_subblock_ptr)

    def smoother(A, x, b):
        relaxation.schwarz(lvl.Acsr, x, b, iterations=iterations,
                           subdomain=subdomain,
                           subdomain_ptr=subdomain_ptr,
                           inv_subblock=inv_subblock,
                           inv_subblock_ptr=inv_subblock_ptr, sweep=sweep)
    return smoother


def setup_strength_based_schwarz(lvl, iterations=DEFAULT_NITER,
                                 sweep=DEFAULT_SWEEP):
    """Set up strength-based Schwarz."""
    # Use the overlapping regions defined by strength of connection matrix C
    # for the overlapping Schwarz method
    if not hasattr(lvl, 'C'):
        C = lvl.A.tocsr()
    else:
        C = lvl.C.tocsr()

    C.sort_indices()
    subdomain_ptr = C.indptr.copy()
    subdomain = C.indices.copy()

    return setup_schwarz(lvl, iterations=iterations, subdomain=subdomain,
                         subdomain_ptr=subdomain_ptr, sweep=sweep)


def setup_block_jacobi(lvl, iterations=DEFAULT_NITER, omega=1.0, Dinv=None,
                       blocksize=None, withrho=True):
    """Set up block Jacobi."""
    # Determine Blocksize
    if blocksize is None and Dinv is None:
        if sparse.isspmatrix_csr(lvl.A):
            blocksize = 1
        elif sparse.isspmatrix_bsr(lvl.A):
            blocksize = lvl.A.blocksize[0]
    elif blocksize is None:
        blocksize = Dinv.shape[1]

    if blocksize == 1:
        # Block Jacobi is equivalent to normal Jacobi
        return setup_jacobi(lvl, iterations=iterations, omega=omega,
                            withrho=withrho)
    # Use Block Jacobi
    if Dinv is None:
        Dinv = get_block_diag(lvl.A, blocksize=blocksize, inv_flag=True)
    if withrho:
        omega = omega/rho_block_D_inv_A(lvl.A, Dinv)

    def smoother(A, x, b):
        relaxation.block_jacobi(A, x, b, iterations=iterations,
                                omega=omega, Dinv=Dinv,
                                blocksize=blocksize)
    return smoother


def setup_block_gauss_seidel(lvl, iterations=DEFAULT_NITER,
                             sweep=DEFAULT_SWEEP,
                             Dinv=None, blocksize=None):
    """Set up block Gauss-Seidel."""
    # Determine Blocksize
    if blocksize is None and Dinv is None:
        if sparse.isspmatrix_csr(lvl.A):
            blocksize = 1
        elif sparse.isspmatrix_bsr(lvl.A):
            blocksize = lvl.A.blocksize[0]
    elif blocksize is None:
        blocksize = Dinv.shape[1]

    if blocksize == 1:
        # Block GS is equivalent to normal GS
        return setup_gauss_seidel(lvl, iterations=iterations, sweep=sweep)

    # Use Block GS
    if Dinv is None:
        Dinv = get_block_diag(lvl.A, blocksize=blocksize, inv_flag=True)

    def smoother(A, x, b):
        relaxation.block_gauss_seidel(A, x, b, iterations=iterations,
                                      Dinv=Dinv, blocksize=blocksize,
                                      sweep=sweep)

    return smoother


def setup_richardson(lvl, iterations=DEFAULT_NITER, omega=1.0):
    """Set up Richardson."""
    omega = omega/approximate_spectral_radius(lvl.A)

    def smoother(A, x, b):
        relaxation.polynomial(A, x, b, coefficients=[omega],
                              iterations=iterations)
    return smoother


def setup_sor(lvl, omega=0.5, iterations=DEFAULT_NITER, sweep=DEFAULT_SWEEP):
    """Set up SOR."""
    def smoother(A, x, b):
        relaxation.sor(A, x, b, omega=omega, iterations=iterations,
                       sweep=sweep)
    return smoother


def setup_chebyshev(lvl, lower_bound=1.0/30.0, upper_bound=1.1, degree=3,
                    iterations=DEFAULT_NITER):
    """Set up Chebyshev."""
    rho = approximate_spectral_radius(lvl.A)
    a = rho * lower_bound
    b = rho * upper_bound
    # drop the constant coefficient
    coefficients = -chebyshev_polynomial_coefficients(a, b, degree)[:-1]

    def smoother(A, x, b):
        relaxation.polynomial(A, x, b, coefficients=coefficients,
                              iterations=iterations)
    return smoother


def setup_jacobi_ne(lvl, iterations=DEFAULT_NITER, omega=1.0, withrho=True):
    """Set up Jacobi NE."""
    matrix_asformat(lvl, 'A', 'csr')
    if withrho:
        omega = omega/rho_D_inv_A(lvl.Acsr)**2

    def smoother(A, x, b):
        relaxation.jacobi_ne(lvl.Acsr, x, b, iterations=iterations,
                             omega=omega)
    return smoother


def setup_gauss_seidel_ne(lvl, iterations=DEFAULT_NITER, sweep=DEFAULT_SWEEP,
                          omega=1.0):
    """Set up Gauss-Seidel NE."""
    matrix_asformat(lvl, 'A', 'csr')

    def smoother(A, x, b):
        relaxation.gauss_seidel_ne(lvl.Acsr, x, b, iterations=iterations,
                                   sweep=sweep, omega=omega)
    return smoother


def setup_gauss_seidel_nr(lvl, iterations=DEFAULT_NITER, sweep=DEFAULT_SWEEP,
                          omega=1.0):
    """Set up Gauss-Seidel NR."""
    matrix_asformat(lvl, 'A', 'csc')

    def smoother(A, x, b):
        relaxation.gauss_seidel_nr(lvl.Acsc, x, b, iterations=iterations,
                                   sweep=sweep, omega=omega)
    return smoother


def setup_gmres(lvl, tol=1e-12, maxiter=1, restrt=None, M=None, callback=None,
                residuals=None):
    """Set up GMRES smoothing."""
    def smoother(A, x, b):
        x[:] = (
            gmres(
                A,
                b,
                x0=x,
                tol=tol,
                maxiter=maxiter,
                restrt=restrt,
                M=M,
                callback=callback,
                residuals=residuals)[0]).reshape(
            x.shape)
    return smoother


def setup_cg(lvl, tol=1e-12, maxiter=1, M=None, callback=None, residuals=None):
    """Set up CG smoothing."""
    def smoother(A, x, b):
        x[:] = (cg(A, b, x0=x, tol=tol, maxiter=maxiter, M=M,
                   callback=callback, residuals=residuals)[0]).reshape(x.shape)
    return smoother


def setup_cgne(lvl, tol=1e-12, maxiter=1, M=None, callback=None,
               residuals=None):
    """Set up CGNE smoothing."""
    def smoother(A, x, b):
        x[:] = (
            cgne(
                A,
                b,
                x0=x,
                tol=tol,
                maxiter=maxiter,
                M=M,
                callback=callback,
                residuals=residuals)[0]).reshape(
            x.shape)
    return smoother


def setup_cgnr(lvl, tol=1e-12, maxiter=1, M=None, callback=None,
               residuals=None):
    """Set up CGNR smoothing."""
    def smoother(A, x, b):
        x[:] = (
            cgnr(
                A,
                b,
                x0=x,
                tol=tol,
                maxiter=maxiter,
                M=M,
                callback=callback,
                residuals=residuals)[0]).reshape(
            x.shape)
    return smoother


def setup_none(lvl):
    """Set up default, empty smoother."""
    def smoother(A, x, b):
        pass
    return smoother


def _setup_call(fn):
    """Register setup functions.

    This is a helper function to call the setup methods and avoids use of eval().
    """
    setup_register = {
        'gauss_seidel':           setup_gauss_seidel,
        'jacobi':                 setup_jacobi,
        'jacobi_krylov':          setup_jacobi_krylov,
        'schwarz':                setup_schwarz,
        'strength_based_schwarz': setup_strength_based_schwarz,
        'block_jacobi':           setup_block_jacobi,
        'block_gauss_seidel':     setup_block_gauss_seidel,
        'richardson':             setup_richardson,
        'sor':                    setup_sor,
        'chebyshev':              setup_chebyshev,
        'jacobi_ne':              setup_jacobi_ne,
        'gauss_seidel_ne':        setup_gauss_seidel_ne,
        'gauss_seidel_nr':        setup_gauss_seidel_nr,
        'gmres':                  setup_gmres,
        'cg':                     setup_cg,
        'cgne':                   setup_cgne,
        'cgnr':                   setup_cgnr,
        'none':                   setup_none,
    }

    if fn not in setup_register:
        raise ValueError(f'Function {fn} does not have a setup')

    return setup_register[fn]
