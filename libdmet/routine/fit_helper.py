#! /usr/bin/env python

"""
Helper functions for fit.
Modified from Scipy.

Author:
    Zhi-Hao Cui
"""

import sys
import warnings
import time
import numpy as np
import scipy.linalg as la

from scipy.optimize import minimize_scalar, fmin
from scipy.optimize._trustregion_ncg import CGSteihaugSubproblem 
from scipy.optimize import OptimizeResult
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS

from pyscf.soscf import ciah
from pyscf.lib import logger

from libdmet.utils.misc import max_abs
from libdmet.utils import logger as log

#norm = la.norm
norm = max_abs
_epsilon = np.sqrt(np.finfo(float).eps)

_status_message = {'success': 'Optimization terminated successfully.',
                   'maxfev': 'Maximum number of function evaluations has '
                              'been exceeded.',
                   'maxiter': 'Maximum number of iterations has been '
                              'exceeded.',
                   'pr_loss': 'Desired error not necessarily achieved due '
                              'to precision loss.',
                   'nan': 'NaN result encountered.',
                   'out_of_bounds': 'The result is outside of the provided '
                                    'bounds.'}

class OptimizeWarning(UserWarning):
    pass

def _check_unknown_options(unknown_options):
    if unknown_options:
        msg = ", ".join(map(str, unknown_options.keys()))
        # Stack level 4: this is called from _minimize_*, which is
        # called from another function in SciPy. Level 4 is the first
        # level in user code.
        warnings.warn("Unknown solver options: %s" % msg, OptimizeWarning, 4)

def _prepare_scalar_function(fun, x0, jac=None, args=(), bounds=None,
                             epsilon=None, finite_diff_rel_step=None,
                             hess=None):
    """
    Creates a ScalarFunction object for use with scalar minimizers
    (BFGS/LBFGSB/SLSQP/TNC/CG/etc).

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.

            ``fun(x, *args) -> float``

        where ``x`` is an 1-D array with shape (n,) and ``args``
        is a tuple of the fixed parameters needed to completely
        specify the function.
    x0 : ndarray, shape (n,)
        Initial guess. Array of real elements of size (n,),
        where 'n' is the number of independent variables.
    jac : {callable,  '2-point', '3-point', 'cs', None}, optional
        Method for computing the gradient vector. If it is a callable, it
        should be a function that returns the gradient vector:

            ``jac(x, *args) -> array_like, shape (n,)``

        If one of `{'2-point', '3-point', 'cs'}` is selected then the gradient
        is calculated with a relative step for finite differences. If `None`,
        then two-point finite differences with an absolute step is used.
    args : tuple, optional
        Extra arguments passed to the objective function and its
        derivatives (`fun`, `jac` functions).
    bounds : sequence, optional
        Bounds on variables. 'new-style' bounds are required.
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    hess : {callable,  '2-point', '3-point', 'cs', None}
        Computes the Hessian matrix. If it is callable, it should return the
        Hessian matrix:

            ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``

        Alternatively, the keywords {'2-point', '3-point', 'cs'} select a
        finite difference scheme for numerical estimation.
        Whenever the gradient is estimated via finite-differences, the Hessian
        cannot be estimated with options {'2-point', '3-point', 'cs'} and needs
        to be estimated using one of the quasi-Newton strategies.

    Returns
    -------
    sf : ScalarFunction
    """
    if callable(jac):
        grad = jac
    elif jac in FD_METHODS:
        # epsilon is set to None so that ScalarFunction is made to use
        # rel_step
        epsilon = None
        grad = jac
    else:
        # default (jac is None) is to do 2-point finite differences with
        # absolute step size. ScalarFunction has to be provided an
        # epsilon value that is not None to use absolute steps. This is
        # normally the case from most _minimize* methods.
        grad = '2-point'
        epsilon = epsilon

    if hess is None:
        # ScalarFunction requires something for hess, so we give a dummy
        # implementation here if nothing is provided, return a value of None
        # so that downstream minimisers halt. The results of `fun.hess`
        # should not be used.
        def hess(x, *args):
            return None

    if bounds is None:
        bounds = (-np.inf, np.inf)

    # ScalarFunction caches. Reuse of fun(x) during grad
    # calculation reduces overall function evaluations.
    # ZHC NOTE for scipy < 1.5, epsilon is not an arg
    try:
        sf = ScalarFunction(fun, x0, args, grad, hess,
                            finite_diff_rel_step, bounds, epsilon=epsilon)
    except:
        sf = ScalarFunction(fun, x0, args, grad, hess,
                            finite_diff_rel_step, bounds)
    return sf

def wrap_function(function, args):
    # wraps a minimizer function to count number of evaluations
    # and to easily provide an args kwd.
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(x, *wrapper_args):
        ncalls[0] += 1
        # A copy of x is sent to the user function (gh13740)
        return function(np.copy(x), *(wrapper_args + args))

    return ncalls, function_wrapper

# ****************************************************************************
# CG, BFGS
# ****************************************************************************

def _minimize_cg(fun, x0, args=(), jac=None, callback=None,
                 ytol=1e-7, gtol=1e-3, dx_tol=1e-7, norm=norm, 
                 eps=_epsilon, maxiter=None,
                 disp=False, return_all=False, 
                 init_step=1.0, min_step=0.1, 
                 xatol=1e-5, finite_diff_rel_step=None, **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    conjugate gradient algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.

    """
    _check_unknown_options(unknown_options)

    retall = return_all

    x0 = np.asarray(x0).flatten()
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = _prepare_scalar_function(fun, x0, jac=jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    gfk = myfprime(x0)
    k = 0
    xk = x0

    old_fval = f(xk)
    steps = [init_step]
    xatol = [xatol]
    min_step = [min_step]

    if retall:
        allvecs = [xk]
    warnflag = 0
    pk = -gfk
    gnorm = norm(gfk)

    while (k < maxiter):
        deltak = np.dot(gfk, gfk)

        def polak_ribiere_powell_step(alpha, gfkp1=None):
            xkp1 = xk + alpha * pk
            if gfkp1 is None:
                gfkp1 = myfprime(xkp1)
            yk = gfkp1 - gfk
            beta_k = max(0, np.dot(yk, gfkp1) / deltak)
            pkp1 = -gfkp1 + beta_k * pk
            gnorm = norm(gfkp1)
            return (alpha, xkp1, pkp1, gfkp1, gnorm)
        
        LineSearchFn = lambda step: f(xk + step * pk)
        
        def FindStep():
            scale = max(abs(np.average(steps[-2:])), min_step[0])
            res = minimize_scalar(LineSearchFn, bounds=(0.0, scale), 
                                  method="bounded", options={"maxiter": 100, 
                                                             "xatol": xatol[0]})
            if res.fun > old_fval: 
                # ZHC NOTE line search is not accurate enough
                # minimize_scalar can give a local minima higher than f(0).
                xopt, fopt, _, _, _ = fmin(LineSearchFn, 0.0, disp=False, 
                                           xtol=xatol[0]*0.1, full_output=True)
                if fopt > old_fval:
                    log.warn("line search fails, resulting value  %20.12f is\n"
                             "larger than the previous step value %20.12f",
                             fopt, old_fval)
                    res = (0.0, old_fval)
                else:
                    res = (xopt[0], fopt)
            else:
                res = (res.x, res.fun)
            return res
        
        alpha_k, new_fval = FindStep()
        steps.append(alpha_k)
        dy = abs(new_fval - old_fval)
        norm_dx = norm(pk) * alpha_k

        if abs(norm_dx) < dx_tol:
            log.debug(0, "CG: dx (%20.12g) < %15.8g reached.", norm_dx, dx_tol)
            break
        old_fval = new_fval
        
        alpha_k, xk, pk, gfk, gnorm = polak_ribiere_powell_step(alpha_k)
        log.debug(0, "%4d %20.12f %20.12f %20.12f %15.3e", k, old_fval, gnorm,
                  norm_dx, alpha_k)
        
        if retall:
            allvecs.append(xk)
        if callback is not None:
            callback(xk)

        k += 1

        # ZHC NOTE it is better to do at least 1 step optimize.
        if gnorm < gtol:
            log.debug(0, "CG: gnorm (%20.12g) < %15.8g reached.", gnorm, gtol)
            break
        
        if dy < ytol:
            log.debug(0, "CG: dy (%20.12g) < %15.8g reached.", dy, ytol)
            break

    fval = old_fval
    if k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    else:
        msg = _status_message['success']

    if disp:
        log.info("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        log.info("         Current function value: %f" % fval)
        log.info("         Iterations: %d" % k)
        log.info("         Function evaluations: %d" % sf.nfev)
        log.info("         Gradient evaluations: %d" % sf.ngev)

    result = OptimizeResult(fun=fval, jac=gfk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result

def _minimize_bfgs(fun, x0, args=(), jac=None, callback=None,
                   ytol=1e-7, gtol=1e-5, dx_tol=1e-7, norm=norm, 
                   eps=_epsilon, maxiter=None,
                   disp=False, return_all=False, 
                   init_step=1.0, min_step=0.1, xatol=1e-5,  
                   finite_diff_rel_step=None, **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.

    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac` is approximated, use this value for the step size.

    """
    _check_unknown_options(unknown_options)

    retall = return_all

    x0 = np.asarray(x0).flatten()
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = _prepare_scalar_function(fun, x0, jac=jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    gfk = myfprime(x0)
    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I

    old_fval = f(x0)
    steps    = [init_step]
    min_step = [min_step]
    xatol    = [xatol]

    xk = x0
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = norm(gfk)

    while (gnorm > gtol) and (k < maxiter):
        pk = -np.dot(Hk, gfk)
        
        LineSearchFn = lambda step: f(xk + step * pk)
        
        def FindStep():
            scale = max(abs(np.average(steps[-2:])), min_step[0])
            res = minimize_scalar(LineSearchFn, bounds=(0.0, scale), 
                                  method="bounded", options={"maxiter": 100,
                                                             "xatol": xatol[0]})
            if res.fun > old_fval: # line search is not accurate enough
                xopt, fopt, _, _, _ = fmin(LineSearchFn, 0.0, disp=False, 
                                           xtol=xatol[0]*0.1, full_output=True)
                if fopt > old_fval:
                    log.warn("line search fails, resulting value  %20.12f is\n"
                             "larger than the previous step value %20.12f",
                             fopt, old_fval)
                    res = (0.0, old_fval)
                else:
                    res = (xopt[0], fopt)
            else:
                res = (res.x, res.fun)
            return res
        
        alpha_k, new_fval = FindStep()
        steps.append(alpha_k)
        dy = abs(new_fval - old_fval)
        norm_dx = norm(pk) * alpha_k
        old_fval = new_fval

        xkp1 = xk + alpha_k * pk
        if retall:
            allvecs.append(xkp1)
        sk = xkp1 - xk
        xk = xkp1
        gfkp1 = myfprime(xkp1)

        yk = gfkp1 - gfk
        gfk = gfkp1
        if callback is not None:
            callback(xk)
        
        k += 1
        gnorm = norm(gfk)
        
        if gnorm < gtol:
            log.debug(0, "BFGS: gnorm (%20.12g) < %15.8g reached.", gnorm, gtol)
            break
        
        if dy < ytol:
            log.debug(0, "BFGS: dy (%20.12g) < %15.8g reached.", dy, ytol)
            break

        log.debug(0, "%4d %20.12f %20.12f %20.12f %15.3e", k, old_fval, gnorm,
                  norm_dx, alpha_k)

        if abs(norm_dx) < dx_tol:
            log.debug(0, "BFGS: dx (%20.12f) = 0 reached.", norm_dx)
            break

        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        try:  # this was handled in numeric, let it remaines for more safety
            rhok = 1.0 / (np.dot(yk, sk))
        except ZeroDivisionError:
            rhok = 1000.0
            if disp:
                log.warn("Divide-by-zero encountered: rhok assumed large")
        if np.isinf(rhok):  # this is patch for np
            rhok = 1000.0
            if disp:
                log.warn("Divide-by-zero encountered: rhok assumed large")
        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                                 sk[np.newaxis, :])

    fval = old_fval
    if np.isnan(fval):
        # This can happen if the first call to f returned NaN;
        # the loop is then never entered.
        warnflag = 2

    if warnflag == 2:
        msg = _status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = _status_message['maxiter']
    else:
        msg = _status_message['success']

    if disp:
        log.info("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        log.info("         Current function value: %f" % fval)
        log.info("         Iterations: %d" % k)
        log.info("         Function evaluations: %d" % sf.nfev)
        log.info("         Gradient evaluations: %d" % sf.ngev)

    result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result

# ****************************************************************************
# trust region Newton-CG
# ****************************************************************************

def _minimize_trust_region(fun, x0, args=(), jac=None, hess=None, hessp=None,
                           subproblem=CGSteihaugSubproblem,
                           initial_trust_radius=0.001,
                           max_trust_radius=0.01, eta=0.10, 
                           ytol=1e-7, gtol=1e-3, dx_tol=1e-7,
                           maxiter=None, disp=False, return_all=False,
                           callback=None, inexact=True, **unknown_options):
    """
    Minimization of scalar function of one or more variables using a
    trust-region algorithm.

    Options for the trust-region algorithm are:
        initial_trust_radius : float
            Initial trust radius.
        max_trust_radius : float
            Never propose steps that are longer than this value.
        eta : float
            Trust region related acceptance stringency for proposed steps.
        gtol : float
            Gradient norm must be less than `gtol`
            before successful termination.
        maxiter : int
            Maximum number of iterations to perform.
        disp : bool
            If True, print convergence message.
        inexact : bool
            Accuracy to solve subproblems. If True requires less nonlinear
            iterations, but more vector products. Only effective for method
            trust-krylov.

    This function is called by the `minimize` function.
    It is not supposed to be called directly.
    """
    _check_unknown_options(unknown_options)

    if jac is None:
        raise ValueError('Jacobian is currently required for trust-region '
                         'methods')
    if hess is None and hessp is None:
        raise ValueError('Either the Hessian or the Hessian-vector product '
                         'is currently required for trust-region methods')
    if subproblem is None:
        raise ValueError('A subproblem solving strategy is required for '
                         'trust-region methods')
    if not (0 <= eta < 0.25):
        raise Exception('invalid acceptance stringency')
    if max_trust_radius <= 0:
        raise Exception('the max trust radius must be positive')
    if initial_trust_radius <= 0:
        raise ValueError('the initial trust radius must be positive')
    if initial_trust_radius >= max_trust_radius:
        raise ValueError('the initial trust radius must be less than the '
                         'max trust radius')

    # force the initial guess into a nice format
    x0 = np.asarray(x0).flatten()

    # Wrap the functions, for a couple reasons.
    # This tracks how many times they have been called
    # and it automatically passes the args.
    nfun, fun = wrap_function(fun, args)
    njac, jac = wrap_function(jac, args)
    nhess, hess = wrap_function(hess, args)
    nhessp, hessp = wrap_function(hessp, args)

    # limit the number of iterations
    if maxiter is None:
        maxiter = len(x0)*200
    
    # init the search status
    warnflag = 0

    # initialize the search
    trust_radius = initial_trust_radius
    x = x0
    if return_all:
        allvecs = [x]
    m = subproblem(x, fun, jac, hess, hessp)
    k = 0
    
    x_old = np.array(x, copy=True)
    # search for the function min
    # do not even start if the gradient is small enough
    while m.jac_mag >= gtol:

        # Solve the sub-problem.
        # This gives us the proposed step relative to the current position
        # and it tells us whether the proposed step
        # has reached the trust region boundary or not.
        try:
            p, hits_boundary = m.solve(trust_radius)
        except np.linalg.linalg.LinAlgError as e:
            warnflag = 3
            break

        # calculate the predicted value at the proposed point
        predicted_value = m(p)

        # define the local approximation at the proposed point
        x_proposed = x + p
        m_proposed = subproblem(x_proposed, fun, jac, hess, hessp)

        # evaluate the ratio defined in equation (4.4)
        actual_reduction = m.fun - m_proposed.fun
        predicted_reduction = m.fun - predicted_value
        if predicted_reduction <= 0:
            warnflag = 2
            break
        rho = actual_reduction / predicted_reduction

        # update the trust radius according to the actual/predicted ratio
        if rho < 0.25:
            trust_radius *= 0.25
        elif rho > 0.75 and hits_boundary:
            trust_radius = min(1.75 * trust_radius, max_trust_radius)

        # if the ratio is high enough then accept the proposed step
        if rho > eta:
            x = x_proposed
            m = m_proposed

        # append the best guess, call back, increment the iteration count
        if return_all:
            allvecs.append(np.copy(x))
        if callback is not None:
            callback(np.copy(x))
        
        # ZHC NOTE printing the information on the fly.
        dx = x - x_old
        norm_dx = norm(dx)
        x_old = np.array(x, copy=True)
        log.debug(0, "%4d %20.12f %20.12f %20.12f %15.3e", k, m.fun, m.jac_mag, norm_dx, trust_radius)
        k += 1

        # check if the gradient is small enough to stop
        if m.jac_mag < gtol:
            log.debug(0, "NCG: g = 0 condition reached.")
            warnflag = 0
            break
        
        # ZHC NOTE terminate if m.fun is close to 0
        if m.fun < ytol * 0.1:
            log.debug(0, "NCG: y = 0 condition reached.")
            warnflag = 0
            break
        
        # ZHC NOTE terminate if x changes very small
        if abs(norm_dx) < dx_tol:
            log.debug(0, "NCG: dx = 0 condition reached.")
            warnflag = 0
            break

        # check if we have looked at enough iterations
        if k >= maxiter:
            warnflag = 1
            break

    # print some stuff if requested
    status_messages = (
            _status_message['success'],
            _status_message['maxiter'],
            'A bad approximation caused failure to predict improvement.',
            'A linalg error occurred, such as a non-psd Hessian.',
            )
    if disp:
        if warnflag == 0:
            log.info(status_messages[warnflag])
        else:
            log.warn('Warning: ' + status_messages[warnflag])
        log.info("         Current function value: %f" % m.fun)
        log.info("         Iterations: %d" % k)
        log.info("         Function evaluations: %d" % nfun[0])
        log.info("         Gradient evaluations: %d" % njac[0])
        log.info("         Hessian evaluations: %d" % (nhess[0] + nhessp[0]))

    result = OptimizeResult(x=x, success=(warnflag == 0), status=warnflag,
                            fun=m.fun, jac=m.jac, nfev=nfun[0], njev=njac[0],
                            nhev=nhess[0] + nhessp[0], nit=k,
                            message=status_messages[warnflag])

    if hess is not None:
        result['hess'] = m.hess

    if return_all:
        result['allvecs'] = allvecs

    return result

# ****************************************************************************
# CIAHMinimizer
# ****************************************************************************

def rotate_orb_cc(iah, u0, conv_tol_grad=None, verbose=logger.NOTE):
    t2m = (logger.process_clock(), logger.perf_counter())
    if isinstance(verbose, logger.Logger):
        pyscflog = verbose
    else:
        pyscflog = logger.Logger(sys.stdout, verbose)

    if conv_tol_grad is None:
        conv_tol_grad = iah.conv_tol_grad

    g_orb, h_op, h_diag = iah.gen_g_hop(u0)
    g_kf = g_orb
    norm_gkf = norm_gorb = norm(g_orb)
    pyscflog.debug('    |g|= %4.3g (keyframe)', norm_gorb)
    t3m = pyscflog.timer('gen h_op', *t2m)

    if h_diag is None:
        def precond(x, e):
            return x
    else:
        def precond(x, e):
            hdiagd = h_diag - (e-iah.ah_level_shift)
            hdiagd[abs(hdiagd) < 1e-8] = 1e-8
            x = x / hdiagd
            return x

    def scale_down_step(dxi, hdxi, norm_gorb):
        dxmax = max_abs(dxi)
        if dxmax > iah.max_stepsize:
            scale = iah.max_stepsize / dxmax
            pyscflog.debug1('Scale rotation by %g', scale)
            dxi *= scale
            hdxi *= scale
        return dxi, hdxi

    class Statistic:
        def __init__(self):
            self.imic = 0
            self.tot_hop = 0
            self.tot_kf = 0

    kf_trust_region = iah.kf_trust_region
    g_op = lambda: g_orb
    x0_guess = g_orb
    while True:
        g_orb0 = g_orb
        stat = Statistic()
        dr = 0
        ikf = 0
        # ZHC NOTE
        ukf = 0.0

        for ah_conv, ihop, w, dxi, hdxi, residual, seig \
                in ciah.davidson_cc(h_op, g_op, precond, x0_guess,
                               tol=iah.ah_conv_tol, max_cycle=iah.ah_max_cycle,
                               lindep=iah.ah_lindep, verbose=pyscflog):
            stat.tot_hop = ihop
            norm_residual = norm(residual)
            if (ah_conv or ihop == iah.ah_max_cycle or # make sure to use the last step
                ((norm_residual < iah.ah_start_tol) and (ihop >= iah.ah_start_cycle)) or
                (seig < iah.ah_lindep)):
                stat.imic += 1
                dxmax = max_abs(dxi)
                dxi, hdxi = scale_down_step(dxi, hdxi, norm_gorb)

                dr = dr + dxi
                g_orb = g_orb + hdxi
                norm_dr = norm(dr)
                norm_gorb = norm(g_orb)
                pyscflog.debug('    imic %d(%d)  |g|= %4.3g  |dxi|= %4.3g  '
                          'max(|x|)= %4.3g  |dr|= %4.3g  eig= %4.3g  seig= %4.3g',
                          stat.imic, ihop, norm_gorb, norm(dxi),
                          dxmax, norm_dr, w, seig)

                max_cycle = max(iah.max_iters,
                                iah.max_iters-int(np.log(norm_gkf+1e-9)*2))
                pyscflog.debug1('Set max_cycle %d', max_cycle)
                ikf += 1
                if stat.imic > 3 and norm_gorb > norm_gkf*iah.ah_trust_region:
                    g_orb = g_orb - hdxi
                    dr -= dxi
                    norm_gorb = norm(g_orb)
                    pyscflog.debug('|g| >> keyframe, Restore previouse step')
                    break

                elif (stat.imic >= max_cycle or norm_gorb < conv_tol_grad*.2):
                    break

                elif (ikf > 2 and # avoid frequent keyframe
                      (ikf >= iah.kf_interval or
# Insert keyframe if the keyframe and the esitimated g_orb are too different
                       norm_gorb < norm_gkf/kf_trust_region)):
                    ikf = 0

                    # ZHC NOTE
                    ukf += dr

                    dr[:] = 0
                    # ZHC NOTE
                    g_kf1 = iah.get_grad(u0 + ukf)
                    stat.tot_kf += 1
                    norm_gkf1 = norm(g_kf1)
                    norm_dg = norm(g_kf1-g_orb)
                    pyscflog.debug('Adjust keyframe g_orb to |g|= %4.3g  '
                              '|g-correction|= %4.3g', norm_gkf1, norm_dg)

                    if (norm_dg < norm_gorb*iah.ah_trust_region  # kf not too diff
                        #or norm_gkf1 < norm_gkf  # grad is decaying
                        # close to solution
                        or norm_gkf1 < conv_tol_grad*iah.ah_trust_region):
                        kf_trust_region = min(max(norm_gorb/(norm_dg+1e-9), iah.kf_trust_region), 10)
                        pyscflog.debug1('Set kf_trust_region = %g', kf_trust_region)
                        g_orb = g_kf = g_kf1
                        norm_gorb = norm_gkf = norm_gkf1
                    else:
                        g_orb = g_orb - hdxi
                        dr -= dxi
                        norm_gorb = norm(g_orb)
                        pyscflog.debug('Out of trust region. Restore previouse step')
                        break
        
        # ZHC NOTE
        u = ukf + dr 
        pyscflog.debug('    tot inner=%d  |g|= %4.3g ', stat.imic, norm_gorb)
        h_op = h_diag = None
        t3m = pyscflog.timer('aug_hess in %d inner iters' % stat.imic, *t3m)
        u0 = (yield u, g_kf, stat)

        g_kf, h_op, h_diag = iah.gen_g_hop(u0)
        norm_gkf = norm(g_kf)
        norm_dg = norm(g_kf-g_orb)
        pyscflog.debug('    |g|= %4.3g (keyframe), |g-correction|= %4.3g',
                  norm_gkf, norm_dg)
        kf_trust_region = min(max(norm_gorb/(norm_dg+1e-9), iah.kf_trust_region), 10)
        pyscflog.debug1('Set  kf_trust_region = %g', kf_trust_region)
        g_orb = g_kf
        norm_gorb = norm_gkf
        x0_guess = dxi

def kernel(localizer, u0, callback=None, verbose=None):
    """
    Kernel for VcorMinimizer.
    """
    if localizer.verbose >= logger.WARN:
        localizer.check_sanity()
    localizer.dump_flags()

    cput0 = (logger.process_clock(), logger.perf_counter())
    pyscflog = logger.new_logger(localizer, verbose=verbose)

    if localizer.conv_tol_grad is None:
        conv_tol_grad = np.sqrt(localizer.conv_tol*.1)
        pyscflog.info('Set conv_tol_grad to %g', conv_tol_grad)
    else:
        conv_tol_grad = localizer.conv_tol_grad

    rotaiter = rotate_orb_cc(localizer, u0, conv_tol_grad, verbose=pyscflog)
    u, g_orb, stat = next(rotaiter)
    cput1 = pyscflog.timer('initializing CIAH', *cput0)

    tot_kf = stat.tot_kf
    tot_hop = stat.tot_hop
    conv = False
    e_last = 0
    for imacro in range(localizer.max_cycle):
        norm_gorb = norm(g_orb)
        # ZHC NOTE
        u0 = u0 + u
        norm_dx = norm(u)
        e = localizer.cost_function(u0)
        e_last, de = e, e-e_last
        
        pyscflog.info('macro= %d  f(x)= %.14g  delta_f= %g  |g|= %g  %d KF %d Hx',
                 imacro+1, e, de, norm_gorb, stat.tot_kf+1, stat.tot_hop)
        cput1 = pyscflog.timer('cycle= %d'%(imacro+1), *cput1)
        
        log.debug(0, "%4d %20.12f %20.12f %20.12f ", imacro, e, norm_gorb, norm_dx)
        if imacro > 0 and de > localizer.conv_tol:
            log.warn("CIAH: function value increasing, de = %20.12f", de)
            u0 = u0 - u
            e = localizer.cost_function(u0)
            e_last, de = e, e-e_last
            break
        
        if (norm_gorb < conv_tol_grad or abs(de) < localizer.conv_tol):
            conv = True

        if callable(callback):
            callback(locals())

        if conv:
            break

        u, g_orb, stat = rotaiter.send(u0)
        tot_kf += stat.tot_kf
        tot_hop += stat.tot_hop

    rotaiter.close()
    pyscflog.info('macro X = %d  f(x)= %.14g  |g|= %g  %d intor %d KF %d Hx',
             imacro+1, e, norm_gorb,
             (imacro+1)*2, tot_kf+imacro+1, tot_hop)
    localizer.e = e
    localizer.norm_gorb = norm_gorb
    return u0

class CIAHMinimizer(ciah.CIAHOptimizer):
    r"""
    Minimize a scalar function using CIAH algorithm.

    Args:
        func: function to minimze.
        grad: gradient of func.
        h_op: hessian vector product, hess(x, p).
        h_diag: diagonal elements of hessian, for preconditioner.

    Attributes for CIAHMinimizer class:
        verbose : int
            Print level.  Default value equals DEBUG0.
        conv_tol : float
            Converge threshold.  Default 1e-6
        conv_tol_grad : float
            Converge threshold for orbital rotation gradients.  Default 1e-3
        max_cycle : int
            The max. number of macro iterations. Default 300
        max_iters : int
            The max. number of iterations in each macro iteration. Default 10
        max_stepsize : float
            The step size for x change.  Small step (0.005 - 0.05) is prefered.
            Default 0.01.
    """

    conv_tol        = 1e-6
    conv_tol_grad   = 1e-3
    max_cycle       = 300
    max_iters       = 5
    max_stepsize    = 0.005
    ah_trust_region = 2.0
    ah_start_tol    = 1.0
    ah_start_cycle  = 2
    ah_max_cycle    = 40
    kf_interval     = 1
    kf_trust_region = 2.0

    def __init__(self, func, grad, h_op, h_diag=None):
        ciah.CIAHOptimizer.__init__(self)
        self.stdout = sys.stdout
        self.verbose = 5
        
        # ZHC NOTE check whether the diagonal hessian can be computed efficiently?
        self.func   = func
        self.grad   = grad
        self.h_op   = h_op
        self.h_diag = h_diag

        keys = set(('conv_tol', 'conv_tol_grad', 'max_cycle', 'max_iters',
                    'max_stepsize', 'ah_trust_region', 'ah_start_tol',
                    'ah_max_cycle'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self, verbose=None):
        pyscflog = logger.new_logger(self, verbose)
        pyscflog.info('\n')
        pyscflog.info('******** %s ********', self.__class__)
        pyscflog.info('conv_tol = %s'       , self.conv_tol       )
        pyscflog.info('conv_tol_grad = %s'  , self.conv_tol_grad  )
        pyscflog.info('max_cycle = %s'      , self.max_cycle      )
        pyscflog.info('max_stepsize = %s'   , self.max_stepsize   )
        pyscflog.info('max_iters = %s'      , self.max_iters      )
        pyscflog.info('kf_interval = %s'    , self.kf_interval    )
        pyscflog.info('kf_trust_region = %s', self.kf_trust_region)
        pyscflog.info('ah_start_tol = %s'   , self.ah_start_tol   )
        pyscflog.info('ah_start_cycle = %s' , self.ah_start_cycle )
        pyscflog.info('ah_level_shift = %s' , self.ah_level_shift )
        pyscflog.info('ah_conv_tol = %s'    , self.ah_conv_tol    )
        pyscflog.info('ah_lindep = %s'      , self.ah_lindep      )
        pyscflog.info('ah_max_cycle = %s'   , self.ah_max_cycle   )
        pyscflog.info('ah_trust_region = %s', self.ah_trust_region)
    
    def gen_g_hop(self, u):
        g = self.grad(u)
        h_op = lambda p: self.h_op(u, p)
        h_diag = self.h_diag
        return g, h_op, h_diag

    def get_grad(self, u):
        return self.grad(u)

    def cost_function(self, u):
        return self.func(u)

    kernel = kernel

