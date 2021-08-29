#! /usr/bin/env python

"""
Fitting functions to minimize (rdm1 - rdm1_target)^2.

Author:
    Zhi-Hao Cui
    Bo-Xiao Zheng
"""

import libdmet.routine.fit_helper as fit_hp
from libdmet.routine.fit_helper import *

def minimize(fn, x0, MaxIter=300, fgrad=None, callback=None, method='CG', \
        ytol=1e-7, gtol=1e-3, dx_tol=1e-7, **kwargs):
    """
    Main wrapper for minimizers.
    """
    if "serial" not in kwargs:
        kwargs["serial"] = True

    log.info("%s used in minimizer", method)
    method = method.lower().strip()
    if method == 'ciah':
        driver = minimize_CIAH
    elif method == 'cg':
        driver = minimize_CG
    elif method == 'bfgs':
        driver = minimize_BFGS
    elif method == 'trust-ncg':
        driver = minimize_NCG
    elif method == 'sd':
        driver = minimize_SD
    else:
        raise ValueError("Unknown method %s" % method)
    return driver(fn, x0, MaxIter=MaxIter, fgrad=fgrad, callback=callback, \
            ytol=ytol, gtol=gtol, dx_tol=dx_tol, **kwargs)

def minimize_SD(fn, x0, MaxIter=300, fgrad=None, callback=None, \
        ytol=1e-7, gtol=1e-3, dx_tol=1e-7, **kwargs):
    """
    Downhill minimize using steepest descent.
    """
    nx        = x0.shape[0]
    eps       = kwargs.get("eps", 1e-5)
    init_step = kwargs.get("init_step", 1.0)
    min_step  = [kwargs.get("min_step",  0.1)]
    xatol     = [kwargs.get("xatol", 1e-5)]

    def grad(x):
        if callback is not None:
            ref = callback(x)
            fn1 = lambda x1: fn(x1, ref=ref)
        else:
            fn1 = fn

        def gix(ix):
            dx = np.zeros_like(x)
            dx[ix] = eps
            return (0.5 / eps) * (fn1(x + dx) - fn1(x - dx))
        
        g = np.asarray([gix(x) for x in range(nx)])
        return g

    if fgrad is None:
        fgrad = grad
    
    def GetDir(y, g):
        h = 10 * g / y
        h2 = np.sum(h * h)
        return h * 10 / (1 + h2)
    
    log.debug(0, "  Iter           Value               Grad"
                 "                 Step           Step relative\n"
                 "------------------------------------------------"
                 "---------------------------------------")
    
    x = x0
    y = fn(x)
    steps = [init_step]
    converge_pattern = 0

    for iter in range(MaxIter):
        if (y < ytol * 0.1 and iter != 0):
            converge_pattern = 1
            break

        g = fgrad(x)
        if norm(g) < min(1e-5, gtol):
            converge_pattern = 2
            break
        
        dx = GetDir(y, g)

        if callback is None:
            LineSearchFn = lambda step: fn(x - step * dx)
        else:
            ref_ = callback(x)
            LineSearchFn = lambda step: fn(x - step * dx, ref_)

        def FindStep():
            scale = max(abs(np.average(steps[-2:])), min_step[0])
            res = minimize_scalar(LineSearchFn, bounds=(0.0, scale), \
                    method="bounded", options={"maxiter": 100, \
                    "xatol": xatol[0]})
            if res.fun > y: # line search is not accurate enough
                log.warn("line search fails, resulting value %20.12f is\n" \
                        "larger than the previous step value %20.12f", \
                        res.fun, y)
                res = (0.0, y)
            else:
                res = (res.x, res.fun)
            return res
        
        step, y_new = FindStep()
        steps.append(step)
        dx *= step
        
        # in case y is blow up
        if y_new > y * 1.5:
            log.debug(0, "y_new > y * 1.5, %s", y_new - y * 1.5)
            converge_pattern = 3
            break
        
        # y change is very small
        if abs(y - y_new) < ytol and norm(g) < gtol:
            log.debug(0, "abs(y - y_new) < %s, dy: %s, gnorm: %s", ytol, abs(y - y_new), norm(g))
            converge_pattern = 3
            x -= dx
            y = y_new
            break
        
        # x change is very small
        if norm(dx) < dx_tol:
            log.debug(0, "norm(dx) < dx_tol, %s", norm(dx))
            converge_pattern = 3
            x -= dx
            y = y_new
            break

        x -= dx
        y  = y_new
        normg = norm(g)
        log.debug(0, "%4d %20.12f %20.12f %20.12f %15.3e", iter, y, normg, \
                norm(dx), step)
    
    normg = norm(g)
    return x, y, converge_pattern, normg

def minimize_CG(fn, x0, MaxIter=300, fgrad=None, callback=None, ytol=1e-7, \
        gtol=1e-3, dx_tol=1e-7, **kwargs):
    return minimize_downhill(fn, x0, method='CG', MaxIter=MaxIter, fgrad=fgrad, \
            callback=callback, ytol=ytol, gtol=gtol, dx_tol=dx_tol, **kwargs)

def minimize_BFGS(fn, x0, MaxIter=300, fgrad=None, callback=None, ytol=1e-7, \
        gtol=1e-3, dx_tol=1e-7, **kwargs):
    return minimize_downhill(fn, x0, method='BFGS', MaxIter=MaxIter, fgrad=fgrad, \
            callback=callback, ytol=ytol, gtol=gtol, dx_tol=dx_tol, **kwargs)

def minimize_downhill(fn, x0, method='BFGS', MaxIter=300, fgrad=None, \
        callback=None, ytol=1e-7, gtol=1e-3, dx_tol=1e-7, **kwargs):
    """
    Downhill + line search.
    See Numerical Recipe 3rd ed. P517
    """
    nx        = x0.shape[0]
    eps       = kwargs.get("eps", 1e-5)
    init_step = kwargs.get("init_step", 1.0)
    min_step  = kwargs.get("min_step",  0.1)
    xatol     = kwargs.get("xatol", 1e-5)

    def grad(x):
        if callback is not None:
            ref = callback(x)
            fn1 = lambda x1: fn(x1, ref = ref)
        else:
            fn1 = fn

        def gix(ix):
            dx = np.zeros_like(x)
            dx[ix] = eps
            return (0.5 / eps) * (fn1(x + dx) - fn1(x - dx))
        g = np.asarray([gix(x) for x in range(nx)])
        return g

    if fgrad is None:
        fgrad = grad

    log.debug(0, "  Iter           Value               Grad"
                 "                 Step           Step relative\n"
                 "------------------------------------------------"
                 "---------------------------------------")
    if method == 'CG':
        driver = fit_hp._minimize_cg
    elif method == 'BFGS':
        driver = fit_hp._minimize_bfgs
    res = driver(fn, x0, jac=fgrad, callback=callback, \
            maxiter=MaxIter, disp=True, ytol=ytol, gtol=gtol, dx_tol=dx_tol, \
            norm=norm, eps=eps, init_step=init_step, min_step=min_step, \
            xatol=xatol)
    x = res.x
    y = res.fun
    g = res.jac
    return x, y, 3, norm(g)

def minimize_NCG(fn, x0, MaxIter=300, fgrad=None, callback=None, ytol=1e-7, \
        gtol=1e-3, dx_tol=1e-7, **kwargs):
    """
    Trust region newton CG, modified from scipy.optimize.
    """
    x0 = np.asarray(x0)
    nx = x0.shape[0]
    
    # ZHC NOTE check these values!
    initial_trust_radius = kwargs.get("initial_trust_radius", 1e-5) * np.sqrt(nx)
    max_trust_radius     = kwargs.get("max_trust_radius", 3e-3) * np.sqrt(nx)
    eta                  = kwargs.get("eta", 0.001)
    eps                  = kwargs.get("eps", 1e-5)

    if "serial" in kwargs.keys() and kwargs["serial"]:
        multi = False
    else:
        multi = True
        try:
            from pathos.multiprocessing import ProcessingPool, cpu_count
        except ImportError:
            multi = False

    if multi:
        log.debug(0, "Fitting: using %d cores to evaluate objective function", \
                cpu_count())
        p = ProcessingPool()
    else:
        log.debug(0, "Fitting: serial specified or failed" \
                " to load multiprocessing module, using single core")
    
    def grad(x):
        if callback is not None:
            ref = callback(x)
            fn1 = lambda x1: fn(x1, ref=ref)
        else:
            fn1 = fn

        def gix(ix):
            dx = np.zeros_like(x)
            dx[ix] = eps
            return (0.5 / eps) * (fn1(x + dx) - fn1(x - dx))
        if multi:
            g = np.asarray(p.map(gix, range(nx)))
        else:
            g = np.asarray([gix(x) for x in range(nx)])
        return g

    if fgrad is None:
        fgrad = grad

    def hessp(x, p):
        dp = p * eps
        res  = fgrad(x + dp)
        res -= fgrad(x - dp)
        res *= (0.5 / eps)
        return res

    log.debug(0, "NCG: initial_trust_radius: %.2e", initial_trust_radius)
    log.debug(0, "NCG: max_trust_radius: %.2e"    , max_trust_radius)
    log.debug(0, "  Iter           Value               Grad                 Step              Radius\n"
            "-----------------------------------------------------------------------------------------")
    res = fit_hp._minimize_trust_region(fn, x0, jac=fgrad, \
            hess=None, hessp=hessp, callback=callback, \
            maxiter=MaxIter, \
            disp=True, ytol=ytol, gtol=gtol, dx_tol=dx_tol, \
            initial_trust_radius=initial_trust_radius, \
            max_trust_radius=max_trust_radius, eta=eta)
    x = res.x
    y = res.fun
    g = res.jac
    return x, y, 3, norm(g)

def minimize_CIAH(fn, x0, MaxIter=300, fgrad=None, callback=None, ytol=1e-6, \
        gtol=1e-3, dx_tol=1e-6, **kwargs):
    """
    Trust region newton CG, modified from scipy.optimize.
    """
    x0 = np.asarray(x0)
    nx = x0.shape[0]
    
    # ZHC NOTE check these values!
    max_stepsize    = kwargs.get("max_stepsize", 0.005)
    ah_trust_region = kwargs.get("ah_trust_region", 2)
    eps             = kwargs.get("eps", 1e-5)
    num_cg_steps    = kwargs.get("num_cg_steps", 10)

    def grad(x):
        if callback is not None:
            ref = callback(x)
            fn1 = lambda x1: fn(x1, ref=ref)
        else:
            fn1 = fn

        def gix(ix):
            dx = np.zeros_like(x)
            dx[ix] = eps
            return (0.5 / eps) * (fn1(x + dx) - fn1(x - dx))
        g = np.asarray([gix(x) for x in range(nx)])
        return g

    if fgrad is None:
        fgrad = grad

    def hessp(x, p):
        dp = p * eps
        res  = fgrad(x + dp)
        res -= fgrad(x - dp)
        res *= (0.5 / eps)
        return res
    
    log.debug(0, "CIAH: CG stage. num_cg_steps = %d", num_cg_steps)
    log.debug(0, "  Iter           Value               Grad"
                 "                 Step           Step relative\n"
                 "------------------------------------------------"
                 "---------------------------------------")
    res = fit_hp._minimize_cg(fn, x0, jac=fgrad, callback=callback, \
            maxiter=num_cg_steps, disp=True, ytol=ytol, gtol=gtol, dx_tol=dx_tol, \
            norm=norm, eps=eps, init_step=1.0, min_step=0.2)
    if res.success:
        x = res.x
        y = res.fun
        g = res.jac
    else:
        x0 = res.x
        log.debug(0, "CIAH: max_stepsize: %.2e", max_stepsize)
        log.debug(0, "CIAH: ah_trust_region: %.2e", ah_trust_region)
        log.debug(0, "  Iter           Value               Grad                 Step             \n"
                "----------------------------------------------------------------------------------")
        
        myopt = CIAHMinimizer(fn, fgrad, hessp, h_diag=None)
        myopt.conv_tol        = ytol
        myopt.conv_tol_grad   = gtol
        myopt.max_cycle       = MaxIter
        myopt.max_stepsize    = max_stepsize
        myopt.ah_trust_region = ah_trust_region
        myopt.verbose         = 3
        
        x = myopt.kernel(x0)
        y = myopt.e
        g = myopt.norm_gorb
    return x, y, 3, norm(g)

if __name__ == "__main__":
    log.verbose = "DEBUG1"
    x0 = np.asarray([10., 20.])
    x, y, converge_pattern, _ = minimize(lambda x: x[0]**2 + x[1]**4 + 2*x[1]**2 + 2*x[0] + 2., \
            x0, MaxIter=300, method='SD')
    log.result("x = %s\ny=%20.12f", x, y)
    
    print ("CG:")
    x0 = np.asarray([10., 20.])
    x, y, converge_pattern, _ = minimize(lambda x: x[0]**2 + x[1]**4 + 2*x[1]**2 + 2*x[0] + 2., \
            x0, MaxIter=300, method='CG')
    log.result("x = %s\ny=%20.12f", x, y)
    
    print ("NCG:")
    x0 = np.asarray([10., 20.])
    x, y, converge_pattern, _ = minimize(lambda x: x[0]**2 + x[1]**4 + 2*x[1]**2 + 2*x[0] + 2., \
            x0, MaxIter=300, method='trust-NCG', max_trust_radius=1000.0)
    log.result("x = %s\ny=%20.12f", x, y)
    
    print ("CIAH:")
    x0 = np.asarray([10., 20.])
    x, y, converge_pattern, _ = minimize(lambda x: x[0]**2 + x[1]**4 + 2*x[1]**2 + 2*x[0] + 2., \
            x0, MaxIter=300, method='CIAH', max_stepsize=0.5, num_cg_steps=0)
    log.result("x = %s\ny=%20.12f", x, y)
