#! /usr/bin/env python
'''
Quadratic fitting of three points.

Author:
    Zhihao Cui <zhcui0408@gmail.com>
'''

import math
import cmath # for complex sqrt
import numpy as np
from scipy import stats
from libdmet.utils import logger as log

def get_parabola_vertex(x, y, tol=1e-12):
    """
    Give x = [x1, x2, x3], y = [y1, y2, y3]
    return a, b, c that y = ax^2 + bx + c.

    Args:
        x: [x1, x2, x3]
        y: [y1, y2, y3]

    Returns:
        a, b, c
        status: True if sucess.
    """
    x1, x2, x3 = x
    y1, y2, y3 = y
    denom = float((x1 - x2) * (x1 - x3) * (x2 - x3))
    if abs(denom) < tol:
        a = b = c = 0
        status = False
    else:
        a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
        b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
        c = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
        status = True
    return a, b, c, status

def get_roots(a, b, c, tol=1e-12):
    """
    Find roots for quadratic equation.

    Args:
        a, b, c
        tol: tolerance for zero

    Returns:
        roots: roots
        status: status 0: no root,
                       1: root1, root2
                       2: root1 (linear equation)
                       3: root1, root2 (complex).
    """
    
    if abs(a) < tol and abs(b) < tol:
        log.warn('a = 0, b = 0, not a quadratic equation.')
        status = 0
        return [], status

    if abs(a) < tol:
        log.warn('a = 0, single solution is x = %s', -c / b)
        status = 2
        return [-c / float(b)], status

    D = b ** 2 - 4.0 * a * c
    if D >= 0.0:
        root1 = (-b + np.sqrt(D)) / (2.0 * a)
        root2 = (-b - np.sqrt(D)) / (2.0 * a)
        status = 1
        return [root1, root2], status
    else:
        root1 = (-b + cmath.sqrt(D)) / (2.0 * a)
        root2 = (-b - cmath.sqrt(D)) / (2.0 * a)
        status = 3
        return [root1, root2], status

def quad_fit(mu, dnelecs, tol=1e-12):
    """
    Quadratic fit of mu and nelec.

    Args:
        mu: (3,)
        dnelecs: (3,) nelecs - target.
        tol: tolerance.

    Returns:
        mu_new: new mu.
        status: True for sucess.   
    """
    mu_lst      = np.array(mu, copy=True)
    dnelecs_lst = np.array(dnelecs, copy=True)
    assert len(mu_lst) == len(dnelecs_lst) and len(mu_lst) == 3

    idx1 = np.argsort(mu_lst, kind='mergesort')
    idx2 = np.argsort(dnelecs_lst, kind='mergesort')
    if not (idx1 == idx2).all():
        log.warn("dnelecs is not a monotonic function of mu...")
    mu_lst = mu_lst[idx1]
    dnelecs_lst = dnelecs_lst[idx1]
    
    a, b, c, status = get_parabola_vertex(mu_lst, dnelecs_lst, tol=tol)
    if not status:
        log.warn("Duplicated points among three dots:\nmu:     %s\ndnelec: %s", \
                mu_lst, dnelecs_lst)
        return 0, False
    
    roots, status = get_roots(a, b, c, tol=tol)
    if status == 0:
        log.warn("Root finding error")
        return 0, False
    elif status == 2:
        mu_new = roots[0]
        return mu_new, True
    elif status == 3:
        if abs(roots[0].imag) + abs(roots[1].imag) > 1e-3:
            log.warn('Complex root finding:\nroot1: %s\nroot2: %s',\
                    roots[0], roots[1])
            return 0, False
        else:
            roots = [roots[0].real, roots[1].real]
     
    if dnelecs_lst[0] >= 0.0:
        left  = -np.inf
        right = mu_lst[0]
    elif dnelecs_lst[1] >= 0.0:
        left  = mu_lst[0]
        right = mu_lst[1]
    elif dnelecs_lst[2] >= 0.0:
        left  = mu_lst[1]
        right = mu_lst[2]
    else:
        left  = mu_lst[2]
        right = np.inf
    
    if roots[0] < right and roots[0] > left:
        if roots[1] < right and roots[1] > left: 
            if abs(roots[0] - mu[0]) < abs(roots[1] - mu[0]):
                return roots[0], True
            else:
                return roots[1], True
        else:
            return roots[0], True
    else:
        if roots[1] < right and roots[1] > left: 
            return roots[1], True
        else:
            log.warn("Can not find proper root within the range, "
                      "[%15.6f, %15.6f] \nroots: %s", left, right, roots)
            return 0, False

def has_duplicate(dmu, mus, tol=1e-7):
    dmus_abs = np.abs(mus - dmu)
    return (dmus_abs < tol).any()

def violate_previous_mu(dmu, mus, target, nelecs):
    x = dmu - mus
    y = target - nelecs
    return ((x * y) < 0.0).any()

def quad_fit_mu(mus, nelecs, filling, step):
    """
    Use quadratic fitting to predict chemical potential.

    Args:
        mus: a list of old mu
        nelecs: a list of old nelectron number
        filling: filling * 2.0 is the target nelec.
        step: max trust step

    Returns:
        dmu: the change of mu.
    """
    mus = np.asarray(mus)
    nelecs = np.asarray(nelecs)
    log.info("use quadratic fitting # %d", len(mus) - 2)
 
    target = filling * 2.0
    dnelec = nelecs - target
    dnelec_abs = np.abs(dnelec)
    
    # get three nearest points
    idx_dnelec  = np.argsort(dnelec_abs, kind='mergesort')
    mus_sub     = mus[idx_dnelec][:3]
    dnelec_sub  = dnelec[idx_dnelec][:3]
   
    # quadratic fit
    dmu, status = quad_fit(mus_sub, dnelec_sub, tol=1e-12)
    
    # check duplicates
    if has_duplicate(dmu, mus):
        log.info("duplicate in extrapolation.")
        status = False

    if not status:
        log.info("quadratic fails or duplicates, use linear regression.")
        slope, intercept, r_value, p_value, std_err = stats.linregress(dnelec_sub, mus_sub)
        dmu = intercept
    
    # check monotonic for the predict mu:
    if violate_previous_mu(dmu, mus, target, nelecs):
        log.info("predicted mu violates previous mus. Try linear regression.")
        slope, intercept, r_value, p_value, std_err = stats.linregress(dnelec_sub, mus_sub)
        dmu = intercept
        if violate_previous_mu(dmu, mus, target, nelecs):
            log.info("predicted mu from linear regression also violates. use finite step (%10.5g).",
                     max(step, 1e-3))
            dmu = math.copysign(max(step, 1e-3), (target - nelecs[-1])) +  mus[-1]

    if abs(dmu - mus[-1]) > step:
        log.info("extrapolation dMu %20.12f more than trust step %20.12f", dmu - mus[-1], step)
        dmu = math.copysign(step, dmu - mus[-1]) + mus[-1]
   
    # TODO determine the range mu should be and use middle point to predict next mu.
    # check duplicates
    if has_duplicate(dmu, mus):
        log.info("duplicate in extrapolation.")
        dmu = math.copysign(step, dmu - mus[-1]) + mus[-1]
    
    if (dmu - mus[-1]) * (target - nelecs[-1]) < 0 and abs(dmu - mus[-1]) > 2e-3 :
        log.info("extrapolation gives wrong direction, use finite difference")
        dmu = math.copysign(step, (target - nelecs[-1])) +  mus[-1]
    
    log.result("extrapolated to dMu = %20.12f", dmu)
    return dmu

if __name__ == '__main__':
    x1, y1 = 6.035156250e-01, 1.508072e+00
    x2, y2 = 5.432540625e-01, 1.488130e+00
    x3, y3 = 4.890186562e-01, 1.465602e+00

    x = [x1, x2, x3]
    y = [y1, y2, y3]

    mu = x
    deltaN = y
    mu_new, status = quad_fit(mu, deltaN, tol = 1e-12)

    print (mu_new)
    print (status)
