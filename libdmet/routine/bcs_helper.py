#! /usr/bin/env python

from functools import reduce
import itertools as it

import numpy as np
import scipy.linalg as la
from scipy.optimize import brentq

from libdmet.utils import logger as log
from libdmet.utils.misc import mdot
from libdmet.settings import IMAG_DISCARD_TOL 

def extractRdm(GRho):
    """
    Generalized density matrix convention:
    GRho = [[rho_A k_ba^dg]
            [k_ba  1-rho_B]]
    k_ba = - k_ab^dg
    """
    norbs = GRho.shape[0] // 2
    log.eassert(norbs * 2 == GRho.shape[0], \
            "generalized density matrix dimension error")
    rhoA = GRho[:norbs, :norbs].copy()
    rhoB = np.eye(norbs) - GRho[norbs:, norbs:]
    kappaBA = GRho[norbs:, :norbs].copy()
    return rhoA, rhoB, kappaBA

def extractH1(GFock):
    norbs = GFock.shape[0] // 2
    log.eassert(norbs * 2 == GFock.shape[0], \
            "generalized density matrix dimension error")
    HA  = GFock[:norbs, :norbs].copy()
    HB  = -GFock[norbs:, norbs:]
    HDT = GFock[norbs:,:norbs].copy()
    return HA, HB, HDT

def combineRdm(rhoA, rhoB, kappaAB):
    norbs = rhoA.shape[0]
    return np.block([[rhoA,                 -kappaAB],
                     [-kappaAB.T, np.eye(norbs)-rhoB]])

def swapSpin(GRho):
    rhoA, rhoB, kappaBA = extractRdm(GRho)
    norbs = rhoA.shape[0]
    return np.block([[      rhoB,           -kappaBA],
                     [-kappaBA.T, np.eye(norbs)-rhoA]])

def basisToCanonical(basis):
    assert(basis.shape[0] == 2)
    shape = list(basis.shape[1:])
    nbasis = shape[-1]
    nsites = shape[-2] // 2
    shape[-1] *= 2
    newbasis = np.empty(tuple(shape))
    newbasis[...,:nbasis] = basis[0]
    newbasis[...,:nsites,nbasis:], newbasis[...,nsites:,nbasis:] = \
            basis[1,...,nsites:,:], basis[1,...,:nsites,:]
    return newbasis

def basisToSpin(basis):
    shape = [2] + list(basis.shape)
    shape[-1] = shape[-1] // 2
    nbasis = shape[-1]
    nsites = shape[-2] // 2
    newbasis = np.empty(tuple(shape))
    newbasis[0] = basis[...,:nbasis]
    newbasis[1,...,:nsites,:], newbasis[1,...,nsites:,:] = \
            basis[...,nsites:,nbasis:], basis[...,:nsites,nbasis:]
    return newbasis

def mono_fit(fn, y0, x0, thr, increase=True, dx=1.0, verbose=True):
    if not increase:
        return mono_fit(lambda x: -fn(x), -y0, x0, thr, True)

    from libdmet.utils.misc import counted
    @counted
    def evaluate(xx):
        yy = fn(xx)
        if verbose:
            log.debug(1, "Iter %2d, x = %20.12f, f(x) = %20.12f", \
                    evaluate.count-1, xx, yy)
        return yy
    
    if verbose:
        log.debug(0, "target f(x) = %20.12f", y0)
    # first section search
    x = x0
    y = evaluate(x)
    if abs(y - y0) < thr:
        return x

    if y > y0:
        dx = -dx

    while 1:
        x1 = x + dx
        y1 = evaluate(x1)
        if abs(y1 - y0) < thr:
            return x1
        if (y-y0) * (y1-y0) < 0:
            break
        else:
            x = x1
            y = y1

    if x < x1:
        sec_x, sec_y = [x, x1], [y, y1]
    else:
        sec_x, sec_y = [x1, x], [y1, y]

    while sec_x[1] - sec_x[0] > 0.1 * thr:
        f = (y0-sec_y[0]) / (sec_y[1] - sec_y[0])
        if f < 0.2:
            f = 0.2
        elif f > 0.8:
            f = 0.8
        x1 = sec_x[0] * (1.-f) + sec_x[1] * f
        y1 = evaluate(x1)
        if abs(y1 - y0) < thr:
            return x1
        if (y1 - y0) * (sec_y[0] - y0) < 0:
            sec_x = [sec_x[0], x1]
            sec_y = [sec_y[0], y1]
        else:
            sec_x = [x1, sec_x[1]]
            sec_y = [y1, sec_y[1]]
    
    return 0.5 * (sec_x[0] + sec_x[1])

def mono_fit_2(fn, y0, x0, thr, increase=True, dx=1.0, verbose=True, maxiter=1000):
    if not increase:
        return mono_fit(lambda x: -fn(x), -y0, x0, thr, True)
    
    if verbose:
        log.debug(0, "target f(x) = %20.12f", y0)
    x = x0
    y = fn(x)
    if abs(y - y0) < thr:
        return x

    if y > y0:
        dx = -dx

    for i in range(maxiter * 50):
        x1 = x + dx
        y1 = fn(x1)
        if abs(y1 - y0) < thr:
            return x1
        if (y-y0) * (y1-y0) < 0:
            break
        else:
            x = x1
            y = y1
    else:
        raise RuntimeError("Cannot find the section.")

    if x < x1:
        sec_x = [x, x1]
    else:
        sec_x = [x1, x]
    
    def error(xx):
        yy = fn(xx)
        return yy - y0
    
    res = brentq(error, sec_x[0], sec_x[1], xtol=thr, rtol=thr,
                 maxiter=maxiter, full_output=True, disp=False)
    mu = res[0]
    if not res[1].converged:
        log.warn("mono_fit_2: brentq fails. x: %s, y: %s", mu, error(mu))
    if verbose:
        log.debug(2, "mono_fit_2: brentq res:\n%s", res[1])
    return mu

def separate_basis(basis):
    nscsites = basis.shape[2] // 2
    # VA, VB, UA, UB
    return basis[0, :, :nscsites], basis[1, :, :nscsites], \
            basis[1, :, nscsites:], basis[0, :, nscsites:]

def contract_trans_inv(basisL, basisR, lattice, H):
    ncells = lattice.ncells
    nbasisL = basisL.shape[2]
    nbasisR = basisR.shape[2]
    res = np.zeros((nbasisL, nbasisR))
    for i, j in it.product(range(ncells), repeat=2):
        res += mdot(basisL[i].T, H[lattice.subtract(i, j)], basisR[j])
    return res

def transform_trans_inv(basis, lattice, H):
    VA, VB, UA, UB = separate_basis(basis)
    if H.ndim == 3:
        HA = HB = H
        D = np.zeros_like(HA)
    elif H.shape[0] == 2:
        HA, HB = H[0], H[1]
        D = np.zeros_like(HA)
    elif H.shape[0] == 3:
        HA, HB, D = H[0], H[1], H[2]
    D_T = lattice.transpose(D)
    ctr = lambda L, H, R: contract_trans_inv(L, R, lattice, H)
    resHA = ctr(VA, HA, VA) - ctr(UB, HB, UB) + ctr(VA, D, UB) + ctr(UB, D_T, VA)
    resHB = ctr(VB, HB, VB) - ctr(UA, HA, UA) - ctr(VB, D_T, UA) - ctr(UA, D, VB)
    resD  = ctr(VA, HA, UA) - ctr(UB, HB, VB) + ctr(VA, D, VB) + ctr(UB, D_T, UA)
    resE0 = np.trace(ctr(UA, HA, UA) + ctr(UB, HB, UB) + ctr(UA, D, VB) + ctr(VB, D_T, UA))
    return np.asarray((resHA, resHB)), resD, resE0

def contract_trans_inv_sparse(basisL, basisR, lattice, H, thr = 1e-7):
    ncells = lattice.ncells
    nbasisL = basisL.shape[2]
    nbasisR = basisR.shape[2]
    res = np.zeros((nbasisL, nbasisR))
    from libdmet.utils.misc import find
    mask_basisL = set(find(True, map(lambda a: la.norm(a) > thr, basisL)))
    mask_basisR = set(find(True, map(lambda a: la.norm(a) > thr, basisR)))
    mask_H = set(find(True, map(lambda a: la.norm(a) > thr, H)))

    for i, k in it.product(mask_basisL, mask_H):
        #j = lattice.add(i, k) 
        j = lattice.subtract(i, k) # J=I-K -> K=I-J=I,J
        if j in mask_basisR:
            res += mdot(basisL[i].T, H[k], basisR[j])
    #for i, j in it.product(mask_basisL, mask_basisR):
    #    Hidx = lattice.subtract(i, j)
    #    if Hidx in mask_H:
    #        res += mdot(basisL[i].T, H[Hidx], basisR[j])
    return res

def transform_trans_inv_sparse(basis, lattice, H, thr = 1e-7):
    VA, VB, UA, UB = separate_basis(basis)
    if H.ndim == 3:
        HA = HB = H
        D = np.zeros_like(HA)
    elif H.shape[0] == 2:
        HA, HB = H[0], H[1]
        D = np.zeros_like(HA)
    elif H.shape[0] == 3:
        HA, HB, D = H[0], H[1], H[2]
    D_T = lattice.transpose(D)
    ctr = lambda L, H, R: contract_trans_inv_sparse(L, R, lattice, H, thr = thr)
    resHA = ctr(VA, HA, VA) - ctr(UB, HB, UB) + ctr(VA, D, UB) + ctr(UB, D_T, VA)
    resHB = ctr(VB, HB, VB) - ctr(UA, HA, UA) - ctr(VB, D_T, UA) - ctr(UA, D, VB)
    resD = ctr(VA, HA, UA) - ctr(UB, HB, VB) + ctr(VA, D, VB) + ctr(UB, D_T, UA)
    resE0 = np.trace(ctr(UA, HA, UA) + ctr(UB, HB, UB) + ctr(UA, D, VB) + ctr(VB, D_T, UA))
    return np.asarray((resHA, resHB)), resD, resE0

def contract_local(basisL, basisR, lattice, H):
    ncells = lattice.ncells
    return reduce(np.ndarray.__add__, \
            map(lambda i: mdot(basisL[i].T, H, basisR[i]), range(ncells)))

def transform_local(basis, lattice, H):
    VA, VB, UA, UB = separate_basis(basis)
    if len(H.shape) == 2:
        HA = HB = H
        D = np.zeros_like(HA)
    elif H.shape[0] == 2:
        HA, HB = H[0], H[1]
        D = np.zeros_like(HA)
    elif H.shape[0] == 3:
        HA, HB, D = H[0], H[1], H[2]
    ctr = lambda L, H, R: contract_local(L, R, lattice, H)
    resHA = ctr(VA, HA, VA) - ctr(UB, HB, UB) + ctr(VA, D, UB) + ctr(UB, D.T, VA)
    resHB = ctr(VB, HB, VB) - ctr(UA, HA, UA) - ctr(VB, D.T, UA) - ctr(UA, D, VB)
    resD = ctr(VA, HA, UA) - ctr(UB, HB, VB) + ctr(VA, D, VB) + ctr(UB, D.T, UA)
    resE0 = np.trace(ctr(UA, HA, UA) + ctr(UB, HB, UB) + ctr(UA, D, VB) + ctr(VB, D.T, UA))
    return np.asarray((resHA, resHB)), resD, resE0

def contract_local_grad(basisL, basisR, lattice):
    ncells = lattice.ncells
    return reduce(np.ndarray.__add__, \
            map(lambda i: np.tensordot(basisL[i], basisR[i], axes = 0),\
            range(ncells))).transpose((0, 2, 1, 3))
            # ipjq- > ijpq

def contract_local_grad_DT(basisL, basisR, lattice):
    # special treatment for dV / dD, for DT terms which include D_ij
    ncells = lattice.ncells
    return reduce(np.ndarray.__add__, \
            map(lambda i: np.tensordot(basisL[i], basisR[i], axes = 0),\
            range(ncells))).transpose((2, 0, 1, 3))
            # jpiq- > ijpq

def transform_local_grad_A(sep_basis, lattice):
    VA, VB, UA, UB = sep_basis
    ctr = lambda L, R: contract_local_grad(L, R, lattice)
    resHA = ctr(VA, VA) 
    resHB = -ctr(UA, UA)
    resD = ctr(VA, UA)
    resE0 = None
    return np.asarray([resHA, resHB]), resD, resE0

def transform_local_grad_B(sep_basis, lattice):
    VA, VB, UA, UB = sep_basis
    ctr = lambda L, R: contract_local_grad(L, R, lattice)
    resHA = -ctr(UB, UB)
    resHB = ctr(VB, VB)
    resD = -ctr(UB, VB)
    resE0 = None
    return np.asarray([resHA, resHB]), resD, resE0

def transform_local_grad_D(sep_basis, lattice):
    VA, VB, UA, UB = sep_basis
    ctr = lambda L, R: contract_local_grad(L, R, lattice)
    ctr_DT = lambda L, R: contract_local_grad_DT(L, R, lattice)
    resHA = ctr(VA, UB) + ctr_DT(UB, VA)
    resHB = -ctr_DT(VB, UA) -ctr(UA, VB)
    resD = ctr(VA, VB) + ctr_DT(UB, UA)
    resE0 = None
    return np.asarray([resHA, resHB]), resD, resE0

def transform_local_grad(basis, lattice):
    sep_basis = separate_basis(basis)
    return transform_local_grad_A(sep_basis, lattice), transform_local_grad_B(sep_basis, lattice), \
            transform_local_grad_D(sep_basis, lattice)

def contract_local_sparseH(basisL, basisR, lattice, H, thr = 1e-7):
    ncells = lattice.ncells
    nbasisL = basisL.shape[-1]
    nbasisR = basisR.shape[-1]
    res = np.zeros((nbasisL, nbasisR))
    mask_H = np.nonzero(abs(H) > thr)
    mask_H = zip(*map(lambda a: a.tolist(), mask_H))
    for j, k in mask_H:
        res += np.dot(basisL[:,j].T, basisR[:,k]) * H[j,k]
    return res

def transform_local_sparseH(basis, lattice, H, thr = 1e-7):
    VA, VB, UA, UB = separate_basis(basis)
    if len(H.shape) == 2:
        HA = HB = H
        D = np.zeros_like(HA)
    elif H.shape[0] == 2:
        HA, HB = H[0], H[1]
        D = np.zeros_like(HA)
    elif H.shape[0] == 3:
        HA, HB, D = H[0], H[1], H[2]
    ctr = lambda L, H, R: contract_local_sparseH(L, R, lattice, H, thr = 1e-7)
    resHA = ctr(VA, HA, VA) - ctr(UB, HB, UB) + ctr(VA, D, UB) + ctr(UB, D.T, VA)
    resHB = ctr(VB, HB, VB) - ctr(UA, HA, UA) - ctr(VB, D.T, UA) - ctr(UA, D, VB)
    resD = ctr(VA, HA, UA) - ctr(UB, HB, VB) + ctr(VA, D, VB) + ctr(UB, D.T, UA)
    resE0 = np.trace(ctr(UA, HA, UA) + ctr(UB, HB, UB) + ctr(UA, D, VB) + ctr(VB, D.T, UA))
    return np.asarray((resHA, resHB)), resD, resE0

def transform_imp(basis, lattice, H):
    VA, VB, UA, UB = separate_basis(basis)
    if len(H.shape) == 2:
        HA = HB = H
        D = np.zeros_like(HA)
    elif H.shape[0] == 2:
        HA, HB = H[0], H[1]
        D = np.zeros_like(HA)
    elif H.shape[0] == 3:
        HA, HB, D = H[0], H[1], H[2]
    ctr = lambda L, H, R: mdot(L[0].T, H, R[0])
    resHA = ctr(VA, HA, VA) - ctr(UB, HB, UB) + ctr(VA, D, UB) + ctr(UB, D.T, VA)
    resHB = ctr(VB, HB, VB) - ctr(UA, HA, UA) - ctr(VB, D.T, UA) - ctr(UA, D, VB)
    resD = ctr(VA, HA, UA) - ctr(UB, HB, VB) + ctr(VA, D, VB) + ctr(UB, D.T, UA)
    resE0 = np.trace(ctr(UA, HA, UA) + ctr(UB, HB, UB) + ctr(UA, D, VB) + ctr(VB, D.T, UA))
    return np.asarray((resHA, resHB)), resD, resE0

def contract_imp_env(basisL, basisR, lattice, H):
    ncells = lattice.ncells
    res1 = reduce(np.ndarray.__add__, \
            map(lambda i: mdot(basisL[0].T, H[i], basisR[i]), range(ncells)))
    res2 = reduce(np.ndarray.__add__, \
            map(lambda i: mdot(basisL[i].T, H[lattice.subtract(i, 0)], basisR[0]), \
            range(ncells)))
    return 0.5 * (res1 + res2)

def transform_imp_env(basis, lattice, H):
    VA, VB, UA, UB = separate_basis(basis)
    if len(H.shape) == 3:
        HA = HB = H
        D = np.zeros_like(HA)
    elif H.shape[0] == 2:
        HA, HB = H[0], H[1]
        D = np.zeros_like(HA)
    elif H.shape[0] == 3:
        HA, HB, D = H[0], H[1], H[2]
    D_T = lattice.transpose(D)
    ctr = lambda L, H, R: contract_imp_env(L, R, lattice, H)
    resHA = ctr(VA, HA, VA) - ctr(UB, HB, UB) + ctr(VA, D, UB) + ctr(UB, D_T, VA)
    resHB = ctr(VB, HB, VB) - ctr(UA, HA, UA) - ctr(VB, D_T, UA) - ctr(UA, D, VB)
    resD = ctr(VA, HA, UA) - ctr(UB, HB, VB) + ctr(VA, D, VB) + ctr(UB, D_T, UA)
    resE0 = np.trace(ctr(UA, HA, UA) + ctr(UB, HB, UB) + ctr(UA, D, VB) + ctr(VB, D_T, UA))
    return np.asarray([resHA, resHB]), resD, resE0
    
def get_dV_dparam(basis, lattice, vcor):
    # this is a differential version of dV_dparam
    # and is faster
    def sym_triu(a):
        a = a + a.transpose((1, 0, 2, 3))
        a[np.arange(a.shape[0]), np.arange(a.shape[1])] *= 0.5
        return a[np.triu_indices(a.shape[0])]
    
    nbasis = basis.shape[-1]
    dV_dp = np.empty((vcor.length(), nbasis*2, nbasis*2))
    resA, resB, resD = transform_local_grad(basis, lattice)
    (resA_A, resA_B), resA_D, _ = resA
    resA_A = sym_triu(resA_A)
    resA_B = sym_triu(resA_B)
    resA_D = sym_triu(resA_D)

    (resB_A, resB_B), resB_D, _ = resB
    resB_A = sym_triu(resB_A)
    resB_B = sym_triu(resB_B)
    resB_D = sym_triu(resB_D)
    
    (resD_A, resD_B), resD_D, _ = resD
    resD_A = resD_A.reshape((-1,) + resD_A.shape[-2:]) 
    resD_B = resD_B.reshape((-1,) + resD_B.shape[-2:]) 
    resD_D = resD_D.reshape((-1,) + resD_D.shape[-2:]) 
    
    dA_dV_collect = reduce(lambda x, y : np.append(x, y, axis = 0), \
            (resA_A, resB_A, resD_A)) 
    dB_dV_collect = reduce(lambda x, y : np.append(x, y, axis = 0), \
            (resA_B, resB_B, resD_B)) 
    dD_dV_collect = reduce(lambda x, y : np.append(x, y, axis = 0), \
            (resA_D, resB_D, resD_D))
    
    for ip in range(vcor.length()):
        dA_dV, dB_dV, dD_dV = \
                dA_dV_collect[ip], dB_dV_collect[ip], dD_dV_collect[ip]
        dV_dp[ip, :nbasis, :nbasis] = dA_dV
        dV_dp[ip, nbasis:, nbasis:] = -dB_dV
        dV_dp[ip, :nbasis, nbasis:] = dD_dV
        dV_dp[ip, nbasis:, :nbasis] = dD_dV.T
    return dV_dp
