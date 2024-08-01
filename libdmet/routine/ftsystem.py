#! /usr/bin/env python

"""
Routines for finite temperature.

Author:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

import numpy as np
import scipy
from scipy.optimize import brentq
import scipy.linalg as la

from pyscf import lib

from libdmet.utils import logger as log
from libdmet.utils.misc import *
from libdmet.basis_transform import make_basis

FIT_TOL = 1e-12
ZERO_TOL = 1e-10

def fermi_smearing_occ(mu, mo_energy, beta, ncore=0, nvirt=0):
    """
    Fermi smearing function for mo_occ.
    By using broadcast, mu can be a list of values 
    (i.e. each sector can have different mu)
    
    Args:
        mu: chemical potential, can have shape (), (1,), (spin,).
        mo_energy: orbital energy, can be (nmo,) or (s, k, ..., nmo) array.
        beta: inverse temperature, float.
        ncore: number of core orbitals with occupation 1.
        nvirt: number of virt orbitals with occupation 0.

    Returns:
        occ: orbital occupancy, the same shape as mo_energy.
    """
    mo_energy = np.asarray(mo_energy)
    mu = np.asarray(mu).reshape(-1, *([1] * (mo_energy.ndim - 1)))
    de = beta * (mo_energy - mu) 
    occ = np.zeros_like(mo_energy)
    idx = (de < 100)
    if ncore != 0:
        assert mo_energy.ndim == 1
        idx[:ncore] = False 
        occ[:ncore] = 1.0
    if nvirt != 0:
        assert mo_energy.ndim == 1
        idx[-nvirt:] = False
    
    occ[idx] = 1.0 / (np.exp(de[idx]) + 1.0)
    return occ

def gaussian_smearing_occ(mu, mo_energy, beta, ncore=0, nvirt=0):
    """
    Gaussian smearing function for mo_occ.

    Args:
        mu: chemical potential, can have shape (), (1,), (spin,).
        mo_energy: orbital energy, can be (nmo,) or (s, k, ..., nmo) array.
        beta: inverse temperature, float.

    Returns:
        occ: orbital occupancy, the same shape as mo_energy.
    """
    mo_energy = np.asarray(mo_energy)
    mu = np.asarray(mu).reshape(-1, *([1] * (mo_energy.ndim - 1)))
    return 0.5 * scipy.special.erfc((mo_energy - mu) * beta)

def find_mu(nelec, mo_energy, beta, mu0=None, f_occ=fermi_smearing_occ, 
            tol=FIT_TOL, ncore=0, nvirt=0):
    """
    Find chemical potential mu for a target nelec.
    Assume mo_energy has no spin dimension.
    
    Returns:
        mu: chemical potential.
    """
    def nelec_cost_fn_brentq(mu):
        mo_occ = f_occ(mu, mo_energy, beta, ncore=ncore, nvirt=nvirt)
        return mo_occ.sum() - nelec
    
    nelec_int = int(np.round(nelec))
    if nelec_int >= len(mo_energy):
        lval = mo_energy[-1] - (1.0 / beta)
        rval = mo_energy[-1] + max(10.0, 1.0 / beta)
    elif nelec_int <= 0:
        lval = mo_energy[0]  - max(10.0, 1.0 / beta)
        rval = mo_energy[0]  + (1.0 / beta)
    else:
        lval = mo_energy[nelec_int - 1] - (1.0 / beta)
        rval = mo_energy[nelec_int]     + (1.0 / beta)
    
    # for the corner case where all empty or all occupied
    if nelec_cost_fn_brentq(lval) * nelec_cost_fn_brentq(rval) > 0:
        lval -= max(100.0, 1.0 / beta)
        rval += max(100.0, 1.0 / beta)
    res = brentq(nelec_cost_fn_brentq, lval, rval, xtol=tol, rtol=tol,
                 maxiter=10000, full_output=True, disp=False)
    if (not res[1].converged):
        log.warn("fitting mu (fermi level) brentq fails.")
    mu = res[0]
    return mu

def find_mu_by_density(density, mo_energy, beta, mu0=None, 
                       f_occ=fermi_smearing_occ, tol=FIT_TOL, ncore=0,
                       nvirt=0):
    norb = mo_energy.size
    nelec = density * norb
    return find_mu(nelec, mo_energy, beta, mu0=mu0, f_occ=f_occ, tol=tol*norb,
                   ncore=ncore, nvirt=nvirt)

def kernel(h, nelec, beta, mu0=None, fix_mu=False):
    mo_energy, mo_coeff = la.eigh(h)
    if not fix_mu:
        mu = find_mu(nelec, mo_energy, beta, mu0)
    else:
        mu = mu0
    mo_occ = fermi_smearing_occ(mu, mo_energy, beta)
    return mo_energy, mo_coeff, mo_occ, mu

def make_rdm1(mo_coeff, mo_occ):
    return (mo_coeff*mo_occ).dot(mo_coeff.conj().T)

def get_h_random(norb, seed = None):
    if seed is not None:
        np.random.seed(seed)
    h = np.random.random((norb, norb))
    h = h + h.T.conj()
    return h

def get_h_random_deg(norb, deg_orbs=[], deg_energy=[], seed=None):
    if seed is not None: 
        np.random.seed(seed)
    h = np.random.random((norb, norb))
    h = h + h.T.conj()
    mo_energy, mo_coeff = la.eigh(h)

    for i in range(len(deg_orbs)):
        mo_energy[deg_orbs[i]] = deg_energy[i]

    h = (mo_coeff*mo_energy).dot(mo_coeff.T.conj())
    return h

def get_rho_grad(mo_energy, mo_coeff, mu, beta, fix_mu=True, compact=False):
    """
    full gradient corresponding to rho change term.
    d rho_{ij} / d v_{kl} [where kl is tril part of the potential]

    Math:
        d rho_ij / d v_kl = partial rho_ij / partial v_kl 
            + partial rho_ij / partial mu * partial mu / partial v_kl

    Args:
        mo_energy
        mo_coeff
        mu
        beta
        fix_mu
        compact

    Returns:
        drho_dv: rho's reponse to v, 
                 shape (nao_pair, nao, nao) or (nao_pair, nao_pair) if compact
                 the first is the tril indices of v, the later is rho.
    """
    norb = mo_coeff.shape[-1]
    rho_elec = fermi_smearing_occ(mu, mo_energy, beta)
    rho_hole = 1.0 - rho_elec
    
    # ep - eq matrix
    de_mat = mo_energy[:, None] - mo_energy
    zero_mask = np.abs(de_mat) < ZERO_TOL
    nonzero_mask = ~zero_mask
    de_mat_inv = np.zeros_like(de_mat)
    de_mat_inv[nonzero_mask] = 1.0 / de_mat[nonzero_mask]
    
    # K_{pq}
    K = de_mat_inv * (rho_elec - rho_elec[:, None])
    K[zero_mask] = (rho_elec[:, None] * rho_hole)[zero_mask] * beta

    # **compute drho / dv
    #rho_grad = -np.einsum('mp, lp, pq, sq, nq -> lsmn', \
    #   mo_coeff, mo_coeff.conj(), K, mo_coeff, mo_coeff.conj()) #[slower]
    mo_coeff_conj = mo_coeff.conj()
    scr = np.einsum('lp, mp -> lmp', mo_coeff_conj, mo_coeff)
    rho_grad = -np.dot(scr, K)
    #rho_grad = np.einsum('lmq, nsq -> lsmn', rho_grad, scr)
    rho_grad = np.tensordot(rho_grad, scr, axes = ((-1,), (-1,))).transpose((0, 3, 1, 2))

    # **symmetrize
    rho_grad = rho_grad + rho_grad.transpose(1, 0, 2, 3)
    rho_grad[np.arange(norb), np.arange(norb)] *= 0.5
    rho_grad = rho_grad[np.tril_indices(norb)]

    # contribution from mu change
    if not fix_mu:
        f = rho_elec * rho_hole    
        f_sum = f.sum()
        if abs(f_sum) > ZERO_TOL: # not almost zero T
            # partial rho_ij / partial mu
            drho_dmu = np.dot(mo_coeff * f, mo_coeff.conj().T)
            drho_dmu *= beta
            
            # partial mu / partial v_{kl}
            E_grad = np.einsum('ki, li -> kli', mo_coeff.conj(), mo_coeff)
            mu_grad = np.dot(E_grad, f) / (f.sum())
            mu_grad = mu_grad + mu_grad.T
            mu_grad[np.arange(norb), np.arange(norb)] *= 0.5
            mu_grad = mu_grad[np.tril_indices(norb)]
            
            # partial rho_{ij} / partial mu * partial mu / partial v_{kl}
            rho_grad_mu_part = np.einsum('k, ij -> kij', mu_grad, drho_dmu)
            rho_grad += rho_grad_mu_part
        
    if compact:
        rho_grad = rho_grad.transpose(1, 2, 0)[np.tril_indices(norb)].transpose(1, 0)
    
    return rho_grad

def get_dw_dv(mo_energy, mo_coeff, drho, mu, beta, fix_mu=True, compact=False,
              fit_idx=None):
    """
    Full finite T gradient of dw over dv, i.e. dnorm_dv in 0 T code.
    
    In a real dmet calculation, we should use:
    dw/dparam = dw/dv * dv/dparam
    This function is for dw/dv = dw/drho * drho/dv.
    Assume orthogonal basis

    Returns:
        dw_dv: d w / d v_{kl} with shape (spin, norb, norb) or 
               (spin, norb_pair) if compact=True
    """
    if mo_coeff.ndim == 2:
        mo_energy = mo_energy[None]
        mo_coeff = mo_coeff[None]
        drho = drho[None]
    spin, _, norb = mo_coeff.shape
    if fit_idx is None:
        fit_idx = range(norb)

    rho_elec = fermi_smearing_occ(mu, mo_energy, beta)
    rho_hole = 1.0 - rho_elec

    # ep - eq matrix
    de_mat = np.zeros((spin, norb, norb))
    for s in range(spin):
        de_mat[s] = mo_energy[s, :, None] - mo_energy[s]
    zero_mask = np.abs(de_mat) < ZERO_TOL
    nonzero_mask = ~zero_mask
    de_mat_inv = np.zeros_like(de_mat)
    de_mat_inv[nonzero_mask] = 1.0 / de_mat[nonzero_mask]
    
    # K_{pq}
    dw_dv = np.zeros((spin, norb, norb), dtype=mo_coeff.dtype)
    for s in range(spin):
        K = de_mat_inv[s] * (rho_elec[s, :, None] - rho_elec[s])
        K[zero_mask[s]] = (rho_elec[s, :, None] * rho_hole[s])[zero_mask[s]] * (-beta)
        # pk, kl, lq -> pq; pq, pq -> pq
        tmp  = mdot(mo_coeff[s, fit_idx].T, 2.0*drho[s], mo_coeff[s, fit_idx].conj()) * K
        # tmp += mdot(mo_coeff[s, fit_idx].T.conj(), 2.0*drho[s], mo_coeff[s, fit_idx]) * K / 2.0
        dw_dv[s] = mdot(mo_coeff[s].conj(), tmp, mo_coeff[s].T) # ip, pq, qj -> ij
    
    # contribution from mu change
    if not fix_mu:
        dw_dv_mu_part = np.zeros((spin, norb, norb), dtype=dw_dv.dtype)
        for s in range(spin):
            f = rho_elec[s] * rho_hole[s]
            f_sum = np.sum(f)
            if abs(f_sum) > ZERO_TOL: # not almost zero T
                # partial rho_ij / partial mu
                drho_dmu = np.dot(mo_coeff[s] * f, mo_coeff[s].conj().T) # should * beta
                #dw_dmu = np.sum(drho[s] * drho_dmu[fit_mesh]) * 2.0 * beta
                dw_dmu = np.einsum('ij, ij ->', drho[s], drho_dmu[fit_idx][:, fit_idx],
                                   optimize=True) * 2.0 * beta
                # dw_dv = dw_dmu * dmu_dv (dmu_dv = drho_dmu / f_sum)
                dw_dv_mu_part[s] = drho_dmu * (dw_dmu / f_sum) # ji
        dw_dv += dw_dv_mu_part
    
    if compact:
        # symmetrize
        dw_dv = lib.pack_tril(dw_dv)
        dw_dv *= 2.0
        diag_idx = tril_diag_indices(norb)
        dw_dv[:, diag_idx] *= 0.5
        
    return dw_dv 

if __name__ == '__main__':
    np.set_printoptions(5, linewidth =1000)
    np.random.seed(1)

    norb = 10
    nelec = 4
    beta = 10.0

    #deg_orbs = []
    #deg_energy = []
    deg_orbs = [[0,3], [1,2], [4,5,6], [7]]
    deg_energy = [1.0 , 0.1, 0.8, 3.0]
    #h = get_h_random(norb)
    h = get_h_random_deg(norb, deg_orbs=deg_orbs, deg_energy=deg_energy)
    print ("h: \n%s" %h)

    mo_energy, mo_coeff, mo_occ, mu = kernel(h, nelec, beta)
    print ("mo_energy: \n%s" % mo_energy)
    print ("mo_occ: \n%s" %mo_occ)
    print ("mu: %s" % mu)
    
    drho = make_rdm1(mo_coeff, mo_occ)
    f0 = (drho*drho).sum()
    mo_energy = mo_energy[None]
    mo_coeff = mo_coeff[None]
    drho = drho[None]

    fix_mu = False
    dw_dv = get_dw_dv(mo_energy, mo_coeff, drho, mu, beta, fix_mu=fix_mu, compact=True)
    
    h_arr_ref = tril_mat2arr(h)
    grad = np.zeros_like(h_arr_ref)
    dx = 1e-6
    for i in range(len(h_arr_ref)):
        h_arr = h_arr_ref.copy()
        h_arr[i] += dx
        h_mat = tril_arr2mat(h_arr)
        mo_energy, mo_coeff, mo_occ, _ = kernel(h_mat, nelec, beta, mu0=mu, fix_mu=fix_mu)
        rho = make_rdm1(mo_coeff, mo_occ)
        f = (rho*rho).sum()
        grad[i] = (f - f0) / dx
    
    print ("diff of gradients: %s" %(la.norm(grad - dw_dv)))
