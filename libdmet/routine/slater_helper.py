#! /usr/bin/env python
"""
Helper functions for embedding Slater determinant.

Authors:
    Zhi-Hao Cui
    Bo-Xiao Zheng

"""
import numpy as np
import scipy.linalg as la
import itertools as it

from pyscf import ao2mo

from libdmet.routine import mfd
from libdmet.utils.misc import (mdot, max_abs, find, add_spin_dim, \
        Iterable, format_idx)
from libdmet.utils import logger as log
from libdmet.settings import IMAG_DISCARD_TOL 

def transform_trans_inv(basis, lattice, H, symmetric=True):
    ncells = lattice.ncells
    nbasis = basis.shape[-1]
    res = np.zeros((nbasis, nbasis))
    if symmetric:
        for i in range(ncells):
            res += mdot(basis[i].T, H[0], basis[i])
        for i, j in it.combinations(range(ncells), 2):
            temp = mdot(basis[i].T, H[lattice.subtract(i, j)], basis[j])
            res += temp + temp.T
    else:
        for i, j in it.product(range(ncells), repeat = 2):
            res += mdot(basis[i].T, H[lattice.subtract(i, j)], basis[j])
    return res

def transform_trans_inv_k(basis_k, H_k):
    """
    Transform from LO to EO, 
    using k-basis and k-one-particle quantities.
    basis_k should has shape (nkpts, nlo, nbasis)
    """
    nkpts, nlo, nbasis = basis_k.shape 
    res = np.zeros((nbasis, nbasis), dtype=np.complex128)
    for k in range(nkpts):
        res += mdot(basis_k[k].conj().T, H_k[k], basis_k[k])
    if max_abs(res.imag) < IMAG_DISCARD_TOL:
        res = res.real
    else:
        log.warn("transform_trans_inv_k: has imag part %s", \
                max_abs(res.imag))
    res /= float(nkpts)
    return res

def transform_trans_inv_sparse(basis, lattice, H, symmetric=True, thr=1e-7):
    ncells = lattice.ncells
    nbasis = basis.shape[-1]
    res = np.zeros((nbasis, nbasis))
    mask_basis = find(True, map(lambda a: la.norm(a) > thr, basis))
    mask_H = find(True, map(lambda a: la.norm(a) > thr, H))
    if symmetric:
        for i in mask_basis:
            res += mdot(basis[i].T, H[0], basis[i])
        for i, j in it.combinations(mask_basis, 2):
            Hidx = lattice.subtract(i, j)
            if Hidx in mask_H:
                temp = mdot(basis[i].T, H[Hidx], basis[j])
                res += temp + temp.T
    else:
        for i, j in it.product(mask_basis, repeat = 2):
            Hidx = lattice.subtract(i, j)
            if Hidx in mask_H:
                res += mdot(basis[i].T, H[Hidx], basis[j])
    return res

def transform_local(basis, lattice, H):
    # assume H is (nscsites, nscsites)
    ncells = lattice.ncells
    nbasis = basis.shape[-1]
    res = np.zeros((nbasis, nbasis))
    for i in range(ncells):
        res += mdot(basis[i].T, H, basis[i])
    return res

def transform_local_sparse(basis, lattice, H, thr = 1e-7):
    # assume H is (nscsites, nscsites)
    ncells = lattice.ncells
    nbasis = basis.shape[-1]
    res = np.zeros((nbasis, nbasis))
    mask_basis = find(True, map(lambda a: la.norm(a) > thr, basis))
    for i in mask_basis:
        res += mdot(basis[i].T, H, basis[i])
    return res

def transform_local_sparseH(basis, lattice, H, thr = 1e-7):
    ncells = lattice.ncells
    nbasis = basis.shape[-1]
    res = np.zeros((nbasis, nbasis))
    mask_H = np.nonzero(abs(H) > thr)
    mask_H = zip(*map(lambda a: a.tolist(), mask_H))
    for j,k in mask_H:
        #for i in range(ncells):
        res += np.dot(basis[:,j].T, basis[:,k]) *  H[j,k]
    return res

def transform_imp(basis, lattice, H):
    return mdot(basis[0].T, H, basis[0])

def transform_imp_env(basis, lattice, H):
    ncells = lattice.ncells
    nbasis = basis.shape[-1]
    res = np.zeros((nbasis, nbasis))
    for i in range(ncells):
        # this is the proper way to do it. equivalently, we do symmetrization 0.5 * (res+res.T)
        #res += 0.5 * mdot(basis[0].T, H[lattice.subtract(0, i)], basis[i])
        #res += 0.5 * mdot(basis[i].T, H[lattice.subtract(i, 0)], basis[0])
        res += mdot(basis[i].T, H[i], basis[0])
    res = 0.5 * (res + res.T)
    return res

def transform_scalar(basis, lattice, s):
    # for example, chemical potential
    ncells = lattice.ncells
    nbasis = basis.shape[-1]
    res = np.zeros((nbasis, nbasis))
    for i in range(ncells):
        res += np.dot(basis[i].T, basis[i]) # does this actually give I?
    res *= s

def transform_4idx(vijkl, ip, jq, kr, ls):
    """
    ip, ijkl -> pjkl -> jpkl
    jq, jpkl -> qpkl -> pqkl
    pqkl, ls -> pqks -> pqsk
    pqsk, kr -> pqsr -> pqrs
    """
    return np.swapaxes(np.tensordot(\
        np.swapaxes(np.tensordot(\
        np.swapaxes(np.tensordot(jq, \
        np.swapaxes(np.tensordot(ip, vijkl, axes = (0,0)),0,1), \
        axes = (0,0)),0,1), \
        ls, axes = (3,0)),2,3), \
        kr, axes = (3,0)),2,3)

def transform_eri_local(basis, lattice, H2):
    """
    Transform the local H2 of shape ((spin, ) nscsites, nscsites, nscsites, nscsites),
    to embedding space. 
    Used for interacting bath formalism.
    """
    if basis.ndim == 3:
        basis = basis[None]
    spin, ncells, nscsites, nbasis = basis.shape
    res = np.zeros((spin*(spin+1)//2, nbasis, nbasis, nbasis, nbasis))
    if H2.ndim == 4:
        if spin == 1:
            H2 = H2[np.newaxis]
        else:
            H2 = np.asarray((H2, H2, H2))
    if spin == 1:
        for i in range(ncells):
            res[0] += transform_4idx(H2[0], basis[0,i], basis[0,i], basis[0,i], basis[0,i])
    else:
        for i in range(ncells):
            res[0] += transform_4idx(H2[0], basis[0,i], basis[0,i], basis[0,i], basis[0,i])
            res[1] += transform_4idx(H2[1], basis[1,i], basis[1,i], basis[1,i], basis[1,i])
            res[2] += transform_4idx(H2[2], basis[0,i], basis[0,i], basis[1,i], basis[1,i])
    return res

def get_emb_basis_other_cell(lattice, basis, R, reorder_idx=None):
    """
    Get embedding basis for the R cell's problem.
    
    Args:
        lattice: lattice object.
        basis: C_lo_eo, (spin, ncells, nlo, neo)
        R: the id of embedding problem, 
           basis is the 0th embedding problem at the first cell.
    
    Returns:
        basis_R: the embedding basis for Rth problem.
    """
    basis = np.asarray(basis)
    old_shape = basis.shape
    if len(old_shape) == 3:
        basis = basis[None]
    if reorder_idx is None: 
        spin, ncells, nlo, neo = basis.shape
        reorder_idx = [lattice.subtract(I, R) for I in range(ncells)]
    basis_R = basis[:, reorder_idx]
    if len(old_shape) == 3:
        basis_R = basis_R[0]
    return basis_R

def get_rho_glob_R(basis, lattice, rho_emb, symmetric=True, compact=True, \
                   sign=None):
    """
    Get rho_glob in site basis, in stripe shape.
    Use democratic partitioning.
    Average of the IJ blocks from I and from J impurity problem.
    
    Args:
        basis: C_lo_eo, (spin, ncells, nlo, neo), or list of C_lo_eo.
        lattice: lattice object, or list of lattices
        rho_emb: rdm1, (spin, neo, neo), or list of rdm1.

    Returns:
        rho_glob_R: global rdm1, 
                    if compact (spin, ncells, nlo, nlo)
                    else       (spin, ncells*nlo, ncells*nlo).
    """
    if not isinstance(lattice, Iterable):
        basis_col = [basis]
        lattice_col = [lattice]
        rho_emb_col = [rho_emb]
    else:
        basis_col = basis
        lattice_col = lattice
        rho_emb_col = rho_emb
    
    if sign is None:
        sign = np.ones(len(lattice_col), dtype=int)
    else:
        sign = np.asarray(sign)
        compact = False
    assert len(sign) == len(lattice_col)

    rho_glob = 0.0
    I_idx = 0
    
    for basis_I, lattice_I, rho_emb_I, sign_I in zip(basis_col, lattice_col, \
                                                     rho_emb_col, sign):
        log.debug(0, "Build rdm1_glob, impurity %s, indices: %s, sign: %s", \
                  I_idx, format_idx(lattice_I.imp_idx), sign_I)
        basis_I = np.asarray(basis_I)
        if basis_I.ndim == 3:
            basis_I = basis_I[None]
        spin, ncells, nlo, _ = basis_I.shape
        rho_emb_I = add_spin_dim(rho_emb_I, spin, non_spin_dim=2)
        
        if compact:
            rho_R = np.zeros((spin, ncells * nlo, nlo))
            for R in range(ncells):
                basis_other = get_emb_basis_other_cell(lattice_I, basis_I, R)
                imp_idx = np.asarray(lattice_I.imp_idx) + R * nlo
                env_idx = np.where(~np.isin(np.arange(ncells * nlo), imp_idx))[0]
                tmp = np.isin(np.arange(nlo), imp_idx)
                imp_idx_0 = np.where(tmp)[0]
                env_idx_0 = np.where(~tmp)[0]
                imp_env = np.ix_(imp_idx, env_idx_0)
                env_imp = np.ix_(env_idx, imp_idx_0)
                env_env = np.ix_(env_idx, env_idx_0)
                for s in range(spin):
                    log.debug(3, "rdm1_glob: spin %s R %s", s, R)
                    C_R = basis_other[s].reshape(-1, basis_other.shape[-1])
                    rdm1_R = mdot(C_R, rho_emb_I[s], C_R[:nlo].conj().T)
                    rdm1_R[imp_env] *= 0.5
                    rdm1_R[env_imp] *= 0.5
                    rdm1_R[env_env]  = 0.0
                    rho_R[s] += rdm1_R
            rho_R = rho_R.reshape(spin, ncells, nlo, nlo)
        else:
            rho_R = np.zeros((spin, ncells * nlo, ncells * nlo))
            for R in range(ncells):
                basis_other = get_emb_basis_other_cell(lattice_I, basis_I, R)
                imp_idx = (np.asarray(lattice_I.imp_idx) + R * nlo) % (ncells * nlo)
                env_idx = np.where(~np.isin(np.arange(ncells * nlo), imp_idx))[0]
                imp_env = np.ix_(imp_idx, env_idx)
                env_imp = np.ix_(env_idx, imp_idx)
                env_env = np.ix_(env_idx, env_idx)
                for s in range(spin):
                    log.debug(3, "rdm1_glob: spin %s R %s", s, R)
                    C_R = basis_other[s].reshape(-1, basis_other.shape[-1])
                    rdm1_R = mdot(C_R, rho_emb_I[s], C_R.conj().T)
                    rdm1_R[imp_env] *= 0.5
                    rdm1_R[env_imp] *= 0.5
                    rdm1_R[env_env]  = 0.0
                    rho_R[s] += rdm1_R

        rho_glob += (rho_R * sign_I)
        I_idx += 1
    return rho_glob

def get_rho_glob_k(basis, lattice, rho_emb, symmetric=True, compact=True, \
                   sign=None):
    if sign is not None:
        compact = False
    
    rho_R = get_rho_glob_R(basis, lattice, rho_emb, symmetric=symmetric, \
            compact=compact, sign=sign)

    if isinstance(lattice, Iterable):
        if compact:
            rho_k = lattice[0].R2k(rho_R)
        else:
            rho_k = lattice[0].R2k(lattice[0].extract_stripe(rho_R))
    else:
        if compact:
            rho_k = lattice.R2k(rho_R)
        else:
            rho_k = lattice.R2k(lattice.extract_stripe(rho_R))
    return rho_k

def get_rho_glob_full(basis, lattice, rho_emb, symmetric=True, compact=True, \
                      sign=None):
    """
    Get rho_glob in site basis, in full shape.
    Use democratic partitioning.
    """
    if sign is not None:
        compact = False
    rho_glob_R = get_rho_glob_R(basis, lattice, rho_emb, symmetric=symmetric, \
                                compact=compact, sign=sign)
    if compact:
        if isinstance(lattice, Iterable):
            rho_glob_full = lattice[0].expand(rho_glob_R)
        else:
            rho_glob_full = lattice.expand(rho_glob_R)
    else:
        rho_glob_full = rho_glob_R
    return rho_glob_full

def get_rdm2_glob_R(basis_a, basis_b, lattice, rdm2_emb, symmetric=True):
    """
    Get rdm2_glob in site basis, in stripe shape,
    (ncells, ncells, ncells, nscsites, nscsites, nscsites, nscsites)
    Use democratic partitioning.
    """
    from pyscf import lib
    ncells = lattice.ncells
    nscsites = lattice.nscsites
    rdm2_R = np.zeros((ncells, ncells, ncells, nscsites, nscsites, nscsites, nscsites))
    I = 0
    if basis_b is None:
        basis_b = basis_a.copy()
    basis_a = basis_a * (0.25**0.25)
    basis_b = basis_b * (0.25**0.25)

    for J in range(ncells):
        for K in range(ncells):
            for L in range(ncells):
                I_I = lattice.subtract(I, I)
                I_J = lattice.subtract(J, I)
                I_K = lattice.subtract(K, I)
                I_L = lattice.subtract(L, I)
                
                J_I = lattice.subtract(I, J)
                J_J = lattice.subtract(J, J)
                J_K = lattice.subtract(K, J)
                J_L = lattice.subtract(L, J)
                
                K_I = lattice.subtract(I, K)
                K_J = lattice.subtract(J, K)
                K_K = lattice.subtract(K, K)
                K_L = lattice.subtract(L, K)
                
                L_I = lattice.subtract(I, L)
                L_J = lattice.subtract(J, L)
                L_K = lattice.subtract(K, L)
                L_L = lattice.subtract(L, L)

                rdm2_R[I_J, I_K, I_L] += np.einsum('pqrs,ip,jq,kr,ls->ijkl', rdm2_emb,
                          basis_a[I_I], basis_a[I_J].conj(),
                          basis_b[I_K], basis_b[I_L].conj(), optimize=True)
                rdm2_R[I_J, I_K, I_L] += np.einsum('pqrs,ip,jq,kr,ls->ijkl', rdm2_emb,
                          basis_a[J_I], basis_a[J_J].conj(),
                          basis_b[J_K], basis_b[J_L].conj(), optimize=True)
                rdm2_R[I_J, I_K, I_L] += np.einsum('pqrs,ip,jq,kr,ls->ijkl', rdm2_emb,
                          basis_a[K_I], basis_a[K_J].conj(),
                          basis_b[K_K], basis_b[K_L].conj(), optimize=True)
                rdm2_R[I_J, I_K, I_L] += np.einsum('pqrs,ip,jq,kr,ls->ijkl', rdm2_emb,
                          basis_a[L_I], basis_b[L_J].conj(),
                          basis_b[L_K], basis_b[L_L].conj(), optimize=True)
    return rdm2_R

def get_phi_k(basis, lattice, rho_glob_k, phi0_k):
    nkpts = lattice.ncells
    phi_k = [rho_glob_k[k].dot(phi0_k[k]) for k in range(nkpts)]
    return phi_k

def get_rho_pdmet_R(basis, lattice, phi_k):
    ncells = nkpts = lattice.ncells
    nscsites = lattice.nscsites
    rho_pdmet_k = np.zeros((nkpts, nscsites, nscsites), dtype=np.complex128)
    for k in range(nkpts):
        phi = phi_k[k]
        norm = la.inv(phi.conj().T.dot(phi))
        rho_pdmet_k[k] = mdot(phi, norm, phi.conj().T)
    rho_pdmet_R = lattice.FFTtoT(rho_pdmet_k)
    return rho_pdmet_R

def get_rdm1_idem(rdm1, nelec, beta):
    """
    Project a rdm1 to the idempotent rdm1, using natural orbitals.
    
    Args:
        rdm1: rdm1, shape (spin, (nkpts), nlo, nlo). 
              For restricted rdm1, the largest occupancy is 1.
        nelec: the electron number (all kpoints).
        beta: possible smearing.

    Returns:
        rdm1_idem: shape (spin, (nkpts), nlo, nlo), projected idempotented rdm1.
    """
    rdm1 = np.asarray(rdm1)
    if len(rdm1.shape) == 3: # no nkpts
        spin, _, nlo = rdm1.shape
        ew = np.empty((spin, nlo))
        ev = np.empty((spin, nlo, nlo), dtype=rdm1.dtype)
        rdm1_idem = np.empty_like(rdm1)
        for s in range(spin):
            ew[s], ev[s] = la.eigh(rdm1[s])
        ew = -ew[:, ::-1]
        ev = ev[:, :, ::-1]
        ewocc, mu, nerr = mfd.assignocc(ew, nelec, beta, mu0=-0.5)
        for s in range(spin):
            rdm1_idem[s] = np.dot(ev[s] * ewocc[s], ev[s].conj().T)
    else:
        spin, nkpts, _, nlo = rdm1.shape
        ew = np.empty((spin, nkpts, nlo))
        ev = np.empty((spin, nkpts, nlo, nlo), dtype=rdm1.dtype)
        rdm1_idem = np.empty_like(rdm1)
        for s in range(spin):
            for k in range(nkpts):
                ew[s, k], ev[s, k] = la.eigh(rdm1[s, k])
        ew = -ew[:, :, ::-1]
        ev = ev[:, :, :, ::-1]
        ewocc, mu, nerr = mfd.assignocc(ew, nelec, beta, mu0=-0.5)
        for s in range(spin):
            for k in range(nkpts):
                rdm1_idem[s, k] = np.dot(ev[s, k] * ewocc[s, k], \
                        ev[s, k].conj().T)
    return rdm1_idem

def get_H1_power_R(lattice, H1_k=None, power=2, return_all_power=False):
    """
    Get power series of H (or D).
    """
    if H1_k is None:
        H1_k = lattice.getH1(kspace=True)
    nkpts = H1_k.shape[0]
    H1_pow_k = H1_k.copy()
    H1_pow_R_collect = [lattice.FFTtoT(H1_pow_k)]
    for p in range(power - 1):
        for k in range(nkpts):
            H1_pow_k[k] = H1_pow_k[k].dot(H1_k[k])
        if return_all_power:
            H1_pow_R = lattice.FFTtoT(H1_pow_k)
            H1_pow_R_collect.append(H1_pow_R)
    if return_all_power:
        return H1_pow_R_collect
    else:
        H1_pow_R = lattice.FFTtoT(H1_pow_k)
        return H1_pow_R 

def init_H2(norb, eri_symmetry, dtype=np.double, spin_dim=None):
    """
    Initialize H2 with zeros.

    Args:
        norb: number of orbitals.
        eri_symmetry: 1 or 4 or 8.
        dtype: data type.

    Returns:
        H2: ERI.
    """
    if spin_dim is None:
        spin_dim = ()
    else:
        spin_dim = (spin_dim,)
    if eri_symmetry == 1:
        H2 = np.zeros(spin_dim + (norb, norb, norb, norb), dtype=dtype)
    elif eri_symmetry == 4:
        norb_pair = norb * (norb + 1) // 2
        H2 = np.zeros(spin_dim + (norb_pair, norb_pair), dtype=dtype)
    elif eri_symmetry == 8:
        norb_pair = norb * (norb + 1) // 2
        norb_pair_pair = norb_pair * (norb_pair + 1) // 2
        H2 = np.zeros(spin_dim + (norb_pair_pair,), dtype=dtype)
    else:
        raise ValueError("unknown ERI symmetry: %s" %(eri_symmetry))
    return H2

def restore_eri_local(H2, norb):
    """
    Retore ERI symmetry of H2 to 4-fold.
    
    Args:
        H2: shape (spin, 1 or 4 fold symmetry).
        norb: number of orbitals.
    
    Returns:
        H2_s4: H2 with 4-fold symmetry.
    """
    spin = H2.shape[0]
    norb_pair = norb * (norb + 1) // 2
    if H2.size == spin * norb_pair * norb_pair:
        H2_s4 = H2.reshape(spin, norb_pair, norb_pair)
    else:
        H2_s4 = np.zeros((spin, norb_pair, norb_pair), dtype=H2.dtype)
        for s in range(spin):
            H2_s4[s] = ao2mo.restore(4, H2[s], norb)
    return H2_s4

def unit2emb(H2_unit, neo):
    """
    Allocate H2_emb and fill the impurity block with H2_unit.

    Args:
        H2_unit: unit ERI, shape (spin_pair, 1 or 4 or 8 fold symmetry).
        neo: number of embedding orbitals.

    Returns:
        H2_emb: embedding ERI, with impurity block filled by H2_unit.
                Has the same symmetry as H2_unit.
    """
    if isinstance(H2_unit, np.ndarray):
        spin_pair = H2_unit.shape[0]
        if H2_unit.ndim == 5:   # 1-fold
            H2_emb = init_H2(neo, 1, spin_dim=spin_pair)
        elif H2_unit.ndim == 3: # 4-fold
            H2_emb = init_H2(neo, 4, spin_dim=spin_pair)
        elif H2_unit.ndim == 2: # 8-fold
            H2_emb = init_H2(neo, 8, spin_dim=spin_pair)
        else:
            raise ValueError
        fill_idx = tuple(map(slice, H2_unit.shape))
        H2_emb[fill_idx] = H2_unit
    else:
        # outcore only support 4-fold symmetry
        H2_unit_old = np.asarray(H2_unit["ccdd"])
        spin_pair, neo_pair, neo_pair = H2_unit_old.shape
        res_shape = (spin_pair, neo*(neo+1)//2, neo*(neo+1)//2)
        H2_emb = H2_unit
        del H2_emb["ccdd"]
        H2_emb.create_dataset("ccdd", res_shape, 'f8')
        H2_emb["ccdd"][:] = 0.0
        H2_emb["ccdd"][:spin_pair, :neo_pair, :neo_pair] = H2_unit_old
    return H2_emb

