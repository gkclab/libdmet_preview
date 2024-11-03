#! /usr/bin/env python

"""
Embedding a superconducting states with spinless formalsim,
using partial particle-hole transform.

Author:
    Zhihao Cui <zhcui0408@gmail.com>
"""

import copy
import h5py
import numpy as np
import scipy.linalg as la

from libdmet.utils import logger as log
from libdmet.system import integral
from libdmet.routine import localizer
from libdmet.routine import ftsystem
from libdmet.routine import mfd
from libdmet.routine.mfd import assignocc
from libdmet.routine import slater
from libdmet.routine.slater import (test_grad, get_H1_scaled, get_H2_scaled,
                                    add_bath)
from libdmet.routine import slater_helper
from libdmet.routine.fit import minimize
from libdmet.routine import pbc_helper as pbc_hp
from libdmet.solver.scf import _get_veff_ghf, restore_Ham
from libdmet.lo.lowdin import vec_lowdin
from libdmet.lo import scdm
from libdmet.utils.misc import mdot, max_abs, format_idx
from libdmet.routine.spinless_helper import *

def get_emb_basis(lattice, GRho, local=True, kind='svd', **kwargs):
    """
    Get embedding basis, i.e. C_lo_eo in R,
    shape (ncell, nso, nso + nval*2).
    """
    if not local:
        raise NotImplementedError

    if kind == 'svd':
        basis = _get_emb_basis_svd(lattice, GRho.real, **kwargs)
    elif kind == 'eig':
        basis = _get_emb_basis_eig(lattice, GRho.real, **kwargs)
    elif kind == 'ph':
        basis = _get_emb_basis_ph(lattice, GRho.real, **kwargs)
    else:
        raise ValueError("get_emb_basis: Unknown kind %s" % kind)

    if kwargs.get("bath_opt", False):
        basis = get_emb_basis_opt(lattice, GRho.real, basis, keep_imp_identity=False,
                                  tol=kwargs.get("tol_bath", 1e-6))
    return basis

embBasis = get_emb_basis

def _get_emb_basis_svd(lattice, rdm1, **kwargs):
    """
    Construct spinless bath based the generalized density matrix,
    using SVD.

    Args:
        lattice: lattice object.
        rdm1: generalized density matrix, shape (ncells, nso, nso)

    Kwargs:
        valence_bath: whether to use valence bath or full bath.
        orth: True, orthogonalize bath by projecting out the virtual orbitals.
              if orth == False and valence_bath, the basis can be non-orthonal.
        tol_bath: 1e-9, tolerance for discarding small singular values.
        nbath: number of bath (all spin channel).

    Returns:
        basis: C_lo_eo, shape (ncells, nso, neo)
    """
    valence_bath = kwargs.get("valence_bath", True)
    orth = kwargs.get("orth", True)
    tol_bath = kwargs.get("tol_bath", 1e-9)
    nbath = kwargs.get("nbath", None)

    ncells = lattice.ncells
    nlo = lattice.nscsites
    nso = nlo * 2

    val_idx = list(lattice.val_idx) + [i + nlo for i in lattice.val_idx]
    imp_idx = list(lattice.imp_idx) + [i + nlo for i in lattice.imp_idx]
    imp_idx_bath = val_idx if valence_bath else imp_idx
    env_idx = []
    # boolean mask of virtual in the env_idx
    virt_mask = []
    # boolean mask of alpha orbitals in the env_idx
    alpha_mask = []
    for R in range(ncells):
        for s in range(2):
            for i in range(nlo):
                idx = R * nso + s * nlo + i
                if not idx in imp_idx_bath:
                    env_idx.append(idx)
                    virt_mask.append(idx in imp_idx)
                    alpha_mask.append(s == 0)
    nimp  = len(imp_idx)
    log.debug(0, "imp_idx for bath : %-15s [val  : %s] ",
              format_idx(imp_idx_bath), format_idx(val_idx))
    log.debug(0, "env_idx for bath : %-15s [virt : %s] ",
              format_idx(env_idx), format_idx(np.array(env_idx)[virt_mask]))

    rdm1 = np.asarray(rdm1)
    assert rdm1.shape == (ncells, nso, nso)
    rdm1_env_imp = rdm1.reshape(ncells * nso, nso)[env_idx][:, imp_idx_bath]

    basis = np.zeros((ncells * nso, nso * 2))

    # SVD
    u, sigma, vt = la.svd(rdm1_env_imp, full_matrices=False)
    if nbath is None:
        nbath = (sigma >= tol_bath).sum()
    log.eassert(nbath % 2 == 0, "nbath (%s) should be even in GSO.", nbath)
    B = u[:, :nbath]

    # check zero singular values
    with np.printoptions(suppress=False):
        log.debug(1, "Singular values:\n%s", sigma[:nbath])
        log.debug(1, "Singular values discarded:\n%s", sigma[nbath:])
    nzero = np.sum(np.abs(sigma[:nbath]) < tol_bath)
    log.debug(0, "Zero singular values number: %s", nzero)
    if nzero > 0:
        log.warn("Zero singular value exists, \n"
                 "this may cause numerical instability.")

    # project out the local virtual component
    # this is actually only used when valence_bath is true.
    if orth:
        B[virt_mask] = 0.0
        B = vec_lowdin(B, np.eye(B.shape[0]))
    else:
        # ZHC TODO support orth == False
        raise NotImplementedError

    # localization of bath
    loc_method = kwargs.get("localize_bath", None)
    if loc_method is not None:
        if not lattice.is_model:
            log.warn("Only model is currently supported "
                     "for localization of bath.")
        B = localizer.localize_bath(B, method=loc_method)

    # analysis on the particle and hole character
    w = np.einsum("ai, ai -> i", B[alpha_mask], B[alpha_mask], optimize=True)
    order = np.argsort(w, kind='mergesort')[::-1]
    w1 = w[order]
    orderA, orderB = order[:nbath//2], order[nbath//2:]
    wA, wB = w1[:nbath//2], 1.0 - w1[nbath//2:]
    log.debug(0, "particle character:\nspin A max %.2f min %.2f mean %.2f"
              "\nspin B max %.2f min %.2f mean %.2f", np.max(wA), np.min(wA),
              np.average(wA), np.max(wB), np.min(wB), np.average(wB))

    basis[imp_idx, :nimp] = np.eye(nimp)
    basis[env_idx, nimp:nimp + nbath] = B[:, order]
    basis = basis[:, :nimp + nbath].reshape(ncells, nso, nimp + nbath)
    log.debug(0, "nimp : %d", nimp)
    log.debug(0, "nbath: %d", nbath)
    return basis

__embBasis_proj = _get_emb_basis_svd

def _get_emb_basis_eig(lattice, rdm1, **kwargs):
    """
    Construct spinless bath based the generalized density matrix,
    using eigenvalue decomposation.

    Args:
        lattice: lattice object.
        rdm1: generalized density matrix, shape (ncells, nso, nso)

    Kwargs:
        valence_bath: whether to use valence bath or full bath.
        orth: True, orthogonalize bath by projecting out the virtual orbitals.
              if orth == False and valence_bath, the basis can be non-orthonal.
        tol_bath: 1e-9, tolerance for discarding small singular values.
        nbath: number of bath (all spin channel).

    Returns:
        basis: C_lo_eo, shape (ncells, nso, neo)
    """
    valence_bath = kwargs.get("valence_bath", True)
    orth = kwargs.get("orth", True)
    tol_bath = kwargs.get("tol_bath", 1e-9)
    nbath = kwargs.get("nbath", None)

    ncells = lattice.ncells
    nlo = lattice.nscsites
    nso = nlo * 2

    val_idx = list(lattice.val_idx) + [i + nlo for i in lattice.val_idx]
    imp_idx = list(lattice.imp_idx) + [i + nlo for i in lattice.imp_idx]
    imp_idx_bath = val_idx if valence_bath else imp_idx
    env_idx = []
    # boolean mask of virtual in the env_idx
    virt_mask = []
    # boolean mask of alpha orbitals in the env_idx
    alpha_mask = []
    for R in range(ncells):
        for s in range(2):
            for i in range(nlo):
                idx = R * nso + s * nlo + i
                if not idx in imp_idx_bath:
                    env_idx.append(idx)
                    virt_mask.append(idx in imp_idx)
                    alpha_mask.append(s == 0)
    nimp  = len(imp_idx)
    log.debug(0, "imp_idx for bath : %-15s [val  : %s] ",
              format_idx(imp_idx_bath), format_idx(val_idx))
    log.debug(0, "env_idx for bath : %-15s [virt : %s] ",
              format_idx(env_idx), format_idx(np.array(env_idx)[virt_mask]))

    rdm1 = np.asarray(rdm1)
    assert rdm1.shape == (ncells, nso, nso)

    # env-env block
    rdm1_env_env = lattice.expand(rdm1)[env_idx][:, env_idx]
    ew, ev = la.eigh(rdm1_env_env)

    bath = []
    e_col = []
    for i, e in enumerate(ew):
        if abs(e) > tol_bath and abs(1 - e) > tol_bath:
            bath.append(ev[:, i])
            e_col.append(e)
    e_col = np.asarray(e_col)
    with np.printoptions(suppress=False):
        log.debug(0, "dm eigenvalues:\n%s", e_col)

    bath = np.asarray(bath).T
    nbath = bath.shape[-1]
    log.eassert(nbath % 2 == 0, "nbath (%s) should be even in GSO.", nbath)

    basis = np.zeros((ncells * nso, nimp + nbath))

    B = bath
    # project out the local virtual component
    # this is actually only used when valence_bath is true.
    if orth:
        B[virt_mask] = 0.0
        B = vec_lowdin(B, np.eye(B.shape[0]))
    else:
        raise NotImplementedError

    # localization of bath
    loc_method = kwargs.get("localize_bath", None)
    if loc_method is not None:
        if not lattice.is_model:
            log.warn("Only model is currently supported "
                     "for localization of bath.")
        B = localizer.localize_bath(B, method=loc_method)

    # analysis on the particle and hole character
    w = np.einsum("ai, ai -> i", B[alpha_mask], B[alpha_mask], optimize=True)
    order = np.argsort(w, kind='mergesort')[::-1]
    w1 = w[order]
    orderA, orderB = order[:nbath//2], order[nbath//2:]
    wA, wB = w1[:nbath//2], 1.0 - w1[nbath//2:]
    log.debug(0, "particle character:\nspin A max %.2f min %.2f mean %.2f"
              "\nspin B max %.2f min %.2f mean %.2f", np.max(wA), np.min(wA),
              np.average(wA), np.max(wB), np.min(wB), np.average(wB))

    basis[imp_idx, :nimp] = np.eye(nimp)
    basis[env_idx, nimp:nimp + nbath] = B[:, order]
    basis = basis.reshape(ncells, nso, nimp + nbath)
    log.debug(0, "nimp : %d", nimp)
    log.debug(0, "nbath: %d", nbath)
    return basis

def get_emb_basis_opt(latt, rdm1_R, basis, keep_imp_identity=False, tol=1e-6):
    """
    Optimize the basis to ensure the number of electron is an integer.
    Useful for metal case: embedding space may not have integer number of electron.
    """
    from scipy import optimize as opt
    rdm1_R = rdm1_R.real
    nemb = basis.shape[-1]
    rdm1_k = latt.R2k(rdm1_R)
    basis_k = latt.R2k_basis(basis)
    basis_full = basis.reshape(-1, nemb)
    rdm1_full = latt.expand(rdm1_R)

    #rdm1_emb = mdot(basis_full.conj().T, rdm1_full, basis_full)
    rdm1_emb = foldRho_k(rdm1_k, basis_k)
    nelec = rdm1_emb.trace()
    nelec_target = np.round(nelec)
    log.debug(0, "get_emb_basis_opt: nelec current: %15.8f , "
              "nelec_target: %15.8f", nelec, nelec_target)
    if abs(nelec - nelec_target) < tol:
        return basis
    elif nelec < nelec_target:
        lval = -1.0
        rval =  0.0
    else:
        lval =  1.0
        rval =  0.0

    basis_sq = np.dot(basis_full, basis_full.conj().T)

    def nelec_cost_fn_brentq(mu):
        rdm1_new = basis_sq - mu * rdm1_full
        ew, ev = la.eigh(rdm1_new)
        ev = ev[:, -nemb:].reshape(basis.shape)
        ev = latt.R2k_basis(ev)
        nelec = foldRho_k(rdm1_k, ev).trace().real
        return nelec - nelec_target

    res = opt.brentq(nelec_cost_fn_brentq, lval, rval, xtol=tol, rtol=tol,
                     maxiter=1000, full_output=True, disp=False)

    if (not res[1].converged):
        log.warn("get_emb_basis_opt fitting mu brentq fails.")
    mu = res[0]

    rdm1_new = basis_sq - mu * rdm1_full
    basis_sq = None
    rdm1_full = None
    ew, ev = la.eigh(rdm1_new)
    rdm1_new = None
    ev = ev[:, -nemb:]

    # keep the first nimp site as identity matrix
    if keep_imp_identity:
        basis_R = basis_full[:, :latt.nimp]
        for i in range(ev.shape[-1]):
            v = ev[:, i]
            coeff = v @ basis_R
            v = v - np.dot(basis_R, coeff)
            norm_v = la.norm(v)
            log.debug(0, "norm of orb %5d : %15.5g ,    keep: %s",
                      i, norm_v, norm_v > tol)
            if norm_v > tol:
                if basis_R.shape[-1] < nemb:
                    basis_R = np.hstack((basis_R, (v / norm_v)[:, None]))
                else:
                    log.warn("basis rank is more than nemb!")
        basis = basis_R.reshape(basis.shape)
    else:
        basis = ev.reshape(basis.shape)

    ev_k = latt.R2k_basis(basis)
    nelec = foldRho_k(rdm1_k, ev_k).trace()
    log.debug(0, "get_emb_basis_opt: nelec after fit: %15.8f, mu: %15.8f",
              nelec, mu)
    return basis

def _get_emb_basis_ph(lattice, rdm1, **kwargs):
    """
    Construct spinless bath based the generalized density matrix,
    using particle and hole projection.

    Args:
        lattice: lattice object.
        rdm1: generalized density matrix, shape (ncells, nso, nso)

    Kwargs:
        valence_bath: whether to use valence bath or full bath.
        orth: True, orthogonalize bath by projecting out the virtual orbitals.
              if orth == False and valence_bath, the basis can be non-orthonal.
        tol_bath: 1e-9, tolerance for discarding small singular values.
        nbath: number of bath (all spin channel).

    Returns:
        basis: C_lo_eo, shape (ncells, nso, neo)
    """
    from libdmet.lo.lowdin import _orth_cano
    valence_bath = kwargs.get("valence_bath", True)
    orth = kwargs.get("orth", True)
    tol_bath = kwargs.get("tol_bath", 1e-9)
    nbath = kwargs.get("nbath", None)

    ncells = lattice.ncells
    nlo = lattice.nscsites
    nso = nlo * 2

    val_idx = list(lattice.val_idx) + [i + nlo for i in lattice.val_idx]
    imp_idx = list(lattice.imp_idx) + [i + nlo for i in lattice.imp_idx]
    imp_idx_bath = val_idx if valence_bath else imp_idx
    env_idx = []
    # boolean mask of virtual in the env_idx
    virt_mask = []
    virt_idx = []
    # boolean mask of alpha orbitals in the env_idx
    alpha_mask = []
    for R in range(ncells):
        for s in range(2):
            for i in range(nlo):
                idx = R * nso + s * nlo + i
                if not idx in imp_idx_bath:
                    env_idx.append(idx)
                    virt_mask.append(idx in imp_idx)
                    if idx in imp_idx:
                        virt_idx.append(idx)
                    alpha_mask.append(s == 0)
    nimp  = len(imp_idx)
    log.debug(0, "imp_idx for bath : %-15s [val  : %s] ",
              format_idx(imp_idx_bath), format_idx(val_idx))
    log.debug(0, "env_idx for bath : %-15s [virt : %s] ",
              format_idx(env_idx), format_idx(np.array(env_idx)[virt_mask]))

    # 1. rdm1 p and h
    rdm1_p = np.asarray(rdm1)
    assert rdm1_p.shape == (ncells, nso, nso)
    bath_p = rdm1_p.reshape(ncells * nso, nso)[:, imp_idx_bath]

    rdm1_h = -rdm1
    rdm1_h[0, range(nso), range(nso)] += 1.0
    bath_h = rdm1_h.reshape(ncells * nso, nso)[:, imp_idx_bath]

    # 2. add virtual
    nval = len(imp_idx_bath) * 2
    nvirt = len(virt_idx)
    nbasis = nval + nvirt

    basis = np.zeros((ncells*nso, nbasis))
    basis[virt_idx, range(nbasis-nvirt, nbasis)] = 1.0
    basis[:, :nval//2] = bath_p
    basis[:, nval//2:nval] = bath_h

    # 3. orthogonalization
    basis = _orth_cano(basis, s=None, tol=tol_bath)
    basis = basis.reshape(ncells, nso, nbasis)

    log.debug(0, "nimp + nbath: %d", nbasis)
    return basis

def get_emb_Ham(lattice, basis, vcor, mu, local=True, **kwargs):
    """
    Get embedding Hamiltonian.
    """
    nbasis = basis.shape[-1]

    # First transform two-body,
    # since we need ERI to calculate JK_emb for interacting bath.
    log.info("Two-body part")
    H2_fname = kwargs.get("H2_fname", None)
    H2_given = kwargs.get("H2_given", None)
    if H2_given is None:
        if H2_fname is None:
            H2 = __embHam2e(lattice, basis, vcor, local, **kwargs)
        else:
            log.debug(1, "Load H2 from %s", H2_fname)
            feri = h5py.File(H2_fname, 'r')
            H2 = np.asarray(feri['emb_eri'])
            feri.close()
    else:
        H2 = H2_given

    log.info("One-body part")
    H1, ovlp = __embHam1e(lattice, basis, vcor, mu, H2, **kwargs)

    H0_add = kwargs.get("H0_add", 0.0)
    H0 = lattice.getH0() + H0_add

    # ZHC FIXME what spin value should I return?
    return integral.Integral(nbasis, True, False, H0, {"cd": H1},
                             {"ccdd": H2}, ovlp=ovlp), None

embHam = get_emb_Ham

def __embHam2e(lattice, basis, vcor, local, int_bath=True, last_aabb=True, **kwargs):
    """
    Internal routine to construct H2_emb.
    """
    nao = lattice.nao
    nso = nao * 2
    neo = basis.shape[-1]
    neo_pair = neo * (neo + 1) // 2
    basis_Ra, basis_Rb = separate_basis(basis)
    eri_symmetry = lattice.eri_symmetry
    max_memory = kwargs.get("max_memory", None)

    if lattice.is_model:
        LatH2 = lattice.getH2(compact=False, kspace=False, use_Ham=True)
        if local:
            log.eassert(max_abs(basis[0, :, :nso] - np.eye(nso)) < 1e-10,
                        "the embedding basis is not local")
            if lattice.H2_format == 'spin local':
                LatH2 = restore_eri_local(LatH2, nao) # 4-fold symm
                if int_bath:
                    H2 = transform_eri_local(basis_Ra, basis_Rb, LatH2,
                                             symm=eri_symmetry)[None]
                else:
                    H2 = unit2emb(LatH2, neo)
                    H2 = ao2mo.restore(eri_symmetry, H2, neo)[None]
            elif latiice.H2_format == "spin nearest":
                if int_bath:
                    raise NotImplementedError
                else:
                    H2[:, :nso, :nso, :nso, :nso] = LatH2[:, 0]
            elif lattice.H2_format == "spin full":
                if int_bath:
                    raise NotImplementedError
                else:
                    H2[:, :nso, :nso, :nso, :nso] = LatH2[:, 0, 0, 0]
            else:
                raise ValueError
        else:
            raise NotImplementedError
    else: # ab initio system
        cell = lattice.cell
        mydf = lattice.df
        C_ao_lo = lattice.C_ao_lo
        swap_idx = kwargs.get("swap_idx", None)
        kscaled_center = kwargs.get("kscaled_center", None)
        t_reversal_symm = kwargs.get("t_reversal_symm", True)
        incore = kwargs.get("incore", True)
        fout = kwargs.get("fout", "H2.h5")
        use_mpi = kwargs.get("use_mpi", False)

        if int_bath: # interacting bath
            if getattr(cell, 'pbc_intor', None):
                if use_mpi:
                    from libdmet.basis_transform.eri_transform_mpi import get_emb_eri_gso
                    H2 = get_emb_eri_gso(mydf.cell, mydf._cderi, mydf.kpts,
                                         C_ao_lo=C_ao_lo, basis=basis,
                                         kscaled_center=kscaled_center,
                                         symmetry=eri_symmetry, max_memory=max_memory,
                                         swap_idx=swap_idx,
                                         t_reversal_symm=t_reversal_symm, incore=incore,
                                         fout=fout)
                else:
                    from libdmet.basis_transform.eri_transform import get_emb_eri_gso
                    H2 = get_emb_eri_gso(cell, mydf, C_ao_lo=C_ao_lo, basis=basis,
                                         kscaled_center=kscaled_center,
                                         symmetry=eri_symmetry, max_memory=max_memory,
                                         swap_idx=swap_idx,
                                         t_reversal_symm=t_reversal_symm, incore=incore,
                                         fout=fout)
            else:
                raise NotImplementedError
        else: # non-interacting bath
            if use_mpi:
                from libdmet.basis_transform.eri_transform_mpi import get_emb_eri_gso
                H2 = get_emb_eri_gso(mydf.cell, mydf._cderi, mydf.kpts,
                                     C_ao_lo=C_ao_lo, basis=basis,
                                     kscaled_center=kscaled_center,
                                     symmetry=eri_symmetry, max_memory=max_memory,
                                     swap_idx=swap_idx,
                                     t_reversal_symm=t_reversal_symm, incore=incore,
                                     fout=fout, unit_eri=True)
            else:
                from libdmet.basis_transform.eri_transform import get_emb_eri_gso
                H2 = get_emb_eri_gso(cell, mydf, C_ao_lo=C_ao_lo, basis=basis,
                                     kscaled_center=kscaled_center,
                                     symmetry=eri_symmetry, max_memory=max_memory,
                                     swap_idx=swap_idx,
                                     t_reversal_symm=t_reversal_symm, incore=incore,
                                     fout=fout, unit_eri=True)
            H2 = unit2emb(H2, nbasis)

    if isinstance(H2, np.ndarray):
        log.info("H2 memory allocated size = %d MB", H2.size * 8. / 1024 / 1024)
    return H2

def __embHam1e(lattice, basis, vcor, mu, H2_emb, int_bath=True,
               add_vcor=False, **kwargs):
    """
    Internal routine to construct H1_emb.
    """
    log.eassert(vcor.islocal(),
                "nonlocal correlation potential cannot be treated in this routine")
    nao = lattice.nscsites
    ncells = lattice.ncells
    nso = nao * 2
    nbasis = basis.shape[-1]
    basis_k = lattice.R2k_basis(basis)
    basis_Ra, basis_Rb = separate_basis(basis)
    basis_ka, basis_kb = separate_basis(basis_k)

    hcore_k = lattice.getH1(kspace=True)
    fock_k  = lattice.getFock(kspace=True)
    ovlp_k  = lattice.get_ovlp(kspace=True)
    JK_imp = lattice.get_JK_imp()
    if not isinstance(H2_emb, np.ndarray):
        H2_emb = np.asarray(H2_emb["ccdd"])

    log.debug(1, "transform hcore")
    hcore_custom = kwargs.get("hcore_custom", None)
    if hcore_custom is not None:
        # using customized hcore
        hcore_emb = transform_trans_inv_k(basis_ka, basis_kb, hcore_custom)
    else:
        hcore_emb = transform_trans_inv_k(basis_ka, basis_kb, hcore_k)

    log.debug(1, "transform ovlp")
    ovlp_emb = transform_trans_inv_k(basis_ka, basis_kb, ovlp_k)

    dft = kwargs.get("dft", False)
    vxc_dc = kwargs.get("vxc_dc", False)
    vhf_for_energy = kwargs.get("vhf_for_energy", True)
    if dft:
        hyb = pbc_hp.get_hybrid_param(lattice.kmf)[-1]
        vxc_emb = transform_trans_inv_k(basis_ka, basis_kb, lattice.vxc_lo_k)
    else:
        hyb = 1.0
        vxc_loc = vxc_emb = None

    if int_bath:
        rdm1_emb = foldRho_k(lattice.rdm1_lo_k, basis_k)
        if dft:
            if vxc_dc:
                raise NotImplementedError
                #vxc_loc = get_vxc_loc(lattice, rdm1_emb, lattice.C_ao_lo, basis_k)
            else:
                vxc_loc = vxc_emb
            # J + scaled K + vxc
            veff_emb = transform_trans_inv_k(basis_ka, basis_kb,
                                             lattice.veff_lo_k)

            H1 = hcore_emb + veff_emb
            if vxc_loc is not None:
                log.debug(1, "remove the contribution of vxc")
                H1 -= vxc_loc

            # after subtract vxc_emb, H1 has hcore, veff_loc, veff_core
            # now subtract veff_loc (i.e. local J + scaled K)
            log.debug(1, "Construct JK_emb")
            veff_loc = slater.get_veff(rdm1_emb, H2_emb, hyb=hyb, ghf=True)
            H1 -= veff_loc

            # add ph terms for the embedding problem due to different scaled K
            if kwargs.get("add_ph_vk", True):
                log.debug(1, "add addtional terms due to different scaled k, "
                          "hyb = %.5g", hyb)
                rdm1_lo_k_eye = np.zeros_like(lattice.rdm1_lo_k)
                norb = rdm1_lo_k_eye.shape[-1]
                rdm1_lo_k_eye[:, range(norb//2, norb), range(norb//2, norb)] = 1.0
                rdm1_lo_k_eye = foldRho_k(rdm1_lo_k_eye, basis_k)
                vk_ph_add = slater.get_veff(rdm1_lo_k_eye, H2_emb, hyb=(1.0-hyb),
                                            ghf=True, hyb_j=0.0)
                H1 -= vk_ph_add

            log.debug(1, "Construct JK_core")
            if vhf_for_energy:
                # the JK core used for energy evaluation is HF one
                vhf_emb = transform_trans_inv_k(basis_ka, basis_kb,
                                                lattice.vhf_lo_k)
                vhf_loc = slater.get_veff(rdm1_emb, H2_emb, hyb=1.0, ghf=True)
                vhf_emb -= vhf_loc
                lattice.JK_core = vhf_emb
                # ZHC FIXME
                # we need the normal ordered term for energy.
            else:
                # only J and scaled K are used for energy
                lattice.JK_core = H1 - hcore_emb
        else:
            log.debug(1, "transform fock")
            if not lattice.is_model:
                # ZHC NOTE do not use DFT fock here, restruct the fock from HF
                # potential.
                #fock_k = lattice.hcore_lo_k + lattice.vhf_lo_k
                fock_k = lattice.fock_hf_lo_k
            H1 = transform_trans_inv_k(basis_ka, basis_kb, fock_k)

            # add extra local term to H1, only to impurity
            hcore_add = kwargs.get("hcore_add", None)
            if hcore_add is not None:
                H1 += transform_imp(basis_Ra, basis_Rb, hcore_add)

            log.debug(1, "Construct JK_emb")
            JK_emb = slater.get_veff(rdm1_emb, H2_emb, hyb=1.0, ghf=True)
            # subtract JK_emb
            H1 -= JK_emb

            log.debug(1, "Construct JK_core")
            # save JK_core for energy evaluation
            lattice.JK_core = H1 - hcore_emb
    else: # NIB
        add_vcor = True
        if lattice.use_hcore_as_emb_ham:
            log.debug(1, "Use hcore as embedding Hamiltonian.")
            H1 = hcore_emb

            # add extra local term to H1, only to impurity
            hcore_add = kwargs.get("hcore_add", None)
            if hcore_add is not None:
                H1 += transform_imp(basis_Ra, basis_Rb, hcore_add)

            # NIB and use hcore, do not include JK_core in energy.
            lattice.JK_core = None
        else: # NIB but use fock as embedding hamiltonian.
            log.debug(1, "transform fock")
            H1 = transform_trans_inv_k(basis_ka, basis_kb, fock_k)

            log.debug(1, "Construct JK_emb")
            rdm1_emb = foldRho_k(lattice.rdm1_lo_k, basis_k)
            JK_emb = slater.get_veff(rdm1_emb, H2_emb, hyb=1.0, ghf=True)
            H1 -= JK_emb

            # add extra local term to H1, only to impurity
            hcore_add = kwargs.get("hcore_add", None)
            if hcore_add is not None:
                H1 += transform_imp(basis_Ra, basis_Rb, hcore_add)

            log.debug(1, "Construct JK_core")
            lattice.JK_core = H1 - hcore_emb

    # transform Mu (-mu*I in impurity and environment)
    log.debug(1, "transform Mu")
    mu_mat = np.zeros((2, nao, nao))
    np.fill_diagonal(mu_mat[0], -mu)
    np.fill_diagonal(mu_mat[1],  mu)
    H1 += transform_local(basis_Ra, basis_Rb, mu_mat)

    if add_vcor:
        # then add Vcor, only in environment;
        # add it everywhere then subtract impurity part
        log.debug(1, "transform Vcor")
        H1 += transform_local(basis_Ra, basis_Rb, vcor.get())

        if not "fitting" in kwargs or not kwargs["fitting"]:
            # for fitting purpose, we need H1 with vcor on impurity
            H1 -= transform_imp(basis_Ra, basis_Rb, vcor.get())

        # subtract impurity Fock if necessary
        # i.e. rho_kl[2(ij||kl)-(il||jk)] where i,j,k,l are all impurity
        if JK_imp is not None:
            log.debug(1, "transform impurity JK")
            H1 -= transform_imp(basis_Ra, basis_Rb, JK_imp)
    return H1[np.newaxis], ovlp_emb

def foldRho_k(GRho_k, basis_k):
    """
    Fold the density matrix to the embedding space.

    Args:
        GRho_k: (nkpts, nso, nso),
        basis_k: (nkpts, nso, nbasis)
    Return:
        GRho_emb: (nbasis, nbasis)
    """
    return slater_helper.transform_trans_inv_k(basis_k, GRho_k)

def addDiag(v, scalar):
    rep = v.get()
    nscsites = rep.shape[1]
    rep[0] += np.eye(nscsites) * scalar
    rep[1] -= np.eye(nscsites) * scalar
    v.assign(rep)
    return v

def keep_vcor_trace_fixed(vcor_new, vcor):
    dvcor_mat = vcor_new.get() - vcor.get()
    ddiagV = (np.average(np.diagonal(dvcor_mat[0])) -
              np.average(np.diagonal(dvcor_mat[1]))) * 0.5
    vcor_new = addDiag(vcor_new, -ddiagV)
    return vcor_new

def transformResults(GRhoEmb, E, lattice, basis, ImpHam, H1e, mu,
                     fit_ghf=False, **kwargs):
    """
    Transform results.
    Get density matrix, dmet energy (of non-interacting bath)
    and electron number.

    Edmet = E1 + E2
    where E1 = partial Tr(rho, H1),
    H1 should not include contribution from mu and last_dmu
    E2 = E_solver - <psi |hemb| psi>, psi is the wavefunction
    """
    if fit_ghf:
        nelec = 0.0
        for i, C in enumerate(basis):
            dm = mdot(C, GRhoEmb, C.conj().T)
            norb = dm.shape[-1] // 2
            nelec += dm[range(norb), range(norb)].sum() - \
                     dm[range(norb, norb*2), range(norb, norb*2)].sum() + \
                     norb
        GRhoImp = mdot(basis[0], GRhoEmb, basis[0].conj().T)
    else:
        ncells, nso, nbasis = basis.shape
        nao = nso // 2
        # ZHC NOTE we need lattice.imp_idx for multi-frag version.
        #imp_idx0 = kwargs.get("imp_idx", lattice.imp_idx)
        imp_idx0 = lattice.imp_idx
        imp_idxa, imp_idxb = idx_ao2so(imp_idx0, nao)
        log.debug(1, "transformResults:\nimp_idxa: %s imp_idxb: %s",
                  format_idx(imp_idxa), format_idx(imp_idxb))

        # special treatment for rdm1 from uhf type solver
        if GRhoEmb.ndim == 3:
            if GRhoEmb.shape[0] == 1:
                GRhoEmb = GRhoEmb[0]
            elif GRhoEmb.shape[0] == 2:
                GRhoEmb = GRhoEmb.sum(axis=0)
            else:
                raise ValueError

        GRhoImp = mdot(basis[0], GRhoEmb, basis[0].conj().T)
        nelec = GRhoImp[imp_idxa, imp_idxa].sum() - \
                GRhoImp[imp_idxb, imp_idxb].sum() + len(imp_idxb)
    if E is not None:
        last_dmu = kwargs["last_dmu"]
        basis_Ra, basis_Rb = separate_basis(basis)

        # first compute E2
        # ZHC NOTE here, when using fock as embedding hamiltonian,
        # be careful about the double counting.
        E2 = E - np.einsum('pq, qp ->', ImpHam.H1["cd"][0], GRhoEmb) - ImpHam.H0

        # then compute E1 from effective Ham
        dmu_idx = kwargs.get("dmu_idx", None)
        if dmu_idx is None:
            dmu_idx = imp_idx0

        # ZHC NOTE here imp_idx is in the embedding basis
        imp_idx0 = kwargs.get("imp_idx", np.arange(lattice.nimp))
        imp_idxa, imp_idxb = idx_ao2so(imp_idx0, lattice.nimp)
        imp_idx = imp_idxa + imp_idxb

        env_idx = [idx for idx in range(nbasis) if idx not in imp_idx]
        imp_env = np.ix_(imp_idx, env_idx)
        env_imp = np.ix_(env_idx, imp_idx)
        env_env = np.ix_(env_idx, env_idx)

        H1_scaled = ImpHam.H1["cd"][0].copy()
        # add back last_dmu only on impurity
        mu_mat = np.zeros((2, nao, nao))
        mu_mat[0][dmu_idx, dmu_idx] =  last_dmu
        mu_mat[1][dmu_idx, dmu_idx] = -last_dmu
        H1_scaled += transform_imp(basis_Ra, basis_Rb, mu_mat)

        # add back mu global
        np.fill_diagonal(mu_mat[0],  mu)
        np.fill_diagonal(mu_mat[1], -mu)
        H1_scaled += transform_local(basis_Ra, basis_Rb, mu_mat)

        # remove the JK_core if possible
        if lattice.JK_core is not None:
            H1_scaled -= 0.5 * lattice.JK_core

        # scale by the number of imp indices
        H1_scaled[imp_env] *= 0.5
        H1_scaled[env_imp] *= 0.5
        H1_scaled[env_env]  = 0.0

        E1 = np.einsum('pq, qp ->', H1_scaled, GRhoEmb)
        Efrag = E1 + E2 + ImpHam.H0
        log.debug(1, "NIB energy: E0 = %15.8g , E1 = %15.8g , "
                  "E2 = %15.8g, Efrag = %15.8g", ImpHam.H0, E1, E2, Efrag)
    else:
        Efrag = None
    return GRhoImp, Efrag, nelec

def get_veff_from_rdm1_emb(lattice, rdm1_emb, basis, kmf=None, C_ao_lo=None,
                           return_update=False, sign=None):
    """
    First construct rdm1_glob and then compute the veff in LO basis.
    rdm1_emb, (nso, nso).
    """
    if not isinstance(lattice, Iterable):
        lattice = [lattice]
        rdm1_emb = [rdm1_emb]
        basis = [basis]
    if kmf is None:
        kmf = lattice[0].kmf
    if C_ao_lo is None:
        C_ao_lo = np.asarray(lattice[0].C_ao_lo)
    nkpts = lattice[0].nkpts
    nao = lattice[0].mol.nao
    nlo = lattice[0].nlo
    nso = nlo * 2

    if C_ao_lo.ndim == 4:
        if C_ao_lo.shape[0] == 1:
            C_ao_lo = la.block_diag(C_ao_lo[0], C_ao_lo[0])
        elif C_ao_lo.shape[0] == 2:
            C_ao_lo = la.block_diag(C_ao_lo[0], C_ao_lo[1])
        else:
            raise ValueError
    assert C_ao_lo.shape == (nkpts, nao*2, nso)

    rdm1_glob = get_rho_glob_k(basis, lattice, rdm1_emb, sign=sign)
    np.save("rdm1_glob_lo_k.npy", rdm1_glob)
    rdm1_veff = make_basis.transform_rdm1_to_ao(rdm1_glob, C_ao_lo)

    if getattr(lattice[0].cell, 'pbc_intor', None):
        vj, vk = kmf.get_jk(dm_kpts=rdm1_veff)
        veff_ao = vj - vk
    else:
        vj, vk = kmf.get_jk(dm=rdm1_veff[0])
        veff_ao = vj - vk
        veff_ao = veff_ao[None]

    veff = make_basis.transform_h1_to_lo(veff_ao, C_ao_lo)
    if return_update:
        return veff, veff_ao, lattice[0].k2R(rdm1_glob)
    else:
        return veff

def get_veff_from_rdm1_emb_mpi(Lat, rdm1_emb, basis, comm, return_update=False):
    """
    Get veff from embedding rdm1 in LO basis, MPI version.
    Each process treat 1 impurity problem.
    """
    from libdmet.system import lattice
    nfrag = comm.Get_size()
    rank  = comm.Get_rank()
    comm.Barrier()
    val_idx_col  = comm.gather(Lat.val_idx, root=0)
    virt_idx_col = comm.gather(Lat.virt_idx, root=0)
    core_idx_col = comm.gather(Lat.core_idx, root=0)
    basis_col    = comm.gather(basis, root=0)
    rdm1_emb_col = comm.gather(rdm1_emb, root=0)
    if isinstance(Lat, lattice.Lattice):
        res_type = complex
    else:
        res_type = float

    if return_update:
        if rank == 0:
            Lat_col = [copy.copy(Lat) for i in range(nfrag)]
            with lib.temporary_env(log, verbose="FATAL"):
                for i in range(nfrag):
                    Lat_col[i].set_val_virt_core(val_idx_col[i], virt_idx_col[i], core_idx_col[i])
            veff, veff_ao, rdm1_glob_R = get_veff_from_rdm1_emb(Lat_col, rdm1_emb_col, basis_col, return_update=True)
            rdm1_glob_R = np.asarray(rdm1_glob_R.real, order='C')
        else:
            nkpts, nso, _ = basis.shape
            nao = Lat.C_ao_lo.shape[-2]
            veff = np.zeros((nkpts, nso, nso), dtype=res_type)
            veff_ao = np.zeros((nkpts, nao*2, nao*2), dtype=res_type)
            rdm1_glob_R = np.zeros((nkpts, nso, nso))
        comm.Barrier()
        comm.Bcast(veff, root=0)
        comm.Bcast(veff_ao, root=0)
        comm.Bcast(rdm1_glob_R, root=0)
        return veff, veff_ao, rdm1_glob_R
    else:
        if rank == 0:
            Lat_col = [copy.copy(Lat) for i in range(nfrag)]
            with lib.temporary_env(log, verbose="FATAL"):
                for i in range(nfrag):
                    Lat_col[i].set_val_virt_core(val_idx_col[i], virt_idx_col[i], core_idx_col[i])
                veff = get_veff_from_rdm1_emb(Lat_col, rdm1_emb_col, basis_col)
        else:
            nkpts, nso, _ = basis.shape
            veff = np.zeros((nkpts, nso, nso), dtype=res_type)
        comm.Barrier()
        comm.Bcast(veff, root=0)
        return veff

def get_H_dmet(basis, lattice, ImpHam, last_dmu=None, mu=None,
               imp_idx=None, dmu_idx=None, add_vcor_to_E=False,
               vcor=None, compact=True, rdm1_emb=None,
               veff=None, rebuild_veff=False, E1=None,
               GV0=None, GV1=None, **kwargs):
    """
    Get a DMET hamiltonian, which is scaled by number of impurity indices,
    and can be directly used for evaluation of DMET energy.
    The impurity index can be specified by imp_idx (should include alpha and
    beta indices).

    rdm1_emb: embedding rdm1, (neo, neo).
    veff: if provide, use veff as JK_core, (nkpts, nso, nso).
    rebuild_veff: re-evaluate the JK_core from the global density matrix.
    E1: if provide, will use it as E1 (including hcore and J, K).
    """
    log.debug(0, "Construct Heff for DMET.")
    nbasis = basis.shape[-1]
    nao = nscsites = lattice.nscsites
    basis_Ra, basis_Rb = separate_basis(basis)
    basis_k = lattice.R2k_basis(basis)
    basis_ka, basis_kb = separate_basis(basis_k)

    if imp_idx is None:
        imp_idx0 = np.arange(lattice.nimp)
    else:
        imp_idx0 = imp_idx
    imp_idxa, imp_idxb = idx_ao2so(imp_idx0, lattice.nimp)
    imp_idx = imp_idxa + imp_idxb
    env_idx = np.asarray([idx for idx in range(nbasis)
                          if idx not in imp_idx], dtype=int)
    log.debug(1, "imp_idx: %s", format_idx(imp_idx))
    log.debug(1, "env_idx: %s", format_idx(env_idx))

    if E1 is None:
        hcore_k = lattice.getH1(kspace=True)
        H1_scaled = transform_trans_inv_k(basis_ka, basis_kb, hcore_k)

        # note the double counting from JK_core cf. HF energy
        if (veff is not None) or rebuild_veff:
            if veff is None:
                veff = get_veff_from_rdm1_emb(lattice, rdm1_emb, basis)
            JK_core = slater_helper.transform_trans_inv_k(basis_k, veff)
            veff_loc = _get_veff_ghf(rdm1_emb, ImpHam.H2["ccdd"])
            JK_core -= veff_loc
            if lattice.JK_core is not None:
                log.debug(1, "difference between JK_glob and JK_HF: %15.8g",
                          max_abs(JK_core - lattice.JK_core))
        else:
            if lattice.JK_core is not None:
                JK_core = lattice.JK_core
            else:
                JK_core = 0.0

        H1_scaled += 0.5 * JK_core
        if add_vcor_to_E:
            log.debug(0, "Add Vcor to energy expression")
            H1_scaled += transform_local(basis_Ra, basis_Rb, vcor.get() * 0.5)
            H1_scaled -= transform_imp(basis_Ra, basis_Rb, vcor.get() * 0.5)
        # ZHC NOTE
        if GV1 is not None:
            GV1_emb = slater_helper.transform_trans_inv_k(basis_k, GV1)
            H1_scaled -= GV1_emb
        H1_scaled = get_H1_scaled(H1_scaled[None], imp_idx, env_idx)

        # H0 part
        H0 = lattice.getH0()
    else:
        # manually specify E1 (contribution from hcore and J, K)
        H1_scaled = -get_veff_ghf(rdm1_emb, ImpHam.H2["ccdd"])
        H1_scaled = get_H1_scaled(H1_scaled[None], imp_idx, env_idx)
        H0 = (E1 + lattice.getH0()).real

    # ZHC NOTE
    if GV0 is not None:
        H0 = H0 - GV0 * 0.5

    # H2 part
    # restore 4-fold symmetry
    H2_scaled = ao2mo.restore(4, ImpHam.H2["ccdd"][0], nbasis)
    H2_scaled = get_H2_scaled(H2_scaled[None], imp_idx, env_idx)

    ImpHam_dmet = integral.Integral(nbasis, True, False, H0,
                                    {"cd": H1_scaled}, {"ccdd": H2_scaled})
    if not compact:
        log.warn("Restoring 1-fold symmetry in dmet Hamiltonian...")
        ImpHam_dmet = restore_Ham(ImpHam_dmet, 1, in_place=True)
    return ImpHam_dmet

def get_E_dmet(basis, lattice, ImpHam, solver, solver_args={}, **kwargs):
    ImpHam_scaled = get_H_dmet(basis, lattice, ImpHam, **kwargs)
    E = solver.run_dmet_ham(ImpHam_scaled, **solver_args)
    return E

def get_E_dmet_HF(basis, lattice, ImpHam, last_dmu, mu, solver, **kwargs):
    """
    Get a DMET energy for a given mean-field solver.
    imp_idx should be spatial orbital labels.
    """
    nbasis = basis.shape[-1]
    nscsites = lattice.nscsites
    basis_Ra, basis_Rb = separate_basis(basis)

    # spatial orb
    imp_idx0 = kwargs.get("imp_idx", np.arange(lattice.nimp))
    # spin orb
    imp_idxa, imp_idxb = idx_ao2so(imp_idx0, lattice.nimp)
    imp_idx = imp_idxa + imp_idxb
    env_idx = [idx for idx in range(nbasis) if idx not in imp_idx]
    imp_env = np.ix_(imp_idx, env_idx)
    env_imp = np.ix_(env_idx, imp_idx)
    env_env = np.ix_(env_idx, env_idx)

    rdm1 = solver.mf.make_rdm1()
    h1e = solver.mf.get_hcore()
    fock = solver.mf.get_fock(h1e=h1e, dm=rdm1)
    heff = (h1e + fock) * 0.5

    # remove the double counting from JK_core cf. HF energy
    if lattice.JK_core is not None:
        heff -= 0.5*lattice.JK_core

    # add back mu only on impurity
    imp_idx0 = lattice.imp_idx
    mu_mat = np.zeros((2, nscsites, nscsites))
    mu_mat[0][imp_idx0, imp_idx0] =  last_dmu
    mu_mat[1][imp_idx0, imp_idx0] = -last_dmu
    heff += transform_imp(basis_Ra, basis_Rb, mu_mat)

    # add back mu global
    np.fill_diagonal(mu_mat[0],  mu)
    np.fill_diagonal(mu_mat[1], -mu)
    heff += transform_local(basis_Ra, basis_Rb, mu_mat)

    # scaled
    heff[imp_env] *= 0.5
    heff[env_imp] *= 0.5
    heff[env_env]  = 0.0
    # ZHC FIXME should there be any contribution of constant?
    E = np.sum(heff * rdm1) #+ solver.mf.energy_nuc()
    return E

def get_dV_dparam(vcor, basis, basis_k, lattice, P_act=None, compact=True):
    """
    Get dV / dparam.

    Args:
        compact: if true dv is the tril part, dv_tril / dparam

    Returns:
        dv_dparam.
    """
    nkpts, nlo, nbasis = basis.shape
    basis_Ra, basis_Rb = separate_basis(basis)
    if compact:
        nbasis_pair = nbasis * (nbasis + 1) // 2
        tril_idx = np.tril_indices(nbasis)
        dV_dparam = np.empty((vcor.length(), nbasis_pair))
    else:
        dV_dparam = np.empty((vcor.length(), nbasis, nbasis))

    if vcor.is_local():
        if P_act is None:
            for ip in range(vcor.length()):
                if compact:
                    dV_dparam[ip] = \
                            transform_local(basis_Ra, basis_Rb,
                                            vcor.gradient()[ip])[tril_idx]
                else:
                    dV_dparam[ip] = transform_local(basis_Ra, basis_Rb,
                                                    vcor.gradient()[ip])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # release the grad memory.
    vcor.grad   = None
    vcor.grad_k = None
    return dV_dparam

def FitVcorEmb(rho, lattice, basis, vcor, mu, beta=np.inf, MaxIter=300,
               imp_fit=False, imp_idx=None, det=False, det_idx=None,
               CG_check=False, BFGS=False, diff_criterion=None, **kwargs):
    """
    Fitting the correlation potential in the embedding space.

    Args:
        imp_idx: a list of spatial orbital idx for imp_fit
        det_idx: a list of spatial orbital idx for diagonal fit

    Additional kwargs:
        fix_mu: [False], do not search fermi level
        mu0: [None], used for fix_mu (fermi level)
        num_grad: [False], use numerical gradient
        test_grad: [False], test the analytical gradient with numerical
    """
    param_begin = vcor.param.copy()
    nbasis = basis.shape[-1]
    nao = lattice.nscsites
    basis_Ra, basis_Rb = separate_basis(basis)
    basis_k = lattice.R2k_basis(basis)
    basis_ka, basis_kb = separate_basis(basis_k)
    nelec = kwargs.get("nelec", None)
    if nelec is None:
        nelec = nbasis // 2
    tol_deg = kwargs.get("tol_deg", 1e-3)

    mu0      = kwargs.get("mu0", None)
    fix_mu   = kwargs.get("fix_mu", False)
    num_grad = kwargs.get("num_grad", False)

    if lattice.use_hcore_as_emb_ham:
        fock_k = lattice.getH1(kspace=True)
    else:
        fock_k = lattice.getFock(kspace=True)
    fock_k = np.array(fock_k, copy=True)
    ovlp_k = lattice.get_ovlp(kspace=True)
    assert fock_k.ndim == 4 and fock_k.shape[0] == 3
    assert ovlp_k.ndim == 4 and ovlp_k.shape[0] == 3

    # initialize imp_idx and det_idx:
    imp_bath_fit = False
    if imp_fit:
        imp_idx = list(range(lattice.nimp))
        det_idx = []
    elif det:
        imp_idx = []
        det_idx = list(range(lattice.nimp))
    elif imp_idx is None:
        if det_idx is None: # imp + bath fitting
            imp_idx = list(range(nbasis))
            det_idx = []
            imp_bath_fit = True
        else:
            imp_idx = []
    elif det_idx is None:
        det_idx = []
    imp_idx = list(imp_idx)
    det_idx = list(det_idx)

    def convert_idx(idx0, norb):
        idx0_a, idx0_b = idx_ao2so(idx0, norb)
        return idx0_a + idx0_b

    if not imp_bath_fit:
        imp_idx = convert_idx(imp_idx, lattice.nimp)
        det_idx = convert_idx(det_idx, lattice.nimp)

    log.info("impurity fitting? %s", imp_fit)
    log.debug(1, "imp_idx: %s", format_idx(imp_idx))
    log.info("det (diagonal fitting)? %s", det)
    log.debug(1, "det_idx: %s", format_idx(det_idx))

    fit_idx = imp_idx + det_idx
    nimp, nidx = len(imp_idx), len(fit_idx)
    imp_mesh = np.ix_(imp_idx, imp_idx)
    det_mesh = (det_idx, det_idx)
    imp_fill_mesh = (slice(nimp), slice(nimp))
    det_fill_mesh = (range(nimp, nidx), range(nimp, nidx))
    if len(np.unique(fit_idx)) != nidx:
        log.warn("fit_idx has repeated indices: %s", format_idx(fit_idx))

    # pre-allocate the objects for efficiency:
    rho_target = np.zeros((nidx, nidx))
    rho1 = np.zeros_like(rho_target)
    ew = np.empty((nbasis))
    ev = np.empty((nbasis, nbasis))

    # fock in the embedding space
    vcor_mat = kwargs.get("vcor_mat", None)
    if vcor_mat is not None:
        for s in range(3):
            fock_k[s] += vcor_mat[s]

    embH1 = transform_trans_inv_k(basis_ka, basis_kb, fock_k)
    # ZHC NOTE add mu
    mu_mat = np.zeros((2, nao, nao))
    np.fill_diagonal(mu_mat[0], -mu)
    np.fill_diagonal(mu_mat[1],  mu)
    embH1 += transform_local(basis_Ra, basis_Rb, mu_mat)

    ovlp_emb = transform_trans_inv_k(basis_ka, basis_kb, ovlp_k)

    # dV / dparam
    dV_dparam = get_dV_dparam(vcor, basis, basis_k, lattice, P_act=None,
                              compact=True)
    diag_idx = (np.arange(nbasis), np.arange(nbasis))
    tril_idx = np.tril_indices(nbasis)

    # rho_target
    rho_target[imp_fill_mesh] = rho[imp_mesh]
    rho_target[det_fill_mesh] = rho[det_mesh]

    def Vemb_param(param):
        """
        Give param return corresponding Vemb.
        """
        if dV_dparam.ndim == 2:
            tmp = np.tensordot(param, dV_dparam, axes=(0, 0))
            v_emb = np.zeros((nbasis, nbasis))
            # only the lower part is needed
            v_emb[tril_idx] = tmp
        else:
            v_emb = np.tensordot(param, dV_dparam, axes=(0, 0))
        return v_emb

    def errfunc(param):
        embHeff = embH1 + Vemb_param(param)
        ew, ev = la.eigh(embHeff, ovlp_emb)
        if not fix_mu:
            mu_quasi = 0.5 * (ew[nelec-1] + ew[nelec])
        else:
            mu_quasi = mu0

        ewocc, mu_quasi, _ = assignocc(ew, nelec, beta, mu_quasi, fix_mu=fix_mu,
                                       thr_deg=tol_deg)
        tmp = np.dot(ev*ewocc, ev.T)
        rho1[imp_fill_mesh] = tmp[imp_mesh]
        rho1[det_fill_mesh] = tmp[det_mesh]
        drho = rho1 - rho_target
        return la.norm(drho) / np.sqrt(2.0)

    def gradfunc(param):
        embHeff = embH1 + Vemb_param(param)
        ew, ev = la.eigh(embHeff, ovlp_emb)
        if not fix_mu:
            mu_quasi = 0.5 * (ew[nelec-1] + ew[nelec])
        else:
            mu_quasi = mu0
        ewocc, mu_quasi, _ = assignocc(ew, nelec, beta, mu_quasi, fix_mu=fix_mu,
                                       thr_deg=tol_deg)

        tmp = np.dot(ev*ewocc, ev.T)
        rho1[imp_fill_mesh] = tmp[imp_mesh]
        rho1[det_fill_mesh] = tmp[det_mesh]

        drho = rho1 - rho_target
        val = la.norm(drho)

        occ = nelec
        ewocc, ewvirt = ew[:occ],    ew[occ:]
        evocc, evvirt = ev[:, :occ], ev[:, occ:]

        if dV_dparam.ndim == 2:
            e_mn = 1.0 / (-ewvirt.reshape((-1,1)) + ewocc)
            temp_mn = mdot(evvirt[fit_idx].T, drho,
                           evocc[fit_idx]) * e_mn / (val * np.sqrt(2.0))
            dw_dV_full  = mdot(evvirt, temp_mn, evocc.T)
            dw_dV_full  = dw_dV_full + dw_dV_full.T
            dw_dV_full *= 2.0
            dw_dV_full[diag_idx] *= 0.5
            dw_dV = dw_dV_full[tril_idx]
            res = np.tensordot(dV_dparam, dw_dV, axes=((1,), (0,)))
        else:
            dw_dV = np.empty((nbasis, nbasis))
            e_mn = 1.0 / (-ewvirt.reshape((-1,1)) + ewocc)
            temp_mn = mdot(evvirt[fit_idx].T, drho,
                           evocc[fit_idx]) * e_mn / (val * np.sqrt(2.0))
            dw_dV = mdot(evvirt, temp_mn, evocc.T)
            dw_dV = dw_dV + dw_dV.T
            res = np.tensordot(dV_dparam, dw_dV, axes=((1, 2), (0, 1)))

        # project out the diagonal component
        #if remove_diag_grad:
        #    for s in range(spin):
        #        res[vcor.diag_indices()[s]] -= \
        #                np.average(res[vcor.diag_indices()[s]])
        return res

    def gradfunc_ft(param):
        embHeff = embH1 + Vemb_param(param)

        ew, ev = la.eigh(embHeff, ovlp_emb)
        if not fix_mu:
            mu_quasi = 0.5 * (ew[nelec-1] + ew[nelec])
        else:
            mu_quasi = mu0
        ewocc, mu_quasi, _ = assignocc(ew, nelec, beta, mu_quasi, fix_mu=fix_mu,
                                       thr_deg=tol_deg)

        tmp = np.dot(ev*ewocc, ev.T)
        rho1[imp_fill_mesh] = tmp[imp_mesh]
        rho1[det_fill_mesh] = tmp[det_mesh]

        drho = rho1 - rho_target
        val = la.norm(drho)

        dw_dv = ftsystem.get_dw_dv(ew, ev, drho, mu_quasi, beta, fix_mu=fix_mu,
                                   fit_idx=fit_idx,
                                   compact=(dV_dparam.ndim == 2))
        dw_dparam = dV_dparam.reshape(dV_dparam.shape[0], -1).dot(dw_dv.ravel()) \
                    / (2.0 * val * np.sqrt(2.0))

        # project out the diagonal component
        #if remove_diag_grad:
        #    for s in range(spin):
        #        dw_dparam[vcor.diag_indices()[s]] -= \
        #                np.average(dw_dparam[vcor.diag_indices()[s]])
        return dw_dparam

    err_begin = errfunc(vcor.param)
    if beta == np.inf:
        log.info("Using analytic gradient for 0 T")
    else:
        log.info("Using analytic gradient for finite T, beta = %s", beta)
        gradfunc = gradfunc_ft

    if kwargs.get("test_grad", False):
        """
        This block is for testing gradient.
        """
        param_rand = kwargs.get("param_rand", None)
        if param_rand is None:
            np.random.seed(10086)
            param_rand = np.random.random(vcor.param.shape)
            param_rand = (param_rand - 0.5) * 0.1
        test_grad(param_rand.copy(), errfunc, gradfunc, dx=1e-4)
        test_grad(param_rand.copy(), errfunc, gradfunc, dx=1e-5)

    if num_grad:
        log.warn("You are using numerical gradient...")
        gradfunc = None

    param, err_end, pattern, gnorm_res = minimize(errfunc, vcor.param, MaxIter, gradfunc, **kwargs)
    vcor.update(param)

    log.info("Minimizer converge pattern: %d ", pattern)
    log.info("Current function value: %15.8f", err_end)
    log.info("Norm of gradients: %s", gnorm_res)
    log.info("Norm diff of x: %15.8f", max_abs(param - param_begin))

    # Check with scipy minimizer
    if CG_check and (pattern == 0 or gnorm_res > 1.0e-4):
        log.info("Check with optimizer in Scipy...")
        from scipy import optimize as opt
        param_new = param.copy()
        gtol = max(5.0e-5, gnorm_res*0.1)
        gtol = min(gtol, 1.0e-2)
        if BFGS:
            log.info("BFGS used in check")
            method = 'BFGS'
        else:
            log.info("CG used in check")
            method = 'CG'

        min_result = opt.minimize(errfunc, param_new, method=method,
                                  jac=gradfunc,
                                  options={'maxiter':
                                           min(len(param_new)*10, MaxIter),
                                           'disp': True, 'gtol': gtol})
        param_new_2 = min_result.x
        log.info("CG Final Diff: %s", min_result.fun)
        log.info("Converged: %s", min_result.status)
        log.info("Jacobian: %s", max_abs(min_result.jac))
        if(not min_result.success):
            log.warn("Minimization unsuccessful. Message:\n%s", min_result.message)

        gnorm_new = max_abs(min_result.jac)
        diff_CG_old = max_abs(param_new_2 - param_new)
        log.info("max diff in x between %s and old: %s", method, diff_CG_old)
        if diff_criterion is None:
            if pattern == 0:
                diff_criterion = 2.0
            else:
                diff_criterion = 1.0
        if (gnorm_new < gnorm_res * 0.9) and (min_result.fun < err_end) \
                and (diff_CG_old < diff_criterion):
            log.info("New result used")
            vcor.update(param_new_2)
            err_end = min_result.fun
        else:
            log.info("Old result used")
            vcor.update(param_new)
    else:
        log.info("Old result used")

    if log.Level[log.verbose] > log.Level["DEBUG0"]:
        errfunc(vcor.param)
        log.debug(1, "rdm1 target:\n%s", rho_target)
        log.debug(1, "rdm1 fitted:\n%s", rho1)
    return vcor, err_begin, err_end

def get_dV_dparam_full(vcor, lattice, P_act=None, compact=True):
    """
    Get dV / dparam for the full problem.

    Args:
        compact: if true dv is the tril part, dv_tril / dparam

    Returns:
        dv_dparam.
    """
    nao = lattice.nao
    nso = nao * 2
    nkpts = lattice.nkpts
    assert vcor.is_local()
    grad = vcor.gradient()

    dV_dparam = np.empty((vcor.length(), nso, nso))
    dV_dparam[:, :nao, :nao] = grad[:, 0]
    dV_dparam[:, nao:, nao:] = grad[:, 1]
    dV_dparam[:, :nao, nao:] = grad[:, 2]
    dV_dparam[:, nao:, :nao] = grad[:, 2].transpose(0, 2, 1).conj()

    grad = None
    vcor.grad   = None
    vcor.grad_k = None

    if compact:
        nso_pair = nso * (nso + 1) // 2
        dV_dparam = lib.pack_tril(dV_dparam.reshape(-1, nso, nso))
        dV_dparam = dV_dparam.reshape(vcor.length(), nso_pair)

    return dV_dparam

def FitVcorFull(rho, lattice, basis, vcor, mu, beta, filling, MaxIter=20,
                imp_fit=False, imp_idx=None, det=False, det_idx=None,
                CG_check=False, BFGS=False, diff_criterion=None, scf=False,
                **kwargs):
    """
    Fit the correlation potential in the full lattice space.
    """
    # ZHC FIXME
    # When potential gives exact degeneracy,
    # the error compared to numerical gradient is large
    # not sure if this is correct
    param_begin = vcor.param.copy()
    if basis is not None:
        nkpts, nso, nbasis = basis.shape
        nao = nso // 2
        basis_Ra, basis_Rb = separate_basis(basis)
        basis_k = lattice.R2k_basis(basis)
        basis_ka, basis_kb = separate_basis(basis_k)
    else:
        nao = lattice.nscsites
        nkpts = lattice.nkpts
        nso = nao * 2

    mu0      = kwargs.get("mu0", None)
    fix_mu   = kwargs.get("fix_mu", False)
    num_grad = kwargs.get("num_grad", False)
    bogo_only = kwargs.get("bogo_only", False)

    # initialize imp_idx and det_idx:
    imp_bath_fit = False
    if imp_fit:
        if imp_idx is None:
            imp_idx = list(range(lattice.nimp))
        det_idx = []
    elif det:
        imp_idx = []
        if det_idx is None:
            det_idx = list(range(lattice.nimp))
    elif imp_idx is None:
        if det_idx is None: # imp + bath fitting
            imp_idx = list(range(nbasis))
            det_idx = []
            imp_bath_fit = True
        else:
            imp_idx = []
    elif det_idx is None:
        det_idx = []
    imp_idx = list(imp_idx)
    det_idx = list(det_idx)

    def convert_idx(idx0, norb):
        idx0_a, idx0_b = idx_ao2so(idx0, norb)
        return idx0_a + idx0_b

    if not imp_bath_fit:
        # ZHC NOTE here should use nao, not nimp
        imp_idx = convert_idx(imp_idx, nao)
        det_idx = convert_idx(det_idx, nao)

    log.info("impurity fitting? %s", imp_fit)
    log.debug(1, "imp_idx: %s", format_idx(imp_idx))
    log.info("det (diagonal fitting)? %s", det)
    log.debug(1, "det_idx: %s", format_idx(det_idx))
    log.debug(1, "only fit bogoliubov part: %s", bogo_only)

    fit_idx = imp_idx + det_idx
    nimp, nidx = len(imp_idx), len(fit_idx)
    nimp_half = nimp // 2
    ndet_half = len(det_idx) // 2
    nidx_half = nidx // 2

    imp_mesh = np.ix_(imp_idx, imp_idx)
    det_mesh = (det_idx, det_idx)
    imp_fill_mesh = (slice(nimp), slice(nimp))
    det_fill_mesh = (range(nimp, nidx), range(nimp, nidx))

    imp_fill_mesh_aa = (slice(nimp_half), slice(nimp_half))
    imp_fill_mesh_bb = (slice(nimp_half, nimp), slice(nimp_half, nimp))
    det_fill_mesh_aa = (range(nimp, nimp + ndet_half), range(nimp, nimp + ndet_half))
    det_fill_mesh_bb = (range(nimp + ndet_half, nidx), range(nimp + ndet_half, nidx))
    if len(np.unique(fit_idx)) != nidx:
        log.warn("fit_idx has repeated indices: %s", format_idx(fit_idx))

    # ZHC NOTE
    # rho should be rho_glob (R = 0) with shape (nso, nso)
    if rho.shape[-1] != nso:
        log.warn("FitVcorFull: target rho should has shape (%s, %s) , "
                 "now has shape %s ...", nso, nso, str(rho.shape))
    rho_target = np.zeros((nidx, nidx))
    rho_target[imp_fill_mesh] = rho[imp_mesh]
    rho_target[det_fill_mesh] = rho[det_mesh]

    # ZHC NOTE
    # choose to fit only bogoliubov part
    if bogo_only:
        rho_target[imp_fill_mesh_aa] = 0.0
        rho_target[imp_fill_mesh_bb] = 0.0
        rho_target[det_fill_mesh_aa] = 0.0
        rho_target[det_fill_mesh_bb] = 0.0

    rho1 = np.zeros_like(rho_target)

    # precompute the GFock
    Fock = lattice.getFock(kspace=True)
    GFock = pbc_hp.combine_H1_k(Fock)
    GFock[:, range(nao), range(nao)] -= mu
    GFock[:, range(nao, nso), range(nao, nso)] += mu
    assert GFock.shape[-1] == lattice.nao * 2

    nelec = nkpts * nso * 0.5
    nelec = mfd.check_nelec(nelec, None)[0]
    dV_dparam = get_dV_dparam_full(vcor, lattice, compact=True)

    if kwargs.get("use_mpi", False):
        log.info("fitting with MPI.")
        from libdmet.routine import mfd_mpi
        kpairs, kidx = mfd_mpi.get_kpairs_kidx(lattice.cell, lattice.kpts)

        def errfunc(param):
            vcor.update(param)
            vcor_mat = vcor.get(0, True)
            ew, ev = mfd_mpi.DiagGHF_symm(lattice.cell, GFock, vcor_mat, mu=None, kpairs=kpairs, kidx=kidx)
            ewocc, mu_quasi, nerr = assignocc(ew, nelec, beta, mu0=0.0, fix_mu=fix_mu)

            if imp_bath_fit:
                GRho = np.empty_like(ev)
                for k in range(nkpts):
                    GRho[k] = np.dot(ev[k]*ewocc[k], ev[k].conj().T)
                rho1[:] = foldRho_k(GRho, basis_k)
            else:
                GRhoT = 0.0
                for k in range(nkpts):
                    GRhoT += np.dot(ev[k]*ewocc[k], ev[k].conj().T)
                GRhoT /= nkpts
                if max_abs(GRhoT.imag) > mfd.IMAG_DISCARD_TOL:
                    log.warn("GRhoT has imag part %s", max_abs(GRhoT.imag))
                GRhoT = GRhoT.real

                rho1[imp_fill_mesh] = GRhoT[imp_mesh]
                rho1[det_fill_mesh] = GRhoT[det_mesh]

                if bogo_only:
                    rho1[imp_fill_mesh_aa] = 0.0
                    rho1[imp_fill_mesh_bb] = 0.0
                    rho1[det_fill_mesh_aa] = 0.0
                    rho1[det_fill_mesh_bb] = 0.0

            return la.norm((rho1 - rho_target)) / np.sqrt(2)

        def gradfunc_ft(param):
            vcor.update(param)
            vcor_mat = vcor.get(0, True)
            ew, ev = mfd_mpi.DiagGHF_symm(lattice.cell, GFock, vcor_mat, mu=None, kpairs=kpairs, kidx=kidx)
            ewocc, mu_quasi, nerr = assignocc(ew, nelec, beta, mu0=0.0, fix_mu=fix_mu)

            if imp_bath_fit:
                GRho = np.empty_like(ev)
                for k in range(nkpts):
                    GRho[k] = np.dot(ev[k]*ewocc[k], ev[k].conj().T)
                rho1[:] = foldRho_k(GRho, basis_k)
            else:
                GRhoT = 0.0
                for k in range(nkpts):
                    GRhoT += np.dot(ev[k]*ewocc[k], ev[k].conj().T)
                GRhoT /= nkpts
                GRhoT = GRhoT.real

                rho1[imp_fill_mesh] = GRhoT[imp_mesh]
                rho1[det_fill_mesh] = GRhoT[det_mesh]

                if bogo_only:
                    rho1[imp_fill_mesh_aa] = 0.0
                    rho1[imp_fill_mesh_bb] = 0.0
                    rho1[det_fill_mesh_aa] = 0.0
                    rho1[det_fill_mesh_bb] = 0.0

            if imp_bath_fit:
                # ZHC TODO implement the gradient for imp+bath fitting
                raise NotImplementedError

            drho = rho1 - rho_target
            val = la.norm(drho)

            dw_dparam = 0.0
            for k in range(nkpts):
                dw_dv = ftsystem.get_dw_dv(ew[k], ev[k], drho, mu_quasi, beta, fix_mu=fix_mu,
                                           fit_idx=fit_idx,
                                           compact=(dV_dparam.ndim == 2))
                dw_dparam += dV_dparam.reshape(dV_dparam.shape[0], -1).dot(dw_dv.ravel())

            dw_dparam = dw_dparam.real / (2.0 * val * np.sqrt(2.0) * nkpts)
            return dw_dparam
    else:
        def errfunc(param):
            vcor.update(param)
            ew, ev = mfd.DiagGHF_symm(GFock, vcor, mu=None, lattice=lattice)
            ewocc, mu_quasi, nerr = assignocc(ew, nelec, beta, mu0=0.0, fix_mu=fix_mu)

            if imp_bath_fit:
                GRho = np.empty_like(ev)
                for k in range(nkpts):
                    GRho[k] = np.dot(ev[k]*ewocc[k], ev[k].conj().T)
                rho1[:] = foldRho_k(GRho, basis_k)
            else:
                GRhoT = 0.0
                for k in range(nkpts):
                    GRhoT += np.dot(ev[k]*ewocc[k], ev[k].conj().T)
                GRhoT /= nkpts
                if max_abs(GRhoT.imag) > mfd.IMAG_DISCARD_TOL:
                    log.warn("GRhoT has imag part %s", max_abs(GRhoT.imag))
                GRhoT = GRhoT.real

                rho1[imp_fill_mesh] = GRhoT[imp_mesh]
                rho1[det_fill_mesh] = GRhoT[det_mesh]

                if bogo_only:
                    rho1[imp_fill_mesh_aa] = 0.0
                    rho1[imp_fill_mesh_bb] = 0.0
                    rho1[det_fill_mesh_aa] = 0.0
                    rho1[det_fill_mesh_bb] = 0.0

            return la.norm((rho1 - rho_target)) / np.sqrt(2)

        def gradfunc_ft(param):
            vcor.update(param)
            ew, ev = mfd.DiagGHF_symm(GFock, vcor, mu=None, lattice=lattice)
            ewocc, mu_quasi, nerr = assignocc(ew, nelec, beta, mu0=0.0, fix_mu=fix_mu)

            if imp_bath_fit:
                GRho = np.empty_like(ev)
                for k in range(nkpts):
                    GRho[k] = np.dot(ev[k]*ewocc[k], ev[k].conj().T)
                rho1[:] = foldRho_k(GRho, basis_k)
            else:
                GRhoT = 0.0
                for k in range(nkpts):
                    GRhoT += np.dot(ev[k]*ewocc[k], ev[k].conj().T)
                GRhoT /= nkpts
                GRhoT = GRhoT.real

                rho1[imp_fill_mesh] = GRhoT[imp_mesh]
                rho1[det_fill_mesh] = GRhoT[det_mesh]

                if bogo_only:
                    rho1[imp_fill_mesh_aa] = 0.0
                    rho1[imp_fill_mesh_bb] = 0.0
                    rho1[det_fill_mesh_aa] = 0.0
                    rho1[det_fill_mesh_bb] = 0.0

            if imp_bath_fit:
                # ZHC TODO implement the gradient for imp+bath fitting
                raise NotImplementedError

            drho = rho1 - rho_target
            val = la.norm(drho)

            dw_dparam = 0.0
            for k in range(nkpts):
                dw_dv = ftsystem.get_dw_dv(ew[k], ev[k], drho, mu_quasi, beta, fix_mu=fix_mu,
                                           fit_idx=fit_idx,
                                           compact=(dV_dparam.ndim == 2))
                dw_dparam += dV_dparam.reshape(dV_dparam.shape[0], -1).dot(dw_dv.ravel())

            dw_dparam = dw_dparam.real / (2.0 * val * np.sqrt(2.0) * nkpts)
            return dw_dparam

    if beta == np.inf:
        log.info("Using analytic gradient for 0 T")
        if not num_grad:
            raise NotImplementedError
    else:
        log.info("Using analytic gradient for finite T, beta = %s", beta)
        gradfunc = gradfunc_ft

    if kwargs.get("test_grad", False):
        """
        This block is for testing gradient.
        """
        param_rand = kwargs.get("param_rand", None)
        if param_rand is None:
            np.random.seed(10086)
            param_rand = np.random.random(vcor.param.shape)
            param_rand = (param_rand - 0.5) * 0.1
        test_grad(param_rand.copy(), errfunc, gradfunc, dx=1e-4)
        test_grad(param_rand.copy(), errfunc, gradfunc, dx=1e-5)

    if num_grad:
        log.warn("You are using numerical gradient...")
        gradfunc = None

    err_begin = errfunc(param_begin)
    param, err_end, pattern, gnorm_res = minimize(errfunc, param_begin.copy(),
                                                  MaxIter, gradfunc, **kwargs)
    vcor.update(param)

    log.info("Minimizer converge pattern: %d ", pattern)
    log.info("Current function value: %15.8f", err_end)
    log.info("Norm of gradients: %s", gnorm_res)
    log.info("Norm diff of x: %15.8f", max_abs(param - param_begin))

    if log.Level[log.verbose] > log.Level["DEBUG0"]:
        errfunc(vcor.param)
        log.debug(1, "rdm1 target:\n%s", rho_target)
        log.debug(1, "rdm1 fitted:\n%s", rho1)

    return vcor, err_begin, err_end

def FitVcorFull_mu(rho, lattice, basis, vcor, mu, beta, filling, MaxIter=20,
                   imp_fit=False, imp_idx=None, det=False, det_idx=None,
                   CG_check=False, BFGS=False, diff_criterion=None, scf=False,
                   use_cvx_frac=False, **kwargs):
    """
    Fit the correlation potential in the full lattice space, with chemical mu fitting.
    """
    # ZHC FIXME
    # When potential gives exact degeneracy,
    # the error compared to numerical gradient is large
    # not sure if this is correct
    param_begin = vcor.param.copy()
    if basis is not None:
        nkpts, nso, nbasis = basis.shape
        nao = nso // 2
        basis_Ra, basis_Rb = separate_basis(basis)
        basis_k = lattice.R2k_basis(basis)
        basis_ka, basis_kb = separate_basis(basis_k)
    else:
        nao = lattice.nscsites
        nkpts = lattice.nkpts
        nso = nao * 2

    mu0      = kwargs.get("mu0", None)
    fix_mu   = kwargs.get("fix_mu", False)
    num_grad = kwargs.get("num_grad", False)
    bogo_only = kwargs.get("bogo_only", False)

    # initialize imp_idx and det_idx:
    imp_bath_fit = False
    if imp_fit:
        if imp_idx is None:
            imp_idx = list(range(lattice.nimp))
        det_idx = []
    elif det:
        imp_idx = []
        if det_idx is None:
            det_idx = list(range(lattice.nimp))
    elif imp_idx is None:
        if det_idx is None: # imp + bath fitting
            imp_idx = list(range(nbasis))
            det_idx = []
            imp_bath_fit = True
        else:
            imp_idx = []
    elif det_idx is None:
        det_idx = []
    imp_idx = list(imp_idx)
    det_idx = list(det_idx)

    def convert_idx(idx0, norb):
        idx0_a, idx0_b = idx_ao2so(idx0, norb)
        return idx0_a + idx0_b

    if not imp_bath_fit:
        # ZHC NOTE here should use nao, not nimp
        imp_idx = convert_idx(imp_idx, nao)
        det_idx = convert_idx(det_idx, nao)

    log.info("impurity fitting? %s", imp_fit)
    log.debug(1, "imp_idx: %s", format_idx(imp_idx))
    log.info("det (diagonal fitting)? %s", det)
    log.debug(1, "det_idx: %s", format_idx(det_idx))
    log.debug(1, "only fit bogoliubov part: %s", bogo_only)

    fit_idx = imp_idx + det_idx
    nimp, nidx = len(imp_idx), len(fit_idx)
    nimp_half = nimp // 2
    ndet_half = len(det_idx) // 2
    nidx_half = nidx // 2

    imp_mesh = np.ix_(imp_idx, imp_idx)
    det_mesh = (det_idx, det_idx)
    imp_fill_mesh = (slice(nimp), slice(nimp))
    det_fill_mesh = (range(nimp, nidx), range(nimp, nidx))

    imp_fill_mesh_aa = (slice(nimp_half), slice(nimp_half))
    imp_fill_mesh_bb = (slice(nimp_half, nimp), slice(nimp_half, nimp))
    det_fill_mesh_aa = (range(nimp, nimp + ndet_half), range(nimp, nimp + ndet_half))
    det_fill_mesh_bb = (range(nimp + ndet_half, nidx), range(nimp + ndet_half, nidx))
    if len(np.unique(fit_idx)) != nidx:
        log.warn("fit_idx has repeated indices: %s", format_idx(fit_idx))

    # ZHC NOTE
    # rho should be rho_glob (R = 0) with shape (nso, nso)
    if rho.shape[-1] != nso:
        log.warn("FitVcorFull: target rho should has shape (%s, %s) , "
                 "now has shape %s ...", nso, nso, str(rho.shape))
    rho_target = np.zeros((nidx, nidx))
    rho_target[imp_fill_mesh] = rho[imp_mesh]
    rho_target[det_fill_mesh] = rho[det_mesh]

    # ZHC NOTE
    # choose to fit only bogoliubov part
    if bogo_only:
        rho_target[imp_fill_mesh_aa] = 0.0
        rho_target[imp_fill_mesh_bb] = 0.0
        rho_target[det_fill_mesh_aa] = 0.0
        rho_target[det_fill_mesh_bb] = 0.0

    rho1 = np.zeros_like(rho_target)

    # precompute the GFock
    Fock = lattice.getFock(kspace=True)
    GFock = pbc_hp.combine_H1_k(Fock)
    assert GFock.shape[-1] == lattice.nao * 2

    nelec = nkpts * nso * 0.5
    nelec = mfd.check_nelec(nelec, None)[0]
    nelec_phys_target = nso * filling

    # CVX fractional fit
    if use_cvx_frac:
        from libdmet.routine.fit import cvx_frac
        if bogo_only:
            imp_idx, imp_idx_col = imp_idx[:len(imp_idx)//2], imp_idx[len(imp_idx)//2:]
        else:
            imp_idx_col = imp_idx
        kpts_symm = kwargs.get("kpts_symm", None)
        tol_cvx = kwargs.get("tol_cvx", 1e-6)
        maxiter_cvx = kwargs.get("maxiter_cvx", 10000)
        rho = rho[:nso, :nso]

        rdm1_fit = cvx_frac(rho, GFock, nelec=nao, imp_idx=imp_idx, imp_idx_col=imp_idx_col,
                            tol=tol_cvx, maxiter=maxiter_cvx, nelec_phys=nelec_phys_target,
                            kpts_symm=kpts_symm)

        tmp_dir = kwargs.get("tmp_dir", None)
        if tmp_dir is not None:
            cvx_fit_file = "%s/cvx_fit.h5"%(tmp_dir)
            log.info("Save CVX fit information to %s"%cvx_fit_file)
            cvx_fit_dict = {
                "rho": rho, "fock": GFock, "nelec": nao,
                "imp_idx": imp_idx, "imp_idx_col": imp_idx_col,
                "kpts_symm": kpts_symm, "tol_cvx": tol_cvx,
                "maxiter_cvx": maxiter_cvx,
                "nelec_phys_target": nelec_phys_target,
                "rdm1_fit": rdm1_fit,
            }

            with h5py.File(cvx_fit_file, "w") as f:
                for k, v in cvx_fit_dict.items():
                    if k == "kpts_symm":
                        if v is not None:
                            log.info("Will use k-point symmetry")
                    else:
                        if v is not None:
                            f[k] = v

        err_begin = err_end = 0.0
        return rdm1_fit, err_begin, err_end

    dV_dparam = get_dV_dparam_full(vcor, lattice, compact=True)

    mu_guess = [mu]
    step_guess = [0.1]

    if kwargs.get("use_mpi", False):
        log.info("fitting with MPI.")
        from libdmet.routine import mfd_mpi
        kpairs, kidx = mfd_mpi.get_kpairs_kidx(lattice.cell, lattice.kpts)

        def get_nelec_phys(mu):
            vcor_mat = vcor.get(0, True)
            ew, ev = mfd_mpi.DiagGHF_symm(lattice.cell, GFock, vcor_mat, mu=mu, kpairs=kpairs, kidx=kidx)
            ewocc = assignocc(ew, nelec, beta, mu0=0.0, fix_mu=fix_mu)[0]
            ev_sq = np.abs(ev)
            ev_sq **= 2
            ev_sq_sum  = ev_sq[:, :nao].sum(axis=1)
            ev_sq_sum -= ev_sq[:, nao:].sum(axis=1)
            nelec_phys = np.dot(ev_sq_sum.ravel(), ewocc.ravel()) / nkpts + nao
            #log.info("nelec_phys %15.8g (target %15.8g)", nelec_phys, nelec_phys_target)
            return nelec_phys

        def errfunc(param):
            vcor.update(param)
            vcor_mat = vcor.get(0, True)
            mu_elec = mono_fit_2(get_nelec_phys, nelec_phys_target, mu_guess[0], thr=1e-6,
                                 dx=step_guess[0], verbose=False, maxiter=20)
            ew, ev = mfd_mpi.DiagGHF_symm(lattice.cell, GFock, vcor_mat, mu=mu_elec, kpairs=kpairs, kidx=kidx)
            ewocc, mu_quasi, nerr = assignocc(ew, nelec, beta, mu0=0.0, fix_mu=fix_mu)

            if imp_bath_fit:
                GRho = np.empty_like(ev)
                for k in range(nkpts):
                    GRho[k] = np.dot(ev[k]*ewocc[k], ev[k].conj().T)
                rho1[:] = foldRho_k(GRho, basis_k)
            else:
                GRhoT = 0.0
                for k in range(nkpts):
                    GRhoT += np.dot(ev[k]*ewocc[k], ev[k].conj().T)
                GRhoT /= nkpts
                if max_abs(GRhoT.imag) > mfd.IMAG_DISCARD_TOL:
                    log.warn("GRhoT has imag part %s", max_abs(GRhoT.imag))
                GRhoT = GRhoT.real

                rho1[imp_fill_mesh] = GRhoT[imp_mesh]
                rho1[det_fill_mesh] = GRhoT[det_mesh]

                if bogo_only:
                    rho1[imp_fill_mesh_aa] = 0.0
                    rho1[imp_fill_mesh_bb] = 0.0
                    rho1[det_fill_mesh_aa] = 0.0
                    rho1[det_fill_mesh_bb] = 0.0

            return la.norm((rho1 - rho_target)) / np.sqrt(2)

        def gradfunc_ft(param):
            vcor.update(param)
            vcor_mat = vcor.get(0, True)
            mu_elec = mono_fit_2(get_nelec_phys, nelec_phys_target, mu_guess[0], thr=1e-6,
                                 dx=step_guess[0], verbose=False, maxiter=20)

            # ZHC NOTE update mu_guess and step_guess
            step_guess[0] = min(max(0.05, abs(mu_elec - mu_guess[0])), 0.2)
            mu_guess[0] = mu_elec

            ew, ev = mfd_mpi.DiagGHF_symm(lattice.cell, GFock, vcor_mat, mu=mu_elec, kpairs=kpairs, kidx=kidx)
            ewocc, mu_quasi, nerr = assignocc(ew, nelec, beta, mu0=0.0, fix_mu=fix_mu)

            if imp_bath_fit:
                GRho = np.empty_like(ev)
                for k in range(nkpts):
                    GRho[k] = np.dot(ev[k]*ewocc[k], ev[k].conj().T)
                rho1[:] = foldRho_k(GRho, basis_k)
            else:
                GRhoT = 0.0
                for k in range(nkpts):
                    GRhoT += np.dot(ev[k]*ewocc[k], ev[k].conj().T)
                GRhoT /= nkpts
                GRhoT = GRhoT.real

                rho1[imp_fill_mesh] = GRhoT[imp_mesh]
                rho1[det_fill_mesh] = GRhoT[det_mesh]

                if bogo_only:
                    rho1[imp_fill_mesh_aa] = 0.0
                    rho1[imp_fill_mesh_bb] = 0.0
                    rho1[det_fill_mesh_aa] = 0.0
                    rho1[det_fill_mesh_bb] = 0.0

            if imp_bath_fit:
                # ZHC TODO implement the gradient for imp+bath fitting
                raise NotImplementedError

            drho = rho1 - rho_target
            val = la.norm(drho)

            #dw_dparam = 0.0
            #for k in range(nkpts):
            #    dw_dv = ftsystem.get_dw_dv(ew[k], ev[k], drho, mu_quasi, beta, fix_mu=fix_mu,
            #                               fit_idx=fit_idx,
            #                               compact=(dV_dparam.ndim == 2))
            #    dw_dparam += dV_dparam.reshape(dV_dparam.shape[0], -1).dot(dw_dv.ravel())
            #
            #dw_dparam = dw_dparam.real / (2.0 * val * np.sqrt(2.0) * nkpts)
            dw_dparam = mfd_mpi.get_dw_dparam(lattice.cell, ew, ev, drho, dV_dparam, val,
                                              mu_quasi, beta, fix_mu=fix_mu, fit_idx=fit_idx,
                                              kpairs=kpairs, kidx=kidx)
            return dw_dparam
    else:
        def get_nelec_phys(mu):
            ew, ev = mfd.DiagGHF_symm(GFock, vcor, mu=mu, lattice=lattice)
            ewocc = assignocc(ew, nelec, beta, mu0=0.0, fix_mu=fix_mu)[0]
            ev_sq = np.abs(ev)
            ev_sq **= 2
            ev_sq_sum  = ev_sq[:, :nao].sum(axis=1)
            ev_sq_sum -= ev_sq[:, nao:].sum(axis=1)
            nelec_phys = np.dot(ev_sq_sum.ravel(), ewocc.ravel()) / nkpts + nao
            #log.info("nelec_phys %15.8g (target %15.8g)", nelec_phys, nelec_phys_target)
            return nelec_phys

        def errfunc(param):
            vcor.update(param)
            mu_elec = mono_fit_2(get_nelec_phys, nelec_phys_target, mu_guess[0], thr=1e-6,
                                 dx=step_guess[0], verbose=False, maxiter=20)
            ew, ev = mfd.DiagGHF_symm(GFock, vcor, mu=mu_elec, lattice=lattice)
            ewocc, mu_quasi, nerr = assignocc(ew, nelec, beta, mu0=0.0, fix_mu=fix_mu)

            if imp_bath_fit:
                GRho = np.empty_like(ev)
                for k in range(nkpts):
                    GRho[k] = np.dot(ev[k]*ewocc[k], ev[k].conj().T)
                rho1[:] = foldRho_k(GRho, basis_k)
            else:
                GRhoT = 0.0
                for k in range(nkpts):
                    GRhoT += np.dot(ev[k]*ewocc[k], ev[k].conj().T)
                GRhoT /= nkpts
                if max_abs(GRhoT.imag) > mfd.IMAG_DISCARD_TOL:
                    log.warn("GRhoT has imag part %s", max_abs(GRhoT.imag))
                GRhoT = GRhoT.real

                rho1[imp_fill_mesh] = GRhoT[imp_mesh]
                rho1[det_fill_mesh] = GRhoT[det_mesh]

                if bogo_only:
                    rho1[imp_fill_mesh_aa] = 0.0
                    rho1[imp_fill_mesh_bb] = 0.0
                    rho1[det_fill_mesh_aa] = 0.0
                    rho1[det_fill_mesh_bb] = 0.0

            return la.norm((rho1 - rho_target)) / np.sqrt(2)

        def gradfunc_ft(param):
            vcor.update(param)
            mu_elec = mono_fit_2(get_nelec_phys, nelec_phys_target, mu_guess[0], thr=1e-6,
                                 dx=step_guess[0], verbose=False, maxiter=20)

            # ZHC NOTE update mu_guess and step_guess
            step_guess[0] = min(max(0.05, abs(mu_elec - mu_guess[0])), 0.2)
            mu_guess[0] = mu_elec

            ew, ev = mfd.DiagGHF_symm(GFock, vcor, mu=mu_elec, lattice=lattice)
            ewocc, mu_quasi, nerr = assignocc(ew, nelec, beta, mu0=0.0, fix_mu=fix_mu)

            if imp_bath_fit:
                GRho = np.empty_like(ev)
                for k in range(nkpts):
                    GRho[k] = np.dot(ev[k]*ewocc[k], ev[k].conj().T)
                rho1[:] = foldRho_k(GRho, basis_k)
            else:
                GRhoT = 0.0
                for k in range(nkpts):
                    GRhoT += np.dot(ev[k]*ewocc[k], ev[k].conj().T)
                GRhoT /= nkpts
                GRhoT = GRhoT.real

                rho1[imp_fill_mesh] = GRhoT[imp_mesh]
                rho1[det_fill_mesh] = GRhoT[det_mesh]

                if bogo_only:
                    rho1[imp_fill_mesh_aa] = 0.0
                    rho1[imp_fill_mesh_bb] = 0.0
                    rho1[det_fill_mesh_aa] = 0.0
                    rho1[det_fill_mesh_bb] = 0.0

            if imp_bath_fit:
                # ZHC TODO implement the gradient for imp+bath fitting
                raise NotImplementedError

            drho = rho1 - rho_target
            val = la.norm(drho)

            dw_dparam = 0.0
            for k in range(nkpts):
                dw_dv = ftsystem.get_dw_dv(ew[k], ev[k], drho, mu_quasi, beta, fix_mu=fix_mu,
                                           fit_idx=fit_idx,
                                           compact=(dV_dparam.ndim == 2))
                dw_dparam += dV_dparam.reshape(dV_dparam.shape[0], -1).dot(dw_dv.ravel())

            dw_dparam = dw_dparam.real / (2.0 * val * np.sqrt(2.0) * nkpts)
            return dw_dparam

    if beta == np.inf:
        log.info("Using analytic gradient for 0 T")
        if not num_grad:
            raise NotImplementedError
    else:
        log.info("Using analytic gradient for finite T, beta = %s", beta)
        gradfunc = gradfunc_ft

    if kwargs.get("test_grad", False):
        """
        This block is for testing gradient.
        """
        param_rand = kwargs.get("param_rand", None)
        if param_rand is None:
            np.random.seed(10086)
            param_rand = np.random.random(vcor.param.shape)
            param_rand = (param_rand - 0.5) * 0.1
        test_grad(param_rand.copy(), errfunc, gradfunc, dx=1e-4)
        test_grad(param_rand.copy(), errfunc, gradfunc, dx=1e-5)

    if num_grad:
        log.warn("You are using numerical gradient...")
        gradfunc = None

    err_begin = errfunc(param_begin)
    param, err_end, pattern, gnorm_res = minimize(errfunc, param_begin.copy(),
                                                  MaxIter, gradfunc, **kwargs)
    vcor.update(param)

    log.info("Minimizer converge pattern: %d ", pattern)
    log.info("Current function value: %15.8f", err_end)
    log.info("Norm of gradients: %s", gnorm_res)
    log.info("Norm diff of x: %15.8f", max_abs(param - param_begin))

    if log.Level[log.verbose] > log.Level["DEBUG0"]:
        errfunc(vcor.param)
        log.debug(1, "rdm1 target:\n%s", rho_target)
        log.debug(1, "rdm1 fitted:\n%s", rho1)

    return vcor, err_begin, err_end

def FitVcorTwoStep(GRho, lattice, basis, vcor, mu, beta=np.inf, MaxIter1=300,
                   MaxIter2=0, kinetic=False, CG_check=False, BFGS=False,
                   serial=True, method='CG', ytol=1e-7, gtol=1e-3,
                   filling=None, **kwargs):
    """
    Main wrapper for correlation potential fitting.
    """

    full_return  = kwargs.get("full_return", False)
    use_cvx_frac = kwargs.get("use_cvx_frac", False)

    rdm1_fit = None
    vcor_new = None
    res_dict = {}

    vcor_new = copy.deepcopy(vcor)
    log.result("Using two-step vcor fitting")
    err_begin = None

    if MaxIter1 > 0:
        log.info("Impurity model stage max %d steps", MaxIter1)
        log.info("Finite temperature used in fitting? beta = %s ", beta)
        vcor_new, err_begin1, err_end1 = FitVcorEmb(GRho, lattice, basis, vcor_new,
                                                    mu, beta=beta, MaxIter=MaxIter1,
                                                    CG_check=CG_check, serial=serial,
                                                    BFGS=BFGS, method=method,
                                                    ytol=ytol, gtol=gtol, **kwargs)
        log.info("Embedding Stage:\nbegin %20.12f    end %20.12f" % (err_begin1, err_end1))

        err_begin = err_begin1
        err_end = err_end1

    if MaxIter2 > 0:
        log.info("Full lattice stage  max %d steps", MaxIter2)
        if filling is not None:
            log.info("fit chemical potential while fitting.")

            res = FitVcorFull_mu(GRho, lattice, basis, vcor_new, mu=mu, beta=beta,
                                 filling=filling, MaxIter=MaxIter2, method=method,
                                 ytol=ytol, gtol=gtol, **kwargs)

            if use_cvx_frac:
                rdm1_fit, err_begin2, err_end = res
            else:
                vcor_new, err_begin2, err_end = res
                res_dict["rdm1_fit"] = None

        else:
            vcor_new, err_begin2, err_end = FitVcorFull(GRho, lattice, basis, vcor_new, mu=mu, beta=beta,
                                                        filling=None, MaxIter=MaxIter2, method=method,
                                                        ytol=ytol, gtol=gtol,
                                                        **kwargs)

        if err_begin is None:
            err_begin = err_begin2

    log.result("residue (begin) = %20.12f", err_begin)
    log.result("residue (end)   = %20.12f", err_end)

    if full_return:
        return vcor_new, rdm1_fit, err_end, res_dict
    else:
        if use_cvx_frac:
            return rdm1_fit, err_end
        else:
            return vcor_new, err_end

def localize_mo(mo_coeff, basis):
    neo, nmo = mo_coeff.shape
    ncells, nso, neo = basis.shape
    nao = nso // 2
    alpha_mask = []
    beta_mask  = []
    for R in range(ncells):
        alpha_mask.append(np.arange(nao) + R * nao * 2)
        beta_mask.append(np.arange(nao) + (R * nao * 2 + nao))
    alpha_mask = np.hstack(alpha_mask)
    beta_mask  = np.hstack(beta_mask)

    basis = basis.reshape(-1, neo)
    mo = np.dot(basis, mo_coeff)

    # occ before localization
    mo_o = mo[:, :mo.shape[-1]//2]
    w_o = np.einsum("ai, ai -> i", mo_o[alpha_mask], mo_o[alpha_mask],
                    optimize=True)
    order_o = np.argsort(w_o, kind='mergesort')[::-1]
    w_o = w_o[order_o]
    log.debug(0, "occupied MO character (before localization):\n%s", w_o)
    log.debug(0, "nocc_a: %s (%s)", np.sum(w_o), np.sum(w_o > 0.5))

    # occ localization
    mo_o, u_o = scdm.scdm_model(mo_o, return_C_mo_lo=True)
    mo_o = mo_o[0]
    u_o = u_o[0]
    w_o = np.einsum("ai, ai -> i", mo_o[alpha_mask], mo_o[alpha_mask],
                    optimize=True)
    order_o = np.argsort(w_o, kind='mergesort')[::-1]
    w_o = w_o[order_o]
    log.debug(0, "occupied MO character (after localization):\n%s", w_o)
    log.debug(0, "nocc_a: %s (%s)", np.sum(w_o), np.sum(w_o > 0.5))
    u_o = u_o[:, order_o]

    # vir before localization
    mo_v = mo[:, mo.shape[-1]//2:]
    w_v = np.einsum("ai, ai -> i", mo_v[alpha_mask], mo_v[alpha_mask],
                    optimize=True)
    order_v = np.argsort(w_v, kind='mergesort')[::-1]
    w_v = w_v[order_v]
    log.debug(0, "virtual MO character (before localization):\n%s", w_v)
    log.debug(0, "nvir_a: %s (%s)", np.sum(w_v), np.sum(w_v > 0.5))

    # vir localization
    mo_v, u_v = scdm.scdm_model(mo_v, return_C_mo_lo=True)
    mo_v = mo_v[0]
    u_v = u_v[0]
    w_v = np.einsum("ai, ai -> i", mo_v[alpha_mask], mo_v[alpha_mask],
                    optimize=True)
    order_v = np.argsort(w_v, kind='mergesort')[::-1]
    w_v = w_v[order_v]
    log.debug(0, "virtual MO character (after localization):\n%s", w_v)
    log.debug(0, "nvir_a: %s (%s)", np.sum(w_v), np.sum(w_v > 0.5))
    u_v = u_v[:, order_v]

    u = la.block_diag(u_o, u_v)
    mo = np.dot(mo_coeff, u)
    return mo, u
