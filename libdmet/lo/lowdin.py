#!/usr/bin/env python

"""
Lowdin orbitals and related check routine.

Author:
    Zhi-Hao Cui
"""

import numpy as np
import scipy.linalg as la
from pyscf import lo
from libdmet.utils.misc import mdot, max_abs
from libdmet.utils import logger as log

ORTH_TOL = 1e-12

# ****************************************************************************
# Meta-Lowdin orbitals.
# ****************************************************************************

def lowdin_k(mf_or_lattice, method='meta_lowdin', s=None, pre_orth_ao=lo.orth.REF_BASIS):
    """
    Meta Lowdin orbitals with k point sampling.
    
    Args:
        kmf: kscf or lattice object
        method : str
            One of
            | lowdin : Symmetric orthogonalization
            | meta-lowdin : Lowdin orth within core, valence, virtual space 
                            separately (JCTC, 10, 3784)
            | NAO:
        s: overlap matrix, shape (nkpts, nao, nao)
        pre_orth_ao: if None, use ANO or MINAO as reference basis

    Returns:
        C_ao_lo: shape (nkpts, nao, nlo)
    """
    from pyscf import scf
    from pyscf.pbc.scf import khf

    if isinstance(mf_or_lattice, khf.KSCF):
        mf = mf_or_lattice
    elif isinstance(mf_or_lattice, scf.hf.SCF): # mol
        C_lowdin = lo.orth_ao(mf_or_lattice, method, s=s, \
                pre_orth_ao=pre_orth_ao)
        if isinstance(mf_or_lattice, scf.uhf.UHF): # unrestricted
            C_lowdin = (C_lowdin, C_lowdin)
        return np.asarray(C_lowdin)
    else:
        mf = mf_or_lattice.kmf
    if mf is None:
        from pyscf.pbc.scf import khf
        mf = khf.KRHF(mf_or_lattice.cell, kpts=mf_or_lattice.kpts)
    cell = mf.cell

    if s is None:
        s1e = mf.get_ovlp()
    else:
        s1e = s
    nkpts = len(mf.kpts)
    assert nkpts == len(s1e)
    C_lowdin = [lo.orth_ao(mf, method, s=s1e[k], pre_orth_ao=pre_orth_ao) \
            for k in range(nkpts)]

    from pyscf.scf import uhf
    from pyscf.pbc.scf import uhf as pbcuhf
    from pyscf.pbc.scf import kuhf
    # ZHC FIXME support istype in the future.
    is_uhf = isinstance(mf, uhf.UHF) or isinstance(mf, pbcuhf.UHF) or isinstance(mf, kuhf.KUHF)

    if is_uhf: # unrestricted
        C_lowdin = (C_lowdin, C_lowdin)
    return np.asarray(C_lowdin)

orth_ao = orth_ao_k = lowdin = lowdin_k

# ****************************************************************************
# Lowdin orthogonalization
# ****************************************************************************

def _lowdin(s, tol=1e-14):
    """
    New basis is |mu> c^{lowdin}_{mu i}
    """
    e, v = la.eigh(s)
    idx = e > tol
    if not idx.all():
        log.warn("_vec_lowdin has almost zero eigenvalues:\n%s", e[~idx])
    return np.dot(v[:, idx] / np.sqrt(e[idx]), v[:, idx].conj().T)

def _vec_lowdin(c, s=1, f=None):
    """
    Lowdin orth for the metric c.T*s*c and get x, then c*x
    """
    if f is None:
        res = np.dot(c, _lowdin(mdot(c.conj().T, s, c)))
    else:
        res = np.dot(c * f, _lowdin(mdot(c.conj().T, s, c)))
    return res

def vec_lowdin(C, S, f=None):
    """
    Lowdin orthogonalization for a set of orbitals with (spin and) kpts. 
    f is a factor array to scale C, i.e. C_pm (* f_m) * X_mi.
    """
    S = np.asarray(S)
    if S.ndim == 3:
        C_orth = np.zeros_like(C)
        nkpts = C.shape[-3]
        if C.ndim == 3:
            for k in range(nkpts):
                if f is not None:
                    f = f[k]
                C_orth[k] = _vec_lowdin(C[k], S[k], f)
        else:
            spin = C.shape[0]
            for s in range(spin):
                for k in range(nkpts):
                    if f is not None:
                        f = f[s][k]
                    C_orth[s, k] = _vec_lowdin(C[s, k], S[k], f)
    else:
        if C.ndim == 2:
            C_orth = _vec_lowdin(C, S, f)
        else:
            spin = C.shape[0]
            C_orth = np.zeros_like(C)
            for s in range(spin):
                if f is not None:
                    f = f[s]
                C_orth[s] = _vec_lowdin(C[s], S, f)
    return C_orth

vec_lowdin_k = vec_lowdin

def _cano(s, tol=1e-12):
    e, v = la.eigh(s)
    idx = e > tol
    with np.printoptions(suppress=False):
        log.debug(2, "canonical orthogonalization eigenvals:\n%s", e)
    return v[:, idx] / np.sqrt(e[idx])

def _orth_cano(c, s, tol=1e-12, f=None):
    if s is None:
        if f is None:
            res = np.dot(c, _cano(np.dot(c.conj().T, c), tol=tol))
        else:
            res = np.dot(c * f, _cano(np.dot(c.conj().T, c), tol=tol))
    else:
        if f is None:
            res = np.dot(c, _cano(mdot(c.conj().T, s, c), tol=tol))
        else:
            res = np.dot(c * f, _cano(mdot(c.conj().T, s, c), tol=tol))
    return res
    
def orth_cano(C, S, tol=1e-12, f=None):
    """
    Canonical orthogonalization for a set of orbitals with (spin and) kpts. 
    tol is the tolerance to discard functions.
    """
    S = np.asarray(S)
    if S.ndim == 3:
        C_orth = []
        nkpts = C.shape[-3]
        if C.ndim == 3:
            for k in range(nkpts):
                if f is not None:
                    f = f[k]
                C_orth.append(_orth_cano(C[k], S[k], tol, f))
        else:
            spin = C.shape[0]
            for s in range(spin):
                C_s = []
                for k in range(nkpts):
                    if f is not None:
                        f = f[s][k]
                    C_s.append(_orth_cano(C[s][k], S[k], tol, f))
                C_orth.append(C_s)
    else:
        if C.ndim == 2:
            C_orth = _orth_cano(C, S, tol, f)
        else:
            spin = C.shape[0]
            C_orth = []
            for s in range(spin):
                if f is not None:
                    f = f[s]
                C_orth.append(_orth_cano(C[s], S, tol, f))
    return C_orth
    
# ****************************************************************************
# Check functions
# ****************************************************************************

def check_orthonormal(C, S, tol=ORTH_TOL):
    r"""
    check whether C^{\dagger} S C = I
    """
    C = np.asarray(C)
    S = np.asarray(S)
    nmo = C.shape[-1]
    I = np.eye(nmo)
    flag = True

    if S.ndim == 3:
        nkpts = C.shape[-3]
        if C.ndim == 3:
            for k in range(nkpts):
                diff_identity = max_abs(mdot(C[k].conj().T, S[k], C[k]) - I)
                if diff_identity > tol:
                    log.info("orbital not orthonormal at k: %s, diff_identity: %s", \
                            k, diff_identity)
                    flag = False
        else:
            spin = C.shape[0]
            for s in range(spin):
                for k in range(nkpts):
                    diff_identity = max_abs(mdot(C[s, k].conj().T, S[k], C[s, k]) \
                            - I)
                    if diff_identity > tol:
                        log.info("orbital not orthonormal at s: %s, k: %s, diff_identity: %s", \
                                s, k, diff_identity)
                        flag = False
    else:
        if C.ndim == 2:
            diff_identity = max_abs(mdot(C.conj().T, S, C) - I)
            if diff_identity > tol:
                log.info("orbital not orthonormal, diff_identity: %s", diff_identity)
                flag = False
        else:
            spin = C.shape[0]
            for s in range(spin):
                diff_identity = max_abs(mdot(C[s].conj().T, S, C[s]) - I)
                if diff_identity > tol:
                    log.info("orbital not orthonormal at s: %s, diff_identity: %s", \
                            s, diff_identity)
                    flag = False
    return flag

def check_orthogonal(C1, C2, S, tol=ORTH_TOL):
    r"""
    Check whether C1^{\dagger} S C2 = 0
    """
    C1 = np.asarray(C1)
    C2 = np.asarray(C2)
    S = np.asarray(S)
    nmo1 = C1.shape[-1]
    nmo2 = C2.shape[-1]
    flag = True

    if S.ndim == 3:
        nkpts = C1.shape[-3]
        if C1.ndim == 3:
            for k in range(nkpts):
                ovlp_max = max_abs(mdot(C1[k].conj().T, S[k], C2[k]))
                if ovlp_max > tol:
                    log.info("orbital not orthogonal at k: %s, ovlp_max: %s", \
                            k, ovlp_max)
                    flag = False
        else:
            spin = C1.shape[0]
            for s in range(spin):
                for k in range(nkpts):
                    ovlp_max = max_abs(mdot(C1[s, k].conj().T, S[k], C2[s, k]))
                    if ovlp_max > tol:
                        log.info("orbital not orthogonal at s: %s, k: %s, ovlp_max: %s", \
                                s, k, ovlp_max)
                        flag = False
    else:
        if C1.ndim == 2:
            ovlp_max = max_abs(mdot(C1.conj().T, S, C2))
            if ovlp_max > tol:
                log.info("orbital not orthogonal, ovlp_max: %s", ovlp_max)
                flag = False
        else:
            spin = C1.shape[0]
            for s in range(spin):
                ovlp_max = max_abs(mdot(C1[s].conj().T, S, C2[s]))
                if ovlp_max > tol:
                    log.info("orbital not orthogonal at s: %s, ovlp_max: %s", \
                            s, ovlp_max)
                    flag = False
    return flag

check_orthogonal_two_sets = check_orthogonal

def check_span_same_space(C1, C2, S, tol=ORTH_TOL):
    """
    Check whether C1, C2 span the same space.
    """
    C1 = np.asarray(C1)
    C2 = np.asarray(C2)
    S = np.asarray(S)
    nmo1 = C1.shape[-1]
    nmo2 = C2.shape[-1]
    flag = True
    if S.ndim == 3:
        nkpts = C1.shape[-3]
        if C1.ndim == 3:
            for k in range(nkpts):
                det_max = abs(np.abs(la.det(mdot(C1[k].conj().T, S[k], C2[k]))) \
                        - 1.0)
                if det_max > tol:
                    log.info("Not span the same space at k: %s, det_max: %s", k, det_max)
                    flag = False
        else:
            spin = C1.shape[0]
            for s in range(spin):
                for k in range(nkpts):
                    det_max = abs(np.abs(la.det(mdot(C1[s, k].conj().T, S[k], \
                            C2[s, k]))) - 1.0)
                    if det_max > tol:
                        log.info("Not span the same space at s: %s, k: %s, det_max: %s", \
                                s, k, det_max)
                        flag = False
    else:
        if C1.ndim == 2:
            det_max = abs(np.abs(la.det(mdot(C1.conj().T, S, C2))) - 1.0)
            if det_max > tol:
                log.info("Not span the same space, det_max: %s", det_max)
                flag = False
        else:
            spin = C1.shape[0]
            for s in range(spin):
                det_max = abs(np.abs(la.det(mdot(C1[s].conj().T, S, C2[s]))) - 1.0)
                if det_max > tol:
                    log.info("Not span the same space at s: %s, det_max: %s", \
                            s, det_max)
                    flag = False
    return flag

def check_positive_definite(ovlp, tol=ORTH_TOL):
    """
    Check whether a symmetric matrix is positive definite.
    """
    ovlp = np.asarray(ovlp)
    flag = True
    if ovlp.ndim == 2:
        ew, ev = la.eigh(ovlp)
        if (ew < tol).any():
            log.info("Matrix is not positive definite, eigvals:\n%s", \
                    ew[ew < tol])
            flag = False
    elif ovlp.ndim == 3:
        nkpts, nlo, nlo = ovlp.shape
        for k in range(nkpts):
            ew, ev = la.eigh(ovlp[k])
            if (ew < tol).any():
                log.info("Matrix is not positive definite at k: %s, eigvals:\n%s", \
                        k, ew[ew < tol])
                flag = False
    else:
        spin, nkpts, nlo, nlo = ovlp.shape
        for s in range(spin):
            for k in range(nkpts):
                ew, ev = la.eigh(ovlp[s, k])
                if (ew < tol).any():
                    log.info("Matrix is not positive definite at s: %s k: %s, eigvals:\n%s", \
                            s, k, ew[ew < tol])
                    flag = False
    return flag

def give_labels_to_lo(kmf, C_ao_lo, order=1, C_ao_lo_ref=None, labels_ref=None):
    """
    Get the closest (Lowdin) AO labels to each LO.

    Args:
        kmf: kmf object.
        C_ao_lo: local orbitals.
        order: default is 1, only the largest component labels.
        C_ao_lo_ref: reference orbitals.
        labels_ref: reference orbital labels.

    Returns:
        labels: labels for local orbitals.
    """
    if C_ao_lo_ref is None:
        C_ao_ref = lowdin_k(kmf)
    else:
        C_ao_ref = np.asarray(C_ao_lo_ref)
    if labels_ref is None:
        labels_ref = np.asarray(kmf.cell.ao_labels())
    else:
        labels_ref = np.asarray(labels_ref)

    ovlp = kmf.get_ovlp()
    C_ao_lo = np.asarray(C_ao_lo)
    assert C_ao_lo.ndim == 3 
    assert C_ao_ref.ndim == 3
    nkpts, nao, nlo = C_ao_lo.shape
    nref = C_ao_ref.shape[-1]

    C_ref_lo_ave = np.zeros((nref, nlo))
    for k in range(nkpts): 
        C_ref_lo_ave += abs(mdot(C_ao_ref[k].conj().T, ovlp[k], C_ao_lo[k]))
    C_ref_lo_ave /= nkpts  
    idx = np.argsort(C_ref_lo_ave, kind='mergesort', axis=0)[-order:][::-1]

    for j in range(idx.shape[-1]):
        print ("%5s" % j, end='   ')
        for i in range(idx.shape[-2]):
            print ("%-20s  [%.2f]"%(labels_ref[idx[i, j]], \
                    C_ref_lo_ave[idx[i, j], j]), end=' ')
        print ()

    return labels_ref[idx]

get_ao_labels = assign_labels_to_lo = give_labels_to_lo
