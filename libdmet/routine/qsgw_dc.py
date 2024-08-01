#!/usr/bin/env python

"""
QSGW functions for DMET.

Authors:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

import numpy as np
import scipy.linalg as la

from libdmet.routine import mfd
from libdmet.routine import pbc_helper as pbc_hp
from libdmet.utils.misc import max_abs, mdot
from libdmet.utils import logger as log

def get_vsig_emb(fock, eri, nelec, beta=np.inf, ef=None, ovlp=None,
                 mode='b', chol_tol=1e-6, rdm1_emb=None, max_memory=None):
    from pyscf.pbc import gto, scf, df
    from libdmet.utils import cholesky
    from fcdmft.gw.pbc import krqsgw, kuqsgw
    log.debug(1, "get_vsig_emb: init")
    assert fock[0].ndim == 2
    spin = len(fock)
    norb = fock[0].shape[-1]
    if ovlp is None:
        ovlp = np.eye(norb)[None]
    if max_memory is None:
        max_memory = 1800000

    cell = gto.Cell(a=np.eye(3))
    cell.nao_nr = lambda *args: norb
    cell.nelectron = nelec
    cell.max_memory = max_memory
    if log.Level[log.verbose] > log.Level["DEBUG0"]:
        cell.verbose = 5
    cell.incore_anyway = True
    cell.build(dump_input=False)
    kpts = kpts=np.zeros((1, 3))
    
    if spin == 1:
        fock = fock[0]
        eri = eri[0]
        
        # first diagonalize fock
        ew, ev = la.eigh(fock, ovlp[0])
        ewocc, mu, _ = mfd.assignocc(ew, nelec * 0.5, beta)
        ewocc *= 2.0

        # fack kmf
        kmf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
        kmf.with_df._cderi = None
        kmf = pbc_hp.smearing_(kmf, sigma=1.0/beta)
        kmf.mo_coeff  = ev[None]
        kmf.mo_energy = ew[None]
        kmf.mo_occ    = ewocc[None]
        kmf.get_ovlp = lambda *args: ovlp

        log.debug(1, "get_vsig_emb: prepare Lpq")
        gw = krqsgw.KRQSGW(kmf)
        Lpq = cholesky.get_cderi_rhf(eri, norb, tol=chol_tol)[None]
    elif spin == 2:
        # first diagonalize fock
        if ovlp.ndim == 3:
            ovlp = [ovlp[0], ovlp[0]]
        else:
            ovlp = ovlp[0]
        
        if rdm1_emb is None:
            ew_a, ev_a = la.eigh(fock[0], ovlp[0])
            ew_b, ev_b = la.eigh(fock[1], ovlp[1])
            ew = np.asarray((ew_a, ew_b))
            ev = np.asarray((ev_a, ev_b))
            ewocc, mu, _ = mfd.assignocc(ew, nelec, beta)
        else:
            log.debug(1, "gw_dc: rdm1")
            assert ef is not None
            ewocc_a, ev_a = la.eigh(rdm1_emb[0])
            ewocc_a = ewocc_a[::-1]
            ev_a = ev_a[:, ::-1]
            ewocc_b, ev_b = la.eigh(rdm1_emb[1])
            ewocc_b = ewocc_b[::-1]
            ev_b = ev_b[:, ::-1]
            ew_a = mdot(ev_a.T, fock[0], ev_a).diagonal()
            ew_b = mdot(ev_b.T, fock[1], ev_b).diagonal()
            ew = np.asarray((ew_a, ew_b))
            ev = np.asarray((ev_a, ev_b))
            ewocc = np.asarray((ewocc_a, ewocc_b))

        # fack kmf
        kmf = scf.KUHF(cell, kpts=kpts, exxdiv=None)
        kmf.with_df._cderi = None
        kmf = pbc_hp.smearing_(kmf, sigma=1.0/beta)
        kmf.mo_coeff  = ev[:, None]
        kmf.mo_energy = ew[:, None]
        kmf.mo_occ    = ewocc[:, None]
        kmf.get_ovlp = lambda *args: ovlp

        log.debug(1, "get_vsig_emb: prepare Lpq")
        gw = kuqsgw.KUQSGW(kmf)
        Lpq = cholesky.get_cderi_uhf(eri, norb, tol=chol_tol)[None]
    else:
        raise ValueError
    
    log.debug(1, "get_vsig_emb: qsgw kernel (naux = %d)", Lpq.shape[-3])
    gw.ac = 'pade'
    gw.mode = mode
    gw.eta = 1.0 / beta
    if ef is None:
        ef = mu
    gw.ef = ef
    gw.fc = False
    gw.max_cycle = -1
    gw.Lpq = Lpq
    gw.mpi_freq = True
    gw.kernel()
    if spin == 1:
        vsig = gw.vsig[0]
    else:
        vsig = gw.vsig[:, 0]
    
    vsig_imag = max_abs(vsig.imag)
    if vsig_imag > mfd.IMAG_DISCARD_TOL:
        log.warn("vsig has non-zero imaginary part %15.8g", vsig_imag)
    vsig = vsig.real
    return vsig

def get_vsig_emb_2(kmf, C_mo_eo, eri, nelec, beta=np.inf, ef=None,
                   mode='b', chol_tol=1e-6, max_memory=None,
                   nw=100, nt=2000):
    from pyscf import gto, scf
    from libdmet.utils import cholesky
    from fcdmft.gw.mol import qsgw_dc, uqsgw_dc
    if max_memory is None:
        max_memory = 1800000
    mol = gto.M()
    mol.verbose = 5
    mol.max_memory = max_memory
    if log.Level[log.verbose] > log.Level["DEBUG0"]:
        mol.verbose = 5
    mol.incore_anyway = True
    
    spin, nkpts, nmo, neo = C_mo_eo.shape
    assert ef is not None
    
    if spin == 1:
        raise NotImplementedError
        fock = fock[0]
        eri = eri[0]
        log.debug(1, "get_vsig_emb: prepare Lpq")
        Lpq = cholesky.get_cderi_rhf(eri, norb, tol=chol_tol)[None]
    elif spin == 2:
        # ZHC NOTE nelec here is the lattice unit cell nelec
        nocca, noccb = nelec
        mf = scf.UHF(mol)
        gw = uqsgw_dc.UGWGF(mf)
        gw.nmo = (neo, neo)
        gw.nocc = (nocca, noccb)
        gw.eta = 1.0 / beta
        gw.ac = 'pade'
        gw.ef = ef
        gw.fullsigma = True
        omega = np.array([ef])
        
        log.debug(1, "get_vsig_emb_2: prepare Lpq")
        Lpq = cholesky.get_cderi_uhf(eri, norb=neo, tol=chol_tol)
        
        # first get sigmaI -> use QSGW AC
        log.debug(1, "get_vsig_emb_2: kernel")
        vsig = gw.kernel(Lpq=Lpq, omega=omega, kmf=kmf, C_mo_lo=C_mo_eo, nw=nw, nt=nt)
    else:
        raise ValueError

    vsig_imag = max_abs(vsig.imag)
    if vsig_imag > mfd.IMAG_DISCARD_TOL:
        log.warn("vsig has non-zero imaginary part %15.8g", vsig_imag)
    vsig = vsig.real
    
    return vsig 
 
