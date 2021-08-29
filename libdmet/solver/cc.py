#! /usr/bin/env python

"""
CC impurity solver.

Author:
    Zhi-Hao Cui
"""

import os
import numpy as np
import scipy.linalg as la
import h5py
import time
from functools import partial

from libdmet.utils import logger as log
from libdmet.solver import scf
from libdmet.solver.scf import (ao2mo_Ham, restore_Ham, _get_veff)
from libdmet.solver.ccd import CCD
from libdmet.solver.uccd import UCCD
from libdmet.basis_transform.make_basis import \
        transform_rdm1_to_ao_mol, transform_rdm2_to_ao_mol, rotate_emb_basis, \
        find_closest_mo, trans_mo, get_mo_ovlp
from libdmet.utils.misc import mdot, max_abs, take_eri

from pyscf.scf import hf
from pyscf import cc
from pyscf import lib
from pyscf import ao2mo
from pyscf.cc.uccsd import _ChemistsERIs

einsum = partial(np.einsum, optimize=True)

def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    cput0 = (time.process_time(), time.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]

    if callable(ao2mofn):
        eri_aa = ao2mofn(moa).reshape([nmoa]*4)
        eri_bb = ao2mofn(mob).reshape([nmob]*4)
        eri_ab = ao2mofn((moa,moa,mob,mob))
    else:
        if len(mycc._scf._eri) == 1:
            eri_aa = ao2mo.restore(1, ao2mo.full(mycc._scf._eri[0], moa), nmoa)
            eri_bb = ao2mo.restore(1, ao2mo.full(mycc._scf._eri[0], mob), nmob)
            eri_ab = ao2mo.general(mycc._scf._eri[0], (moa, moa, mob, mob), compact=False)
        elif len(mycc._scf._eri) == 3:
            eri_aa = ao2mo.restore(1, ao2mo.full(mycc._scf._eri[0], moa), nmoa)
            eri_bb = ao2mo.restore(1, ao2mo.full(mycc._scf._eri[1], mob), nmob)
            eri_ab = ao2mo.general(mycc._scf._eri[2], (moa, moa, mob, mob), compact=False)
        else:
            raise ValueError

    eri_ba = eri_ab.reshape(nmoa,nmoa,nmob,nmob).transpose(2,3,0,1)

    eri_aa = eri_aa.reshape(nmoa,nmoa,nmoa,nmoa)
    eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)
    eri_ba = eri_ba.reshape(nmob,nmob,nmoa,nmoa)
    eri_bb = eri_bb.reshape(nmob,nmob,nmob,nmob)
    eris.oooo = eri_aa[:nocca,:nocca,:nocca,:nocca].copy()
    eris.ovoo = eri_aa[:nocca,nocca:,:nocca,:nocca].copy()
    eris.ovov = eri_aa[:nocca,nocca:,:nocca,nocca:].copy()
    eris.oovv = eri_aa[:nocca,:nocca,nocca:,nocca:].copy()
    eris.ovvo = eri_aa[:nocca,nocca:,nocca:,:nocca].copy()
    eris.ovvv = eri_aa[:nocca,nocca:,nocca:,nocca:].copy()
    eris.vvvv = eri_aa[nocca:,nocca:,nocca:,nocca:].copy()

    eris.OOOO = eri_bb[:noccb,:noccb,:noccb,:noccb].copy()
    eris.OVOO = eri_bb[:noccb,noccb:,:noccb,:noccb].copy()
    eris.OVOV = eri_bb[:noccb,noccb:,:noccb,noccb:].copy()
    eris.OOVV = eri_bb[:noccb,:noccb,noccb:,noccb:].copy()
    eris.OVVO = eri_bb[:noccb,noccb:,noccb:,:noccb].copy()
    eris.OVVV = eri_bb[:noccb,noccb:,noccb:,noccb:].copy()
    eris.VVVV = eri_bb[noccb:,noccb:,noccb:,noccb:].copy()

    eris.ooOO = eri_ab[:nocca,:nocca,:noccb,:noccb].copy()
    eris.ovOO = eri_ab[:nocca,nocca:,:noccb,:noccb].copy()
    eris.ovOV = eri_ab[:nocca,nocca:,:noccb,noccb:].copy()
    eris.ooVV = eri_ab[:nocca,:nocca,noccb:,noccb:].copy()
    eris.ovVO = eri_ab[:nocca,nocca:,noccb:,:noccb].copy()
    eris.ovVV = eri_ab[:nocca,nocca:,noccb:,noccb:].copy()
    eris.vvVV = eri_ab[nocca:,nocca:,noccb:,noccb:].copy()

    #eris.OOoo = eri_ba[:noccb,:noccb,:nocca,:nocca].copy()
    eris.OVoo = eri_ba[:noccb,noccb:,:nocca,:nocca].copy()
    #eris.OVov = eri_ba[:noccb,noccb:,:nocca,nocca:].copy()
    eris.OOvv = eri_ba[:noccb,:noccb,nocca:,nocca:].copy()
    eris.OVvo = eri_ba[:noccb,noccb:,nocca:,:nocca].copy()
    eris.OVvv = eri_ba[:noccb,noccb:,nocca:,nocca:].copy()
    #eris.VVvv = eri_ba[noccb:,noccb:,nocca:,nocca:].copy()

    if not callable(ao2mofn):
        ovvv = eris.ovvv.reshape(nocca*nvira,nvira,nvira)
        eris.ovvv = lib.pack_tril(ovvv).reshape(nocca,nvira,nvira*(nvira+1)//2)
        eris.vvvv = ao2mo.restore(4, eris.vvvv, nvira)

        OVVV = eris.OVVV.reshape(noccb*nvirb,nvirb,nvirb)
        eris.OVVV = lib.pack_tril(OVVV).reshape(noccb,nvirb,nvirb*(nvirb+1)//2)
        eris.VVVV = ao2mo.restore(4, eris.VVVV, nvirb)

        ovVV = eris.ovVV.reshape(nocca*nvira,nvirb,nvirb)
        eris.ovVV = lib.pack_tril(ovVV).reshape(nocca,nvira,nvirb*(nvirb+1)//2)
        vvVV = eris.vvVV.reshape(nvira**2,nvirb**2)
        idxa = np.tril_indices(nvira)
        idxb = np.tril_indices(nvirb)
        eris.vvVV = lib.take_2d(vvVV, idxa[0]*nvira+idxa[1], idxb[0]*nvirb+idxb[1])

        OVvv = eris.OVvv.reshape(noccb*nvirb,nvira,nvira)
        eris.OVvv = lib.pack_tril(OVvv).reshape(noccb,nvirb,nvira*(nvira+1)//2)
    return eris

class UICCSD(cc.uccsd.UCCSD):
    def ao2mo(self, mo_coeff=None):
        nmoa, nmob = self.get_nmo()
        nao = self.mo_coeff[0].shape[0]
        nmo_pair = nmoa * (nmoa+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmoa**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory or self.incore_complete)):
            return _make_eris_incore(self, mo_coeff)

        elif getattr(self._scf, 'with_df', None):
            logger.warn(self, 'UCCSD detected DF being used in the HF object. '
                        'MO integrals are computed based on the DF 3-index tensors.\n'
                        'It\'s recommended to use dfccsd.CCSD for the '
                        'DF-CCSD calculations')
            raise NotImplementedError

        else:
            raise NotImplementedError
            return _make_eris_outcore(self, mo_coeff)

class UICCD(UCCD):
    def ao2mo(self, mo_coeff=None):
        nmoa, nmob = self.get_nmo()
        nao = self.mo_coeff[0].shape[0]
        nmo_pair = nmoa * (nmoa+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmoa**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory or self.incore_complete)):
            return _make_eris_incore(self, mo_coeff)

        elif getattr(self._scf, 'with_df', None):
            logger.warn(self, 'UCCSD detected DF being used in the HF object. '
                        'MO integrals are computed based on the DF 3-index tensors.\n'
                        'It\'s recommended to use dfccsd.CCSD for the '
                        'DF-CCSD calculations')
            raise NotImplementedError

        else:
            raise NotImplementedError
            return _make_eris_outcore(self, mo_coeff)

class CCSD(object):
    def __init__(self, nproc=1, nnode=1, TmpDir="./tmp", SharedDir=None, 
                 restricted=False, Sz=0, bcs=False, ghf=False, tol=1e-9, 
                 tol_normt=1e-6, max_cycle=200, level_shift=0.0, frozen=0, 
                 max_memory=40000, compact_rdm2=False, scf_newton=True):
        """
        CCSD solver.
        """
        self.restricted = restricted
        self.max_cycle = max_cycle
        self.max_memory = max_memory
        self.conv_tol = tol
        self.conv_tol_normt = tol_normt
        self.level_shift = level_shift
        self.frozen = frozen
        self.verbose = 5
        self.bcs = bcs
        self.ghf = ghf
        self.Sz = Sz
        self.scfsolver = scf.SCF(newton_ah=scf_newton)
        
        self.t12 = None
        self.l12 = None
        self.onepdm = None
        self.twopdm = None

        self.optimized = False
    
    def run(self, Ham=None, nelec=None, guess=None, restart=False, \
            dump_tl=False, fcc_name="fcc.h5", calc_rdm2=False, \
            **kwargs):
        """
        Main kernel function of the solver.
        """
        log.info("CC solver: start")
        spin = Ham.H1["cd"].shape[0]
        if spin > 1:
            log.eassert(not self.restricted, "CC solver: spin (%s) > 1 " \
                        "requires unrestricted", spin)
        if nelec is None:
            if self.bcs:
                nelec = Ham.norb * 2
            else:
                raise ValueError("CC solver: nelec cannot be None " \
                                 "for RCC or UCC.")
        nelec_a, nelec_b = (nelec + self.Sz) // 2, (nelec - self.Sz) // 2
        log.eassert(nelec_a >= 0 and nelec_b >=0, "CC solver: " \
                    "nelec_a (%s), nelec_b (%s) should >= 0", nelec_a, nelec_b)
        log.eassert(nelec_a + nelec_b == nelec, "CC solver: " \
                    "nelec_a (%s) + nelec_b (%s) should == nelec (%s)", \
                    nelec_a, nelec_b, nelec)
        
        log.debug(1, "CC solver: mean-field")
        self.scfsolver.set_system(nelec, self.Sz, False, self.restricted, \
                                  max_memory=self.max_memory)
        self.scfsolver.set_integral(Ham)

        dm0 = kwargs.get("dm0", None)
        scf_max_cycle = kwargs.get("scf_max_cycle", 200)
        bcc = kwargs.get("bcc", False)
        bcc_verbose = kwargs.get("bcc_verbose", 2)
        bcc_restart = kwargs.get("bcc_restart", False)
        if bcc and bcc_restart and self.optimized and restart:
            bcc_restart = True
            scf_max_cycle = 1 # need not to do scf
        else:
            bcc_restart = False
        
        if self.ghf:
            raise NotImplementedError
        else:
            E_HF, rhoHF = self.scfsolver.HF(tol=self.conv_tol*0.1, \
                    MaxIter=scf_max_cycle, InitGuess=dm0)
        log.debug(1, "CC solver: mean-field converged: %s", \
                  self.scfsolver.mf.converged)
        
        if "mo_energy_custom" in kwargs:
            self.scfsolver.mf.mo_energy = kwargs["mo_energy_custom"]
        if "mo_occ_custom" in kwargs:
            self.scfsolver.mf.mo_occ = kwargs["mo_occ_custom"]
        if "mo_coeff_custom" in kwargs:
            log.info("Use customized MO as CC reference.")
            self.scfsolver.mf.mo_coeff = kwargs["mo_coeff_custom"]
            self.scfsolver.mf.e_tot = self.scfsolver.mf.energy_tot()
        
        log.debug(2, "CC solver: mean-field rdm1: \n%s", 
                  self.scfsolver.mf.make_rdm1())
        
        if kwargs.get("ccd", False):
            log.info("Using CCD as CC solver.")
            if self.ghf:
                raise NotImplementedError
            elif Ham.restricted:
                self.cisolver = CCD(self.scfsolver.mf)
            else:
                self.cisolver = UICCD(self.scfsolver.mf)
        else:
            if self.ghf:
                raise NotImplementedError
            elif Ham.restricted:
                self.cisolver = cc.CCSD(self.scfsolver.mf)
            else:
                self.cisolver = UICCSD(self.scfsolver.mf)

        self.cisolver.max_cycle = self.max_cycle
        self.cisolver.conv_tol = self.conv_tol
        self.cisolver.conv_tol_normt = self.conv_tol_normt
        self.cisolver.level_shift = self.level_shift
        self.cisolver.set(frozen = self.frozen)
        self.cisolver.verbose = self.verbose
        
        if restart:
            log.eassert("basis" in kwargs, "restart requires basis passed in")
        if restart and self.optimized:
            t1, t2, l1, l2 = self.load_t12_from_h5(fcc_name, kwargs["basis"], \
                    self.scfsolver.mf.mo_coeff, bcc_restart=bcc_restart)
        else:
            if guess is not None:
                if len(guess) == 2:
                    t1, t2 = guess
                    l1, l2 = None, None
                else:
                    t1, t2, l1, l2 = guess
            else:
                t1, t2, l1, l2 = None, None, None, None

        log.debug(1, "CC solver: solve t amplitudes")
        eris = self.cisolver.ao2mo(self.cisolver.mo_coeff)
        E_corr, t1, t2 = self.cisolver.kernel(t1=t1, t2=t2, eris=eris)
        
        if bcc:
            log.info("Using Brueckner CC.")
            self.cisolver = bcc_loop(self.cisolver, utol=self.conv_tol_normt,
                    verbose=bcc_verbose)
            self.scfsolver.mf.mo_coeff = self.cisolver.mo_coeff
            self.scfsolver.mf.e_tot = self.cisolver._scf.e_tot
            t1, t2 = self.cisolver.t1, self.cisolver.t2
            eris = self.cisolver.ao2mo(self.cisolver.mo_coeff)
        
        log.debug(1, "CC solver: solve l amplitudes")
        if kwargs.get("ccsdt", False):
            log.info("CCSD(T) correction")
            if kwargs.get("ccsdt_energy", False):
                e_t = self.cisolver.ccsd_t(eris=eris)
                E_corr += e_t
                log.info("CCSD(T) E(T): %20.12f", e_t)
            else:
                e_t = 0.0

            lambda_drv, rdm1_drv, rdm2_drv = self._get_ccsdt_drv(eris=eris)
            l1, l2 = self.cisolver.l1, self.cisolver.l2 = \
                    lambda_drv(self.cisolver, eris=eris, t1=t1, t2=t2, \
                    max_cycle=self.max_cycle, tol=self.conv_tol_normt, \
                    verbose=self.verbose)[1:]
        else:
            l1, l2 = self.cisolver.solve_lambda(t1=t1, t2=t2, l1=l1, l2=l2, eris=eris)
            rdm1_drv = rdm2_drv = None
            e_t = 0.0
        
        E = self.cisolver.e_tot + e_t

        self.make_rdm1(Ham, drv=rdm1_drv)
        if calc_rdm2:
            self.make_rdm2(Ham, drv=rdm2_drv)

        if dump_tl or restart:
            self.save_t12_to_h5(fcc_name, kwargs["basis"], self.cisolver.mo_coeff)
        
        if not self.cisolver.converged:
            log.warn("CC solver not converged...")
        self.optimized = True
        return self.onepdm, E
    
    def _get_ccsdt_drv(self, eris=None):
        if self.restricted:
            from pyscf.cc.ccsd_t_lambda_slow import kernel as lambda_drv
            from pyscf.cc import ccsd_t_rdm_slow as rdm_mod
        else:
            from pyscf.cc.uccsd_t_lambda import kernel as lambda_drv
            from pyscf.cc import uccsd_t_rdm as rdm_mod
        rdm1_drv = partial(rdm_mod.make_rdm1, mycc=self.cisolver, eris=eris)
        rdm2_drv = partial(rdm_mod.make_rdm2, mycc=self.cisolver, eris=eris)
        return lambda_drv, rdm1_drv, rdm2_drv 
    
    def run_dmet_ham(self, Ham, last_aabb=True, save_dmet_ham=False, \
            dmet_ham_fname='dmet_ham.h5', use_calculated_twopdm=False, \
            **kwargs):
        """
        Run scaled DMET Hamiltonian.
        NOTE: the spin order for H2 is aa, bb, ab, the same as ImpHam.
        """
        if kwargs.get("ccsdt", False) or use_calculated_twopdm:
            return self.run_dmet_ham_slow(Ham, last_aabb=last_aabb, \
                save_dmet_ham=save_dmet_ham, dmet_ham_fname=dmet_ham_fname, \
                use_calculated_twopdm=use_calculated_twopdm, **kwargs)
        
        log.info("CC solver Run DMET Hamiltonian.")
        log.debug(0, "ao2mo for DMET Hamiltonian.")
        Ham = ao2mo_Ham(Ham, self.cisolver.mo_coeff, compact=True, in_place=True)

        if Ham.restricted:
            H1 = Ham.H1["cd"][0]
            H2 = Ham.H2["ccdd"][0]
            H0 = Ham.H0
            E = exp_val_rccsd(self.cisolver, H1, H2, H0=H0)
        else:
            H1 = Ham.H1["cd"]
            H2 = Ham.H2["ccdd"]
            H0 = Ham.H0
            E = exp_val_uccsd(self.cisolver, H1, H2, H0=H0)
                  
        if save_dmet_ham:
            fdmet_ham = h5py.File(dmet_ham_fname, 'w')
            fdmet_ham['H0'] = Ham.H0
            fdmet_ham['H1'] = H1
            fdmet_ham['H2'] = H2
            fdmet_ham['mo_coeff'] = self.cisolver.mo_coeff
            fdmet_ham.close()
        return E
    
    def run_dmet_ham_slow(self, Ham, last_aabb=True, save_dmet_ham=False, \
            dmet_ham_fname='dmet_ham.h5', use_calculated_twopdm=False, \
            **kwargs):
        """
        Run scaled DMET Hamiltonian.
        NOTE: the spin order for H2 is aa, bb, ab, the same as ImpHam.
        """
        log.info("CC solver Run DMET Hamiltonian.")
        log.debug(0, "ao2mo for DMET Hamiltonian.")
        Ham = ao2mo_Ham(Ham, self.cisolver.mo_coeff, \
                compact=True, in_place=True)
        Ham = restore_Ham(Ham, 1, in_place=True)
        if use_calculated_twopdm:
            log.info("Using exisiting twopdm in MO basis...")
            assert self.twopdm_mo is not None
        else:
            if kwargs.get("ccsdt", False):
                eris = self.cisolver.ao2mo(self.cisolver.mo_coeff)
                rdm2_drv = self._get_ccsdt_drv(eris=eris)[-1]
            else:
                rdm2_drv = None
            self.make_rdm2(drv=rdm2_drv)

        if Ham.restricted:
            h1 = Ham.H1["cd"]
            h2 = Ham.H2["ccdd"]
            r1 = self.onepdm_mo
            r2 = self.twopdm_mo
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape
            
            E1 = 2.0 * einsum('ij, ji', h1[0], r1[0])
            E2 = 0.5 * einsum('ijkl, ijkl', h2[0], r2[0])
        else:
            h1 = Ham.H1["cd"]
            h2 = Ham.H2["ccdd"]
            r1 = self.onepdm_mo
            r2 = self.twopdm_mo
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape
            
            # energy
            E1 = einsum('sij, sji', h1, r1)
            E2_aa = 0.5 * einsum('ijkl, ijkl', h2[0], r2[0])
            E2_bb = 0.5 * einsum('ijkl, ijkl', h2[1], r2[1])
            E2_ab =       einsum('ijkl, ijkl', h2[2], r2[2])
            E2 = E2_aa + E2_bb + E2_ab
        
        E = E1 + E2
        E += Ham.H0
        log.debug(0, "run DMET Hamiltonian:\nE0 = %20.12f, E1 = %20.12f, " 
                "E2 = %20.12f, E = %20.12f", Ham.H0, E1, E2, E)
        
        if save_dmet_ham:
            fdmet_ham = h5py.File(dmet_ham_fname, 'w')
            fdmet_ham['H0'] = Ham.H0
            fdmet_ham['H1'] = h1
            fdmet_ham['H2'] = h2
            fdmet_ham['mo_coeff'] = self.cisolver.mo_coeff
            fdmet_ham.close()
        return E
    
    def make_rdm1(self, Ham=None, drv=None):
        log.debug(1, "CC solver: solve rdm1")
        if drv is None:
            drv = self.cisolver.make_rdm1
        onepdm = drv(t1=self.cisolver.t1, t2=self.cisolver.t2, \
                l1=self.cisolver.l1, l2=self.cisolver.l2)

        if self.restricted:
            self.onepdm_mo = np.asarray(onepdm)[np.newaxis] * 0.5
        else:
            self.onepdm_mo = np.asarray(onepdm)

        log.debug(1, "CC solver: rotate rdm1 to AO")
        self.onepdm = transform_rdm1_to_ao_mol(self.onepdm_mo, \
                self.cisolver.mo_coeff)
        return self.onepdm

    def make_rdm2(self, Ham=None, ao_repr=False, drv=None, with_dm1=True):
        log.debug(1, "CC solver: solve rdm2")
        if drv is None:
            drv = self.cisolver.make_rdm2
        if with_dm1:
            twopdm_mo = drv(t1=self.cisolver.t1, t2=self.cisolver.t2, 
                            l1=self.cisolver.l1, l2=self.cisolver.l2)
        else:
            twopdm_mo = drv(t1=self.cisolver.t1, t2=self.cisolver.t2, 
                            l1=self.cisolver.l1, l2=self.cisolver.l2,
                            with_dm1=with_dm1)

        if self.restricted:
            self.twopdm_mo = twopdm_mo[np.newaxis]
        else:
            self.twopdm_mo = np.asarray(twopdm_mo)
            twopdm_mo = None

        # rotate back to the AO basis
        if ao_repr:
            log.debug(1, "CC solver: rotate rdm2 to AO")
            self.twopdm = transform_rdm2_to_ao_mol(self.twopdm_mo, 
                                                   self.cisolver.mo_coeff)
            self.twopdm_mo = None
        else:
            self.twopdm = None
            
        if not self.restricted and not self.ghf:
            if self.twopdm_mo is not None:
                self.twopdm_mo = self.twopdm_mo[[0, 2, 1]]
            if self.twopdm is not None:
                self.twopdm = self.twopdm[[0, 2, 1]]
        return self.twopdm

    def load_t12_from_h5(self, fcc_name, basis_new, mo_coeff_new, bcc_restart=False):
        log.debug(1, "CC solver: read previous t and basis")
        if not os.path.isfile(fcc_name):
            log.info("CC solver: read previous t and basis failed, "
                     "file %s does not exist.", fcc_name)
            return None, None, None, None
        fcc = h5py.File(fcc_name, 'r')
        basis_old    = np.asarray(fcc['basis'])
        mo_coeff_old = np.asarray(fcc['mo_coeff'])
        if mo_coeff_old.ndim == 2:
            t1_old = np.asarray(fcc['t1'])
            t2_old = np.asarray(fcc['t2'])
            l1_old = np.asarray(fcc['l1'])
            l2_old = np.asarray(fcc['l2'])
        else:
            spin = mo_coeff_old.shape[0]
            t1_old = [np.asarray(fcc['t1_%s'%s]) for s in range(spin)]
            t2_old = [np.asarray(fcc['t2_%s'%s]) for s in range(spin*(spin+1)//2)]
            l1_old = [np.asarray(fcc['l1_%s'%s]) for s in range(spin)]
            l2_old = [np.asarray(fcc['l2_%s'%s]) for s in range(spin*(spin+1)//2)]
        fcc.close()
        
        mo_coeff_new = np.asarray(mo_coeff_new)
        if mo_coeff_new.shape != mo_coeff_old.shape:
            log.warn("CC solver: mo_coeff shape changed (%s -> %s).", 
                     mo_coeff_old.shape, mo_coeff_new.shape)
            return None, None, None, None
        nao, nmo = mo_coeff_new.shape[-2:]
        # frozen
        mo_idx = self.cisolver.get_frozen_mask()
        is_same_basis = (basis_new is basis_old) or \
                        (max_abs(basis_new - basis_old) < 1e-12)
        
        try:
            if mo_coeff_new.ndim == 2: # RHF and GHF 
                if bcc_restart:
                    if is_same_basis:
                        mo_coeff_new = mo_coeff_old
                    else:
                        basis_old = basis_old.reshape(-1, nmo)
                        basis_new = basis_new.reshape(-1, nmo)
                        umat = find_closest_mo(basis_old, basis_new, return_rotmat=True)[1]
                        mo_coeff_new = np.dot(umat.conj().T, mo_coeff_old)
                
                if is_same_basis:
                    log.debug(2, "restart with the same basis.")
                else:
                    log.debug(2, "restart with the different basis.")
                basis_cas_old = basis_old.reshape(-1, nmo).dot(mo_coeff_old[:, mo_idx])
                basis_cas_new = basis_new.reshape(-1, nmo).dot(mo_coeff_new[:, mo_idx])
                umat = find_closest_mo(basis_cas_old, basis_cas_new, return_rotmat=True)[1]
            else: # UHF
                if bcc_restart:
                    if is_same_basis:
                        mo_coeff_new = mo_coeff_old
                    else:
                        basis_old = basis_old.reshape(spin, -1, nmo)
                        basis_new = basis_new.reshape(spin, -1, nmo)
                        umat = find_closest_mo(basis_old, basis_new, return_rotmat=True)[1]
                        mo_coeff_new = trans_mo(umat.conj().transpose(0, 2, 1), mo_coeff_old)
                
                if is_same_basis:
                    log.debug(2, "restart with the same basis.")
                else:
                    log.debug(2, "restart with the different basis.")
                umat = []
                for s in range(2):
                    basis_cas_old = np.dot(basis_old[s].reshape(-1, nmo), \
                                           mo_coeff_old[s][:, mo_idx[s]])
                    basis_cas_new = np.dot(basis_new[s].reshape(-1, nmo), \
                                           mo_coeff_new[s][:, mo_idx[s]])
                    umat.append(find_closest_mo(basis_cas_old, basis_cas_new, \
                                                return_rotmat=True)[1])
            
            t1 = transform_t1_to_bo(t1_old, umat)
            t2 = transform_t2_to_bo(t2_old, umat)
            l1 = transform_l1_to_bo(l1_old, umat)
            l2 = transform_l2_to_bo(l2_old, umat)
        except np.linalg.LinAlgError:
            log.warn("SVD error catched during matching basis...")
            t1 = None
            t2 = None
            l1 = None
            l2 = None
        return t1, t2, l1, l2
    
    def save_t12_to_h5(self, fcc_name, basis_new, mo_coeff_new):
        log.debug(1, "CC solver: dump t and l")
        mo_coeff_new = np.asarray(mo_coeff_new)
        fcc = h5py.File(fcc_name, 'w')
        fcc['mo_coeff'] = mo_coeff_new
        fcc['basis'] = np.asarray(basis_new)
        if mo_coeff_new.ndim == 2: 
            fcc['t1'] = np.asarray(self.cisolver.t1)
            fcc['t2'] = np.asarray(self.cisolver.t2)
            fcc['l1'] = np.asarray(self.cisolver.l1)
            fcc['l2'] = np.asarray(self.cisolver.l2)
        else:
            spin = mo_coeff_new.shape[0]
            for s in range(spin):
                fcc['t1_%s'%s] = np.asarray(self.cisolver.t1[s])
                fcc['l1_%s'%s] = np.asarray(self.cisolver.l1[s])
            for s in range(spin*(spin+1)//2):
                fcc['t2_%s'%s] = np.asarray(self.cisolver.t2[s])
                fcc['l2_%s'%s] = np.asarray(self.cisolver.l2[s])
        fcc.close()

    def save_rdm_mo(self, rdm_fname='rdm_mo_cc.h5'):
        frdm = h5py.File(rdm_fname, 'w')
        frdm['rdm1'] = np.asarray(self.onepdm_mo)
        frdm['rdm2'] = np.asarray(self.twopdm_mo)
        frdm["mo_coeff"] = np.asarray(self.cisolver.mo_coeff)
        frdm.close()
    
    def load_rdm_mo(self, rdm_fname='rdm_mo_cc.h5'):
        frdm = h5py.File(rdm_fname, 'r')
        rdm1 = np.asarray(frdm["rdm1"])
        rdm2 = np.asarray(frdm["rdm2"])
        mo_coeff = np.asarray(frdm["mo_coeff"])
        frdm.close()
        return rdm1, rdm2, mo_coeff 
    
    def load_dmet_ham(self, dmet_ham_fname='dmet_ham.h5'):
        fdmet_ham = h5py.File(dmet_ham_fname, 'r')
        H1 = np.asarray(fdmet_ham["H1"])
        H2 = np.asarray(fdmet_ham["H2"])
        fdmet_ham.close()
        return H1, H2 
    
    def onepdm(self):
        log.debug(1, "Compute 1pdm")
        return self.onepdm

    def twopdm(self):
        log.debug(1, "Compute 2pdm")
        return self.twopdm

    def cleanup(self):
        pass

# ****************************************************************************
# Expectation value from CCSD rdm, outcore version.
# ****************************************************************************

def exp_val_rccsd(mycc, H1, H2, H0=0.0, rdm2_tmp_fname=None, blksize=None):
    """
    Expectation value of H0, H1 and H2, using an outcore routine.
    H1 and H2 are in MO basis. 
    H2 is real and has 8-fold symmetry, 1, 4, 8-fold array are ok.

    Args:
        mycc: cc object.
        H1: (nmo, nmo)
        H2: (nmo,) * 4, or (nmo_pair, nmo_pair) or (nmo_pair_pair)
        H0: scalar
        rdm2_tmp_fname: if given, will save the rdm2 intermidiates.

    Returns:
        E_tot: the expectation value of H0, H1 and H2.
    """
    mo_idx = mycc.get_frozen_mask()
    if not all(mo_idx):
        nocc = np.count_nonzero(mycc.mo_occ > 0)
        core_idx = np.arange(nocc)[~mo_idx[:nocc]]
        act_idx  = np.arange(H1.shape[-1])[mo_idx]
        rdm1_core = np.zeros_like(H1)
        rdm1_core[core_idx, core_idx] = 2.0
        veff_core = _get_veff(rdm1_core, H2)[0]
        E_core = einsum('ij, ji', H1 + veff_core*0.5, rdm1_core)
        
        H0 += E_core
        H1 = (H1 + veff_core)[np.ix_(act_idx, act_idx)]
        H2 = take_eri(H2, act_idx, act_idx, act_idx, act_idx, compact=True)

    f = lib.H5TmpFile(filename=rdm2_tmp_fname, mode='w')
    t1, t2, l1, l2 = mycc.t1, mycc.t2, mycc.l1, mycc.l2
    cc.ccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, f, False)
    nocc, nvir = t1.shape
    norb = nocc + nvir
    
    d1 = cc.ccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    E1 =  (einsum('ij, ji', H1[:nocc, :nocc], doo) \
         + einsum('ij, ji', H1[nocc:, :nocc], dov) \
         + einsum('ij, ji', H1[:nocc, nocc:], dvo) \
         + einsum('ij, ji', H1[nocc:, nocc:], dvv) \
         + np.sum(H1[range(nocc), range(nocc)])) * 2
    
    rdm1_mo = cc.ccsd_rdm._make_rdm1(mycc, d1, with_frozen=False)
    rdm1_mo[np.diag_indices(nocc)] -= 2.0
    rdm1_hf = np.zeros((norb, norb))
    rdm1_hf[range(nocc), range(nocc)] = 2.0
    veff_hf = _get_veff(rdm1_hf, H2)[0]
    E2_prod = 0.5 * einsum('ij, ji', veff_hf, rdm1_hf + rdm1_mo * 2.0)

    if blksize is None:
        max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
        unit = nvir**3*6
        blksize = min(nocc, nvir, max(cc.ccsd.BLKMIN, int(max_memory*.95e6/8/unit)))
    
    oidx = np.arange(nocc)
    vidx = np.arange(nocc, norb)
    E2_cum = 0.0
    
    for p0, p1 in lib.prange(0, nocc, blksize):
        cidx = oidx[p0:p1]

        eri_ovov = take_eri(H2, cidx, vidx, oidx, vidx)
        E_ovov = einsum('ijkl, ijkl', eri_ovov, f['dovov'][p0:p1])
        E2_cum += E_ovov
        eri_ovov = None

        eri_oovv = take_eri(H2, cidx, oidx, vidx, vidx)
        E_oovv = einsum('ijkl, ijkl', eri_oovv, f['doovv'][p0:p1]) 
        E2_cum += E_oovv
        eri_oovv = None
    
        eri_ovvo = take_eri(H2, cidx, vidx, vidx, oidx)
        E_ovvo = einsum('ijkl, ijkl', eri_ovvo, f['dovvo'][p0:p1])
        E2_cum += E_ovvo
        eri_ovvo = None
        
        eri_oooo = take_eri(H2, cidx, oidx, oidx, oidx)
        E_oooo = einsum('ijkl, ijkl', eri_oooo, f['doooo'][p0:p1])
        E2_cum += E_oooo
        eri_oooo = None
    
        eri_ovvv = take_eri(H2, cidx, vidx, vidx, vidx)
        E_ovvv = einsum('ijkl, ijkl', eri_ovvv, f['dovvv'][p0:p1])
        E2_cum += E_ovvv
        eri_ovvv = None
        
        eri_ooov = take_eri(H2, cidx, oidx, oidx, vidx)
        E_ooov = einsum('ijkl, ijkl', eri_ooov, f['dooov'][p0:p1])
        E2_cum += E_ooov
        eri_ooov = None
        
    for p0, p1 in lib.prange(0, nvir, blksize):
        cidx = vidx[p0:p1]
        eri_vvvv = take_eri(H2, cidx, vidx, vidx, vidx)
        E_vvvv = einsum('ijkl, ijkl', eri_vvvv, f['dvvvv'][p0:p1])
        E2_cum += E_vvvv
        eri_vvvv = None
    
    E2_cum *= 2.0
    E2 = E2_prod + E2_cum
    
    E_tot = H0 + E1 + E2
    log.debug(0, "CC exp_val: E0: %20.12f, E1: %20.12f, E2: %20.12f \n"
              "(prod: %20.12f, cum: %20.12f), E_tot: %20.12f", 
              H0, E1, E2, E2_prod, E2_cum, E_tot)
    return E_tot

def exp_val_uccsd(mycc, H1, H2, H0=0.0, rdm2_tmp_fname=None, blksize=None):
    """
    Expectation value of H0, H1 and H2, using an outcore routine.
    H1 and H2 are in MO basis. 
    H2 is real and has 4-fold symmetry, 1, 4-fold array are ok.

    Args:
        mycc: cc object.
        H1: (2, nmo, nmo)
        H2: (3, nmo, nmo, nmo, nmo) or (3, nmo_pair, nmo_pair),
            aa, bb, ab order.
        H0: scalar
        rdm2_tmp_fname: if given, will save the rdm2 intermidiates.

    Returns:
        E_tot: the expectation value of H0, H1 and H2.
    """
    mo_idx = mycc.get_frozen_mask()
    if not (all(mo_idx[0]) and all(mo_idx[1])):
        nocc = [np.count_nonzero(mycc.mo_occ[s] > 0) for s in range(2)]
        core_idx = [np.arange(nocc[s])[~mo_idx[s][:nocc[s]]] for s in range(2)]
        act_idx  = [np.arange(H1.shape[-1])[mo_idx[s]] for s in range(2)]
        rdm1_core = np.zeros_like(H1)
        for s in range(2):
            rdm1_core[s, core_idx[s], core_idx[s]] = 1.0
        veff_core = _get_veff(rdm1_core, H2)
        E_core = einsum('sij, sji', H1 + veff_core*0.5, rdm1_core)
        
        H0 += E_core
        H1 = [(H1[s] + veff_core[s])[np.ix_(act_idx[s], act_idx[s])] for s in range(2)]
        H2 = [take_eri(H2[0], act_idx[0], act_idx[0], act_idx[0], act_idx[0], compact=True),
              take_eri(H2[1], act_idx[1], act_idx[1], act_idx[1], act_idx[1], compact=True),
              take_eri(H2[2], act_idx[0], act_idx[0], act_idx[1], act_idx[1], compact=True)]

    f = lib.H5TmpFile(filename=rdm2_tmp_fname, mode='w')
    t1, t2, l1, l2 = mycc.t1, mycc.t2, mycc.l1, mycc.l2
    d2 = cc.uccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, f, False)
    d2 = [(np.zeros((0)) if d is None else d for d in dx) for dx in d2]
    f["dovov"], f["dovOV"], f["dOVov"], f["dOVOV"] = d2[0]
    f["dvvvv"], f["dvvVV"], f["dVVvv"], f["dVVVV"] = d2[1]
    f["doooo"], f["dooOO"], f["dOOoo"], f["dOOOO"] = d2[2]
    f["doovv"], f["dooVV"], f["dOOvv"], f["dOOVV"] = d2[3]
    f["dovvo"], f["dovVO"], f["dOVvo"], f["dOVVO"] = d2[4]
    f["dvvov"], f["dvvOV"], f["dVVov"], f["dVVOV"] = d2[5]
    f["dovvv"], f["dovVV"], f["dOVvv"], f["dOVVV"] = d2[6]
    f["dooov"], f["dooOV"], f["dOOov"], f["dOOOV"] = d2[7]
    d2 = None

    nocca, nvira = t1[0].shape
    norba = nocca + nvira
    noccb, nvirb = t1[1].shape
    norbb = noccb + nvirb
    
    d1 = cc.uccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
    doo, dOO = d1[0]
    dov, dOV = d1[1]
    dvo, dVO = d1[2]
    dvv, dVV = d1[3]
    E1 =   einsum('ij, ji', H1[0][:nocca, :nocca], doo) \
         + einsum('ij, ji', H1[0][nocca:, :nocca], dov) \
         + einsum('ij, ji', H1[0][:nocca, nocca:], dvo) \
         + einsum('ij, ji', H1[0][nocca:, nocca:], dvv) \
         + np.sum(H1[0][range(nocca), range(nocca)])
    E1 +=  einsum('ij, ji', H1[1][:noccb, :noccb], dOO) \
         + einsum('ij, ji', H1[1][noccb:, :noccb], dOV) \
         + einsum('ij, ji', H1[1][:noccb, noccb:], dVO) \
         + einsum('ij, ji', H1[1][noccb:, noccb:], dVV) \
         + np.sum(H1[1][range(noccb), range(noccb)])
    
    rdm1_mo = cc.uccsd_rdm._make_rdm1(mycc, d1, with_frozen=False)
    rdm1_mo = list(rdm1_mo)
    rdm1_mo[0][np.diag_indices(nocca)] -= 1.0
    rdm1_mo[1][np.diag_indices(noccb)] -= 1.0

    rdm1_hf = [np.zeros((norba, norba)), np.zeros((norbb, norbb))]
    rdm1_hf[0][range(nocca), range(nocca)] = 1.0
    rdm1_hf[1][range(noccb), range(noccb)] = 1.0
    
    vj00, vk00 = hf.dot_eri_dm(H2[0], rdm1_hf[0], hermi=1)
    vj11, vk11 = hf.dot_eri_dm(H2[1], rdm1_hf[1], hermi=1)
    if rdm1_hf[0].shape == rdm1_hf[1].shape:
        vj01 = hf.dot_eri_dm(H2[2], rdm1_hf[1], hermi=1, with_j=True,
                             with_k=False)[0]
        vj10 = hf.dot_eri_dm(H2[2].T, rdm1_hf[0], hermi=1, with_j=True,
                             with_k=False)[0]
    else:
        rdm1_b = rdm1_hf[1] * 2.0
        rdm1_b[range(norbb), range(norbb)] *= 0.5
        rdm1_b = lib.pack_tril(rdm1_b)
        vj01 = np.dot(H2[2], rdm1_b)
        vj01 = lib.unpack_tril(vj01)
        
        rdm1_a = rdm1_hf[0] * 2.0
        rdm1_a[range(norba), range(norba)] *= 0.5
        rdm1_a = lib.pack_tril(rdm1_a)
        vj10 = np.dot(H2[2].T, rdm1_a)
        vj10 = lib.unpack_tril(vj10)
        rdm1_a = rdm1_b = None

    veff_hf = [vj00 + vj01 - vk00, vj11 + vj10 - vk11]
    E2_prod = 0.0
    for s in range(2):
        E2_prod += 0.5 * einsum('ij, ji', veff_hf[s], rdm1_hf[s] + rdm1_mo[s] * 2.0)
    
    if blksize is None:
        max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
        unit = max(nvira, nvirb)**3 * 6
        blksize = min(nocca, noccb, nvira, nvirb, max(cc.ccsd.BLKMIN, 
                      int(max_memory*.95e6/8/unit)))
    
    E2_aa = 0.0
    oidxa = np.arange(nocca)
    vidxa = np.arange(nocca, norba)
    
    for p0, p1 in lib.prange(0, nocca, blksize):
        cidx = oidxa[p0:p1]

        eri_ovov = take_eri(H2[0], cidx, vidxa, oidxa, vidxa)
        E_ovov = einsum('ijkl, ijkl', eri_ovov, f['dovov'][p0:p1])
        E2_aa += E_ovov * 2
        eri_ovov = None

        eri_oovv = take_eri(H2[0], cidx, oidxa, vidxa, vidxa)
        E_oovv = einsum('ijkl, ijkl', eri_oovv, f['doovv'][p0:p1]) 
        E2_aa += E_oovv * 2
        eri_oovv = None
    
        eri_ovvo = take_eri(H2[0], cidx, vidxa, vidxa, oidxa)
        E_ovvo = einsum('ijkl, ijkl', eri_ovvo, f['dovvo'][p0:p1])
        E2_aa += E_ovvo * 2
        eri_ovvo = None
        
        eri_oooo = take_eri(H2[0], cidx, oidxa, oidxa, oidxa)
        E_oooo = einsum('ijkl, ijkl', eri_oooo, f['doooo'][p0:p1])
        E2_aa += E_oooo
        eri_oooo = None
    
        eri_ovvv = take_eri(H2[0], cidx, vidxa, vidxa, vidxa)
        E_ovvv = einsum('ijkl, ijkl', eri_ovvv, f['dovvv'][p0:p1])
        E2_aa += E_ovvv * 4
        eri_ovvv = None
        
        eri_ooov = take_eri(H2[0], cidx, oidxa, oidxa, vidxa)
        E_ooov = einsum('ijkl, ijkl', eri_ooov, f['dooov'][p0:p1])
        E2_aa += E_ooov * 4
        eri_ooov = None
        
    for p0, p1 in lib.prange(0, nvira, blksize):
        cidx = vidxa[p0:p1]
        eri_vvvv = take_eri(H2[0], cidx, vidxa, vidxa, vidxa)
        E_vvvv = einsum('ijkl, ijkl', eri_vvvv, f['dvvvv'][p0:p1])
        E2_aa += E_vvvv
        eri_vvvv = None

    E2_bb = 0.0
    Oidxb = np.arange(noccb)
    Vidxb = np.arange(noccb, norbb)
    
    for p0, p1 in lib.prange(0, noccb, blksize):
        cidx = Oidxb[p0:p1]

        eri_ovov = take_eri(H2[1], cidx, Vidxb, Oidxb, Vidxb)
        E_ovov = einsum('ijkl, ijkl', eri_ovov, f['dOVOV'][p0:p1])
        E2_bb += E_ovov * 2
        eri_ovov = None

        eri_oovv = take_eri(H2[1], cidx, Oidxb, Vidxb, Vidxb)
        E_oovv = einsum('ijkl, ijkl', eri_oovv, f['dOOVV'][p0:p1]) 
        E2_bb += E_oovv * 2
        eri_oovv = None
    
        eri_ovvo = take_eri(H2[1], cidx, Vidxb, Vidxb, Oidxb)
        E_ovvo = einsum('ijkl, ijkl', eri_ovvo, f['dOVVO'][p0:p1])
        E2_bb += E_ovvo * 2
        eri_ovvo = None
        
        eri_oooo = take_eri(H2[1], cidx, Oidxb, Oidxb, Oidxb)
        E_oooo = einsum('ijkl, ijkl', eri_oooo, f['dOOOO'][p0:p1])
        E2_bb += E_oooo
        eri_oooo = None
    
        eri_ovvv = take_eri(H2[1], cidx, Vidxb, Vidxb, Vidxb)
        E_ovvv = einsum('ijkl, ijkl', eri_ovvv, f['dOVVV'][p0:p1])
        E2_bb += E_ovvv * 4
        eri_ovvv = None
        
        eri_ooov = take_eri(H2[1], cidx, Oidxb, Oidxb, Vidxb)
        E_ooov = einsum('ijkl, ijkl', eri_ooov, f['dOOOV'][p0:p1])
        E2_bb += E_ooov * 4
        eri_ooov = None
        
    for p0, p1 in lib.prange(0, nvirb, blksize):
        cidx = Vidxb[p0:p1]
        eri_vvvv = take_eri(H2[1], cidx, Vidxb, Vidxb, Vidxb)
        E_vvvv = einsum('ijkl, ijkl', eri_vvvv, f['dVVVV'][p0:p1])
        E2_bb += E_vvvv
        eri_vvvv = None
    
    E2_ab = 0.0
    for p0, p1 in lib.prange(0, nocca, blksize):
        cidx = oidxa[p0:p1]

        eri_ovOV = take_eri(H2[2], cidx, vidxa, Oidxb, Vidxb)
        E_ovOV = einsum('ijkl, ijkl', eri_ovOV, f['dovOV'][p0:p1])
        E2_ab += E_ovOV * 2
        eri_ovOV = None
        
        eri_ooVV = take_eri(H2[2], cidx, oidxa, Vidxb, Vidxb)
        E_ooVV = einsum('ijkl, ijkl', eri_ooVV, f['dooVV'][p0:p1])
        E2_ab += E_ooVV
        eri_ooVV = None

        eri_ovVO = take_eri(H2[2], cidx, vidxa, Vidxb, Oidxb)
        E_ovVO = einsum('ijkl, ijkl', eri_ovVO, f['dovVO'][p0:p1])
        E2_ab += E_ovVO * 2
        eri_ovVO = None

        eri_ooOO = take_eri(H2[2], cidx, oidxa, Oidxb, Oidxb)
        E_ooOO = einsum('ijkl, ijkl', eri_ooOO, f['dooOO'][p0:p1])
        E2_ab += E_ooOO
        eri_ooOO = None
        
        eri_ovVV = take_eri(H2[2], cidx, vidxa, Vidxb, Vidxb)
        E_ovVV = einsum('ijkl, ijkl', eri_ovVV, f['dovVV'][p0:p1])
        E2_ab += E_ovVV * 2
        eri_ovVV = None
        
        eri_ooOV = take_eri(H2[2], cidx, oidxa, Oidxb, Vidxb)
        E_ooOV = einsum('ijkl, ijkl', eri_ooOV, f['dooOV'][p0:p1])
        E2_ab += E_ooOV * 2
        eri_ooOV = None
        
        eri_ovOO = take_eri(H2[2], cidx, vidxa, Oidxb, Oidxb)
        E_ovOO = einsum('ijkl, klij', eri_ovOO, f['dOOov'][:, :, p0:p1])
        E2_ab += E_ovOO * 2
        eri_ovOO = None
    
    for p0, p1 in lib.prange(0, nvira, blksize):
        cidx = vidxa[p0:p1]
        
        eri_vvOO = take_eri(H2[2], cidx, vidxa, Oidxb, Oidxb)
        E_vvOO = einsum('ijkl, klij', eri_vvOO, f['dOOvv'][:, :, p0:p1])
        E2_ab += E_vvOO
        eri_vvOO = None

        eri_vvVV = take_eri(H2[2], cidx, vidxa, Vidxb, Vidxb)
        E_vvVV = einsum('ijkl, ijkl', eri_vvVV, f['dvvVV'][p0:p1])
        E2_ab += E_vvVV
        eri_vvVV = None

        eri_vvOV = take_eri(H2[2], cidx, vidxa, Oidxb, Vidxb)
        E_vvOV = einsum('ijkl, klij', eri_vvOV, f['dOVvv'][:, :, p0:p1])
        E2_ab += E_vvOV * 2
        eri_vvOV = None
    
    E2_cum = 0.5 * (E2_aa + E2_bb) + E2_ab
    E2 = E2_prod + E2_cum
    
    E_tot = H0 + E1 + E2
    log.debug(0, "CC exp_val: E0: %20.12f, E1: %20.12f, E2: %20.12f \n"
              "(prod: %20.12f, cum: %20.12f), E_tot: %20.12f", 
              H0, E1, E2, E2_prod, E2_cum, E_tot)
    return E_tot

def transform_t1_to_bo(t1, umat):
    if isinstance(t1, np.ndarray) and t1.ndim == 2:
        nocc, nvir = t1.shape
        umat_occ = umat[:nocc, :nocc]
        umat_vir = umat[nocc:, nocc:] 
        return mdot(umat_occ.conj().T, t1, umat_vir)
    else:
        spin = len(t1)
        return [transform_t1_to_bo(t1[s], umat[s]) \
                for s in range(spin)]

def transform_t2_to_bo(t2, umat, umat_b=None):
    if isinstance(t2, np.ndarray) and t2.ndim == 4:
        umat_a = umat
        if umat_b is None:
            umat_b = umat_a

        nocc_a, nocc_b, nvir_a, nvir_b = t2.shape
        umat_occ_a = umat_a[:nocc_a, :nocc_a]
        umat_occ_b = umat_b[:nocc_b, :nocc_b]
        umat_vir_a = umat_a[nocc_a:, nocc_a:]
        umat_vir_b = umat_b[nocc_b:, nocc_b:]
        t2_bo = einsum("ijab, iI, jJ, aA, bB -> IJAB", t2, \
                          umat_occ_a, umat_occ_b, umat_vir_a, umat_vir_b)
    else:
        t2_bo = [None, None, None]
        t2_bo[0] = transform_t2_to_bo(t2[0], umat[0])
        t2_bo[1] = transform_t2_to_bo(t2[1], umat[0], umat_b=umat[1])
        t2_bo[2] = transform_t2_to_bo(t2[2], umat[1])
    return t2_bo

transform_l1_to_bo = transform_t1_to_bo
transform_l2_to_bo = transform_t2_to_bo

if __name__ == '__main__':
    pass
