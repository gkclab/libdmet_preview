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
from libdmet.system import integral
from libdmet.solver import scf
from libdmet.solver import lgccsd, lgccd
from libdmet.solver.scf import (ao2mo_Ham, restore_Ham, _get_veff,
                                _get_veff_ghf)
from libdmet.solver.ccd import CCD
from libdmet.solver.uccd import UCCD
from libdmet.solver.gccd import GCCD
from libdmet.solver import uccsd_ite
from libdmet.basis_transform.make_basis import (transform_rdm1_to_ao_mol,
                                                transform_rdm2_to_ao_mol,
                                                rotate_emb_basis,
                                                find_closest_mo, trans_mo,
                                                get_mo_ovlp)
from libdmet.utils.misc import mdot, max_abs, take_eri

from pyscf.scf import hf
from pyscf import cc
from pyscf import ci
from pyscf import lib
from pyscf import ao2mo
from pyscf.cc import gccsd
from pyscf.cc.uccsd import _ChemistsERIs
from pyscf.cc.gccsd import _PhysicistsERIs
from pyscf.lib import logger

einsum = partial(np.einsum, optimize=True)

# ****************************************************************************
# UCC
# ****************************************************************************

def _make_eris_incore_uhf(mycc, mo_coeff=None, ao2mofn=None):
    """
    Hacked CC make eri function. NOTE the order.
    """
    from libdmet.utils import tril_take_idx
    assert ao2mofn is None
    cput0 = (time.process_time(), time.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    naoa, nmoa = moa.shape
    naob, nmob = mob.shape
    assert naoa == naob

    if len(mycc._scf._eri) == 1:
        eri_ao = [mycc._scf._eri[0], mycc._scf._eri[0], mycc._scf._eri[0]]
    elif len(mycc._scf._eri) == 3:
        eri_ao = mycc._scf._eri
    else:
        raise ValueError("Unknown ERI length %s"%(len(mycc._scf._eri)))

    # aa
    o = np.arange(0, nocca)
    v = np.arange(nocca, nmoa)

    # ZHC NOTE special treatment for OO-CCD,
    if (naoa == nmoa) and (max_abs(moa - np.eye(nmoa)) < 1e-13):
        eri_aa = ao2mo.restore(4, eri_ao[0], naoa)
    else:
        eri_aa = ao2mo.full(ao2mo.restore(4, eri_ao[0], naoa),
                            moa, compact=True)

    eris.oooo = take_eri(eri_aa, o, o, o, o)
    eris.ovoo = take_eri(eri_aa, o, v, o, o)
    eris.ovov = take_eri(eri_aa, o, v, o, v)
    eris.oovv = take_eri(eri_aa, o, o, v, v)
    eris.ovvo = take_eri(eri_aa, o, v, v, o)

    idx1 = tril_take_idx(o, v, compact=False)
    idx2 = tril_take_idx(v, v, compact=True)
    eris.ovvv = eri_aa[np.ix_(idx1, idx2)].reshape(nocca, nvira, nvira*(nvira+1)//2)

    eris.vvvv = take_eri(eri_aa, v, v, v, v, compact=True)
    eri_aa = None

    # bb
    O = np.arange(0, noccb)
    V = np.arange(noccb, nmob)

    if (naob == nmob) and (max_abs(mob - np.eye(nmob)) < 1e-13):
        eri_bb = ao2mo.restore(4, eri_ao[1], naob)
    else:
        eri_bb = ao2mo.full(ao2mo.restore(4, eri_ao[1], naob),
                            mob, compact=True)

    eris.OOOO = take_eri(eri_bb, O, O, O, O)
    eris.OVOO = take_eri(eri_bb, O, V, O, O)
    eris.OVOV = take_eri(eri_bb, O, V, O, V)
    eris.OOVV = take_eri(eri_bb, O, O, V, V)
    eris.OVVO = take_eri(eri_bb, O, V, V, O)

    idx1 = tril_take_idx(O, V, compact=False)
    idx2 = tril_take_idx(V, V, compact=True)
    eris.OVVV = eri_bb[np.ix_(idx1, idx2)].reshape(noccb, nvirb, nvirb*(nvirb+1)//2)

    eris.VVVV = take_eri(eri_bb, V, V, V, V, compact=True)
    eri_bb = None

    # ab
    if ((naoa == naob) and (nmoa == nmob)) and \
       ((naoa == nmoa) and (max_abs(moa - np.eye(nmoa)) < 1e-13)) and \
       ((naob == nmob) and (max_abs(mob - np.eye(nmob)) < 1e-13)):
        eri_ab = ao2mo.restore(4, eri_ao[2], naoa)
    else:
        eri_ab = ao2mo.general(ao2mo.restore(4, eri_ao[2], naoa),
                               (moa, moa, mob, mob), compact=True)
    eri_ao = None

    eris.ooOO = take_eri(eri_ab, o, o, O, O)
    eris.ovOO = take_eri(eri_ab, o, v, O, O)
    eris.ovOV = take_eri(eri_ab, o, v, O, V)
    eris.ooVV = take_eri(eri_ab, o, o, V, V)
    eris.ovVO = take_eri(eri_ab, o, v, V, O)

    idx1 = tril_take_idx(o, v, compact=False)
    eris.ovVV = eri_ab[np.ix_(idx1, idx2)].reshape(nocca, nvira, nvirb*(nvirb+1)//2)

    eris.vvVV = take_eri(eri_ab, v, v, V, V, compact=True)

    # ba
    eri_ba = eri_ab.T
    eri_ab = None

    eris.OVoo = take_eri(eri_ba, O, V, o, o)
    eris.OOvv = take_eri(eri_ba, O, O, v, v)
    eris.OVvo = take_eri(eri_ba, O, V, v, o)

    idx1 = tril_take_idx(O, V, compact=False)
    idx2 = tril_take_idx(v, v, compact=True)
    eris.OVvv = eri_ba[np.ix_(idx1, idx2)].reshape(noccb, nvirb, nvira*(nvira+1)//2)
    eri_ba = None
    return eris

def ao2mo_uhf(mycc, mo_coeff=None):
    nmoa, nmob = mycc.get_nmo()
    nao = mycc.mo_coeff[0].shape[0]
    nmo_pair = nmoa * (nmoa+1) // 2
    nao_pair = nao * (nao+1) // 2
    mem_incore = (max(nao_pair**2, nmoa**4) + nmo_pair**2) * 8/1e6
    mem_now = lib.current_memory()[0]
    if (mycc._scf._eri is not None and
        (mem_incore+mem_now < mycc.max_memory or mycc.incore_complete)):
        return _make_eris_incore_uhf(mycc, mo_coeff)

    elif getattr(mycc._scf, 'with_df', None):
        log.warn('UCCSD detected DF being used in the HF object. '
                 'MO integrals are computed based on the DF 3-index tensors.\n'
                 'It\'s recommended to use dfccsd.CCSD for the '
                 'DF-CCSD calculations')
        raise NotImplementedError

    else:
        raise NotImplementedError
        return _make_eris_outcore(mycc, mo_coeff)

def init_amps_uhf(mycc, eris=None):
    time0 = logger.process_clock(), logger.perf_counter()
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    nocca, noccb = mycc.nocc

    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]
    mo_ea_o = eris.mo_energy[0][:nocca]
    mo_ea_v = eris.mo_energy[0][nocca:] + mycc.level_shift
    mo_eb_o = eris.mo_energy[1][:noccb]
    mo_eb_v = eris.mo_energy[1][noccb:] + mycc.level_shift
    eia_a = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eia_b = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)

    t1a = fova.conj() / eia_a
    t1b = fovb.conj() / eia_b

    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    t2aa = eris_ovov.transpose(0,2,1,3) / lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
    t2ab = eris_ovOV.transpose(0,2,1,3) / lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
    t2bb = eris_OVOV.transpose(0,2,1,3) / lib.direct_sum('ia+jb->ijab', eia_b, eia_b)
    t2aa = t2aa - t2aa.transpose(0,1,3,2)
    t2bb = t2bb - t2bb.transpose(0,1,3,2)
    e  =      np.einsum('iJaB,iaJB', t2ab, eris_ovOV)
    e += 0.25*np.einsum('ijab,iajb', t2aa, eris_ovov)
    e -= 0.25*np.einsum('ijab,ibja', t2aa, eris_ovov)
    e += 0.25*np.einsum('ijab,iajb', t2bb, eris_OVOV)
    e -= 0.25*np.einsum('ijab,ibja', t2bb, eris_OVOV)
    mycc.emp2 = e.real
    logger.info(mycc, 'Init t2, MP2 energy = %.15g', mycc.emp2)
    logger.timer(mycc, 'init mp2', *time0)
    return mycc.emp2, (t1a,t1b), (t2aa,t2ab,t2bb)

def make_rdm2_uhf(mycc, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
    from libdmet.solver import uccsd_rdm
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = mycc.l1
    if l2 is None: l2 = mycc.l2
    if l1 is None: l1, l2 = mycc.solve_lambda(t1, t2)
    return uccsd_rdm.make_rdm2(mycc, t1, t2, l1, l2, ao_repr=ao_repr)

class UICCSD(cc.uccsd.UCCSD):
    init_amps = init_amps_uhf
    ao2mo = ao2mo_uhf
    make_rdm2 = make_rdm2_uhf

class UICCSDITE(uccsd_ite.UCCSDITE):
    init_amps = init_amps_uhf
    ao2mo = ao2mo_uhf
    make_rdm2 = make_rdm2_uhf

class UICCD(UCCD):
    init_amps = init_amps_uhf
    ao2mo = ao2mo_uhf

try:
    from tccsd import DMRGUTCCSD

    class DMRGUITCCSD(DMRGUTCCSD):
        ao2mo = ao2mo_uhf
except ImportError:
    DMRGUITCCSD = None

# ****************************************************************************
# Main solver
# ****************************************************************************

class UTCCSD(object):
    def __init__(self, nproc=1, nnode=1, TmpDir="./tmp", SharedDir=None,
                 restricted=False, Sz=0, bcs=False, ghf=False, tol=1e-9,
                 tol_normt=1e-6, max_cycle=200, level_shift=0.0, frozen=0,
                 max_memory=40000, compact_rdm2=False, scf_newton=True,
                 diis_space=8, iterative_damping=1.0, linear=False,
                 approx_l=False, alpha=None, beta=np.inf, tcc=False,
                 ite=None):
        """
        CCSD solver.
        """
        self.restricted = restricted
        self.max_cycle = max_cycle
        self.max_memory = max_memory
        self.conv_tol = tol
        self.conv_tol_normt = tol_normt
        self.level_shift = level_shift
        self.diis_space = diis_space
        self.iterative_damping = iterative_damping
        self.frozen = frozen
        self.verbose = 5
        self.bcs = bcs
        self.ghf = ghf
        self.Sz = Sz
        self.alpha = alpha
        self.beta = beta
        self.linear = linear
        self.tcc = tcc
        self.ite = ite
        self.approx_l = approx_l
        self.scfsolver = scf.SCF(newton_ah=scf_newton)

        self.onepdm = None
        self.twopdm = None

        self.optimized = False

    def run(self, Ham=None, nelec=None, guess=None, restart=False,
            dump_tl=False, fcc_name="fcc.h5", calc_rdm2=False, Mu=None,
            **kwargs):
        """
        Main kernel function of the solver.
        NOTE: the spin order for unrestricted H2 is aa, bb, ab.
        """
        # *********************************************************************
        # 1. sanity check
        # *********************************************************************
        log.info("CC solver: start")
        spin = Ham.H1["cd"].shape[0]
        if spin > 1:
            log.eassert(not self.restricted, "CC solver: spin (%s) > 1 "
                        "requires unrestricted", spin)
        if nelec is None:
            if self.bcs:
                nelec = Ham.norb * 2
            elif self.ghf:
                nelec = Ham.norb // 2
            else:
                raise ValueError("CC solver: nelec cannot be None "
                                 "for RCC or UCC.")
        nelec_a, nelec_b = (nelec + self.Sz) // 2, (nelec - self.Sz) // 2
        log.eassert(nelec_a >= 0 and nelec_b >=0, "CC solver: "
                    "nelec_a (%s), nelec_b (%s) should >= 0", nelec_a, nelec_b)
        log.eassert(nelec_a + nelec_b == nelec, "CC solver: "
                    "nelec_a (%s) + nelec_b (%s) should == nelec (%s)",
                    nelec_a, nelec_b, nelec)

        # *********************************************************************
        # 2. mean-field calculation
        # *********************************************************************
        log.debug(1, "CC solver: mean-field")
        self.scfsolver.set_system(nelec, self.Sz, False, self.restricted,
                                  max_memory=self.max_memory)
        self.scfsolver.set_integral(Ham)

        dm0 = kwargs.get("dm0", None)
        scf_max_cycle = kwargs.get("scf_max_cycle", 200)
        bcc = kwargs.get("bcc", False)
        bcc_verbose = kwargs.get("bcc_verbose", 2)
        bcc_restart = kwargs.get("bcc_restart", False)
        bcc_tol = kwargs.get("bcc_tol", self.conv_tol_normt * 10.0)
        if bcc and bcc_restart and self.optimized and restart:
            bcc_restart = True
            scf_max_cycle = 1 # need not to do scf
        else:
            bcc_restart = False

        if self.ghf:
            E_HF, rhoHF = self.scfsolver.GGHF(tol=min(self.conv_tol*0.1, 1e-10),
                                              MaxIter=scf_max_cycle,
                                              InitGuess=dm0,
                                              Mu=Mu,
                                              alpha=self.alpha,
                                              beta=self.beta)
        else:
            E_HF, rhoHF = self.scfsolver.HF(tol=min(self.conv_tol*0.1, 1e-10),
                                            MaxIter=scf_max_cycle,
                                            InitGuess=dm0, Mu=Mu,
                                            alpha=self.alpha, beta=self.beta)
        if self.alpha != 1.0:
            # ZHC NOTE alpha is adjusted to 1 after converge mf
            log.info("adjust mf.alpha to 1.0 for CC.")
            self.scfsolver.mf.alpha = 1.0
        if self.beta < np.inf:
            log.info("adjust mf.mo_occ to integer for CC.")
            self.scfsolver.mf.mo_occ = np.round(self.scfsolver.mf.mo_occ)
            self.scfsolver.mf.sigma = 0.0

        log.debug(1, "CC solver: mean-field converged: %s",
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


        ncas = kwargs["ncas"]
        nelecas = kwargs["nelecas"]
        maxM = kwargs.get("maxM", 500)
        sweep_tol = kwargs.get("sweep_tol", 1e-6)
        sample_tol = kwargs.get("sample_tol", sweep_tol * 0.01)

        from pyscf import mcscf
        import tccsd

        print (nelec_a, nelec_b)

        print (self.scfsolver.mf.nelec)
        print (self.scfsolver.mf.mol.nelectron)
        print (self.scfsolver.mf.mol.spin)



        mc_dmrg = mcscf.UCASCI(self.scfsolver.mf, ncas, nelecas)
        mc_dmrg.fcisolver = tccsd.DMRGCI(self.scfsolver.mf)
        mc_dmrg.fcisolver.dmrg_args['maxM'] = maxM
        mc_dmrg.fcisolver.dmrg_args['scratch'] = "./tmp"
        mc_dmrg.fcisolver.dmrg_args['sweep_tol'] = sweep_tol
        mc_dmrg.fcisolver.dmrg_args['sample_tol'] = sample_tol
        mc_dmrg.kernel()

        tcc_dmrg = DMRGUITCCSD(mc_dmrg)
        self.cisolver = tcc_dmrg

        self.cisolver.max_cycle = self.max_cycle
        self.cisolver.conv_tol = self.conv_tol
        self.cisolver.conv_tol_normt = self.conv_tol_normt
        self.cisolver.level_shift = self.level_shift
        if self.diis_space <= 0:
            self.cisolver.diis = False
        self.cisolver.diis_space = self.diis_space
        self.cisolver.iterative_damping = self.iterative_damping
        self.cisolver.set(frozen=self.frozen)
        self.cisolver.verbose = self.verbose

        # *********************************************************************
        # 4. solve t1, t2, restart can use the previously saved t1 and t2
        # *********************************************************************
        eris = self.cisolver.ao2mo(self.cisolver.mo_coeff)

        if restart:
            log.eassert("basis" in kwargs, "restart requires basis passed in")
        if restart and self.optimized:
            t1, t2, l1, l2 = self.load_t12_from_h5(fcc_name, kwargs["basis"],
                                                   self.scfsolver.mf.mo_coeff,
                                                   bcc_restart=bcc_restart)
        else:
            if isinstance(guess, str) and guess == 'cisd':
                if self.ghf:
                    myci = GGCISD(self.scfsolver.mf)
                elif Ham.restricted:
                    raise NotImplementedError
                else:
                    raise NotImplementedError
                myci.conv_tol = self.conv_tol
                ecisd, civec = myci.kernel(eris=eris)
                _, t1, t2 = myci.cisdvec_to_amplitudes(civec, myci.nmo, myci.nocc)
                l1 = l2 = None
            elif guess is not None:
                if len(guess) == 2:
                    t1, t2 = guess
                    l1, l2 = None, None
                else:
                    t1, t2, l1, l2 = guess
            else:
                t1, t2, l1, l2 = None, None, None, None

        # solve t1, t2
        log.debug(1, "CC solver: solve t amplitudes")

        E_corr, t1, t2 = self.cisolver.kernel(t1=t1, t2=t2, eris=eris)

        # brueckner CC
        if bcc:
            log.info("Using Brueckner CC.")
            verbose_old = self.cisolver.verbose
            mo_coeff_old = self.cisolver.mo_coeff
            self.cisolver = bcc_loop(self.cisolver, utol=bcc_tol,
                                     verbose=bcc_verbose)
            self.cisolver.verbose = verbose_old
            self.scfsolver.mf.verbose = verbose_old
            self.scfsolver.mf.mo_coeff = self.cisolver.mo_coeff
            self.scfsolver.mf.e_tot = self.cisolver._scf.e_tot
            t1, t2 = self.cisolver.t1, self.cisolver.t2
            eris = self.cisolver.ao2mo(self.cisolver.mo_coeff)

            if l1 is not None and l2 is not None:
                # transform l1, l2 to the new BCC basis
                log.info("transform old l1, l2 to new BCC basis...")
                if isinstance(mo_coeff_old, np.ndarray) and mo_coeff_old.ndim == 2:
                    umat = get_mo_ovlp(mo_coeff_old, self.cisolver.mo_coeff,
                                       self.scfsolver.mf.get_ovlp())
                    l1 = transform_l1_to_bo(l1, umat)
                    l2 = transform_l2_to_bo(l2, umat)
                else: # UHF
                    umat = []
                    for s in range(mo_coeff.shape[0]):
                        umat.append(get_mo_ovlp(mo_coeff_old[s],
                                                self.cisolver.mo_coeff[s],
                                                self.scfsolver.mf.get_ovlp()))
                    l1 = transform_l1_to_bo(l1, umat)
                    l2 = transform_l2_to_bo(l2, umat)

        # *********************************************************************
        # 5. (T) and solve lambda
        # *********************************************************************
        log.debug(1, "CC solver: solve l amplitudes")
        if kwargs.get("ccsdt", False):
            log.info("CCSD(T) correction")
            # CCSD(T)
            # ZHC NOTE the CCSD(T) lambda does not support fov != 0
            # thus the correctness of BCCD(T) needs to be checked!
            if kwargs.get("ccsdt_energy", False):
                e_t = self.cisolver.ccsd_t(eris=eris)
                E_corr += e_t
                log.info("CCSD(T) E(T): %20.12f", e_t)
            else:
                e_t = 0.0

            lambda_drv, rdm1_drv, rdm2_drv = self._get_ccsdt_drv(eris=eris)
            l1, l2 = self.cisolver.l1, self.cisolver.l2 = \
                    lambda_drv(self.cisolver, eris=eris, t1=t1, t2=t2,
                               max_cycle=self.max_cycle,
                               tol=self.conv_tol_normt,
                               verbose=self.verbose)[1:]
        else:
            if self.linear or self.approx_l:
                l1 = self.cisolver.l1 = self.cisolver.t1
                l2 = self.cisolver.l2 = self.cisolver.t2
            else:
                l1, l2 = self.cisolver.solve_lambda(t1=t1, t2=t2, l1=l1, l2=l2,
                                                    eris=eris)
            rdm1_drv = rdm2_drv = None
            e_t = 0.0

        # *********************************************************************
        # 6. collect properties
        # *********************************************************************
        # energy
        E = self.cisolver.e_tot + e_t

        # rdm1 and rdm2
        self.make_rdm1(Ham, drv=rdm1_drv)
        if calc_rdm2:
            self.make_rdm2(Ham, drv=rdm2_drv)

        # dump t1, t2, l1, l2, basis, mo_coeff
        if dump_tl or restart:
            self.save_t12_to_h5(fcc_name, kwargs["basis"], self.cisolver.mo_coeff)

        if not self.cisolver.converged:
            log.warn("CC solver not converged...")
        self.optimized = True
        return self.onepdm, E

    def _get_ccsdt_drv(self, eris=None):
        if self.ghf:
            from pyscf.cc.gccsd_t_lambda import kernel as lambda_drv
            from pyscf.cc import gccsd_t_rdm as rdm_mod
        elif self.restricted:
            from pyscf.cc.ccsd_t_lambda_slow import kernel as lambda_drv
            from pyscf.cc import ccsd_t_rdm_slow as rdm_mod
        else:
            from pyscf.cc.uccsd_t_lambda import kernel as lambda_drv
            from pyscf.cc import uccsd_t_rdm as rdm_mod
        rdm1_drv = partial(rdm_mod.make_rdm1, mycc=self.cisolver, eris=eris)
        rdm2_drv = partial(rdm_mod.make_rdm2, mycc=self.cisolver, eris=eris)
        return lambda_drv, rdm1_drv, rdm2_drv

    def run_dmet_ham(self, Ham, last_aabb=True, save_dmet_ham=False,
                     dmet_ham_fname='dmet_ham.h5', use_calculated_twopdm=False,
                     **kwargs):
        """
        Run scaled DMET Hamiltonian.
        NOTE: the spin order for H2 is aa, bb, ab, the same as ImpHam.
        """
        if kwargs.get("ccsdt", False) or use_calculated_twopdm:
            return self.run_dmet_ham_slow(Ham, last_aabb=last_aabb,
                                          save_dmet_ham=save_dmet_ham,
                                          dmet_ham_fname=dmet_ham_fname,
                                          use_calculated_twopdm=use_calculated_twopdm,
                                          **kwargs)

        log.info("CC solver Run DMET Hamiltonian.")
        log.debug(0, "ao2mo for DMET Hamiltonian.")
        Ham = ao2mo_Ham(Ham, self.cisolver.mo_coeff, compact=True, in_place=True)

        if self.ghf:
            H1 = Ham.H1["cd"][0]
            H2 = Ham.H2["ccdd"][0]
            H0 = Ham.H0
            E = exp_val_gccsd(self.cisolver, H1, H2, H0=H0)
        elif Ham.restricted:
            H1 = Ham.H1["cd"][0]
            H2 = Ham.H2["ccdd"][0]
            H0 = Ham.H0
            E = exp_val_rccsd(self.cisolver, H1, H2, H0=H0)
        else:
            # H2 is in aa, bb, ab order
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

    def run_dmet_ham_slow(self, Ham, last_aabb=True, save_dmet_ham=False,
                          dmet_ham_fname='dmet_ham.h5',
                          use_calculated_twopdm=False, **kwargs):
        """
        Run scaled DMET Hamiltonian.
        NOTE: the spin order for H2 is aa, bb, ab, the same as ImpHam.
        """
        log.info("CC solver Run DMET Hamiltonian.")
        log.debug(0, "ao2mo for DMET Hamiltonian.")
        Ham = ao2mo_Ham(Ham, self.cisolver.mo_coeff, compact=True,
                        in_place=True)
        Ham = restore_Ham(Ham, 1, in_place=True)
        # calculate rdm2 in aa, bb, ab order
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

        if self.ghf:
            h1 = Ham.H1["cd"][0]
            h2 = Ham.H2["ccdd"][0]
            r1 = self.onepdm
            r2 = self.twopdm
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape

            E1 = einsum('ij, ji', h1, r1)
            E2 = 0.5 * einsum('ijkl, ijkl', h2, r2)
        elif Ham.restricted:
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
            # h2 is in aa, bb, ab order
            h2 = Ham.H2["ccdd"]
            r1 = self.onepdm_mo
            # r2 is in aa, bb, ab order
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
        onepdm = drv(t1=self.cisolver.t1, t2=self.cisolver.t2,
                     l1=self.cisolver.l1, l2=self.cisolver.l2)

        if self.ghf: # GHF
            self.onepdm_mo = np.asarray(onepdm)
        elif self.restricted:
            self.onepdm_mo = np.asarray(onepdm)[np.newaxis] * 0.5
        else:
            self.onepdm_mo = np.asarray(onepdm)

        # rotate back to the AO basis
        log.debug(1, "CC solver: rotate rdm1 to AO")
        self.onepdm = transform_rdm1_to_ao_mol(self.onepdm_mo,
                                               self.cisolver.mo_coeff)
        return self.onepdm

    def make_rdm2(self, Ham=None, ao_repr=False, drv=None, with_dm1=True):
        """
        Compute rdm2.
        NOTE: the returned value's spin order for rdm2 is aa, bb, ab.
        """
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

        if self.ghf: # GHF
            self.twopdm_mo = twopdm_mo
        elif self.restricted:
            self.twopdm_mo = twopdm_mo[np.newaxis]
        else:
            # NOTE: here is aa, ab, bb order
            self.twopdm_mo = np.asarray(twopdm_mo)
            twopdm_mo = None

        # rotate back to the AO basis
        # NOTE: the transform function use aa, ab, bb order.
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

    def load_t12_from_h5(self, fcc_name, basis_new, mo_coeff_new,
                         bcc_restart=False):
        """
        Load t1, t2, l1, l2 and rotate to current basis.
        """
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
            if not (self.linear or self.approx_l):
                l1_old = np.asarray(fcc['l1'])
                l2_old = np.asarray(fcc['l2'])
        else:
            spin = mo_coeff_old.shape[0]
            t1_old = [np.asarray(fcc['t1_%s'%s]) for s in range(spin)]
            t2_old = [np.asarray(fcc['t2_%s'%s]) for s in range(spin*(spin+1)//2)]
            if not (self.linear or self.approx_l):
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
        if is_same_basis:
            log.debug(2, "restart with the same basis.")
        else:
            log.debug(2, "restart with the different basis.")

        try:
            if bcc_restart:
                # ZHC NOTE
                # restart for a bcc calculation,
                # new mo is estimated by maximizing the overlap
                # w.r.t. last calculation
                # self.scfsolver.mf.mo_coeff and
                # self.cisolver.mo_coeff are overwritten
                # t1, t2, l1, l2 are unchanged!

                if mo_coeff_new.ndim == 2: # RHF and GHF
                    basis = basis_new.reshape(-1, nmo)
                    basis_cas_old = basis_old.reshape(-1, nmo).dot(mo_coeff_old[:, mo_idx])
                    mo_coeff_new = find_closest_mo(basis, basis_cas_old,
                                                   return_rotmat=True)[1]
                else: # UHF
                    mo_coeff_new = []
                    for s in range(2):
                        basis = basis_new[s].reshape(-1, nmo)
                        basis_cas_old = basis_old[s].reshape(-1, nmo).dot(mo_coeff_old[s][:, mo_idx[s]])
                        mo_coeff_new.append(find_closest_mo(basis, basis_cas_old,
                                                            return_rotmat=True)[1])
                self.scfsolver.mf.mo_coeff = mo_coeff_new
                self.cisolver.mo_coeff = mo_coeff_new
                t1 = t1_old
                t2 = t2_old
                if not (self.linear or self.approx_l):
                    l1 = l1_old
                    l2 = l2_old

            else: # normal CCSD restart
                if mo_coeff_new.ndim == 2: # RHF and GHF
                    # umat maximally match the basis, C_old U ~ C_new
                    basis_cas_old = basis_old.reshape(-1, nmo).dot(mo_coeff_old[:, mo_idx])
                    basis_cas_new = basis_new.reshape(-1, nmo).dot(mo_coeff_new[:, mo_idx])
                    umat = find_closest_mo(basis_cas_old, basis_cas_new, return_rotmat=True)[1]
                else: # UHF
                    umat = []
                    for s in range(2):
                        basis_cas_old = np.dot(basis_old[s].reshape(-1, nmo),
                                               mo_coeff_old[s][:, mo_idx[s]])
                        basis_cas_new = np.dot(basis_new[s].reshape(-1, nmo),
                                               mo_coeff_new[s][:, mo_idx[s]])
                        umat.append(find_closest_mo(basis_cas_old, basis_cas_new,
                                                    return_rotmat=True)[1])

                t1 = transform_t1_to_bo(t1_old, umat)
                t2 = transform_t2_to_bo(t2_old, umat)
                if not (self.linear or self.approx_l):
                    l1 = transform_l1_to_bo(l1_old, umat)
                    l2 = transform_l2_to_bo(l2_old, umat)
        except np.linalg.LinAlgError:
            log.warn("SVD error catched during matching basis...")
            t1 = None
            t2 = None
            l1 = None
            l2 = None

        if (self.linear or self.approx_l):
            l1 = t1
            l2 = t2
        return t1, t2, l1, l2

    def save_t12_to_h5(self, fcc_name, basis_new, mo_coeff_new):
        """
        Save t1, t2, l1, l2, basis and mo_coeff.
        """
        log.debug(1, "CC solver: dump t and l")
        mo_coeff_new = np.asarray(mo_coeff_new)
        fcc = h5py.File(fcc_name, 'w')
        fcc['mo_coeff'] = mo_coeff_new
        fcc['basis'] = np.asarray(basis_new)
        if mo_coeff_new.ndim == 2:
            fcc['t1'] = np.asarray(self.cisolver.t1)
            fcc['t2'] = np.asarray(self.cisolver.t2)
            if not (self.linear or self.approx_l):
                fcc['l1'] = np.asarray(self.cisolver.l1)
                fcc['l2'] = np.asarray(self.cisolver.l2)
        else:
            spin = mo_coeff_new.shape[0]
            for s in range(spin):
                fcc['t1_%s'%s] = np.asarray(self.cisolver.t1[s])
                if not (self.linear or self.approx_l):
                    fcc['l1_%s'%s] = np.asarray(self.cisolver.l1[s])
            for s in range(spin*(spin+1)//2):
                fcc['t2_%s'%s] = np.asarray(self.cisolver.t2[s])
                if not (self.linear or self.approx_l):
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


if __name__ == '__main__':
    pass
