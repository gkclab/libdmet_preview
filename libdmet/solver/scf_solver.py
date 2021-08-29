#! /usr/bin/env python

"""
SCF impurity solver.
"""

import numpy as np
import scipy.linalg as la

from libdmet.utils import logger as log
from libdmet.solver import scf

class SCFSolver(object):
    def __init__(self, nproc=1, nnode=1, TmpDir="./tmp", SharedDir=None, \
            restricted=False, Sz=0, bcs=False, ghf=False, tol=1e-9, \
            max_cycle=200, level_shift=0.0, max_memory=40000, scf_newton=True):
        """
        HF solver.
        """
        self.restricted = restricted
        self.max_cycle = max_cycle
        self.max_memory = max_memory
        self.conv_tol = tol
        self.level_shift = level_shift
        self.verbose = 5
        self.bcs = bcs
        self.ghf = ghf
        self.Sz = Sz

        self.scfsolver = scf.SCF(newton_ah=scf_newton)
        self.onepdm = None
        self.twopdm = None
    
    def run(self, Ham=None, nelec=None, restart=False, calc_rdm2=False, \
            **kwargs):
        """
        Main function of the solver.
        """
        log.info("HF solver Run")
        spin = Ham.H1["cd"].shape[0]
        if spin > 1:
            assert not self.restricted
        if nelec is None:
            if self.bcs:
                nelec = Ham.norb * 2
            else:
                nelec = Ham.norb
        
        nelec_a = (nelec + self.Sz) // 2
        nelec_b = (nelec - self.Sz) // 2
        assert (nelec_a >= 0) and (nelec_b >=0) and (nelec_a + nelec_b == nelec)
        
        log.debug(1, "HF solver: mean-field")
        dm0 = kwargs.get("dm0", None)
        scf_max_cycle = kwargs.get("scf_max_cycle", 200)

        self.scfsolver.set_system(nelec, self.Sz, False, \
                self.restricted, max_memory=self.max_memory)
        self.scfsolver.set_integral(Ham)
        
        E_HF, rhoHF = self.scfsolver.HF(tol=self.conv_tol, \
                MaxIter=scf_max_cycle, InitGuess=dm0)
        log.debug(1, "HF solver: mean-field converged: %s", \
                self.scfsolver.mf.converged)
        
        self.onepdm = rhoHF
        E = E_HF
        return self.onepdm, E
    
    def run_dmet_ham(self, Ham, last_aabb=True, save_dmet_ham=False, \
            dmet_ham_fname='dmet_ham.h5', use_calculated_twopdm=False, \
            ao_repr=False, **kwargs):
        """
        Run scaled DMET Hamiltonian.
        """
        log.info("mf solver Run DMET Hamiltonian.")
        if not ao_repr:
            log.info("Use MO representation.")
            Ham = scf.ao2mo_Ham(Ham, self.scfsolver.mf.mo_coeff, compact=True, in_place=True)
        Ham = scf.restore_Ham(Ham, 1, in_place=True)

        if use_calculated_twopdm:
            log.info("Using exisiting twopdm in MO basis...")
        else:
            self.make_rdm2(ao_repr=ao_repr)
        
        if ao_repr:
            r1 = self.onepdm
            r2 = self.twopdm
        else:
            r1 = self.onepdm_mo
            r2 = self.twopdm_mo

        if Ham.restricted:
            h1 = Ham.H1["cd"]
            h2 = Ham.H2["ccdd"]
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape
            
            # energy
            E1 = np.einsum('pq, qp', h1[0], r1[0]) * 2.0
            E2 = np.einsum('pqrs, pqrs', h2[0], r2[0]) * 0.5
            E = E1 + E2
        else:
            h1 = Ham.H1["cd"]
            h2 = Ham.H2["ccdd"]
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape
            
            # energy
            E1 = np.einsum('spq, sqp', h1, r1)
            E2_aa = 0.5 * np.einsum('pqrs, pqrs', r2[0], h2[0])
            E2_bb = 0.5 * np.einsum('pqrs, pqrs', r2[1], h2[1])
            E2_ab = np.einsum('pqrs, pqrs', r2[2], h2[2])
            E = E1 + E2_aa + E2_bb + E2_ab 
        E += Ham.H0
        
        if save_dmet_ham:
            fdmet_ham = h5py.File(dmet_ham_fname, 'w')
            fdmet_ham['H1'] = h1
            fdmet_ham['H2'] = h2
            fdmet_ham.close()
        
        return E
    
    def make_rdm1(self):
        return self.onepdm

    def make_rdm2(self, ao_repr=True):
        """
        Compute rdm2.
        NOTE: the returned value's spin order for H2 is aa, bb, ab.
        """
        if ao_repr:
            if self.restricted:
                self.twopdm = (4.0 * np.einsum('qp, sr -> pqrs', self.onepdm[0], \
                               self.onepdm[0]) \
                           - 2.0 * np.einsum('sp, qr -> pqrs', self.onepdm[0], \
                               self.onepdm[0]))[None]
            else:
                rdm2_aa = np.einsum('qp, sr -> pqrs', self.onepdm[0], \
                        self.onepdm[0]) \
                        - np.einsum('sp, qr -> pqrs', self.onepdm[0], \
                        self.onepdm[0])
                rdm2_bb = np.einsum('qp, sr -> pqrs', self.onepdm[1], \
                        self.onepdm[1]) \
                        - np.einsum('sp, qr -> pqrs', self.onepdm[1], \
                        self.onepdm[1])
                rdm2_ab = np.einsum('qp, sr -> pqrs', self.onepdm[0], \
                        self.onepdm[1])
                self.twopdm = np.asarray((rdm2_aa, rdm2_bb, rdm2_ab))
            return self.twopdm
        else:
            if self.restricted:
                onepdm = np.diag(self.scfsolver.mf.mo_occ)
                self.onepdm_mo = (onepdm * 0.5)[None]
                self.twopdm_mo = (np.einsum('qp, sr -> pqrs', onepdm, onepdm) \
                           - 0.5 * np.einsum('sp, qr -> pqrs', onepdm, onepdm))[None]
            else:
                onepdm_a = np.diag(self.scfsolver.mf.mo_occ[0])
                onepdm_b = np.diag(self.scfsolver.mf.mo_occ[1])
                self.onepdm_mo = np.asarray((onepdm_a, onepdm_b))
                rdm2_aa = np.einsum('qp, sr -> pqrs', onepdm_a, onepdm_a) \
                        - np.einsum('sp, qr -> pqrs', onepdm_a, onepdm_a)
                rdm2_bb = np.einsum('qp, sr -> pqrs', onepdm_b, onepdm_b) \
                        - np.einsum('sp, qr -> pqrs', onepdm_b, onepdm_b)
                rdm2_ab = np.einsum('qp, sr -> pqrs', onepdm_a, onepdm_b)
                self.twopdm_mo = np.asarray((rdm2_aa, rdm2_bb, rdm2_ab))
            return self.twopdm_mo

    def onepdm(self):
        log.debug(1, "Compute 1pdm")
        return self.onepdm

    def twopdm(self):
        log.debug(1, "Compute 2pdm")
        return self.twopdm
