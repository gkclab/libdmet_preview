#! /usr/bin/env python

"""
SHCI solver.

Author:
    Zhi-Hao Cui <zhcui0408@gmail.com>
    Tianyu Zhu <tyzhu@caltech.edu>
"""
# ZHC TODO
# 1. tolerance
# 2. perturbation in energy, rdm?
# 3. optimized schedule
# 4. restart?
# 5. tmpdir

import os
import subprocess as sub
from tempfile import mkdtemp

import numpy as np
import scipy.linalg as la

from pyscf import gto
from pyscf.cornell_shci import shci
import pyscf.lib.logger as pyscflogger

from libdmet.utils import logger as log
from libdmet.solver import scf
from libdmet.solver.scf import ao2mo_Ham, restore_Ham
from libdmet.basis_transform.make_basis import \
        transform_rdm1_to_ao_mol, transform_rdm2_to_ao_mol 
from libdmet.utils.misc import mdot

class SHCI(object):
    name = "SHCI"
    nnode = 1
    def __init__(self, nproc=1, nnode=1, TmpDir="./tmp", SharedDir=None, \
            restricted=False, Sz=0, bcs=False, ghf=False, tol=2e-9, \
            max_cycle=200, max_memory=40000, compact_rdm2=False, \
            scf_newton=True, mpiprefix=None):
        """
        SHCI solver.
        """
        self.restricted = restricted
        self.Sz = Sz
        self.bcs = bcs
        self.ghf = ghf
        self.conv_tol = tol
        self.max_memory = max_memory

        # mol object
        self.mol = gto.Mole()
        self.mol.incore_anyway = True
        self.mol.max_memory = self.max_memory
        if log.Level[log.verbose] >= log.Level["RESULT"]:
            self.mol.build(verbose=4, dump_input=False)
        else:
            self.mol.build(verbose=2, dump_input=False)

        # SHCI attributes
        self.cisolver = shci.SHCI(mol=self.mol, tol=self.conv_tol)
        self.cisolver.spin = self.Sz
        self.cisolver.config['get_1rdm_csv'] = True
        #self.createTmp(tmp=TmpDir, shared=SharedDir)
        sub.check_call(["mkdir", "-p", TmpDir])
        self.cisolver.runtimedir = TmpDir
        if mpiprefix is not None:
            self.cisolver.mpiprefix = mpiprefix

        #self.scfsolver = scf.SCF(newton_ah=scf_newton)
        self.fcivec = None
        self.onepdm = None
        self.twopdm = None
        self.compact_rdm2 = compact_rdm2 # consider symm of rdm2
        self.optimized = False
    
    def run(self, Ham, nelec=None, guess=None, calc_rdm2=False, \
            restart=False, Mu=None, var_only=True, \
            eps_vars=[1e-4, 5e-5, 1e-5], \
            eps_vars_schedule=[5e-3, 2e-3, 1e-3, 5e-4], \
            get_green=False, w_green=None, n_green=None, **kwargs):
        """
        Main function of the solver.
        """
        log.info("SHCI solver Run")
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
        assert((nelec_a >= 0) and (nelec_b >=0) and (nelec_a + nelec_b == nelec))
        self.nelec = (nelec_a, nelec_b)

        ## first do a mean-field calculation
        #log.debug(1, "SHCI solver: mean-field")
        #dm0 = kwargs.get("dm0", None)
        #scf_max_cycle = kwargs.get("scf_max_cycle", 200)

        #if self.ghf:
        #    log.eassert(nelec_b == 0, "GHF SHCI need all particle alpha spin.")
        #    self.scfsolver.set_system(nelec, 0, False, False, \
        #            max_memory=self.max_memory)
        #    self.scfsolver.set_integral(Ham)
        #    E_HF, rhoHF = self.scfsolver.GGHF(tol=min(1e-9, self.conv_tol*0.1), \
        #            MaxIter=scf_max_cycle, InitGuess=dm0, Mu=Mu)
        #else:
        #    self.scfsolver.set_system(nelec, self.Sz, False, self.restricted, \
        #            max_memory=self.max_memory)
        #    self.scfsolver.set_integral(Ham)
        #    E_HF, rhoHF = self.scfsolver.HF(tol=min(1e-9, self.conv_tol*0.1), \
        #            MaxIter=scf_max_cycle, InitGuess=dm0, Mu=Mu)

        #log.debug(1, "SHCI solver: mean-field converged: %s", self.scfsolver.mf.converged)
        #log.debug(2, "SHCI solver: mean-field rdm1: \n%s", self.scfsolver.mf.make_rdm1())

        #Ham = ao2mo_Ham(Ham, self.scfsolver.mf.mo_coeff)
        #if Ham.restricted: # RHF-SHCI and GHF-SHCI
        #    h1 = Ham.H1["cd"][0]
        #    h2 = Ham.H2["ccdd"][0]
        #else: # UHF-SHCI
        #    h1 = Ham.H1["cd"].copy()
        #    if Mu is not None:
        #        Mu_mat = np.eye(h1.shape[-1])
        #        nao = Mu_mat.shape[-1] // 2 
        #        Mu_mat[range(nao), range(nao)] = -Mu
        #        Mu_mat[range(nao, nao*2), range(nao, nao*2)] = Mu
        #        mo_coeff = self.scfsolver.mf.mo_coeff[0]
        #        Mu_mat = mdot(mo_coeff.conj().T, Mu_mat, mo_coeff)
        #        h1[0] += Mu_mat
        #        
        #    # always convert to the convention of pyscf: aaaa, aabb, bbbb
        #    h2 = Ham.H2["ccdd"][[0, 2, 1]] # ZHC NOTE order
        if spin == 1:
            h1 = Ham.H1["cd"][0]
            h2 = Ham.H2["ccdd"][0]
        else:
            raise NotImplementedError

        self.cisolver.config["eps_vars"] = eps_vars
        self.cisolver.config["eps_vars_schedule"] = eps_vars_schedule
        self.cisolver.config["var_only"] = var_only 
        # Run Green's function
        if get_green:
            self.cisolver.config["get_green"] = get_green
            self.cisolver.config["w_green"] = w_green
            self.cisolver.config["n_green"] = n_green
        E, self.fcivec = self.cisolver.kernel(h1, h2, Ham.norb, self.nelec, \
                ci0=guess, ecore=Ham.H0, restart=False, **kwargs)
        
        self.make_rdm1(Ham)
        # remove the contribution of Mu if exists
        if Mu is not None:
            E -= np.sum(Mu_mat * self.onepdm_mo[0])
        if calc_rdm2:
            self.make_rdm2(Ham)
        
        self.optimized = True
        self.E = E
        #log.info("SHCI solver converged: %s", self.cisolver.converged)
        log.info("SHCI total energy: %s", self.E)
        return self.onepdm, E
    
    def run_dmet_ham(self, Ham, last_aabb=True, **kwargs):
        """
        Run scaled DMET Hamiltonian.
        """
        raise NotImplementedError
        log.info("SHCI solver Run DMET Hamiltonian.")
        log.debug(0, "ao2mo for DMET Hamiltonian.")
        Ham = ao2mo_Ham(Ham, self.scfsolver.mf.mo_coeff, compact=True, in_place=True)
        Ham = restore_Ham(Ham, 1, in_place=True)
        
        # calculate rdm2 in aa, bb, ab order
        self.make_rdm2(Ham)
        if Ham.restricted:
            h1 = Ham.H1["cd"]
            h2 = Ham.H2["ccdd"]
            r1 = self.onepdm_mo
            r2 = self.twopdm_mo
            assert(h1.shape == r1.shape)
            assert(h2.shape == r2.shape)
            
            # energy 
            E1 = np.sum(h1[0].T*r1[0]) * 2.0
            E2 = np.sum(h2[0]*r2[0]) * 0.5
            E = E1 + E2
        else:
            h1 = Ham.H1["cd"]
            # h2 is in aa, bb, ab order
            h2 = Ham.H2["ccdd"]
            r1 = self.onepdm_mo
            # r2 is in aa, bb, ab order
            r2 = self.twopdm_mo
            assert(h1.shape == r1.shape)
            assert(h2.shape == r2.shape)
            
            # energy
            E1 = np.tensordot(h1, r1, axes=((0,1,2), (0,2,1)))
            E2_aa = 0.5 * np.sum(r2[0] * h2[0])
            E2_bb = 0.5 * np.sum(r2[1] * h2[1])
            E2_ab = np.sum(r2[2] * h2[2])
            E = E1 + E2_aa + E2_bb + E2_ab 
        E += Ham.H0
        return E
    
    def make_rdm1(self, Ham):
        log.debug(1, "SHCI solver: solve rdm1")
        onepdm = self.cisolver.make_rdm1(None, Ham.norb, self.nelec)
        if Ham.restricted:
            self.onepdm = (onepdm * 0.5)[np.newaxis]
        else:
            raise NotImplementedError
            onepdm = self.cisolver.make_rdm1s(self.fcivec, Ham.norb, self.nelec)
            self.onepdm_mo = np.asarray(onepdm)
            
        ## rotate back to the AO basis
        #log.debug(1, "SHCI solver: rotate rdm1 to AO")
        #self.onepdm = transform_rdm1_to_ao_mol(self.onepdm_mo, \
        #        self.scfsolver.mf.mo_coeff)
    
    def make_rdm2(self, Ham, ao_repr=False):
        raise NotImplementedError
        log.debug(1, "SHCI solver: solve rdm2")
        if self.ghf:
            self.twopdm_mo = make_rdm2_ghf(self.fcivec, Ham.norb, self.nelec)
        elif Ham.restricted:
            self.twopdm_mo = self.cisolver.make_rdm2(self.fcivec, Ham.norb, \
                    self.nelec)[np.newaxis]
        else:
            self.twopdm_mo = np.asarray(self.cisolver.make_rdm12s(self.fcivec, \
                    Ham.norb, self.nelec)[1])
        
        if ao_repr:
            log.debug(1, "SHCI solver: rotate rdm2 to AO")
            self.twopdm = transform_rdm2_to_ao_mol(self.twopdm_mo, \
                    self.scfsolver.mf.mo_coeff)
        else:
            self.twopdm = None
        
        if not Ham.restricted and not self.ghf:
            self.twopdm_mo = self.twopdm_mo[[0, 2, 1]]
            if self.twopdm is not None:
                self.twopdm = self.twopdm[[0, 2, 1]]

    def onepdm(self):
        log.debug(1, "Compute 1pdm")
        return self.onepdm

    def twopdm(self):
        log.debug(1, "Compute 2pdm")
        return self.twopdm

    def createTmp(self, tmp = "./tmp", shared = None):
        sub.check_call(["mkdir", "-p", tmp])
        self.tmpDir = mkdtemp(prefix = type(self).name, dir = tmp)
        log.info("%s working dir %s", type(self).name, self.tmpDir)
        if type(self).nnode > 1:
            log.eassert(shared is not None, "when running on multiple nodes," \
                    " a shared tmporary folder is required")
            sub.check_call(["mkdir", "-p", shared])
            self.tmpShared = mkdtemp(prefix = type(self).name, dir = shared)
            sub.check_call(type(self).mpipernode + ["mkdir", "-p", self.tmpDir])
            log.info("%s shared dir %s", type(self).name, self.tmpShared)

    def cleanup(self):
        pass
        # FIXME first copy and save restart files
