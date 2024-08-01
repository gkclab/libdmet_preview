#! /usr/bin/env python

"""
SHCI solver.

Author:
    Zhi-Hao Cui <zhcui0408@gmail.com>
    Tianyu Zhu <tyzhu@caltech.edu>
"""
# ZHC TODO
# uhf based integrals
# perturbation in rdm
# rdm2
# optimized schedule
# restart

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
from libdmet.basis_transform.make_basis import (transform_rdm1_to_ao_mol, 
                                                transform_rdm2_to_ao_mol)
from libdmet.utils.misc import mdot

class SHCI(object):
    """
    SHCI solver.
    """

    name = "SHCI"
    nnode = 1

    def __init__(self, nproc=1, nnode=1, TmpDir="./tmp", SharedDir=None, 
                 restricted=False, Sz=0, bcs=False, ghf=False, tol=1e-8, 
                 max_cycle=200, max_memory=40000, compact_rdm2=False,
                 scf_newton=False, mpiprefix=None, alpha=None, beta=np.inf,
                 **kwargs):
        """
        SHCI solver.
        """
        self.restricted = restricted
        self.Sz = Sz
        self.bcs = bcs
        self.ghf = ghf
        self.alpha = alpha
        self.beta = beta
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
        sub.check_call(["mkdir", "-p", TmpDir])
        self.cisolver.runtimedir = TmpDir
        if mpiprefix is not None:
            self.cisolver.mpiprefix = mpiprefix
        self.scfsolver = scf.SCF(newton_ah=scf_newton)

        self.fcivec = None
        self.onepdm = None
        self.twopdm = None
        self.compact_rdm2 = compact_rdm2 
        self.optimized = False
        self.count = 0

    def run(self, Ham, nelec=None, guess=None, calc_rdm2=False, restart=False,
            Mu=None, var_only=True, eps_vars=[2e-4, 1e-4, 5e-5], 
            eps_vars_schedule=[2e-3, 1e-3, 5e-4], 
            get_green=False, w_green=None, n_green=None, **kwargs):
        """
        Main function of the solver.
        Mu is a possible chemical potential adding to the h1e[0]
        to simulate GHF type calculation.
        """
        log.info("SHCI solver Run")
        spin = Ham.H1["cd"].shape[0]
        if spin > 1:
            assert not self.restricted
        if nelec is None:
            if self.bcs:
                nelec = Ham.norb * 2
            elif self.ghf:
                nelec = Ham.norb // 2
            else:
                raise ValueError
        if self.ghf:
            self.Sz = nelec
        nelec_a = (nelec + self.Sz) // 2
        nelec_b = (nelec - self.Sz) // 2
        assert (nelec_a >= 0) and (nelec_b >=0) and (nelec_a + nelec_b == nelec)
        self.nelec = (nelec_a, nelec_b)

        # first do a mean-field calculation
        log.debug(1, "SHCI solver: mean-field")
        dm0 = kwargs.get("dm0", None)
        scf_max_cycle = kwargs.get("scf_max_cycle", 200)

        if self.ghf:
            log.eassert(nelec_b == 0, "GHF SHCI need all particle alpha spin.")
            self.scfsolver.set_system(nelec, 0, False, False, 
                                      max_memory=self.max_memory)
            self.scfsolver.set_integral(Ham)
            E_HF, rhoHF = self.scfsolver.GGHF(tol=min(1e-10, self.conv_tol*0.1),
                                              MaxIter=scf_max_cycle,
                                              InitGuess=dm0, Mu=Mu,
                                              alpha=self.alpha,
                                              beta=self.beta)
        else:
            self.scfsolver.set_system(nelec, self.Sz, False, self.restricted,
                                      max_memory=self.max_memory)
            self.scfsolver.set_integral(Ham)
            E_HF, rhoHF = self.scfsolver.HF(tol=min(1e-10, self.conv_tol*0.1),
                                            MaxIter=scf_max_cycle,
                                            InitGuess=dm0, Mu=Mu, 
                                            alpha=self.alpha,
                                            beta=self.beta)

        log.debug(1, "SHCI solver: mean-field converged: %s",
                  self.scfsolver.mf.converged)

        if "mo_energy_custom" in kwargs:
            self.scfsolver.mf.mo_energy = kwargs["mo_energy_custom"]
        if "mo_occ_custom" in kwargs:
            self.scfsolver.mf.mo_occ = kwargs["mo_occ_custom"]
        if "mo_coeff_custom" in kwargs:
            log.info("Use customized MO as reference.")
            self.scfsolver.mf.mo_coeff = kwargs["mo_coeff_custom"]
            self.scfsolver.mf.e_tot = self.scfsolver.mf.energy_tot()

        log.debug(2, "SHCI solver: mean-field rdm1: \n%s",
                  self.scfsolver.mf.make_rdm1())
        
        # ZHC NOTE directly write the integrals
        self.cisolver.cleanup(remove_wf=True)
        Ham = ao2mo_Ham(Ham, self.scfsolver.mf.mo_coeff)
        Ham.dump(filename="%s/%s"%(self.cisolver.runtimedir,
                 self.cisolver.integralfile), fmt="FCIDUMP",
                 thr=min(eps_vars[-1], self.conv_tol, 1e-10) * 0.1,
                 buffered_io=True)
        h1 = None
        h2 = None
        
        #if Ham.restricted: # RHF-SHCI and GHF-SHCI
        #    h1 = Ham.H1["cd"][0]
        #    h2 = Ham.H2["ccdd"][0]
        #    if Mu is not None:
        #        if self.ghf:
        #            ovlp = self.scfsolver.mf.get_ovlp()
        #            nso = self.scfsolver.mf.mo_coeff.shape[-2]
        #            nao = nso // 2
        #            Mu_mat = np.zeros((nso, nso))
        #            Mu_mat[:nao, :nao] = ovlp[:nao, :nao] * (-Mu)
        #            Mu_mat[nao:, nao:] = ovlp[nao:, nao:] * (Mu)
        #            mo_coeff = self.scfsolver.mf.mo_coeff
        #            Mu_mat = mdot(mo_coeff.conj().T, Mu_mat, mo_coeff)
        #            h1 = np.array(h1, copy=True)
        #            h1 += Mu_mat
        #        else:
        #            raise NotImplementedError
        #else: # UHF-SHCI
        #    h1 = Ham.H1["cd"].copy()
        #    if Mu is not None:
        #        ovlp = self.scfsolver.mf.get_ovlp()
        #        nso = self.scfsolver.mf.mo_coeff.shape[-2]
        #        nao = nso // 2
        #        Mu_mat = np.zeros((nso, nso))
        #        Mu_mat[:nao, :nao] = ovlp[:nao, :nao] * (-Mu)
        #        Mu_mat[nao:, nao:] = ovlp[nao:, nao:] * (Mu)
        #        mo_coeff = self.scfsolver.mf.mo_coeff[0]
        #        Mu_mat = mdot(mo_coeff.conj().T, Mu_mat, mo_coeff)
        #        h1[0] += Mu_mat
        #        
        #    # always convert to the convention of pyscf: aaaa, aabb, bbbb
        #    h2 = Ham.H2["ccdd"][[0, 2, 1]] # ZHC NOTE order

        self.cisolver.config["eps_vars"] = eps_vars
        self.cisolver.config["eps_vars_schedule"] = eps_vars_schedule
        self.cisolver.config["var_only"] = var_only 
        # Run Green's function
        if get_green:
            self.cisolver.config["get_green"] = get_green
            self.cisolver.config["w_green"] = w_green
            self.cisolver.config["n_green"] = n_green
        E, self.fcivec = self.cisolver.kernel(h1, h2, Ham.norb, self.nelec,
                                              ci0=guess, ecore=Ham.H0, restart=False,
                                              **kwargs)
        
        self.make_rdm1(Ham)
        # remove the contribution of Mu if exists
        if Mu is not None:
            if self.ghf:
                E -= np.einsum('pq, qp', Mu_mat, self.onepdm_mo)
            else:
                E -= np.einsum('pq, qp', Mu_mat, self.onepdm_mo[0])
        if calc_rdm2:
            self.make_rdm2(Ham)
        
        self.optimized = True
        self.E = E
        log.info("SHCI total energy: %s", self.E)
        os.rename("%s/config.json"%self.cisolver.runtimedir, 
                  "%s/config.json_%03d"%(self.cisolver.runtimedir, self.count))
        os.rename("%s/result.json"%self.cisolver.runtimedir, 
                  "%s/result.json_%03d"%(self.cisolver.runtimedir, self.count))
        os.rename("%s/output.dat"%self.cisolver.runtimedir, 
                  "%s/output.dat_%03d"%(self.cisolver.runtimedir, self.count))
        os.rename("%s/1rdm.csv"%self.cisolver.runtimedir, 
                  "%s/1rdm.csv_%03d"%(self.cisolver.runtimedir, self.count))
        self.count += 1
        return self.onepdm, E
    
    def run_dmet_ham(self, Ham, last_aabb=True, **kwargs):
        """
        Run scaled DMET Hamiltonian.
        """
        log.info("SHCI solver Run DMET Hamiltonian.")
        log.debug(0, "ao2mo for DMET Hamiltonian.")
        Ham = ao2mo_Ham(Ham, self.scfsolver.mf.mo_coeff, compact=True,
                        in_place=True)
        Ham = restore_Ham(Ham, 1, in_place=True)
        
        # calculate rdm2 in aa, bb, ab order
        self.make_rdm2(Ham)
        if self.ghf:
            h1 = Ham.H1["cd"][0]
            h2 = Ham.H2["ccdd"][0]
            r1 = self.onepdm_mo
            r2 = self.twopdm_mo
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape
            
            E1 = np.einsum('pq, qp', h1, r1)
            E2 = np.einsum('pqrs, pqrs', h2, r2) * 0.5
        elif Ham.restricted:
            h1 = Ham.H1["cd"]
            h2 = Ham.H2["ccdd"]
            r1 = self.onepdm_mo
            r2 = self.twopdm_mo
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape
            
            E1 = np.einsum('pq, qp', h1[0], r1[0]) * 2.0
            E2 = np.einsum('pqrs, pqrs', h2[0], r2[0]) * 0.5
        else:
            h1 = Ham.H1["cd"]
            # h2 is in aa, bb, ab order
            h2 = Ham.H2["ccdd"]
            r1 = self.onepdm_mo
            # r2 is in aa, bb, ab order
            r2 = self.twopdm_mo
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape
            
            E1 = np.einsum('spq, sqp', h1, r1)
            
            E2_aa = 0.5 * np.einsum('pqrs, pqrs', h2[0], r2[0])
            E2_bb = 0.5 * np.einsum('pqrs, pqrs', h2[1], r2[1])
            E2_ab = np.einsum('pqrs, pqrs', h2[2], r2[2])
            E2 = E2_aa + E2_bb + E2_ab
        
        E = E1 + E2
        E += Ham.H0
        log.debug(0, "run DMET Hamiltonian:\nE0 = %20.12f, E1 = %20.12f, " 
                  "E2 = %20.12f, E = %20.12f", Ham.H0, E1, E2, E)
        return E
    
    def make_rdm1(self, Ham):
        log.debug(1, "SHCI solver: solve rdm1")
        self.onepdm_mo = self.cisolver.make_rdm1(None, Ham.norb, self.nelec)
        if self.ghf: # GHF
            pass
        elif Ham.restricted:
            self.onepdm_mo = (self.onepdm_mo * 0.5)[np.newaxis]
        
        # rotate back to the AO basis
        log.debug(1, "SHCI solver: rotate rdm1 to AO")
        self.onepdm = transform_rdm1_to_ao_mol(self.onepdm_mo, 
                                               self.scfsolver.mf.mo_coeff)
    
    def make_rdm2(self, Ham, ao_repr=False):
        # ZHC NOTE I think there may be a bug in the rdm2 code of Arrow.
        log.debug(1, "SHCI solver: solve rdm2")
        _, self.twopdm_mo = self.cisolver.make_rdm12(None, Ham.norb, self.nelec)
        
        if self.ghf:
            self.twopdm_mo = np.sum(self.twopdm_mo, axis=0)
        elif Ham.restricted:
            self.twopdm_mo = self.twopdm_mo[np.newaxis]
        else:
            raise NotImplementedError
        
        if ao_repr:
            log.debug(1, "SHCI solver: rotate rdm2 to AO")
            self.twopdm = transform_rdm2_to_ao_mol(self.twopdm_mo, 
                                                   self.scfsolver.mf.mo_coeff)
        else:
            self.twopdm = None
        
        # ZHC NOTE change to aa, bb, ab
        if not Ham.restricted and not self.ghf:
            self.twopdm_mo = self.twopdm_mo[[0, 2, 1]]
            if self.twopdm is not None:
                self.twopdm = self.twopdm[[0, 2, 1]]
    
    def onepdm(self):
        return self.onepdm

    def twopdm(self):
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
