#! /usr/bin/env python

"""
DQMC solver interface.

Author:
    Ankit Mahajan <ankitmahajan76@gmail.com>
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

import h5py
import os
import subprocess as sub
from tempfile import mkdtemp

import numpy as np
import scipy.linalg as la

from pyscf import gto

from libdmet.utils import logger as log
from libdmet.solver import scf
from libdmet.solver.scf import ao2mo_Ham, restore_Ham
from libdmet.basis_transform import make_basis 

from libdmet.utils import cholesky
from libdmet.utils.misc import mdot
from libdmet.solver.settings import DQMC_PATH, DQMC_BLOCKING

def write_ints_dqmc_uhf(hcore, hcore_mod, chol, nelec, nmo, enuc, ms=0,
                    filename='FCIDUMP_chol'):
    with h5py.File(filename, 'w') as fh5:
        fh5['header'] = np.array([nelec, nmo, ms, chol[0].shape[0]])
        fh5['hcore_up'] = hcore[0].ravel()
        fh5['hcore_dn'] = hcore[1].ravel()
        fh5['hcore_mod_up'] = hcore_mod[0].ravel()
        fh5['hcore_mod_dn'] = hcore_mod[1].ravel()
        fh5['chol_up'] = chol[0].ravel()
        fh5['chol_dn'] = chol[1].ravel()
        fh5['energy_core'] = enuc

def make_ints_dqmc_uhf(ImpHam, nelec, tol=1e-7, filename="FCIDUMP_chol"):
    norb = ImpHam.norb
    h1 = ImpHam.H1['cd']
    eri = ImpHam.H2['ccdd']
    enuc = float(ImpHam.H0)
    chol = cholesky.get_cderi_uhf(eri, norb, tol=tol) 
    nchol = chol.shape[1]

    # writing dqmc ints
    v0_up = 0.5 * np.einsum('nik, njk -> ij', chol[0], chol[0], optimize=True)
    v0_dn = 0.5 * np.einsum('nik, njk -> ij', chol[1], chol[1], optimize=True)
    h1_mod = [h1[0] - v0_up, h1[1] - v0_dn]
    chol_flat = [chol[0].reshape(nchol, -1), chol[1].reshape(nchol, -1)]
    write_ints_dqmc_uhf(h1, h1_mod, chol_flat, sum(nelec), norb, enuc, 
                        ms=nelec[0] - nelec[1], filename=filename)

def write_ints_dqmc_ghf(hcore, hcore_mod, chol, nelec, nmo, enuc, ms=0,
                        filename='FCIDUMP_chol'):
    with h5py.File(filename, 'w') as fh5:
        fh5['header'] = np.array([nelec, nmo, ms, chol.shape[0]])
        fh5['hcore'] = hcore.ravel()
        fh5['hcore_mod'] = hcore_mod.ravel()
        fh5['chol'] = chol.ravel()
        fh5['energy_core'] = enuc

def make_ints_dqmc_ghf(ImpHam, nelec, tol=1e-7, filename="FCIDUMP_chol"):
    norb = ImpHam.norb
    h1 = ImpHam.H1['cd'][0]
    eri = ImpHam.H2['ccdd'][0]
    enuc = float(ImpHam.H0)
    chol = cholesky.get_cderi_rhf(eri, norb, tol=tol)
    nchol = chol.shape[0]
  
    # writing dqmc ints
    v0 = 0.5 * np.einsum('nik, njk -> ij', chol, chol, optimize=True)
    h1_mod = h1 - v0
    chol_flat = chol.reshape(nchol, -1)
    write_ints_dqmc_ghf(h1, h1_mod, chol_flat, sum(nelec), norb, enuc, 
                        ms=nelec[0] - nelec[1], filename=filename)

def write_input_dqmc(int_type, dt=0.005, nsteps=50, ndets=100, fname='dqmc.json',
                     integrals="FCIDUMP_chol", left="uhf", right="uhf",
                     determinants="dets.bin",
                     nwalk=60, choleskyThreshold=1e-3, orthoSteps=20,
                     stochasticIter=500, prefix="./"):
    dqmc_input =  \
"""
{
  "system":
  {
    "intType": "%s",
    "integrals": "%s"
  },
  "wavefunction":
  {
    "left":  "%s",
    "right": "%s",
    "determinants": "%s",
    "ndets": %i
  },
  "sampling":
  {
    "seed": %i,
    "phaseless": true,
    "dt": %f,
    "nsteps": %i,
    "nwalk": %d,
    "choleskyThreshold": %s,
    "orthoSteps": %s,
    "stochasticIter": %s
  },
  "print":
  {
    "writeOneRDM": true,
    "scratchDir": "%s"
  }
}
"""%(int_type, integrals, left, right, determinants, ndets, 
     np.random.randint(1, 1e6), dt, nsteps, nwalk,
     choleskyThreshold, orthoSteps, stochasticIter, prefix)
    with open(fname, "w") as f:
        f.write(dqmc_input)

def write_coeff(fname, mo_coeff):
    if isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 2:
        np.savetxt(fname + ".txt", mo_coeff)
    else:
        norb = mo_coeff[0].shape[-1]
        coeff = np.empty((norb, norb * 2))
        coeff[:, :norb] = mo_coeff[0]
        coeff[:, norb:] = mo_coeff[1]
        np.savetxt(fname + ".txt", coeff)

# ****************************************************************************
# Dice
# ****************************************************************************

def write_conf_dice(fname, nelec, prefix="./", orbitals="./FCIDUMP",
                    ndets=10000, eps_var=1e-5, tol_davidson=5e-5, tol=1e-6,
                    maxiter=6):
    f = open(fname, 'w')
    f.write("#system \n")
    f.write("nocc %d \n" % sum(nelec))
    for i in range(nelec[0]):
        f.write("%d " % (2 * i))
    for i in range(nelec[1]):
        f.write("%d " % (2 * i + 1))
    #for i in range(nelec[0] + nelec[1]):
    #    f.write("%d " % i)
    
    f.write("\n")
    f.write("end \n")
    f.write("orbitals %s \n" % orbitals)
    f.write("nroots 1 \n")
    f.write("\n")
    f.write("#variational \n")
    f.write("schedule \n")
    f.write("0 %g \n" % eps_var)
    f.write("end \n")
    f.write("davidsonTol %g \n" % tol_davidson)
    f.write("dE %g \n" % tol)
    f.write("maxiter %d \n" % maxiter)
    pt_string = \
"""
#pt
nPTiter 0
epsilon2 1e-07
epsilon2Large 1000
targetError 0.0001
sampleN 200

#misc
io 
prefix %s
""" % prefix
    f.write(pt_string)
    f.write("readText \n")
    f.write("writebestdeterminants %d \n" % ndets)
    #f.write("DoSpinRDM \n")
    string = \
    f.close()

class DQMC(object):
    """
    DQMC solver.
    """

    name = "DQMC"
    nnode = 1

    def __init__(self, nproc=1, nnode=1, TmpDir="./tmp", SharedDir=None, 
                 restricted=False, Sz=0, bcs=False, ghf=False, tol=1e-6, 
                 max_cycle=200, max_memory=40000, compact_rdm2=False,
                 scf_newton=False, mpiprefix=None, alpha=None, beta=np.inf,
                 **kwargs):
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

        sub.check_call(["mkdir", "-p", TmpDir])
        self.tmp_dir = self.TmpDir = TmpDir
        self.scfsolver = scf.SCF(newton_ah=scf_newton)

        self.fcivec = None
        self.onepdm = None
        self.twopdm = None
        self.compact_rdm2 = compact_rdm2 
        self.optimized = False
        
        self.nnode = nnode
        self.nproc = nproc
        self.count = 0

    @property
    def mpipernode(self):
        return ["mpirun", "-np", "%d"%(self.nnode * self.nproc)]
    
    def run(self, Ham, nelec=None, guess=None, calc_rdm2=False, restart=False,
            Mu=None, dt=0.005, nsteps=50, ndets=50, fname='dqmc.json',
            integrals="FCIDUMP_chol", left=None, right=None,
            nwalk=None, nwalk_tot=1000, choleskyThreshold=1e-3, orthoSteps=20,
            stochasticIter=300, **kwargs):
        """
        Main function of the solver.
        Mu is a possible chemical potential adding to the h1e[0]
        to simulate GHF type calculation.

        nwalk_tot is the total number of walkers.
        """
        log.info("DQMC solver Run")
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
        log.debug(1, "DQMC solver: mean-field")
        dm0 = kwargs.get("dm0", None)
        scf_max_cycle = kwargs.get("scf_max_cycle", 200)

        if self.ghf:
            log.eassert(nelec_b == 0, "GHF DQMC need all particle alpha spin.")
            self.scfsolver.set_system(nelec, 0, False, False, 
                                      max_memory=self.max_memory)
            self.scfsolver.set_integral(Ham)
            E_HF, rhoHF = self.scfsolver.GGHF(tol=min(1e-10, self.conv_tol*0.1),
                                              MaxIter=scf_max_cycle,
                                              InitGuess=dm0, Mu=Mu,
                                              alpha=self.alpha,
                                              beta=self.beta)
            int_type = "g"
        else:
            self.scfsolver.set_system(nelec, self.Sz, False, self.restricted,
                                      max_memory=self.max_memory)
            self.scfsolver.set_integral(Ham)
            E_HF, rhoHF = self.scfsolver.HF(tol=min(1e-10, self.conv_tol*0.1),
                                            MaxIter=scf_max_cycle,
                                            InitGuess=dm0, Mu=Mu, 
                                            alpha=self.alpha,
                                            beta=self.beta)
            if self.restricted:
                int_type = 'r'
            else:
                int_type = 'u'
        if left is None:
            left = "%shf"%int_type
        if right is None:
            right = "%shf"%int_type

        log.debug(1, "DQMC solver: mean-field converged: %s",
                  self.scfsolver.mf.converged)

        if "mo_energy_custom" in kwargs:
            self.scfsolver.mf.mo_energy = kwargs["mo_energy_custom"]
        if "mo_occ_custom" in kwargs:
            self.scfsolver.mf.mo_occ = kwargs["mo_occ_custom"]
        if "mo_coeff_custom" in kwargs:
            log.info("Use customized MO as reference.")
            self.scfsolver.mf.mo_coeff = kwargs["mo_coeff_custom"]
            self.scfsolver.mf.e_tot = self.scfsolver.mf.energy_tot()

        log.debug(2, "DQMC solver: mean-field rdm1: \n%s",
                  self.scfsolver.mf.make_rdm1())
        
        fname = "%s/%s.%03d"%(self.TmpDir, fname, self.count)
        fints = "%s/%s"%(self.TmpDir, integrals)

        if nwalk is None:
            nwalk = max(int(nwalk_tot / (self.nnode * self.nproc) + 0.5), 1)
        

        write_input_dqmc(int_type, dt=dt, nsteps=nsteps, ndets=ndets, 
                         fname=fname,
                         integrals=fints, 
                         left=left, right=right,
                         nwalk=nwalk, choleskyThreshold=choleskyThreshold,
                         orthoSteps=orthoSteps, stochasticIter=stochasticIter,
                         prefix=self.TmpDir)
        
        Ham_mo = ao2mo_Ham(Ham, self.scfsolver.mf.mo_coeff)
        
        # SHCI prepare
        if left == "multislater":
            log.info("shci prepare for multislater.")
            if self.ghf:
                raise NotImplementedError

            f_shci = self.TmpDir + "/dice.conf.%03d" % self.count
            fints_shci = self.TmpDir + "/FCIDUMP"
            
            from libdmet.system.integral import dumpFCIDUMP_as_ghf
            #dumpFCIDUMP_as_ghf(fints_shci, Ham_mo, thr=min(self.conv_tol * 0.01, 1e-7), 
            #                   buffered_io=False, nelec=self.nelec, spin=None,
            #                   dump_as_complex=True, aabb=False)
            #Ham_mo.dump(fints_shci, nelec=self.nelec, dump_as_complex=True)
            write_conf_dice(f_shci, self.nelec, prefix=self.TmpDir, orbitals=fints_shci,
                            ndets=10000, eps_var=1e-5, tol_davidson=5e-5, tol=1e-6,
                            maxiter=6)

            import QMCUtils
            ham_ints = {'enuc': Ham_mo.H0, 'h1': Ham_mo.H1["cd"], 'eri': Ham_mo.H2["ccdd"]}
            QMCUtils.write_hci_ghf_uhf_integrals(ham_ints, Ham_mo.norb, sum(self.nelec), filename=fints_shci)
        
        if self.ghf:
            make_ints_dqmc_ghf(Ham_mo, self.nelec, tol=min(self.conv_tol * 0.01, 1e-7), 
                               filename=fints)
        else:
            if self.restricted:
                raise NotImplementedError
            else:
                make_ints_dqmc_uhf(Ham_mo, self.nelec, tol=min(self.conv_tol * 0.01, 1e-7), 
                                   filename=fints)

        Ham_mo = None
        # coefficient should be in the basis of hamiltonian
        mo = make_basis.get_mo_ovlp(self.scfsolver.mf.mo_coeff, self.scfsolver.mf.mo_coeff,
                                    self.scfsolver.mf.get_ovlp())
        # SHCI run
        if left == "multislater":
            from libdmet.solver.settings import DICE_PATH
            log.info("shci run for multislater.")
            outputfile = os.path.join(self.TmpDir, "dice.out.%03d" % self.count)
            log.info("%s call No. %d", self.name, self.count)
            log.debug(0, "Written to file %s", outputfile)
            with open(outputfile, "w", buffering=1) as f:
                log.debug(1, "normal environment, mpirun used.")
                cmd = [*self.mpipernode, DICE_PATH, f_shci]
                log.debug(1, " ".join(cmd))
                sub.check_call(cmd, stdout=f)
        
        if self.ghf:
            write_coeff("ghf", mo)
        else:
            if self.restricted:
                raise NotImplementedError
            else:
                write_coeff("uhf", mo)

        outputfile = os.path.join(self.TmpDir, "dqmc.out.%03d" % self.count)
        log.info("%s call No. %d", self.name, self.count)
        log.debug(0, "Written to file %s", outputfile)
        with open(outputfile, "w", buffering=1) as f:
            log.debug(1, "normal environment, mpirun used.")
            cmd = [*self.mpipernode, DQMC_PATH, fname]
            log.debug(1, " ".join(cmd))
            sub.check_call(cmd, stdout=f)
        
        # compute average energy
        outputfile = os.path.join(self.TmpDir, "blocking.out.%03d" % self.count)
        with open(outputfile, "w", buffering=1) as f:
            cmd = ["python", DQMC_BLOCKING, "samples.dat", "50"]
            log.debug(1, " ".join(cmd))
            sub.check_call(cmd, stdout=f)
            os.rename("samples.dat", 
                      os.path.join(self.TmpDir, "samples.dat.%03d" % self.count))
            os.rename("%s.txt" % ("%shf"%int_type), 
                      os.path.join(self.TmpDir, "%s.txt.%03d" % ("%shf"%int_type, self.count)))
        
        with open(outputfile, "r") as f:
            lines = f.readlines()
            E = float(lines[1].split()[-1])
        
        if left == "multislater": 
            self.make_rdm1(Ham, ndets=ndets)
        else:
            self.make_rdm1(Ham, ndets=0)

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
        log.info("DQMC total energy: %s", self.E)
        self.count += 1
        
        return self.onepdm, E
    
    def run_dmet_ham(self, Ham, last_aabb=True, **kwargs):
        """
        Run scaled DMET Hamiltonian.
        """
        log.info("DQMC solver Run DMET Hamiltonian.")
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
    
    def make_rdm1(self, Ham, ndets=0, extrap=False, hermi=True):
        """
        hermi: if True, will hermitize D = (D + D.conj().T) * 0.5
        extrap: if True, will use mixed rdm1. D = D * 2 - D_mf
        """
        log.debug(1, "DQMC solver: solve rdm1")
        if self.ghf:
            rdm1 = 0.0
            cols = np.arange(Ham.norb)
            
            weight_tot = 0.0
            for i in range(self.nproc):
                filename = self.TmpDir + "/rdm_%d.dat"%i
                with open(filename) as fh:
                    weight = float(fh.readline())
                df = np.loadtxt(filename, skiprows=1)
                rdm1 += df * weight
                weight_tot += weight

            rdm1 /= weight_tot
            self.onepdm_mo = rdm1
        else:
            if self.restricted:
                raise NotImplementedError
            else:
                rdm1_a = 0.0
                rdm1_b = 0.0
                cols = np.arange(Ham.norb)
                
                weight_tot = 0.0
                for i in range(self.nproc):
                    filename = self.TmpDir + "/rdm_up_%d.dat"%i
                    with open(filename) as fh:
                        weight = float(fh.readline())
                    df = np.loadtxt(filename, skiprows=1)
                    rdm1_a += df * weight
                    weight_tot += weight
                    
                    filename = self.TmpDir + "/rdm_dn_%d.dat"%i
                    with open(filename) as fh:
                        weight = float(fh.readline())
                    df = np.loadtxt(filename, skiprows=1)
                    rdm1_b += df * weight

                rdm1 = np.asarray((rdm1_a, rdm1_b))
                rdm1 /= weight_tot
                self.onepdm_mo = rdm1
        
        if self.ghf: # GHF
            pass
        elif Ham.restricted:
            self.onepdm_mo = (self.onepdm_mo * 0.5)[np.newaxis]
        
        if hermi:
            self.onepdm_mo = 0.5 * (self.onepdm_mo + np.swapaxes(self.onepdm_mo.conj(), -1, -2))

        # extrapolated rdm1
        if extrap:
            if ndets == 0:
                rdm1_mf = self.scfsolver.mf.make_rdm1()
                rdm1_mf = make_basis.transform_rdm1_to_mo_mol(rdm1_mf, self.scfsolver.mf.mo_coeff,
                                                              self.scfsolver.mf.get_ovlp()[0])
                self.onepdm_mo = self.onepdm_mo * 2 - rdm1_mf
            else:
                import QMCUtils
                norb_dice, state_dice = QMCUtils.read_dets(ndets=ndets)
                rdm1_mf = QMCUtils.calculate_ci_1rdm(norb_dice, state_dice, ndets=ndets)
                rdm1_mf = np.asarray(rdm1_mf)
                self.onepdm_mo = self.onepdm_mo * 2 - rdm1_mf

        # rotate back to the AO basis
        log.debug(1, "DQMC solver: rotate rdm1 to AO")
        self.onepdm = make_basis.transform_rdm1_to_ao_mol(self.onepdm_mo, 
                                                          self.scfsolver.mf.mo_coeff)
        
    def make_rdm2(self, Ham, ao_repr=False):
        raise NotImplementedError
    
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
