#! /usr/bin/env python

"""
Test spinless formalism for molecules.
"""

import os
import numpy as np
import scipy.linalg as la

import pytest

def test_spinless_mol():
    from pyscf import fci

    from libdmet.system.integral import Integral 
    from libdmet.solver import scf
    from libdmet.solver.fci import FCI
    from libdmet.routine.spinless_helper import \
            transform_spinless_mol, Ham_compact2uhf, extractRdm 
    from libdmet.utils.misc import max_abs, mdot
    from libdmet.utils import logger as log

    np.set_printoptions(4, linewidth=1000, suppress=True)
    np.random.seed(4)
    log.verbose = "DEBUG2"

    # ********************************
    # set up system
    # first test D = 0 case
    # ********************************
    norb = 4
    nelec_a = 2
    nelec_b = 2

    h1_a = np.random.random((norb, norb))
    h1_a = h1_a + h1_a.conj().T
    
    h1_b = np.random.random((norb, norb))
    h1_b = h1_b + h1_b.conj().T
    
    D = np.zeros((norb, norb))
    
    h1 = np.asarray((h1_a, h1_b))

    h2_aa = np.random.random((norb, norb, norb, norb))
    h2_aa = h2_aa + h2_aa.transpose(1,0,2,3)
    h2_aa = h2_aa + h2_aa.transpose(0,1,3,2)
    h2_aa = h2_aa + h2_aa.transpose(2,3,0,1)
    
    h2_bb = np.random.random((norb, norb, norb, norb))
    h2_bb = h2_bb + h2_bb.transpose(1,0,2,3)
    h2_bb = h2_bb + h2_bb.transpose(0,1,3,2)
    h2_bb = h2_bb + h2_bb.transpose(2,3,0,1)
     
    h2_ab = np.random.random((norb, norb, norb, norb))
    h2_ab = h2_ab + h2_ab.transpose(1,0,2,3)
    h2_ab = h2_ab + h2_ab.transpose(0,1,3,2)
    # 4-fold is enough
    #h2_ab = h2_ab + h2_ab.transpose(2,3,0,1)
    
    h2 = np.asarray((h2_aa, h2_bb, h2_ab))

    # ********************************
    # UHF-FCI reference
    # ********************************

    log.note("UHF-FCI reference")
    Ham_u = Integral(norb, restricted=False, bogoliubov=False, \
            H0=0.0, H1={"cd": h1}, H2={"ccdd": h2})
    fci_u = FCI(restricted=False, Sz=(nelec_a-nelec_b), bcs=False, \
            tol=1e-10, max_cycle=200, max_memory=40000)

    rdm1_u, e_u = fci_u.run(Ham_u, nelec=(nelec_a+nelec_b))
    log.result("UHF-FCI Energy: %s", e_u)
    log.result("UHF-FCI rdm1: \n%s", rdm1_u)
    log.result("UHF-FCI rdm1 trace: %s", rdm1_u[0].trace()+rdm1_u[1].trace())

    # ********************************
    # Spinless-FCI
    # ********************************

    log.note("Spinless system with UIHF-FCI")
    Ham_sl = transform_spinless_mol(h1, D, h2)
    Ham_slu = Ham_compact2uhf(Ham_sl, eri_spin=1)

    fci_sl = FCI(restricted=False, Sz=norb, bcs=False, \
            tol=1e-10, max_cycle=200, max_memory=40000)
    dm0 = np.zeros((2, norb*2, norb*2))
    dm0[0, np.arange(nelec_a), np.arange(nelec_a)] = 1.0
    dm0[0, np.arange(norb, norb*2), np.arange(norb, norb*2)] = 1.0
    dm0[0, np.arange(norb, norb+nelec_b), np.arange(norb, norb+nelec_b)] = 0.0

    # Mu chemical potential
    # ZHC NOTE here Mu influence the mean-field solution,
    # however, does not influence the FCI solution.
    # is this reasonable?
    Mu = 4.0
    rdm1_sl, e_sl = fci_sl.run(Ham_slu, nelec=norb, dm0=dm0, Mu=Mu)

    rdm1_sla, rdm1_slb, rdm1_ba = extractRdm(rdm1_sl[0]) 

    log.result("Spinless-FCI Energy: %s", e_sl)
    log.result("Spinless-FCI rdm1: \n%s", \
            np.asarray((rdm1_sla, rdm1_slb, rdm1_ba)))
    log.result("Spinless-FCI rdm1 trace: %s", \
            rdm1_sla.trace() + rdm1_slb.trace())

    log.result("Diff: GHF-UFCI - UHF-FCI: %s", e_sl - e_u)
    log.eassert(abs(e_sl - e_u) < 1e-8, \
            "FCI Energy should be the same, if particle number is correct.")

    # ********************************
    # HFB vs UIHF(all alpha) vs GHF
    # D != 0
    # ********************************
    Mu = 4.0
    D = np.random.random((norb, norb))
    #D = D + D.conj().T
    newton_ah = True
    do_diis = True
    InitGuess = None

    # HFB
    log.note("HFB")
    Ham_b = Integral(norb, restricted=False, bogoliubov=True, \
            H0=0.0, H1={"cd": h1, "cc": np.asarray((D,))}, \
            H2={"ccdd": h2, "cccd": None, "cccc": None})

    myscf_b = scf.SCF(newton_ah=newton_ah)
    myscf_b.set_system(None, 0, True, False) # nelec, spin, bogo, res
    myscf_b.set_integral(Ham_b)
    e_b, rdm1_b = myscf_b.HFB(Mu, tol=1e-10, do_diis=do_diis,
            InitGuess=InitGuess)
    rdm1_hfb_a, rdm1_hfb_b, rdm1_hfb_ba = extractRdm(rdm1_b)

    log.result("HFB MO energy:\n%s", myscf_b.mf.mo_energy)
    log.result("trace of normal rdm1: %s", \
            rdm1_hfb_a.trace() + rdm1_hfb_b.trace())
    log.result("E (HFB): %s", e_b)

    # GHF
    log.note("Spinless HF use GIHF")
    Ham_sl = transform_spinless_mol(Ham_b.H1["cd"], Ham_b.H1["cc"], \
            Ham_b.H2["ccdd"])

    myscf_sl = scf.SCF(newton_ah=newton_ah)
    myscf_sl.set_system(norb, 0, False, False) # nelec, spin, bogo, res
    myscf_sl.set_integral(Ham_sl)
    e_sl, rdm1_sl = myscf_sl.GHF(Mu=Mu, tol=1e-10, do_diis=do_diis,\
            InitGuess=InitGuess)
    rdm1_sl_a, rdm1_sl_b, rdm1_sl_ba = extractRdm(rdm1_sl)

    log.result("GHF MO energy:\n%s", myscf_sl.mf.mo_energy)
    log.result("trace of normal rdm1: %s", \
            rdm1_sl_a.trace() + rdm1_sl_b.trace())
    log.result("E (GHF): %s", e_sl)

    log.result("Diff GHF - HFB: %s", e_sl - e_b)
    log.eassert(abs(e_sl - e_b) < 1e-8, \
            "GHF and HFB should give the same energy")

    # UIHF
    log.note("Spinless HF use UIHF")
    dm0 = np.zeros((2, norb*2, norb*2))
    dm0[0] = np.eye(norb*2) * 0.5
    Ham_slu = Ham_compact2uhf(Ham_sl, eri_spin=3)

    myscf_slu = scf.SCF(newton_ah=newton_ah)
    myscf_slu.set_system(norb, norb, False, False) # nelec, spin, bogo, res
    myscf_slu.set_integral(Ham_slu)
    e_slu, rdm1_slu = myscf_slu.HF(Mu=Mu, tol=1e-10, do_diis=do_diis,\
            InitGuess=dm0)
    rdm1_slu_a, rdm1_slu_b, rdm1_slu_ba = extractRdm(rdm1_slu)

    log.result("E (GHF from UIHF): %s", e_slu)
    log.eassert(abs(e_slu - e_b) < 1e-8, \
            "GHF from UIHF and HFB should give the same energy")

    # ********************************
    # GHF-FCI vs HFB-DMRG
    # D != 0
    # ********************************
    # GHF-FCI using UIHF-FCI 
    log.note("GHF-FCI using UIHF-FCI")
    fci_slu = FCI(restricted=False, Sz=norb, bcs=False, \
            tol=1e-10, max_cycle=200, max_memory=40000)

    rdm1_slu, e_slu = fci_slu.run(Ham_slu, nelec=norb, dm0=dm0, Mu=Mu)
    log.result("E (FCI from UIHF): %s", e_slu)

    with pytest.raises(Exception):
        # HFB-DMRG
        log.note("HFB-DMRG")
        from libdmet.solver import impurity_solver
        maxM = 1000
        ncas = norb
        block = impurity_solver.StackBlock(nproc=1, nthread=28, nnode=1, \
                bcs=True, tol=1e-11, maxM=maxM, SharedDir="./shdir")
        solver = impurity_solver.BCSDmrgCI(ncas=ncas, cisolver=block, \
                splitloc=True, algo="energy", mom_reorder=False)


        Ham_b = Integral(norb, restricted=False, bogoliubov=True, \
                H0=0.0, H1={"cd": h1, "cc": D[None]}, \
                H2={"ccdd": h2, "cccd": None, "cccc": None})

        GRho_b, e_b = solver.run(Ham_b, guess=dm0[0], Mu=Mu)
        log.result("E (BCS-DMRG): %s", e_b)
        log.result("Diff: %s", e_b - e_slu)

        log.eassert(abs(e_b - e_slu) < 1e-8, \
                "GHF-FCI and BCS-DMRG should give the same energy. diff: %s"%(e_b - e_slu))

if __name__ == "__main__":
    test_spinless_mol()
