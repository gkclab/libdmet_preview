#! /usr/bin/env python

import os, sys
import pytest

@pytest.mark.parametrize(
    "mp2", [True, False]
)
def test_gsodmrgci(mp2):
    import sys
    import numpy as np
    from scipy import linalg as la
    from pyscf import lib
    from libdmet.utils import logger as log
    import libdmet.dmet.HubbardGSO as dmet
    from libdmet.utils.misc import max_abs, mdot
    from libdmet.system.hamiltonian import HamNonInt

    np.set_printoptions(3, linewidth=1000, suppress=False)
    log.verbose = "DEBUG2"
    
    U = 6.0 
    LatSize = [40, 40]
    ImpSize = [2, 2]
    Filling = 1.0 / 2
    int_bath = False
    restricted = False
    use_hcore_as_emb_ham = True
    MaxIter = 50

    Mu = U * Filling
    last_dmu = 0.0
    beta = 1000.0
    imp_fit = False

    DiisStart = 3
    TraceStart = 2
    DiisDim = 4
    dc = dmet.FDiisContext(DiisDim)
    adiis = lib.diis.DIIS()
    adiis.space = DiisDim

    ntotal = Filling * np.prod(LatSize)
    if ntotal - np.round(ntotal) > 1e-5:
        log.warning("rounded total number of electrons to integer %d", \
                np.round(ntotal))
        Filling=float(np.round(ntotal)) / np.prod(LatSize)

    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nao = nscsites = Lat.supercell.nsites
    nso = nao * 2
    
    Ham = dmet.Ham(Lat, U)
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)
    
    H1_k = Lat.getH1(kspace=True)
    H2_loc = Lat.getH2(kspace=False)
    
    vcor = dmet.AFInitGuess(ImpSize, U, Filling, rand=0.001)
    vcor_mat = vcor.get()
    vcor_mat[1] = -vcor_mat[1].T
    vcor.assign(vcor_mat)
    
    # p-h transformed hamiltonian.
    GH1, GH0_from_H1 = dmet.transform_H1_k(H1_k)
    GH2, GH1_from_H2, GH0_from_H2 = dmet.transform_H2_local(H2_loc)
    GH0_from_vcor = -vcor.get()[1].trace() * 0.5
    GH0 = GH0_from_H1 + GH0_from_vcor
    
    Ham_sl = HamNonInt(Lat, GH1, GH2, Fock=None, ImpJK=None, \
                kspace_input=True, spin_dim_H2=3, H0=GH0)
    Lat.setHam(Ham_sl, use_hcore_as_emb_ham=use_hcore_as_emb_ham)
    
    
    cisolver = dmet.impurity_solver.CCSD(restricted=restricted, tol=1e-6, ghf=True)
    ncas = Lat.nval * 2 + Lat.nvirt #- 2
    nelecas = Lat.nval * 2 #- 2
    solver = dmet.impurity_solver.GSODmrgCI(ncas, nelecas, MP2natorb=mp2,
                                            spinAverage=False, splitloc=True,
                                            cisolver=cisolver,
                                            mom_reorder=False,
                                            scf_newton=True)

    E_old = 0.0
    conv = False
    history = dmet.IterHistory()
    dVcor_per_ele = None
    #dmet.SolveImpHam_with_fitting.load("./frecord")

    for iter in range(MaxIter):
        log.section("\nDMET Iteration %d\n", iter)
        
        log.section("\nsolving mean-field problem\n")
        log.result("Vcor =\n%s", vcor.get())
        log.result("Mu (guess) = %20.12f", Mu)
        
        GH0_from_vcor = -vcor.get()[1].trace() * 0.5
        Lat.H0 = Lat.Ham.H0 = GH0_from_H1 + GH0_from_vcor

        GRho, Mu, ires = dmet.GHartreeFock(Lat, vcor, Filling, mu0_elec=Mu, \
                beta=beta, fix_mu=False, mu0=None, thrnelec=1e-10, scf=False,
                full_return=True, verbose=1, conv_tol=1e-10)
        GRho_k = ires["rho_k"]
        E_mf = ires["E"] / nscsites
        log.result("Mean-field energy (per site): %s", E_mf)
        
        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, GRho, vcor, Mu, \
                matching=False, int_bath=int_bath, hcore_add=GH1_from_H2,
                H0_add=GH0_from_H2 - GH0_from_vcor, localize_bath='scdm')
         
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu) 
        basis_k = Lat.R2k_basis(basis)
        
        log.section("\nsolving impurity problem\n")
        ci_args = {"restart": True}
        solver_args = {"guess": dmet.foldRho_k(GRho_k, basis_k),
                       "basis": basis, "ci_args": ci_args}

        GRhoEmb, EnergyEmb, ImpHam, dmu = \
                dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, \
                solver, solver_args, thrnelec=2e-5)
        dmet.SolveImpHam_with_fitting.save("./frecord")
        
        last_dmu += dmu
        GRhoImp, EnergyImp, nelecImp = \
                dmet.transformResults(GRhoEmb, EnergyEmb, Lat, basis, ImpHam, \
                H1e, Mu, last_dmu=last_dmu, int_bath=int_bath, \
                solver=solver, solver_args=solver_args)
        log.result("last_dmu : %s", last_dmu)
        log.result("E (DMET) : %s", EnergyImp)

        dE = EnergyImp - E_old
        E_old = EnergyImp 
        dump_res_iter = np.array([Mu, last_dmu, vcor.param, GRhoEmb, basis, GRhoImp], dtype = object)
        np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter)
        
        log.section("\nfitting correlation potential\n")
        vcor_new, err = dmet.FitVcor(GRhoEmb, Lat, basis, vcor, Mu, \
                beta=beta, CG_check=False, test_grad=True, imp_fit=imp_fit, \
                fix_mu=True, mu0=0.0, method='CG', MaxIter1=1000, \
                ytol=1e-8, gtol=1e-4)
        
        # Fix trace
        if iter >= TraceStart:
            # to avoid spiral increase of vcor and mu
            log.result("Keep trace of vcor unchanged")
            vcor_new = dmet.keep_vcor_trace_fixed(vcor_new, vcor)
        
        if iter >= DiisStart:
            pvcor = adiis.update(vcor_new.param)
            dc.nDim = adiis.get_num_vec()
        else:
            pvcor = vcor_new.param
        
        dVcor_per_ele = la.norm(pvcor - vcor.param) / (len(pvcor))
        vcor.update(pvcor)
        
        history.update(EnergyImp, err, nelecImp, dVcor_per_ele, dc)
        history.write_table()
        
        if dVcor_per_ele < 5.0e-5 and abs(dE) < 1.0e-5 and iter > 3 :
            conv = True
            break
    if conv:
        log.result("DMET converged")
    else:
        log.result("DMET cannot converge")
    
    log.result("difference with ref data: %.3e", EnergyImp - -0.650255236756)
    assert abs(EnergyImp - -0.650255236756) < 1e-4
    
    sys.modules.pop("libdmet.dmet.Hubbard", None)
    return EnergyImp

def test_matching():
    import numpy as np
    from scipy import linalg as la
    from pyscf import gto, scf, lib, ao2mo
    
    from libdmet.utils import logger as log
    import libdmet.dmet.HubbardGSO as dmet
    from libdmet.basis_transform import make_basis
    from libdmet.routine import spinless
    from libdmet.system import integral
    from libdmet.solver import scf_solver
    from libdmet.solver import scf_mu

    log.verbose = "DEBUG1"
    np.set_printoptions(3, linewidth=1000, suppress=True)

    mol = gto.Mole()
    mol.verbose = 5
    mol.atom = '''
    O 0 0      0
    H 0 -2.757 2.587
    H 0  2.757 2.587'''
    mol.basis = '321g'
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    
    nelec = mol.nelectron
    nao = mol.nao_nr()
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    eri = mf._eri
    rdm1 = mf.make_rdm1()
    e_nuc = mf.energy_nuc()

    C_ao_lo = make_basis.get_C_ao_lo_iao_mol(mf, minao='minao', orth_virt=True, full_virt=False,
                                             full_return=False, pmol_val=None, pmol_core=None,
                                             tol=1e-10)
    
    hcore_lo = make_basis.transform_h1_to_lo_mol(hcore, C_ao_lo)
    ovlp_lo = make_basis.transform_h1_to_lo_mol(ovlp, C_ao_lo)
    rdm1_lo = make_basis.transform_rdm1_to_lo_mol(rdm1, C_ao_lo, ovlp)
    eri_lo = ao2mo.kernel(eri, C_ao_lo)

    nso = nao * 2
    GH1, GH0 = spinless.transform_H1_local(hcore_lo, ovlp=None, C_ao_lo=None, compact=False)
    GV2, GV1, GV0 = spinless.transform_H2_local(eri_lo, ovlp=None, C_ao_lo=None, compact=False, hyb=1.0,
                                                hyb_j=1.0, ao_repr=False)
    GH0 += GV0 + e_nuc
    GH1 += GV1
    
    Grdm1 = spinless.transform_rdm1_local(rdm1_lo * 0.5, compact=False)

    Ham = integral.Integral(nso, True, False, H1=GH1[None], H2=GV2[None], H0=GH0)
    
    scfsolver = scf_solver.SCFSolver(
                 restricted=True, ghf=True, tol=1e-10,
                 max_cycle=200,  max_memory=4000, scf_newton=False,
                 Sz=GH1.shape[-1]//2) 
    rdm1, E = scfsolver.run(Ham, nelec=GH1.shape[-1]//2, dm0=Grdm1, fit_mu=True,
                         nelec_target=nelec, mu_elec=0.0)
    mu_elec = scfsolver.scfsolver.mf.mu_elec

    GH1 += scfsolver.scfsolver.mf.v_mu
    
    ncas = GH1.shape[-1] // 2 
    nelecas = GH1.shape[-1] // 2
    basis = np.eye(ncas * 2)
    basis = basis[None]
    print ("mean-field nelec:", scf_mu._get_nelec(rdm1, basis))

    cisolver = dmet.impurity_solver.CCSD(restricted=True, tol=1e-6, ghf=True,
            Sz=nelecas, tol_normt=1e-4)
    solver = dmet.impurity_solver.GSODmrgCI(ncas, nelecas, MP2natorb=False,
                                            spinAverage=False, splitloc=True,
                                            cisolver=cisolver,
                                            mom_reorder=False,
                                            scf_newton=False, tmpDir='./',
                                            fiedler=True)
    
    rdm1, E = solver.run(Ham, guess=Grdm1, basis=basis, match_basis_ghf=True)
    print (rdm1)
    print ("CC nelec:", scf_mu._get_nelec(rdm1, basis))
    
    # test the fit of mu for correlated solver
    from libdmet.dmet import HubbardGSO as dmet
    solver = dmet.impurity_solver.CCSD(restricted=False, ghf=True, tol=1e-7, 
                                       tol_normt=1e-4, Sz=nelecas, scf_newton=False)
    
    print ("fit electron number for solver")
    class Lattice(object):
        nimp = nao
        imp_idx = np.arange(nimp)

    Lat = Lattice()
    Filling = nelec / nso
    
    GH1 -= scfsolver.scfsolver.mf.v_mu * 1e+5
    
    mo_coeff = scfsolver.scfsolver.mf.mo_coeff
    mo_energy = scfsolver.scfsolver.mf.mo_energy
    mo_occ = scfsolver.scfsolver.mf.mo_occ

    solver_args = {"dm0": Grdm1, "basis": basis, "scf_max_cycle": 0,
                   "mo_coeff_custom": mo_coeff,
                   "mo_occ_custom": mo_occ,
                   "mo_energy_custom": mo_energy}

    rhoEmb, EnergyEmb, Ham, dmu = \
            dmet.SolveImpHam_with_fitting(Lat, Filling, Ham, basis, solver, \
            solver_args=solver_args, fit_ghf=True)
    
    print ("CC nelec:", scf_mu._get_nelec(rhoEmb, basis))
    sys.modules.pop("libdmet.dmet.Hubbard", None)

if __name__ == "__main__":
    test_matching()
    test_gsodmrgci(mp2=True)
