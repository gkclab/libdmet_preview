#! /usr/bin/env python

"""
Test spinless formalism.
"""

import os, sys
import numpy as np
import scipy.linalg as la

def test_spinless_1shot():
    from pyscf import lib
    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet_hub
    import libdmet.dmet.HubbardGSO as dmet
    from libdmet.utils.misc import max_abs, mdot
    from libdmet.routine import spinless
    from libdmet.routine import slater
    from libdmet.system.integral import Integral
    from libdmet.solver import scf
    from libdmet.dmet import Hubbard
    from libdmet.system import lattice
    from libdmet.routine import mfd
    from libdmet.routine.spinless import get_E_dmet_HF

    np.set_printoptions(4, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"

    # Lattice settings
    LatSize = [40, 40]
    ImpSize = [2, 2]
    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites
    nso = nao * 2
    nkpts = Lat.nkpts

    # Hamiltonian
    U = 4.0
    Filling = 1.0 / 2.0
    ntotal = Filling * np.prod(LatSize)
    if ntotal - np.round(ntotal) > 1e-5:
        ntotal = int(np.round(ntotal))
        log.warning("rounded total number of electrons to integer %d", ntotal)
        Filling=float(ntotal) / np.prod(LatSize)
    Ham = dmet.Ham(Lat, U, tlist=[1.0])
    restricted = False
    bogoliubov = True
    use_hcore_as_emb_ham = True
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

    H1_k = Lat.getH1(kspace=True)
    H2_loc = Lat.getH2(kspace=False)

    vcor = dmet.AFInitGuess(ImpSize, U, Filling, rand=0.001)
    vcor_mat = vcor.get()
    vcor_mat[1] = -vcor_mat[1].T
    vcor_mat[2] = 0.0
    vcor.assign(vcor_mat)

    # p-h transformed hamiltonian.
    GH1, GH0_from_H1 = spinless.transform_H1_k(H1_k)
    GH2, GH1_from_H2, GH0_from_H2 = spinless.transform_H2_local(H2_loc)
    GH0_from_vcor = -vcor.get()[1].trace() * 0.5
    GH0 = GH0_from_H1 + GH0_from_vcor

    from libdmet.system.hamiltonian import HamNonInt
    Ham_sl = HamNonInt(Lat, GH1, GH2, Fock=None, ImpJK=None, \
                kspace_input=True, spin_dim_H2=3, H0=GH0)
    Lat.setHam(Ham_sl, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

    log.info("vcor:\n%s", vcor.get())
    beta = 2000.0
    Mu = U * Filling
    _, mu, ires = dmet.GHartreeFock(Lat, vcor, Filling, mu0_elec=Mu, \
            beta=beta, fix_mu=False, mu0=None, thrnelec=1e-10, scf=False,
            full_return=True, verbose=1, conv_tol=1e-10)

    GRho, mu, ires = dmet.GHartreeFock(Lat, vcor, None, mu0_elec=mu, \
            beta=beta, fix_mu=False, mu0=None, thrnelec=1e-8, scf=False,
            full_return=True, verbose=1, conv_tol=1e-10)

    E_mf = (ires["E"]) / nscsites
    print ("E_mf: %12.6f \n" %E_mf)
    print (dmet.extractRdm(GRho[0])[0].trace() + \
            dmet.extractRdm(GRho[0])[1].trace())

    GRho_k = ires["rho_k"]
    print (np.asarray(spinless.extractRdm(GRho[0])))
    print (np.sort(ires["e"].ravel(), kind='mergesort'))

    basis = spinless.embBasis(Lat, GRho, local=True, localize_bath="scdm")
    basis_k = Lat.R2k_basis(basis)

    last_dmu = 0.0
    int_bath = False
    ImpHam, _ = spinless.get_emb_Ham(Lat, basis, vcor, mu, local=True,
            int_bath=int_bath, hcore_add=GH1_from_H2, H0_add=GH0_from_H2 \
                    + vcor.get()[1].trace() * 0.5)
    GRho_emb = spinless.foldRho_k(GRho_k, basis_k)

    from libdmet.solver.impurity_solver import FCI
    solver = FCI(restricted=True, ghf=True)
    last_dmu = 0.0
    H1e = 0.0
    GRhoEmb, EnergyEmb, ImpHam, dmu = \
        dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
        solver_args={"dm0": dmet.foldRho_k(GRho_k, basis_k)}, thrnelec = 1e-5)
    last_dmu += dmu
    GRhoImp, EnergyImp, nelecImp = \
        dmet.transformResults(GRhoEmb, EnergyEmb, Lat, basis, ImpHam, H1e, \
        mu, last_dmu=last_dmu, int_bath=int_bath, solver=solver)

    log.result("E (DMET) : %s", EnergyImp)
    log.eassert(abs(EnergyImp - -0.8649303805028655) < 1e-7, "should be the "
                "same as UHF-DMET")


    print ("test global density matrix:")
    rdm1_glob_R = spinless.get_rho_glob_R(basis, Lat, GRhoEmb, compact=True)
    rdm1_glob_full = spinless.get_rho_glob_R(basis, Lat, GRhoEmb, compact=False)

    sys.modules.pop("libdmet.dmet.Hubbard", None)

def spinless_05(solver_type):
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
    #beta = np.inf
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

    if solver_type == "FCI":
        solver = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-10, \
                ghf=True)
    elif solver_type == "CC":
        solver = dmet.impurity_solver.CCSD(restricted=restricted, ghf=True, \
                max_memory=40000)
    elif solver_type == 'OOMP2':
        solver = dmet.impurity_solver.SCFSolver(restricted=restricted, ghf=True,
                                                oomp2=True, restart=True,
                                                mc_conv_tol=1e-5, ci_conv_tol=1e-8,
                                                tol_normt=1e-6)
    elif solver_type == 'OOCCD':
        solver = dmet.impurity_solver.SCFSolver(restricted=restricted, ghf=True,
                                                ooccd=True, restart=True,
                                                mc_conv_tol=1e-5, ci_conv_tol=1e-8,
                                                tol_normt=1e-6)
    else:
        raise ValueError

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
        GRho, Mu, ires = dmet.GHartreeFock(Lat, vcor, None, mu0_elec=Mu, \
                beta=beta, fix_mu=False, mu0=None, thrnelec=1e-10, scf=False,
                full_return=True, verbose=1, conv_tol=1e-10)
        GRho_k = ires["rho_k"]
        E_mf = ires["E"] / nscsites
        log.result("Mean-field energy (per site): %s", E_mf)

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, GRho, vcor, Mu, \
                matching=False, int_bath=int_bath, hcore_add=GH1_from_H2,
                H0_add=GH0_from_H2 - GH0_from_vcor, localize_bath='scdm', kind='svd', tol_bath=1e-6)

        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)

        log.section("\nsolving impurity problem\n")
        solver_args = {"dm0": dmet.foldRho_k(GRho_k, basis_k), "basis": basis}

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
    sys.modules.pop("libdmet.dmet.Hubbard", None)
    return EnergyImp

def test_spinless_fci_05():
    from libdmet.utils import logger as log
    EnergyImp = spinless_05(solver_type="FCI")
    log.result("difference with ref data: %.3e", EnergyImp - -0.652114179764)
    assert abs(EnergyImp - -0.652114179764) < 1e-4

def test_spinless_gccsd_05():
    from libdmet.utils import logger as log
    EnergyImp = spinless_05(solver_type="CC")
    log.result("difference with ref data: %.3e", EnergyImp - -0.650255236756)
    assert abs(EnergyImp - -0.650255236756) < 1e-4

def test_spinless_oomp2_05():
    from libdmet.utils import logger as log
    EnergyImp = spinless_05(solver_type="OOMP2")
    E_ref = -0.620348938953
    log.result("difference with ref data: %.3e", EnergyImp - E_ref)
    assert abs(EnergyImp - E_ref) < 1e-4

def test_spinless_ooccd_05():
    from libdmet.utils import logger as log
    EnergyImp = spinless_05(solver_type="OOCCD")
    E_ref = -0.650709068552
    log.result("difference with ref data: %.3e", EnergyImp - E_ref)
    assert abs(EnergyImp - E_ref) < 1e-4

def test_spinless_fci_08():
    import sys
    from pyscf import lib
    from libdmet.utils import logger as log
    import libdmet.dmet.HubbardGSO as dmet
    from libdmet.utils.misc import max_abs, mdot
    from libdmet.system.hamiltonian import HamNonInt

    np.set_printoptions(3, linewidth=1000, suppress=False)
    log.verbose = "DEBUG2"

    U = 6.0
    LatSize = [60, 60]
    ImpSize = [2, 2]
    Filling = 0.8 / 2
    int_bath = False
    restricted = False
    use_hcore_as_emb_ham = True
    MaxIter = 50

    Mu = U * Filling
    last_dmu = -0.23898961883345338
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

    vcor = dmet.AFInitGuess(ImpSize, U, Filling, rand=0.001, bogo_res=True)
    vcor_mat = vcor.get()
    vcor_mat[1] = -vcor_mat[1].T
    vcor.assign(vcor_mat)

    # p-h transformed hamiltonian.
    GH1, GH0_from_H1 = dmet.transform_H1_k(H1_k)
    GH2, GH1_from_H2, GH0_from_H2 = dmet.transform_H2_local(H2_loc)
    GH0_from_vcor = -vcor.get()[1].trace() * 0.5
    GH0 = GH0_from_H1 + GH0_from_vcor

    Ham_sl = HamNonInt(Lat, GH1, GH2, Fock=None, ImpJK=None,
                       kspace_input=True, spin_dim_H2=3, H0=GH0)
    Lat.setHam(Ham_sl, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

    FCI = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-10, ghf=True)
    solver = FCI

    E_old = 0.0
    conv = False
    history = dmet.IterHistory()
    dVcor_per_ele = None

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
        solver_args = {"dm0": dmet.foldRho_k(GRho_k, basis_k)}

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
                beta=beta, CG_check=False, test_grad=True, imp_fit=imp_fit,
                fix_mu=True, mu0=0.0, method='SD', MaxIter1=1000, \
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

    # test unrestricted bogoliubov potential
    vcor_mat = vcor.get()
    vcor = dmet.AFInitGuess(ImpSize, U, Filling, rand=0.001, bogo_res=False)
    vcor.assign(vcor_mat)

    adiis = lib.diis.DIIS()
    adiis.space = DiisDim

    E_old = 0.0
    conv = False
    history = dmet.IterHistory()
    dVcor_per_ele = None

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
        solver_args = {"dm0": dmet.foldRho_k(GRho_k, basis_k)}

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
                beta=beta, CG_check=False, test_grad=False, imp_fit=imp_fit,
                fix_mu=True, mu0=0.0, method='SD', MaxIter1=1000, \
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

        if dVcor_per_ele < 5.0e-5 and abs(dE) < 1.0e-5 and iter > 2:
            conv = True
            break

    log.result("difference with ref data: %.3e", EnergyImp - -1.001725641814)
    assert abs(EnergyImp - -1.001725641814) < 2e-4

    sys.modules.pop("libdmet.dmet.Hubbard", None)

if __name__ == "__main__":
    test_spinless_ooccd_05()
    test_spinless_oomp2_05()
    test_spinless_fci_08()
    test_spinless_1shot()
    test_spinless_fci_05()
    test_spinless_gccsd_05()
