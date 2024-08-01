#! /usr/bin/env python

def t_mpicc(ooccd=False):
    import sys
    import numpy as np
    import scipy.linalg as la

    from pyscf import lib
    from libdmet.solver.scf_solver import SCFSolver
    from libdmet.solver.mpicc import GCCSD
    from libdmet.utils import logger as log
    import libdmet.dmet.HubbardGSO as dmet
    from libdmet.utils.misc import max_abs, mdot
    from libdmet.system.hamiltonian import HamNonInt

    np.set_printoptions(3, linewidth=1000, suppress=True)
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

    H1_k   = Lat.getH1(kspace=True)
    H2_loc = Lat.getH2(kspace=False)

    vcor        = dmet.AFInitGuess(ImpSize, U, Filling, rand=0.001)
    vcor_mat    = vcor.get()
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

    if ooccd:
        cisolver = SCFSolver(nproc=2, nthread=4, nnode=1, TmpDir="./tmp", SharedDir=None,
                 restricted=False, Sz=0, bcs=False, ghf=True, tol=1e-10,
                 max_cycle=200,  max_memory=40000, scf_newton=False,
                 mp2=False, oomp2=False, ooccd=True,
                 mc_conv_tol=1e-4, mc_max_cycle=20,
                 ci_conv_tol=None, ci_diis_space=10,
                 tol_normt=1e-5, level_shift=0.01, frozen=0, restart=True,
                 approx_l=False, fix_fcivec=False, use_mpi=True, verbose=5)
    else:
        cisolver = GCCSD(
                nproc = 2, nnode = 1, nthread = 4,
                restricted = restricted, tol = 1e-6
                )

    history = dmet.IterHistory()
    dvcor_per_ele = None

    istep = 0

    is_converged = False
    is_max_cycle = False

    tol_ene  = 1.0e-5
    err_ene  = 1.0

    tol_vcor = 5.0e-5
    err_vcor = 1.0

    e_new  = 0.0
    e_prev = 0.0

    while not is_converged and not is_max_cycle:
        log.section("\nDMET Iteration %d\n", istep)

        log.section("\nsolving mean-field problem\n")
        log.result("Vcor =\n%s", vcor.get())
        log.result("Mu (guess) = %20.12f", Mu)

        GH0_from_vcor = -vcor.get()[1].trace() * 0.5
        Lat.H0 = Lat.Ham.H0 = GH0_from_H1 + GH0_from_vcor

        GRho, Mu, ires = dmet.GHartreeFock(Lat, vcor, Filling, mu0_elec=Mu, \
                beta=beta, fix_mu=False, mu0=None, thrnelec=1e-10, scf=False,
                full_return=True, verbose=1, conv_tol=1e-10)
        GRho_k = ires["rho_k"]
        E_mf   = ires["E"] / nscsites
        log.result("Mean-field energy (per site): %s", E_mf)

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, GRho, vcor, Mu, \
                matching=False, int_bath=int_bath, hcore_add=GH1_from_H2,
                H0_add=GH0_from_H2 - GH0_from_vcor, localize_bath='scdm')

        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)

        solver_args={"dm0": dmet.foldRho_k(GRho_k, basis_k), "basis": basis, "fcc_name": "MPICC"}

        GRhoEmb, EnergyEmb, ImpHam, dmu = \
                dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, \
                cisolver, thrnelec=2e-5, solver_args=solver_args)

        dmet.SolveImpHam_with_fitting.save("./frecord")

        last_dmu += dmu
        GRhoImp, EnergyImp, nelecImp = \
                dmet.transformResults(GRhoEmb, EnergyEmb, Lat, basis, ImpHam, \
                H1e, Mu, last_dmu=last_dmu, int_bath=int_bath, \
                solver=cisolver)
        log.result("last_dmu : %s", last_dmu)
        log.result("E (DMET) : %s", EnergyImp)

        e_new   = EnergyImp
        de      = EnergyImp - e_prev
        e_prev  = e_new
        err_ene = abs(de)
        dump_res_iter = np.array([Mu, last_dmu, vcor.param, GRhoEmb, basis, GRhoImp], dtype = object)
        np.save('./dmet_iter_%s.npy'%(istep), dump_res_iter)

        log.section("\nfitting correlation potential\n")
        vcor_new, err = dmet.FitVcor(GRhoEmb, Lat, basis, vcor, Mu, \
                beta=beta, CG_check=False, test_grad=True, imp_fit=imp_fit, \
                fix_mu=True, mu0=0.0, method='CG', MaxIter1=1000, \
                ytol=1e-8, gtol=1e-4)

        # Fix trace
        if istep >= TraceStart:
            # to avoid spiral increase of vcor and mu
            log.result("Keep trace of vcor unchanged")
            vcor_new = dmet.keep_vcor_trace_fixed(vcor_new, vcor)

        if istep >= DiisStart:
            pvcor = adiis.update(vcor_new.param)
            dc.nDim = adiis.get_num_vec()
        else:
            pvcor = vcor_new.param

        err_vcor = la.norm(pvcor - vcor.param) / (len(pvcor))
        vcor.update(pvcor)

        history.update(EnergyImp, err, nelecImp, err_vcor, dc)
        history.write_table()

        is_converged = (err_vcor < tol_vcor) and (err_ene < tol_ene) and (istep > 3)
        is_max_cycle = (istep >= MaxIter)
        istep       += 1

    if is_converged:
        log.result("DMET converged")
    else:
        log.result("DMET cannot converge")

    return

if __name__ == "__main__":
    t_mpicc(ooccd=True)
