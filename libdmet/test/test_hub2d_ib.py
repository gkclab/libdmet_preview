#! /usr/bin/env python

import numpy as np
import scipy.linalg as la

def test_hub2d_ib():
    from pyscf import lib
    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet

    np.set_printoptions(4, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"

    U = 2.0
    LatSize = [40, 40]
    ImpSize = [2, 2]
    Filling = 1.0 / 2
    int_bath = True
    restricted = False
    use_hcore_as_emb_ham = True
    beta = np.inf
    #beta = 100.0

    ytol = 1e-7
    gtol = 1e-3
    MaxIter = 50

    Mu = U * Filling
    last_dmu = 0.0

    DiisStart = 4
    TraceStart = 3
    DiisDim = 4
    dc = dmet.FDiisContext(DiisDim)
    adiis = lib.diis.DIIS()
    adiis.space = DiisDim

    ntotal = Filling * np.prod(LatSize)
    if ntotal - np.round(ntotal) > 1e-5:
        log.warning("rounded total number of electrons to integer %d", np.round(ntotal))
        Filling=float(np.round(ntotal)) / np.prod(LatSize)

    log.info("Actual filling: %s", Filling)

    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nscsites = Lat.supercell.nsites
    Ham = dmet.Ham(Lat, U)
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)
    vcor = dmet.AFInitGuess(ImpSize, U, Filling)

    FCI = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-12)
    #FCI = dmet.impurity_solver.FCI_AO(restricted=restricted, tol=1e-12)
    solver = FCI

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

        rho, Mu, res = dmet.HartreeFock(Lat, vcor, Filling, Mu, ires=True, beta=beta)
        Lat.update_Ham(rho)
        E_mf = res["E"] / nscsites

        log.result("Mean-field energy (per site): %s", E_mf)

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching=False, int_bath=int_bath, beta=beta)
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)

        log.section("\nsolving impurity problem\n")
        solver_args = {"nelec": (Lat.ncore+Lat.nval)*2, \
                "dm0": dmet.foldRho_k(res["rho_k"], basis_k)}

        rhoEmb, EnergyEmb, ImpHam, dmu = \
                dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
                solver_args)
        dmet.SolveImpHam_with_fitting.save("./frecord")

        last_dmu += dmu
        rhoImp, EnergyImp, nelecImp = \
                dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
                lattice=Lat, last_dmu=last_dmu, int_bath=int_bath, \
                solver=solver, solver_args=solver_args)
        log.result("E (DMET) : %s", EnergyImp)

        log.section("\nfitting correlation potential\n")
        vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, \
                vcor, beta, Filling, MaxIter2 = 0, \
                ytol=ytol, gtol=gtol, test_grad=False, \
                imp_fit=False, CG_check=True, BFGS=False)

        # Fix trace
        if iter >= TraceStart:
            # to avoid spiral increase of vcor and mu
            log.result("Keep trace of vcor unchanged")
            ddiagV = np.average(np.diagonal(\
                    (vcor_new.get()-vcor.get())[:2], 0, 1, 2))
            vcor_new = dmet.addDiag(vcor_new, -ddiagV)

        # DIIS
        if iter >= DiisStart:
            pvcor = adiis.update(np.hstack((vcor_new.param)))
            dc.nDim = adiis.get_num_vec()
        else:
            pvcor = np.hstack((vcor_new.param))

        dVcor_per_ele = la.norm(pvcor - vcor.param) / (len(vcor.param))
        vcor.update(pvcor)

        dE = EnergyImp - E_old
        E_old = EnergyImp

        log.info("trace of vcor: %s", np.sum(np.diagonal((vcor.get())[:2], 0, 1, 2)))

        history.update(EnergyImp, err, nelecImp, dVcor_per_ele, dc)
        history.write_table()
        dump_res_iter = np.array([Mu, last_dmu, vcor.param, rhoEmb, basis, rhoImp], dtype = object)
        np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter)

        if dVcor_per_ele < 1.0e-5 and abs(dE) < 1.0e-6 and iter > 3 :
            conv = True
            break

    assert abs(EnergyImp - -1.179836342898) < 1e-4

    if conv:
        log.result("DMET converged")
    else:
        log.result("DMET cannot converge")

def test_hub2d_ib_fock():
    from pyscf import lib
    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet

    np.set_printoptions(4, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"

    U = 8.0
    LatSize = [6, 6]
    ImpSize = [2, 2]
    Filling = 1.0 / 2
    int_bath = True
    restricted = False
    use_hcore_as_emb_ham = False
    beta = 1000.0

    ytol = 1e-7
    gtol = 1e-3
    MaxIter = 50

    Mu = U * Filling
    last_dmu = 0.0

    DiisStart = 4
    TraceStart = 3
    DiisDim = 4
    dc = dmet.FDiisContext(DiisDim)
    adiis = lib.diis.DIIS()
    adiis.space = DiisDim

    ntotal = Filling * np.prod(LatSize)
    if ntotal - np.round(ntotal) > 1e-5:
        log.warning("rounded total number of electrons to integer %d", np.round(ntotal))
        Filling=float(np.round(ntotal)) / np.prod(LatSize)

    log.info("Actual filling: %s", Filling)

    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nscsites = Lat.supercell.nsites
    Ham = dmet.Ham(Lat, U)
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)
    vcor = dmet.VcorZeros(restricted=False, bogoliubov=False, nscsites=nscsites)

    nkpts = Lat.nkpts
    dm_a = [np.diag([1.0, 0.0, 0.0, 1.0])] * nkpts
    dm_b = [np.diag([0.0, 1.0, 1.0, 0.0])] * nkpts
    dm0 = np.array((dm_a, dm_b))

    rho, Mu, res = dmet.HartreeFock(Lat, vcor, Filling, Mu, ires=True, beta=beta,
                                    scf=True, dm0=dm0)

    #Lat.fock_lo_k = Lat.kmf_lo.get_fock()
    #Lat.rdm1_lo_k = res["rho_k"]
    Lat.update_Ham(rho)

    FCI = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-10)
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

        rho, Mu, res = dmet.HartreeFock(Lat, vcor, Filling, Mu, ires=True, beta=beta)

        print (rho[:, 0])

        Lat.update_Ham(rho)
        E_mf = res["E"] / nscsites

        log.result("Mean-field energy (per site): %s", E_mf)

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching=False, int_bath=int_bath, beta=beta)
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)

        log.section("\nsolving impurity problem\n")
        solver_args = {"nelec": (Lat.ncore+Lat.nval)*2, \
                "dm0": dmet.foldRho_k(res["rho_k"], basis_k)}

        rhoEmb, EnergyEmb, ImpHam, dmu = \
                dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
                solver_args)
        dmet.SolveImpHam_with_fitting.save("./frecord")

        last_dmu += dmu
        rhoImp, EnergyImp, nelecImp = \
                dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
                lattice=Lat, last_dmu=last_dmu, int_bath=int_bath, \
                solver=solver, solver_args=solver_args)
        log.result("E (DMET) : %s", EnergyImp)


        log.section("\nfitting correlation potential\n")
        vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, \
                vcor, beta, Filling, MaxIter1=0, MaxIter2=300, \
                ytol=ytol, gtol=gtol, test_grad=False, \
                imp_fit=True, CG_check=True, BFGS=True)

        # DIIS
        if iter >= DiisStart:
            pvcor = adiis.update(np.hstack((vcor_new.param)))
            dc.nDim = adiis.get_num_vec()
        else:
            pvcor = np.hstack((vcor_new.param))

        dVcor_per_ele = la.norm(pvcor - vcor.param) / (len(vcor.param))
        vcor.update(pvcor)

        dE = EnergyImp - E_old
        E_old = EnergyImp

        log.info("trace of vcor: %s", np.sum(np.diagonal((vcor.get())[:2], 0, 1, 2)))

        history.update(EnergyImp, err, nelecImp, dVcor_per_ele, dc)
        history.write_table()
        dump_res_iter = np.array([Mu, last_dmu, vcor.param, rhoEmb, basis, rhoImp], dtype = object)
        np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter)

        if dVcor_per_ele < 1.0e-5 and abs(dE) < 1.0e-6 and iter > 3 :
            conv = True
            break

    assert abs(EnergyImp - -0.51685) < 1e-4

if __name__ == "__main__":
    test_hub2d_ib_fock()
    test_hub2d_ib()

