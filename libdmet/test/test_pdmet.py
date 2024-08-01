#! /usr/bin/env python

"""
PDMET, JCP 151, 064108 (2019).

Author:
    Zhi-Hao Cui
"""

import numpy as np
import scipy.linalg as la
import pytest

def test_pdmet():
    from pyscf import lib
    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet
    from libdmet.utils.misc import mdot, max_abs
    from libdmet.routine.slater import get_rho_glob_R, get_rdm1_idem

    np.set_printoptions(4, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"

    U = 4.0
    LatSize = [40, 40]
    ImpSize = [2, 2]
    Filling = 1.0 / 2
    int_bath = True
    restricted = False
    use_hcore_as_emb_ham = False
    #beta = np.inf
    beta = 1000.0

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
    vcor.update(np.zeros(vcor.length()))

    nkpts = Lat.nkpts
    nao = Lat.nao

    dm0 = np.zeros((2, nkpts, nao, nao), dtype=np.complex128)
    dm_a = np.diag([0.8, 0.2, 0.2, 0.8])
    dm_b = np.diag([0.2, 0.8, 0.8, 0.2])
    dm0[0] = dm_a
    dm0[1] = dm_b

    FCI = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-12)
    solver = FCI

    E_old = 0.0
    conv = False
    history = dmet.IterHistory()
    dVcor_per_ele = None
    #dmet.SolveImpHam_with_fitting.load("./frecord")

    # PDMET need to do mean-field scf first,
    # since the results depend on the initial rho_glob.
    # use HF rho as the guess of rho_glob.
    rho, Mu, res = dmet.HartreeFock(Lat, vcor, Filling, Mu, ires=True, \
            beta=beta, scf=True, dm0=dm0, conv_tol=1e-12, max_cycle=100)
    rho_glob = rho_glob_old = rho
    rho_glob_k = Lat.R2k(rho_glob)
    #Lat.update_Ham(rho_glob)

    for iter in range(MaxIter):
        log.section("\nDMET Iteration %d\n", iter)

        log.section("\nsolving mean-field problem\n")
        log.result("Vcor =\n%s", vcor.get())
        log.result("Mu (guess) = %20.12f", Mu)

        log.result("rdm1 glob:\n%s", rho_glob[:, 0])
        Lat.update_Ham(rho_glob)
        rho = rho_glob

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, \
                matching=False, int_bath=int_bath, beta=beta)
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)

        log.section("\nsolving impurity problem\n")
        solver_args = {"nelec": (Lat.ncore+Lat.nval)*2, \
                "dm0": dmet.foldRho_k(rho_glob_k, basis_k)}

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

        # construct rho_glob and get the idempotent part
        #rhoEmb = dmet.foldRho_k(rho_glob_k, basis_k)
        rho_glob_R = get_rho_glob_R(basis, Lat, rhoEmb)
        rho_glob_k = Lat.R2k(rho_glob_R)
        nelec = [nkpts * nao * Filling, nkpts * nao * Filling]
        rho_glob_k = get_rdm1_idem(rho_glob_k, nelec, beta)
        rho_glob = Lat.k2R(rho_glob_k)

        # DIIS on rho_glob
        if iter >= DiisStart:
            rho_glob = adiis.update(rho_glob)
            dc.nDim = adiis.get_num_vec()

        rho_glob_k = Lat.R2k(rho_glob)

        drho = rho_glob - rho_glob_old
        dVcor_per_ele = max_abs(drho)
        rho_glob_old = rho_glob

        dE = EnergyImp - E_old
        E_old = EnergyImp

        history.update(EnergyImp, 0.0, nelecImp, dVcor_per_ele, dc)
        history.write_table()
        dump_res_iter = np.array([Mu, last_dmu, rho_glob, rhoEmb, basis, rhoImp], dtype = object)
        np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter)

        if dVcor_per_ele < 1.0e-5 and abs(dE) < 1.0e-6 and iter > 3 :
            conv = True
            break

    # the number should be the same as reference
    assert abs(EnergyImp - -0.86792) < 5e-5

def test_pdmet_2():
    """
    Using the idea of global density matrix,
    it is possible to design a self-consistency,
    which updates the lattice fock based on the rdm1_glob.
    In this way, the vcor fit no longer exists.
    """
    from pyscf import lib
    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet
    from libdmet.utils.misc import mdot, max_abs
    from libdmet.routine.slater import get_rho_glob_R, get_rdm1_idem

    np.set_printoptions(4, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"

    U = 4.0
    LatSize = [40, 40]
    ImpSize = [2, 2]
    Filling = 1.0 / 2
    int_bath = True
    restricted = False
    use_hcore_as_emb_ham = False
    #beta = np.inf
    beta = 1000.0

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
    vcor.update(np.zeros(vcor.length()))

    nkpts = Lat.nkpts
    nao = Lat.nao

    dm0 = np.zeros((2, nkpts, nao, nao), dtype=np.complex128)
    dm_a = np.diag([0.8, 0.2, 0.2, 0.8])
    dm_b = np.diag([0.2, 0.8, 0.8, 0.2])
    dm0[0] = dm_a
    dm0[1] = dm_b

    FCI = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-12)
    solver = FCI

    E_old = 0.0
    conv = False
    history = dmet.IterHistory()
    dVcor_per_ele = None
    #dmet.SolveImpHam_with_fitting.load("./frecord")

    # PDMET need to do mean-field scf first,
    # since the results depend on the initial rho_glob.
    # use HF rho as the guess of rho_glob.
    rho, Mu, res = dmet.HartreeFock(Lat, vcor, Filling, Mu, ires=True, \
            beta=beta, scf=True, dm0=dm0, conv_tol=1e-12, max_cycle=100)
    rho_glob = rho_glob_old = rho
    rho_glob_k = Lat.R2k(rho_glob)
    #Lat.update_Ham(rho_glob)

    for iter in range(MaxIter):
        log.section("\nDMET Iteration %d\n", iter)

        log.section("\nsolving mean-field problem\n")
        log.result("Vcor =\n%s", vcor.get())
        log.result("Mu (guess) = %20.12f", Mu)

        log.result("rdm1 glob:\n%s", rho_glob[:, 0])
        Lat.update_Ham(rho_glob)
        rho, Mu, res = dmet.HartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)
        #Lat.update_Ham(rho)

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, \
                matching=False, int_bath=int_bath, beta=beta)
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)

        log.section("\nsolving impurity problem\n")
        solver_args = {"nelec": (Lat.ncore+Lat.nval)*2, \
                "dm0": dmet.foldRho_k(rho_glob_k, basis_k)}

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

        # construct rho_glob and get the idempotent part
        #rhoEmb = dmet.foldRho_k(rho_glob_k, basis_k)
        rho_glob_R = get_rho_glob_R(basis, Lat, rhoEmb)
        rho_glob_k = Lat.R2k(rho_glob_R)
        rho_glob = rho_glob_R
        #nelec = [nkpts * nao * Filling, nkpts * nao * Filling]
        #rho_glob_k = get_rdm1_idem(rho_glob_k, nelec, beta)
        #rho_glob = Lat.k2R(rho_glob_k)

        # DIIS on rho_glob
        if iter >= DiisStart:
            rho_glob = adiis.update(rho_glob)
            dc.nDim = adiis.get_num_vec()

        rho_glob_k = Lat.R2k(rho_glob)

        drho = rho_glob - rho_glob_old
        dVcor_per_ele = max_abs(drho)
        rho_glob_old = rho_glob

        dE = EnergyImp - E_old
        E_old = EnergyImp

        history.update(EnergyImp, 0.0, nelecImp, dVcor_per_ele, dc)
        history.write_table()
        dump_res_iter = np.array([Mu, last_dmu, rho_glob, rhoEmb, basis, rhoImp], dtype = object)
        np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter)

        if dVcor_per_ele < 1.0e-5 and abs(dE) < 1.0e-6 and iter > 3 :
            conv = True
            break

    # the number should be the same as reference
    assert abs(EnergyImp - -0.876942444093) < 5e-5

if __name__ == "__main__":
    test_pdmet_2()
    test_pdmet()
