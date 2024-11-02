#! /usr/bin/env python

import os, sys
import numpy as np
import scipy.linalg as la

import pytest

def test_hub2dbcs_nib():
    from pyscf import lib
    from libdmet.utils import logger as log
    import libdmet.dmet.HubbardBCS as dmet
    from libdmet.utils.misc import max_abs

    np.set_printoptions(3, linewidth=1000, suppress=False)
    log.verbose = "DEBUG2"

    with pytest.raises(OSError):
        # Lattice settings
        LatSize = [60, 60]
        ImpSize = [2, 2]
        Lat = dmet.SquareLattice(*(LatSize+ImpSize))
        nscsites = Lat.nscsites

        # Hamiltonian
        U = 6.0
        Filling = 0.8 / 2.0
        ntotal = Filling * np.prod(LatSize)
        if ntotal - np.round(ntotal) > 1e-5:
            ntotal = int(np.round(ntotal))
            log.warning("rounded total number of electrons to integer %d", ntotal)
            Filling=float(ntotal) / np.prod(LatSize)
        Ham = dmet.Ham(Lat, U)
        restricted = False
        bogoliubov = True
        use_hcore_as_emb_ham = True
        Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

        # chemical potential and finite T
        Mu = U * Filling
        #last_dmu = -0.312223942888
        last_dmu = -0.233103840590
        #last_dmu = 0.0
        fix_mu = False # quasi particle fermi level is fixed?
        #beta = np.inf
        beta = 2000.0

        # DMET control
        int_bath = False
        MaxIter = 50
        u_tol = 5.0e-5
        E_tol = 1.0e-5
        iter_tol = 4

        # DIIS trace
        DiisStart = 4
        TraceStart = 3
        DiisDim = 4
        dc = dmet.FDiisContext(DiisDim)
        adiis = lib.diis.DIIS()
        adiis.space = DiisDim

        # Vcor
        vcor = dmet.AFInitGuess(ImpSize, U, Filling, rand=0.0)
        #np.save('./vcor0_param.npy', vcor.param)
        #vcor.update(np.load('./vcor0_param.npy'))
        #vcor_mat = vcor.get()
        #vcor_mat[2] = vcor_mat[2] + vcor_mat[2].conj().T

        # Solver
        LMO = True
        maxM = 800
        if LMO:
            ncas = Lat.nscsites + Lat.nval # imp + bath
            block = dmet.impurity_solver.StackBlock(nproc=1, nthread=28, nnode=1, \
                    bcs=True, tol=1e-6, maxM=maxM, SharedDir="./shdir")
            solver = dmet.impurity_solver.BCSDmrgCI(ncas=ncas, \
                    cisolver=block, splitloc=True, algo="energy",\
                    mom_reorder=True)
        else:
            solver = dmet.impurity_solver.StackBlock(nproc=1, nthread=28, nnode=1, \
                    bcs=True, reorder=True, tol=1e-6, maxM=maxM)

        # loop parameters
        E_old = 0.0
        conv = False
        history = dmet.IterHistory()
        dVcor_per_ele = None
        if os.path.exists("./frecord"):
            dmet.SolveImpHam_with_fitting.load("./frecord")
            log.info("Load frecord success.")
        else:
            log.info("Load frecord failed.")

        GRho, Mu = dmet.HartreeFockBogoliubov(Lat, vcor, Filling, Mu, \
                beta=beta, fix_mu=False, thrnelec=1e-10)

        for iter in range(MaxIter):
            log.section("\nDMET Iteration %d\n", iter)

            # ***********************************************************
            # Mean field
            # ***********************************************************
            log.section("\nsolving mean-field problem\n")
            log.result("Vcor =\n%s", vcor.get())
            log.result("Mu (guess) = %20.12f", Mu)

            GRho, Mu, res = dmet.HartreeFockBogoliubov(Lat, vcor, None, Mu, \
                    beta=beta, fix_mu=fix_mu, thrnelec=1e-10, full_return=True)
            E_mf = res["E"] / nscsites
            log.result("Mean-field energy (per site): %s", E_mf)

            # ***********************************************************
            # DMET bath and Hamiltonian
            # ***********************************************************
            log.section("\nconstructing impurity problem\n")
            ImpHam, H_energy, basis = dmet.ConstructImpHam(Lat, GRho, vcor, Mu,
                    localize_bath=None, matching=True)
            ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)

            # ***********************************************************
            # Solver
            # ***********************************************************
            log.section("\nsolving impurity problem\n")
            if LMO:
                if iter <= 3:
                    solver.localized_cas = None
                    solver_args = {"guess": dmet.foldRho(GRho, Lat, basis), "basis": basis, "ci_args": {"restart": False}}
                elif iter <= 5:
                    solver.localized_cas = None
                    solver.cisolver.cisolver.optimized = False # not do restart among different dmet iters
                    solver_args = {"guess": dmet.foldRho(GRho, Lat, basis), "basis": basis}
                else:
                    solver_args = {"guess": dmet.foldRho(GRho, Lat, basis), "basis": basis}
            else:
                solver.cisolver.optimized = False # not do restart among different dmet iters
                solver_args = {}

            GRhoEmb, EnergyEmb, ImpHam, dmu = \
                    dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
                    delta=0.01, step=0.2, thrnelec=1e-5, solver_args=solver_args)
            dmet.SolveImpHam_with_fitting.save("./frecord")
            last_dmu += dmu
            log.result("last_dmu : %20.12f", last_dmu)

            # ***********************************************************
            # Collect results and compute energy
            # ***********************************************************
            GRhoImp, EnergyImp, nelecImp = \
                    dmet.transformResults(GRhoEmb, EnergyEmb, Lat, basis, ImpHam, H_energy, \
                    Mu, last_dmu=last_dmu, int_bath=int_bath)
            log.result("E (DMET) per site: %20.12f", EnergyImp / Lat.nscsites)
            dump_res_iter = np.array([Mu, last_dmu, vcor.param, GRhoEmb, basis, GRhoImp],\
                    dtype=object)
            np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter, allow_pickle=True)

            # ***********************************************************
            # Fit correlation potential
            # ***********************************************************
            log.section("\nfitting correlation potential\n")
            vcor_new, err = dmet.FitVcor(GRhoEmb, Lat, basis, vcor, Mu, \
                    MaxIter1=max(len(vcor.param)*5, 1000), MaxIter2=0, \
                    CG_check=True, fix_mu=True, beta=beta)

            # ***********************************************************
            # Fix trace
            # ***********************************************************
            if iter >= TraceStart:
                log.result("Keep trace of vcor unchanged")
                ddiagV = np.average(np.diagonal(\
                        (vcor_new.get()-vcor.get())[:2], 0, 1, 2))
                vcor_new = dmet.addDiag(vcor_new, -ddiagV)

            dVcor_per_ele = max_abs(vcor_new.param - vcor.param) #/ (len(vcor.param))
            dE = EnergyImp - E_old
            E_old = EnergyImp

            # ***********************************************************
            # DIIS and new HFB
            # ***********************************************************
            GRho, Mu_new = dmet.HartreeFockBogoliubov(Lat, vcor_new, Filling, Mu, \
                    beta=beta, fix_mu=fix_mu, thrnelec=1e-10)

            if iter >= DiisStart:
                pvcor = adiis.update(np.hstack((vcor_new.param, Mu_new)))
                dc.nDim = adiis.get_num_vec()
            else:
                pvcor = np.hstack((vcor_new.param, Mu_new))

            dVcor_per_ele = max_abs(pvcor[:-1] - vcor.param)
            vcor.update(pvcor[:-1])
            Mu = pvcor[-1]
            log.info("trace of vcor: %s", \
                    np.sum(np.diagonal((vcor.get())[:2], 0, 1, 2)))

            history.update(EnergyImp, err, nelecImp, dVcor_per_ele, dc)
            history.write_table()
            if dVcor_per_ele < u_tol and abs(dE) < E_tol and iter > iter_tol :
                conv = True
                break
        if conv:
            log.result("DMET converged")
        else:
            log.result("DMET cannot converge")
    sys.modules.pop("libdmet.dmet.Hubbard", None)

if __name__ == "__main__":
    test_hub2dbcs_nib()
