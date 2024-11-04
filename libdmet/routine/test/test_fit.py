#! /usr/bin/env python

import os
import sys
import numpy as np
import scipy.linalg as la
import pytest

@pytest.mark.parametrize(
    "method", ['SD', 'CG', 'BFGS', 'trust-NCG', 'CIAH']
)
def test_fit(method):
    from libdmet.routine.fit import minimize
    from libdmet.utils import logger as log

    log.verbose = "DEBUG1"
    func = lambda x: x[0]**2 + x[1]**4 + 2*x[1]**2 + 2*x[0] + 2.0
    x0 = np.asarray([10.0, 20.0])
    x, y, converge_pattern, _ = minimize(func, x0, MaxIter=3000, \
            method=method, initial_trust_radius=1.0, max_trust_radius=1000.0, \
            num_cg_steps=1, max_stepsize=100.0)
    log.result("x = %s\ny=%20.12f", x, y)


def test_cvx_frac_gso():
    from pyscf import lib
    import libdmet.dmet.HubbardGSO as dmet
    from libdmet.utils.misc import max_abs, mdot
    from libdmet.system.hamiltonian import HamNonInt
    from libdmet.utils.get_order_param import get_order_param
    from libdmet.routine import mfd, spinless
    import libdmet.utils.logger as log

    np.set_printoptions(4, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"

    U = 8.0
    LatSize = [40, 40]
    ImpSize = [2, 2]
    Filling = 0.92 / 2.0
    int_bath = True
    restricted = False
    use_hcore_as_emb_ham = False
    MaxIter = 50

    Mu = U * Filling
    last_dmu = 0.0
    beta = 1000.0
    imp_fit = True

    DiisStart = 4
    DiisDim = 2
    dc = dmet.FDiisContext(DiisDim)
    adiis = lib.diis.DIIS()
    adiis.space = DiisDim

    ntotal = Filling * np.prod(LatSize)
    if abs(ntotal - np.round(ntotal)) > 1e-5:
        log.warning("rounded total number of electrons to integer %d",
                    np.round(ntotal))
        Filling=float(np.round(ntotal)) / np.prod(LatSize)

    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nkpts = Lat.nkpts
    nao = nscsites = Lat.supercell.nsites
    nso = nao * 2

    Ham = dmet.Ham(Lat, U)
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

    H1_k = Lat.getH1(kspace=True)
    H2_loc = Lat.getH2(kspace=False)

    # p-h transformed hamiltonian
    H1_k, H0 = dmet.transform_H1_k(H1_k)
    H2_loc, H1_from_H2_loc, H0_from_H2 = dmet.transform_H2_local(H2_loc)
    H1_k = mfd.add_H1_loc_to_k(H1_from_H2_loc, H1_k)
    H0 += H0_from_H2

    vcor = dmet.AFInitGuess(ImpSize, U, Filling, rand=0.01)
    vcor_mat = vcor.get()
    vcor_mat[1] = -vcor_mat[1].T
    vcor_mat[2] = 0.0
    vcor.assign(vcor_mat)

    vcor_param = vcor.param
    vcor_param[:] = 0.0
    vcor.update(vcor_param)

    # KUHF
    dm0_a = np.diag([0.0, Filling*2, Filling*2, 0.0])
    dm0_b = np.diag([Filling*2, 0.0, 0.0, Filling*2])
    dm0 = np.array(([dm0_a] * nkpts, [dm0_b] * nkpts))

    GRho, Mu, ires = dmet.HartreeFock(Lat, vcor, Filling, beta=beta, fix_mu=False,
                                      mu0=None, thrnelec=1e-10, scf=True,
                                      full_return=True, ires=True, verbose=4,
                                      conv_tol=1e-10, dm0=dm0)
    H0_from_vcor = vcor.get()[1].trace()

    Ham_sl = HamNonInt(Lat, H1_k, H2_loc, Fock=None, ImpJK=None, kspace_input=True,
                       spin_dim_H2=3, H0=H0)
    Lat.setHam(Ham_sl, use_hcore_as_emb_ham=use_hcore_as_emb_ham, eri_symmetry=1)
    dm0 = spinless.transform_rdm1_k(ires["rho_k"])
    GRho = Lat.k2R(dm0)
    Lat.update_Ham_ghf(GRho)

    vcor = dmet.AFInitGuess(ImpSize, U, Filling, rand=-0.15, d_wave=True, trace_zero=True, bogo_res=True)
    vcor_mat = vcor.get()
    vcor_mat[0] = 0.0
    vcor_mat[1] = 0.0

    # no symmetry is used
    Ca = [np.eye(nscsites)]
    Cb = [np.eye(nscsites)]

    vcor = dmet.VcorSymmBogo(restricted=False, bogoliubov=True, nscsites=nscsites,
                             Ca=Ca, Cb=Cb, idx_range=None, bogo_res=True)
    vcor.assign(vcor_mat)

    cisolver = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-10, ghf=True, scf_newton=False, beta=beta)
    #cisolver = dmet.impurity_solver.CCSD(restricted=restricted, tol=1e-7, tol_normt=1e-5, ghf=True, level_shift=0.05,
    #                                     diis_space=10, scf_newton=False)
    solver = cisolver

    myscf = dmet.impurity_solver.SCFSolver(restricted=restricted, tol=1e-8,
                                           max_memory=120000, ghf=True,
                                           scf_newton=False, alpha=1.0, max_cycle=100,
                                           beta=beta)

    E_old = 0.0
    conv = False
    history = dmet.IterHistory()
    dVcor_per_ele = None
    GRho_old = 0.0

    for iter in range(MaxIter):
        log.section("\nDMET Iteration %d\n", iter)

        log.section("\nsolving mean-field problem\n")
        log.result("Vcor =\n%s", vcor.get())
        log.result("Mu (guess) = %20.12f", Mu)

        if iter == 0:
            GRho, Mu, ires = dmet.GHartreeFock(Lat, vcor, Filling, mu0_elec=Mu, \
                    beta=beta, fix_mu=False, mu0=None, thrnelec=1e-10, scf=False,
                    full_return=True, verbose=1, conv_tol=1e-10)
            GRho, Mu, ires = dmet.GHartreeFock(Lat, vcor, None, mu0_elec=Mu, \
                    beta=beta, fix_mu=False, mu0=None, thrnelec=1e-10, scf=False,
                    full_return=True, verbose=1, conv_tol=1e-10)
        else:
            GRho = Lat.k2R(rdm1_fit).real

        if iter >= 3:
            GRho = adiis.update(GRho)
            dc.nDim = adiis.get_num_vec()

        GRho_k = Lat.R2k(GRho)
        dVcor_per_ele = max_abs(GRho - GRho_old)
        GRho_old = GRho.copy()
        E_mf = (ires["E"] - 0.5*H0_from_vcor) / nscsites
        log.result("Mean-field energy (per site): %s", E_mf)

        log.section("\nconstructing impurity problem\n")
        basis = spinless.get_emb_basis(Lat, GRho, localize_bath='scdm')
        ImpHam, H1e = spinless.get_emb_Ham(Lat, basis, vcor, Mu, local=True, int_bath=True)
        basis_k = Lat.R2k_basis(basis)

        # fit the chemical potential for the solver
        dm0 = dmet.foldRho_k(GRho_k, basis_k)
        nelec_target = 0.0
        for i, C in enumerate(basis):
            dm = mdot(C, dm0, C.conj().T)
            norb = dm.shape[-1] // 2
            nelec_target += dm[range(norb), range(norb)].sum() - dm[range(norb, norb*2), range(norb, norb*2)].sum() + norb

        print ("nelec target", nelec_target)

        Filling_Mu = nelec_target / float(basis.shape[-2])
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu, fit_ghf=True)

        solver_args = {"dm0": dm0, "restart": True, "basis": basis}

        GRhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(Lat, Filling_Mu, ImpHam, \
            basis, solver, \
            solver_args=solver_args, thrnelec=1e-6, \
            delta=0.1, step=0.5, fit_ghf=True)
        dmet.SolveImpHam_with_fitting.save("./frecord")
        last_dmu += dmu

        GRhoImp, EnergyImp, nelecImp = \
                dmet.transformResults(GRhoEmb, EnergyEmb, Lat, basis, ImpHam, \
                H1e, Mu, last_dmu=0.0, int_bath=False, \
                solver=solver, solver_args=solver_args)
        log.result("last_dmu : %s", last_dmu)
        log.result("E (DMET) : %s", EnergyImp)
        m_AF, m_SC = get_order_param(GRhoImp, idx=[0, 1, 2, 3], return_abs=True)
        log.result("m_AF : %s", m_AF)
        log.result("m_SC : %s", m_SC)

        dE = EnergyImp - E_old
        E_old = EnergyImp
        dump_res_iter = np.array([Mu, last_dmu, vcor.param, GRhoEmb, basis, GRhoImp], dtype = object)

        log.section("\nfitting correlation potential\n")
        rdm1_fit, err = dmet.FitVcor(GRhoEmb, Lat, basis, vcor, Mu, \
                beta=beta, CG_check=False, test_grad=False, imp_fit=imp_fit,
                fix_mu=False, mu0=0.0, method='BFGS', MaxIter1=0, \
                num_grad=True, MaxIter2=500, init_step=4.0, min_step=2.0,
                bogo_only=True, filling=Filling, use_cvx_frac=True)

        print (rdm1_fit.shape)
        break

    sys.modules.pop("libdmet.dmet.Hubbard", None)
    sys.modules.pop("libdmet.dmet.HubbardGSO", None)

sys.modules.pop("libdmet.dmet.Hubbard", None)
sys.modules.pop("libdmet.dmet.HubbardGSO", None)

if __name__ == "__main__":
    test_cvx_frac_gso()
    test_fit('CIAH')
