#!/usr/bin/env python

import pytest
import numpy as np
import scipy.linalg as la
from libdmet.lo import check_orthonormal, check_span_same_space
from libdmet.utils import max_abs, mdot
np.set_printoptions(4, linewidth=1000, suppress=True)

@pytest.mark.parametrize(
    "nelec_lat", [8, 9]
)
def test_deg(nelec_lat):
    from libdmet.routine import mfd 
    from libdmet.routine import ftsystem
    np.random.seed(1)

    norb = 16

    deg_orbs = [[0, 3, 8], [1, 2], [4, 5, 6], [7]]
    deg_energy = [1.0 , 0.1, 0.8, 3.0]
    h = ftsystem.get_h_random_deg(norb, deg_orbs=deg_orbs, deg_energy=deg_energy)
    print ("h: \n", h)

    ew, ev = la.eigh(h)
    print ("ew:\n", ew) 
    
    ewocc, mu, err = mfd.assignocc(ew, nelec_lat, beta=np.inf, mu0=0.0)
    
    print ("occ:\n", ewocc)
    rdm1_full = (ev * ewocc) @ ev.T
    
    def get_bath(rdm1_env, tol=1e-6):
        ew, ev = la.eigh(rdm1_env)
        bath = []
        e_sum = 0.0 
        for i, e in enumerate(ew):
            if abs(e) > tol and abs(1 - e) > tol:
                print (e) 
                e_sum += e
                bath.append(ev[:, i]) 
        bath = np.asarray(bath).T
        return bath

    nimp = 2
    bath = get_bath(rdm1_full[nimp:, nimp:])
    nbath = bath.shape[-1]
    basis = np.zeros((norb, nimp + nbath))
    basis[:nimp, :nimp] = np.eye(nimp)
    basis[nimp:, nimp:] = bath

    h1_emb = mdot(basis.T, h, basis)
    rdm1_emb = mdot(basis.T, rdm1_full, basis)
    
    ew, ev = la.eigh(h1_emb)
    nelec = int(np.round(rdm1_emb.trace()))
    ewocc, mu, err = mfd.assignocc(ew, nelec=nelec, beta=np.inf, mu0=0.0)
    rdm1 = (ev * ewocc) @ ev.T
    
    print ("from embedding")
    print (rdm1)
    print ("from projection")
    print (rdm1_emb)
    print ("diff rdm1 emb: ", max_abs(rdm1 - rdm1_emb))
    assert max_abs(rdm1 - rdm1_emb) < 1e-12
    print ("-" * 79)

    ## check the case 
    #ew, ev = la.eigh(h)
    #print ("ew:\n", ew) 
    #
    #ewocc, mu, err = mfd.assignocc(ew, nelec_lat, beta=np.inf, mu0=0.0)
    #
    #print ("occ:\n", ewocc)
    #nocc = np.sum(ewocc > 1e-10)
    #C_occ = ev[:, :nocc]
    #
    #u, s, vt = la.svd(C_occ[:nimp], full_matrices=True)
    #
    #C_occ_rot = C_occ @ vt.T
    #
    #print ("C_occ_rot\n", C_occ_rot)

    #print (la.norm(C_occ_rot[:nimp, nimp:]))

def test_hub1d_deg():
    from pyscf import lib
    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet
    log.verbose = "DEBUG2"
    
    U = 4.0 
    LatSize = 20
    ImpSize = 2
    Filling = 1.0 / 2
    int_bath = False
    restricted = True
    use_hcore_as_emb_ham = True
    MaxIter = 20

    Mu = U * Filling
    last_dmu = 0.0

    DiisStart = 4
    TraceStart = 3
    DiisDim = 4
    dc = dmet.FDiisContext(DiisDim)
    adiis = lib.diis.DIIS()
    adiis.space = DiisDim

    ntotal = Filling * np.prod(LatSize)
    if abs(ntotal - np.round(ntotal)) > 1e-5:
        log.warning("rounded total number of electrons to integer %d", np.round(ntotal))
        Filling=float(np.round(ntotal)) / np.prod(LatSize)

    Lat = dmet.ChainLattice(LatSize, ImpSize)
    nscsites = Lat.supercell.nsites
    Ham = dmet.Ham(Lat, U)
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)
    vcor = dmet.PMInitGuess(ImpSize, U, Filling)

    FCI = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-11, scf_newton=True)
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

        rho, Mu, res = dmet.RHartreeFock(Lat, vcor, Filling, Mu, ires=True, tol_deg=1e-3)
        E_mf = res["E"] / nscsites

        log.result("Mean-field energy (per site): %s", E_mf)

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching=False, int_bath=int_bath, kind='eig',
                tol_bath=1e-6)
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu) 
        basis_k = Lat.R2k_basis(basis)
        
        rdm1_emb = dmet.foldRho_k(res["rho_k"], basis_k)*2.0
        nelec_emb = int(np.round(rdm1_emb[0].trace()))
        
        log.section("\nsolving impurity problem\n")
        solver_args = {"nelec": nelec_emb, "dm0": rdm1_emb}
        
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
                vcor, np.inf, Filling, MaxIter2 = 0, nelec=nelec_emb//2)

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
        
        if dVcor_per_ele < 1.0e-5 and abs(dE) < 1.0e-5 and iter > 3 :
            conv = True
            break
    
    assert abs(EnergyImp - -0.550862936544) < 1e-4

    if conv:
        log.result("DMET converged")
    else:
        log.result("DMET cannot converge")

if __name__ == "__main__":
    test_deg(nelec_lat=8)
    test_hub1d_deg()
