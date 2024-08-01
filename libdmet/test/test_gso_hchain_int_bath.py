#!/usr/bin/env python

'''
Test DMET with self-consistency, GSO checked with RHF.
'''

import os
import sys
import numpy as np
import scipy.linalg as la
import pytest

@pytest.mark.parametrize(
    "full_fit", [True, False]
)
def test_gso_rhf(full_fit):
    from libdmet.routine import spinless
    from libdmet.utils import max_abs
    E_rhf, rho_rhf = t_rhf(full_fit=full_fit)
    E_gso, rho_gso = t_gso(full_fit=full_fit)
    rho_rhf = spinless.transform_rdm1_local(rho_rhf, compact=False)
    
    print ("rdm1 diff compare to RHF SCF")
    print (max_abs(rho_gso - rho_rhf))
    assert max_abs(rho_gso - rho_rhf) < 1e-5

def t_rhf(full_fit=False):
    """
    RHF H6 @ 321G.
    """
    from pyscf import lib, fci, ao2mo
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, dft, cc

    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.basis_transform import eri_transform

    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet
    from libdmet.utils import max_abs

    log.verbose = "DEBUG1"
    np.set_printoptions(4, linewidth=1000, suppress=True)
    
    ### ************************************************************
    ### System settings
    ### ************************************************************

    cell = lattice.HChain()
    cell.basis = '321G'
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 3]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts

    exxdiv = None

    ### ************************************************************
    ### DMET settings 
    ### ************************************************************

    # system
    Filling = cell.nelectron / float(Lat.nscsites*2.0)
    restricted = True
    bogoliubov = False
    int_bath = True
    nscsites = Lat.nscsites
    Mu = 0.0
    #last_dmu = 0.0
    last_dmu = 0.000349788363
    #beta = np.inf
    beta = 100.0

    # DMET SCF control
    MaxIter = 100
    u_tol = 1.0e-6
    E_tol = 1.0e-6
    iter_tol = 4

    # DIIS
    adiis = lib.diis.DIIS()
    adiis.space = 4
    diis_start = 4
    dc = dmet.FDiisContext(adiis.space)
    trace_start = 3

    # solver and mu fit
    FCI = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-12)
    solver = FCI
    nelec_tol = 1.0e-6
    delta = 0.01
    step = 0.1
    load_frecord = False

    # vcor fit
    if full_fit:
        imp_fit = True
        emb_fit_iter = 0
        full_fit_iter = 500
    else:
        imp_fit = False
        emb_fit_iter = 500 # embedding fitting
        full_fit_iter = 0

    # vcor initialization
    vcor = dmet.VcorLocal(restricted, bogoliubov, nscsites)
    z_mat = np.zeros((2, nscsites, nscsites))
    vcor.assign(z_mat)

    ### ************************************************************
    ### SCF Mean-field calculation
    ### ************************************************************

    log.section("\nSolving SCF mean-field problem\n")

    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    #if not os.path.isfile(gdf_fname):
    if True:
        gdf.build()

    chkfname = 'hchain.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        kmf.kernel()
        assert(kmf.converged)

    ### ************************************************************
    ### Pre-processing, LO and subspace partition
    ### ************************************************************

    log.section("\nPre-process, orbital localization and subspace partition\n")
    kmf = Lat.symmetrize_kmf(kmf)
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao='minao', full_return=True)
    C_ao_iao = Lat.symmetrize_lo(C_ao_iao)

    ncore = 0
    nval = C_ao_iao_val.shape[-1]
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)

    # use IAO orbital as Wannier's guess
    C_ao_lo = C_ao_iao
    Lat.set_Ham(kmf, gdf, C_ao_lo)

    ### ************************************************************
    ### DMET procedure
    ### ************************************************************

    # DMET main loop
    E_old = 0.0
    conv = False
    history = dmet.IterHistory()
    dVcor_per_ele = None
    if load_frecord:
        dmet.SolveImpHam_with_fitting.load("./frecord")

    for iter in range(MaxIter):
        log.section("\nDMET Iteration %d\n", iter)
        
        log.section("\nsolving mean-field problem\n")
        log.result("Vcor =\n%s", vcor.get())
        log.result("Mu (guess) = %20.12f", Mu)
        rho, Mu, res = dmet.RHartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)
        #Lat.update_Ham(rho*2.0)
        
        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, \
                matching=True, int_bath=int_bath, incore=True)
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)

        log.section("\nsolving impurity problem\n")
        solver_args = {"nelec": min((Lat.ncore+Lat.nval)*2, \
                cell.nelectron*nkpts), \
                "dm0": dmet.foldRho_k(res["rho_k"], basis_k)*2.0}
        rhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
            solver_args=solver_args, thrnelec=nelec_tol, \
            delta=delta, step=step)
        dmet.SolveImpHam_with_fitting.save("./frecord")
        last_dmu += dmu
        rhoImp, EnergyImp, nelecImp = \
                dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
                lattice=Lat, last_dmu=last_dmu, int_bath=int_bath, \
                solver=solver, solver_args=solver_args, add_vcor_to_E=False, \
                vcor=vcor, rebuild_veff=False)
        EnergyImp *= nscsites
        log.result("last_dmu = %20.12f", last_dmu)
        log.result("E(DMET) = %20.12f", EnergyImp)
        solver.twopdm = None
        
        #np.save("rdm1_imp_ref0", rhoImp)
        #exit()

        dump_res_iter = np.array([Mu, last_dmu, vcor.param, rhoEmb, basis, rhoImp, \
                C_ao_lo, rho, Lat.getFock(kspace=False)], dtype=object)
        np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter, allow_pickle=True)
        
        log.section("\nfitting correlation potential\n")
        vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, \
                vcor, beta, Filling, MaxIter1=emb_fit_iter,
                MaxIter2=full_fit_iter, method='CG', \
                imp_fit=imp_fit, ytol=1e-9, gtol=1e-5, test_grad=True)
        
        #if iter >= trace_start:
        #    # to avoid spiral increase of vcor and mu
        #    log.result("Keep trace of vcor unchanged")
        #    vcor_new = dmet.make_vcor_trace_unchanged(vcor_new, vcor)

        #dVcor_per_ele = la.norm(vcor_new.param - vcor.param) / (len(vcor.param))
        dVcor_per_ele = max_abs(vcor_new.param - vcor.param)
        dE = EnergyImp - E_old
        E_old = EnergyImp 
        
        if iter >= diis_start:
            pvcor = adiis.update(vcor_new.param)
            dc.nDim = adiis.get_num_vec()
        else:
            pvcor = vcor_new.param
        
        #dVcor_per_ele = la.norm(pvcor - vcor.param) / (len(vcor.param))
        dVcor_per_ele = max_abs(pvcor - vcor.param)
        vcor.update(pvcor)
        #log.result("trace of vcor: %s", \
        #        np.sum(np.diagonal((vcor.get())[:2], 0, 1, 2), axis=1))
        
        history.update(EnergyImp, err, nelecImp, dVcor_per_ele, dc)
        history.write_table()

        if dVcor_per_ele < u_tol and abs(dE) < E_tol and iter > iter_tol :
            conv = True
            break

    return EnergyImp, rhoImp

def t_gso(full_fit=False):
    """
    GSO H6 chain @ 321G.
    """
    from pyscf import lib, fci, ao2mo
    from pyscf import scf as mol_scf
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, dft, cc

    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.basis_transform import eri_transform

    from libdmet.utils import logger as log
    import libdmet.dmet.HubbardGSO as dmet
    from libdmet.routine import spinless
    from libdmet.utils import max_abs, mdot
    
    log.verbose = "DEBUG1"
    np.set_printoptions(4, linewidth=1000, suppress=True)
    
    ### ************************************************************
    ### System settings
    ### ************************************************************

    cell = lattice.HChain()
    cell.basis = '321G'
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 3]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts

    exxdiv = None
    #exxdiv = 'ewald'

    ### ************************************************************
    ### DMET settings 
    ### ************************************************************

    # system
    Filling = cell.nelectron / float(Lat.nscsites*2.0)
    restricted = True
    bogoliubov = True
    int_bath = True
    nscsites = Lat.nscsites
    Mu = 0.0
    last_dmu = 0.000349788363
    #beta = np.inf
    beta = 100.0

    # DMET SCF control
    MaxIter = 100
    u_tol = 1.0e-6
    E_tol = 1.0e-6
    iter_tol = 4

    # DIIS
    adiis = lib.diis.DIIS()
    adiis.space = 4
    diis_start = 4
    dc = dmet.FDiisContext(adiis.space)
    trace_start = 3

    # solver and mu fit
    FCI = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-12, ghf=True)
    solver = FCI
    nelec_tol = 1.0e-6
    delta = 0.01
    step = 0.1
    load_frecord = False

    # vcor fit
    if full_fit:
        imp_fit = True
        emb_fit_iter = 0
        full_fit_iter = 500
    else:
        imp_fit = False
        emb_fit_iter = 500 # embedding fitting
        full_fit_iter = 0

    # vcor initialization
    vcor = dmet.VcorLocal(restricted, bogoliubov, nscsites, ghf=True)
    
    ### ************************************************************
    ### SCF Mean-field calculation
    ### ************************************************************

    log.section("\nSolving SCF mean-field problem\n")

    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    #if not os.path.isfile(gdf_fname):
    if True:
        gdf.build()

    chkfname = 'hchain.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        kmf.kernel()
        assert(kmf.converged)

    ### ************************************************************
    ### Pre-processing, LO and subspace partition
    ### ************************************************************
    
    log.section("\nPre-process, orbital localization and subspace partition\n")
    kmf = Lat.symmetrize_kmf(kmf)
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao='minao', full_return=True)
    C_ao_iao = Lat.symmetrize_lo(C_ao_iao)
    
    ncore = 0
    nval = C_ao_iao_val.shape[-1]
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)
    
    C_ao_lo = C_ao_iao
    Lat.set_Ham(kmf, gdf, C_ao_lo)
    
    # transform everything to LO
    hcore = kmf.get_hcore()
    ovlp  = kmf.get_ovlp()
    rdm1  = kmf.make_rdm1()
    e_nuc = kmf.energy_nuc()
    
    # p-h transformed hamiltonian.
    C_sao_slo = spinless.combine_mo_coeff_k(C_ao_lo)
    ovlp_ghf = spinless.combine_mo_coeff_k(ovlp)

    H1_k = Lat.getH1(kspace=True)
    GH1, GH0 = spinless.transform_H1_k(H1_k, compact=False)
    GV1, GV0 = spinless.get_H1_H0_from_df(gdf, C_ao_lo=C_ao_lo, compact=False)
    GH1 += GV1
    GH0 += GV0
    #GH0 += e_nuc
    
    GRho_k = spinless.transform_rdm1_k(Lat.rdm1_lo_k * 0.5)
    
    # transform back to AO
    GH1_ao_k = make_basis.transform_h1_to_ao(GH1, C_sao_slo, ovlp_ghf)
    GRho_ao_k = make_basis.transform_rdm1_to_ao(GRho_k, C_sao_slo)
    
    cell.nelectron = GRho_k.shape[-1] // 2
    kmf = spinless.KGHFPH(cell, kpts, exxdiv=exxdiv).density_fit()
    kmf.with_df = gdf
    kmf.with_df._cderi = 'gdf_ints.h5'
    kmf.conv_tol = 1e-12
    kmf.max_cycle = 300
    kmf.chkfile = chkfname
    kmf.get_hcore  = lambda *args: GH1_ao_k
    kmf.get_ovlp   = lambda *args: ovlp_ghf
    kmf.energy_nuc = lambda *args: GH0 + e_nuc
    kmf.kernel(dm0=GRho_ao_k)
    
    Lat.set_Ham(kmf, gdf, C_ao_lo, H0=GH0)
    
    ### ************************************************************
    ### DMET procedure
    ### ************************************************************

    # DMET main loop
    E_old = 0.0
    conv = False
    history = dmet.IterHistory()
    dVcor_per_ele = None
    if load_frecord:
        dmet.SolveImpHam_with_fitting.load("./frecord")

    for iter in range(MaxIter):
        log.section("\nDMET Iteration %d\n", iter)
        
        log.section("\nsolving mean-field problem\n")
        log.result("Vcor =\n%s", vcor.get())
        log.result("Mu (guess) = %20.12f", Mu)
        rho, Mu, res = dmet.GHartreeFock(Lat, vcor, Filling, Mu, beta=beta, full_return=True)
        rho, Mu, res = dmet.GHartreeFock(Lat, vcor, None, Mu, beta=beta, full_return=True)
        #Lat.update_Ham(rho)
        

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, Mu, localize_bath='scdm')
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)

        log.section("\nsolving impurity problem\n")
        solver_args = {"nelec": ImpHam.H1["cd"].shape[-1] // 2, \
                "dm0": dmet.foldRho_k(res["rho_k"], basis_k)}
        rhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
            solver_args=solver_args, thrnelec=nelec_tol, \
            delta=delta, step=step)
        dmet.SolveImpHam_with_fitting.save("./frecord")
        last_dmu += dmu
        
        rhoImp, EnergyImp, nelecImp = \
                dmet.transformResults(rhoEmb, EnergyEmb, Lat, basis, ImpHam, H1e, \
                Mu, last_dmu=last_dmu, int_bath=int_bath, \
                solver=solver, solver_args=solver_args, add_vcor_to_E=False, \
                vcor=vcor, rebuild_veff=False)
        
        if iter == 0:
            rdm1_file = os.path.dirname(os.path.realpath(__file__)) + "/rdm1_imp_ref0"
            rdm1_imp_ref0 = np.load(rdm1_file)
            rdm1_imp_ref0 = spinless.transform_rdm1_local(rdm1_imp_ref0, 
                                                      compact=False)
            print ("rdm1 diff compare to RHF ref 1st iteration")
            print (max_abs(rhoImp - rdm1_imp_ref0))
            assert max_abs(rhoImp - rdm1_imp_ref0) < 1e-7

        EnergyImp *= nscsites
        log.result("last_dmu = %20.12f", last_dmu)
        log.result("E(DMET) = %20.12f", EnergyImp)
        
        #rdm1_glob_k = spinless.get_rho_glob_k(basis, Lat, rhoEmb)
        #rdm1_glob_k = res["rho_k"]
        #E_GV1 = 0.5 * np.einsum('kpq, kqp ->', GV1, rdm1_glob_k) / nkpts

        solver.twopdm = None

        dump_res_iter = np.array([Mu, last_dmu, vcor.param, rhoEmb, basis,
                                  rhoImp, C_ao_lo, rho, 
                                  Lat.getFock(kspace=False)], dtype=object)
        np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter, allow_pickle=True)
        
        log.section("\nfitting correlation potential\n")
        vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis,
                   vcor, Mu, beta=beta, MaxIter1=emb_fit_iter,
                   MaxIter2=full_fit_iter, kinetic=False, CG_check=False, BFGS=False, 
                   serial=True, method='CG', ytol=1e-9, gtol=1e-5,
                   num_grad=False, imp_fit=imp_fit, test_grad=True)
        
        #if iter >= trace_start:
        #    # to avoid spiral increase of vcor and mu
        #    log.result("Keep trace of vcor unchanged")
        #    vcor_new = dmet.make_vcor_trace_unchanged(vcor_new, vcor)

        #dVcor_per_ele = la.norm(vcor_new.param - vcor.param) / (len(vcor.param))
        dE = EnergyImp - E_old
        E_old = EnergyImp 
        
        if iter >= diis_start:
            pvcor = adiis.update(vcor_new.param)
            dc.nDim = adiis.get_num_vec()
        else:
            pvcor = vcor_new.param
        
        #dVcor_per_ele = la.norm(pvcor - vcor.param) / (len(vcor.param))
        dVcor_per_ele = max_abs(pvcor - vcor.param)
        vcor.update(pvcor)
        #log.result("trace of vcor: %s", \
        #        np.sum(np.diagonal((vcor.get())[:2], 0, 1, 2), axis=1))
        
        history.update(EnergyImp, err, nelecImp, dVcor_per_ele, dc)
        history.write_table()

        if dVcor_per_ele < u_tol and abs(dE) < E_tol and iter > iter_tol :
            conv = True
            break
    sys.modules.pop("libdmet.dmet.Hubbard", None)
    return EnergyImp, rhoImp

if __name__ == "__main__":
    test_gso_rhf(full_fit=True)
