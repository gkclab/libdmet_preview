#!/usr/bin/env python

'''
Test UCCSD as DMET solver.
'''

import os, sys
import numpy as np
import scipy.linalg as la
import pytest

@pytest.mark.parametrize(
    "incore", [True, False]
)
def test_ucc_solver(incore):

    from pyscf import lib
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, cc, tools

    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.lo.iao import reference_mol

    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet

    log.verbose = "DEBUG2"
    np.set_printoptions(4, linewidth=1000, suppress=True)
    name = "ucc"

    ### ************************************************************
    ### System settings
    ### ************************************************************

    max_memory = 4000 # 4 G
    cell = gto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''
    cell.atom = ''' H 5.0      5.0      0.75
                    H 5.0      5.0      2.25 '''
    cell.basis = '321g'
    cell.verbose = 5
    cell.max_memory = max_memory
    cell.precision = 1e-13
    cell.build(unit='Angstrom')

    cell_mesh = [1, 1, 1]
    ncell_sc = np.prod(cell_mesh)
    cell = tools.pbc.super_cell(cell, cell_mesh)
    natom_sc = cell.natm

    kmesh = [1, 1, 3]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts

    minao = 'MINAO'
    pmol = reference_mol(cell, minao=minao)
    ncore = 0
    nval = pmol.nao_nr()
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)

    exxdiv = None
    kmf_conv_tol = 1e-12
    kmf_max_cycle = 300

    gdf_fname = '%s_gdf_ints.h5'%name
    chkfname = '%s.chk'%name

    ### ************************************************************
    ### DMET settings 
    ### ************************************************************

    # system
    Filling = cell.nelectron / (Lat.nscsites*2.0)
    restricted = False
    bogoliubov = False
    int_bath = True
    add_vcor = False
    nscsites = Lat.nscsites
    Mu = 0.0
    last_dmu = -0.006707758235
    #beta = np.inf
    beta = 1000.0

    # DMET SCF control
    MaxIter = 100
    u_tol = 1.0e-4
    E_tol = 1.0e-5 # energy diff per orbital
    iter_tol = 4

    # DIIS
    adiis = lib.diis.DIIS()
    diis_start = 300 # ZHC NOTE FIXME the DIIS
    adiis.space = 4
    dc = dmet.FDiisContext(adiis.space)
    trace_start = 3

    # solver and mu fit
    ncas = nscsites + nval
    nelecas = min((Lat.ncore+Lat.nval)*2, nkpts*cell.nelectron)
    fci_etol = natom_sc * 1e-13
    cc_etol = natom_sc * 1e-8
    cc_ttol = 1e-5
    cisolver = dmet.impurity_solver.CCSD(restricted=restricted, tol=cc_etol, tol_normt=cc_ttol, max_memory=max_memory)
    #cisolver = dmet.impurity_solver.FCI(restricted=restricted, tol=fci_etol, max_memory=max_memory)
    #solver = dmet.impurity_solver.CASCI(ncas=ncas, nelecas=nelecas, \
    #            splitloc=False, cisolver=fcisolver, mom_reorder=True, tmpDir="./tmp")
    solver = cisolver
    nelec_tol = 2.5e-6 # per orbital
    delta = 0.01
    step = 0.1
    load_frecord = False

    # vcor fit
    imp_fit = False
    emb_fit_iter = 300 # embedding fitting
    full_fit_iter = 0
    ytol = 1e-6
    gtol = 1e-3
    CG_check = False

    # vcor initialization
    vcor = dmet.VcorLocal(restricted, bogoliubov, nscsites)
    z_mat = np.zeros((2, nscsites, nscsites))
    z_mat[0, 0, 0] -= 0.1
    z_mat[0, 1, 1] += 0.1
    z_mat[1, 0, 0] += 0.1
    z_mat[1, 1, 1] -= 0.1
    vcor.assign(z_mat)

    ### ************************************************************
    ### SCF Mean-field calculation
    ### ************************************************************

    log.section("\nSolving SCF mean-field problem\n")

    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    #if not os.path.isfile(gdf_fname):
    if True:
        gdf.build()

    #if os.path.isfile(chkfname):
    if False:
        kmf = scf.KUHF(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = kmf_conv_tol
        kmf.max_cycle = kmf_max_cycle
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KUHF(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = kmf_conv_tol
        kmf.max_cycle = kmf_max_cycle
        kmf.chkfile = chkfname
        dm0_a = np.diag([0.5, 0.5, 0.0, 0.0])
        dm0_b = np.diag([0.0, 0.0, 0.5, 0.5])
        dm0 = np.asarray([[dm0_a]*3, [dm0_b]*3])
        dm0 = None
        kmf.kernel(dm0=dm0)
        assert kmf.converged

    log.result("kmf electronic energy: %20.12f", (kmf.e_tot - kmf.energy_nuc())/ncell_sc)

    ### ************************************************************
    ### Pre-processing, LO and subspace partition
    ### ************************************************************

    log.section("\nPre-process, orbital localization and subspace partition\n")
    # IAO
    S_ao_ao = kmf.get_ovlp()
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt, lo_labels = \
            make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao, full_return=True,
                                       return_labels=True)
    C_ao_iao = Lat.symmetrize_lo(C_ao_iao)

    assert(nval == C_ao_iao_val.shape[-1])
    C_ao_mo = np.asarray(kmf.mo_coeff)

    # use IAO
    C_ao_lo = C_ao_iao
    Lat.set_Ham(kmf, gdf, C_ao_lo, eri_symmetry=4)

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

    fm = open('m-H-chain.dat', 'w')

    for iter in range(MaxIter):
        log.section("\nDMET Iteration %d\n", iter)
        
        log.section("\nsolving mean-field problem\n")
        log.result("Vcor =\n%s", vcor.get())
        log.result("Mu (guess) = %s", Mu)
        rho, Mu, res = dmet.HartreeFock(Lat, vcor, [Filling, Filling], Mu,
                                        beta=beta, ires=True, labels=lo_labels)

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching=True, int_bath=int_bath, add_vcor=add_vcor,\
                max_memory=max_memory, incore=incore)
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)

        log.section("\nsolving impurity problem\n")
        if iter < 1:
            restart = False
        else:
            restart = True
        solver_args = {"restart": restart, "basis": basis, "nelec": min((Lat.ncore+Lat.nval)*2, nkpts*cell.nelectron), \
                "dm0": dmet.foldRho_k(res["rho_k"], basis_k), 'ccd': False}
        
        rhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
            solver_args=solver_args, thrnelec=nelec_tol, \
            delta=delta, step=step)

        dmet.SolveImpHam_with_fitting.save("./frecord")
        last_dmu += dmu
        rhoImp, EnergyImp, nelecImp = \
            dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
            lattice=Lat, last_dmu=last_dmu, int_bath=int_bath, \
            solver=solver, solver_args=solver_args, labels=lo_labels)
        
        r_a = rhoImp[0][0, 0]
        r_b = rhoImp[1][0, 0]
        m = 0.5*(r_a - r_b)
        fm.write('%s %s\n'%(iter, m))
        
        
        E_DMET_per_cell = EnergyImp*nscsites / ncell_sc
        log.result("last_dmu = %20.12f", last_dmu)
        log.result("E(DMET) = %20.12f", E_DMET_per_cell)
        
        # DUMP results:
        dump_res_iter = np.array([Mu, last_dmu, vcor.param, rhoEmb, basis, rhoImp, C_ao_lo], dtype=object)
        np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter)
        
        log.section("\nfitting correlation potential\n")
        vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, \
                vcor, beta, Filling, MaxIter1=emb_fit_iter, \
                MaxIter2=full_fit_iter, method='SD', \
                imp_fit=imp_fit, ytol=ytol, gtol=gtol, CG_check=CG_check, \
                serial=True, test_grad=True)

        if iter >= trace_start:
            # to avoid spiral increase of vcor and mu
            log.result("Keep trace of vcor unchanged")
            vcor_new = dmet.make_vcor_trace_unchanged(vcor_new, vcor)

        dVcor_per_ele = np.max(np.abs(vcor_new.param - vcor.param))
        dE = EnergyImp - E_old
        E_old = EnergyImp 
        
        if iter >= diis_start:
            pvcor = adiis.update(vcor_new.param)
            dc.nDim = adiis.get_num_vec()
        else:
            pvcor = vcor_new.param
        
        dVcor_per_ele = np.max(np.abs(pvcor - vcor.param))
        vcor.update(pvcor)
        log.result("Trace of vcor: %s", np.sum(np.diagonal((vcor.get())[:2], 0, 1, 2), axis=1))
        
        history.update(E_DMET_per_cell, err, nelecImp, dVcor_per_ele, dc)
        history.write_table()
        
        if dVcor_per_ele < u_tol and abs(dE) < E_tol and iter > iter_tol :
            conv = True
            break

    fm.close()
    if conv:
        log.result("DMET converge.")
    else:
        log.result("DMET does not converge.")

    ### ************************************************************
    ### compare with KCCSD
    ### ************************************************************

    log.section("Reference Energy")
    mycc = cc.KUCCSD(kmf)
    mycc.kernel()
    log.result("KUCCSD energy (per unit cell)")
    log.result("%20.12f", mycc.e_tot - cell.energy_nuc())

if __name__ == "__main__":
    test_ucc_solver(incore=False)
