#!/usr/bin/env python

'''
Test DMET with non-interacting bath.
'''

import os, sys
import numpy as np
import scipy.linalg as la

def test_non_int_bath_uhf():
    from pyscf import lib, fci, ao2mo
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, dft, cc

    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.basis_transform import eri_transform
    from libdmet.lo.iao import reference_mol

    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet

    log.verbose = "DEBUG1"
    np.set_printoptions(4, linewidth=1000, suppress=True)

    ### ************************************************************
    ### System settings
    ### ************************************************************

    cell = gto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''
    cell.atom = ''' H 5.0      5.0      0.75
                    H 5.0      5.0      2.25 '''
    #cell.a = ''' 10.0    0.0     0.0
    #             0.0     10.0    0.0
    #             0.0     0.0     6.0 '''
    #cell.atom = ''' H 5.0      5.0      0.75
    #                H 5.0      5.0      2.25
    #                H 5.0      5.0      3.75 
    #                H 5.0      5.0      5.25'''
    cell.basis = '321g'
    cell.verbose = 4
    cell.precision = 1e-13
    cell.build(unit='Angstrom', dump_input=False)

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
    #exxdiv = 'ewald'

    ### ************************************************************
    ### DMET settings 
    ### ************************************************************

    # system
    Filling = cell.nelectron / float(Lat.nscsites*2.0)
    restricted = False
    bogoliubov = False
    int_bath = False
    nscsites = Lat.nscsites
    Mu = 0.0
    last_dmu = 0.0
    #beta = 1000.0
    beta = np.inf

    # DMET SCF control
    MaxIter = 100
    u_tol = 5.0e-5
    E_tol = 5.0e-6
    iter_tol = 4

    # DIIS
    adiis = lib.diis.DIIS()
    adiis.space = 4
    diis_start = 4
    dc = dmet.FDiisContext(adiis.space)
    trace_start = 3

    # solver and mu fit
    ncas = nscsites + nval
    nelecas = (Lat.ncore+Lat.nval)*2
    #cisolver = dmet.impurity_solver.CCSD(restricted=True, tol=1e-9, tol_normt=1e-6)
    cisolver = dmet.impurity_solver.FCI(restricted=restricted)
    solver = cisolver
    nelec_tol = 5.0e-6
    delta = 0.01
    step = 0.1
    load_frecord = False

    # vcor fit
    imp_fit = False
    #imp_fit = True
    emb_fit_iter = 300 # embedding fitting
    full_fit_iter = 0
    #emb_fit_iter = 0 
    #full_fit_iter = 100 # full fitting
    ytol = 1e-8
    gtol = 1e-4 
    CG_check = True

    # vcor initialization
    vcor = dmet.VcorLocal(restricted, bogoliubov, nscsites)
    z_mat = np.zeros((2, nscsites, nscsites))
    #np.random.seed(1)
    #z_mat = np.random.random((nscsites, nscsites))
    #z_mat = z_mat + z_mat.T
    #z_mat = np.asarray([z_mat]*2)
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
        kmf = scf.KUHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KUHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        dm0_a = np.diag([0.5, 0.5, 0.0, 0.0])
        dm0_b = np.diag([0.0, 0.0, 0.5, 0.5])
        dm0 = np.asarray([[dm0_a]*3, [dm0_b]*3])
        #dm0 = None
        kmf.kernel(dm0=dm0)
        assert(kmf.converged)

    log.result("kmf electronic energy: %20.12f", kmf.e_tot - kmf.energy_nuc())

    ### ************************************************************
    ### Pre-processing, LO and subspace partition
    ### ************************************************************

    log.section("\nPre-process, orbital localization and subspace partition\n")
    # IAO
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao, full_return=True)
    assert(nval == C_ao_iao_val.shape[-1])

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

    for iter in range(MaxIter):
        log.section("\nDMET Iteration %d\n", iter)
        
        log.section("\nsolving mean-field problem\n")
        log.result("Vcor =\n%s", vcor.get())
        log.result("Mu (guess) = %20.12f", Mu)
        rho, Mu, res = dmet.HartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True)
        Lat.update_Ham(rho)

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, 
                matching=True, int_bath=int_bath, add_vcor=True, incore=False)
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)

        log.section("\nsolving impurity problem\n")
        if iter < 3:
            restart = False
        else:
            solver.optimized = False
            restart = (dVcor_per_ele < 1e-3)
        solver_args = {"restart": restart, "nelec": (Lat.ncore+Lat.nval)*2, \
                "dm0": dmet.foldRho_k(res["rho_k"], basis_k)}

        rhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
            solver_args=solver_args, thrnelec=nelec_tol, \
            delta=delta, step=step)
        dmet.SolveImpHam_with_fitting.save("./frecord")
        last_dmu += dmu
        rhoImp, EnergyImp, nelecImp = \
            dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
            lattice=Lat, last_dmu=last_dmu, int_bath=int_bath, \
            solver=solver, solver_args=solver_args)
         
        EnergyImp *= nscsites
        log.result("last_dmu = %20.12f", last_dmu)
        log.result("E(DMET) = %20.12f", EnergyImp)
        print ("diff: ", abs(EnergyImp - -1.238248899089))
        assert abs(EnergyImp - -1.238248899089) < 1e-7
        return 

if __name__ == "__main__":
    test_non_int_bath_uhf()
