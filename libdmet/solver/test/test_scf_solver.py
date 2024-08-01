#! /usr/bin/env python

"""
Test SCFSolver.
"""

def test_rhf_solver():
    import numpy as np
    import pyscf
    from pyscf import ao2mo
    from libdmet.system.integral import Integral
    from libdmet.utils.misc import mdot, max_abs
    from libdmet.solver import impurity_solver 

    np.set_printoptions(3, linewidth=1000, suppress=True)

    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = 'ccpvdz',
    )

    myhf = mol.HF()
    myhf.kernel()
    E_ref = myhf.e_tot
    rdm1_ref = myhf.make_rdm1()

    restricted = True
    bogoliubov = False
    norb = mol.nao_nr()
    H1 = myhf.get_hcore()[None]
    ovlp = myhf.get_ovlp()
    H2 = myhf._eri[None]
    H0 = myhf.energy_nuc()

    Ham = Integral(norb, restricted, bogoliubov, H0, {"cd": H1}, \
            {"ccdd": H2}, ovlp=ovlp)

    solver = impurity_solver.SCFSolver(restricted=restricted, tol=1e-10)
    rdm1, E = solver.run(Ham, nelec=mol.nelectron, dm0=myhf.make_rdm1())
    assert abs(E - E_ref) < 1e-8
    assert max_abs(rdm1 - rdm1_ref * 0.5) < 1e-7

    rdm2 = solver.make_rdm2(ao_repr=True)
    E_from_rdm = np.einsum('pq, qp ->', H1[0], rdm1[0] * 2.0) + \
            0.5 * np.einsum('pqrs, pqrs ->', ao2mo.restore(1, H2[0], norb), \
            rdm2[0]) + H0
    assert abs(E_from_rdm - E_ref) < 1e-8

def test_uhf_solver():
    import os
    import numpy as np
    
    from pyscf import lib
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, cc, tools

    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.lo.iao import reference_mol
    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet

    log.verbose = "DEBUG2"
    np.set_printoptions(3, linewidth=1000, suppress=False)
    name = "uhf"

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
    cell.precision = 1e-10
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

    minao = 'minao'
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
    last_dmu = 0.0
    beta = np.inf
    #beta = 1000.0

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
    cisolver = dmet.impurity_solver.SCFSolver(restricted=restricted, tol=cc_etol, max_memory=max_memory)
    solver = cisolver
    nelec_tol = 2.5e-6 # per orbital
    delta = 0.01
    step = 0.1
    load_frecord = False

    # vcor fit
    imp_fit = False
    emb_fit_iter = 200 # embedding fitting
    full_fit_iter = 0
    ytol = 1e-9
    gtol = 1e-5 
    CG_check = False

    # vcor initialization
    vcor = dmet.VcorLocal(restricted, bogoliubov, nscsites)
    z_mat = np.zeros((2, nscsites, nscsites))
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
        kmf = scf.KUHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = kmf_conv_tol
        kmf.max_cycle = kmf_max_cycle
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KUHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = kmf_conv_tol
        #kmf.max_cycle = 1
        kmf.chkfile = chkfname
        dm0_a = np.diag([0.5, 0.5, 0.0, 0.0])
        dm0_b = np.diag([0.0, 0.0, 0.5, 0.5])
        dm0 = np.asarray([[dm0_a]*nkpts, [dm0_b]*nkpts])
        kmf.kernel(dm0=dm0)

    log.result("kmf electronic energy: %20.12f", (kmf.e_tot - kmf.energy_nuc())/ncell_sc)

    ### ************************************************************
    ### Pre-processing, LO and subspace partition
    ### ************************************************************

    log.section("\nPre-process, orbital localization and subspace partition\n")
    # IAO
    S_ao_ao = kmf.get_ovlp()
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao, full_return=True)
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

    for iter in range(MaxIter):
        log.section("\nDMET Iteration %d\n", iter)
        
        log.section("\nsolving mean-field problem\n")
        log.result("Vcor =\n%s", vcor.get())
        log.result("Mu (guess) = %s", Mu)
        rho, Mu, res = dmet.HartreeFock(Lat, vcor, [Filling, Filling], Mu, beta=beta, ires=True)
        #Lat.update_Ham(rho, rdm1_lo_k=res["rho_k"])
        #Lat.rdm1_lo_k = res["rho_k"]
        #Lat.rdm1_lo_R = rho

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching=True, int_bath=int_bath, add_vcor=add_vcor,\
                max_memory=max_memory, t_reversal_symm=False, incore=True)
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)

        log.section("\nsolving impurity problem\n")
        if iter < 3:
            restart = False
        else:
            solver.optimized = False
            #restart = (dVcor_per_ele < 1e-4)
            restart = False
        solver_args = {"restart": restart, "basis": basis, "nelec": min((Lat.ncore+Lat.nval)*2, nkpts*cell.nelectron), \
                "dm0": dmet.foldRho_k(Lat.rdm1_lo_k, basis_k)}
        
        rhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
            solver_args=solver_args, thrnelec=1e+5, \
            delta=delta, step=step)
        dmet.SolveImpHam_with_fitting.save("./frecord")
        last_dmu += dmu
        
        rhoEmb = dmet.foldRho_k(Lat.rdm1_lo_k, basis_k)
        solver.onepdm = rhoEmb

        rhoImp, EnergyImp, nelecImp = \
            dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
            lattice=Lat, last_dmu=last_dmu, int_bath=int_bath, \
            solver=solver, solver_args=solver_args)
        
        E_DMET_per_cell = EnergyImp*nscsites / ncell_sc
        log.result("last_dmu = %20.12f", last_dmu)
        log.result("E(DMET) = %20.12f", E_DMET_per_cell)
        
        assert abs(E_DMET_per_cell - (kmf.e_tot - kmf.energy_nuc())/ncell_sc) < 1e-8
        assert abs(E_DMET_per_cell - res["E"]) < 1e-8
        break

def test_oomp2_solver():
    import numpy as np
    from pyscf import gto, scf, ao2mo
    from libdmet_solid.system import integral
    from libdmet_solid.solver import scf_solver
    from libdmet_solid.utils.misc import tile_eri
    from libdmet.utils import logger as log 
    np.set_printoptions(3, linewidth=1000)
    log.verbose = "DEBUG2"
    
    mol = gto.M(atom = 'H 0 0 0; F 0 0 1.2',
                basis = 'ccpvdz',
                verbose = 4)
    mf = scf.RHF(mol).run()
    
    mf = scf.addons.convert_to_ghf(mf)
    dm0 = mf.make_rdm1()
    mf.kernel(dm0=dm0)
    
    ovlp = mf.get_ovlp()
    H0 = mf.energy_nuc()
    H1 = mf.get_hcore()
    H2 = mf._eri
    H2 = ao2mo.restore(4, H2, mol.nao_nr())
    
    H2 = tile_eri(H2, H2, H2)
    dm0 = mf.make_rdm1()
    
    Ham = integral.Integral(H1.shape[-1], True, False, H0, {"cd": H1[None]},
                            {"ccdd": H2[None]}, ovlp=ovlp)
    
    solver = scf_solver.SCFSolver(ghf=True, tol=1e-10, max_cycle=200,
                                  oomp2=True, tol_normt=1e-6, ci_conv_tol=1e-8,
                                  level_shift=0.1, restart=True, mc_conv_tol=1e-6)
    
    rdm1, E = solver.run(Ham, nelec=mol.nelectron)
    
    print ("diff to ROOMP2", abs(E - -100.176641888785))
    assert abs(E - -100.176641888785) < 1e-7

def test_uooccd_solver():
    import numpy as np
    from pyscf import gto, scf, ao2mo
    from libdmet_solid.system import integral
    from libdmet_solid.solver import scf_solver
    from libdmet_solid.utils.misc import tile_eri
    from libdmet.utils import logger as log 
    np.set_printoptions(3, linewidth=1000)
    log.verbose = "DEBUG2"
    
    mol = gto.M(atom = 'H 0 0 0; F 0 0 1.2',
                basis = 'ccpvdz',
                verbose = 4)
    mf = scf.RHF(mol).run()
    
    mf = scf.addons.convert_to_uhf(mf)
    dm0 = mf.make_rdm1()
    mf.kernel(dm0=dm0)
    
    ovlp = mf.get_ovlp()
    H0 = mf.energy_nuc()
    H1 = mf.get_hcore()
    H2 = mf._eri
    H2 = ao2mo.restore(4, H2, mol.nao_nr())
    
    dm0 = mf.make_rdm1()
    
    Ham = integral.Integral(H1.shape[-1], False, False, H0, {"cd": H1[None]},
                            {"ccdd": H2[None]}, ovlp=ovlp)
    
    solver = scf_solver.SCFSolver(restricted=False, ghf=False, tol=1e-13, max_cycle=200,
                                  ooccd=True, ci_conv_tol=1e-9, tol_normt=1e-7,
                                  level_shift=0.05, mc_conv_tol=1e-6, ci_diis_space=10,
                                  restart=True, approx_l=False, fix_fcivec=False, scf_newton=False)
    
    rdm1, E = solver.run(Ham, nelec=mol.nelectron, scf_max_cycle=0, dm0=dm0)
    E_ref = -100.180073031997
    print ("diff to ref", abs(E - E_ref))
    assert abs(E - E_ref) < 1e-7

def test_gooccd_solver():
    import numpy as np
    from pyscf import gto, scf, ao2mo
    from libdmet_solid.system import integral
    from libdmet_solid.solver import scf_solver
    from libdmet_solid.utils.misc import tile_eri
    from libdmet.utils import logger as log 
    np.set_printoptions(3, linewidth=1000)
    log.verbose = "DEBUG2"
    
    mol = gto.M(atom = 'H 0 0 0; F 0 0 1.2',
                basis = 'ccpvdz',
                verbose = 4)
    mf = scf.RHF(mol).run()
    
    mf = scf.addons.convert_to_ghf(mf)
    dm0 = mf.make_rdm1()
    mf.kernel(dm0=dm0)
    
    ovlp = mf.get_ovlp()
    H0 = mf.energy_nuc()
    H1 = mf.get_hcore()
    H2 = mf._eri
    H2 = ao2mo.restore(4, H2, mol.nao_nr())
    
    H2 = tile_eri(H2, H2, H2)
    dm0 = mf.make_rdm1()
    
    Ham = integral.Integral(H1.shape[-1], True, False, H0, {"cd": H1[None]},
                            {"ccdd": H2[None]}, ovlp=ovlp)
    
    solver = scf_solver.SCFSolver(ghf=True, tol=1e-13, max_cycle=200,
                                  ooccd=True, ci_conv_tol=1e-9, tol_normt=1e-7,
                                  level_shift=0.05, mc_conv_tol=1e-6, ci_diis_space=10,
                                  restart=True, approx_l=False, fix_fcivec=False, scf_newton=False)
    
    rdm1, E = solver.run(Ham, nelec=mol.nelectron, scf_max_cycle=0, dm0=dm0)
    E_ref = -100.180073031997
    print ("diff to ref", abs(E - E_ref))
    assert abs(E - E_ref) < 1e-7

def test_hf_scaled_beta():
    import numpy as np
    from scipy import linalg as la
    from pyscf import gto, scf, ao2mo
    from libdmet_solid.system import integral
    from libdmet_solid.solver import scf_solver
    from libdmet_solid.utils.misc import tile_eri
    from libdmet.utils import logger as log 
    np.set_printoptions(3, linewidth=1000)
    log.verbose = "DEBUG2"
    
    mol = gto.M(atom = 'H 0 0 0; F 0 0 1.2',
                basis = 'ccpvdz',
                verbose = 4)
    mf = scf.RKS(mol)
    mf.xc = 'HF*0.25'
    from libdmet.routine import pbc_helper as pbc_hp
    #beta = np.inf
    beta = 10.0
    mf = pbc_hp.smearing_(mf, sigma=1.0/beta)
    mf.conv_tol = 1e-11
    E_ref = mf.kernel()
    
    nao = mol.nao_nr()
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    eri = ao2mo.restore(4, mf._eri, nao)
    H0 = mf.energy_nuc()
    rdm1 = mf.make_rdm1()
    
    # UHF
    Ham = integral.Integral(hcore.shape[-1], True, False, H0, {"cd": hcore[None]},
                            {"ccdd": eri[None]}, ovlp=ovlp)
    
    solver = scf_solver.SCFSolver(ghf=False, restricted=False, tol=1e-7, max_cycle=200,
                                  scf_newton=False, alpha=0.25, beta=beta)
    
    _, E = solver.run(Ham, nelec=mol.nelectron, scf_max_cycle=200,
                         dm0=np.array((rdm1, rdm1))* 0.5)
    
    print ("diff to ref", abs(E - E_ref))
    assert abs(E - E_ref) < 1e-7

    # GHF
    H1 = la.block_diag(hcore, hcore)
    ovlp = la.block_diag(ovlp, ovlp)
    H2 = tile_eri(eri, eri, eri)
    rdm1 = la.block_diag(rdm1, rdm1) * 0.5
    
    Ham = integral.Integral(H1.shape[-1], True, False, H0, {"cd": H1[None]},
                            {"ccdd": H2[None]}, ovlp=ovlp)
    
    solver = scf_solver.SCFSolver(ghf=True, tol=1e-8, max_cycle=200,
                                  scf_newton=False, alpha=0.25, beta=beta)
    
    rdm1, E = solver.run(Ham, nelec=mol.nelectron, scf_max_cycle=200, dm0=rdm1)
    
    print ("diff to ref", abs(E - E_ref))
    assert abs(E - E_ref) < 1e-7

def test_rhf_beta():
    import numpy as np
    from scipy import linalg as la
    from pyscf import gto, scf, ao2mo
    from libdmet_solid.system import integral
    from libdmet_solid.solver import scf_solver
    from libdmet_solid.utils.misc import tile_eri
    from libdmet.utils import logger as log 
    np.set_printoptions(3, linewidth=1000)
    log.verbose = "DEBUG2"
    
    mol = gto.M(atom = 'H 0 0 0; F 0 0 1.2',
                basis = 'ccpvdz',
                verbose = 4)
    mf = scf.RKS(mol)
    mf.xc = 'HF*0.5'
    from libdmet.routine import pbc_helper as pbc_hp
    #beta = np.inf
    beta = 10.0
    mf = pbc_hp.smearing_(mf, sigma=1.0/beta)
    mf.conv_tol = 1e-11
    E_ref = mf.kernel()
    
    nao = mol.nao_nr()
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    eri = ao2mo.restore(4, mf._eri, nao)
    H0 = mf.energy_nuc()
    rdm1 = mf.make_rdm1()
    
    # RHF
    Ham = integral.Integral(hcore.shape[-1], True, False, H0, {"cd": hcore[None]},
                            {"ccdd": eri[None]}, ovlp=ovlp)
    
    solver = scf_solver.SCFSolver(ghf=False, restricted=True, tol=1e-7, max_cycle=200,
                                  scf_newton=False, alpha=0.5, beta=beta)
    
    _, E = solver.run(Ham, nelec=mol.nelectron, scf_max_cycle=200,
                      dm0=rdm1)
    
    print ("diff to ref", abs(E - E_ref))
    assert abs(E - E_ref) < 1e-7

if __name__ == "__main__":
    test_uooccd_solver()
    test_gooccd_solver()
    test_rhf_beta()
    test_hf_scaled_beta()
    test_oomp2_solver()
    test_rhf_solver()
    test_uhf_solver()
