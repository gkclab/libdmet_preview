#!/usr/bin/env python

'''
Test DMET with self-consistency.
Currently using NIB.
'''

def test_compare_rhf_ghf_nib():
    """
    Compare rdm1 and vcor between RHF- and GHF-based DMET.
    """
    from libdmet.routine import spinless
    from libdmet.utils.misc import max_abs
    rdm1_rhf, vcor_rhf = t_self_consistency_rhf()
    rdm1_rhf = rdm1_rhf[0]
    nao = rdm1_rhf.shape[-1]
    nso = nao * 2
    rdm1_ghf, vcor_ghf = t_self_consistency_ghf()
    rdm1_aa, rdm1_bb, rdm1_ab = spinless.extract_rdm1(rdm1_ghf)
    diff_aa = max_abs(rdm1_aa - rdm1_rhf)
    diff_bb = max_abs(rdm1_bb - rdm1_rhf)
    diff_ab = max_abs(rdm1_ab)
    print ("diff rdm1_aa with rdm1_rhf: ", diff_aa)
    print ("diff rdm1_bb with rdm1_rhf: ", diff_bb)
    print ("diff rdm1_ab with 0: ", diff_ab)
    assert diff_aa < 2e-5
    assert diff_bb < 2e-5
    assert diff_ab < 5e-5

def t_self_consistency_rhf():
    """
    Test self consistency with H6 321G, FCI@RHF.
    """
    import os, sys
    import numpy as np
    import scipy.linalg as la

    from pyscf import lib, fci, ao2mo
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, dft, cc

    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.basis_transform import eri_transform
    from libdmet.routine import slater

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
    cell.basis = '321g'
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
    int_bath = False
    nscsites = Lat.nscsites
    Mu = 0.0
    last_dmu = 0.0
    beta = np.inf
    #beta = 1000.0

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
    trace_start = 3000

    # solver and mu fit
    FCI = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-12)
    solver = FCI
    nelec_tol = 5.0e-6
    delta = 0.01
    step = 0.1
    load_frecord = False

    # vcor fit
    imp_fit = False
    emb_fit_iter = 1000 # embedding fitting
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
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
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
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao='minao', full_return=True)
    C_ao_iao = Lat.symmetrize_lo(C_ao_iao)

    ncore = 0
    nval = C_ao_iao_val.shape[-1]
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)

    # use IAO orbital as Wannier's guess
    C_ao_lo = C_ao_iao
    Lat.set_Ham(kmf, gdf, C_ao_lo)

    # compute and store the H2_emb and reuse it
    H2_unit = eri_transform.get_unit_eri(cell, gdf, C_ao_lo=C_ao_lo, symmetry=4)
    neo = Lat.nao + Lat.nval
    H2_emb = slater.unit2emb(H2_unit, neo)
    H2_unit = None

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

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, \
                matching=True, int_bath=int_bath, H2_given=H2_emb)
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
                solver=solver, solver_args=solver_args, add_vcor_to_E=False, vcor=vcor)
        EnergyImp *= nscsites
        log.result("last_dmu = %20.12f", last_dmu)
        log.result("E(DMET) = %20.12f", EnergyImp)
        solver.twopdm = None

        dump_res_iter = np.array([Mu, last_dmu, vcor.param, rhoEmb, basis, rhoImp, \
                C_ao_lo, rho, Lat.getFock(kspace=False)], dtype=object)
        np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter, allow_pickle=True)

        log.section("\nfitting correlation potential\n")
        vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, \
                vcor, beta, Filling, MaxIter1=emb_fit_iter, \
                MaxIter2=full_fit_iter, method='CG', \
                imp_fit=imp_fit, ytol=1e-8, gtol=1e-4)

        #if iter >= trace_start:
        #    # to avoid spiral increase of vcor and mu
        #    log.result("Keep trace of vcor unchanged")
        #    ddiagV = np.average(np.diagonal(\
        #            (vcor_new.get()-vcor.get())[:2], 0, 1, 2))
        #    vcor_new = dmet.addDiag(vcor_new, -ddiagV)

        dVcor_per_ele = la.norm(vcor_new.param - vcor.param) / (len(vcor.param))
        dE = EnergyImp - E_old
        E_old = EnergyImp

        if iter >= diis_start:
            pvcor = adiis.update(vcor_new.param)
            dc.nDim = adiis.get_num_vec()
        else:
            pvcor = vcor_new.param

        dVcor_per_ele = la.norm(pvcor - vcor.param) / (len(vcor.param))
        vcor.update(pvcor)
        log.result("trace of vcor: %20.12f ", np.sum(np.diagonal((vcor.get())[:2], 0, 1, 2)))

        history.update(EnergyImp, err, nelecImp, dVcor_per_ele, dc)
        history.write_table()

        if dVcor_per_ele < u_tol and abs(dE) < E_tol and iter > iter_tol :
            conv = True
            break
    assert abs(EnergyImp - (-1.250922494778)) < 2e-5
    return rhoImp, vcor

def t_self_consistency_ghf():
    """
    Test self consistency with H6 321G, FCI@GHF.
    """
    import os, sys
    import numpy as np
    import scipy.linalg as la

    from pyscf import lib, fci, ao2mo
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, dft, cc

    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.basis_transform import eri_transform
    from libdmet.routine import spinless
    from libdmet.system.hamiltonian import HamNonInt

    from libdmet.utils import logger as log
    import libdmet.dmet.HubbardGSO as dmet

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
    cell.basis = '321g'
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
    restricted = False
    bogoliubov = True
    int_bath = False
    nscsites = Lat.nscsites
    Mu = 0.0
    last_dmu = 0.0
    beta = np.inf
    #beta = 1000.0
    use_hcore_as_emb_ham = False

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
    trace_start = 3000

    # solver and mu fit
    FCI = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-12, ghf=True)
    #FCI = dmet.impurity_solver.SCFSolver(restricted=restricted, tol=1e-10, ghf=True)
    solver = FCI
    nelec_tol = 5.0e-6
    delta = 0.01
    step = 0.1
    load_frecord = False

    # vcor fit
    imp_fit = False
    emb_fit_iter = 1000 # embedding fitting
    full_fit_iter = 0

    # vcor initialization
    vcor = dmet.VcorLocal(restricted, bogoliubov, nscsites)
    vcor_mat = np.zeros((3, nao, nao))
    vcor.assign(vcor_mat)

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
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
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
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao='minao', full_return=True)
    C_ao_iao = Lat.symmetrize_lo(C_ao_iao)

    ncore = 0
    nval = C_ao_iao_val.shape[-1]
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)

    # use IAO orbital as Wannier's guess
    C_ao_lo = C_ao_iao
    Lat.set_Ham(kmf, gdf, C_ao_lo)

    # compute and store the H2_emb and reuse it
    neo = (Lat.nao + Lat.nval) * 2

    # GH2
    H2_unit = eri_transform.get_unit_eri(cell, gdf, C_ao_lo=C_ao_lo, symmetry=4)
    GH2_loc, GH1_from_H2_loc, GH0_from_H2 = spinless.transform_H2_local(H2_unit[0])
    GH2_emb = spinless.unit2emb(GH2_loc, neo)[None]

    # GFock and GH1
    Fock  = np.asarray(Lat.getFock(kspace=False))[0]
    Fock_k = Lat.R2k(Fock)
    H1 = np.asarray((Lat.getH1(kspace=False)))[0]
    H1_k = Lat.R2k(H1)

    GFock, GH0_from_Fock = spinless.transform_H1_k(Fock_k)
    GH1_from_H1, GH0_from_H1 = spinless.transform_H1_k(H1_k)

    # GH1 from H2 full
    GH1_from_H2_full, GH0_from_H2_full = \
            spinless.get_H1_H0_from_df(gdf, C_ao_lo=C_ao_lo)
    GH1 = GH1_from_H1 + GH1_from_H2_full
    GH0 = GH0_from_H1 + GH0_from_H2_full #+ GH0_from_vcor #+ GH0_from_Fock

    # GRdm1
    GRdm1 = dmet.transform_rdm1_k(Lat.rdm1_lo_k[0] * 0.5)

    Ham_ghf = HamNonInt(Lat, GH1, GH2_loc, Fock=GFock, H0=GH0, spin_dim_H2=3,
                        kspace_input=True)

    Lat.setHam_model(Ham_ghf, use_hcore_as_emb_ham=use_hcore_as_emb_ham,
                     rdm1=Lat.k2R(GRdm1))
    Lat.is_model = True

    GRho, Mu, ires = dmet.GHartreeFock(Lat, vcor, Filling, mu0_elec=Mu, \
            beta=beta, fix_mu=False, mu0=None, thrnelec=1e-10, scf=False,
            full_return=True, verbose=1, conv_tol=1e-10, ph_trans=False)
    GRho, Mu, ires = dmet.GHartreeFock(Lat, vcor, None, mu0_elec=Mu, \
            beta=beta, fix_mu=False, mu0=None, thrnelec=1e-10, scf=False,
            full_return=True, verbose=1, conv_tol=1e-10, ph_trans=False)

    E_mf = ires["E"]
    E_mf_ref = kmf.e_tot - kmf.energy_nuc()
    print ("diff mean-field energy: ", E_mf - E_mf_ref)
    assert abs(E_mf - E_mf_ref) < 1e-10

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
        GRho, Mu, ires = dmet.GHartreeFock(Lat, vcor, Filling, mu0_elec=Mu, \
                beta=beta, fix_mu=False, mu0=None, thrnelec=1e-10, scf=False,
                full_return=True, verbose=1, conv_tol=1e-10, ph_trans=False)
        GRho, Mu, ires = dmet.GHartreeFock(Lat, vcor, None, mu0_elec=Mu, \
                beta=beta, fix_mu=False, mu0=None, thrnelec=1e-10, scf=False,
                full_return=True, verbose=1, conv_tol=1e-10, ph_trans=False)
        GRho_k = ires["rho_k"]

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, GRho, vcor, Mu, \
                matching=True, int_bath=int_bath, H2_given=None)
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)

        log.section("\nsolving impurity problem\n")
        solver_args = {"dm0": dmet.foldRho_k(GRho_k, basis_k)}

        GRhoEmb, EnergyEmb, ImpHam, dmu = \
                dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, \
                solver, solver_args, thrnelec=nelec_tol, delta=delta, step=step)
        dmet.SolveImpHam_with_fitting.save("./frecord")
        last_dmu += dmu

        GRhoImp, EnergyImp, nelecImp = \
                dmet.transformResults(GRhoEmb, EnergyEmb, Lat, basis, ImpHam, \
                H1e, Mu, last_dmu=last_dmu, int_bath=int_bath, \
                solver=solver, solver_args=solver_args)
        EnergyImp *= nscsites
        log.result("last_dmu : %s", last_dmu)
        log.result("E (DMET) : %s", EnergyImp)

        dump_res_iter = np.array([Mu, last_dmu, vcor.param, GRhoEmb, basis, GRhoImp, \
                C_ao_lo, GRho, Lat.getFock(kspace=False)], dtype=object)
        np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter, allow_pickle=True)

        log.section("\nfitting correlation potential\n")
        vcor_new, err = dmet.FitVcor(GRhoEmb, Lat, basis, \
                vcor, Mu, beta, MaxIter1=emb_fit_iter, \
                MaxIter2=full_fit_iter, method='CG', \
                imp_fit=imp_fit, ytol=1e-8, gtol=1e-4)

        #if iter >= trace_start:
        #    # to avoid spiral increase of vcor and mu
        #    log.result("Keep trace of vcor unchanged")
        #    ddiagV = np.average(np.diagonal(\
        #            (vcor_new.get()-vcor.get())[:2], 0, 1, 2))
        #    vcor_new = dmet.addDiag(vcor_new, -ddiagV)

        dVcor_per_ele = la.norm(vcor_new.param - vcor.param) / (len(vcor.param))
        dE = EnergyImp - E_old
        E_old = EnergyImp

        if iter >= diis_start:
            pvcor = adiis.update(vcor_new.param)
            dc.nDim = adiis.get_num_vec()
        else:
            pvcor = vcor_new.param

        dVcor_per_ele = la.norm(pvcor - vcor.param) / (len(vcor.param))
        vcor.update(pvcor)
        log.result("trace of vcor: %20.12f ", np.sum(np.diagonal((vcor.get())[:2], 0, 1, 2)))

        history.update(EnergyImp, err, nelecImp, dVcor_per_ele, dc)
        history.write_table()

        if dVcor_per_ele < u_tol and abs(dE) < E_tol and iter > iter_tol :
            conv = True
            break
    sys.modules.pop("libdmet.dmet.Hubbard", None)
    return GRhoImp, vcor

if __name__ == "__main__":
    #t_self_consistency_rhf()
    #t_self_consistency_ghf()
    test_compare_rhf_ghf_nib()
