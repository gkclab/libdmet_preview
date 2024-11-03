#!/usr/bin/env python

'''
Test DMET with self-consistency, GSO checked with UHF.
'''

import os
import sys
import numpy as np
import scipy.linalg as la

def test_gso_uks():
    from libdmet.routine import spinless
    from libdmet.utils import max_abs
    rho_gso = t_gso()
    sys.modules.pop("libdmet.dmet.Hubbard", None)

    rho_uks = t_uks()
    rho_uks = spinless.transform_rdm1_local(rho_uks, compact=False)

    print ("rdm1 diff compare to UHF SCF")
    print (max_abs(rho_gso - rho_uks))
    assert max_abs(rho_gso - rho_uks) < 1e-5

def t_gso():
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

    log.verbose = "DEBUG2"
    np.set_printoptions(4, linewidth=1000, suppress=True)

    ### ************************************************************
    ### System settings
    ### ************************************************************

    cell = lattice.HChain()
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
    #exxdiv = 'ewald'

    ### ************************************************************
    ### DMET settings
    ### ************************************************************

    # system
    Filling = cell.nelectron / float(Lat.nscsites*2.0)
    restricted = False
    bogoliubov = True
    int_bath = True
    nscsites = Lat.nscsites
    Mu = 0.0
    #last_dmu = 0.000349788363
    last_dmu = 0.0
    #beta = np.inf
    beta = 1000.0

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
    #myci = dmet.impurity_solver.FCI(restricted=True, tol=1e-12, ghf=True)
    myci = dmet.impurity_solver.CCSD(restricted=True, tol=1e-9, tol_normt=1e-8, ghf=True,
            scf_newton=False, alpha=1.0)
    myscf = dmet.impurity_solver.SCFSolver(restricted=True, tol=1e-9, tol_normt=1e-8, ghf=True,
                                           scf_newton=False, alpha=1.0)
    solver = myci
    nelec_tol = 1.0e+6
    delta = 0.01
    step = 0.1
    load_frecord = False

    # vcor fit
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
        kmf = scf.KUKS(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.xc = 'pbe0'
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-13
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KUKS(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.xc = 'pbe0'
        #kmf.xc = '0.01 * pbe + 0.99 * HF'
        #kmf = scf.KUHF(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-13
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        dm0_a = np.diag([0.5, 0.5, 0.0, 0.0])
        dm0_b = np.diag([0.0, 0.0, 0.5, 0.5])
        dm0 = np.asarray([[dm0_a]*nkpts, [dm0_b]*nkpts])
        #dm0 = None
        kmf.kernel(dm0=dm0)

        assert kmf.converged

    Lat.analyze(kmf)

    ### ************************************************************
    ### Pre-processing, LO and subspace partition
    ### ************************************************************

    log.section("\nPre-process, orbital localization and subspace partition\n")
    kmf = Lat.symmetrize_kmf(kmf)
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt, lo_labels = \
            make_basis.get_C_ao_lo_iao(Lat, kmf, minao='minao', full_return=True,
                                       return_labels=True)
    C_ao_iao = Lat.symmetrize_lo(C_ao_iao)

    ncore = 0
    nval = C_ao_iao_val.shape[-1]
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)

    C_ao_lo = C_ao_iao
    Lat.set_Ham(kmf, gdf, C_ao_lo)

    mo_coeff = kmf.mo_coeff
    # transform everything to LO
    hcore = kmf.get_hcore()
    ovlp  = kmf.get_ovlp()
    rdm1  = kmf.make_rdm1()
    e_nuc = kmf.energy_nuc()
    from libdmet.routine import mfd
    ewocc, Mu, err = mfd.assignocc(kmf.mo_energy, cell.nelectron * nkpts,
                                   beta=1000.0, mu0=0.0)

    # p-h transformed hamiltonian.
    C_sao_slo = spinless.combine_mo_coeff_k(C_ao_lo)
    ovlp_ghf = spinless.combine_mo_coeff_k(ovlp)

    H1_k = Lat.getH1(kspace=True)
    GH1, GH0 = spinless.transform_H1_k(H1_k, compact=False)
    omega, alpha, hyb = spinless.get_hybrid_param(kmf)#[-1]
    GV1, GV0, GV1_hf, GV0_hf = spinless.get_H1_H0_from_df(gdf, C_ao_lo=C_ao_lo,
                                                          compact=False,
                                                          hyb=hyb,
                                                          return_hf=True)

    hcore_hf_add = (GV1_hf - GV1)
    GH1 += GV1
    GH0 += GV0
    #GH0 += e_nuc
    GRho_k = spinless.transform_rdm1_k(Lat.rdm1_lo_k)

    # transform back to AO
    GH1_ao_k = make_basis.transform_h1_to_ao(GH1, C_sao_slo, ovlp_ghf)
    hcore_hf_add = make_basis.transform_h1_to_ao(hcore_hf_add, C_sao_slo, ovlp_ghf)
    GRho_ao_k = make_basis.transform_rdm1_to_ao(GRho_k, C_sao_slo)

    cell.nelectron = GRho_k.shape[-1] // 2
    if hasattr(kmf, "_numint"):
        xc_code = getattr(kmf, "xc", None)
        kmf = spinless.KGKSPH(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.xc = xc_code
    else:
        kmf = spinless.KGHFPH(cell, kpts, exxdiv=exxdiv).density_fit()
    kmf.with_df = gdf
    kmf.with_df._cderi = 'gdf_ints.h5'
    kmf.conv_tol = 1e-10
    kmf.max_cycle = 300

    def get_hcore(cell=None, kpts=None):
        hcore_ori = np.array(GH1_ao_k, copy=True)
        ovlp = kmf.get_ovlp()
        nkpts, nso, _ = hcore_ori.shape
        nao = nso // 2
        if getattr(kmf, "Mu", None) is not None:
            hcore_ori[:, :nao, :nao] -= ovlp[:, :nao, :nao] * kmf.Mu
            hcore_ori[:, nao:, nao:] += ovlp[:, nao:, nao:] * kmf.Mu
        return hcore_ori

    kmf.get_hcore = get_hcore
    kmf.get_ovlp   = lambda *args: ovlp_ghf
    kmf.energy_nuc = lambda *args: GH0 + e_nuc
    kmf.Mu = Mu
    kmf.kernel(dm0=GRho_ao_k)
    kmf.get_hcore = lambda *args: GH1_ao_k

    Lat.set_Ham(kmf, gdf, C_ao_lo, H0=GH0, hcore_hf_add=hcore_hf_add)

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
        rho, Mu, res = dmet.GHartreeFock(Lat, vcor, None, Mu, beta=beta,
                                         full_return=True, labels=lo_labels)

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, Mu,
                                                  localize_bath='scdm',
                                                  dft=True)
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)

        log.section("\nsolving impurity problem\n")

        # first find proper mu
        dm0 = dmet.foldRho_k(res["rho_k"], basis_k)
        nelec_target = 0.0
        for i, C in enumerate(basis):
            dm = mdot(C, dm0, C.conj().T)
            norb = dm.shape[-1] // 2
            nelec_target += dm[range(norb), range(norb)].sum() - dm[range(norb, norb*2), range(norb, norb*2)].sum() + norb

        dm0, E = myscf.run(ImpHam, nelec=ImpHam.H1["cd"].shape[-1]//2, dm0=dm0,
                           fit_mu=True, nelec_target=nelec_target, basis=basis, mu_elec=Mu)
        Mu_new = myscf.scfsolver.mf.mu_elec

        # modify hamiltonian
        v_mu0 = myscf.scfsolver.mf.v_mu0
        v_mu = myscf.scfsolver.mf.v_mu
        ImpHam.H1["cd"] += (v_mu - v_mu0)

        solver_args = {"nelec": dm0.shape[-1]//2, "dm0": dm0, "scf_max_cycle": 200}

        rhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
            solver_args=solver_args, thrnelec=nelec_tol, \
            delta=delta, step=step)
        dmet.SolveImpHam_with_fitting.save("./frecord")
        last_dmu += dmu

        rhoImp, EnergyImp, nelecImp = \
                dmet.transformResults(rhoEmb, EnergyEmb, Lat, basis, ImpHam, H1e, \
                Mu_new, last_dmu=last_dmu, int_bath=int_bath, \
                solver=solver, solver_args=solver_args, add_vcor_to_E=False, \
                vcor=vcor, rebuild_veff=False, labels=lo_labels)

        EnergyImp *= nscsites
        log.result("last_dmu = %20.12f", last_dmu)
        log.result("E(DMET) = %20.12f", EnergyImp)

        return rhoImp

def t_uks():
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
    import libdmet.dmet.Hubbard as dmet
    from libdmet.routine import spinless
    from libdmet.utils import max_abs, mdot

    log.verbose = "DEBUG2"
    np.set_printoptions(4, linewidth=1000, suppress=True)

    ### ************************************************************
    ### System settings
    ### ************************************************************

    cell = lattice.HChain()
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
    #exxdiv = 'ewald'

    ### ************************************************************
    ### DMET settings
    ### ************************************************************

    # system
    Filling = cell.nelectron / float(Lat.nscsites*2.0)
    restricted = False
    bogoliubov = True
    int_bath = True
    nscsites = Lat.nscsites
    Mu = 0.0
    #last_dmu = 0.000349788363
    last_dmu = 0.0
    #beta = np.inf
    beta = 1000.0

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
    #FCI = dmet.impurity_solver.FCI(restricted=True, tol=1e-12, ghf=True)
    FCI = dmet.impurity_solver.CCSD(restricted=restricted, tol=1e-9, tol_normt=1e-8, ghf=False,
            scf_newton=False, alpha=1.0)
    solver = FCI
    nelec_tol = 1.0e+6
    delta = 0.01
    step = 0.1
    load_frecord = False

    # vcor fit
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
        kmf = scf.KUKS(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.xc = 'pbe0'
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-13
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KUKS(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.xc = 'pbe0'
        #kmf.xc = '0.01 * pbe + 0.99 * HF'
        #kmf = scf.KUHF(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-13
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        dm0_a = np.diag([0.5, 0.5, 0.0, 0.0])
        dm0_b = np.diag([0.0, 0.0, 0.5, 0.5])
        dm0 = np.asarray([[dm0_a]*nkpts, [dm0_b]*nkpts])
        #dm0 = None
        kmf.kernel(dm0=dm0)

        assert kmf.converged

    Lat.analyze(kmf)

    ### ************************************************************
    ### Pre-processing, LO and subspace partition
    ### ************************************************************

    log.section("\nPre-process, orbital localization and subspace partition\n")
    kmf = Lat.symmetrize_kmf(kmf)
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt, lo_labels = \
            make_basis.get_C_ao_lo_iao(Lat, kmf, minao='minao', full_return=True,
                                       return_labels=True)
    C_ao_iao = Lat.symmetrize_lo(C_ao_iao)

    ncore = 0
    nval = C_ao_iao_val.shape[-1]
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)

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
        rho, Mu, res = dmet.HartreeFock(Lat, vcor, Filling, beta=beta, full_return=True)

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor,
                                                  localize_bath='scdm',
                                                  dft=True)
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)

        log.section("\nsolving impurity problem\n")
        solver_args = {"nelec": Lat.nval * 2, \
                "dm0": dmet.foldRho_k(res["rho_k"], basis_k), "scf_max_cycle":
                200}
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
                vcor=vcor, rebuild_veff=False, labels=lo_labels)

        EnergyImp *= nscsites
        log.result("last_dmu = %20.12f", last_dmu)
        log.result("E(DMET) = %20.12f", EnergyImp)

        return rhoImp

if __name__ == "__main__":
    test_gso_uks()
