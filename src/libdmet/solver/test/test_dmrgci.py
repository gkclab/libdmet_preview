#! /usr/bin/env python

"""
Test dmrgci.
"""

def test_cas_from_rdm1():
    import numpy as np
    import scipy.linalg as la
    from libdmet.solver.dmrgci import cas_from_rdm1
    from libdmet.utils.misc import mdot
    from libdmet.utils import logger as log
    log.verbose = "DEBUG2"

    natocc = np.array([0.99, 0.0, 0.0, 1.0, 0.9, 0.6, 0.6, 0.6, 0.2, 0.29, 1.0])
    nao = len(natocc)
    C = la.qr(np.random.random((nao, nao)))[0]
    rdm1 = mdot(C, np.diag(natocc), C.T)
    nelec = 8
    nelecas = 5
    ncas = 6

    print ("nao: ", nao)
    print ("nelec: ", nelec)
    print ("ncas: ", ncas)
    print ("nelecas: ", nelecas)
    core, cas, virt, casinfo = cas_from_rdm1(rdm1, ncas, nelecas, nelec)
    assert casinfo == (1, 3, 2)

    # no core, no virt
    nelec = 7
    nelecas = 7
    ncas = nao
    print ("nao: ", nao)
    print ("nelec: ", nelec)
    print ("ncas: ", ncas)
    print ("nelecas: ", nelecas)
    core, cas, virt, casinfo = cas_from_rdm1(rdm1, ncas, nelecas, nelec)
    assert casinfo == (4, 3, 4)

    # no virt
    nelec = nao
    nelecas = 7
    ncas = 7
    print ("nao: ", nao)
    print ("nelec: ", nelec)
    print ("ncas: ", ncas)
    print ("nelecas: ", nelecas)
    core, cas, virt, casinfo = cas_from_rdm1(rdm1, ncas, nelecas, nelec)
    assert casinfo == (0, 3, 4)

    # no core
    nelec = 5
    nelecas = 5
    ncas = 8
    print ("nao: ", nao)
    print ("nelec: ", nelec)
    print ("ncas: ", ncas)
    print ("nelecas: ", nelecas)
    core, cas, virt, casinfo = cas_from_rdm1(rdm1, ncas, nelecas, nelec)
    assert casinfo == (4, 3, 1)

def test_cas_from_energy():
    import numpy as np
    import scipy.linalg as la
    from libdmet.solver.dmrgci import cas_from_energy
    from libdmet.utils import logger as log
    log.verbose = "DEBUG2"

    nao = 12
    fock = np.random.random((nao, nao))
    fock = fock + fock.T
    mo_energy, mo_coeff = la.eigh(fock)

    nelec = 8
    nelecas = 5
    ncas = 6
    print ("nao: ", nao)
    print ("nelec: ", nelec)
    print ("ncas: ", ncas)
    print ("nelecas: ", nelecas)
    core, cas, virt, casinfo = cas_from_energy(mo_coeff, mo_energy, ncas, nelecas, nelec)
    assert casinfo == (5, 0, 1)

    nelec = 8
    nelecas = nelec
    ncas = nao
    print ("nao: ", nao)
    print ("nelec: ", nelec)
    print ("ncas: ", ncas)
    print ("nelecas: ", nelecas)
    core, cas, virt, casinfo = cas_from_energy(mo_coeff, mo_energy, ncas, nelecas, nelec)
    assert casinfo == (8, 0, 4)

    # all occupied all cas
    nelec = nao
    nelecas = nelec
    ncas = nao
    print ("nao: ", nao)
    print ("nelec: ", nelec)
    print ("ncas: ", ncas)
    print ("nelecas: ", nelecas)
    core, cas, virt, casinfo = cas_from_energy(mo_coeff, mo_energy, ncas, nelecas, nelec)
    assert casinfo == (12, 0, 0)

def test_match_cas_basis():
    """
    Test matching the cas basis.
    """
    import numpy as np
    import scipy.linalg as la
    from libdmet.routine.ftsystem import get_h_random_deg, kernel
    from libdmet.solver.dmrgci import match_cas_basis
    from libdmet.utils.misc import max_abs

    np.set_printoptions(3, linewidth =1000, suppress=True)

    norb = 12
    nelec = 5
    beta = 1000.0
    deg_orbs = [[0,3], [1,2], [4,5,6], [7]]
    deg_energy = [1.0 , 0.1, 0.8, 3.0]
    h = get_h_random_deg(norb, deg_orbs=deg_orbs, deg_energy=deg_energy)

    a = np.random.random((norb, norb))
    C_lo_eo_old = la.qr(a)[0].reshape(1, 3, 4, 12)
    ew, ev = la.eigh(h)
    C_eo_mo_old = ev[:, :8][None]
    C_lo_mo_old = np.einsum('sRle, sec -> sRlc', C_lo_eo_old, C_eo_mo_old)

    noise = 0.001
    ew, ev = la.eigh(h + np.random.random((norb, norb)) * noise)
    C_lo_eo = la.qr(a + np.random.random((norb, norb)) * noise)[0].reshape(1, 3, \
            4, 12)
    C_eo_mo = ev[:, :8][None]
    C_lo_mo = np.einsum('sRle, sec -> sRlc', C_lo_eo, C_eo_mo)

    print ("Before matching")
    print ("diff C_lo_eo")
    print (max_abs(C_lo_eo - C_lo_eo_old))
    print ("diff C_eo_mo")
    print (max_abs(C_eo_mo - C_eo_mo_old))
    print ("diff C_lo_mo")
    val_old = max_abs(C_lo_mo - C_lo_mo_old)
    print (val_old)

    casinfo = [(5, 0, 3)]
    u_mat, diff = match_cas_basis(C_lo_eo, C_eo_mo, C_lo_eo_old, C_eo_mo_old, casinfo)
    C_eo_mo_new = np.einsum('sec, scC -> seC', C_eo_mo, u_mat)
    print ()
    print ("After matching")
    print ("diff C_eo_mo")
    print (max_abs(C_eo_mo_new - C_eo_mo_old))
    print ("diff C_lo_mo")
    print (diff)
    assert diff < val_old

def test_split_localize():
    import numpy as np
    import scipy.linalg as la

    from pyscf import lib
    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet

    np.set_printoptions(4, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"

    beta = 1000.0

    U = 6.0
    LatSize = [40, 40]
    ImpSize = [2, 2]
    Filling = 0.8 / 2
    int_bath = False
    restricted = False
    use_hcore_as_emb_ham = True
    MaxIter = 20

    Mu = U * Filling
    last_dmu = -0.235384414266

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

    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nscsites = Lat.supercell.nsites
    Ham = dmet.Ham(Lat, U, compact=True)
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)
    vcor = dmet.AFInitGuess(ImpSize, U, Filling)

    FCI = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-10)
    ncas = nscsites * 2
    nelecas = (Lat.ncore + Lat.nval)*2
    solver = dmet.impurity_solver.CASCI(ncas=ncas, nelecas=nelecas, \
                splitloc=True, MP2natorb=True, cisolver=FCI, \
                mom_reorder=False, tmpDir="./tmp")

    E_old = 0.0
    conv = False
    history = dmet.IterHistory()
    dVcor_per_ele = None

    for iter in range(MaxIter):
        log.section("\nDMET Iteration %d\n", iter)

        log.section("\nsolving mean-field problem\n")
        log.result("Vcor =\n%s", vcor.get())
        log.result("Mu (guess) = %s", Mu)

        rho, Mu, res = dmet.HartreeFock(Lat, vcor, [Filling, Filling], Mu, ires=True, beta=beta)
        E_mf = res["E"] / nscsites
        log.result("Mean-field energy (per site): %s", E_mf)

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching=False, int_bath=int_bath)
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)

        log.section("\nsolving impurity problem\n")
        solver_args = {"guess": dmet.foldRho_k(Lat.R2k(rho), basis_k), "basis": basis}
        solver.loc_method = 'ciah'

        rhoEmb, EnergyEmb, ImpHam, dmu = \
                dmet.SolveImpHam_with_fitting(Lat, [Filling, Filling], ImpHam, basis, solver, \
                solver_args)
        dmet.SolveImpHam_with_fitting.save("./frecord")

        last_dmu += dmu
        rhoImp, EnergyImp, nelecImp = \
                dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
                lattice=Lat, last_dmu=last_dmu, int_bath=int_bath, \
                solver=solver, solver_args=solver_args)
        log.result("E (DMET) : %s", EnergyImp)
        assert abs(EnergyImp - -0.982681432008679) < 1e-7

        # use pyscf's ER
        solver.loc_method = 'ciah'
        rhoEmb, EnergyEmb, ImpHam, dmu = \
                dmet.SolveImpHam_with_fitting(Lat, [Filling, Filling], ImpHam, basis, solver, \
                solver_args)
        dmet.SolveImpHam_with_fitting.save("./frecord")

        last_dmu += dmu
        rhoImp, EnergyImp, nelecImp = \
                dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
                lattice=Lat, last_dmu=last_dmu, int_bath=int_bath, \
                solver=solver, solver_args=solver_args)
        log.result("E (DMET) : %s", EnergyImp)
        assert abs(EnergyImp - -0.982681432008679) < 1e-7

        break

def test_gaopt():
    import numpy as np
    from pyscf import ao2mo
    from libdmet.system import integral
    from libdmet.solver.dmrgci import gaopt
    norb = 5
    H0 = 1.0
    H1 = np.random.random((norb, norb))
    H1 = H1 + H1.T

    H2 = np.random.random((norb, norb, norb, norb))
    H2 = H2 + H2.transpose(0, 1, 3, 2)
    H2 = H2 + H2.transpose(1, 0, 2, 3)
    H2 = H2 + H2.transpose(2, 3, 0, 1)

    Ham = integral.Integral(norb, True, False, H0, {"cd": H1[None]}, \
            {"ccdd": H2[None]}, ovlp=None)
    K_s1 = gaopt(Ham, tmp='.', only_file=True)

    Ham = integral.Integral(norb, True, False, H0, {"cd": H1[None]}, \
            {"ccdd": ao2mo.restore(4, H2, norb)[None]}, ovlp=None)
    K_s4 = gaopt(Ham, tmp='.', only_file=True)

    Ham = integral.Integral(norb, True, False, H0, {"cd": H1[None]}, \
            {"ccdd": ao2mo.restore(8, H2, norb)[None]}, ovlp=None)
    K_s8 = gaopt(Ham, tmp='.', only_file=True)

    assert np.allclose(K_s4, K_s1)
    assert np.allclose(K_s8, K_s1)

    H2_a = H2
    H2_b = H2 + 1.0
    H2_ab = H2 - 0.4

    Ham = integral.Integral(norb, False, False, H0, {"cd": np.array((H1, H1))}, \
            {"ccdd": np.array((H2_a, H2_b, H2_ab))}, ovlp=None)
    K_s1 = gaopt(Ham, tmp='.', only_file=True)

    Ham = integral.Integral(norb, False, False, H0, {"cd": np.array((H1, H1))}, \
            {"ccdd": np.array((ao2mo.restore(4, H2_a, norb), \
                               ao2mo.restore(4, H2_b, norb),
                               ao2mo.restore(4, H2_ab, norb)))}, ovlp=None)
    K_s4 = gaopt(Ham, tmp='.', only_file=True)

    assert np.allclose(K_s4, K_s1)

def test_momopt():
    import numpy as np
    import scipy.linalg as la
    from libdmet.solver import dmrgci
    from libdmet.utils.misc import max_abs

    np.set_printoptions(3, linewidth=1000, suppress=True)
    norb = 6
    a1 = np.random.random((norb, norb)) - 0.5
    basis_1 = la.qr(a1)[0].reshape((1, 3, 2, 6))[:, :, :, :5]

    # 1. only the basis order is changed
    perm_idx = np.random.permutation(5)
    basis_2 = basis_1[:, :, :, perm_idx]
    reorder, val = dmrgci.momopt(basis_1, basis_2)
    perm_re = np.argsort(reorder, kind='mergesort')

    assert max_abs(basis_2[:, :, :, reorder] - basis_1) < 1e-12
    assert (perm_re == perm_idx).all()
    assert abs(val - 1.0) < 1e-10

    # 2. has noise, not perfect matching.
    # note that when degenracy of orbitals exist,
    # the orbitals can have a phsae difference
    # so that the reordered basis can have phase difference
    np.random.seed(10)
    a2 = a1 + (np.random.random((norb, norb)) - 0.5) * 1e-5
    basis_2 = la.qr(a2)[0].reshape((1, 3, 2, 6))[:, :, :, :5]

    perm_idx = np.random.permutation(5)
    basis_2 = basis_2[:, :, :, perm_idx]
    reorder, val = dmrgci.momopt(basis_1, basis_2)
    perm_re = np.argsort(reorder, kind='mergesort')

    assert (perm_re == perm_idx).all()
    assert abs(val - 1.0) < 1e-3
    assert max_abs(basis_2[:, :, :, reorder] - basis_1) < 1e-2

def test_reorder():
    import numpy as np
    import scipy.linalg as la
    from pyscf import ao2mo
    from libdmet.system import integral
    from libdmet.solver.dmrgci import reorder
    from libdmet.utils.misc import max_abs
    norb = 5
    H0 = 1.0
    H1 = np.random.random((norb, norb))
    H1 = H1 + H1.T

    H2 = np.random.random((norb, norb, norb, norb))
    H2 = H2 + H2.transpose(0, 1, 3, 2)
    H2 = H2 + H2.transpose(1, 0, 2, 3)
    H2 = H2 + H2.transpose(2, 3, 0, 1)

    Ham = integral.Integral(norb, True, False, H0, {"cd": H1.copy()[None]}, \
            {"ccdd": H2.copy()[None]}, ovlp=None)
    mo = la.qr(np.random.random((norb, norb)))[0][None]

    order = np.random.permutation(norb)

    H0_ref = H0
    H1_ref = H1[np.ix_(order, order)].copy()
    H2_ref = H2[np.ix_(order, order, order, order)].copy()
    H2_ref_s4 = ao2mo.restore(4, H2_ref, norb)
    H2_ref_s8 = ao2mo.restore(8, H2_ref, norb)

    # s1
    Ham, mo = reorder(order, Ham, mo, rot=None)
    assert max_abs(Ham.H0 - H0_ref) < 1e-10
    assert max_abs(Ham.H1["cd"] - H1_ref) < 1e-10
    assert max_abs(Ham.H2["ccdd"] - H2_ref) < 1e-10

    # s4
    Ham = integral.Integral(norb, True, False, H0, {"cd": H1.copy()[None]}, \
            {"ccdd": ao2mo.restore(4, H2, norb)[None]}, ovlp=None)
    Ham, mo = reorder(order, Ham, mo, rot=None)
    assert max_abs(Ham.H0 - H0_ref) < 1e-10
    assert max_abs(Ham.H1["cd"] - H1_ref) < 1e-10
    assert max_abs(Ham.H2["ccdd"] - H2_ref_s4) < 1e-10

    # s8
    Ham = integral.Integral(norb, True, False, H0, {"cd": H1.copy()[None]}, \
            {"ccdd": ao2mo.restore(8, H2, norb)[None]}, ovlp=None)
    Ham, mo = reorder(order, Ham, mo, rot=None)
    assert max_abs(Ham.H0 - H0_ref) < 1e-10
    assert max_abs(Ham.H1["cd"] - H1_ref) < 1e-10
    assert max_abs(Ham.H2["ccdd"] - H2_ref_s8) < 1e-10

    # UHF
    H2_a = H2
    H2_b = H2 + 1.0
    H2_ab = H2 - 0.4

    H0_ref = H0
    H1_ref = H1[np.ix_(order, order)].copy()
    H1_ref = np.array((H1_ref, H1_ref))

    H2_a_ref  = H2_a[np.ix_(order, order, order, order)].copy()
    H2_b_ref  = H2_b[np.ix_(order, order, order, order)].copy()
    H2_ab_ref = H2_ab[np.ix_(order, order, order, order)].copy()
    H2_ref =  np.array((H2_a_ref, H2_b_ref, H2_ab_ref))

    H2_a_ref_s4  = ao2mo.restore(4, H2_a_ref, norb)
    H2_b_ref_s4  = ao2mo.restore(4, H2_b_ref, norb)
    H2_ab_ref_s4 = ao2mo.restore(4, H2_ab_ref, norb)
    H2_ref_s4 = np.array((H2_a_ref_s4, H2_b_ref_s4, H2_ab_ref_s4))

    # s1
    Ham = integral.Integral(norb, False, False, H0, {"cd": np.array((H1, H1))}, \
            {"ccdd": np.array((H2_a, H2_b, H2_ab))}, ovlp=None)
    Ham, mo = reorder(order, Ham, mo, rot=None)
    assert max_abs(Ham.H1["cd"] - H1_ref) < 1e-10
    assert max_abs(Ham.H2["ccdd"] - H2_ref) < 1e-10

    # s4
    Ham = integral.Integral(norb, False, False, H0, {"cd": np.array((H1, H1))}, \
            {"ccdd": np.array((ao2mo.restore(4, H2_a, norb), \
                               ao2mo.restore(4, H2_b, norb),
                               ao2mo.restore(4, H2_ab, norb)))}, ovlp=None)
    Ham, mo = reorder(order, Ham, mo, rot=None)
    assert max_abs(Ham.H1["cd"] - H1_ref) < 1e-10
    assert max_abs(Ham.H2["ccdd"] - H2_ref_s4) < 1e-10

def test_dmrgci_rhf():
    import os
    import numpy as np
    import scipy.linalg as la

    from pyscf import lib, fci, ao2mo
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, dft, cc

    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.basis_transform import eri_transform

    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet

    log.verbose = "DEBUG1"
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
    trace_start = 3

    # solver and mu fit
    nelec_tol = 5.0e-6
    delta = 0.01
    step = 0.1
    load_frecord = False

    # vcor fit
    imp_fit = False
    emb_fit_iter = 100 # embedding fitting
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
    Lat.set_Ham(kmf, gdf, C_ao_lo, eri_symmetry=4)

    ### ************************************************************
    ### DMET procedure
    ### ************************************************************

    ncas = Lat.nao + Lat.nval
    nelecas = (Lat.ncore + Lat.nval)*2
    fcisolver = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-12)
    solver = dmet.impurity_solver.CASCI(ncas=ncas, nelecas=nelecas, \
                splitloc=True, MP2natorb=True, cisolver=fcisolver, \
                mom_reorder=False, tmpDir="./tmp")

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
        Lat.update_Ham(rho*2.0, rdm1_lo_k=res["rho_k"]*2.0)

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching=True, int_bath=int_bath)
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)

        log.section("\nsolving impurity problem\n")
        solver_args = {"guess": dmet.foldRho_k(res["rho_k"], basis_k)*2.0, \
                "basis": basis, "nelec":(Lat.ncore+Lat.nval)*2}

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

        print ("E diff: ",  abs(EnergyImp - -1.243371414159))
        assert abs(EnergyImp - -1.243371414159) < 1e-8

        # use pyscf's ER
        solver.loc_method = 'ciah'

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

        print ("E diff: ",  abs(EnergyImp - -1.243371414159))
        assert abs(EnergyImp - -1.243371414159) < 1e-8
        break

def test_dmrgci_uhf():
    import os
    import numpy as np
    import scipy.linalg as la

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

    ### ************************************************************
    ### System settings
    ### ************************************************************
    name = 'dmrgci_uhf'
    max_memory = 4000 # 4 G
    cell = lattice.HChain()
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
    #last_dmu = -0.006707758235
    last_dmu = -0.000950441344
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
    ncas = nscsites + nval - 2
    nelecas = min((Lat.ncore+Lat.nval)*2, nkpts*cell.nelectron) - 2
    cc_etol = natom_sc * 1e-8
    cc_ttol = 1e-5
    cisolver = dmet.impurity_solver.CCSD(restricted=restricted, tol=cc_etol,
                                         tol_normt=cc_ttol, max_memory=max_memory)
    solver = dmet.impurity_solver.CASCI(ncas=ncas, nelecas=nelecas,
                                        splitloc=True, MP2natorb=False,
                                        cisolver=cisolver, mom_reorder=False,
                                        tmpDir="./tmp")

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
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao, full_return=True)
    C_ao_iao = Lat.symmetrize_lo(C_ao_iao)

    assert(nval == C_ao_iao_val.shape[-1])
    C_ao_mo = np.asarray(kmf.mo_coeff)

    # use IAO
    C_ao_lo = C_ao_iao
    Lat.set_Ham(kmf, gdf, C_ao_lo, eri_symmetry=1)

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

        log.section("\nconstructing impurity problem\n")
        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching=True, int_bath=int_bath, add_vcor=add_vcor,\
                max_memory=max_memory)
        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
        basis_k = Lat.R2k_basis(basis)

        log.section("\nsolving impurity problem\n")
        if iter < 3:
            restart = False
        else:
            solver.optimized = False
            restart = False

        solver_args = {"guess": dmet.foldRho_k(res["rho_k"], basis_k),
                       "basis": basis, "nelec":(Lat.ncore+Lat.nval) * 2}

        #solver.loc_method = 'ciah'
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

        E_DMET_per_cell = EnergyImp*nscsites / ncell_sc
        log.result("last_dmu = %20.12f", last_dmu)
        log.result("E(DMET) = %20.12f", E_DMET_per_cell)

        print ("E diff:", abs(E_DMET_per_cell - -1.211826367632))
        assert abs(E_DMET_per_cell - -1.211826367632) < 1e-7

        # use pyscf's ER
        #solver.loc_method = 'jacobi'
        solver.loc_method = 'ciah'
        rhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
            solver_args=solver_args, thrnelec=nelec_tol, \
            delta=delta, step=step)

        rhoImp, EnergyImp, nelecImp = \
            dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
            lattice=Lat, last_dmu=last_dmu, int_bath=int_bath, \
            solver=solver, solver_args=solver_args)

        E_DMET_per_cell = EnergyImp*nscsites / ncell_sc
        log.result("last_dmu = %20.12f", last_dmu)
        log.result("E(DMET) = %20.12f", E_DMET_per_cell)

        print ("E diff:", abs(E_DMET_per_cell - -1.211826367632))
        assert abs(E_DMET_per_cell - -1.211826367632) < 1e-7

        # customized core, cas, virt
        core = np.load("dmrgci_core.npy")
        cas  = np.load("dmrgci_cas.npy")
        virt = np.load("dmrgci_virt.npy")
        solver_args = {"guess": dmet.foldRho_k(res["rho_k"], basis_k),
                       "basis": basis, "nelec":(Lat.ncore+Lat.nval) * 2,
                       "orbs": (core, cas, virt)}
        rhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
            solver_args=solver_args, thrnelec=nelec_tol, \
            delta=delta, step=step)

        rhoImp, EnergyImp, nelecImp = \
            dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
            lattice=Lat, last_dmu=last_dmu, int_bath=int_bath, \
            solver=solver, solver_args=solver_args)

        E_DMET_per_cell = EnergyImp*nscsites / ncell_sc
        log.result("last_dmu = %20.12f", last_dmu)
        log.result("E(DMET) = %20.12f", E_DMET_per_cell)

        print ("E diff:", abs(E_DMET_per_cell - -1.211826367632))
        assert abs(E_DMET_per_cell - -1.211826367632) < 1e-7

        break

if __name__ == "__main__":
    test_cas_from_rdm1()
    test_momopt()
    test_dmrgci_uhf()
    test_split_localize()
    test_dmrgci_rhf()
    test_cas_from_energy()

    test_match_cas_basis()
    test_gaopt()
    test_reorder()


