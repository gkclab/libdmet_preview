#! /usr/bin/env python

"""
Test slater.
"""
import pytest

def test_get_emb_basis():
    import os
    import numpy as np 

    from libdmet.system import lattice
    from libdmet.routine.slater import get_emb_basis
    from libdmet.lo.lowdin import check_span_same_space
    from libdmet.utils import logger as log
    log.verbose = "DEBUG2"

    np.set_printoptions(4, linewidth=1000, suppress=True)

    cell = lattice.HChain()
    cell.basis = '321g'
    cell.verbose = 4
    cell.precision = 1e-10
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 3]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts
    
    nval  = 2
    nvirt = 2
    ncore = 0
    Lat.set_val_virt_core(nval, nvirt, ncore)
    
    rdm1_lo_file = os.path.dirname(os.path.realpath(__file__)) + "/rdm1_lo"
    rdm1_lo = np.load(rdm1_lo_file)
    
    basis = get_emb_basis(Lat, rdm1_lo)
    print (basis)
    
    Lat.val_idx = list(range(4))
    Lat.virt_idx = []
    basis_trunc = get_emb_basis(Lat, rdm1_lo, nbath=2, valence_bath=False)
    print (basis_trunc)
    ovlp = np.eye(nkpts * nao)
    assert check_span_same_space(basis.reshape(-1, 6), basis_trunc.reshape(-1, 6), ovlp)

    basis_trunc = get_emb_basis(Lat, np.array((rdm1_lo, rdm1_lo)),
            tol_bath=1e-7, localize_bath='scdm', valence_bath=False)
    print (basis_trunc)
    assert check_span_same_space(basis.reshape(-1, 6), basis_trunc[0].reshape(-1, 6), ovlp)
    assert check_span_same_space(basis.reshape(-1, 6), basis_trunc[0].reshape(-1, 6), ovlp)

def test_active_projector():
    """
    Test with H6 321G, FCI@RHF.
    """
    import os
    import numpy as np 

    from pyscf import lib, fci, ao2mo
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, dft, cc

    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.routine.slater import get_active_projector, \
            get_active_projector_full, make_rdm1_P

    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet
    from libdmet.utils.misc import mdot, max_abs, kdot

    log.verbose = "DEBUG1"
    np.set_printoptions(4, linewidth=1000, suppress=True)

    cell = lattice.HChain()
    cell.basis = '321g'
    cell.verbose = 4
    cell.precision = 1e-10
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
    FCI = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-12)
    solver = FCI
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
        kmf.conv_tol = 1e-10
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-10
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

    C_ao_lo = C_ao_iao
    nlo = C_ao_lo.shape[-1]
    Lat.set_Ham(kmf, gdf, C_ao_lo)
    
    # check the hole density matrix
    ovlp = Lat.ovlp_ao_k
    rdm1_lo_k = Lat.rdm1_lo_k
    ovlp_lo_k = Lat.ovlp_lo_k 
    rdm1_lo_R = Lat.rdm1_lo_R
    C_ao_mo = np.asarray(kmf.mo_coeff)
    C_lo_mo = make_basis.get_mo_ovlp_k(C_ao_lo, C_ao_mo, ovlp)
    
    rdm1_lo_k_h = ovlp_lo_k - rdm1_lo_k * 0.5
    
    # ref1 from the lattice supercell
    rdm1_lo_sc = Lat.expand(rdm1_lo_R)
    rdm1_lo_sc_h = Lat.R2k(Lat.extract_stripe(np.eye(rdm1_lo_sc.shape[-1]) - \
            rdm1_lo_sc * 0.5))
    assert max_abs(rdm1_lo_k_h - rdm1_lo_sc_h) < 1e-12
    
    # ref2 from lattice virtuals
    mo_occ_h = np.array(kmf.mo_occ)
    mo_occ_h = (mo_occ_h == 0).astype(int)
    rdm1_h = np.asarray(kmf.make_rdm1(C_lo_mo, mo_occ_h))
    assert max_abs(rdm1_lo_k_h - rdm1_h) < 1e-12
    
    act_idx = np.asarray([0])
    nact = len(act_idx)
    P_act, nocc = get_active_projector(act_idx, rdm1_lo_k, ovlp_lo_k)
    #P_act_no_orth = get_active_projector(act_idx, rdm1_lo_k, ovlp_lo_k, orth=False)
    
    # reference from definition
    #spin, nkpts, nlo, _ = rdm1_lo_k.shape
    #mo_occ = np.asarray(kmf.mo_occ)
    #occ_idx = mo_occ > 1e-10
    #C_lo_mo_occ  = [C_lo_mo[k][:, occ_idx[k]] for k in range(nkpts)]
    #C_lo_mo_virt = [C_lo_mo[k][:, ~occ_idx[k]] for k in range(nkpts)]
    #
    #P_lo_act = np.empty((nkpts, nlo, nact*2), dtype=np.complex128)
    #for k in range(nkpts):
    #    P_lo_act[k, :, :nact] = np.dot(C_lo_mo_occ[k],  C_lo_mo_occ[k][act_idx].conj().T)
    #    P_lo_act[k, :, nact:] = np.dot(C_lo_mo_virt[k], C_lo_mo_virt[k][act_idx].conj().T)
    #assert max_abs(P_lo_act - P_act_no_orth) < 1e-12

    rdm1_P = make_rdm1_P(Lat.fock_lo_k, ovlp_lo_k, None, P_act, nocc, project_back=True)
    rdm1_Q = rdm1_lo_k - rdm1_P
    # rdm1_Q should be all zero, since the embedding space contains all
    # electrons
    assert max_abs(rdm1_Q) < 1e-10
    P_act_full = get_active_projector_full(P_act, ovlp_lo_k)

def test_unit2emb():
    import numpy as np
    import scipy.linalg as la
    from pyscf import ao2mo
    from libdmet.routine.slater import unit2emb
    from libdmet.utils.misc import max_abs
    nao = 5
    neo = 8
    
    # 1-fold
    H2_unit = np.random.random((nao, nao, nao, nao))
    H2_unit = H2_unit + H2_unit.transpose(1, 0, 2, 3)
    H2_unit = H2_unit + H2_unit.transpose(0, 1, 3, 2)
    H2_unit = H2_unit + H2_unit.transpose(2, 3, 1, 0)
    H2_unit_s4 = ao2mo.restore(4, H2_unit, nao)
    H2_unit_s8 = ao2mo.restore(8, H2_unit, nao)

    H2_emb_ref = np.zeros((neo, neo, neo, neo))
    H2_emb_ref[:nao, :nao, :nao, :nao] = H2_unit
    H2_emb_ref_s4 = ao2mo.restore(4, H2_emb_ref, neo)
    H2_emb_ref_s8 = ao2mo.restore(8, H2_emb_ref, neo)
    
    # 1-fold
    H2_emb = unit2emb(H2_unit[None], neo) 
    assert max_abs(H2_emb - H2_emb_ref) < 1e-12

    # 4-fold
    H2_emb_s4 = unit2emb(H2_unit_s4[None], neo) 
    assert max_abs(H2_emb_s4 - H2_emb_ref_s4) < 1e-12
    
    # 8-fold
    H2_emb_s8 = unit2emb(H2_unit_s8[None], neo) 
    assert max_abs(H2_emb_s8 - H2_emb_ref_s8) < 1e-12

def test_trace():
    import numpy as np
    from libdmet.dmet.Hubbard import vcor_zeros, VcorLocal
    from libdmet.routine.slater import make_vcor_trace_unchanged, \
            vcor_diag_average, addDiag
    from libdmet.utils.misc import max_abs
    restricted = False
    bogoliubov = False
    nscsites = 5
    # zero vcor
    vcor = vcor_zeros(restricted, bogoliubov, nscsites)
    print (vcor.get()) 
    
    ave = vcor_diag_average(vcor)
    assert (max_abs(ave) < 1e-12)
    addDiag(vcor, ave)
    
    idx_range = [0, 1, 3]
    vcor = vcor_zeros(restricted, bogoliubov, nscsites, idx_range=idx_range)
    vcor_mat = np.arange(nscsites*nscsites).reshape(nscsites, nscsites)
    vcor_mat = np.array((vcor_mat, -vcor_mat * 0.5))
    vcor.assign(vcor_mat)
    vcor_mat = vcor.get()
    ave_old = vcor_diag_average(vcor, idx_range=idx_range)
    
    vcor_new = vcor_zeros(restricted, bogoliubov, nscsites, idx_range=idx_range)
    vcor_new.assign(vcor_mat + np.diag(np.random.random(nscsites)))
    vcor_new = make_vcor_trace_unchanged(vcor_new, vcor, idx_range=idx_range)
    vcor_mat_new = vcor_new.get()

    ave_new = vcor_diag_average(vcor, idx_range=idx_range)
    assert max_abs(ave_new - ave_old) < 1e-12

def test_get_rho_idem():
    import numpy as np
    import pyscf
    from pyscf import lo
    from libdmet.utils.misc import max_abs
    from libdmet.basis_transform import make_basis
    from libdmet.routine import slater
    from libdmet.utils import logger as log
    log.verbose = "DEBUG2"

    np.set_printoptions(3, linewidth=1000, suppress=True)

    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',  # in Angstrom
        basis = '321g',
    )
    myhf = mol.HF()
    myhf.kernel()
    E_ref = myhf.e_tot
    rdm1_mf = myhf.make_rdm1()
    ovlp = myhf.get_ovlp()
    C_ao_lo = lo.orth_ao(mol, 'meta_lowdin')
    
    # mf rdm1
    rdm1_lo = make_basis.transform_rdm1_to_mo_mol(rdm1_mf, C_ao_lo, ovlp)[None] \
            * 0.5
    rdm1_idem = slater.get_rdm1_idem(rdm1_lo, mol.nelectron * 0.5, beta=1000.0) 
    
    assert abs(rdm1_idem[0].trace() - mol.nelectron * 0.5) < 1e-10
    assert max_abs(np.dot(rdm1_idem[0], rdm1_idem[0]) - rdm1_idem) < 1e-10
    assert max_abs(rdm1_idem - rdm1_lo) < 1e-10 
    
    # CC rdm1
    mycc = myhf.CCSD().run()
    E_cc_ref = mycc.e_tot
    rdm1 = mycc.make_rdm1(ao_repr=True)
    rdm1_lo = make_basis.transform_rdm1_to_mo_mol(rdm1, C_ao_lo, ovlp)[None] \
            * 0.5
    rdm1_idem = slater.get_rdm1_idem(rdm1_lo, mol.nelectron * 0.5, beta=np.inf) 
    assert abs(rdm1_idem[0].trace() - mol.nelectron * 0.5) < 1e-12
    assert max_abs(np.dot(rdm1_idem[0], rdm1_idem[0]) - rdm1_idem) < 1e-12

def test_drho_dparam():
    """
    Test drho_dparam.
    """
    import os
    import numpy as np
    
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, df

    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    import libdmet.dmet.Hubbard as dmet
    from libdmet.utils import logger as log

    log.verbose = "DEBUG1"
    np.set_printoptions(3, linewidth=1000, suppress=True)
    
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

    # system
    Filling = cell.nelectron / float(Lat.nscsites*2.0)
    restricted = True
    bogoliubov = False
    int_bath = True
    nscsites = Lat.nscsites
    Mu = 0.0
    last_dmu = 0.0
    beta = 1000.0

    # vcor initialization
    vcor = dmet.VcorLocal(restricted, bogoliubov, nscsites)
    z_mat = np.zeros((2, nscsites, nscsites))
    vcor.assign(z_mat)

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

    log.section("\nsolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    log.result("Mu (guess) = %20.12f", Mu)
    rho, Mu, res = dmet.RHartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)
    Lat.update_Ham(rho*2.0, rdm1_lo_k=res["rho_k"]*2.0)
    
    log.section("\nconstructing impurity problem\n")
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, \
            matching=True, int_bath=int_bath, orth=True)
    ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
    basis_k = Lat.R2k_basis(basis)
    
    drho_dparam = dmet.FitVcor(dmet.foldRho_k(res["rho_k"], basis_k)*2.0, \
            Lat, basis, vcor, beta, Filling, return_drho_dparam=True)
    
    print (drho_dparam.shape)
    print (drho_dparam)

def test_get_H2_scaled():
    import numpy as np
    from pyscf import ao2mo
    from libdmet.routine.slater import get_H2_scaled
    from libdmet.utils import max_abs
    
    norb = 7
    imp_idx = np.random.permutation(norb)[:4]
    H2 = np.random.random((norb, norb, norb, norb))
    H2 = H2 + H2.transpose(1, 0, 2, 3)
    H2 = H2 + H2.transpose(0, 1, 3, 2)
    H2_s4 = ao2mo.restore(4, H2, norb)
    H2 = H2[None]
    H2_s4 = H2_s4[None]

    H2_scaled = get_H2_scaled(H2, imp_idx)
    H2_scaled_s4 = get_H2_scaled(H2_s4, imp_idx)
    H2_scaled_re = ao2mo.restore(1, H2_scaled_s4[0], norb)
    assert max_abs(H2_scaled_re - H2_scaled) < 1e-12

def test_rho_glob():
    """
    Test global density matrix and compared with molecular definition.
    """
    import os
    import copy
    import numpy as np 
    import scipy.linalg as la
    from pyscf import lib, fci, ao2mo
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, dft, cc

    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.routine import slater
    from libdmet.utils import mdot, max_abs

    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet

    log.verbose = "DEBUG1"
    np.set_printoptions(4, linewidth=1000, suppress=False)
    
    cell = gto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''
    cell.atom = ''' H1 5.0      5.0      0.75
                    H2 5.0      5.0      2.25 '''
    cell.basis = {'H1': 'sto3g', 'H2': 'minao'}
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 5]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts
    exxdiv = None
    Lat_col = [lattice.Lattice(cell, kmesh) for k in range(nkpts)]

    # system
    Filling = cell.nelectron / float(Lat.nscsites*2.0)
    restricted = True
    bogoliubov = False
    int_bath = True
    nscsites = Lat.nscsites
    Mu = 0.0
    last_dmu = 0.0
    beta = np.inf

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
    solver = [dmet.impurity_solver.FCI(restricted=restricted, tol=1e-12) for k
            in range(nkpts)]
    nelec_tol = 5.0e+6
    delta = 0.01
    step = 0.1
    load_frecord = False

    # vcor fit
    imp_fit = False
    emb_fit_iter = 500 # embedding fitting
    full_fit_iter = 0

    # vcor initialization
    vcor = dmet.VcorLocal(restricted, bogoliubov, nscsites)
    z_mat = np.zeros((2, nscsites, nscsites))
    vcor.assign(z_mat)

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

    log.section("\nPre-process, orbital localization and subspace partition\n")
    kmf = Lat.symmetrize_kmf(kmf)
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat,
            kmf, minao='sto3g', full_return=True)
    C_ao_iao = Lat.symmetrize_lo(C_ao_iao)
    C_ao_lo = C_ao_iao

    ncore = 0
    nval = C_ao_iao_val.shape[-1]
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)
    Lat.set_Ham(kmf, gdf, C_ao_lo)
    for k in range(nkpts):
        Lat_col[k].set_val_virt_core(np.arange(nval*k, nval*(k+1)), 0, 0)
        Lat_col[k].set_Ham(kmf, gdf, C_ao_lo)
        

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
        #Lat.update_Ham(rho*2.0, rdm1_lo_k=res["rho_k"]*2.0)
        
        log.section("\nconstructing impurity problem\n")
        basis_col = []
        ImpHam_col = []
        solver_args_col = []
        
        for i, imp_idx in enumerate(np.arange(nkpts*nao).reshape(nkpts, nao)):
            ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, \
                    matching=False, int_bath=int_bath, orth=True,
                    imp_idx=imp_idx, val_idx=imp_idx, localize_bath='scdm',
                    fout="H2_%s.h5"%i)
            #ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
            basis_k = Lat.R2k_basis(basis)
            solver_args = {"nelec": min((Lat.ncore+Lat.nval)*2, \
                    cell.nelectron*nkpts), \
                    "dm0": dmet.foldRho_k(res["rho_k"], basis_k)*2.0}
            basis_col.append(basis)
            ImpHam_col.append(ImpHam)
            solver_args_col.append(solver_args)    
        
        log.section("\nsolving impurity problem\n")
        rhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(Lat_col, Filling * nkpts, ImpHam_col,
                                          basis_col, solver, 
                                          solver_args=solver_args_col, 
                                          thrnelec=nelec_tol, delta=delta,
                                          step=step, 
                                          dmu_idx=np.arange(len(imp_idx)))

        rho_col = []
        for rho, basis in zip(rhoEmb, basis_col):
            C = basis.reshape((nkpts*nao, nao*2))
            rho_col.append(mdot(C, rho[0], C.conj().T))
            print (rho_col[-1])
        
        rdm1_glob = np.zeros((nkpts*nao, nkpts*nao))
        for I in range(nkpts):
            for J in range(nkpts):
                rdm1_glob[I*nao:(I+1)*nao, J*nao:(J+1)*nao] = \
                        0.5 * (rho_col[I][I*nao:(I+1)*nao, J*nao:(J+1)*nao] + \
                               rho_col[J][I*nao:(I+1)*nao, J*nao:(J+1)*nao])
        print (rdm1_glob)
        rho_glob_re = Lat.expand(slater.get_rho_glob_R(basis_col[0], Lat, rhoEmb[0]))
        assert max_abs(rho_glob_re - rdm1_glob) < 1e-12    
        
        break 

@pytest.mark.parametrize(
    "incore", [True, False]
)
def test_rho_glob_multi_frag(incore):
    """
    test global density matrix for multiple fragments.
    """
    import os
    import copy
    import numpy as np 
    import scipy.linalg as la
    from pyscf import lib, fci, ao2mo
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, dft, cc

    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.routine import slater
    from libdmet.utils import mdot, max_abs

    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet

    log.verbose = "DEBUG1"
    np.set_printoptions(4, linewidth=1000, suppress=True)
    
    cell = gto.Cell()
    cell.a = ''' 4.0    0.0     0.0
                 0.0     4.0    0.0
                 0.0     0.0     6.0 '''
    cell.atom = ''' H1 5.0      5.0      0.75
                    H1 5.0      5.0      2.25
                    H2 5.0      5.0      3.75 
                    H2 5.0      5.0      5.25'''
    cell.basis = {'H1': 'sto3g', 'H2': 'minao'}
    cell.verbose = 4
    cell.precision = 1e-13
    cell.build(unit='Angstrom')

    kmesh = [3, 1, 2]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts
    exxdiv = None
    Lat_col = [lattice.Lattice(cell, kmesh) for k in range(2)]

    # system
    Filling = cell.nelectron / float(Lat.nscsites*2.0)
    restricted = True
    bogoliubov = False
    int_bath = True
    nscsites = Lat.nscsites
    Mu = 0.0
    last_dmu = 0.0
    beta = np.inf

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
    solver = [dmet.impurity_solver.SCFSolver(restricted=restricted, tol=1e-13) for k
            in range(2)]
    nelec_tol = 5.0e+6
    delta = 0.01
    step = 0.1
    load_frecord = False

    # vcor fit
    imp_fit = False
    emb_fit_iter = 500 # embedding fitting
    full_fit_iter = 0

    # vcor initialization
    vcor = dmet.VcorLocal(restricted, bogoliubov, nscsites)
    z_mat = np.zeros((2, nscsites, nscsites))
    vcor.assign(z_mat)

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
        kmf.conv_tol = 1e-13
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-13
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        kmf.kernel()
        assert(kmf.converged)

    log.section("\nPre-process, orbital localization and subspace partition\n")
    kmf = Lat.symmetrize_kmf(kmf)
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat,
            kmf, minao='sto3g', full_return=True)
    C_ao_iao = Lat.symmetrize_lo(C_ao_iao)
    C_ao_lo = C_ao_iao

    ncore = 0
    nval = C_ao_iao_val.shape[-1]
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)
    Lat.set_Ham(kmf, gdf, C_ao_lo)
    Lat_col[0].set_val_virt_core([0], 0, 0)
    Lat_col[0].set_Ham(kmf, gdf, C_ao_lo)
    Lat_col[1].set_val_virt_core([1, 2, 3], 0, 0)
    Lat_col[1].set_Ham(kmf, gdf, C_ao_lo)

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
        #Lat.update_Ham(rho*2.0, rdm1_lo_k=res["rho_k"]*2.0)
        
        log.section("\nconstructing impurity problem\n")
        basis_col = []
        ImpHam_col = []
        solver_args_col = []
        for i in range(2):
            ImpHam, H1e, basis = dmet.ConstructImpHam(Lat_col[i], rho, vcor, \
                    matching=False, int_bath=int_bath, orth=True, \
                    localize_bath='scdm', fout="H2_%s.h5"%i, incore=incore)
            #ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
            basis_k = Lat.R2k_basis(basis)
            solver_args = {"nelec": min((Lat_col[i].ncore+Lat_col[i].nval)*2, \
                    cell.nelectron*nkpts), \
                    "dm0": dmet.foldRho_k(res["rho_k"], basis_k)*2.0}
            basis_col.append(basis)
            ImpHam_col.append(ImpHam)
            solver_args_col.append(solver_args)    
        
        log.section("\nsolving impurity problem\n")
        rhoEmb, EnergyEmb, ImpHam, dmu = \
            dmet.SolveImpHam_with_fitting(Lat_col, Filling, ImpHam_col, basis_col,
                    solver, solver_args=solver_args_col, thrnelec=nelec_tol, \
                            delta=delta, step=step)
        
        rho_glob_re = slater.get_rho_glob_R(basis_col, Lat_col, rhoEmb)
        assert max_abs(rho_glob_re - rho) < 1e-8
        
        rho_glob_full = slater.get_rho_glob_R(basis_col, Lat_col, rhoEmb, \
                compact=False, sign=[1, 1])
        assert max_abs(Lat.expand(rho_glob_re) - rho_glob_full) < 1e-10
        
        rho_glob_k = slater.get_rho_glob_k(basis_col, Lat_col, rhoEmb, \
                sign=[1, 1])
        assert max_abs(rho_glob_k - res["rho_k"]) < 1e-8
        
        break

def test_get_emb_basis_other_cell():
    """
    Test PBC embedding basis construction from the the first cell.
    """
    import numpy as np
    from libdmet.system import lattice
    from libdmet.system import hamiltonian as ham
    from libdmet.dmet import Hubbard as dmet
    from libdmet.utils.get_order_param import get_3band_order
    from libdmet.routine import slater
    from libdmet.basis_transform import make_basis
    from libdmet.lo import check_span_same_space
    from libdmet.utils.misc import max_abs
    import libdmet.utils.logger as log

    np.set_printoptions(4, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"

    x_dop  = 0.0
    beta = np.inf

    # Lattice settings
    LatSize = [4, 3]
    ImpSize = [1, 1]

    Lat = dmet.Square3BandAFM(*(LatSize+ImpSize))
    Ham = ham.Hubbard3band_ref(Lat, "Hanke", min_model=True, hole_rep=False, \
            factor=1.0, ignore_intercell=True, tol=1e-10)
    nao = nscsites = Lat.nscsites
    nkpts = Lat.nkpts
    natm = np.prod(ImpSize) * 6

    # Hamiltonian
    nCu_tot = np.prod(LatSize) * 2 # 4 is number of Cu site per 2x2 cell
    nO_tot = np.prod(LatSize) * 4
    nao_tot = nao * nkpts 
    nelec_half = np.prod(LatSize) * 10 # 20 electron per cell
    nelec_half_Cu = np.prod(LatSize) * 2
    nelec_half_O = np.prod(LatSize) * 8

    nelec_dop = int(np.round(x_dop * nCu_tot))
    if nelec_dop % 2 == 1:
        diff_l = abs(nelec_dop - 1 - x_dop * nCu_tot)
        diff_r = abs(nelec_dop + 1 - x_dop * nCu_tot)
        if diff_l < diff_r:
            nelec_dop = nelec_dop - 1
        else:
            nelec_dop = nelec_dop + 1
    x_dop = nelec_dop / float(nCu_tot)

    Filling = (nelec_half - nelec_dop) / (nao_tot * 2.0)
    if nelec_dop >= 0: # hole doping
        Filling_Cu = (nelec_half_Cu) / (nCu_tot * 2.0)
        Filling_O = (nelec_half_O - nelec_dop) / (nO_tot * 2.0)
    else: # electron doping
        Filling_Cu = (nelec_half_Cu - nelec_dop) / (nCu_tot * 2.0)
        Filling_O = (nelec_half_O) / (nO_tot * 2.0)

    log.info("doping x = %s", x_dop)
    log.info("nelec_half = %s", nelec_half)
    log.info("nelec_dop = %s", nelec_dop)

    restricted = False
    bogoliubov = False
    use_hcore_as_emb_ham = True
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)
    Lat.cell.verbose = 4

    H1_k = Lat.getH1(kspace=True)
    H2_loc = Lat.getH2(kspace=False)
    vcor = dmet.vcor_zeros(restricted, bogoliubov, nscsites)

    polar = 0.5
    # make AFM guess dm0
    fCu_a = Filling_Cu * (1.0 - polar)
    fCu_b = Filling_Cu * (1.0 + polar)
    fO = Filling_O
    dm0_a = np.diag([fCu_a, fCu_b, fO, fO, fO, fO])
    dm0_b = np.diag([fCu_b, fCu_a, fO, fO, fO, fO])
    dm0 = np.zeros((2, nkpts, nao, nao))
    dm0[0] = dm0_a
    dm0[1] = dm0_b

    rho, mu, ires = dmet.HartreeFock(Lat, vcor, Filling, mu0=None, beta=beta, \
            ires=True, scf=True, dm0=dm0, conv_tol=1e-9)
    log.result("\n2D Hubbard E per site (KUHF): %s \n", (Lat.kmf_lo.e_tot / natm))
    np.save("dm0_HF.npy", ires["rho_k"])

    rdm1_a, rdm1_b = rho[:, 0]
    m_AFM = 0.5 * (abs(rdm1_a[0, 0] - rdm1_b[0, 0]) + abs(rdm1_a[1, 1] - rdm1_b[1, 1]))
    log.result("m_AFM = %s", m_AFM)

    veff = Lat.kmf_lo.get_veff()
    veff_R = Lat.k2R(veff) 

    vcor_mat = np.zeros((2, nao, nao))
    frac = 1.0 / 3.0
    vcor_mat[0] = (veff_R[0, 0] * (1.0 - frac) + veff_R[1, 0] * frac)
    vcor_mat[1] = (veff_R[0, 0] * frac + veff_R[1, 0] * (1.0 - frac))

    Cu_idx = [0, 1]
    O_idx = [i for i in range(nao) if (not i in Cu_idx)]
    Cu_mesh = np.ix_(Cu_idx, Cu_idx)
    O_mesh = np.ix_(O_idx, O_idx)

    Mu = 0.0 
    vcor.assign(vcor_mat)
    fvcor = "vcor.npy"

    AFidx = [[0, 6], [3, 9]]
    PMlist1 = [1, 4, 7, 10]
    PMlist2 = [2, 5, 8, 11]

    E_old = 0.0
    conv = False
    history = dmet.IterHistory()
    dVcor_per_ele = None

    log.section ("\nsolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    log.result("Mu (guess) : %20.12f", Mu)
    rho, Mu = dmet.HartreeFock(Lat, vcor, Filling, Mu, beta=beta)
    
    res = get_3band_order(rho[:, 0], Lat)
    m_AFM = res["m_AFM"]
    print ("m_AFM (mean-field): ", m_AFM)
    
    idx_col = np.arange(nkpts * nao).reshape(nkpts, nao)
    basis = []
    for i, idx in enumerate(idx_col):
        basis.append(slater.get_emb_basis(Lat, rho, imp_idx=idx, val_idx=idx))
    
    def make_d(basis):
        basis = basis.reshape(-1, basis.shape[-1])[:, nao:]
        dm = np.dot(basis, basis.conj().T)
        return dm
    
    def check_span(C1, C2):
        C1 = C1.reshape(-1, C1.shape[-1])[:, nao:]
        C2 = C2.reshape(-1, C2.shape[-1])[:, nao:]
        print ("u")
        print (make_basis.find_closest_mo(C1, C2, np.eye(C1.shape[0]), True)[-1])
        return check_span_same_space(C1, C2, np.eye(C1.shape[0]))
    
    def check_diff(C1, C2):
        print ("C1")
        print (C1[:, :, nao:])
        print ("C2")
        print (C2[:, :, nao:])
        print (max_abs(C1 - C2))
    
    for J in range(nkpts):
        print ("\nJ = %s" % J)
        basis_J = slater.get_emb_basis_other_cell(Lat, basis[0], J)
        for s in range(2):
            check_diff(basis_J[s], basis[J][s])
            dm_J = make_d(basis_J[s])
            dm = make_d(basis[J][s])
            assert max_abs(dm - dm_J) < 1e-12
            assert check_span(basis_J[s], basis[J][s])

#def test_rho_glob_rdm2():
#    """
#    rdm2.
#    """
#    import os
#    import numpy as np
#
#    from pyscf import lib, fci, ao2mo
#    from pyscf.pbc.lib import chkfile
#    from pyscf.pbc import scf, gto, df, dft, cc
#
#    from libdmet.system import lattice
#    from libdmet.basis_transform import make_basis
#    from libdmet.basis_transform import eri_transform
#
#    from libdmet.utils import logger as log
#    import libdmet.dmet.Hubbard as dmet
#    
#    from libdmet.routine import slater
#
#    log.verbose = "DEBUG1"
#    np.set_printoptions(4, linewidth=1000, suppress=False)
#
#    cell = lattice.HChain()
#    cell.basis = '321G'
#    cell.verbose = 4
#    cell.precision = 1e-12
#    cell.build(unit='Angstrom')
#
#    kmesh = [1, 1, 3]
#    Lat = lattice.Lattice(cell, kmesh)
#    kpts = Lat.kpts
#    nao = Lat.nao
#    nkpts = Lat.nkpts
#    exxdiv = None
#
#    Filling = cell.nelectron / float(Lat.nscsites*2.0)
#    restricted = True
#    bogoliubov = False
#    int_bath = True
#    nscsites = Lat.nscsites
#    Mu = 0.0
#    last_dmu = 0.0
#    beta = np.inf
#
#    MaxIter = 100
#    u_tol = 1.0e-6
#    E_tol = 1.0e-6
#    iter_tol = 4
#
#    adiis = lib.diis.DIIS()
#    adiis.space = 4
#    diis_start = 4
#    dc = dmet.FDiisContext(adiis.space)
#    trace_start = 3
#
#    FCI = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-12)
#    solver = FCI
#    nelec_tol = 5.0e-6
#    delta = 0.01
#    step = 0.1
#    load_frecord = False
#
#    imp_fit = False
#    emb_fit_iter = 500 # embedding fitting
#    full_fit_iter = 0
#
#    vcor = dmet.VcorLocal(restricted, bogoliubov, nscsites)
#    z_mat = np.zeros((2, nscsites, nscsites))
#    vcor.assign(z_mat)
#
#    log.section("\nSolving SCF mean-field problem\n")
#
#    gdf_fname = 'gdf_ints.h5'
#    gdf = df.GDF(cell, kpts)
#    gdf._cderi_to_save = gdf_fname
#    if not os.path.isfile(gdf_fname):
#    #if True:
#        gdf.build()
#
#    chkfname = 'hchain.chk'
#    if os.path.isfile(chkfname):
#    #if False:
#        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
#        kmf.with_df = gdf
#        kmf.with_df._cderi = 'gdf_ints.h5'
#        kmf.conv_tol = 1e-12
#        kmf.max_cycle = 300
#        data = chkfile.load(chkfname, 'scf')
#        kmf.__dict__.update(data)
#    else:
#        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
#        kmf.with_df = gdf
#        kmf.with_df._cderi = 'gdf_ints.h5'
#        kmf.conv_tol = 1e-12
#        kmf.max_cycle = 300
#        kmf.chkfile = chkfname
#        kmf.kernel()
#        assert(kmf.converged)
#
#    log.section("\nPre-process, orbital localization and subspace partition\n")
#    kmf = Lat.symmetrize_kmf(kmf)
#    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao='minao', full_return=True)
#    C_ao_iao = Lat.symmetrize_lo(C_ao_iao)
#
#    ncore = 0
#    nval = C_ao_iao_val.shape[-1]
#    nvirt = cell.nao_nr() - ncore - nval
#    Lat.set_val_virt_core(nval, nvirt, ncore)
#
#    C_ao_lo = C_ao_iao
#    Lat.set_Ham(kmf, gdf, C_ao_lo)
#
#    E_old = 0.0
#    conv = False
#    history = dmet.IterHistory()
#    dVcor_per_ele = None
#    if load_frecord:
#        dmet.SolveImpHam_with_fitting.load("./frecord")
#
#    for iter in range(MaxIter):
#        log.section("\nDMET Iteration %d\n", iter)
#        
#        log.section("\nsolving mean-field problem\n")
#        log.result("Vcor =\n%s", vcor.get())
#        log.result("Mu (guess) = %20.12f", Mu)
#        rho, Mu, res = dmet.RHartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)
#        Lat.update_Ham(rho*2.0)
#        
#        log.section("\nconstructing impurity problem\n")
#        ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, \
#                matching=True, int_bath=int_bath)
#        ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
#        basis_k = Lat.R2k_basis(basis)
#
#        log.section("\nsolving impurity problem\n")
#        solver_args = {"nelec": min((Lat.ncore+Lat.nval)*2, \
#                cell.nelectron*nkpts), \
#                "dm0": dmet.foldRho_k(res["rho_k"], basis_k)*2.0}
#        rhoEmb, EnergyEmb, ImpHam, dmu = \
#            dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
#            solver_args=solver_args, thrnelec=nelec_tol, \
#            delta=delta, step=step)
#        dmet.SolveImpHam_with_fitting.save("./frecord")
#        last_dmu += dmu
#        rhoImp, EnergyImp, nelecImp = \
#                dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
#                lattice=Lat, last_dmu=last_dmu, int_bath=int_bath, \
#                solver=solver, solver_args=solver_args, add_vcor_to_E=False, \
#                vcor=vcor, rebuild_veff=True)
#        EnergyImp *= nscsites
#        log.result("last_dmu = %20.12f", last_dmu)
#        log.result("E(DMET) = %20.12f", EnergyImp)
#        #solver.twopdm = None
#        #print (solver.twopdm_mo)
#        
#        
#        slater.get_rho_glob_R(basis, lattice, rho_emb, symmetric=True, compact=True, \
#                   sign=None)
#
#        mo = solver.scfsolver.mf.mo_coeff
#        rdm2_eo = np.einsum('IJKL, iI, jJ, kK, lL -> ijkl', solver.twopdm_mo[0], 
#                            mo, mo, mo, mo, optimize=True)
#
#        rdm2_glob = slater.get_rdm2_glob_R(basis[0], None, Lat, rdm2_eo)
#        
#        print (rdm2_glob.shape)
#        
#
#        exit()
#
#        dump_res_iter = np.array([Mu, last_dmu, vcor.param, rhoEmb, basis, rhoImp, \
#                C_ao_lo, rho, Lat.getFock(kspace=False)], dtype=object)
#        np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter, allow_pickle=True)
#        
#        log.section("\nfitting correlation potential\n")
#        vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, \
#                vcor, beta, Filling, MaxIter1=emb_fit_iter,
#                MaxIter2=full_fit_iter, method='CIAH', \
#                imp_fit=imp_fit, ytol=1e-6, gtol=1e-3)
#
#        if iter >= trace_start:
#            # to avoid spiral increase of vcor and mu
#            log.result("Keep trace of vcor unchanged")
#            vcor_new = dmet.make_vcor_trace_unchanged(vcor_new, vcor)
#
#        dVcor_per_ele = la.norm(vcor_new.param - vcor.param) / (len(vcor.param))
#        dE = EnergyImp - E_old
#        E_old = EnergyImp 
#        
#        if iter >= diis_start:
#            pvcor = adiis.update(vcor_new.param)
#            dc.nDim = adiis.get_num_vec()
#        else:
#            pvcor = vcor_new.param
#        
#        dVcor_per_ele = la.norm(pvcor - vcor.param) / (len(vcor.param))
#        vcor.update(pvcor)
#        log.result("trace of vcor: %s", \
#                np.sum(np.diagonal((vcor.get())[:2], 0, 1, 2), axis=1))
#        
#        history.update(EnergyImp, err, nelecImp, dVcor_per_ele, dc)
#        history.write_table()
#
#        if dVcor_per_ele < u_tol and abs(dE) < E_tol and iter > iter_tol :
#            conv = True
#            break

if __name__ == "__main__":
#    test_rho_glob_rdm2()
    test_rho_glob()
    test_rho_glob_multi_frag(incore=True)
    test_get_emb_basis_other_cell()
    test_get_H2_scaled()
    test_active_projector()
    test_drho_dparam()
    test_get_emb_basis()
    test_get_rho_idem()
    test_trace()
    test_unit2emb()
