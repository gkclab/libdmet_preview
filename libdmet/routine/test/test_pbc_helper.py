#! /usr/bin/env python

"""
Test pbc_helper.
"""

import os, sys
import numpy as np
import scipy.linalg as la
import pytest

def test_KGHF_KRHF():
    import os, sys
    from libdmet.utils import logger as log
    import libdmet.dmet.HubbardGSO as dmet
    from libdmet.utils.misc import max_abs, mdot
    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis

    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df
    
    np.set_printoptions(4, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"

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

    # system
    Filling = cell.nelectron / float(Lat.nscsites*2.0)
    restricted = True
    bogoliubov = False
    use_hcore_as_emb_ham = False
    nscsites = Lat.nscsites
    Mu = 0.0
    last_dmu = 0.0
    beta = np.inf

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
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, \
            kmf, minao='minao', full_return=True)

    ncore = 0
    nval = C_ao_iao_val.shape[-1]
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)

    # use IAO orbital as Wannier's guess
    C_ao_lo = C_ao_iao
    Lat.set_Ham(kmf, gdf, C_ao_lo, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

    rho, Mu, res = dmet.RHartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)
    E_mf_ref = (kmf.e_tot - kmf.energy_nuc())
    E_rhf = res["E"]
    
    restricted = False
    bogoliubov = True
    # vcor initialization
    vcor = dmet.VcorLocal(restricted, bogoliubov, nscsites)
    z_mat = np.zeros((3, nscsites, nscsites))
    vcor.assign(z_mat)
     
    GRho, Mu, ires = dmet.GHartreeFock(Lat, vcor, Filling, mu0_elec=Mu, \
            beta=beta, fix_mu=False, mu0=None, thrnelec=1e-10, scf=False,
            full_return=True, verbose=1, conv_tol=1e-10, ph_trans=True)
    GRho, Mu, ires = dmet.GHartreeFock(Lat, vcor, None, mu0_elec=Mu, \
            beta=beta, fix_mu=False, mu0=None, thrnelec=1e-10, scf=False,
            full_return=True, verbose=1, conv_tol=1e-10, ph_trans=True)
    E_ghf = ires["E"]
    
    diff_E = E_ghf - E_mf_ref
    print ("difference between GHF and RHF: %s" % diff_E)
    assert abs(E_ghf - E_mf_ref) < 1e-8
    sys.modules.pop("libdmet.dmet.Hubbard", None)

def test_ph_trans_mol():
    """
    P-H transform for molecular Hamiltonian.
    """
    import numpy as np
    import pyscf
    from pyscf import ao2mo, lo, gto, scf
    from libdmet.system.integral import Integral
    from libdmet.utils.misc import mdot, max_abs
    from libdmet.solver import impurity_solver 
    from libdmet.routine import spinless
    from libdmet.basis_transform import make_basis

    np.set_printoptions(3, linewidth=1000, suppress=True)

    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',
        basis = 'sto3g',
    )

    myhf = mol.HF()
    myhf.kernel()
    E_ref = myhf.e_tot
    rdm1_ref = myhf.make_rdm1()

    restricted = True
    bogoliubov = False
    norb = mol.nao_nr()
    H1 = myhf.get_hcore()
    ovlp = myhf.get_ovlp()
    H2 = ao2mo.restore(4, myhf._eri, norb)
    H0 = myhf.energy_nuc()
    rdm1 = myhf.make_rdm1()
    vj, vk = myhf.get_jk()
    
    print ("*" * 79)
    print ("LO basis")

    C = lo.orth_ao(mol)
    H1_lo = make_basis.transform_h1_to_mo_mol(H1, C)
    vj_lo = make_basis.transform_h1_to_mo_mol(vj, C)
    vk_lo = make_basis.transform_h1_to_mo_mol(vk, C)
    ovlp_lo = make_basis.transform_h1_to_mo_mol(ovlp, C)
    H2_lo = ao2mo.kernel(H2, C)
    rdm1_lo = make_basis.transform_rdm1_to_mo_mol(rdm1, C, ovlp)

    GH1, GH0 = spinless.transform_H1_local(H1_lo, compact=False)
    GV2, GV1, GV0 = spinless.transform_H2_local(H2_lo, compact=False)
    
    GH1 += GV1
    GH0 += GV0
    GH0 += H0

    S = la.block_diag(ovlp, ovlp)
    GS = la.block_diag(ovlp_lo, ovlp_lo)
    GRdm1 = spinless.transform_rdm1_local(rdm1_lo * 0.5, compact=False)
    GC = la.block_diag(C, C)

    # in LO basis
    Ham = Integral(norb * 2, True, False, GH0, {"cd": GH1[None]}, 
                   {"ccdd": GV2[None]}, ovlp=GS)

    solver = impurity_solver.SCFSolver(restricted=restricted, tol=1e-10,
                                       ghf=True)
    rdm1, E_lo = solver.run(Ham, nelec=norb, dm0=GRdm1)

    print ("LO diff")
    print (E_lo - E_ref)
    assert max_abs(E_lo - E_ref) < 1e-10

    # vj, vk
    GJ1, GJ0 = spinless.transform_H1_local(vj_lo, compact=False)
    GV2 = ao2mo.restore(1, GV2, norb*2)
    GJ1_add_aa = np.einsum('pprs -> rs', GV2[norb:, norb:, :norb, :norb])
    GJ1[:norb, :norb] += GJ1_add_aa
    GJ1_add_bb = np.einsum('pprs -> rs', GV2[norb:, norb:, norb:, norb:])
    GJ1[norb:, norb:] += GJ1_add_bb

    GK1, GK0 = spinless.transform_H1_local(vk_lo * 0.5, compact=False)
    GK1_add_bb = np.einsum('pqqs -> ps', GV2[norb:, norb:, norb:, norb:])
    GK1[norb:, norb:] += GK1_add_bb
    GJ1_ref, GK1_ref = solver.scfsolver.mf.get_jk()
    
    print ("J1 check")
    print (max_abs(GJ1 - GJ1_ref))
    assert max_abs(GJ1 - GJ1_ref) < 1e-9
    
    print ("K1 check")
    print (max_abs(GK1 - GK1_ref))
    assert max_abs(GK1 - GK1_ref) < 1e-9
    
    # in AO basis
    print ("*" * 79)
    print ("AO basis")
    GC_inv = GC.conj().T.dot(S)
    GH1_ao = mdot(GC_inv.T, GH1, GC_inv)
    GV2_ao = ao2mo.kernel(GV2, GC_inv)
    GRdm1_ao = mdot(GC, GRdm1, GC.T)

    Ham = Integral(norb * 2, True, False, GH0, {"cd": GH1_ao[None]}, 
                   {"ccdd": GV2_ao[None]}, ovlp=S)

    solver = impurity_solver.SCFSolver(restricted=restricted, tol=1e-10,
                                       ghf=True)
    rdm1, E_ao = solver.run(Ham, nelec=norb, dm0=GRdm1_ao)
    
    print ("AO diff")
    print (E_ao - E_ref)
    assert max_abs(E_ao - E_ref) < 1e-10

def test_ph_trans_mol_frozen_core():
    """
    P-H transform for molecular Hamiltonian, with frozen core
    """
    import copy
    import numpy as np
    import pyscf
    from pyscf import ao2mo, lo, gto, scf
    from pyscf import fci
    from libdmet.system.integral import Integral
    from libdmet.utils.misc import mdot, max_abs
    from libdmet.solver import impurity_solver 
    from libdmet.routine import spinless
    from libdmet.basis_transform import make_basis

    np.set_printoptions(3, linewidth=1000, suppress=True)
    mol = pyscf.M(
        atom = 'H 0 0 0; F 0 0 1.1',
        basis = 'sto3g',
        verbose = 4,
    )
    print ("AO labels")
    print (mol.ao_labels())
    
    # ------------------------------------------------------------------------
    # AO basis
    mf = mol.HF()
    E_mf_ao = mf.kernel()
    mf.analyze()
    
    cisolver = fci.FCI(mf)
    E_fci_ao, vec = cisolver.kernel()
    Ecorr_ao = E_fci_ao - E_mf_ao
    print ("Ecorr (FCI AO)", Ecorr_ao)
    # ------------------------------------------------------------------------
    
    # ------------------------------------------------------------------------
    # LO basis
    from libdmet import lo
    from libdmet.lo.iao import reference_mol
    
    nao = mol.nao
    E_ref = mf.e_tot
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    eri = ao2mo.restore(4, mf._eri, nao)
    e_nuc = mf.energy_nuc()
    rdm1 = rdm1_ref = mf.make_rdm1()
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy
    
    print (mo_energy)

    # first construct a set of LOs that froze F 1s and F 2s.
    minao = 'sto3g'
    pmol = reference_mol(mol, minao=minao)
    basis = pmol._basis
    basis_val = {}
    basis_core = {}
    
    # H, no core
    basis_val["H"] = copy.deepcopy(basis["H"])
    # F core is 1s and 2s
    basis_val["F"] = copy.deepcopy(basis["F"])
    basis_core["F"] = copy.deepcopy(basis["F"])
    basis_val["F"] = basis_val["F"][2:]
    basis_core["F"] = basis_core["F"][:2]

    pmol_val = pmol.copy()
    pmol_val.basis = basis_val
    pmol_val.build()

    print ("Valence:")
    val_labels = pmol_val.ao_labels()
    for i in range(len(val_labels)):
        val_labels[i] = val_labels[i].replace("F 1s", "F 2s")
    pmol_val.ao_labels = lambda *args: val_labels
    print (len(pmol_val.ao_labels()))
    print (pmol_val.ao_labels())
    pmol_core = pmol.copy()
    pmol_core.basis = basis_core
    pmol_core.build()
    core_labels = pmol_core.ao_labels()

    print ("Core:")
    print (len(pmol_core.ao_labels()))
    print (pmol_core.ao_labels())
    ncore = len(pmol_core.ao_labels())
    nval = pmol_val.nao_nr()
    nvirt = mol.nao_nr() - ncore - nval
    
    C_ao_lo, C_ao_lo_val, C_ao_lo_virt, C_ao_lo_core = \
            make_basis.get_C_ao_lo_iao_mol(mf, minao='sto3g', orth_virt=True,
                                           full_virt=False, full_return=True,
                                           pmol_val=pmol_val, 
                                           pmol_core=pmol_core, tol=1e-10)
    C_ao_lo_xcore = make_basis.tile_C_ao_iao(C_ao_lo_val, C_virt=C_ao_lo_virt, C_core=None)
    
    lo_labels, val_labels, virt_labels = lo.get_labels(mol, minao='sto3g', full_virt=False, 
                                                       B2_labels=val_labels,
                                                       core_labels=core_labels)
    lo_labels = core_labels + lo_labels

    # core potential
    mo_core = mo_coeff[:, :ncore]
    mo_xcore = mo_coeff[:, ncore:]
    mo_occ_core = mo_occ[:ncore]
    mo_occ_xcore = mo_occ[ncore:]
    rdm1_core = mf.make_rdm1(mo_core, mo_occ_core)
    rdm1_xcore = mf.make_rdm1(mo_xcore, mo_occ_xcore)
    
    vj_core, vk_core = mf.get_jk(mol, rdm1_core)
    vj_ao, vk_ao = mf.get_jk(mol, rdm1)

    vj_xcore = vj_ao - vj_core
    vk_xcore = vk_ao - vk_core
    
    veff_core = vj_core - vk_core * 0.5
    hcore_xcore = hcore + veff_core
    E0 = e_nuc + np.einsum('pq, qp ->', hcore + veff_core * 0.5, rdm1_core)
    
    # transform to LO
    hcore_xcore_lo = make_basis.transform_h1_to_lo_mol(hcore_xcore, C_ao_lo_xcore)
    ovlp_xcore_lo = make_basis.transform_h1_to_lo_mol(ovlp, C_ao_lo_xcore)
    eri_xcore_lo = ao2mo.kernel(eri, C_ao_lo_xcore)
    rdm1_xcore_lo = make_basis.transform_rdm1_to_lo_mol(rdm1_xcore, C_ao_lo_xcore, ovlp) 
    rdm1_core_lo = make_basis.transform_rdm1_to_lo_mol(rdm1_core, C_ao_lo_core, ovlp) 

    nelec = mol.nelectron 
    mol.nelectron = nelec - ncore * 2
    
    mf = mol.HF()
    mf.get_hcore = lambda *args: hcore_xcore_lo
    mf.get_ovlp = lambda *args: ovlp_xcore_lo
    mf._eri = eri_xcore_lo
    mf.energy_nuc = lambda *args: E0
    E_mf_lo = mf.kernel(dm0=rdm1_xcore_lo)

    cisolver = fci.FCI(mf)
    E_fci_lo, vec = cisolver.kernel()
    Ecorr_lo = E_fci_lo - E_mf_lo
    print ("Ecorr (FCI LO)", Ecorr_lo)
    
    assert abs(E_mf_lo - E_mf_ao) < 1e-10
    # ------------------------------------------------------------------------
    
    # ------------------------------------------------------------------------
    # GSO
    from libdmet.routine import mfd
    beta = 1000.0
    #beta = np.inf
    ewocc, mu, err = mfd.assignocc(mo_energy, nelec * 0.5, beta, mu0=0.0,
                                   fix_mu=False, thr_deg=1e-6, Sz=None,
                                   fit_tol=1e-12)

    ovlp_ghf = spinless.combine_mo_coeff(ovlp)
    C_sao_slo = spinless.combine_mo_coeff(C_ao_lo)
    C_sao_slo_core = spinless.combine_mo_coeff(C_ao_lo_core)
    C_sao_slo_xcore = spinless.combine_mo_coeff(C_ao_lo_xcore)

    # full 
    GH1, GH0 = spinless.transform_H1_local(hcore, ovlp=ovlp, compact=False)

    hcore_lo = make_basis.transform_h1_to_lo_mol(hcore, C_ao_lo)
    GH1_lo, GH0_lo = spinless.transform_H1_local(hcore_lo, compact=False)
    GH1_ao = make_basis.transform_h1_to_ao_mol(GH1_lo, C_sao_slo, ovlp_ghf)
    GH0_ao = GH0_lo

    print ("GH0 diff", max_abs(GH0 - GH0_ao))
    print ("GH1 diff", max_abs(GH1 - GH1_ao))
    assert max_abs(GH0 - GH0_ao) < 1e-12
    assert max_abs(GH1 - GH1_ao) < 1e-12

    eri_lo = ao2mo.kernel(eri, C_ao_lo)
    GV2_lo, GV1_lo, GV0_lo = spinless.transform_H2_local(eri_lo, compact=False)
    
    C_inv_gso = C_sao_slo.T @ ovlp_ghf
    GV2_ao = ao2mo.kernel(GV2_lo, C_inv_gso)
    GV1_ao = make_basis.transform_h1_to_ao_mol(GV1_lo, C_sao_slo, ovlp_ghf)
    GV0_ao = GV0_lo

    GV2, GV1, GV0 = spinless.transform_H2_local(eri, ovlp, compact=False)
    print ("GV0 diff", max_abs(GV0 - GV0_ao))
    print ("GV1 diff", max_abs(GV1 - GV1_ao))
    print ("GV2 diff", max_abs(GV2 - GV2_ao))
    assert max_abs(GV0 - GV0_ao) < 1e-12
    assert max_abs(GV1 - GV1_ao) < 1e-12
    assert max_abs(GV2 - GV2_ao) < 1e-12
    
    Grdm1 = spinless.transform_rdm1_local(rdm1 * 0.5, ovlp, compact=False)
    Govlp = spinless.combine_mo_coeff(ovlp)

    GH1 += GV1
    E0 = GH0 + GV0 + e_nuc
    
    Ham = Integral(GH1.shape[-1], True, False, E0, {"cd": GH1[None]}, 
                   {"ccdd": GV2[None]}, ovlp=Govlp)
    solver = impurity_solver.FCI(restricted=True, tol=1e-10, ghf=True)
    rdm1, E_fci_gso = solver.run(Ham, nelec=GH1.shape[-1]//2, dm0=Grdm1, Mu=mu)
    E_mf_gso = solver.scfsolver.mf.e_tot
    
    Ecorr_gso = E_fci_gso - E_mf_gso
    print ("Ecorr (FCI full GSO)", Ecorr_gso)
    assert abs(E_mf_gso - E_mf_ao) < 1e-10
    assert abs(E_fci_gso - E_fci_ao) < 1e-10

    # ------------------------------------------------------------------------
    # GSO frozen core
    mf = solver.scfsolver.mf
    Grdm1_core_lo = spinless.transform_rdm1_local(rdm1_core_lo * 0.5, compact=False)
    Grdm1_core = make_basis.transform_rdm1_to_ao_mol(Grdm1_core_lo, C_sao_slo_core)
    Grdm1_xcore_lo = spinless.transform_rdm1_local(rdm1_xcore_lo * 0.5, compact=False)

    Gveff_core = mf.get_veff(dm=Grdm1_core)
    Ghcore = GH1 + Gveff_core
    GH0_core = np.einsum('pq, qp ->', GH1 + 0.5 * Gveff_core, Grdm1_core)
    
    Ghcore_lo = make_basis.transform_h1_to_lo_mol(Ghcore, C_sao_slo_xcore)
    Govlp_lo = make_basis.transform_h1_to_lo_mol(Govlp, C_sao_slo_xcore)
    GV2_lo = ao2mo.kernel(GV2, C_sao_slo_xcore)
    E0 = GH0 + GV0 + e_nuc + GH0_core
    
    Ham = Integral(Ghcore_lo.shape[-1], True, False, E0, {"cd": Ghcore_lo[None]}, 
                   {"ccdd": GV2_lo[None]}, ovlp=Govlp_lo)
    solver = impurity_solver.FCI(restricted=True, tol=1e-10, ghf=True)
    rdm1, E_fci_gso_frozen = solver.run(Ham, nelec=Ghcore_lo.shape[-1]//2, dm0=Grdm1_xcore_lo, Mu=mu)
    E_mf_gso_frozen = solver.scfsolver.mf.e_tot
    
    Ecorr_gso_frozen = E_fci_gso_frozen - E_mf_gso_frozen
    print ("Ecorr (FCI frozen GSO)", Ecorr_gso_frozen)

    assert abs(E_mf_gso_frozen - E_mf_lo) < 1e-10
    assert abs(E_fci_gso_frozen - E_fci_lo) < 1e-10
    # ------------------------------------------------------------------------

def test_ph_integral():
    """
    Test ab initio particle-hole transformation.
    """
    import libdmet.dmet.Hubbard as dmet
    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.basis_transform.eri_transform import \
            get_unit_eri, get_emb_eri
    from libdmet.routine import spinless
    from libdmet.utils import logger as log
    from libdmet.utils.misc import max_abs, mdot

    import pyscf
    from pyscf import lib, ao2mo
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, dft
    
    np.set_printoptions(4, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"

    cell = gto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''
    cell.atom = ''' H 5.0      5.0      0.0
                    H 5.0      5.0      1.5 '''
    cell.basis = 'sto3g'
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 3]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = nscsites = Lat.nao
    nso = nao * 2
    nkpts = Lat.nkpts
    exxdiv = None

    # system
    Filling = cell.nelectron / float(Lat.nscsites*2.0)
    restricted = True
    bogoliubov = False
    int_bath = False
    use_hcore_as_emb_ham = False
    H2_fname = "emb_eri_slater.h5"
    nscsites = Lat.nscsites
    Mu = 0.0
    last_dmu = 0.0
    beta = np.inf

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

    log.section("\nSolving SCF mean-field problem\n")
    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    if True:
        gdf.build()

    chkfname = 'hchain.chk'
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
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = \
            make_basis.get_C_ao_lo_iao(Lat, kmf, minao='minao', full_return=True)

    ncore = 0
    nval = C_ao_iao_val.shape[-1]
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)

    # use IAO orbital as Wannier's guess
    C_ao_lo = C_ao_iao
    Lat.set_Ham(kmf, gdf, C_ao_lo, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

    rho, Mu, res = dmet.RHartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)
    E_mf_ref = (kmf.e_tot - kmf.energy_nuc())
    print ("E_mf_ref")
    print (E_mf_ref)
    
    int_bath = False
    restricted = False
    bogoliubov = True
    # vcor is initialized as all zero
    vcor = dmet.VcorLocal(restricted, bogoliubov, nscsites)
    vcor_mat = np.zeros((3, nao, nao))
    vcor.assign(vcor_mat)
    
    snao = nao * nkpts
    snso = snao * 2
    imp_ao = list(range(nao))
    env_ao = [idx for idx in range(snao) if not idx in imp_ao]
    dm_imp = np.zeros((snao, snao))
    dm_imp[imp_ao, imp_ao] = 1.0
    dm_env = np.zeros_like(dm_imp)
    dm_env[env_ao, env_ao] = 1.0

    Fock  = np.asarray(Lat.getFock(kspace=False))[0]
    Fock_k = Lat.R2k(Fock)
    H1 = np.asarray((Lat.getH1(kspace=False)))[0]
    H1_k = Lat.R2k(H1)
    
    smf = scf.KRHF(Lat.bigcell, exxdiv=None).density_fit()
    Cgamma = np.einsum('Rk, kui, Sk -> RuSi', Lat.phase, C_ao_lo, \
            Lat.phase.conj(), optimize=True)
    Cgamma = Cgamma.reshape(nao*nkpts, nao*nkpts)
    H2_ref = smf.with_df.ao2mo(Cgamma, compact=False).reshape([nao*nkpts]*4)
    assert max_abs(H2_ref.imag) < 1e-10
    H2_ref = H2_ref.real
    V0_ref = 0.5 * (np.einsum('PPRR ->', H2_ref[:nao, :nao]) \
            - np.einsum('PQQP ->', H2_ref[:, :nao, :nao]))
    
    GH1_from_H2_full, GH0_from_H2_full = spinless.get_H1_H0_from_df(gdf, C_ao_lo=C_ao_lo)
    print ("GH0_from_H2_full")
    print (GH0_from_H2_full)
    print ("V0_ref")
    print (V0_ref)
    assert abs(GH0_from_H2_full - V0_ref) < 1e-10

def test_ph_trans_fci():
    from pyscf import fci
    from libdmet_solid.routine import pbc_helper as pbc_hp
    from libdmet_solid.routine import spinless
    np.set_printoptions(3, linewidth=1000, suppress=True)
    #np.random.seed(1)
    norb = 5

    H0 = 0.0
    H1 = np.random.random((norb, norb)) - 0.5
    H1 = H1 + H1.T
    V2 = np.random.random((norb, norb, norb, norb)) - 0.5
    V2 = V2 + V2.transpose(0, 1, 3, 2)
    V2 = V2 + V2.transpose(1, 0, 2, 3)
    V2 = V2 + V2.transpose(2, 3, 0, 1)
    
    cisolver = fci.direct_spin1.FCI()
    cisolver.verbose = 8
    cisolver.max_cycle = 100
    cisolver.conv_tol = 1e-8
    e_ref, fcivec = cisolver.kernel(H1, V2, norb, (3, 3), ecore=H0)
    print ("ref", e_ref)

    GV2, GV1, GV0 =  spinless.transform_H2_local(V2)
    GH1, GH0 = spinless.transform_H1_local(H1)
    GH1 += GV1
    GH0 = H0 + GH0 + GV0
    
    GV2 = spinless.combine_H2(GV2)
    GH1 = spinless.combine_H(GH1)

    e, fcivec = cisolver.kernel(GH1, GV2, norb*2, (norb, 0), ecore=GH0)
    dm1 = cisolver.make_rdm1(fcivec, norb*2, (norb, 0))
    nelec = dm1[:norb, :norb].trace() - dm1[norb:, norb:].trace() + norb
    print ("ghf-fci", e)
    print (nelec)
    print (dm1)
    print (abs(e - e_ref))
    if abs(nelec - 6) < 1e-10:
        assert abs(e - e_ref) < 1e-10

def test_ph_trans_gamma_uhf():
    """
    Compare p-h transform with supercell and UHF.
    2D Hubbard model.
    """
    import libdmet.dmet.HubbardGSO as dmet
    from libdmet.system import integral
    from libdmet.system.hamiltonian import HamNonInt
    from libdmet.routine import mfd, spinless
    from libdmet.solver import scf as scf_hp
    from libdmet.utils.misc import max_abs
    import libdmet.utils.logger as log

    np.set_printoptions(4, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"
    
    U = 6.0 
    LatSize = [6, 4]
    ImpSize = [2, 2]
    Filling = 1.0 / 2.0
    int_bath = True
    restricted = False
    use_hcore_as_emb_ham = False
    MaxIter = 50

    Mu = U * Filling
    last_dmu = 0.0
    beta = np.inf
    imp_fit = False

    ntotal = Filling * np.prod(LatSize)
    if ntotal - np.round(ntotal) > 1e-5:
        log.warning("rounded total number of electrons to integer %d", 
                    np.round(ntotal))
        Filling=float(np.round(ntotal)) / np.prod(LatSize)
    
    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nkpts = Lat.nkpts
    nao = nscsites = Lat.supercell.nsites
    nso = nao * 2
    
    Ham = dmet.Ham(Lat, U)
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)
    
    # PBC case
    H1_k = Lat.getH1(kspace=True)
    H2_loc = Lat.getH2(kspace=False)
    # p-h transformed hamiltonian.
    H1_k, H0 = dmet.transform_H1_k(H1_k)
    H2_loc, H1_from_H2_loc, H0_from_H2 = dmet.transform_H2_local(H2_loc)
    H1_k = mfd.add_H1_loc_to_k(H1_from_H2_loc, H1_k)
    H0 += H0_from_H2
    
    vcor = dmet.AFInitGuess(ImpSize, U, Filling, rand=0.01)
    vcor_param = vcor.param
    vcor_param[:] = 0.0
    vcor.update(vcor_param)

    Ham_sl = HamNonInt(Lat, H1_k, H2_loc, Fock=None, ImpJK=None, kspace_input=True, spin_dim_H2=3, H0=H0)
    Lat.setHam(Ham_sl, use_hcore_as_emb_ham=use_hcore_as_emb_ham, eri_symmetry=1)

    dm0_a = np.diag([1.0, 0.0, 0.0, 1.0])
    dm0_b = np.diag([0.0, 1.0, 1.0, 0.0])
    dm0 = np.array(([dm0_a] * nkpts, [dm0_b] * nkpts))
    dm0 = spinless.transform_rdm1_k(dm0)

    Grdm1, Mu, ires = dmet.GHartreeFock(Lat, vcor, Filling, mu0_elec=Mu, \
                beta=beta, fix_mu=False, mu0=None, thrnelec=1e-10, scf=True,
                full_return=True, verbose=4, conv_tol=1e-12, dm0=dm0)
    
    Grdm1 = Lat.expand(Grdm1)
    Gfock = Lat.expand(Lat.k2R(Lat.kmf_lo.get_fock()))
    Ghcore = Lat.expand(Lat.k2R(Lat.kmf_lo.get_hcore()))
    E_hf = ires["E"]

    # *******************************************************
    Ham = dmet.Ham(Lat, U)
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

    norb = nao * nkpts
    nelec_tot = int(np.round(Filling * 2 * norb, 0))
    H1 = Lat.expand(Lat.getH1(kspace=False))
    H1 = np.array([H1, H1])
    
    # first check GH1, GFock, mf energy, rdm1 from Gamma
    GH1, GH0 = spinless.transform_H1_local(H1)
    H2 = np.zeros((norb, norb, norb, norb))
    H2[range(norb), range(norb), range(norb), range(norb)] = U
    
    GV2, GV1, GV0 = spinless.transform_H2_local(H2)
    GH0 = GV0 + GH0
    GH1 = GH1 + GV1
    
    GH1 = spinless.combine_H(GH1)
    GV2 = spinless.combine_H2(GV2)
    
    dm0_a = np.diag([1.0, 0.0, 0.0, 1.0])
    dm0_b = np.diag([0.0, 1.0, 1.0, 0.0])
    dm0 = np.array(([dm0_a] * nkpts, [dm0_b] * nkpts))
    
    dm0 = Lat.expand(Lat.k2R(dm0))
    Gdm0 = spinless.transform_rdm1_local(dm0, compact=False)
    
    myscf = scf_hp.SCF(newton_ah=True)

    Mu = U * Filling
    # nelec, spin, bogoliubov, spinRestricted
    myscf.set_system(norb, 0, False, False)
    myscf.set_integral(norb*2, 0.0, {"cd": GH1[None]}, {"ccdd": GV2[None]})
    E_hf_ref, rdm1_ref = myscf.GGHF(MaxIter=100, tol=1e-12, InitGuess=Gdm0, Mu=Mu)
    print ("energy (HF) per site", E_hf / norb)
    
    hcore_ref = myscf.mf.get_hcore()
    fock_ref = myscf.mf.get_fock()
    
    # energy
    print ("energy diff")
    print (E_hf - E_hf_ref / nkpts)
    assert abs(E_hf - E_hf_ref / nkpts) < 1e-10
    
    # hcore
    order_a = []
    order_b = []
    for k in range(nkpts):
        order_a.append(np.arange(nao) + (k * 2) * nao)
        order_b.append(np.arange(nao) + (k * 2 + 1) * nao)
    order = np.vstack((order_a, order_b)).ravel()
    
    mesh = np.ix_(order, order)
    print ("hcore diff")
    print (max_abs(Ghcore[mesh] - hcore_ref))
    assert max_abs(Ghcore[mesh] - hcore_ref) < 1e-10 

    print ("fock diff")
    print (max_abs(Gfock[mesh] - fock_ref))
    assert max_abs(Gfock[mesh] - fock_ref) < 1e-8 

    print ("rdm1 diff")
    print (max_abs(Grdm1[mesh] - rdm1_ref))
    assert max_abs(Grdm1[mesh] - rdm1_ref) < 1e-8 
    
    # Now, check the UHF
    dm0_a = np.diag([1.0, 0.0, 0.0, 1.0])
    dm0_b = np.diag([0.0, 1.0, 1.0, 0.0])
    dm0 = np.array(([dm0_a] * nkpts, [dm0_b] * nkpts))

    rdm1_uhf, Mu, ires = dmet.HartreeFock(Lat, vcor, Filling,
                beta=beta, fix_mu=False, mu0=None, thrnelec=1e-10, scf=True,
                full_return=True, ires=True, verbose=4, conv_tol=1e-12, dm0=dm0)
    E_hf_uhf = ires["E"]
    
    # energy
    print ("energy diff to UHF")
    print (E_hf - E_hf_uhf)
    assert abs(E_hf - E_hf_uhf) < 1e-10
    
    # rdm1, transform_rdm1_k
    rdm1_trans = spinless.transform_rdm1_k(ires["rho_k"], compact=False)
    rdm1_trans = Lat.expand(Lat.k2R(rdm1_trans))

    print ("rdm1 diff to UHF")
    print (max_abs(rdm1_trans - Grdm1))
    assert max_abs(rdm1_trans - Grdm1) < 1e-10 

def test_ph_trans_gamma_random():
    """
    Compare p-h transform with supercell and UHF.
    2D Hubbard model.
    """
    import libdmet.dmet.HubbardGSO as dmet
    from libdmet.system import integral
    from libdmet.system.hamiltonian import HamNonInt
    from libdmet.routine import mfd, spinless
    from libdmet.utils.misc import max_abs
    import libdmet.utils.logger as log

    np.set_printoptions(4, linewidth=1000, suppress=True)
    np.random.seed(10086)
    log.verbose = "DEBUG2"
    
    U = 6.0 
    LatSize = [10, 18]
    ImpSize = [2, 2]
    Filling = 1.0 / 2.0
    int_bath = True
    restricted = False
    use_hcore_as_emb_ham = False
    MaxIter = 50

    Mu = U * Filling
    last_dmu = 0.0
    beta = np.inf
    imp_fit = False

    ntotal = Filling * np.prod(LatSize)
    if ntotal - np.round(ntotal) > 1e-5:
        log.warning("rounded total number of electrons to integer %d", 
                    np.round(ntotal))
        Filling=float(np.round(ntotal)) / np.prod(LatSize)
    
    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nkpts = Lat.nkpts
    nao = nscsites = Lat.supercell.nsites
    nso = nao * 2
    
    Ham = dmet.Ham(Lat, U)
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)
    
    H1_A = np.array(Lat.getH1(kspace=False))
    H1_B = np.array(Lat.getH1(kspace=False))
    H1_D = np.array(Lat.getH1(kspace=False))
    
    H1_A += (np.random.random(H1_A.shape) - 0.5) * 0.1
    H1_B += (np.random.random(H1_B.shape) - 0.5) * 0.1
    H1_D += (np.random.random(H1_D.shape) - 0.5) * 0.2
    
    H1_A[0] = (H1_A[0] + H1_A[0].T) * 0.5
    H1_B[0] = (H1_B[0] + H1_B[0].T) * 0.5
    H1_D[0] = (H1_D[0] + H1_D[0].T) * 0.5
    
    for R in range(nkpts):
        mR = Lat.subtract(0, R)
        tmp = (H1_A[R] + H1_A[mR].T) * 0.5
        H1_A[R] = tmp
        H1_A[mR] = tmp
        
        tmp = (H1_B[R] + H1_B[mR].T) * 0.5
        H1_B[R] = tmp
        H1_B[mR] = tmp
        
        tmp = (H1_D[R] + H1_D[mR].T) * 0.5
        H1_D[R] = tmp
        H1_D[mR] = tmp
    
    H1_k = np.asarray((Lat.R2k(H1_A), Lat.R2k(H1_B), Lat.R2k(H1_D)))
    GH1_k, GH0 = spinless.transform_H1_k(H1_k, compact=False)
    
    # hcore
    order_a = []
    order_b = []
    for k in range(nkpts):
        order_a.append(np.arange(nao) + (k * 2) * nao)
        order_b.append(np.arange(nao) + (k * 2 + 1) * nao)
    order = np.vstack((order_a, order_b)).ravel()
    mesh = np.ix_(order, order)
    order_re = np.argsort(order, kind='mergesort')
    mesh_re = np.ix_(order_re, order_re)

    # reference from Gamma
    H1_G = Lat.expand(np.asarray((H1_A, H1_B, H1_D)))
    GH1_G, GH0_G = spinless.transform_H1_local(H1_G, compact=False)
    GH1_k_from_G = Lat.R2k(Lat.extract_stripe(GH1_G[mesh_re]))
    
    assert abs(GH0 - GH0_G / nkpts) < 1e-10
    
    print ("GH1 difference")
    print (max_abs(GH1_k_from_G - GH1_k))
    assert max_abs(GH1_k_from_G - GH1_k) < 1e-10

def test_ph_trans_ab_initio():
    from libdmet.system import lattice
    from libdmet.routine import spinless
    from libdmet.utils.misc import max_abs
    from libdmet import lo
    import libdmet.utils.logger as log

    from pyscf import ao2mo
    from pyscf.pbc import gto, scf, df
    
    np.set_printoptions(4, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"

    cell = gto.Cell()
    cell.a = ''' 5.0     0.0     0.0
                 0.0     5.0     0.0
                 0.0     0.0     5.0 '''
    cell.atom = ''' Li  0.0      0.0      0.0
                    H   0.0      0.0      2.5 '''
    cell.basis = {'Li': 'minao', 'H':'minao'}
    cell.verbose = 5
    cell.precision = 1e-14
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 3]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts
    
    C_ao_lo = lo.lowdin_k(Lat)
    
    C = Lat.expand_orb(Lat.k2R(C_ao_lo))
    print ("C_ao_lo full shape")
    print (C.shape)
    
    # check the C_ao_lo
    from libdmet.basis_transform import make_basis
    hcore_k = np.asarray(cell.pbc_intor('int1e_kin', 1, 1, kpts))
    hcore_lo_k = make_basis.transform_h1_to_lo(hcore_k, C_ao_lo)
    
    ovlp_k = cell.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
    
    Hcore_lo_k, H0 = spinless.transform_H1_k(hcore_lo_k, compact=False)
    
    S_so_so = []
    C_so_lo = []
    for k in range(C_ao_lo.shape[0]):
        C_so_lo.append(la.block_diag(C_ao_lo[k], C_ao_lo[k]))
        S_so_so.append(la.block_diag(ovlp_k[k], ovlp_k[k]))
    C_so_lo = np.asarray(C_so_lo)
    S_so_so = np.asarray(S_so_so)
   
    Hcore_ao_k = make_basis.transform_h1_to_ao(Hcore_lo_k, C_so_lo, S_so_so)
    Hcore_ao_k_AA = Hcore_ao_k[:, :nao, :nao]
    Hcore_ao_k_BB = Hcore_ao_k[:, nao:, nao:]

    assert max_abs(Hcore_ao_k_AA - hcore_k) < 1e-10
    assert max_abs(Hcore_ao_k_BB + hcore_k) < 1e-10

    # check 
    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf.auxbasis = df.aug_etb(cell, beta=5.0)
    gdf._cderi_to_save = gdf_fname
    gdf._cderi = gdf_fname
    if True:
    #if not os.path.exists(gdf_fname):
        gdf.build()
    
    eri_8d = spinless.get_eri_8d(cell, gdf, kpts=None, compact=False)
    eri_8d_R = Lat.k2R_H2_8d(eri_8d)
    
    eri_full_ao = eri_8d_R.transpose(0, 4, 1, 5, 2, 6, 3, 7)\
            .reshape(nkpts*nao, nkpts*nao, nkpts*nao, nkpts*nao)
    eri_full = ao2mo.kernel(eri_full_ao, C) 
    eri_8d = None
    eri_8d_R = None
    
    print ("eri lo full shape")
    print (eri_full.shape)
    
    GV2_ref, GV1_ref, GV0_ref = spinless.transform_H2_local(eri_full)
    GV2_ref = spinless.combine_H2(GV2_ref)
    GV1_ref = spinless.combine_H1(GV1_ref)
    
    # V1, V0 from GDF directly
    GV1, GV0 = spinless.get_H1_H0_from_df(gdf, C_ao_lo=C_ao_lo)
    GV1 = spinless.combine_H1_k(GV1)
    GV1 = Lat.expand(Lat.k2R(GV1))
    
    print ("V0 difference")
    print (GV0 - GV0_ref / nkpts)
    assert abs(GV0 - GV0_ref / nkpts) < 1e-10

    print (GV2_ref.shape)
    print (GV1_ref.shape)
    print (GV1.shape)
    
    order_a = []
    order_b = []
    for k in range(nkpts):
        order_a.append(np.arange(nao) + (k * 2) * nao)
        order_b.append(np.arange(nao) + (k * 2 + 1) * nao)
    order = np.vstack((order_a, order_b)).ravel()
    mesh = np.ix_(order, order)
    order_re = np.argsort(order, kind='mergesort')
    mesh_re = np.ix_(order_re, order_re)
    
    print ("GV1 diff")
    print (max_abs(GV1[mesh] - GV1_ref))

    # ***************************************************************
    # check get_jk in the KGHFPH
    # ***************************************************************
    mf = scf.KRHF(cell, kpts=kpts, exxdiv=None)
    mf = mf.density_fit()
    mf.with_df._cderi = gdf_fname

    ovlp = mf.get_ovlp()
    dm0_ao = mf.get_init_guess(key='1e')
    dm0_lo = make_basis.transform_rdm1_to_lo(dm0_ao, C_ao_lo, ovlp)

    GRho_lo = spinless.transform_rdm1_k(dm0_lo * 0.5, compact=False)
    GRho_lo_R = Lat.k2R(GRho_lo)
    
    for R in range(nkpts):
        tmp = np.random.random((nao, nao))
        GRho_lo_R[R][:nao, nao:] = tmp
        tmp = np.random.random((nao, nao))
        GRho_lo_R[R][nao:, :nao] = tmp
    
    for R in range(nkpts):
        mR = Lat.subtract(0, R)
        tmp = (GRho_lo_R[R] + GRho_lo_R[mR].T) * 0.5
        GRho_lo_R[R] = tmp
        GRho_lo_R[mR] = tmp
    
    GRho_lo = Lat.R2k(GRho_lo_R)
    GRho_ao = make_basis.transform_rdm1_to_ao(GRho_lo, C_so_lo)
    GJ_ao, GK_ao = spinless.get_jk_ph(mf, dm_kpts=GRho_ao)
    GJ = make_basis.transform_h1_to_lo(GJ_ao, C_so_lo)
    GK = make_basis.transform_h1_to_lo(GK_ao, C_so_lo)
    GJ = Lat.expand(Lat.k2R(GJ))[mesh]
    GK = Lat.expand(Lat.k2R(GK))[mesh]
    GJ_ao = Lat.expand(Lat.k2R(GJ_ao))[mesh]
    GK_ao = Lat.expand(Lat.k2R(GK_ao))[mesh]

    # reference from supercell
    GRho_sc = Lat.expand(Lat.k2R(GRho_lo))[mesh]
    J = np.einsum('pqrs, qp -> rs', GV2_ref, GRho_sc)
    K = np.einsum('pqrs, qr -> ps', GV2_ref, GRho_sc)

    print ("vj diff")
    print (max_abs(GJ - J))
    print ("vk diff")
    print (max_abs(GK - K))
    assert max_abs(GJ - J) < 1e-10
    assert max_abs(GK - K) < 1e-10
    
    # ***************************************************************
    # ERI full from supercell
    scell = Lat.bigcell 
    skmesh = [1, 1, 1]

    sLat = lattice.Lattice(scell, skmesh)
    skpts = sLat.kpts
    snao = sLat.nao
    snkpts = sLat.nkpts
    
    sC_ao_lo = lo.lowdin_k(sLat)
    sC = sLat.expand_orb(sLat.k2R(sC_ao_lo))
    print ("sC_ao_lo full shape")
    print (sC.shape)
    
    print ("C ao lo diff")
    print (max_abs(sC_ao_lo - C))
    assert max_abs(sC_ao_lo - C) < 1e-10

    sgdf_fname = 'gdf_ints.h5'
    sgdf = df.GDF(scell, skpts)
    sgdf.auxbasis = df.aug_etb(scell, beta=5.0)
    sgdf._cderi_to_save = sgdf_fname
    if True:
        sgdf.build()

    eri_ref = sgdf.get_eri(compact=False)
    eri_ref = ao2mo.kernel(eri_ref, sC)
    
    print ("eri diff")
    print (max_abs(eri_ref - eri_full))
    assert max_abs(eri_ref - eri_full) < 1e-8

def test_ph_trans_ab_initio_frozen_core():
    """
    Check AO representation and frozen core.
    """
    import copy
    from libdmet.system import lattice
    from libdmet.routine import spinless
    from libdmet.utils.misc import max_abs
    from libdmet import lo
    from libdmet.lo import iao
    import libdmet.utils.logger as log
    from libdmet.basis_transform import make_basis

    from pyscf import ao2mo
    from pyscf.pbc import gto, scf, df
    from pyscf.pbc.lib import chkfile
    
    np.set_printoptions(3, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"

    cell = gto.Cell()
    cell.a = ''' 5.0     0.0     0.0
                 0.0     5.0     0.0
                 0.0     0.0     5.0 '''
    cell.atom = ''' Li  0.0      0.0      0.0
                    H   0.0      0.0      2.0 '''
    cell.basis = {'Li': 'sto3g', 'H':'sto3g'}
    cell.verbose = 5
    cell.precision = 1e-8
    cell.build(unit='Angstrom')
    
    kmesh = [1, 1, 3]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts
    
    # normal RHF
    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf.auxbasis = df.aug_etb(cell, beta=5.0)
    gdf._cderi_to_save = gdf_fname
    gdf._cderi = gdf_fname
    if True:
    #if not os.path.exists(gdf_fname):
        gdf.build()
    
    exxdiv = None
    chkfname = 'LiH.chk'
    #if os.path.exists(chkfname):
    if False:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-10
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-10
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        kmf.kernel()
    
    nelec = cell.nelectron
    E_ref = kmf.e_tot
    Lat.analyze(kmf)

    print (cell.ao_labels())
    
    # first construct a set of LOs that froze F 1s and F 2s.
    minao = 'sto3g'
    pmol = iao.reference_mol(cell, minao=minao)
    basis = pmol._basis
    basis_val = {}
    basis_core = {}
    
    # Li core is 1s
    basis_val["Li"] = copy.deepcopy(basis["Li"])
    basis_core["Li"] = copy.deepcopy(basis["Li"])
    basis_val["Li"] = basis_val["Li"][1:]
    basis_core["Li"] = basis_core["Li"][:1]
    
    # H, no core
    basis_val["H"] = copy.deepcopy(basis["H"])

    pmol_val = pmol.copy()
    pmol_val.basis = basis_val
    pmol_val.build()

    print ("Valence:")
    val_labels = pmol_val.ao_labels()
    for i in range(len(val_labels)):
        val_labels[i] = val_labels[i].replace("Li 1s", "Li 2s")
    pmol_val.ao_labels = lambda *args: val_labels
    print (len(pmol_val.ao_labels()))
    print (pmol_val.ao_labels())
    pmol_core = pmol.copy()
    pmol_core.basis = basis_core
    pmol_core.build()
    core_labels = pmol_core.ao_labels()

    print ("Core:")
    print (len(pmol_core.ao_labels()))
    print (pmol_core.ao_labels())
    ncore = len(pmol_core.ao_labels())
    nval = pmol_val.nao_nr()
    nvirt = cell.nao_nr() - ncore - nval
    
    Lat.set_val_virt_core(nval, nvirt, ncore)
    C_ao_lo, C_ao_lo_val, C_ao_lo_virt, C_ao_lo_core = \
            make_basis.get_C_ao_lo_iao(Lat, kmf, minao='sto3g', orth_virt=True,
                                       full_virt=False, full_return=True,
                                       pmol_val=pmol_val, 
                                       pmol_core=pmol_core, tol=1e-10)

    C_ao_lo_xcore = make_basis.tile_C_ao_iao(C_ao_lo_val, C_virt=C_ao_lo_virt, C_core=None)
    
    lo_labels, val_labels, virt_labels = lo.get_labels(cell, minao='sto3g', full_virt=False, 
                                                       B2_labels=val_labels,
                                                       core_labels=core_labels)
    
    lo_labels = core_labels + lo_labels

    hcore = np.asarray(kmf.get_hcore())
    ovlp = np.asarray(kmf.get_ovlp())
    rdm1 = np.asarray(kmf.make_rdm1())
    e_nuc = kmf.energy_nuc()
    mo_coeff  = np.asarray(kmf.mo_coeff)
    mo_occ    = np.asarray(kmf.mo_occ)
    mo_energy = np.asarray(kmf.mo_energy)
    
    # core potential
    mo_core = mo_coeff[:, :, :ncore]
    mo_xcore = mo_coeff[:, :, ncore:]
    mo_occ_core = mo_occ[:, :ncore]
    mo_occ_xcore = mo_occ[:, ncore:]
    rdm1_core = kmf.make_rdm1(mo_core, mo_occ_core)
    rdm1_xcore = kmf.make_rdm1(mo_xcore, mo_occ_xcore)
    
    vj_core, vk_core = kmf.get_jk(cell, dm_kpts=rdm1_core)
    vj_ao, vk_ao = kmf.get_jk(cell, dm_kpts=rdm1)
    vj_xcore = vj_ao - vj_core
    vk_xcore = vk_ao - vk_core
    
    veff = vj_ao - vk_ao * 0.5
    veff_core = vj_core - vk_core * 0.5
    veff_xcore = veff - veff_core
    
    hcore_xcore = hcore + veff_core
    E0 = e_nuc + \
            np.einsum('kpq, kqp ->', hcore + veff_core * 0.5, rdm1_core).real / nkpts
    
    E_mf_re = np.einsum('kpq, kqp ->', hcore_xcore, rdm1_xcore) / nkpts + \
              np.einsum('kpq, kqp ->', veff_xcore, rdm1_xcore) * 0.5 / nkpts + \
              E0
    
    print ("E_mf re - E_mf ref: ", abs(E_mf_re - E_ref))
    assert abs(E_mf_re - E_ref) < 1e-10
    

    # GSO with frozen core
    from libdmet.routine import mfd
    beta = 100.0
    ewocc, Mu, err = mfd.assignocc(mo_energy, nelec * nkpts * 0.5, beta, mu0=0.0,
                                   fix_mu=False, thr_deg=1e-6, Sz=None,
                                   fit_tol=1e-12)
    print (Mu)
    print (mo_energy)
    print (ewocc)

    ovlp_ghf = spinless.combine_mo_coeff_k(ovlp) 
    C_sao_slo = spinless.combine_mo_coeff_k(C_ao_lo) 
    C_sao_slo_core = spinless.combine_mo_coeff_k(C_ao_lo_core) 
    C_sao_slo_xcore = spinless.combine_mo_coeff_k(C_ao_lo_xcore) 

    rdm1_core_lo = make_basis.transform_rdm1_to_lo(rdm1_core, C_ao_lo_core, ovlp) 
    Grdm1_core_lo = spinless.transform_rdm1_k(rdm1_core_lo * 0.5, compact=False)
    Grdm1_core = make_basis.transform_rdm1_to_ao(Grdm1_core_lo, C_sao_slo_core)
    
    rdm1_xcore_lo = make_basis.transform_rdm1_to_lo(rdm1_xcore, C_ao_lo_xcore, ovlp) 
    Grdm1_xcore_lo = spinless.transform_rdm1_k(rdm1_xcore_lo * 0.5, compact=False)
    Grdm1_xcore = make_basis.transform_rdm1_to_ao(Grdm1_xcore_lo, C_sao_slo_xcore)
    
    Grdm1 = spinless.transform_rdm1_k(rdm1 * 0.5, ovlp, compact=False)

    GH1, GH0 = spinless.transform_H1_k(hcore, ovlp, compact=False)
    GV1, GV0 = spinless.get_GV1_GV0_from_df(gdf, ovlp, compact=False)
    GH1 += GV1
    E0 = e_nuc + GH0 + GV0

    # Gmf
    cell.nelectron = GH1.shape[-1] // 2
    Gkmf = spinless.KGHFPH(cell, kpts, exxdiv=exxdiv).density_fit()
    Gkmf.with_df = gdf
    Gkmf.with_df._cderi = 'gdf_ints.h5'
    Gkmf.conv_tol = 1e-10
    Gkmf.max_cycle = 300
    #Gkmf.chkfile = chkfname
    Gkmf.get_ovlp = lambda *args: ovlp_ghf
    Gkmf.energy_nuc = lambda *args: E0
    Gkmf.Mu = Mu

    def get_hcore(*args):
        GH = np.array(GH1, copy=True)
        nao = GH.shape[-1] // 2
        GH[:, :nao, :nao] -= ovlp_ghf[:, :nao, :nao] * Gkmf.Mu
        GH[:, nao:, nao:] += ovlp_ghf[:, nao:, nao:] * Gkmf.Mu
        return GH

    Gkmf.get_hcore = get_hcore
    E_mf_gso = Gkmf.kernel(dm0=Grdm1)  
    
    print ("E_mf_gso - E_mf ref: ", abs(E_mf_gso - E_ref))
    assert abs(E_mf_gso - E_ref) < 1e-12
    
    Gveff_core = Gkmf.get_veff(dm_kpts=Grdm1_core)
    Gveff_xcore = Gkmf.get_veff(dm_kpts=Grdm1_xcore)
    Ghcore_xcore = GH1 + Gveff_core
    E0 += np.einsum('kpq, kqp ->', GH1 + Gveff_core * 0.5, Grdm1_core).real / nkpts
    E_mf_re_gso = np.einsum('kpq, kqp ->', Ghcore_xcore, Grdm1_xcore) / nkpts + \
                  np.einsum('kpq, kqp ->', Gveff_xcore, Grdm1_xcore) * 0.5 / nkpts + \
                  E0
    
    print ("E_mf re (GSO) - E_mf ref: ", abs(E_mf_re_gso - E_ref))
    assert abs(E_mf_re_gso - E_ref) < 1e-12

def test_ph_trans_ab_initio_frozen_core_dft():
    """
    Check AO representation and frozen core. DFT.
    """
    import copy
    from libdmet.system import lattice
    from libdmet.routine import spinless
    from libdmet.routine import pbc_helper as pbc_hp
    from libdmet.utils.misc import max_abs
    from libdmet import lo
    from libdmet.lo import iao
    import libdmet.utils.logger as log
    from libdmet.basis_transform import make_basis, eri_transform

    from pyscf import ao2mo
    from pyscf.pbc import gto, scf, df
    from pyscf.pbc.lib import chkfile
    
    np.set_printoptions(3, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"

    cell = gto.Cell()
    cell.a = ''' 5.0     0.0     0.0
                 0.0     5.0     0.0
                 0.0     0.0     5.0 '''
    cell.atom = ''' Li  0.0      0.0      0.0
                    H   0.0      0.0      2.0 '''
    cell.basis = {'Li': 'sto3g', 'H':'sto3g'}
    cell.verbose = 5
    cell.precision = 1e-8
    cell.build(unit='Angstrom')
    
    kmesh = [1, 1, 3]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts
    
    # normal RHF
    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf.auxbasis = df.aug_etb(cell, beta=5.0)
    gdf._cderi_to_save = gdf_fname
    gdf._cderi = gdf_fname
    if True:
    #if not os.path.exists(gdf_fname):
        gdf.build()
    
    exxdiv = None
    chkfname = 'LiH.chk'
    #if os.path.exists(chkfname):
    if False:
        kmf = scf.KRKS(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.xc = 'pbe0'
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-10
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRKS(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.xc = 'pbe0'
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-10
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        kmf.kernel()
    
    nelec = cell.nelectron
    E_ref = kmf.e_tot
    Lat.analyze(kmf)
    
    omega, alpha, hyb = pbc_hp.get_hybrid_param(kmf)

    print (cell.ao_labels())
    
    # first construct a set of LOs that froze F 1s and F 2s.
    minao = 'sto3g'
    pmol = iao.reference_mol(cell, minao=minao)
    basis = pmol._basis
    basis_val = {}
    basis_core = {}
    
    # Li core is 1s
    basis_val["Li"] = copy.deepcopy(basis["Li"])
    basis_core["Li"] = copy.deepcopy(basis["Li"])
    basis_val["Li"] = basis_val["Li"][1:]
    basis_core["Li"] = basis_core["Li"][:1]
    
    # H, no core
    basis_val["H"] = copy.deepcopy(basis["H"])

    pmol_val = pmol.copy()
    pmol_val.basis = basis_val
    pmol_val.build()

    print ("Valence:")
    val_labels = pmol_val.ao_labels()
    for i in range(len(val_labels)):
        val_labels[i] = val_labels[i].replace("Li 1s", "Li 2s")
    pmol_val.ao_labels = lambda *args: val_labels
    print (len(pmol_val.ao_labels()))
    print (pmol_val.ao_labels())
    pmol_core = pmol.copy()
    pmol_core.basis = basis_core
    pmol_core.build()
    core_labels = pmol_core.ao_labels()

    print ("Core:")
    print (len(pmol_core.ao_labels()))
    print (pmol_core.ao_labels())
    ncore = len(pmol_core.ao_labels())
    nval = pmol_val.nao_nr()
    nvirt = cell.nao_nr() - ncore - nval
    
    Lat.set_val_virt_core(nval, nvirt, ncore)
    C_ao_lo, C_ao_lo_val, C_ao_lo_virt, C_ao_lo_core = \
            make_basis.get_C_ao_lo_iao(Lat, kmf, minao='sto3g', orth_virt=True,
                                       full_virt=False, full_return=True,
                                       pmol_val=pmol_val, 
                                       pmol_core=pmol_core, tol=1e-10)

    C_ao_lo_xcore = make_basis.tile_C_ao_iao(C_ao_lo_val, C_virt=C_ao_lo_virt, C_core=None)
    
    lo_labels, val_labels, virt_labels = lo.get_labels(cell, minao='sto3g', full_virt=False, 
                                                       B2_labels=val_labels,
                                                       core_labels=core_labels)
    
    lo_labels = core_labels + lo_labels

    hcore = np.asarray(kmf.get_hcore())
    ovlp = np.asarray(kmf.get_ovlp())
    rdm1 = np.asarray(kmf.make_rdm1())
    e_nuc = kmf.energy_nuc()
    mo_coeff  = np.asarray(kmf.mo_coeff)
    mo_occ    = np.asarray(kmf.mo_occ)
    mo_energy = np.asarray(kmf.mo_energy)
    
    # core potential
    mo_core = mo_coeff[:, :, :ncore]
    mo_xcore = mo_coeff[:, :, ncore:]
    mo_occ_core = mo_occ[:, :ncore]
    mo_occ_xcore = mo_occ[:, ncore:]
    rdm1_core = kmf.make_rdm1(mo_core, mo_occ_core)
    rdm1_xcore = kmf.make_rdm1(mo_xcore, mo_occ_xcore)
    
    vj_core, vk_core = kmf.get_jk(cell, dm_kpts=rdm1_core)
    vj_ao, vk_ao = kmf.get_jk(cell, dm_kpts=rdm1)
    vj_xcore = vj_ao - vj_core
    vk_xcore = vk_ao - vk_core
    
    vhf = vj_ao - vk_ao * (0.5 * hyb)
    vhf_core = vj_core - vk_core * (0.5 * hyb)
    vhf_xcore = vhf - vhf_core
    
    hcore_xcore = hcore + vhf_core
    E0 = e_nuc + \
            np.einsum('kpq, kqp ->', hcore + vhf_core * 0.5, rdm1_core).real / nkpts
    
#    # kmf with frozen core
    vxc_ao = pbc_hp.get_vxc(kmf)
    Exc = vxc_ao.exc
    vxc_lo = make_basis.transform_h1_to_lo(vxc_ao, C_ao_lo_xcore)

    hcore_xcore_lo = make_basis.transform_h1_to_lo(hcore_xcore, C_ao_lo_xcore)
    ovlp_xcore_lo = make_basis.transform_h1_to_lo(ovlp, C_ao_lo_xcore)
    gdf_lo = eri_transform.transform_gdf_to_lo(gdf, C_ao_lo_xcore, fname="gdf_ints_lo.h5")
    rdm1_xcore_lo = make_basis.transform_rdm1_to_lo(rdm1_xcore, C_ao_lo_xcore, ovlp)

#    nao = cell.nao_nr()
#    cell.nelectron = nelec - ncore * 2
#    cell.nao_nr = lambda *args: nao - ncore
#    kmf = spinless.KRKS_LO(cell, kpts=kpts, xc='pbe0', C_ao_lo=C_ao_lo_xcore,
#                           dm_core_ao=rdm1_core).density_fit()
#    kmf.exxdiv = None
#    kmf.with_df = gdf_lo
#    kmf.with_df._cderi = "gdf_ints_lo.h5"
#    kmf.conv_tol = 1e-10
#    kmf.max_cycle = 300
#    #kmf.chkfile = chkfname
#    kmf.get_ovlp = lambda *args: ovlp_xcore_lo
#    kmf.get_hcore = lambda *args: hcore_xcore_lo
#    kmf.energy_nuc = lambda *args: E0
#    E_mf_frozen = kmf.kernel(dm0=rdm1_xcore_lo) 
#    vxc_lo_2 = pbc_hp.get_vxc(kmf)
#    
#    assert max_abs(vxc_lo_2 - vxc_lo) < 1e-8
#    
#    print ("normal DFT frozen core")
#    print (abs(E_mf_frozen - E_ref))
#    assert abs(E_mf_frozen - E_ref) < 1e-10
    
    # GSO with frozen core
    from libdmet.routine import mfd
    beta = 1000.0
    ewocc, Mu, err = mfd.assignocc(mo_energy, nelec * nkpts * 0.5, beta, mu0=0.0,
                                   fix_mu=False, thr_deg=1e-6, Sz=None,
                                   fit_tol=1e-12)
    print (Mu)
    print (mo_energy)
    print (ewocc)
    
    ovlp_ghf = spinless.combine_mo_coeff_k(ovlp) 
    C_sao_slo = spinless.combine_mo_coeff_k(C_ao_lo) 
    C_sao_slo_core = spinless.combine_mo_coeff_k(C_ao_lo_core) 
    C_sao_slo_xcore = spinless.combine_mo_coeff_k(C_ao_lo_xcore) 

    rdm1_core_lo = make_basis.transform_rdm1_to_lo(rdm1_core, C_ao_lo_core, ovlp) 
    Grdm1_core_lo = spinless.transform_rdm1_k(rdm1_core_lo * 0.5, compact=False)
    Grdm1_core = make_basis.transform_rdm1_to_ao(Grdm1_core_lo, C_sao_slo_core)
    
    rdm1_xcore_lo = make_basis.transform_rdm1_to_lo(rdm1_xcore, C_ao_lo_xcore, ovlp) 
    Grdm1_xcore_lo = spinless.transform_rdm1_k(rdm1_xcore_lo * 0.5, compact=False)
    Grdm1_xcore = make_basis.transform_rdm1_to_ao(Grdm1_xcore_lo, C_sao_slo_xcore)
    
    Grdm1 = spinless.transform_rdm1_k(rdm1 * 0.5, ovlp, compact=False)

    GH1, GH0 = spinless.transform_H1_k(hcore, ovlp, compact=False)
    GV1, GV0 = spinless.get_GV1_GV0_from_df(gdf, ovlp, compact=False, hyb=hyb)
    GH1 += GV1
    E0 = e_nuc + GH0 + GV0
    
    Gvxc, Gvxc_0 = spinless.transform_H1_k(vxc_ao, ovlp, compact=False)
    
    # Gmf
    cell.nelectron = GH1.shape[-1] // 2
    Gkmf = spinless.KGHFPH(cell, kpts, exxdiv=exxdiv).density_fit()
    Gkmf.with_df = gdf
    Gkmf.with_df._cderi = 'gdf_ints.h5'
    Gkmf.conv_tol = 1e-10
    Gkmf.max_cycle = 300
    Gkmf.get_ovlp = lambda *args: ovlp_ghf
    Gkmf.energy_nuc = lambda *args: E0
    Gkmf.Mu = Mu

    def get_hcore(*args):
        GH = np.array(GH1, copy=True)
        nao = GH.shape[-1] // 2
        GH[:, :nao, :nao] -= ovlp_ghf[:, :nao, :nao] * Gkmf.Mu
        GH[:, nao:, nao:] += ovlp_ghf[:, nao:, nao:] * Gkmf.Mu
        return GH
    Gkmf.get_hcore = get_hcore
    
    Gvj_core, Gvk_core = Gkmf.get_jk(dm_kpts=Grdm1_core)
    Gvj_xcore, Gvk_xcore = Gkmf.get_jk(dm_kpts=Grdm1_xcore)
    
    Gvhf_core = Gvj_core - Gvk_core * hyb
    Gvhf_xcore = Gvj_xcore - Gvk_xcore * hyb
    
    Ghcore_xcore = GH1 + Gvhf_core
    E0 += np.einsum('kpq, kqp ->', GH1 + Gvhf_core * 0.5, Grdm1_core).real / nkpts
    E_mf_re_gso = np.einsum('kpq, kqp ->', Ghcore_xcore, Grdm1_xcore) / nkpts + \
                  np.einsum('kpq, kqp ->', Gvhf_xcore, Grdm1_xcore) * 0.5 / nkpts + \
                  E0 + Exc
    print ("E_mf re (GSO) DFT- E_mf ref: ", abs(E_mf_re_gso - E_ref))
    assert abs(E_mf_re_gso - E_ref) < 1e-10

def test_ph_trans_hchain():
    """
    P-H transform on H6 321G.
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

    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    #if not os.path.isfile(gdf_fname):
    if True:
        gdf.build()

    chkfname = 'hchain.chk'
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
    
    E_ref = kmf.e_tot
    kmf = Lat.symmetrize_kmf(kmf)
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, \
            kmf, minao='minao', full_return=True)
    C_ao_iao = Lat.symmetrize_lo(C_ao_iao)
    
    ncore = 0
    nval = C_ao_iao_val.shape[-1]
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)
    
    C_ao_lo = C_ao_iao
    Lat.set_Ham(kmf, gdf, C_ao_lo)
    
    # LO basis
    # transform everything to LO
    print ("*" * 79)
    print ("normal GHF")

    hcore = kmf.get_hcore()
    ovlp = kmf.get_ovlp()
    rdm1 = kmf.make_rdm1()
    vj, vk = kmf.get_jk()
    
    hcore_lo = make_basis.transform_h1_to_lo(hcore, C_ao_lo)
    ovlp_lo = make_basis.transform_h1_to_lo(ovlp, C_ao_lo)
    vj_lo = make_basis.transform_h1_to_lo(vj, C_ao_lo)
    vk_lo = make_basis.transform_h1_to_lo(vk, C_ao_lo)
    rdm1_lo = make_basis.transform_rdm1_to_lo(rdm1, C_ao_lo, ovlp)
    gdf_lo = eri_transform.transform_gdf_to_lo(gdf, C_ao_lo, 
                                               fname="gdf_ints_lo.h5",
                                               t_reversal_symm=True)
    e_nuc = kmf.energy_nuc()
    C_lo = np.zeros_like(C_ao_lo)
    for k in range(nkpts):
        C_lo[k][range(nao), range(nao)] = 1.0
    
    # normal GHF
    hcore_so = spinless.combine_mo_coeff_k(hcore_lo)
    ovlp_so = spinless.combine_mo_coeff_k(ovlp_lo)
    vj_so = spinless.combine_mo_coeff_k(vj_lo)
    vk_so = spinless.combine_mo_coeff_k(vk_lo)
    rdm1_so = spinless.combine_mo_coeff_k(rdm1_lo * 0.5)

    kmf = spinless.KGHF(cell, kpts, exxdiv=exxdiv)
    kmf.get_hcore = lambda *args: hcore_so
    kmf.get_ovlp = lambda *args: ovlp_so
    kmf.with_df = gdf_lo
    kmf.with_df._cderi = 'gdf_ints_lo.h5'
    
    kmf.conv_tol = 1e-12
    kmf.max_cycle = 300
    kmf.chkfile = chkfname
    E_ghf = kmf.kernel(dm0=rdm1_so)
    
    print ("GHF diff")
    print (abs(E_ghf - E_ref))
    assert abs(E_ghf - E_ref) < 1e-10
    

    print ("*" * 79) 
    print ("LO basis")
    
    #eri_8d = spinless.get_eri_8d(cell, gdf_lo, kpts=None, compact=False)
    #eri_8d_R = Lat.k2R_H2_8d(eri_8d)
    #eri_full = eri_8d_R.transpose(0, 4, 1, 5, 2, 6, 3, 7)\
    #        .reshape(nkpts*nao, nkpts*nao, nkpts*nao, nkpts*nao)
    #GV2_ref, GV1_ref, GV0_ref = spinless.transform_H2_local(eri_full, compact=False)
    #def get_jk_G(cell=None, dm_kpts=None, hermi=0, kpts=None, kpts_band=None,
    #              with_j=True, with_k=True, **kwargs):
    #    vj, vk = mol_scf.hf.dot_eri_dm(GV2_ref, dm_kpts[0], hermi=1, 
    #                                     with_j=with_j, with_k=with_k)
    #    return vj[None], vk[None]

    # PH transform
    GH1, GH0 = spinless.transform_H1_k(hcore_lo, compact=False)
    GV1, GV0, j_add, k_add = spinless.get_H1_H0_from_df(gdf_lo, C_ao_lo=C_lo,
                                                        compact=False,
                                                        return_jk=True)
    GH1 += GV1
    GH0 += GV0
    GH0 += kmf.energy_nuc()
    GRho_k = spinless.transform_rdm1_k(rdm1_lo * 0.5)
    GS = spinless.combine_mo_coeff_k(ovlp_lo)
    
    GJ1, GJ0 = spinless.transform_H1_k(vj_lo, compact=False)
    GK1, GK0 = spinless.transform_H1_k(vk_lo * 0.5, compact=False)

    cell.nelectron = GRho_k.shape[-1] // 2
    kmf = spinless.KGHFPH(cell, kpts, exxdiv=exxdiv)
    kmf.get_hcore = lambda *args: GH1
    kmf.get_ovlp = lambda *args: GS
    kmf.energy_nuc = lambda *args: GH0
    kmf.with_df = gdf_lo
    kmf.with_df._cderi = 'gdf_ints_lo.h5'
    #kmf.get_jk = get_jk_G
    
    kmf.conv_tol = 1e-12
    kmf.max_cycle = 300
    kmf.chkfile = chkfname
    E_lo = kmf.kernel(dm0=GRho_k)
    
    print ("LO diff")
    print (abs(E_lo - E_ref))
    assert abs(E_lo - E_ref) < 1e-10
    
    # check vj, vk
    vj, vk = kmf.get_jk()
    
    GJ1[:, :nao, :nao] -= j_add[0]
    GJ1[:, nao:, nao:] += j_add[1]
    
    print ("J1 check")
    print (max_abs(GJ1 - vj))
    assert max_abs(GJ1 - vj) < 1e-9
    
    GK1[:, nao:, nao:] += k_add
    
    print ("K1 check")
    print (max_abs(GK1 - vk))
    assert max_abs(GK1 - vk) < 1e-9
    
    print ("*" * 79) 
    print ("AO basis")
    # p-h transformed hamiltonian.
    C_sao_slo = spinless.combine_mo_coeff_k(C_ao_lo)
    ovlp_ghf = spinless.combine_mo_coeff_k(ovlp)

    H1_k = Lat.getH1(kspace=True)
    GH1, GH0 = spinless.transform_H1_k(H1_k)
    GV1, GV0 = spinless.get_H1_H0_from_df(gdf, C_ao_lo=C_ao_lo)
    GH1 += GV1
    GH1 = spinless.combine_H1_k(GH1)
    GH0 += GV0
    GH0 += e_nuc
    GRho_k = spinless.transform_rdm1_k(Lat.rdm1_lo_k * 0.5)
    
    # transform back to AO
    GH1_ao_k = make_basis.transform_h1_to_ao(GH1, C_sao_slo, ovlp_ghf)
    GRho_ao_k = make_basis.transform_rdm1_to_ao(GRho_k, C_sao_slo)
    
    cell.nelectron = GRho_k.shape[-1] // 2
    kmf = spinless.KGHFPH(cell, kpts, exxdiv=exxdiv)
    kmf.with_df = gdf
    kmf.with_df._cderi = 'gdf_ints.h5'
    kmf.conv_tol = 1e-12
    kmf.max_cycle = 300
    kmf.chkfile = chkfname
    kmf.get_hcore = lambda *args: GH1_ao_k
    kmf.get_ovlp = lambda *args: ovlp_ghf
    kmf.energy_nuc = lambda *args: GH0
    E_ao = kmf.kernel(dm0=GRho_ao_k)
    
    print ("AO diff")
    print (abs(E_ao - E_ref))
    assert abs(E_ao - E_ref) < 1e-10

@pytest.mark.parametrize(
    "xc",
    ["b3lyp"],
)
def test_pdft_lo_xc(xc):
    import os, sys, copy
    from libdmet.utils import logger as log
    import libdmet.dmet.HubbardGSO as dmet
    from libdmet.utils.misc import max_abs, mdot
    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis, eri_transform
    from libdmet.lo import iao, lowdin
    from libdmet.routine.pbc_helper import KRKS_LO, KUKS_LO
    from libdmet.routine import pbc_helper as pbc_hp

    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, dft, gto, df
    
    np.set_printoptions(4, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"

    ### ************************************************************
    ### System settings
    ### ************************************************************
    cell = gto.Cell()
    cell.a = ''' 5.0     0.0     0.0
                 0.0     5.0     0.0
                 0.0     0.0     5.0 '''
    cell.atom = ''' Li  0.0      0.0      0.0
                    H   0.0      0.0      2.5 '''
    cell.basis = {'Li': 'minao', 'H':'321g'}
    cell.verbose = 5
    cell.precision = 1e-10
    cell.build(unit='Angstrom')
    cell_lo = cell.copy()

    kmesh = [1, 1, 2]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts
    exxdiv = None

    ### ************************************************************
    ### SCF Mean-field calculation
    ### ************************************************************
    log.section("\nSolving SCF mean-field problem\n")

    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    #gdf.auxbasis = {'Li': 'def2-svp-jk', 'H': 'def2-svp-jk'}
    gdf._cderi_to_save = gdf_fname
    #if not os.path.isfile(gdf_fname):
    if True:
        gdf.build()
    
    print ("DFT")
    chkfname = 'hchain.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = dft.KRKS(cell, kpts).density_fit()
        kmf.exxdiv = exxdiv
        kmf.xc = xc
        #kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = dft.KRKS(cell, kpts).density_fit()
        kmf.exxdiv = exxdiv
        kmf.xc = xc
        #kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        kmf.kernel()
    
    # check the get_hybrid_param
    omega, alpha, hyb = pbc_hp.get_hybrid_param(kmf)
    # check get_vxc_krks
    vxc = pbc_hp.get_vxc(kmf)
    if xc == 'b3lyp':
        assert hyb == 0.2
    vj, vk = kmf.get_jk()
    vj = np.asarray(vj)
    vk = np.asarray(vk)
    veff = np.asarray(kmf.get_veff())
    diff_vxc = max_abs(vj - vk * (0.5 * hyb) + vxc - veff)
    print ("diff vxc:", diff_vxc)
    assert diff_vxc < 1e-10
    
    ### ************************************************************
    ### Pre-processing, LO and subspace partition
    ### ************************************************************
    E_kmf = kmf.e_tot
    C_ao_mo = np.asarray(kmf.mo_coeff)
    mo_occ = np.asarray(kmf.mo_occ)
    hcore = np.asarray(kmf.get_hcore())
    ovlp = np.asarray(kmf.get_ovlp())
    rdm1 = np.asarray(kmf.make_rdm1())
    minao = 'minao'
    pmol = iao.reference_mol(cell, minao=minao)

    basis = pmol._basis
    basis_val = {}
    basis_core = {}

    # Li core is 1s
    basis_val["Li"] = copy.deepcopy(basis["Li"])
    basis_core["Li"] = copy.deepcopy(basis["Li"])
    for i in range(1, len(basis_val["Li"][0])):
        basis_val["Li"][0][i].pop(1)
        basis_core["Li"][0][i].pop(2)
    # H has no core
    basis_val["H"] = copy.deepcopy(basis["H"])

    pmol_val = pmol.copy()
    pmol_val.basis = basis_val
    pmol_val.build()

    print ("Valence:")
    print (len(pmol_val.ao_labels()))
    print (pmol_val.ao_labels())

    pmol_core = pmol.copy()
    pmol_core.basis = basis_core
    pmol_core.build()

    print ("Core:")
    print (len(pmol_core.ao_labels()))
    print (pmol_core.ao_labels())

    ncore = len(pmol_core.ao_labels())
    nval = pmol_val.nao_nr()
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)

    log.section("\nPre-process, orbital localization and subspace partition\n")
    C_ao_lo, C_ao_lo_val, C_ao_lo_virt, C_ao_lo_core = make_basis.get_C_ao_lo_iao(Lat, \
            kmf, minao='minao', full_return=True, pmol_val=pmol_val,
            pmol_core=pmol_core)
    C_ao_lo_xcore = make_basis.tile_C_ao_iao(C_ao_lo_val, C_ao_lo_virt) 
    
    assert lowdin.check_orthonormal(C_ao_lo_core, ovlp)
    assert lowdin.check_orthonormal(C_ao_lo_val, ovlp)
    assert lowdin.check_orthonormal(C_ao_lo_virt, ovlp)
    assert lowdin.check_orthonormal(C_ao_lo, ovlp)

    assert lowdin.check_orthogonal(C_ao_lo_core, C_ao_lo_val, ovlp)
    assert lowdin.check_orthogonal(C_ao_lo_core, C_ao_lo_virt, ovlp)
    assert lowdin.check_orthogonal(C_ao_lo_virt, C_ao_lo_val, ovlp)

    assert lowdin.check_span_same_space(C_ao_lo_core, C_ao_mo[:, :, :ncore], ovlp)
    assert lowdin.check_span_same_space(C_ao_lo_xcore, C_ao_mo[:, :, ncore:], ovlp)
    
    rdm1_core = kmf.make_rdm1(C_ao_mo[:, :, :ncore], mo_occ[:, :ncore])
    
    omega, alpha, hyb = pbc_hp.get_hybrid_param(kmf)
    vj, vk = kmf.get_jk(cell, rdm1_core)
    veff_core = vj - vk * (hyb * 0.5)
    E_core = np.einsum('kij, kji ->', hcore + veff_core * 0.5, \
            rdm1_core).real / nkpts
    hcore_lo = make_basis.transform_h1_to_lo(hcore + veff_core, C_ao_lo_xcore)
    ovlp_lo = make_basis.transform_h1_to_lo(ovlp, C_ao_lo_xcore)
    rdm1_lo = make_basis.transform_rdm1_to_lo(rdm1 - rdm1_core, C_ao_lo_xcore, ovlp)
    gdf_lo = eri_transform.transform_gdf_to_lo(gdf, C_ao_lo_xcore, fname="gdf_ints_lo.h5", \
        t_reversal_symm=True)
    
    nlo_xcore = nao - ncore
    cell_lo.nao_nr = lambda *args: nlo_xcore
    cell_lo.nelectron = cell_lo.nelectron - ncore * 2
    ncore = 0
    nval = C_ao_lo_val.shape[-1]
    nvirt = cell_lo.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)
    
    kmf_lo = KRKS_LO(cell_lo, C_ao_lo=C_ao_lo_xcore, kpts=kpts, dm_core_ao=rdm1_core).density_fit()
    kmf_lo.exxdiv = kmf.exxdiv
    kmf_lo.xc = kmf.xc
    kmf_lo.C_ao_lo = C_ao_lo_xcore
    kmf_lo.get_hcore = lambda *args: hcore_lo 
    kmf_lo.get_ovlp = lambda *args: ovlp_lo
    kmf_lo.energy_nuc = lambda *args: cell_lo.energy_nuc() + E_core
    kmf_lo.with_df = gdf_lo
    kmf_lo.with_df._cderi = 'gdf_ints_lo.h5'
    kmf_lo.kernel(dm0=rdm1_lo)
    E_kmf2 = kmf_lo.e_tot
    rdm1_lo2 = kmf_lo.make_rdm1()

    diff_E = abs(E_kmf2 - E_kmf)
    diff_rdm1 = max_abs(rdm1_lo2 - rdm1_lo)
    print ("diff E: ", diff_E)
    print ("diff rdm1: ", diff_rdm1)
    assert diff_E < 1e-9
    assert diff_rdm1 < 1e-6
    
    # check get_vxc_krks
    vxc = kmf_lo.get_vxc()
    vj, vk = kmf_lo.get_jk()
    vj = np.asarray(vj)
    vk = np.asarray(vk)
    veff = np.asarray(kmf_lo.get_veff())
    diff_vxc = max_abs(vj - vk * (0.5 * hyb) + vxc - veff)
    print ("diff vxc:", diff_vxc)
    assert diff_vxc < 1e-9

    rdm1_core_uks = np.asarray((rdm1_core, rdm1_core)) * 0.5
    rdm1_lo_uks = np.asarray((rdm1_lo, rdm1_lo)) * 0.5
    kmf_lo = KUKS_LO(cell_lo, C_ao_lo=C_ao_lo_xcore, kpts=kpts, \
            dm_core_ao=rdm1_core_uks).density_fit()
    kmf_lo.exxdiv = kmf.exxdiv
    kmf_lo.xc = kmf.xc
    kmf_lo.C_ao_lo = C_ao_lo_xcore
    kmf_lo.get_hcore = lambda *args: hcore_lo 
    kmf_lo.get_ovlp = lambda *args: ovlp_lo
    kmf_lo.energy_nuc = lambda *args: cell_lo.energy_nuc() + E_core
    kmf_lo.with_df = gdf_lo
    kmf_lo.with_df._cderi = 'gdf_ints_lo.h5'
    kmf_lo.kernel(dm0=rdm1_lo_uks)
    E_kmf3 = kmf_lo.e_tot
    rdm1_lo3 = np.asarray(kmf_lo.make_rdm1())

    diff_E = abs(E_kmf3 - E_kmf)
    diff_rdm1 = max_abs(rdm1_lo3[0] + rdm1_lo3[1] - rdm1_lo)
    print ("diff E (UB3LYP): ", diff_E)
    print ("diff rdm1 (UB3LYP): ", diff_rdm1)
    assert diff_E < 1e-9
    assert diff_rdm1 < 1e-6
    
    # check get_vxc_krks
    vxc = kmf_lo.get_vxc()
    vj, vk = kmf_lo.get_jk()
    vj = np.asarray(vj)
    vk = np.asarray(vk)
    veff = np.asarray(kmf_lo.get_veff())
    diff_vxc = max_abs(vj[0] + vj[1] - vk * hyb + vxc - veff)
    print ("diff vxc:", diff_vxc)
    assert diff_vxc < 1e-9
    
    print ("*********************************************")
    print ("Hartree Fock")
    cell_lo = cell.copy()

    chkfname = 'hchain.chk'
    #if os.path.isfile(chkfname):
    if False:
        #kmf = dft.KRKS(cell, kpts).density_fit()
        #kmf.exxdiv = exxdiv
        #kmf.xc = xc
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        #kmf = dft.KRKS(cell, kpts).density_fit()
        #kmf.exxdiv = exxdiv
        #kmf.xc = xc
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        kmf.kernel()
    
    ### ************************************************************
    ### Pre-processing, LO and subspace partition
    ### ************************************************************
    E_kmf = kmf.e_tot
    C_ao_mo = np.asarray(kmf.mo_coeff)
    mo_occ = np.asarray(kmf.mo_occ)
    hcore = np.asarray(kmf.get_hcore())
    ovlp = np.asarray(kmf.get_ovlp())
    rdm1 = np.asarray(kmf.make_rdm1())
    minao = 'minao'
    pmol = iao.reference_mol(cell, minao=minao)

    basis = pmol._basis
    basis_val = {}
    basis_core = {}

    # Li core is 1s
    basis_val["Li"] = copy.deepcopy(basis["Li"])
    basis_core["Li"] = copy.deepcopy(basis["Li"])
    for i in range(1, len(basis_val["Li"][0])):
        basis_val["Li"][0][i].pop(1)
        basis_core["Li"][0][i].pop(2)
    # H has no core
    basis_val["H"] = copy.deepcopy(basis["H"])

    pmol_val = pmol.copy()
    pmol_val.basis = basis_val
    pmol_val.build()

    print ("Valence:")
    print (len(pmol_val.ao_labels()))
    print (pmol_val.ao_labels())

    pmol_core = pmol.copy()
    pmol_core.basis = basis_core
    pmol_core.build()

    print ("Core:")
    print (len(pmol_core.ao_labels()))
    print (pmol_core.ao_labels())

    ncore = len(pmol_core.ao_labels())
    nval = pmol_val.nao_nr()
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)

    log.section("\nPre-process, orbital localization and subspace partition\n")
    C_ao_lo, C_ao_lo_val, C_ao_lo_virt, C_ao_lo_core = make_basis.get_C_ao_lo_iao(Lat, \
            kmf, minao='minao', full_return=True, pmol_val=pmol_val,
            pmol_core=pmol_core)
    C_ao_lo_xcore = make_basis.tile_C_ao_iao(C_ao_lo_val, C_ao_lo_virt) 
    
    assert lowdin.check_orthonormal(C_ao_lo_core, ovlp)
    assert lowdin.check_orthonormal(C_ao_lo_val, ovlp)
    assert lowdin.check_orthonormal(C_ao_lo_virt, ovlp)
    assert lowdin.check_orthonormal(C_ao_lo, ovlp)

    assert lowdin.check_orthogonal(C_ao_lo_core, C_ao_lo_val, ovlp)
    assert lowdin.check_orthogonal(C_ao_lo_core, C_ao_lo_virt, ovlp)
    assert lowdin.check_orthogonal(C_ao_lo_virt, C_ao_lo_val, ovlp)

    assert lowdin.check_span_same_space(C_ao_lo_core, C_ao_mo[:, :, :ncore], ovlp)
    assert lowdin.check_span_same_space(C_ao_lo_xcore, C_ao_mo[:, :, ncore:], ovlp)
    
    rdm1_core = kmf.make_rdm1(C_ao_mo[:, :, :ncore], mo_occ[:, :ncore])
    veff_core = kmf.get_veff(dm_kpts=rdm1_core)
    E_core = np.einsum('kij, kji ->', hcore + veff_core * 0.5, \
            rdm1_core).real / nkpts
    hcore_lo = make_basis.transform_h1_to_lo(hcore + veff_core, C_ao_lo_xcore)
    ovlp_lo = make_basis.transform_h1_to_lo(ovlp, C_ao_lo_xcore)
    rdm1_lo = make_basis.transform_rdm1_to_lo(rdm1 - rdm1_core, C_ao_lo_xcore, ovlp)
    gdf_lo = eri_transform.transform_gdf_to_lo(gdf, C_ao_lo_xcore, fname="gdf_ints_lo.h5", \
        t_reversal_symm=True)
    
    nlo_xcore = nao - ncore
    cell_lo.nao_nr = lambda *args: nlo_xcore
    cell_lo.nelectron = cell_lo.nelectron - ncore * 2
    
    ncore = 0
    nval = C_ao_lo_val.shape[-1]
    nvirt = cell_lo.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)
    
    kmf_lo = KRKS_LO(cell_lo, C_ao_lo=C_ao_lo_xcore, kpts=kpts, dm_core_ao=rdm1_core).density_fit()
    kmf_lo.exxdiv = kmf.exxdiv
    kmf_lo.xc = 'hf'
    kmf_lo.C_ao_lo = C_ao_lo_xcore
    kmf_lo.get_hcore = lambda *args: hcore_lo 
    kmf_lo.get_ovlp = lambda *args: ovlp_lo
    kmf_lo.energy_nuc = lambda *args: cell_lo.energy_nuc() + E_core
    kmf_lo.with_df = gdf_lo
    kmf_lo.kernel(dm0=rdm1_lo)
    E_kmf2 = kmf_lo.e_tot
    rdm1_lo2 = kmf_lo.make_rdm1()

    diff_E = abs(E_kmf2 - E_kmf)
    diff_rdm1 = max_abs(rdm1_lo2 - rdm1_lo)
    print ("diff E: ", diff_E)
    print ("diff rdm1: ", diff_rdm1)
    assert diff_E < 1e-9
    assert diff_rdm1 < 1e-6

def test_krhf_smearing():
    import numpy as np
    from pyscf import lib
    import pyscf.pbc.gto as pbcgto
    import pyscf.pbc.scf as pscf
    from libdmet.routine import pbc_helper as pbc_hp
    cell = pbcgto.Cell()
    cell.atom = '''
    He 0 0 1
    He 1 0 1
    '''
    cell.basis = [[0, [1., 1.]], [0, [0.5, 1]]]
    cell.a = np.eye(3) * 3
    cell.mesh = [10] * 3
    cell.verbose = 5
    cell.build()
    nao = cell.nao_nr()
    
    mf = pscf.KRHF(cell, cell.make_kpts([2,1,1]))
    nkpts = len(mf.kpts)
    pbc_hp.smearing_(mf, 0.1, 'fermi')
    mo_energy_kpts = np.array([np.arange(nao)*.2+np.cos(i+.5)*.1
                                  for i in range(nkpts)])
    occ = mf.get_occ(mo_energy_kpts)
    assert abs(mf.entropy - 6.1656394960533021/2) < 1e-9

    mf.smearing_method = 'gauss'
    occ = mf.get_occ(mo_energy_kpts)
    assert abs(mf.entropy - 0.94924016074521311/2) < 1e-9

    mf.kernel()
    assert abs(mf.entropy) < 1e-15

def test_kuhf_smearing():
    import numpy as np
    from pyscf import lib
    import pyscf.pbc.gto as pbcgto
    import pyscf.pbc.scf as pscf
    from libdmet.routine import pbc_helper as pbc_hp
    cell = pbcgto.Cell()
    cell.atom = '''
    He 0 0 1
    He 1 0 1
    '''
    cell.basis = [[0, [1., 1.]], [0, [0.5, 1]]]
    cell.a = np.eye(3) * 3
    cell.mesh = [10] * 3
    cell.verbose = 5
    cell.build()
    nao = cell.nao_nr()
    
    mf = pscf.KUHF(cell, cell.make_kpts([2,1,1]))
    nkpts = len(mf.kpts)
    pbc_hp.smearing_(mf, 0.1, 'fermi', fit_spin=False)
    mo_energy_kpts = np.array([np.arange(nao)*.2+np.cos(i+.5)*.1
                               for i in range(nkpts)])
    mo_energy_kpts = np.array([mo_energy_kpts, 
                               mo_energy_kpts+np.cos(mo_energy_kpts)*.02])

    occ = mf.get_occ(mo_energy_kpts)
    assert abs(mf.entropy - 6.1803390081500869/2) < 1e-9

    mf.smearing_method = 'gauss'
    occ = mf.get_occ(mo_energy_kpts)
    assert abs(mf.entropy - 0.9554526863670467/2) < 1e-9
    
    # fit two spin nelec
    pbc_hp.smearing_(mf, 0.1, 'fermi', fit_spin=True, tol=1e-14)
    occ = mf.get_occ(mo_energy_kpts)
    assert abs(mf.entropy - 3.090861414696259) < 1e-9

    mf.smearing_method = 'gauss'
    occ = mf.get_occ(mo_energy_kpts)
    assert abs(mf.entropy - 0.47667143693189784) < 1e-9

def test_rhf_smearing():
    import numpy as np
    from pyscf import lib
    import pyscf.pbc.gto as pbcgto
    import pyscf.pbc.scf as pscf
    from libdmet.routine import pbc_helper as pbc_hp
    cell = pbcgto.Cell()
    cell.atom = '''
    He 0 0 1
    He 1 0 1
    '''
    cell.basis = [[0, [1., 1.]], [0, [0.5, 1]]]
    cell.a = np.eye(3) * 3
    cell.mesh = [10] * 3
    cell.verbose = 5
    cell.build()
    nao = cell.nao_nr()
    
    mf = pscf.RHF(cell)
    pscf.addons.smearing_(mf, 0.1, 'fermi')
    mo_energy = np.arange(nao)*.2+np.cos(.5)*.1
    mf.get_occ(mo_energy)
    assert abs(mf.entropy - 3.0922723199786408) < 1e-9

    mf.smearing_method = 'gauss'
    occ = mf.get_occ(mo_energy)
    assert abs(mf.entropy - 0.4152467504725415) <1e-9

    mf.kernel()
    assert abs(mf.entropy) < 1e-15

def test_uhf_smearing():
    import numpy as np
    from pyscf import lib
    import pyscf.pbc.gto as pbcgto
    import pyscf.pbc.scf as pscf
    from libdmet.routine import pbc_helper as pbc_hp
    cell = pbcgto.Cell()
    cell.atom = '''
    He 0 0 1
    He 1 0 1
    '''
    cell.basis = [[0, [1., 1.]], [0, [0.5, 1]]]
    cell.a = np.eye(3) * 3
    cell.mesh = [10] * 3
    cell.verbose = 5
    cell.build()
    nao = cell.nao_nr()
    
    mf = pscf.UHF(cell)
    pscf.addons.smearing_(mf, 0.1, 'fermi')
    mo_energy = np.arange(nao)*.2+np.cos(.5)*.1
    mo_energy = np.array([mo_energy, mo_energy+np.cos(mo_energy)*.02])
    mf.get_occ(mo_energy)
    assert abs(mf.entropy - 3.1007387905421022) < 1e-9

    mf.smearing_method = 'gauss'
    occ = mf.get_occ(mo_energy)
    assert abs(mf.entropy - 0.42189309944541731) < 1e-9

def test_kukspu():
    from pyscf.pbc import gto
    from libdmet.system import lattice
    import libdmet.routine.pbc_helper as pbc_hp
    
    cell = gto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 7
    cell.build()
    kmesh = [2, 2, 2]
    kpts = cell.make_kpts(kmesh, wrap_around=True) 
    Lat = lattice.Lattice(cell, kmesh)
    U_idx = ["1 C 2p"]
    U_val = [5.0]
    
    mf = pbc_hp.KUKSpU(cell, kpts, U_idx=U_idx, U_val=U_val)
    mf.kernel()
    Lat.analyze(mf)
    
    mf_r = pbc_hp.KRKSpU(cell, kpts, U_idx=U_idx, U_val=U_val)
    mf_r.kernel()
    Lat.analyze(mf)
    assert abs(mf_r.e_tot - mf.e_tot) < 1e-7
    
    # with kpoints symmetry
    try:
        cell.space_group_symmetry=True
        cell.build()
        kpts = cell.make_kpts(kmesh, wrap_around=True, 
                              space_group_symmetry=True, time_reversal_symmetry=True)
        mf = pbc_hp.KUKSpU(cell, kpts, U_idx=U_idx, U_val=U_val)
        mf.conv_tol = 1e-10
        mf.kernel()
        assert abs(mf_r.e_tot - mf.e_tot) < 1e-7
    except TypeError:
        pass
    
    U_idx = ["1 C 2s", [0, 1]]
    U_val = [2.0, 1.0]
    mf = pbc_hp.KUKSpU(cell, kpts, U_idx=U_idx, U_val=U_val)
    

def test_ksymm():
    try:
        import os
        from pyscf.pbc.lib import chkfile
        from pyscf.pbc import scf, gto, df

        from libdmet.system import lattice
        from libdmet.routine import pbc_helper as pbc_hp
        from libdmet.utils import logger as log

        log.verbose = "DEBUG1"
        np.set_printoptions(4, linewidth=1000, suppress=False)
        
        cell = gto.Cell()
        cell.a = ''' 10.0    0.0     0.0
                     0.0     10.0    0.0
                     0.0     0.0     3.0 '''
        cell.atom = ''' H1 5.0      5.0      0.75
                        H2 5.0      5.0      2.25 '''
        cell.basis = {'H1': '321G', 'H2': 'sto3g'}
        cell.verbose = 4
        cell.precision = 1e-11
        cell.space_group_symmetry=True
        cell.build(unit='Angstrom')

        kmesh = [1, 1, 5]
        Lat = lattice.Lattice(cell, kmesh)
        kpts = Lat.kpts
        nao = Lat.nao
        nkpts = Lat.nkpts
        
        kpts_symm = cell.make_kpts(kmesh, with_gamma_point=True, wrap_around=True, \
                space_group_symmetry=True,time_reversal_symmetry=True)

        gdf_fname = 'gdf_ints.h5'
        gdf = df.GDF(cell, kpts)
        gdf._cderi_to_save = gdf_fname
        #if not os.path.isfile(gdf_fname):
        if True:
            gdf.build()
        
        chkfname = 'hchain.chk'
        #if os.path.isfile(chkfname):
        if False:
            kmf = scf.KRHF(cell, kpts_symm, exxdiv=None)
            kmf.with_df = gdf
            kmf.with_df._cderi = gdf_fname
            kmf.conv_tol = 1e-11
            data = chkfile.load(chkfname, 'scf')
            kmf.__dict__.update(data)
        else:
            kmf = scf.KRHF(cell, kpts_symm, exxdiv=None)
            kmf.with_df = gdf
            kmf.with_df._cderi = gdf_fname
            kmf.conv_tol = 1e-11
            kmf.chkfile = chkfname
            kmf.kernel()
        
        e_ref = kmf.e_tot
        kmf = pbc_hp.kmf_symm_(kmf)
        kmf.chkfile = "hchain2.chk"
        e = kmf.kernel()
        assert abs(e - e_ref) < 1e-10
        
        dm0 = kmf.make_rdm1()
        dm0 = np.asarray((dm0, dm0)) * 0.5
        H_A = [0, 1]
        H_B = [2]
        dm0[0, :, H_A, H_A] *= 2.0
        dm0[0, :, H_B, H_B]  = 0.0
        dm0[1, :, H_A, H_A]  = 0.0
        dm0[1, :, H_B, H_B] *= 2.0
        dm0 = dm0[:, kpts_symm.ibz2bz]

        kmf = scf.KUHF(cell, kpts_symm, exxdiv=None)
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf.conv_tol = 1e-11
        e_ref = kmf.kernel(dm0=dm0)
        
        kmf = pbc_hp.kmf_symm_(kmf)
        dm0 = kmf.make_rdm1()
        e = kmf.kernel(dm0=dm0)
        assert abs(e - e_ref) < 1e-10

    except TypeError:
        pass

if __name__ == "__main__":
    test_pdft_lo_xc(xc="b3lyp")
    test_ksymm()
    test_krhf_smearing()
    test_kuhf_smearing()
    test_rhf_smearing()
    test_uhf_smearing()
    
    test_ph_trans_ab_initio_frozen_core_dft()
    test_ph_trans_ab_initio_frozen_core()
    test_ph_trans_ab_initio()
    test_ph_trans_mol_frozen_core()
    test_ph_trans_mol()
    test_ph_trans_hchain()
    test_ph_trans_gamma_random()
    test_ph_trans_gamma_uhf()

    test_ph_trans_fci()
    test_kukspu()
    test_ph_integral()
    test_KGHF_KRHF()
