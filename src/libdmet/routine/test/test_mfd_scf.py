#! /usr/bin/env python

"""
Test for KRHF and KUHF for lattice models.
"""

import numpy as np
import scipy.linalg as la
import pytest

np.set_printoptions(3, linewidth=1000, suppress=True)

@pytest.mark.parametrize(
    "beta",
    [np.inf, 10.0],
)
def test_assignocc(beta):
    from libdmet.routine.mfd import assignocc
    ew = np.array([[[-1.0, 1.0, 4.0, 5.0],
                    [-2.0, 0.0, 2.0, 5.0],
                    [-1.0, 1.0, 4.0, 6.0]]])
    nelec = 6
    mu0 = 0.0
    ewocc, mu, nerr = assignocc(ew, nelec, beta, mu0)
    if beta == np.inf:
        ewocc_ref = np.zeros_like(ew)
        ewocc_ref[:, :, :2] = 1.0
        assert np.allclose(ewocc, ewocc_ref)

    ew = np.array((ew[0], ew[0]))
    nelec = (6, 6)
    mu0 = 0.0
    ewocc, mu, nerr = assignocc(ew, nelec, beta, mu0)
    if beta == np.inf:
        ewocc_ref = np.zeros_like(ew)
        ewocc_ref[:, :, :2] = 1.0
        assert np.allclose(ewocc, ewocc_ref)

    print ("ew")
    print (ew)
    nelec = 12
    Sz = 2
    mu0 = 0.0
    ewocc, mu, nerr = assignocc(ew, nelec, beta, mu0, Sz=Sz)
    print (ewocc)
    print (mu)
    print (nerr)
    assert abs(ewocc[0].sum() - (nelec + Sz) // 2) < 1e-10
    assert abs(ewocc[1].sum() - (nelec - Sz) // 2) < 1e-10

    ew[1, 1, 1] = -0.3
    ew[1, 0, 1] = -1.0

    print ("ew")
    print (ew)
    nelec = 12
    Sz = 2
    mu0 = (0.0, 0.0)
    ewocc, mu, nerr = assignocc(ew, nelec, beta, mu0, Sz=Sz)
    print (ewocc)
    print (mu)
    print (nerr)
    assert abs(ewocc[0].sum() - (nelec + Sz) // 2) < 1e-10
    assert abs(ewocc[1].sum() - (nelec - Sz) // 2) < 1e-10

def test_1d_hubbard_KRHF():
    """
    1D Hubbard KRHF
    """
    from libdmet.utils import logger as log
    from libdmet.utils.misc import max_abs
    from libdmet.system import lattice
    from libdmet.dmet import Hubbard as dmet
    log.verbose = "DEBUG2"

    # Lattice settings
    LatSize = [10]
    ImpSize = [2]
    Lat = dmet.ChainLattice(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites
    nkpts = Lat.nkpts
    natm = np.prod(ImpSize)

    # Hamiltonian
    U = 4.0
    Filling = 1.0 / 2.0
    ntotal = Filling * np.prod(LatSize)
    if ntotal - np.round(ntotal) > 1e-5:
        ntotal = int(np.round(ntotal))
        log.warning("rounded total number of electrons to integer %d", ntotal)
        Filling=float(ntotal) / np.prod(LatSize)

    Ham = dmet.Ham(Lat, U)
    restricted = True
    bogoliubov = False
    use_hcore_as_emb_ham = True
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

    H1_k = Lat.getH1(kspace=True)
    H2_loc = Lat.getH2(kspace=False)

    vcor = dmet.PMInitGuess(ImpSize, U, Filling)
    vcor_mat = vcor.get()
    vcor_mat *= 0.0
    vcor.assign(vcor_mat)

    # make AFM guess dm0
    rho, mu, ires = dmet.HartreeFock(Lat, vcor, Filling, mu0=None, beta=np.inf,
                                     ires=True, scf=True)
    log.result("\n1D Hubbard E per site: %s \n", (ires["E"] / natm))
    assert max_abs(ires["E"] / natm - -0.294427190999916) < 1e-8

def test_2d_hubbard_KUHF_gamma():
    """
    2D Hubbard KUHF, Gamma point.
    """
    from libdmet.utils import logger as log
    from libdmet.utils.misc import max_abs
    from libdmet.system import lattice
    from libdmet.dmet import Hubbard as dmet
    log.verbose = "DEBUG2"

    # Lattice settings
    LatSize = [2, 2]
    ImpSize = [2, 2]
    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites
    nkpts = Lat.nkpts
    natm = np.prod(ImpSize)

    # Hamiltonian
    U = 4.0
    Filling = 1.0 / 2.0
    ntotal = Filling * np.prod(LatSize)
    if ntotal - np.round(ntotal) > 1e-5:
        ntotal = int(np.round(ntotal))
        log.warning("rounded total number of electrons to integer %d", ntotal)
        Filling=float(ntotal) / np.prod(LatSize)

    Ham = dmet.Ham(Lat, U)
    restricted = False
    bogoliubov = False
    use_hcore_as_emb_ham = True
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

    H1_k = Lat.getH1(kspace=True)
    H2_loc = Lat.getH2(kspace=False)

    vcor = dmet.AFInitGuess(ImpSize, U, Filling)
    vcor_mat = vcor.get()
    vcor_mat *= 0.0
    vcor.assign(vcor_mat)

    # make AFM guess dm0
    dm0_a = np.zeros((nao, nao))
    dm0_b = np.zeros((nao, nao))
    dm0_a[range(nao), range(nao)] = Filling
    dm0_b[range(nao), range(nao)] = Filling
    subA, subB = lattice.BipartiteSquare(ImpSize)
    dm0_a[subA, subA] -= 0.5
    dm0_a[subB, subB] += 0.5
    dm0_b[subA, subA] += 0.5
    dm0_b[subB, subB] -= 0.5
    dm0 = np.zeros((2, nkpts, nao, nao))
    dm0[0] = dm0_a
    dm0[1] = dm0_b

    rho, mu, ires = dmet.HartreeFock(Lat, vcor, Filling, mu0=None, beta=np.inf, \
            ires=True, scf=True, dm0=dm0, conv_tol=1e-9)
    log.result("\n2D Hubbard E per site: %s \n", (ires["E"] / natm))

    norb = nao * nkpts
    nelec = int(np.round(norb * 2 * Filling))
    # 0 1
    # 2 3
    h1 = np.array([[0.0, -1.0, -1.0, 0.0],
                   [-1.0, 0.0, 0.0, -1.0],
                   [-1.0, 0.0, 0.0, -1.0],
                   [0.0, -1.0, -1.0, 0.0]])
    h2 = np.zeros((norb,)*4)
    for i in range(norb):
        h2[i, i, i, i] = U

    from pyscf import scf, gto, ao2mo
    mol = gto.M()
    mol.nelectron = nelec
    mol.verbose = 4
    mf = scf.UHF(mol)

    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(norb)
    mf._eri = ao2mo.restore(4, h2, norb)

    dm0_a = np.zeros((norb, norb))
    dm0_a[1, 1] = 1.0
    dm0_a[2, 2] = 1.0

    dm0_b = np.zeros((norb, norb))
    dm0_b[0, 0] = 1.0
    dm0_b[3, 3] = 1.0

    dm0 = np.asarray((dm0_a, dm0_b))
    e_ref = mf.kernel(dm0=dm0)
    e_ref_per_site = e_ref / float(norb)
    rdm1_ref = mf.make_rdm1()
    assert max_abs(ires["E"] / natm - e_ref_per_site) < 1e-8

def test_2d_hubbard_KUHF():
    """
    2D Hubbard KUHF with kpoint sampling.
    """
    from libdmet.utils import logger as log
    from libdmet.utils.misc import max_abs
    from libdmet.system import lattice
    from libdmet.dmet import Hubbard as dmet
    log.verbose = "DEBUG2"

    # Lattice settings
    LatSize = [6, 12]
    ImpSize = [2, 4]
    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites
    nkpts = Lat.nkpts
    natm = np.prod(ImpSize)

    # Hamiltonian
    U = 6.0
    Filling = 1.0 / 2.0
    ntotal = Filling * np.prod(LatSize)
    if ntotal - np.round(ntotal) > 1e-5:
        ntotal = int(np.round(ntotal))
        log.warning("rounded total number of electrons to integer %d", ntotal)
        Filling=float(ntotal) / np.prod(LatSize)

    Ham = dmet.Ham(Lat, U)
    restricted = False
    bogoliubov = False
    use_hcore_as_emb_ham = True
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

    H1_k = Lat.getH1(kspace=True)
    H2_loc = Lat.getH2(kspace=False)

    vcor = dmet.AFInitGuess(ImpSize, U, Filling)
    vcor_mat = vcor.get()
    vcor_mat *= 0.0
    vcor.assign(vcor_mat)

    # make AFM guess dm0
    dm0_a = np.zeros((nao, nao))
    dm0_b = np.zeros((nao, nao))
    dm0_a[range(nao), range(nao)] = Filling
    dm0_b[range(nao), range(nao)] = Filling
    subA, subB = lattice.BipartiteSquare(ImpSize)
    dm0_a[subA, subA] -= 0.5
    dm0_a[subB, subB] += 0.5
    dm0_b[subA, subA] += 0.5
    dm0_b[subB, subB] -= 0.5
    dm0 = np.zeros((2, nkpts, nao, nao))
    dm0[0] = dm0_a
    dm0[1] = dm0_b

    rho, mu, ires = dmet.HartreeFock(Lat, vcor, Filling, mu0=None, beta=np.inf, \
            ires=True, scf=True, dm0=dm0, conv_tol=1e-9)
    log.result("\n2D Hubbard E per site (KUHF): %s \n", (ires["E"] / natm))

    norb = nkpts * nao
    nelec = int(np.round(norb * 2 * Filling))
    h1 = Lat.expand(Lat.k2R(H1_k))
    h2 = np.zeros((norb,)*4)
    for i in range(norb):
        h2[i, i, i, i] = U

    from pyscf import scf, gto, ao2mo
    mol = gto.M()
    mol.nelectron = nelec
    mol.verbose = 4
    mf = scf.UHF(mol)

    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(norb)
    mf._eri = ao2mo.restore(4, h2, norb)

    dm0 = Lat.expand(Lat.k2R(dm0))
    e_ref = mf.kernel(dm0=dm0)
    e_ref_per_site = e_ref / float(norb)
    rdm1_ref = mf.make_rdm1()

    log.result("\n2D Hubbard E per site (UHF): %s \n", e_ref_per_site)
    assert max_abs(ires["E"] / natm - e_ref_per_site) < 1e-8

def test_2d_hubbard_KUHF_AFM_cell():
    """
    2D Hubbard KUHF with kpoint sampling, AFM cell.
    """
    from libdmet.utils import logger as log
    from libdmet.utils.misc import max_abs
    from libdmet.system import lattice
    from libdmet.dmet import Hubbard as dmet
    log.verbose = "DEBUG2"

    # Lattice settings
    LatSize = [3, 8]
    ImpSize = [1, 2]
    Lat = dmet.SquareAFM(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites
    nkpts = Lat.nkpts
    natm = np.prod(ImpSize) * 2

    # Hamiltonian
    U = 2.0
    Filling = 1.0 / 2.0
    ntotal = Filling * np.prod(LatSize) * 2.0
    if ntotal - np.round(ntotal) > 1e-5:
        ntotal = int(np.round(ntotal))
        log.warning("rounded total number of electrons to integer %d", ntotal)
        Filling=float(ntotal) / (np.prod(LatSize) * 2.0)

    Ham = dmet.Ham(Lat, U)
    restricted = False
    bogoliubov = False
    use_hcore_as_emb_ham = True
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

    H1_k = Lat.getH1(kspace=True)
    H2_loc = Lat.getH2(kspace=False)
    subA = [0, 2]
    subB = [1, 3]

    vcor = dmet.AFInitGuess(ImpSize, U, Filling, subA=subA, subB=subB)
    vcor_mat = vcor.get()
    vcor_mat *= 0.0
    vcor.assign(vcor_mat)

    # make AFM guess dm0
    dm0_a = np.zeros((nao, nao))
    dm0_b = np.zeros((nao, nao))
    dm0_a[range(nao), range(nao)] = Filling
    dm0_b[range(nao), range(nao)] = Filling

    dm0_a[subA, subA] -= 0.5
    dm0_a[subB, subB] += 0.5
    dm0_b[subA, subA] += 0.5
    dm0_b[subB, subB] -= 0.5
    dm0 = np.zeros((2, nkpts, nao, nao))
    dm0[0] = dm0_a
    dm0[1] = dm0_b

    rho, mu, ires = dmet.HartreeFock(Lat, vcor, Filling, mu0=None, beta=np.inf, \
            ires=True, scf=True, dm0=dm0, conv_tol=1e-9)
    log.result("\n2D Hubbard E per site (KUHF): %s \n", (ires["E"] / natm))

    norb = nkpts * nao
    nelec = int(np.round(norb * 2 * Filling))
    h1 = Lat.expand(Lat.k2R(H1_k))
    h2 = np.zeros((norb,)*4)
    for i in range(norb):
        h2[i, i, i, i] = U

    from pyscf import scf, gto, ao2mo
    mol = gto.M()
    mol.nelectron = nelec
    mol.verbose = 4
    mf = scf.UHF(mol)

    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(norb)
    mf._eri = ao2mo.restore(4, h2, norb)

    dm0 = Lat.expand(Lat.k2R(dm0))
    e_ref = mf.kernel(dm0=dm0)
    e_ref_per_site = e_ref / float(norb)
    rdm1_ref = mf.make_rdm1()

    log.result("\n2D Hubbard E per site (UHF): %s \n", e_ref_per_site)
    assert max_abs(ires["E"] / natm - e_ref_per_site) < 1e-8

def test_2d_hubbard_KGHF():
    """
    2D Hubbard KGHF (no p-h transform), compared with KUHF value.
    """
    from libdmet.utils import logger as log
    from libdmet.utils.misc import max_abs
    from libdmet.system import lattice
    from libdmet.dmet import Hubbard as dmet
    from libdmet.system.hamiltonian import HamNonInt
    from libdmet.routine import mfd
    log.verbose = "DEBUG2"

    # Lattice settings
    LatSize = [6, 12]
    ImpSize = [2, 4]
    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites
    nkpts = Lat.nkpts
    natm = np.prod(ImpSize)

    # Hamiltonian
    U = 6.0
    Filling = 1.0 / 2.0
    ntotal = Filling * np.prod(LatSize)
    if ntotal - np.round(ntotal) > 1e-5:
        ntotal = int(np.round(ntotal))
        log.warning("rounded total number of electrons to integer %d", ntotal)
        Filling=float(ntotal) / np.prod(LatSize)

    Ham = dmet.Ham(Lat, U)
    restricted = False
    bogoliubov = True
    use_hcore_as_emb_ham = True
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

    H1_k = Lat.getH1(kspace=True)
    H2_loc = Lat.getH2(kspace=False)

    GH1_k = np.zeros((3, nkpts, nao, nao), dtype=np.complex128)
    GH1_k[0] = GH1_k[1] = H1_k
    GH2_loc = np.asarray((H2_loc,)*3)
    GH0 = 0.0

    Ham_sl = HamNonInt(Lat, GH1_k, GH2_loc, Fock=None, ImpJK=None, \
                kspace_input=True,  H0=GH0, spin_dim_H2=3)
    Lat.setHam(Ham_sl)

    vcor = dmet.AFInitGuess(ImpSize, U, Filling, bogoliubov=bogoliubov)
    vcor_mat = vcor.get()
    vcor_mat *= 0.0
    vcor.assign(vcor_mat)

    # make AFM guess dm0
    dm0_a = np.zeros((nao, nao))
    dm0_b = np.zeros((nao, nao))
    dm0_a[range(nao), range(nao)] = Filling
    dm0_b[range(nao), range(nao)] = Filling
    subA, subB = lattice.BipartiteSquare(ImpSize)
    dm0_a[subA, subA] -= 0.5
    dm0_a[subB, subB] += 0.5
    dm0_b[subA, subA] += 0.5
    dm0_b[subB, subB] -= 0.5
    dm0 = np.zeros((3, nkpts, nao, nao))
    dm0[0] = dm0_a
    dm0[1] = dm0_b
    dm0 = mfd.H_k2GH_k(dm0)

    Mu = 2.0
    GRho, n, E = mfd.GHF(Lat, vcor, restricted, mu=Mu, beta=np.inf, fix_mu=False, \
            ires=False, scf=True, dm0=dm0, newton_ah=False, max_cycle=50)
    log.result("\n2D Hubbard E per site (KGHF): %s \n", (E / natm))
    assert max_abs(E / natm - -0.592623515081266) < 1e-8

def test_2d_hubbard_KGHF_ph():
    """
    2D Hubbard KGHF (with partial p-h transform), HD = 0.0,
    compared with KUHF value.
    """
    from libdmet.utils import logger as log
    from libdmet.utils.misc import max_abs
    from libdmet.system import lattice
    from libdmet.dmet import Hubbard as dmet
    from libdmet.system.hamiltonian import HamNonInt
    from libdmet.routine import mfd
    from libdmet.routine.spinless import \
            transform_H1_k, transform_H2_local
    log.verbose = "DEBUG2"

    # Lattice settings
    LatSize = [6, 12]
    ImpSize = [2, 4]
    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites
    nkpts = Lat.nkpts
    natm = np.prod(ImpSize)

    # Hamiltonian
    U = 6.0
    Filling = 1.0 / 2.0
    ntotal = Filling * np.prod(LatSize)
    if ntotal - np.round(ntotal) > 1e-5:
        ntotal = int(np.round(ntotal))
        log.warning("rounded total number of electrons to integer %d", ntotal)
        Filling=float(ntotal) / np.prod(LatSize)

    Ham = dmet.Ham(Lat, U)
    restricted = False
    bogoliubov = False
    use_hcore_as_emb_ham = True
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

    H1_k = Lat.getH1(kspace=True)
    H2_loc = Lat.getH2(kspace=False)

    GH1_k, GH0 = transform_H1_k(H1_k)
    GH2_loc, GH1_from_H2_loc, GH0_from_H2 = transform_H2_local(H2_loc)

    GH1_k = mfd.add_H1_loc_to_k(GH1_from_H2_loc, GH1_k)
    GH0 += GH0_from_H2

    Ham_sl = HamNonInt(Lat, GH1_k, GH2_loc, Fock=None, ImpJK=None, \
                kspace_input=True,  H0=GH0, spin_dim_H2=3)
    Lat.setHam(Ham_sl)

    vcor = dmet.AFInitGuess(ImpSize, U, Filling, bogoliubov=True)
    vcor_mat = vcor.get()
    vcor_mat *= 0.0
    vcor.assign(vcor_mat)

    # make AFM guess dm0
    dm0_a = np.zeros((nao, nao))
    dm0_b = np.zeros((nao, nao))
    dm0_a[range(nao), range(nao)] = Filling
    dm0_b[range(nao), range(nao)] = Filling
    subA, subB = lattice.BipartiteSquare(ImpSize)
    dm0_a[subA, subA] -= 0.5
    dm0_a[subB, subB] += 0.5
    dm0_b[subA, subA] += 0.5
    dm0_b[subB, subB] -= 0.5
    dm0 = np.zeros((3, nkpts, nao, nao))
    dm0[0] = dm0_a
    dm0[1] = np.eye(nao) - dm0_b
    dm0 = mfd.H_k2GH_k(dm0)

    Mu = U * 0.5
    GRho, n, E = mfd.GHF(Lat, vcor, restricted, mu=Mu, beta=np.inf, fix_mu=False, \
            ires=False, scf=True, dm0=dm0, newton_ah=False, max_cycle=50)
    log.result("\n2D Hubbard E per site (KGHF with ph): %s \n", (E / natm))
    assert max_abs(E / natm - -0.592623515081266) < 1e-8

def test_random_ham_KGHF_ph():
    from libdmet.system.integral import Integral
    from libdmet.solver import scf
    from libdmet.routine.bcs_helper import extractRdm
    from libdmet.routine.spinless_helper import (transform_spinless_mol,
                                                 Ham_compact2uhf)
    from libdmet.utils.misc import max_abs
    from libdmet.dmet import Hubbard as dmet
    from libdmet.system import lattice
    from libdmet.system.hamiltonian import HamNonInt
    from libdmet.routine import mfd
    from libdmet.routine.spinless import (transform_H1_k, transform_H2_local)
    from libdmet.utils import logger as log

    np.random.seed(5)
    # ********************************
    # HFB vs UIHF(all alpha) vs GHF
    # D != 0
    # ********************************
    norb = 4
    nelec_a = 2
    nelec_b = 2

    h1_a = np.random.random((norb, norb))
    h1_a = h1_a + h1_a.conj().T

    h1_b = np.random.random((norb, norb))
    h1_b = h1_b + h1_b.conj().T

    h1 = np.asarray((h1_a, h1_b))

    h2_aa = np.random.random((norb, norb, norb, norb))
    h2_aa = h2_aa + h2_aa.transpose(1,0,2,3)
    h2_aa = h2_aa + h2_aa.transpose(0,1,3,2)
    h2_aa = h2_aa + h2_aa.transpose(2,3,0,1)

    #h2_bb = np.random.random((norb, norb, norb, norb))
    #h2_bb = h2_bb + h2_bb.transpose(1,0,2,3)
    #h2_bb = h2_bb + h2_bb.transpose(0,1,3,2)
    #h2_bb = h2_bb + h2_bb.transpose(2,3,0,1)
    #
    #h2_ab = np.random.random((norb, norb, norb, norb))
    #h2_ab = h2_ab + h2_ab.transpose(1,0,2,3)
    #h2_ab = h2_ab + h2_ab.transpose(0,1,3,2)
    ## 4-fold is enough
    ##h2_ab = h2_ab + h2_ab.transpose(2,3,0,1)
    h2_bb = h2_ab = h2_aa

    h2 = np.asarray((h2_aa, h2_bb, h2_ab))

    Mu = 4.0
    D = np.random.random((norb, norb))
    # D does not have to be symmetric
    #D = D + D.conj().T

    newton_ah = False
    do_diis = True
    InitGuess = None

    # HFB
    log.note("HFB")
    Ham_b = Integral(norb, restricted=False, bogoliubov=True, H0=0.0,
                     H1={"cd": h1, "cc": np.asarray((D,))},
                     H2={"ccdd": h2, "cccd": None, "cccc": None})

    myscf_b = scf.SCF(newton_ah=newton_ah)
    myscf_b.set_system(None, 0, True, False) # nelec, spin, bogo, res
    myscf_b.set_integral(Ham_b)
    e_b, rdm1_b = myscf_b.HFB(Mu, tol=1e-10, do_diis=do_diis,
            InitGuess=InitGuess)
    rdm1_hfb_a, rdm1_hfb_b, rdm1_hfb_ba = extractRdm(rdm1_b)

    log.result("HFB MO energy:\n%s", myscf_b.mf.mo_energy)
    log.result("trace of normal rdm1: %s",
               rdm1_hfb_a.trace() + rdm1_hfb_b.trace())
    log.result("E (HFB): %s", e_b)

    # GHF
    log.note("Spinless HF use GIHF")
    Ham_sl = transform_spinless_mol(Ham_b.H1["cd"], Ham_b.H1["cc"], Ham_b.H2["ccdd"])

    myscf_sl = scf.SCF(newton_ah=newton_ah)
    myscf_sl.set_system(norb, 0, False, False) # nelec, spin, bogo, res
    myscf_sl.set_integral(Ham_sl)
    e_sl, rdm1_sl = myscf_sl.GHF(Mu=Mu, tol=1e-10, do_diis=do_diis,
                                 InitGuess=InitGuess)
    rdm1_sl_a, rdm1_sl_b, rdm1_sl_ba = extractRdm(rdm1_sl)

    log.result("GHF MO energy:\n%s", myscf_sl.mf.mo_energy)
    log.result("trace of normal rdm1: %s", rdm1_sl_a.trace() + rdm1_sl_b.trace())
    log.result("E (GHF): %s", e_sl)

    log.result("Diff GHF - HFB: %s", e_sl - e_b)
    log.eassert(abs(e_sl - e_b) < 1e-8,
                "GHF and HFB should give the same energy")


    # UIHF
    log.note("Spinless HF use UIHF")
    dm0 = np.zeros((2, norb*2, norb*2))
    dm0[0] = np.eye(norb*2) * 0.5
    Ham_slu = Ham_compact2uhf(Ham_sl, eri_spin=3)

    myscf_slu = scf.SCF(newton_ah=newton_ah)
    myscf_slu.set_system(norb, norb, False, False) # nelec, spin, bogo, res
    myscf_slu.set_integral(Ham_slu)
    e_slu, rdm1_slu = myscf_slu.HF(Mu=Mu, tol=1e-10, do_diis=do_diis,\
            InitGuess=dm0)
    rdm1_slu_a, rdm1_slu_b, rdm1_slu_ba = extractRdm(rdm1_slu)

    log.result("E (GHF from UIHF): %s", e_slu)
    log.eassert(abs(e_slu - e_b) < 1e-8, \
            "GHF from UIHF and HFB should give the same energy")

    # KGHF
    # Lattice settings
    LatSize = [2, 2]
    ImpSize = [2, 2]
    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites
    nkpts = Lat.nkpts
    natm = np.prod(ImpSize)
    restricted = False

    H1_k = np.asarray((h1_a[None], h1_b[None], D[None]))
    H2_loc = h2_aa # assume h_aa = h_bb = h_ab

    GH1_k, GH0 = transform_H1_k(H1_k)
    GH2_loc, GH1_from_H2_loc, GH0_from_H2 = transform_H2_local(H2_loc)
    GH1_k = mfd.add_H1_loc_to_k(GH1_from_H2_loc, GH1_k)
    GH0 += GH0_from_H2

    Ham_sl = HamNonInt(Lat, GH1_k, GH2_loc, Fock=None, ImpJK=None, \
                kspace_input=True,  H0=GH0, spin_dim_H2=3)
    Lat.setHam(Ham_sl)

    vcor = dmet.AFInitGuess(ImpSize, 0.0, 0.5, bogoliubov=True)
    vcor_mat = vcor.get()
    vcor_mat *= 0.0
    vcor.assign(vcor_mat)

    dm0 = dm0[0][None]
    GRho, n, E = mfd.GHF(Lat, vcor, restricted, mu=Mu, beta=np.inf, \
            fix_mu=False, ires=False, scf=True, dm0=dm0, newton_ah=False, \
            conv_tol=1e-10, max_cycle=50)
    log.result("E (KGHF): %s", E)
    log.eassert(abs(E - e_b) < 1e-8,
                "KGHF and HFB should give the same energy")

def test_2d_hubbard_KGHF_SC():
    import libdmet.dmet.Hubbard as dmet
    from libdmet.dmet import Hubbard
    from libdmet.routine import spinless
    from libdmet.routine.spinless import \
            transform_H1_k, transform_H2_local, GH2H, H2GH
    from libdmet.system import lattice
    from libdmet.routine import mfd
    from libdmet.system.integral import Integral
    from libdmet.system.hamiltonian import HamNonInt
    from libdmet.solver import scf
    from libdmet.utils.misc import max_abs
    from libdmet.utils import logger as log

    np.set_printoptions(3, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"

    # Lattice settings
    LatSize = [6, 4]
    ImpSize = [2, 2]
    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites
    nso = nao * 2
    nkpts = Lat.nkpts

    # Hamiltonian
    U = 4.0
    Filling = 1.0 / 2.0
    ntotal = Filling * np.prod(LatSize)
    if ntotal - np.round(ntotal) > 1e-5:
        ntotal = int(np.round(ntotal))
        log.warning("rounded total number of electrons to integer %d", ntotal)
        Filling=float(ntotal) / np.prod(LatSize)
    Ham = dmet.Ham(Lat, U)
    restricted = False
    bogoliubov = True
    use_hcore_as_emb_ham = True
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

    H1_k = Lat.getH1(kspace=True)
    H2_loc = Lat.getH2(kspace=False)

    # p-h transformed hamiltonian.
    GH1_k, GH0 = transform_H1_k(H1_k)
    GH2_loc, GH1_from_H2_loc, GH0_from_H2 = transform_H2_local(H2_loc)
    GH1_k = mfd.add_H1_loc_to_k(GH1_from_H2_loc, GH1_k)
    GH0 += GH0_from_H2

    vcor = dmet.AFInitGuess(ImpSize, U, Filling, bogoliubov=True, rand=0.1)
    vcor_mat = vcor.get()
    vcor_mat[1] = -vcor_mat[1].T
    vcor_mat[2] = vcor_mat[2] + vcor_mat[2].conj().T
    vcor.assign(vcor_mat)

    subA, subB = Hubbard.BipartiteSquare(ImpSize)
    dm0_a = np.zeros((nao, nao), dtype=np.complex128)
    dm0_b = np.zeros((nao, nao), dtype=np.complex128)
    dm0_a[subB, subB] = 1.0
    dm0_b[subA, subA] = 1.0
    dm0_b = np.eye(nao) - dm0_b
    dm0_ab = la.block_diag(dm0_a, dm0_b)
    dm0 = np.zeros((nkpts, nso, nso), dtype=np.complex128)
    dm0[:] = dm0_ab

    Ham_sl = HamNonInt(Lat, GH1_k, GH2_loc, Fock=None, ImpJK=None, \
                kspace_input=True,  H0=GH0, spin_dim_H2=3)
    Lat.setHam(Ham_sl)
    Mu = U * 0.5
    GRho, n, E = mfd.GHF(Lat, vcor, restricted, mu=Mu, beta=np.inf, fix_mu=False, \
            ires=False, scf=True, dm0=dm0, newton_ah=False, conv_tol=1e-10, max_cycle=50)
    log.result("KGHF E per cell: %s", E / nkpts)
    log.result("KGHF GRho:\n%s", np.asarray(spinless.extractRdm(GRho[0])))

    H1_k = GH1_k
    H1_k_with_vcor = mfd.add_H1_loc_to_k(vcor.get(), GH1_k)
    H1_R = np.zeros((3, nao*nkpts, nao*nkpts))
    for s in range(H1_R.shape[0]):
        H1_R[s] = Lat.expand(Lat.k2R(H1_k[s]))
    GH1_R = H2GH(H1_R)

    H1_R_with_vcor = np.zeros((3, nao*nkpts, nao*nkpts))
    for s in range(H1_R_with_vcor.shape[0]):
        H1_R_with_vcor[s] = Lat.expand(Lat.k2R(H1_k_with_vcor[s]))
    GH1_R_with_vcor = H2GH(H1_R_with_vcor)

    H2_R = np.zeros((3,) + (nao*nkpts,)*4)
    for k in range(nkpts):
        H2_R[:, k*nao:(k+1)*nao, k*nao:(k+1)*nao, \
                k*nao:(k+1)*nao, k*nao:(k+1)*nao] = GH2_loc
    H0_R = GH0 * nkpts

    newton_ah = False
    InitGuess = Lat.expand(Lat.k2R(dm0))
    norb = H1_R.shape[-1]
    Ham_sl_mol = Integral(norb, False, False, H0_R, {"cd":H1_R_with_vcor},
            {"ccdd":H2_R}, ovlp=None)

    myscf_sl = scf.SCF(newton_ah=newton_ah)
    myscf_sl.set_system(norb, 0, False, False) # nelec, spin, bogo, res
    myscf_sl.set_integral(Ham_sl_mol)
    e_sl, rdm1_sl = myscf_sl.GHF(Mu=Mu, tol=1e-10, do_diis=True,\
            InitGuess=InitGuess, DiisDim=8)

    Gvcor = GH1_R_with_vcor - GH1_R
    Gveff = myscf_sl.mf.get_veff()
    E1 = np.einsum('pq, qp ->', GH1_R, rdm1_sl)
    E2 = 0.5 * np.einsum('pq, qp ->', Gveff + Gvcor, rdm1_sl)
    E0 = H0_R
    E_new = E0 + E1 + E2
    log.result("E (GHF from supercell per cell: %s", E / nkpts)
    assert max_abs(E_new / nkpts - E) < 1e-8

def test_7d_eri_jk():
    """
    Test 7d eri and get_jk_7d.
    """
    from pyscf import lib, fci, ao2mo
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df

    from libdmet.system import lattice
    from libdmet.utils.misc import max_abs
    from libdmet.utils import logger as log

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
    ### SCF Mean-field calculation
    ### ************************************************************

    log.section("\nSolving SCF mean-field problem\n")

    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    #if not os.path.isfile(gdf_fname):
    if True:
        gdf.build()

    from libdmet.routine.pbc_helper import get_eri_7d, get_jk_from_eri_7d
    eri_7d = get_eri_7d(cell, gdf, kpts=None, compact=False)

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

    dm = kmf.make_rdm1()
    vj_ref, vk_ref = kmf.get_jk()
    vj, vk = get_jk_from_eri_7d(eri_7d, dm)
    diff_j = max_abs(vj - vj_ref)
    diff_k = max_abs(vk - vk_ref)
    print ("diff_j", diff_j)
    print ("diff_k", diff_k)
    assert diff_j < 1e-10
    assert diff_k < 1e-10

def test_3band_nearest_Vpd():
    """
    2D 3band Hubbard KUHF with kpoint sampling, AFM cell.
    """
    from libdmet.utils import logger as log
    from libdmet.utils.misc import max_abs
    from libdmet.system import lattice
    from libdmet.system import hamiltonian as ham
    from libdmet.dmet import Hubbard as dmet
    log.verbose = "DEBUG2"

    # half filled
    x_dop = 0.0

    # Hamiltonian
    tpd = -1.2
    tpp = -0.7
    ed = -2.0
    tpp1 = -0.1
    Ud = 8.4
    Up = 2.0
    Vpd = 2.0

    # Lattice settings
    LatSize = [3, 2]
    ImpSize = [1, 1]

    Lat = dmet.Square3BandAFM(*(LatSize+ImpSize))
    Ham = ham.Hubbard3band(Lat, Ud, Up, ed, tpd, tpp, tpp1=tpp1, Vpd=Vpd, \
            ignore_intercell=False)

    nao = nscsites = Lat.nscsites
    nkpts = Lat.nkpts
    natm = np.prod(ImpSize) * 6

    nCu_tot = np.prod(LatSize) * 2 # 2 is number of Cu site per AFM cell
    nO_tot = np.prod(LatSize) * 4
    nao_tot = nao * nkpts
    nelec_half = np.prod(LatSize) * 10 # 10 electron per cell
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
    vcor = dmet.vcor_zeros(restricted, bogoliubov, nscsites)

    # make AFM guess dm0
    polar = 0.5
    dm0_a = np.zeros((nao, nao))
    dm0_b = np.zeros((nao, nao))
    dm0_a[range(nao), range(nao)] = Filling
    dm0_b[range(nao), range(nao)] = Filling

    dm0_a = np.diag([Filling_Cu * (1.0 + polar), Filling_Cu * (1.0 - polar), \
            Filling_O, Filling_O, Filling_O, Filling_O])
    dm0_b = np.diag([Filling_Cu * (1.0 - polar), Filling_Cu * (1.0 + polar), \
            Filling_O, Filling_O, Filling_O, Filling_O])

    dm0 = np.zeros((2, nkpts, nao, nao))
    dm0[0] = dm0_a
    dm0[1] = dm0_b

    rho, mu, ires = dmet.HartreeFock(Lat, vcor, Filling, mu0=None, beta=1.0/0.1, \
            ires=True, scf=True, dm0=dm0, conv_tol=1e-9)
    log.result("\n2D Hubbard E per site (KUHF): %s \n", (Lat.kmf_lo.e_zero / natm))

    rdm1_a, rdm1_b = rho[:, 0]
    m_AFM = 0.5 * (abs(rdm1_a[0, 0] - rdm1_b[0, 0]) + abs(rdm1_a[1, 1] - rdm1_b[1, 1]))
    log.result("m_AFM = %s", m_AFM)

    E = ires["E"] / natm

    # supercell reference
    dm0 = Lat.expand(rho)[:, None]
    #dm0 = Lat.expand(Lat.k2R(dm0))[:, None]
    ImpSize = LatSize

    Lat = dmet.Square3BandAFM(*(LatSize+ImpSize))
    Ham = ham.Hubbard3band(Lat, Ud, Up, ed, tpd, tpp, tpp1=tpp1, Vpd=Vpd, \
            ignore_intercell=False)
    Ham.H2_format = 'local'
    Ham.H2 = Ham.H2[0]

    nao = nscsites = Lat.nscsites
    nkpts = Lat.nkpts
    natm = np.prod(ImpSize) * 6

    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

    vcor = dmet.vcor_zeros(restricted, bogoliubov, nscsites)

    rho, mu, ires = dmet.HartreeFock(Lat, vcor, Filling, mu0=None, beta=1.0/0.1, \
            ires=True, scf=True, dm0=dm0, conv_tol=1e-9)
    log.result("\n2D Hubbard E per site (KUHF): %s \n", (Lat.kmf_lo.e_zero / natm))

    rdm1_a, rdm1_b = rho[:, 0]
    m_AFM = 0.5 * (abs(rdm1_a[0, 0] - rdm1_b[0, 0]) + abs(rdm1_a[1, 1] - rdm1_b[1, 1]))
    log.result("m_AFM = %s", m_AFM)

    E_ref = ires["E"] / natm
    assert abs(E_ref - E) < 1e-10

def test_2d_hubbard_KUHF_from_gdf():
    """
    2D Hubbard KUHF with kpoint sampling.
    Using a manually build GDF.
    """
    from pyscf.pbc import scf, df
    from libdmet.system import lattice
    from libdmet.dmet import Hubbard as dmet
    from libdmet.routine import pbc_helper as pbc_hp
    from libdmet.utils.misc import max_abs
    from libdmet.utils import logger as log
    log.verbose = "DEBUG2"

    # Lattice settings
    LatSize = [6, 6]
    ImpSize = [2, 2]
    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites
    nkpts = Lat.nkpts
    natm = np.prod(ImpSize)

    # Hamiltonian
    U = 6.0
    Filling = 1.0 / 2.0
    ntotal = Filling * np.prod(LatSize)
    if abs(ntotal - np.round(ntotal)) > 1e-5:
        ntotal = int(np.round(ntotal))
        log.warning("rounded total number of electrons to integer %d", ntotal)
        Filling=float(ntotal) / np.prod(LatSize)

    nelec_per_cell = Filling * np.prod(ImpSize) * 2.0

    Ham = dmet.Ham(Lat, U)
    restricted = False
    bogoliubov = False
    use_hcore_as_emb_ham = True
    Lat.setHam(Ham, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

    norb = Lat.nao
    kpts = Lat.kpts
    hcore = Lat.getH1(kspace=True)
    ovlp = np.zeros_like(hcore)
    ovlp[:, range(norb), range(norb)] = 1.0
    H2 = Lat.getH2()

    cell = Lat.cell
    cell.nelectron = nelec_per_cell

    # build a gdf for kmf
    cderi_name = pbc_hp.eri_to_gdf(H2, kpts)
    gdf = df.GDF(cell, kpts)
    gdf._cderi = cderi_name
    gdf._cderi_to_save = cderi_name

    kmf = scf.KUHF(cell, kpts, exxdiv=None).density_fit()
    kmf.get_hcore = lambda *args: hcore
    kmf.get_ovlp = lambda *args: ovlp
    kmf.energy_nuc = lambda *args: 0.0
    kmf.with_df = gdf

    kmf.conv_tol = 1e-12
    kmf.max_cycle = 300

    dm0_a = np.zeros((norb, norb))
    dm0_b = np.zeros((norb, norb))
    dm0_a[range(norb), range(norb)] = Filling
    dm0_b[range(norb), range(norb)] = Filling
    subA, subB = lattice.BipartiteSquare(ImpSize)
    dm0_a[subA, subA] -= 0.5
    dm0_a[subB, subB] += 0.5
    dm0_b[subA, subA] += 0.5
    dm0_b[subB, subB] -= 0.5
    dm0 = np.zeros((2, nkpts, norb, norb))
    dm0[0] = dm0_a
    dm0[1] = dm0_b

    E = kmf.kernel(dm0=dm0)

    dm = kmf.make_rdm1()
    dm_R = Lat.k2R(dm)

    vcor = dmet.AFInitGuess(ImpSize, U, Filling)
    vcor_mat = vcor.get()
    vcor_mat *= 0.0
    vcor.assign(vcor_mat)

    rho, mu, ires = dmet.HartreeFock(Lat, vcor, Filling, mu0=None, beta=np.inf,
                                     ires=True, scf=True, dm0=dm0, conv_tol=1e-9)

    E_diff = abs(E - ires["E"])
    rdm1_diff = max_abs(dm_R - rho)
    print ("E diff", E_diff)
    assert E_diff < 1e-10
    print ("rdm1 diff", rdm1_diff)
    assert rdm1_diff < 1e-8

if __name__ == "__main__":
    test_2d_hubbard_KUHF_from_gdf()
    test_random_ham_KGHF_ph()
    test_2d_hubbard_KGHF_ph()
    test_2d_hubbard_KUHF_AFM_cell()
    test_3band_nearest_Vpd()
    test_assignocc(beta=np.inf)
    test_1d_hubbard_KRHF()
    test_2d_hubbard_KUHF_gamma()
    test_2d_hubbard_KUHF()
    test_2d_hubbard_KGHF()
    test_2d_hubbard_KGHF_SC()
    test_7d_eri_jk()
