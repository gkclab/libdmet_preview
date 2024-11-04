#! /usr/bin/env python

import numpy as np
import scipy.linalg as la
from libdmet.utils import logger as log
from libdmet.system.hamiltonian import HamNonInt

def Hubbard3band_old(lattice, Ud, Up, ed, tpd, tpp, tpp1=0.0, Vpd=0.0, \
        ignore_intercell=True, tol=1e-10):
    """
    3-band Hubbard model in electron representation:
    H = tpd + tpp + tpp' + ed + Ud + Ud + Vpd
    t and ed is in general negative, while U is positive.
    """
    from libdmet.system.lattice import Frac2Real, Real2Frac, round_to_FUC
    ncells = lattice.ncells
    nscsites = lattice.nscsites
    H1 = np.zeros((ncells, nscsites, nscsites))
    if ignore_intercell:
        H2 = np.zeros((nscsites,)*4)
    else:
        if abs(Vpd) < 1e-5:
            log.warn("Vpd %s is very small, ignore_intercell should be set to True...", Vpd)
        H2 = np.zeros((ncells,) + (nscsites,)*4)
    d_pd  = lattice.neighborDist[0]
    d_pp  = lattice.neighborDist[1]
    d_pp1 = lattice.neighborDist[2]
    log.warning("Searching neighbor within only one supercell")

    def get_vec(s1, s2):
        vec = (lattice.sites[s1] - lattice.sites[s2]) % np.diag(lattice.size)
        for i in range(vec.shape[0]):
            if vec[i] > lattice.size[i,i] * 0.5:
                vec[i] -= lattice.size[i,i]
        return vec

    # tpd and Vpd
    pd_pairs = lattice.neighbor(dis=d_pd, sitesA=range(nscsites))
    for i, j in pd_pairs:
        if lattice.names[i] == "Cu":
            vec = get_vec(j, i)
        else:
            vec = get_vec(i, j)

        if abs(vec[0] - 1.0) < tol or abs(vec[1] + 1.0) < tol:
            sign = -1.0
        elif abs(vec[1] - 1.0) < tol or abs(vec[0] + 1.0) < tol:
            sign = 1.0
        else:
            log.error("invalid p-d neighbor, vec = %s", vec)
            raise ValueError

        H1[j//nscsites, j%nscsites, i] = sign * tpd

        if ignore_intercell:
            if j // nscsites == 0:
                H2[j, j, i, i] = Vpd
        else:
            H2[j//nscsites, j%nscsites, j%nscsites, i, i] = Vpd

    # tpp
    pp_pairs = lattice.neighbor(dis=d_pp, sitesA=range(nscsites))
    for i, j in pp_pairs:
        vec = get_vec(j, i)
        if vec[0] * vec[1] > 0:
            sign = -1.0
        else:
            sign = 1.0
        H1[j//nscsites, j%nscsites, i] = sign * tpp

    # tpp'
    # ZHC FIXME should I include tpp' for oxygens between not bridged by Cu?
    Osites = [idx for (idx, name) in \
            zip(range(nscsites), lattice.names[:nscsites]) if name == "O"]
    pp1_pairs = lattice.neighbor(dis=d_pp1, sitesA=Osites)
    for i, j in pp1_pairs:
        H1[j//nscsites, j%nscsites, i] = -tpp1

    # ed (-Delta_pd), Ud, Up
    for i, orb in enumerate(lattice.supercell.names):
        if orb == "Cu":
            H1[0, i, i] = ed
            if ignore_intercell:
                H2[i, i, i, i] = Ud
            else:
                H2[0, i, i, i, i] = Ud
        elif orb == "O":
            if ignore_intercell:
                H2[i, i, i, i] = Up
            else:
                H2[0, i, i, i, i] = Up
        else:
            log.error("wrong orbital name %s in 3-band Hubbard model", orb)
    return HamNonInt(lattice, H1, H2)

def test_hub1d_ham():
    from pyscf import lib
    from libdmet.utils import logger as log
    import libdmet.dmet.Hubbard as dmet
    import libdmet.system.hamiltonian as ham
    from libdmet.utils.misc import max_abs

    log.verbose = "DEBUG2"

    U = 4.0
    LatSize = 10
    ImpSize = 2
    Filling = 1.0 / 2.0

    ntotal = Filling * np.prod(LatSize)
    if ntotal - np.round(ntotal) > 1e-5:
        log.warning("rounded total number of electrons to integer %d", np.round(ntotal))
        Filling=float(np.round(ntotal)) / np.prod(LatSize)

    #Lat = dmet.SquareLattice(*(LatSize + ImpSize))
    Lat = dmet.ChainLattice(LatSize, ImpSize)
    Ham = ham.HubbardHamiltonian(Lat, U)
    Lat.setHam(Ham)
    nscsites = Lat.nscsites
    nkpts = Lat.nkpts
    nsites = Lat.nsites

    #vcor = dmet.AFInitGuess(ImpSize, U, Filling)
    vcor = dmet.PMInitGuess(ImpSize, U, Filling)

    H1_R = Lat.getH1(kspace=False)
    H1_full_1 = Lat.expand(H1_R, dense=False)
    H1_full_2 = Lat.expand(H1_R, dense=True)

    H1_ref = np.zeros((nsites, nsites))
    for i in range(nsites-1):
        H1_ref[i, i+1] = H1_ref[i+1, i] = -1.0
    H1_ref[nsites-1, 0] = H1_ref[0, nsites-1] = -1.0  # PBC

    eri = np.zeros((nsites, nsites, nsites, nsites))
    for i in range(nsites):
        eri[i, i, i, i] = U

    diff_H1_1 = max_abs(H1_full_1 - H1_ref)
    diff_H1_2 = max_abs(H1_full_2 - H1_ref)

    print ("diff_H1 (sparse): %s" %diff_H1_1)
    print ("diff_H1 (dense): %s" %diff_H1_2)
    assert diff_H1_1 < 1e-8
    assert diff_H1_2 < 1e-8

def test_hub2d_ham():
    from pyscf import ao2mo
    import libdmet.dmet.Hubbard as dmet
    import libdmet.system.hamiltonian as ham
    from libdmet.utils.misc import max_abs
    from libdmet.utils import logger as log
    log.verbose = "DEBUG2"

    LatSize = [3, 3]
    ImpSize = [3, 3]
    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites

    # Hubbard Hamiltonian
    U = 4.0
    Ham = dmet.Ham(Lat, U, obc=True, compact=False)
    H1_obc = Ham.getH1()[0]
    H2_obc = Ham.getH2()

    # reference
    # 0 1 2
    # 3 4 5
    # 6 7 8
    H2_ref = np.zeros((nao, nao, nao, nao))
    H2_ref[range(nao), range(nao), range(nao), range(nao)] = U

    # OBC
    H1_obc_ref = np.zeros((nao, nao))
    H1_obc_ref[0, 1] = H1_obc_ref[1, 0] = -1
    H1_obc_ref[0, 3] = H1_obc_ref[3, 0] = -1
    H1_obc_ref[1, 2] = H1_obc_ref[2, 1] = -1
    H1_obc_ref[1, 4] = H1_obc_ref[4, 1] = -1
    H1_obc_ref[2, 5] = H1_obc_ref[5, 2] = -1
    H1_obc_ref[3, 4] = H1_obc_ref[4, 3] = -1
    H1_obc_ref[3, 6] = H1_obc_ref[6, 3] = -1
    H1_obc_ref[4, 5] = H1_obc_ref[5, 4] = -1
    H1_obc_ref[4, 7] = H1_obc_ref[7, 4] = -1
    H1_obc_ref[5, 8] = H1_obc_ref[8, 5] = -1
    H1_obc_ref[6, 7] = H1_obc_ref[7, 6] = -1
    H1_obc_ref[7, 8] = H1_obc_ref[8, 7] = -1
    assert max_abs(H1_obc - H1_obc_ref) < 1e-12
    assert max_abs(H2_obc - H2_ref) < 1e-12

    # PBC
    Ham = dmet.Ham(Lat, U)
    H1_pbc = Ham.getH1()[0]
    H2_pbc = Ham.getH2()

    H1_pbc_ref = np.array(H1_obc_ref)
    H1_pbc_ref[0, 2] = H1_pbc_ref[2, 0] = -1
    H1_pbc_ref[0, 6] = H1_pbc_ref[6, 0] = -1
    H1_pbc_ref[1, 7] = H1_pbc_ref[7, 1] = -1
    H1_pbc_ref[2, 8] = H1_pbc_ref[8, 2] = -1
    H1_pbc_ref[3, 5] = H1_pbc_ref[5, 3] = -1
    H1_pbc_ref[6, 8] = H1_pbc_ref[8, 6] = -1
    assert max_abs(H1_pbc - H1_pbc_ref) < 1e-12
    assert max_abs(H2_pbc - H2_ref) < 1e-12

    # compact H2
    Ham = dmet.Ham(Lat, U, compact=True)
    H2 = Ham.getH2()
    H2_full = ao2mo.restore(1, H2, nao)
    assert max_abs(H2_full - H2_ref) < 1e-12

def test_hubbard_dca_ham():
    import libdmet.dmet.Hubbard as dmet
    import libdmet.system.hamiltonian as ham
    from libdmet.utils import logger as log
    log.verbose = "DEBUG2"

    LatSize = [9, 6]
    ImpSize = [3, 3]
    Lat = dmet.SquareLattice(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites

    # Hubbard Hamiltonian
    U = 4.0
    Ham = ham.HubbardDCA(Lat, U, tlist=[1.0, -0.2])

def test_3band_ham():
    """
    Test the 3band Hamiltonian with different cells.
    """
    import libdmet.dmet.Hubbard as dmet
    import libdmet.system.hamiltonian as ham
    from libdmet.utils import logger as log
    from libdmet.utils.misc import max_abs
    log.verbose = "DEBUG2"

    ## Hubbard Hamiltonian
    Ud = 8.0
    Up = 4.0
    ed = -3.0
    tpd = -1.0
    tpp = 0.5
    tpp1 = 0.1
    Vpd = 2.0

    """
    normal PM cell.
    """
    LatSize = [2, 2]
    ImpSize = [1, 1]
    Lat = dmet.Square3Band(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites
    Ham = ham.Hubbard3band(Lat, Ud, Up, ed, tpd, tpp, tpp1=tpp1, Vpd=Vpd, \
            ignore_intercell=False)
    H1 = Ham.getH1()
    H2 = Ham.getH2()

    #     -          -          -
    #   +0Cu+ -1O+ +3Cu+ -7O+ +0Cu+
    #     -          -          -
    #     +          +          +
    #    5O    1   11O    3    5O
    #     -          -          -
    #     -          -          -
    #   +3Cu+ -4O+ +9Cu+ -10O+ +6Cu+
    #     -          -          -
    #     +          +          +
    #    2O    0    8O    2    2O
    #     -          -          -
    #     -          -          -
    #   +0Cu+ -1O+ +6Cu+ -7O+ +0Cu+
    #     -          -          -
    #
    #                      0      1      2
    H1_ref = np.array([[[ ed,  -tpd,   tpd],
                        [-tpd,  0.0,   tpp],
                        [tpd,   tpp,   0.0]],
                       [[0.0,   0.0,   -tpd],
                        [0.0,   -tpp1, -tpp],
                        [-tpd,  -tpp,  -tpp1]],
                       [[0.0,   tpd,   0.0],
                        [tpd,   -tpp1, -tpp],
                        [0.0,  -tpp,   -tpp1]],
                       [[0.0,   0.0,   0.0],
                        [0.0,   0.0,   tpp],
                        [0.0,   tpp,   0.0]]])
    assert max_abs(H1 - H1_ref) < 1e-10

    H2_ref = np.zeros_like(H2)
    # 0-0 cell
    H2_ref[0, 0, 0, 0, 0] = Ud
    H2_ref[0, 1, 1, 1, 1] = Up
    H2_ref[0, 2, 2, 2, 2] = Up
    H2_ref[0, 0, 0, 1, 1] = Vpd
    H2_ref[0, 1, 1, 0, 0] = Vpd
    H2_ref[0, 0, 0, 2, 2] = Vpd
    H2_ref[0, 2, 2, 0, 0] = Vpd

    # 1-0 cell
    H2_ref[1, 0, 0, 2, 2] = Vpd
    H2_ref[1, 2, 2, 0, 0] = Vpd

    # 2-0 cell
    H2_ref[2, 0, 0, 1, 1] = Vpd
    H2_ref[2, 1, 1, 0, 0] = Vpd

    # 3-0 cell, no coupling
    assert max_abs(H2 - H2_ref) < 1e-10

    Ham_ref = Hubbard3band_old(Lat, Ud, Up, ed, tpd, tpp, tpp1=tpp1, Vpd=Vpd, \
            ignore_intercell=False)
    assert max_abs(Ham.getH1() - Ham_ref.getH1()) < 1e-10
    assert max_abs(Ham.getH2() - Ham_ref.getH2()) < 1e-10

    LatSize = [6, 6]
    ImpSize = [3, 2]
    Lat = dmet.Square3Band(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites
    Ham = ham.Hubbard3band(Lat, Ud, Up, ed, tpd, tpp, tpp1=tpp1, Vpd=Vpd, \
            ignore_intercell=True)

    Ham_ref = Hubbard3band_old(Lat, Ud, Up, ed, tpd, tpp, tpp1=tpp1, Vpd=Vpd, \
            ignore_intercell=True)
    assert max_abs(Ham.getH1() - Ham_ref.getH1()) < 1e-10
    assert max_abs(Ham.getH2() - Ham_ref.getH2()) < 1e-10

    """
    symmetrized 2x2 cell.
    """
    LatSize = [4, 6]
    ImpSize = [1, 1]
    nao = nscsites = Lat.nscsites
    Lat = dmet.Square3BandSymm(*(LatSize+ImpSize))
    Ham = ham.Hubbard3band(Lat, Ud, Up, ed, tpd, tpp, tpp1=tpp1, Vpd=Vpd, \
            ignore_intercell=False)

    Ham_ref = Hubbard3band_old(Lat, Ud, Up, ed, tpd, tpp, tpp1=tpp1, Vpd=Vpd, \
            ignore_intercell=False)
    assert max_abs(Ham.getH1() - Ham_ref.getH1()) < 1e-10
    assert max_abs(Ham.getH2() - Ham_ref.getH2()) < 1e-10

    LatSize = [1, 1]
    ImpSize = [1, 1]
    nao = nscsites = Lat.nscsites
    Lat = dmet.Square3BandSymm(*(LatSize+ImpSize))
    Ham = ham.Hubbard3band(Lat, Ud, Up, ed, tpd, tpp, tpp1=tpp1, Vpd=Vpd, \
            ignore_intercell=False)

    H1 = Ham.getH1()
    H2 = Ham.getH2()

    #         +
    #        4O
    #         -
    #         -          -
    #      +3Cu+ -5O+ +6Cu+ -7O+
    #         -          -
    #         +          +
    #        2O         8O
    #         -          -
    #         -          -
    # -1O+ +0Cu+ -11O+ +9Cu+
    #         -          -
    #                    +
    #                  10O
    #                    -
    #
    #                   Cu    O    O   Cu    O      O    Cu    O     O   Cu    O     O
    #                    0    1    2    3    4      5     6    7     8    9    10    11
    H1_ref = np.array([[[ed, tpd, tpd,  0,  -tpd,   0,    0,   0,    0,   0,    0,  -tpd],
                        [0,   0, -tpp,  0,   tpp,   0,    0, -tpp1,  tpp, -tpd, -tpp, -tpp1],
                        [0,   0,   0,  -tpd, -tpp1, -tpp, 0,  tpp,  -tpp1, 0,   0,  tpp],

                        [0,   0,   0,   ed,  tpd, -tpd,   0,  tpd,   0,   0,   0,    0 ],
                        [0,   0,   0,   0,    0,   tpp,   0,  -tpp,  0,   0,  -tpp1, -tpp],
                        [0,   0,   0,   0,    0,    0,   tpd, -tpp1, tpp, 0,  -tpp, -tpp1],

                        [0,   0,   0,   0,    0,    0,   ed, -tpd,  -tpd, 0,  tpd,    0],
                        [0,   0,   0,   0,    0,    0,   0,    0,   -tpp, 0,  tpp,    0 ],
                        [0,   0,   0,   0,    0,    0,   0,    0,    0,   tpd, -tpp1, -tpp],

                        [0,   0,   0,   0,    0,    0,   0,    0,    0,   ed,  -tpd,  tpd],
                        [0,   0,   0,   0,    0,    0,   0,    0,    0,   0,    0,    tpp],
                        [0,   0,   0,   0,    0,    0,   0,    0,    0,   0,    0,     0]]])

    H1_ref = H1_ref + H1_ref.transpose(0, 2, 1)
    H1_ref[0, range(nao), range(nao)] *= 0.5

    H2_ref = np.zeros_like(H2)
    for i in [0, 3, 6, 9]:
        H2_ref[0, i, i, i, i] = Ud
    for i in [1, 2, 4, 5, 7, 8, 10, 11]:
        H2_ref[0, i, i, i, i] = Up
    H2_ref[0, 0, 0, 1, 1] = Vpd
    H2_ref[0, 1, 1, 0, 0] = Vpd
    H2_ref[0, 0, 0, 2, 2] = Vpd
    H2_ref[0, 2, 2, 0, 0] = Vpd
    H2_ref[0, 0, 0, 4, 4] = Vpd
    H2_ref[0, 4, 4, 0, 0] = Vpd
    H2_ref[0, 0, 0, 11, 11] = Vpd
    H2_ref[0, 11, 11, 0, 0] = Vpd

    H2_ref[0, 3, 3, 2, 2] = Vpd
    H2_ref[0, 2, 2, 3, 3] = Vpd
    H2_ref[0, 3, 3, 4, 4] = Vpd
    H2_ref[0, 4, 4, 3, 3] = Vpd
    H2_ref[0, 3, 3, 5, 5] = Vpd
    H2_ref[0, 5, 5, 3, 3] = Vpd
    H2_ref[0, 3, 3, 7, 7] = Vpd
    H2_ref[0, 7, 7, 3, 3] = Vpd

    H2_ref[0, 6, 6, 5, 5] = Vpd
    H2_ref[0, 5, 5, 6, 6] = Vpd
    H2_ref[0, 6, 6, 7, 7] = Vpd
    H2_ref[0, 7, 7, 6, 6] = Vpd
    H2_ref[0, 6, 6, 8, 8] = Vpd
    H2_ref[0, 8, 8, 6, 6] = Vpd
    H2_ref[0, 6, 6, 10, 10] = Vpd
    H2_ref[0, 10, 10, 6, 6] = Vpd

    H2_ref[0, 9, 9, 1, 1] = Vpd
    H2_ref[0, 1, 1, 9, 9] = Vpd
    H2_ref[0, 9, 9, 8, 8] = Vpd
    H2_ref[0, 8, 8, 9, 9] = Vpd
    H2_ref[0, 9, 9, 10, 10] = Vpd
    H2_ref[0, 10, 10, 9, 9] = Vpd
    H2_ref[0, 9, 9, 11, 11] = Vpd
    H2_ref[0, 11, 11, 9, 9] = Vpd

    assert max_abs(H1 - H1_ref) < 1e-10
    assert max_abs(H2 - H2_ref) < 1e-10

    """
    AFM cell.
    """
    LatSize = [1, 1]
    ImpSize = [1, 1]
    Lat = dmet.Square3BandAFM(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites
    Ham = ham.Hubbard3band(Lat, Ud, Up, ed, tpd, tpp, tpp1=tpp1, Vpd=Vpd, \
            ignore_intercell=True)
    #
    #               -2O+
    #           +          +
    #           4O         O
    #           -          -
    #           -          -
    #    -2O+ +0Cu+ -3O+ +1Cu+ -2O+
    #           -          -
    #           +          +
    #           5O         O
    #           -          -
    #               -2O+
    #
    #                      0      1     2     3     4     5
    H1_ref = np.asarray([[ ed,  0.0,  tpd,  -tpd,  tpd, -tpd],
                         [0.0,   ed,  -tpd,  tpd,  -tpd, tpd],
                         [tpd, -tpd,  0.0,  -tpp1, -tpp, tpp],
                         [-tpd, tpd, -tpp1,  0.0,  tpp, -tpp],
                         [tpd, -tpd, -tpp,   tpp,  0.0, -tpp1],
                         [-tpd, tpd,  tpp,  -tpp, -tpp1, 0.0]])
    assert max_abs(Ham.getH1() - H1_ref) < 1e-10

    LatSize = [1, 2]
    ImpSize = [1, 1]
    Lat = dmet.Square3BandAFM(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites
    Ham = ham.Hubbard3band(Lat, Ud, Up, ed, tpd, tpp, tpp1=tpp1, Vpd=Vpd, \
            ignore_intercell=True)
    #
    #                         -2O+
    #
    #                      +           +
    #                    10O         11O
    #                      -           -
    #                      -           -
    #              -8O+ +6Cu+ -9O+  +7Cu+ -2O+
    #                      -           -
    #           +          +           +
    #          4O         5O         10O
    #           -          -           -
    #           -          -
    #   -2O+ +0Cu+ -3O+ +1Cu+ -8O+
    #           -          -
    #           +          +
    #         11O         4O
    #           -          -
    #
    #              -2O+
    #
    #                        0     1     2     3     4     5
    H1_ref = np.asarray([[[ ed,  0.0,  tpd,  -tpd,  tpd,  0.0],
                          [0.0,   ed,  0.0,  tpd,  -tpd,  tpd],
                          [tpd,  0.0,  0.0,  -tpp1, -tpp, 0.0],
                          [-tpd, tpd, -tpp1,  0.0,  tpp, -tpp],
                          [tpd, -tpd, -tpp,   tpp,  0.0, -tpp1],
                          [0.0,  tpd,  0.0,  -tpp, -tpp1, 0.0]],
                         [[0.0,  0.0,  0.0,   0.0,  0.0, -tpd],
                          [0.0,  0.0, -tpd,   0.0,  0.0,  0.0],
                          [0.0, -tpd,  0.0, -tpp1, -tpp,  tpp],
                          [0.0,  0.0, -tpp1,  0.0,  0.0, -tpp],
                          [0.0,  0.0, -tpp,   0.0,  0.0, -tpp1],
                          [-tpd, 0.0,  tpp,  -tpp, -tpp1, 0.0]]])
    assert max_abs(Ham.getH1() - H1_ref) < 1e-10
    H1_ref_sc = Lat.expand(H1_ref)

    LatSize = [1, 2]
    ImpSize = [1, 2]
    Lat = dmet.Square3BandAFM(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites
    Ham = ham.Hubbard3band(Lat, Ud, Up, ed, tpd, tpp, tpp1=tpp1, Vpd=Vpd, \
            ignore_intercell=True)
    assert max_abs(Ham.getH1() - H1_ref_sc) < 1e-10

    """
    AFM cell asymmetric.
    """
    LatSize = [1, 1]
    ImpSize = [1, 1]
    Lat = dmet.Square3BandAFM(*(LatSize+ImpSize), symm=False)
    nao = nscsites = Lat.nscsites
    Ham = ham.Hubbard3band(Lat, Ud, Up, ed, tpd, tpp, tpp1=tpp1, Vpd=Vpd, \
            ignore_intercell=True)
    #
    #               -2O+
    #           +          +
    #           4O         O
    #           -          -
    #           -          -
    #    -2O+ +0Cu+ -3O+ +1Cu+ -2O+
    #           -          -
    #           +          +
    #           5O         O
    #           -          -
    #               -2O+
    #
    #                      0      1     2     3     4     5
    H1_ref = np.asarray([[ ed,  0.0,  tpd,  -tpd,  tpd, -tpd],
                         [0.0,   ed,  -tpd,  tpd,  -tpd, tpd],
                         [tpd, -tpd,  0.0,  -tpp1, -tpp, tpp],
                         [-tpd, tpd, -tpp1,  0.0,  tpp, -tpp],
                         [tpd, -tpd, -tpp,   tpp,  0.0, -tpp1],
                         [-tpd, tpd,  tpp,  -tpp, -tpp1, 0.0]])
    assert max_abs(Ham.getH1() - H1_ref) < 1e-10

    LatSize = [1, 2]
    ImpSize = [1, 1]
    Lat = dmet.Square3BandAFM(*(LatSize+ImpSize), symm=False)
    nao = nscsites = Lat.nscsites
    Ham = ham.Hubbard3band(Lat, Ud, Up, ed, tpd, tpp, tpp1=tpp1, Vpd=Vpd, \
            ignore_intercell=True)

    # if use old version of code, the arrangement would be the following:
    #
    #                          -2O+
    #
    #                      +           +
    #                    10O          5O
    #                      -           -
    #                      -           -
    #               -8O+ +6Cu+ -9O+  +7Cu+ -2O+
    #                      -           -
    #           +          +           +
    #           4O       11O         10O
    #           -          -           -
    #           -          -
    #    -2O+ +0Cu+ -3O+ +1Cu+ -8O+
    #           -          -
    #           +          +
    #           5O        4O
    #           -          -
    #
    #               -2O+
    #
    # It is clear that the two Cus are equivalent in the lattice sense,
    # but NOT locally.
    # The new version of code gives a symmetric way.
    #
    H1_ref = np.asarray([[[ ed,  0.0,  tpd,  -tpd,  tpd, -tpd],
                          [0.0,   ed,  0.0,  tpd,  -tpd,  0.0],
                          [tpd,  0.0,  0.0,  -tpp1, -tpp, tpp],
                          [-tpd, tpd, -tpp1,  0.0,  tpp, -tpp],
                          [tpd, -tpd, -tpp,   tpp,  0.0, -tpp1],
                          [-tpd, 0.0,  tpp,  -tpp, -tpp1, 0.0]],
                         [[0.0,  0.0,  0.0,   0.0,  0.0,  0.0],
                          [0.0,  0.0, -tpd,   0.0,  0.0,  tpd],
                          [0.0, -tpd,  0.0, -tpp1, -tpp,  0.0],
                          [0.0,  0.0, -tpp1,  0.0,  0.0, -tpp],
                          [0.0,  0.0, -tpp,   0.0,  0.0, -tpp1],
                          [0.0,  tpd,  0.0,  -tpp, -tpp1, 0.0]]])
    assert max_abs(Ham.getH1() - H1_ref) < 1e-10
    H1_ref_sc = Lat.expand(H1_ref)
    # When large lattice is used, the first cell coupling
    #H1_ref = np.asarray([[[ ed,  0.0,  0.0,  -tpd,  tpd,  0.0],
    #                      [0.0,   ed,  0.0,  tpd,   0.0,  tpd],
    #                      [0.0,  0.0,  0.0,  -tpp1, 0.0, 0.0],
    #                      [-tpd, tpd, -tpp1,  0.0,  tpp, -tpp],
    #                      [tpd,  0.0,  0.0,   tpp,  0.0, -tpp1],
    #                      [0.0,  tpd,  0.0,  -tpp, -tpp1, 0.0]],
    #

    LatSize = [1, 2]
    ImpSize = [1, 2]
    Lat = dmet.Square3BandAFM(*(LatSize+ImpSize), symm=False)
    nao = nscsites = Lat.nscsites
    Ham = ham.Hubbard3band(Lat, Ud, Up, ed, tpd, tpp, tpp1=tpp1, Vpd=Vpd, \
            ignore_intercell=True)
    assert max_abs(Ham.getH1() - H1_ref_sc) < 1e-10

    """
    Test 3band model from reference
    """
    for name in ["Hybertsen", "Martin", "Hanke", "Wagner"]:
        for hole_rep in [True, False]:
            for min_model in [True, False]:
                Ham = ham.Hubbard3band_ref(Lat, name=name, \
                        hole_rep=hole_rep,
                        min_model=min_model, \
                        ignore_intercell=True)
                Ham_half = ham.Hubbard3band_ref(Lat, name=name, \
                        hole_rep=hole_rep,
                        min_model=min_model, factor=0.5, \
                        ignore_intercell=True)

                print ("name: ", name)
                print ("hole_rep: ", hole_rep)
                print ("min_model: ", min_model)
                print (Ham.getH1())
                print ()
                assert max_abs(Ham_half.getH1() - Ham.getH1() * 0.5) < 1e-10
                assert max_abs(Ham_half.getH2() - Ham.getH2() * 0.5) < 1e-10

    # using dictionary (hole parameters) to initialize
    dic = {"Ud": 12.0, "tpd": 1.5, "D_pd": 4.5, "Vpd": 2.0}
    Ham_from_dic = ham.Hubbard3band_ref(Lat, name=dic, \
            hole_rep=False,
            min_model=True, \
            ignore_intercell=True)
    Ham_from_str = ham.Hubbard3band_ref(Lat, name="Hanke", \
            hole_rep=False,
            min_model=True, \
            ignore_intercell=True)
    assert max_abs(Ham_from_dic.getH1() - Ham_from_str.getH1()) < 1e-10
    assert max_abs(Ham_from_dic.getH2() - Ham_from_str.getH2()) < 1e-10

    # check the equivalency btw hole and electron repr.
    from pyscf import fci
    from libdmet.system.integral import Integral
    from libdmet.solver import impurity_solver
    np.set_printoptions(4, linewidth=1000, suppress=True)

    LatSize = [1, 1]
    ImpSize = [1, 1]
    Lat = dmet.Square3BandSymm(*(LatSize+ImpSize))
    nao = nscsites = Lat.nscsites

    Ham_h = ham.Hubbard3band_ref(Lat, name="wagner", hole_rep=True, min_model=False,
                                 ignore_intercell=True)
    h1 = Ham_h.getH1()[0]
    h2 = Ham_h.getH2()
    h0 = Ham_h.getH0()
    norb = h1.shape[-1]
    nhole = 4

    Ham = Integral(norb, True, False, h0, {"cd": h1[None]}, {"ccdd": h2[None]})
    solver = impurity_solver.FCI(restricted=True, tol=1e-10, beta=1000.0, scf_newton=False)

    e, fcivec = fci.direct_spin1.kernel(h1, h2, norb, (nhole//2, nhole//2), verbose=5, max_cycle=1000,
            pspace_size=1000, davidson_only=False, nroots=2)
    rdm1_h = fci.direct_spin1.make_rdm1(fcivec[0], norb, (nhole//2, nhole//2))
    rdm1_h2 = fci.direct_spin1.make_rdm1(fcivec[1], norb, (nhole//2, nhole//2))

    Ham_e = ham.Hubbard3band_ref(Lat, name="wagner", hole_rep=False, min_model=False,
                                 ignore_intercell=True)

    h1 = Ham_e.getH1()[0]
    h2 = Ham_e.getH2()
    h0 = Ham_e.getH0()
    norb = h1.shape[-1]
    nelec = norb * 2 - nhole

    Ham = Integral(norb, True, False, h0, {"cd": h1[None]}, {"ccdd": h2[None]})
    solver = impurity_solver.FCI(restricted=True, tol=1e-10, beta=1000.0, scf_newton=False)

    e, fcivec = fci.direct_spin1.kernel(h1, h2, norb, (nelec//2, nelec//2), verbose=5, max_cycle=1000,
            pspace_size=1000, davidson_only=False, nroots=2)
    rdm1_e = fci.direct_spin1.make_rdm1(fcivec[0], norb, (nelec//2, nelec//2))
    rdm1_e2 = fci.direct_spin1.make_rdm1(fcivec[1], norb, (nelec//2, nelec//2))

    rdm1_h_re = np.eye(norb) * 2 - rdm1_e
    rdm1_h_re2 = np.eye(norb) * 2 - rdm1_e2

    print ("ref")
    print (rdm1_h)
    print ("re")
    print (rdm1_h_re)
    print ("diff")
    print (rdm1_h_re - rdm1_h)
    print (max_abs(rdm1_h - rdm1_h_re))
    assert max_abs(rdm1_h - rdm1_h_re) < 1e-6

if __name__ == "__main__":
    test_3band_ham()
    test_hub1d_ham()
    test_hub2d_ham()
    test_hubbard_dca_ham()
