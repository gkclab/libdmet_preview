#! /usr/bin/env python

"""
Hamiltonian of several types of lattice models.

Author:
    Bo-Xiao Zheng
    Zhi-Hao Cui
"""

import numpy as np
import scipy.linalg as la
import itertools as it

from libdmet.utils import misc
from libdmet.utils import logger as log

class HamNonInt(object):
    def __init__(self, lattice, H1, H2, Fock=None, ImpJK=None, kspace_input=False,
                 spin_dim_H2=None, H0=0.0):
        """
        Class of non-interacting Hamiltonian.
        H2 might have cell label.

        Args:
            lattice: lattice object.
            H1: hcore, shape ((spin,), ncells, nao, nao).
            H2: shape ((spin_dim_H2,), (ncells,)) + eri shape.
            Fock: fock, if None will be taken as the same as hcore.
            ImpJK: JK_imp, shape ((spin,))
            kspace_input: H1 and Fock are in k space?
            spin_dim_H2: spin dimension of H2, None for no spin dimension.
            H0: H0.

        Returns:
            HamNonInt object.
        """
        ncells = lattice.ncells
        nao = lattice.nao
        nao_pair = nao * (nao + 1) // 2
        self.spin_dim_H2 = spin_dim_H2

        # 1. H1
        log.eassert(H1.shape[-3:] == (ncells, nao, nao), "H1 shape %s not"
                "compatible with lattice", H1.shape)
        if kspace_input:
            self.H1 = lattice.k2R(H1)
        else:
            self.H1 = H1

        # 2. Fock
        if Fock is None:
            self.Fock = self.H1
        else:
            log.eassert(Fock.shape[-3:] == self.H1.shape[-3:], "Fock shape %s"
                    "not compatible with lattice", Fock.shape)
            if kspace_input:
                self.Fock = lattice.k2R(Fock)
            else:
                self.Fock = Fock

        # 3. H2
        if self.spin_dim_H2 is None:
            if H2.shape == (nao,)*4 or H2.shape == (nao_pair, nao_pair):
                self.H2_format = "local"
            elif H2.shape == (ncells,) + (nao,)*4 \
                    or H2.shape == (ncells, nao_pair, nao_pair):
                self.H2_format = "nearest"
            elif H2.shape == (ncells,)*3 + (nao,)*4 \
                    or H2.shape == (ncells,)*3 + (nao_pair, nao_pair):
                self.H2_format = "full"
            else:
                log.error("H2 shape not compatible with supercell")
                raise ValueError
        else:
            if H2.shape == (self.spin_dim_H2,) + (nao,)*4 \
                    or H2.shape == (self.spin_dim_H2, nao_pair, nao_pair):
                self.H2_format = "spin local"
            elif H2.shape == (self.spin_dim_H2, ncells,) + (nao,)*4 \
                    or H2.shape == (self.spin_dim_H2, ncells, nao_pair, nao_pair):
                self.H2_format = "spin nearest"
            elif H2.shape == (self.spin_dim_H2,) + (ncells,)*3 + (nao,)*4 \
                    or H2.shape == (self.spin_dim_H2,) + (ncells,)*3 \
                    + (nao_pair, nao_pair):
                self.H2_format = "spin full"
            else:
                log.error("H2 shape %s not compatible with supercell", H2.shape)
                raise ValueError

        self.H2 = H2

        # 4. ImpJK
        if ImpJK is None:
            self.ImpJK = None
        else:
            log.eassert(ImpJK.shape[-2:] == self.H1.shape[-2:],
                        "ImpJK shape %s not compatible with supercell", ImpJK.shape)
            self.ImpJK = ImpJK

        # 5. H0
        self.H0 = H0

    def getH0(self):
        return self.H0

    def getH1(self):
        return self.H1

    def getH2(self):
        return self.H2

    def getFock(self):
        return self.Fock

    def getImpJK(self):
        return self.ImpJK

def HubbardHamiltonian(lattice, U, tlist=[1.0], obc=False, compact=False,
                       tol=1e-10, return_H1=False):
    """
    1-band Hubbard model in electron representation:
    H = -t <ij> -t' <<ij>> - ...

    Args:
        lattice: lattice object
        U: Hubabrd U
        tlist: t, t', t'', ... values.
        obc: open boundary condition.
        compact: whether to use 4-fold symmetry for ERI.
        tol: tolerance for ignoring small t.
        return_H1: only return H1.

    Returns:
        Ham: Hubbard Hamitltonian.
    """
    ncells = lattice.ncells
    nscsites = lattice.nscsites
    H1 = np.zeros((ncells, nscsites, nscsites))
    search_range = 0 if obc else 1

    for order, t in enumerate(tlist):
        if abs(t) < tol:
            continue
        log.eassert(order < len(lattice.neighborDist),
                    "%dth near neighbor distance unspecified in Lattice object",
                    order+1)
        dis = lattice.neighborDist[order]
        log.warning("Searching neighbor within only one supercell")
        pairs = lattice.neighbor(dis=dis, sitesA=range(nscsites),
                                 search_range=search_range)
        for i, j in pairs:
            H1[j // nscsites, j % nscsites, i] = -t

    if return_H1:
        return H1
    else:
        if compact:
            nao_pair = nscsites * (nscsites + 1) // 2
            H2 = np.zeros((nao_pair, nao_pair))
            diag_idx = misc.tril_diag_indices(nscsites)
            H2[diag_idx, diag_idx] = U
        else:
            H2 = np.zeros((nscsites,) * 4)
            np.fill_diagonal(H2, U)
        return HamNonInt(lattice, H1, H2)

def HubbardDCA(lattice, U, tlist=[1.0]):
    """
    1-band Hubbard model with DCA.
    """
    assert len(tlist) < 3
    from libdmet.utils import dca_transform

    cells = tuple(lattice.csize)
    scsites = tuple(lattice.supercell.csize)
    dim = lattice.dim
    H = []
    def vec1(d, v1, v2):
        idx = [0] * dim * 2
        idx[d] = v1
        idx[d+dim] = v2
        return tuple(idx)

    for d in range(dim):
        H.append((vec1(d, 0, 1), -tlist[0]))
        H.append((vec1(d, cells[d]-1, scsites[d]-1), -tlist[0]))

    if len(tlist) == 2:
        assert(dim == 2)
        H.append(((0, 0, 1, 1), tlist[1]))
        H.append(((0, cells[1]-1, 1, scsites[1]-1), tlist[1]))
        H.append(((cells[0]-1, 0, scsites[0]-1, 1), tlist[1]))
        H.append(((cells[0]-1, cells[1]-1, scsites[0]-1, scsites[1]-1), tlist[1]))

    H_DCA = dca_transform.transformHam(cells, scsites, H)

    ncells = lattice.ncells
    nscsites = lattice.nscsites
    H1 = np.zeros((ncells, nscsites, nscsites))

    for pos, val in H_DCA:
        cidx = lattice.cell_pos2idx(pos[:dim])
        spos = np.asarray(pos[dim:])
        for s in range(nscsites):
            s1 = lattice.supercell.sitedict[tuple((lattice.supercell.sites[s]+spos) % scsites)]
            #H1[cidx, s, s1] = val
            H1[cidx, s1, s] = val

    H2 = np.zeros((nscsites,)*4)
    np.fill_diagonal(H2, U)
    return HamNonInt(lattice, H1, H2)

def Hubbard3band(lattice, Ud, Up, ed, tpd, tpp, tpp1=0.0, Vpd=0.0,
                 ignore_intercell=True, tol=1e-10):
    """
    3-band Hubbard model in electron representation:
    H = tpd + tpp + tpp' + ed + Ud + Up + Vpd
    t and ed is in general negative, while U is positive.
    """
    from libdmet.system.lattice import Frac2Real, Real2Frac, round_to_FUC
    ncells = lattice.ncells
    nscsites = lattice.nscsites
    H1 = np.zeros((ncells, nscsites, nscsites))
    if ignore_intercell:
        H2 = np.zeros((nscsites,) * 4)
    else:
        if abs(Vpd) < 1e-5:
            log.warn("Vpd %s is very small, ignore_intercell should be set to True...", Vpd)
        H2 = np.zeros((ncells,) + (nscsites,)*4)
    d_pd  = lattice.neighborDist[0]
    d_pp  = lattice.neighborDist[1]
    d_pp1 = lattice.neighborDist[2]
    log.warning("Searching neighbor within only one supercell")

    def get_vec(s1, s2):
        # round vector to [-0.5, 0.5)
        vec_frac = Real2Frac(lattice.size, lattice.sites[s1] - lattice.sites[s2])
        vec_frac = round_to_FUC(vec_frac, tol=tol, wrap_around=True)
        vec = Frac2Real(lattice.size, vec_frac)
        return vec

    # tpd and Vpd
    pd_pairs = lattice.neighbor(dis=d_pd, sitesA=range(nscsites))
    for i, j in pd_pairs:
        # vec = the vector start from Cu to O.
        # the Cu-O bond is assumed to be along x or y axis.
        if lattice.names[i] == "Cu":
            vec = get_vec(j, i)
        else:
            vec = get_vec(i, j)

        if abs(vec[0] - 1.0) < tol or abs(vec[1] + 1.0) < tol:
            # vec = x [1, 0] or vec = -y [0, -1]
            sign = -1.0
        elif abs(vec[1] - 1.0) < tol or abs(vec[0] + 1.0) < tol:
            # vec = y [0, 1] or vec = -x [-1, 0]
            sign = 1.0
        else:
            log.error("invalid p-d neighbor, vec = %s", vec)
            raise ValueError

        H1[j//nscsites, j%nscsites, i] = sign * tpd

        if ignore_intercell:
            if j // nscsites == 0:
                H2[j, j, i, i] = Vpd
        else:
            # V^{P, P, 0, 0}_{pqrs}
            H2[j//nscsites, j%nscsites, j%nscsites, i, i] = Vpd

    # tpp
    pp_pairs = lattice.neighbor(dis=d_pp, sitesA=range(nscsites))
    for i, j in pp_pairs:
        vec = get_vec(j, i)
        if vec[0] * vec[1] > 0:
            # vec = [1, 1] or [-1, -1]
            sign = -1.0
        else:
            # vec = [1, -1] or [-1, 1]
            sign = 1.0
        H1[j//nscsites, j%nscsites, i] = sign * tpp

    # tpp'
    Osites = [idx for (idx, name) in
              zip(range(nscsites), lattice.names[:nscsites]) if name == "O"]
    pp1_pairs = lattice.neighbor(dis=d_pp1, sitesA=Osites)
    for i, j in pp1_pairs:
        # ZHC FIXME should the sign of tpp be postitive?
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

def Hubbard3band_ref(lattice, name, min_model=False, hole_rep=False,
                     factor=1.0, ignore_intercell=True, tol=1e-10):
    """
    3-band Hubbard model in electron representation.
    Using parameters from reference.

    Args:
        name: Currently supported model names:
            Hybertsen, Hybertsen89, PRB
            Martin,    Martin96, PRB
            Hanke,     Hanke10, EPJ
            Wagner,    Vitali18, PRB
        min_model: only keep Ud, Up, tpd and ed.
        hole_rep: use hole representation.
        factor: scale factor.

    Returns:
        Ham: 3band Hamiltonian.
    """
    if isinstance(name, str):
        name = name.lower().strip()
        if name == "hybertsen":
            # Hybertsen (hole)
            Ud   = 10.5
            tpd  = 1.3
            D_pd = 3.6

            Up   = 4.0
            tpp  = 0.65
            tpp1 = 0.0
            Vpd  = 1.2
            # electron rep
            # ed = D_pd - Ud  - 4Vpd    + Up
            #    = 3.6  - 10.5 - 4 * 1.3 + 4.0 = -8.1
        elif name == "martin":
            # Martin (hole)
            Ud   = 16.5
            tpd  = 1.8
            D_pd = 5.4

            Up   = 0.0
            tpp  = 0.6
            tpp1 = 0.0
            Vpd  = 0.0
            # electron rep
            # ed = D_pd - Ud  - 4Vpd    + Up
            #    = 5.4  - 16.5 - 4 * 0.0 + 0.0 = -11.1

        elif name == "hanke":
            # Hanke (hole)
            Ud   = 12.0
            tpd  = 1.5
            D_pd = 4.5

            Up   = 5.25
            tpp  = 0.75
            tpp1 = 0.0
            Vpd  = 0.75
            # electron rep
            # ed = D_pd - Ud - 4Vpd + Up =
            #      4.5 - 12.0 - 4 * 0.75 + 5.25 = -5.25
            #
        elif name == "wagner":
            # Wagner (hole)
            Ud   = 8.4
            tpd  = 1.2
            D_pd = 4.4

            Up   = 2.0
            tpp  = 0.7
            tpp1 = 0.0
            Vpd  = 0.0
            # electron rep
            # ed = D_pd -  Ud  - 4Vpd    + Up
            #    = 4.4  - 8.4  - 4 * 0.0 + 2.0 = -2.0
        else:
            raise ValueError("Unknown name of 3band model: %s" %name)
    else:
        log.debug(0, "input parameters:\n%s", name)
        known_keys = set(["Ud", "tpd", "D_pd", "Up", "tpp", "tpp1", "Vpd"])
        log.eassert(set(name.keys()).issubset(known_keys),
                    "Unknown parameter names.")

        Ud   = name["Ud"]
        tpd  = name["tpd"]
        D_pd = name["D_pd"]

        Up   = name.get("Up", 0.0)
        tpp  = name.get("tpp", 0.0)
        tpp1 = name.get("tpp1", 0.0)
        Vpd  = name.get("Vpd", 0.0)


    if min_model:
        tpp = tpp1 = Up = Vpd = 0.0

    if hole_rep:
        ed = -D_pd
    else: # p-h transform to electron rep
        tpd  = -tpd
        tpp  = -tpp
        tpp1 = -tpp1
        ed = D_pd - Ud - 4 * Vpd + Up

    if factor != 1.0:
        Ud   *= factor
        tpd  *= factor
        ed   *= factor
        Up   *= factor
        tpp  *= factor
        tpp1 *= factor
        Vpd  *= factor

    return Hubbard3band(lattice, Ud, Up, ed, tpd, tpp, tpp1=tpp1, Vpd=Vpd,
                        ignore_intercell=ignore_intercell, tol=tol)

