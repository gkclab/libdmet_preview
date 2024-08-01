#! /usr/bin/env python

"""
Base routines for Hubbard based DMET.

Author:
    Bo-Xiao Zheng
    Zhi-Hao Cui
"""

import types
import numpy as np
import scipy.linalg as la
import itertools as it

from libdmet.system.lattice import ChainLattice, SquareLattice, \
        SquareAFM, Square3Band, Square3BandAFM, Square3BandSymm, CubicLattice, \
        HoneycombLattice, BipartiteSquare
from libdmet.system.hamiltonian import HubbardHamiltonian as Ham
from libdmet.system.hamiltonian import Hubbard3band, Hubbard3band_ref, HubbardDCA
from libdmet.routine import vcor, slater
from libdmet.routine.mfd import HF
from libdmet.routine.diis import FDiisContext
from libdmet.utils import logger as log
from libdmet.solver import impurity_solver
from libdmet.utils.misc import mdot
from libdmet.routine.slater_helper import transform_trans_inv_sparse

def HartreeFock(Lat, v, U):
    rho, mu, E, res = HF(Lat, v, 0.5, False, mu0 = U/2, beta = np.inf, ires = True)
    log.result("Local density matrix (mean-field):\n%s\n%s", rho[0][0], rho[1][0])
    log.result("Chemical potential (mean-field) = %20.12f", mu)
    log.result("Energy per site (mean-field) = %20.12f", E/Lat.nscsites)
    log.result("Gap (mean-field) = %20.12f" % res["gap"])
    return rho, mu

def basisMatching(basis):
    basisA, basisB = basis[0], basis[1]
    S = np.tensordot(basisA, basisB, axes = ((0,1), (0,1)))
    # S=A^T*B svd of S is UGV^T then we let A'=AU, B'=BV
    # yields A'^T*B'=G diagonal and optimally overlapped
    u, gamma, vt = la.svd(S)
    log.result("overlap statistics:\n larger than 0.9: %3d  smaller than 0.9: %3d\n"
            " average: %10.6f  min: %10.6f", \
            np.sum(gamma > 0.9), np.sum(gamma < 0.9), np.average(gamma), np.min(gamma))
    basisA = np.tensordot(basisA, u, axes = (2, 0))
    basisB = np.tensordot(basisB, vt, axes = (2, 1))
    return np.asarray([basisA, basisB])

def afqmc_symmetrize(lattice, basis, h, h1e):
    nbasis = basis.shape[-1]
    nscsites = lattice.nscsites
    nbath = nbasis - nscsites
    ewA, evA = la.eigh(h[0, -nbath:,-nbath:])
    ewB, evB = la.eigh(h[1, -nbath:,-nbath:])
    #log.check(la.norm(ewA + ewB[::-1]) < 1e-7, \
    #        "bath spectra violate particle-hole symmetry")
    log.debug(1, "bath spectra\n%s\n%s", ewA, ewB)
    for i in range(nbath-1):
        if abs(ewA[i] - ewA[i+1]) < 1e-7:
            log.warning("energy spectra degenerate, may have sign problem")
    basis[0,:,:,nscsites:] = np.tensordot(basis[0,:,:,nscsites:], evA, axes = (2, 0))
    basis[1,:,:,nscsites:] = np.tensordot(basis[1,:,:,nscsites:], evB[:,::-1], axes = (2,0))
    rot = np.zeros((nbasis,nbasis))
    rot[:nscsites, :nscsites] = np.eye(nscsites)
    rot[nscsites:, nscsites:] = evA
    h[0] = mdot(rot.T, h[0], rot)
    h1e[0] = mdot(rot.T, h1e[0], rot)
    rot[nscsites:, nscsites:] = evB[:,::-1]
    h[1] = mdot(rot.T, h[1], rot)
    h1e[1] = mdot(rot.T, h1e[1], rot)
    return basis, h, h1e

def ConstructImpHam(Lat, rho, v, mu=None, afqmc=False, matching=True, 
                    local=True, split=False, **kwargs):
    log.result("Making embedding basis")
    basis = slater.embBasis(Lat, rho, local=local, **kwargs)
    if afqmc:
        log.result("Constructing impurity Hamiltonian")
        ImpHam, H1e = slater.embHam(Lat, basis, v, local = True, **kwargs)
        log.info("rotate bath orbitals to achieve particle-hole symmetry")
        basis, ImpHam.H1["cd"], H1e["cd"] = afqmc_symmetrize(Lat, basis, ImpHam.H1["cd"], H1e["cd"])
    else:
        if matching and basis.shape[0] == 2:
            log.result("Rotate bath orbitals to match alpha and beta basis")
            nimp = Lat.nimp
            if local:
                basis[:, :, :, nimp:] = basisMatching(basis[:, :, :, nimp:])
            else:
                # ZHC FIXME should I modify nimp?
                # split matching occ and virt
                if split:
                    basis[:, :, :, :nimp] = basisMatching(basis[:, :, :, :nimp])
                    basis[:, :, :, nimp:] = basisMatching(basis[:, :, :, nimp:])
                else:
                    basis = basisMatching(basis)
            
        log.result("Constructing impurity Hamiltonian")
        ImpHam, H1e = slater.embHam(Lat, basis, v, local=local, **kwargs)
    return ImpHam, H1e, basis

def transformResults(rhoEmb, E, basis, ImpHam, H1e):
    spin = rhoEmb.shape[0]
    nscsites = basis.shape[2]
    rhoImp, Efrag, nelec = slater.transformResults(rhoEmb, E, basis, ImpHam, H1e)
    log.result("Local density matrix (impurity):")
    for s in range(spin):
        log.result("%s", rhoImp[s])
    log.result("nelec per site (impurity) = %20.12f", nelec/nscsites)
    log.result("Energy per site (impurity) = %20.12f", Efrag/nscsites)

    return rhoImp, Efrag/nscsites, nelec/nscsites

def InitGuess(ImpSize, U, polar = None, r = None):
    subA, subB = BipartiteSquare(ImpSize)
    v = VcorLocalPhSymm(U, False, ImpSize, subA, subB, r)
    if polar is None:
        polar = U * 0.5
    nscsites = np.prod(ImpSize)
    init_v = np.eye(nscsites) * U * 0.5
    init_p = np.diag([polar if s in subA else -polar for s in range(nscsites)])
    v.assign(np.asarray([init_v + init_p, init_v - init_p]))
    return v

def VcorLocalPhSymm(U, bogoliubov, ImpSize, subA, subB, r = None):
    # with particle-hole symmetry, on two sublattices
    # specifically for t'=0 Hubbard model at half-filling
    # unrestricted potential is assumed
    # the symmetry is
    # VA_{ij} + (-)^{i+j}VB_{ij} = 0
    # D_{ij} = (-)^{i+j}D_{ji}
    # AA=+, AB=-, BB=+
    assert(np.asarray(ImpSize).shape in [(1,), (2,), (3,)])
    subA, subB = set(subA), set(subB)
    log.eassert(len(subA) == len(subB), "number of sites in two sublattices are equal")
    nscsites = len(subA) * 2
    log.eassert(subA | subB == set(range(nscsites)), "sublattice designation problematic")
    nscsites = np.prod(ImpSize)
    log.eassert(subA | subB == set(range(nscsites)), "sublattice designation problematic")

    if r is None:
        pairs = list(it.combinations_with_replacement(range(nscsites), 2))
    else:
        pairs = []
        sites = list(enumerate(it.product(*map(range, ImpSize))))
        for (i, ri), (j, rj) in it.combinations_with_replacement(sites, 2):
            if la.norm(np.asarray(ri) - np.asarray(rj)) < r+1e-6:
                pairs.append((i, j))

    nV = len(pairs)

    v = vcor.Vcor()
    v.grad = None

    def sign(i, j):
        if (i in subA) == (j in subA):
            return 1
        else:
            return -1

    if bogoliubov:
        nD = nV
        def evaluate(self):
            log.eassert(self.param.shape == (nV+nD,), "wrong parameter shape, require %s", (nV+nD,))
            V = np.zeros((3, nscsites, nscsites))
            for idx, (i, j) in enumerate(pairs):
                V[0,i,j] = V[0,j,i] = self.param[idx]
                V[1,i,j] = V[1,j,i] = -self.param[idx] * sign(i,j)
                V[2,i,j] = self.param[idx+nV]
                if i != j:
                    V[2,j,i] = self.param[idx+nV] * sign(i,j)
            V[0] += np.eye(nscsites) * (U/2)
            V[1] += np.eye(nscsites) * (U/2)
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV+nD, 3, nscsites, nscsites))
                for idx, (i,j) in enumerate(pairs):
                    g[idx,0,i,j] = g[idx,0,j,i] = 1
                    g[idx,1,i,j] = g[idx,1,j,i] = -sign(i,j)
                    g[idx+nV,2,i,j] = 1
                    if i != j:
                        g[idx+nV,2,j,i] = sign(i,j)
                self.grad = g
            return self.grad
    else:
        nD = 0
        def evaluate(self):
            log.eassert(self.param.shape == (nV,), "wrong parameter shape, require %s", (nV,))
            V = np.zeros((2, nscsites, nscsites))
            for idx, (i,j) in enumerate(pairs):
                V[0,i,j] = V[0,j,i] = self.param[idx]
                V[1,i,j] = V[1,j,i] = -self.param[idx] * sign(i,j)
            V[0] += np.eye(nscsites) * (U/2)
            V[1] += np.eye(nscsites) * (U/2)
            return V

        def gradient(self):
            if self.grad is None:
                g = np.zeros((nV, 2, nscsites, nscsites))
                for idx, (i,j) in enumerate(pairs):
                    g[idx,0,i,j] = g[idx,0,j,i] = 1
                    g[idx,1,i,j] = g[idx,1,j,i] = -sign(i,j)
                self.grad = g
            return self.grad

    v.evaluate = types.MethodType(evaluate, v)
    v.gradient = types.MethodType(gradient, v)
    v.length = types.MethodType(lambda self: nV+nD, v)
    return v

def VcorDCAPhSymm(U, ImpSize, subA, subB):
    # Bogoliubov is always False
    assert(np.asarray(ImpSize).shape in [(1,), (2,)])
    subA, subB = set(subA), set(subB)
    log.eassert(len(subA) == len(subB), "number of sites in two sublattices are equal")
    nscsites = len(subA) * 2
    log.eassert(subA | subB == set(range(nscsites)), "sublattice designation problematic")
    nscsites = np.prod(ImpSize)
    log.eassert(subA | subB == set(range(nscsites)), "sublattice designation problematic")

    sites = list(it.product(*map(range, ImpSize)))
    sitedict = dict(zip(sites, range(len(sites))))

    container = set()
    vectors = []

    for s in sites:
        vec = []
        if not s in container:
            vec.append(np.asarray(s))
            container.add(s)
        s1 = tuple((-np.asarray(s)) % ImpSize)
        if not s1 in container:
            vec.append(np.asarray(s1))
            container.add(s1)
        if len(vec) > 0:
            vectors.append(vec)

    nV = len(vectors)

    v = vcor.Vcor()
    v.grad = None

    def evaluate(self):
        log.eassert(self.param.shape == (nV,), "wrong parameter shape, require %s", (nV,))
        V = np.zeros((2, nscsites, nscsites))

        for idxp, p in enumerate(self.param):
            for vec in vectors[idxp]:
                for idx1, site1 in enumerate(sites):
                    idx2 = sitedict[
                            tuple((np.asarray(site1) + vec) % ImpSize)
                    ]
                    if idx1 in subA and idx2 in subA:
                        V[0, idx1, idx2] = p
                        V[1, idx1, idx2] = -p
                    elif idx1 in subB and idx2 in subB:
                        V[0, idx1, idx2] = -p
                        V[1, idx1, idx2] = p
                    else:
                        V[0, idx1, idx2] = p
                        V[1, idx1, idx2] = p

        V[0] += np.eye(nscsites) * (U/2)
        V[1] += np.eye(nscsites) * (U/2)
        return V

    def gradient(self):
        if self.grad is None:
            g = np.zeros((nV, 2, nscsites, nscsites))

            for idxp, p in enumerate(self.param):
                for vec in vectors[idxp]:
                    for idx1, site1 in enumerate(sites):
                        idx2 = sitedict[
                            tuple((np.asarray(site1) + vec) % ImpSize)
                        ]
                        if idx1 in subA and idx2 in subA:
                            g[idxp, 0, idx1, idx2] = 1
                            g[idxp, 1, idx1, idx2] = -1
                        elif idx1 in subB and idx2 in subB:
                            g[idxp, 0, idx1, idx2] = -1
                            g[idxp, 1, idx1, idx2] = 1
                        else:
                            g[idxp, 0, idx1, idx2] = 1
                            g[idxp, 1, idx1, idx2] = 1
            self.grad = g 
        return self.grad

    v.evaluate = types.MethodType(evaluate, v)
    v.gradient = types.MethodType(gradient, v)
    v.length = types.MethodType(lambda self: nV, v)
    return v

def FitVcor(rho, lattice, basis, vcor, beta, MaxIter1 = 300, MaxIter2 = 20):
    log.info("degrees of freedom = %d", vcor.length())
    return slater.FitVcorTwoStep(rho, lattice, basis, vcor, beta, 0.5, \
            MaxIter1, MaxIter2)

class IterHistory(object):
    def __init__(self):
        self.history = []

    def update(self, energy, err, nelec, dvcor, dc):
        if self.history == []:
            self.history.append([energy, energy, err, nelec, dvcor, dc.nDim, dc.iNext])
        else:
            self.history.append([energy, energy - self.history[-1][0], err, nelec, dvcor, dc.nDim, dc.iNext])

        log.section("\nDMET Progress\n")
        log.result("  Iter         Energy                 dE                RdmErr         " \
            "       Nelec                 dVcor      DIIS")
        for idx, item in enumerate(self.history):
            log.result(" %3d %20.12f     %15.3e %20.12f %20.12f %20.5e  %2d %2d", idx, *item)
        log.result("")
    def write_table(self):
        #import os
        #if not os.path.exists('./table.txt'):
        f_table = open('./table.txt', 'w')
        f_table.write("  Iter         Energy                  dE                RdmErr               Nelec                 dVcor      DIIS \n")
        for idx, item in enumerate(self.history):
            f_table.write(" %3d %20.12f     %15.3e %20.12f %20.12f %20.5e  %2d %2d \n"%((idx,) + tuple(item)))
        f_table.close()

foldRho = slater.foldRho
foldRho_k = slater.foldRho_k
