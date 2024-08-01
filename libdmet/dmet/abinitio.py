#! /usr/bin/env python

import os
import numpy as np
import scipy.linalg as la

from libdmet.dmet.Hubbard import *
from libdmet.system.hamiltonian import HamNonInt
import libdmet.system.lattice as Lat
import libdmet.system.integral as integral
from libdmet.utils.misc import mdot

def buildUnitCell(size, atoms, basis):
    sites = []
    count = {}
    for a in atoms:
        coord, name = a
        if not name in count.keys():
            count[name] = 1
        else:
            count[name] += 1
        for b in basis[name]:
            sites.append((coord, name + "%d_" % count[name] + b))
    return Lat.UnitCell(size, sites)

def buildLattice(latSize, impSize, cellSize, atoms, basis):
    log.eassert(np.allclose(latSize % impSize, 0), \
            "incompatible lattice size and supercell size")
    uc = buildUnitCell(cellSize, atoms, basis)
    sc = Lat.SuperCell(uc, impSize)
    lat = Lat.Lattice(sc, latSize / impSize)
    return lat

def __read_bin(dirname, name, shape):
    if os.path.exists(os.path.join(dirname, name + ".npy")):
        temp = np.load(os.path.join(dirname, name + ".npy"))
        log.eassert(temp.shape == shape, \
            "when reading integral, the required shape is %s," \
            " but get %s", shape, temp.shape)
    elif os.path.exists(os.path.join(dirname, name + ".mmap")):
        temp = np.memmap(os.path.join(dirname, name + ".mmap"),\
            dtype = "float", mode = "c", shape = shape)
    else:
        log.error("couldn't find the integral file %s", name)
    return temp

def read_integral(dirname, lattice, cutoff = None):
    log.info("reading integrals from %s", os.path.realpath(dirname))
    nscsites = lattice.supercell.nsites
    dirname = os.path.realpath(dirname)
    if cutoff is None:
        nnonzero = lattice.ncells
    else:
        log.error("Deprecated function, do you know why you're using it?")
        nonzero = [np.asarray(x) for x in it.product(range(cutoff), \
                repeat = lattice.dim)]
        nnonzero = len(nonzero)
    H1 = __read_bin(dirname, "H1", (nnonzero, nscsites, nscsites))
    H2 = __read_bin(dirname, "H2", (nscsites,)*4)
    Fock = __read_bin(dirname, "Fock", (nnonzero, nscsites, nscsites))
    ImpJK = __read_bin(dirname, "ImpJK", (nscsites, nscsites))

    if cutoff is not None and nnonzero < lattice.ncells:
        FockFull = np.zeros((lattice.ncells, nscsites, nscsites))
        H1Full = np.zeros((lattice.ncells, nscsites, nscsites))
        for i, n in enumerate(nonzero):
            FockFull[lattice.cell_pos2idx(n)] = Fock[i]
            FockFull[lattice.cell_pos2idx(-n)] = Fock[i].T
            H1Full[lattice.cell_pos2idx(n)] = H1[i]
            H1Full[lattice.cell_pos2idx(-n)] = H1[i].T
        Fock, H1 = FockFull, H1Full
    return [H1, H2, Fock, ImpJK]

def buildHamiltonian(dirname, lattice, kspace_input = False):
    return HamNonInt(lattice, *(read_integral(dirname, lattice)), kspace_input = kspace_input)

def AFInitGuessIdx(v, nscsites, AFidx, PMidx, shift = 0., polar = 0.5, \
        bogoliubov = False, rand = 0., PMshift = None, PMidx2 = None, PMshift2 = None):
    subA, subB = AFidx
    subC = PMidx
    if PMshift is None:
        PMshift = shift
    if bogoliubov:
        vguess = np.zeros((3, nscsites, nscsites))
    else:
        vguess = np.zeros((2, nscsites, nscsites))
    for site in subA:
        vguess[0, site, site] = shift + polar
        vguess[1, site, site] = shift - polar
    for site in subB:
        vguess[0, site, site] = shift - polar
        vguess[1, site, site] = shift + polar
    for site in subC:
        vguess[0, site, site] = vguess[1, site, site] = PMshift
    if PMidx2 is not None:
        if PMshift2 is None:
            PMshift2 = 0.0
        for site in PMidx2:
            vguess[0, site, site] = vguess[1, site, site] = PMshift2
    if bogoliubov:
        np.random.seed(32499823)
        nact = len(subA) + len(subB)
        s = np.random.rand(1, nact, nact) - 0.5
        s[0] += s[0].T
        vguess[np.ix_([2], subA+subB, subA+subB)] = s * rand

    v.assign(vguess)
    return v

    # FIXME a hack, directly update the parameters
    #p = np.zeros(v.length())
    #psite = lambda site: (2*nscsites+1-site)*site/2
    #for site in subA:
    #    p[psite(site)] = shift + polar
    #    p[psite(site) + psite(nscsites)] = shift - polar
    #for site in subB:
    #    p[psite(site)] = shift - polar
    #    p[psite(site) + psite(nscsites)] = shift + polar
    #for site in subC:
    #    p[psite(site)] = p[psite(site) + psite(nscsites)] = shift
    #if bogoliubov:
    #    for site1 in subA+subB:
    #        for site2 in subA+subB:
    #            p[psite(nscsites)*2+site1*nscsites+site2] = \
    #                    vguess[2, site1, site2]

    #v.update(p)
    #log.eassert(la.norm(v.get() - vguess) < 1e-10, \
    #        "initial guess cannot be assgned directly")
    #return v


def AFInitGuessOrbs(v, lattice, AForbs, PMorbs, shift = 0., polar = 0.5, \
        bogoliubov = False, rand = 0.):
    names = lattice.supercell.names
    nscsites = lattice.supercell.nsites
    subA = [names.index(x) for x in AForbs[0]]
    subB = [names.index(x) for x in AForbs[1]]
    subC = [names.index(x) for x in PMorbs]
    return AFInitGuessIdx(v, nscsites, (subA, subB), subC, shift, polar, \
            bogoliubov, rand)
    #if bogoliubov:
    #    vguess = np.zeros((3, nscsites, nscsites))
    #else:
    #    vguess = np.zeros((2, nscsites, nscsites))
    #for site in subA:
    #    vguess[0, site, site] = shift + polar
    #    vguess[1, site, site] = shift - polar
    #for site in subB:
    #    vguess[0, site, site] = shift - polar
    #    vguess[1, site, site] = shift + polar
    #for site in subC:
    #    vguess[0, site, site] = vguess[1, site, site] = shift
    #if bogoliubov:
    #    np.random.seed(32499823)
    #    nact = len(subA) + len(subB)
    #    vguess[np.ix_([2], subA+subB, subA+subB)] = \
    #        (np.random.rand(1, nact, nact) - 0.5) * rand

    ## FIXME a hack, directly update the parameters
    #p = np.zeros(v.length())
    #psite = lambda site: (2*nscsites+1-site)*site/2
    #for site in subA:
    #    p[psite(site)] = shift + polar
    #    p[psite(site) + psite(nscsites)] = shift - polar
    #for site in subB:
    #    p[psite(site)] = shift - polar
    #    p[psite(site) + psite(nscsites)] = shift + polar
    #for site in subC:
    #    p[psite(site)] = p[psite(site) + psite(nscsites)] = shift
    #if bogoliubov:
    #    for site1 in subA+subB:
    #        for site2 in subA+subB:
    #            p[psite(nscsites)*2+site1*nscsites+site2] = \
    #                    vguess[2, site1, site2]

    #v.update(p)
    #log.eassert(la.norm(v.get() - vguess) < 1e-10, \
    #        "initial guess cannot be assgned directly")
    #return v

def reportOccupation(lattice, rho, names = None):
    rhoImp = [np.diag(x) for x in rho]
    charge = (rhoImp[0] + rhoImp[1]) / 2
    spin = (rhoImp[0] - rhoImp[1]) / 2
    nscsites = lattice.supercell.nsites
    if names is None:
        names = lattice.supercell.names
        indices = range(nscsites)
    else:
        indices = [lattice.supercell.index(x) for x in names]

    results = []
    lines = ["%-3s   ", "charge", "spin  "]
    atom = names[0].split("_")[0]
    lines[0] = lines[0] % atom
    totalc, totals = 0., 0.
    for i, (name, index) in enumerate(zip(names, indices)):
        if atom != name.split("_")[0]:
            lines[0] += "%10s" % "total"
            lines[1] += "%10.5f" % totalc
            lines[2] += "%10.5f" % totals
            totalc, totals = 0., 0.
            results.append("\n".join(lines))
            lines = ["%-3s   ", "charge", "spin  "]
            atom = name.split("_")[0]
            lines[0] = lines[0] % atom

        lines[0] += "%10s" % name.split("_")[1]
        lines[1] += "%10.5f" % charge[index]
        lines[2] += "%10.5f" % spin[index]
        totalc += charge[index]
        totals += spin[index]

    lines[0] += "%10s" % "total"
    lines[1] += "%10.5f" % totalc
    lines[2] += "%10.5f" % totals
    results.append("\n".join(lines))
    log.result("\n".join(results))

