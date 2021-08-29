#! /usr/bin/env python

from libdmet.dmet.HubbardBCS import *
from libdmet.dmet.abinitio import buildUnitCell, buildLattice, \
        read_integral, buildHamiltonian, AFInitGuessIdx, AFInitGuessOrbs
from libdmet.dmet.abinitio import reportOccupation as report

def reportOccupation(lattice, GRho, names = None):
    rhoA, rhoB, kappaBA = extractRdm(GRho)
    report(lattice, np.asarray([rhoA, rhoB]), names)

