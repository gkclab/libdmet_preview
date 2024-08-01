#! /usr/bin/env python

"""
Test for mfd with mpi.
"""

import numpy as np
import scipy.linalg as la
import pytest

np.set_printoptions(3, linewidth=1000, suppress=True)

def test_find_kpairs():
    try:
        from pyscf.pbc import gto
        from libdmet.routine import mfd_mpi
        cell = gto.Cell()
        cell.a = np.eye(3)
        cell.atom = 'H 0.0 0.0 0.0'
        cell.build()
        kpts = cell.make_kpts([4, 4, 3], wrap_around=True)
        kpairs, kidx = mfd_mpi.get_kpairs_kidx(cell, kpts)
        print (kpairs)
        print (kidx)
        assert kpairs[-3] == (29, 34)
    except ImportError:
        pass

def test_ghf_mpi():
    try:
        from mpi4pyscf.tools import mpi
        from libdmet.utils import logger as log
        import libdmet.dmet.HubbardGSO as dmet
        from libdmet.system.hamiltonian import HamNonInt

        np.set_printoptions(3, linewidth=1000, suppress=False)
        log.verbose = "DEBUG2"
        
        U = 6.0 
        LatSize = [16, 16]
        ImpSize = [2, 2]
        Filling = 1.0 / 2

        Mu = U * Filling
        beta = 1000.0

        Lat = dmet.SquareLattice(*(LatSize+ImpSize))
        nao = nscsites = Lat.supercell.nsites
        nso = nao * 2
        
        Ham = dmet.Ham(Lat, U)
        Lat.setHam(Ham, use_hcore_as_emb_ham=False)
        H1 = Lat.getH1(kspace=True)
        Lat.hcore_lo_k = np.asarray((H1, H1, np.zeros_like(H1)))
        Lat.fock_lo_k = np.asarray((H1, H1, np.zeros_like(H1)))

        vcor = dmet.AFInitGuess(ImpSize, U, Filling, rand=0.001)
        
        GRho, Mu, ires = dmet.GHartreeFock(Lat, vcor, None, mu0_elec=Mu, 
                                           beta=beta, fix_mu=False, mu0=None, use_mpi=True,
                                           full_return=True)
    except ImportError:
        pass

if __name__ == "__main__":
    test_ghf_mpi()
    test_find_kpairs()

