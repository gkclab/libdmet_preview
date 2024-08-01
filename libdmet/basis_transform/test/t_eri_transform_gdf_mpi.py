#!/usr/bin/env python

'''
Test ERI transformation from k-space to real space
'''

import numpy as np
import scipy.linalg as la
import os, sys

from pyscf import lib
from pyscf.pbc.lib import chkfile
from pyscf.pbc import scf, gto, df
from pyscf import ao2mo

from libdmet.lo import iao, pywannier90
from libdmet.system import lattice
from libdmet.basis_transform import make_basis
from libdmet.basis_transform import eri_transform
from libdmet.utils.misc import mdot, max_abs
from libdmet.utils import logger as log

import pytest

np.set_printoptions(4, linewidth=1000, suppress=True)
log.verbose = "DEBUG2"

def _test_ERI(cell, gdf, kpts, C_ao_lo):
    # Fast 1st unit cell ERI (for DMFT)
    eri_mpi = eri_transform.get_unit_eri(cell, gdf, C_ao_lo=C_ao_lo, 
                                         t_reversal_symm=True, symmetry=1,
                                         use_mpi=True)
    eri_ref = eri_transform.get_unit_eri(cell, gdf, C_ao_lo=C_ao_lo, 
                                         t_reversal_symm=True, symmetry=1)
    print ('Fast 1st unit cell ERI compared to supcell', 
           max_abs(eri_ref - eri_mpi))
    assert max_abs(eri_ref - eri_mpi) < 1e-10

@pytest.mark.parametrize(
    "kmesh", [[1, 1, 10]]
)
def t_ERI_gdf(kmesh):
    cell = gto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''
    cell.atom = ''' H 5.0      5.0      0.75
                    H 5.0      5.0      2.25 '''
    cell.basis = 'ccpvdz'
    #cell.basis = 'sto3g'
    cell.verbose = 4
    cell.precision = 1e-14
    cell.build(unit='Angstrom')
    nao = cell.nao_nr()

    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi = gdf_fname
    gdf._cderi_to_save = gdf_fname
    
    #if not os.path.isfile(gdf_fname):
    if True:
        gdf.build()
    
    print ('### Test AO ERI transform ###')
    C_ao_lo = np.zeros((nkpts, nao, nao),dtype=complex)
    C_ao_lo[:, range(nao), range(nao)] = 1.0
    _test_ERI(cell, gdf, kpts, C_ao_lo)
    
if __name__ == "__main__":
    from mpi4pyscf.tools import mpi
    t_ERI_gdf(kmesh=[1, 1, 10])
