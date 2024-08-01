#!/usr/bin/env python

'''
Test ERI transformation from k-space to real space. 
Analytical Fourier transform density fitting.
'''

import numpy as np
import scipy.linalg as la
import os, sys

from pyscf import lib
from pyscf.pbc.lib import chkfile
from pyscf.pbc import scf, gto, df

from libdmet.lo import iao, pywannier90
from libdmet.system import lattice
from libdmet.basis_transform import make_basis
from libdmet.basis_transform import eri_transform
from libdmet.utils.misc import mdot, max_abs

from libdmet.utils import logger as log
log.verbose = "DEBUG2"

def _test_ERI(cell, mydf, kpts, C_ao_lo):
    # Fast ERI
    # Give a very small memory to force small blksize (for test only)
    eri_k2gamma = eri_transform.get_emb_eri(cell, mydf, C_ao_lo=C_ao_lo, 
                                            max_memory=2000, symmetry=1)
    eri_k2gamma = eri_k2gamma[0]

    scell, phase = eri_transform.get_phase(cell, kpts)
    mydf_scell = df.AFTDF(scell)
    nao = cell.nao_nr()
    nkpts = len(kpts)
    smf = scf.KRHF(scell,exxdiv=None)
    smf.with_df = mydf_scell
    Cgamma = np.einsum('Rk,kui,Sk->RuSi',phase,C_ao_lo,phase.conj())
    Cgamma = Cgamma.reshape(nao*nkpts,nao*nkpts)
    eri_scell = smf.with_df.ao2mo(Cgamma,compact=False).reshape([nao*nkpts]*4)
    print ('ERI Max Imag', max_abs(eri_scell.imag))
    print ('Fast ERI compared to supcell', max_abs(eri_scell-eri_k2gamma))
    assert max_abs(eri_scell-eri_k2gamma) < 1e-8

    # Fast 1st unit cell ERI (for DMFT)
    eri_k2gamma = eri_transform.get_unit_eri(cell, mydf, C_ao_lo=C_ao_lo, 
                                             symmetry=1)
    eri_k2gamma = eri_k2gamma[0]
    print ('Fast 1st unit cell ERI compared to supcell', 
           max_abs(eri_scell[:nao,:nao,:nao,:nao]-eri_k2gamma))
    assert max_abs(eri_scell[:nao,:nao,:nao,:nao]-eri_k2gamma) < 1e-8

def test_ERI_aft():
    cell = gto.Cell()
    cell.a = ''' 5.0    0.0     0.0
                 0.0     5.0    0.0
                 0.0     0.0     3.0 '''
    cell.atom = ''' H 2.5      2.5      0.75
                    H 2.5      2.5      2.25 '''
    cell.basis = 'minao'
    cell.verbose = 4
    cell.precision = 1e-5
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 1]
    kpts = cell.make_kpts(kmesh)
    nkpts = len(kpts)
    nao = cell.nao_nr()

    mydf = df.AFTDF(cell, kpts)

#    print '### Test AO ERI transform ###'
#    C_ao_lo = np.zeros((nkpts, nao, nao),dtype=complex)
#    C_ao_lo[:, range(nao), range(nao)] = 1.0
#    _test_ERI(cell, mydf, kpts, C_ao_lo)

    print ('### Test Lowdin AO ERI transform ###')
    ovlp = np.asarray(cell.pbc_intor('cint1e_ovlp_sph',kpts=kpts))
    X = np.zeros_like(ovlp)
    for ki in range(nkpts):
        X[ki] = la.inv(la.sqrtm(ovlp[ki]))
    C_ao_lo = X.copy()
    _test_ERI(cell, mydf, kpts, C_ao_lo)

#    print ('### Test IAO+PAO ERI transform ###')
#    chkfname = 'hchain.chk'
#    #if os.path.isfile(chkfname):
#    if False:
#        kmf = scf.KRHF(cell, kpts, exxdiv=None)
#        kmf.with_df = mydf
#        kmf.conv_tol = 1e-4
#        kmf.max_cycle = 100
#        data = chkfile.load(chkfname, 'scf')
#        kmf.__dict__.update(data)
#    else:
#        kmf = scf.KRHF(cell, kpts, exxdiv=None)
#        kmf.with_df = mydf
#        kmf.conv_tol = 1e-4
#        kmf.max_cycle = 100
#        kmf.chkfile = chkfname
#        kmf.kernel()
#
#    # IAO
#    Lat = lattice.Lattice(cell, kmesh)
#    C_ao_lo = make_basis.get_C_ao_lo_iao(Lat, kmf, full_return=False)
#    _test_ERI(cell, mydf, kpts, C_ao_lo)

if __name__ == "__main__":
    test_ERI_aft()
