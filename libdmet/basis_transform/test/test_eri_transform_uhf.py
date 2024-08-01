#!/usr/bin/env python

'''
Test ERI transformation with GDF and UHF integral.
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

log.verbose = 'DEBUG2'

def _test_ERI(cell, gdf, kpts, C_ao_lo):
    # Fast ERI
    # Give a very small memory to force small blksize (for test only)
    eri_k2gamma = eri_transform.get_emb_eri(cell, gdf, C_ao_lo=C_ao_lo,
                                            max_memory=0.01, symmetry=4)

    # outcore routine
    eri_transform.ERI_SLICE = 2
    eri_outcore = eri_transform.get_emb_eri(cell, gdf, C_ao_lo=C_ao_lo,
                                            max_memory=0.02, t_reversal_symm=True,
                                            incore=False)
    diff_outcore = max_abs(np.asarray(eri_outcore["ccdd"])
                           - eri_k2gamma[[0, 2, 1]])
    print ("outcore difference")
    print (diff_outcore)
    assert diff_outcore < 1e-12

    if C_ao_lo.ndim == 3:
        C_ao_lo = C_ao_lo[np.newaxis]
    scell, phase = eri_transform.get_phase(cell, kpts)
    mydf_scell = df.GDF(scell)
    nao = cell.nao_nr()
    nkpts = len(kpts)
    spin = C_ao_lo.shape[0]
    Cgamma = np.empty((spin,nao*nkpts,nao*nkpts), dtype=np.complex128)
    for s in range(spin):
        Cgamma[s] = np.einsum('Rk,kui,Sk->RuSi', phase,C_ao_lo[s],phase.conj()).reshape(nao*nkpts,nao*nkpts)
    Cgamma = Cgamma.real
    eri_scell = mydf_scell.get_eri()
    if spin == 1:
        eri_full = ao2mo.restore(1,ao2mo.full(eri_scell,Cgamma[0]),nao*nkpts)[np.newaxis]
    else:
        eri_full = np.empty((3,nao*nkpts,nao*nkpts,nao*nkpts,nao*nkpts))
        eri_full[0] = ao2mo.restore(1,ao2mo.full(eri_scell,Cgamma[0]),nao*nkpts)
        eri_full[1] = ao2mo.general(eri_scell, (Cgamma[0],Cgamma[0],Cgamma[1],Cgamma[1]), compact=False).reshape([nao*nkpts]*4)
        eri_full[2] = ao2mo.restore(1,ao2mo.full(eri_scell,Cgamma[1]),nao*nkpts)

        eri_k2gamma_s1 = np.empty((3, nao*nkpts,nao*nkpts,nao*nkpts, nao*nkpts))
        eri_k2gamma_s1[0] = ao2mo.restore(1, eri_k2gamma[0], nao*nkpts)
        eri_k2gamma_s1[1] = ao2mo.restore(1, eri_k2gamma[1], nao*nkpts)
        eri_k2gamma_s1[2] = ao2mo.restore(1, eri_k2gamma[2], nao*nkpts)
    diff_full = max_abs(eri_full - eri_k2gamma_s1)
    print ('Fast ERI compared to supcell', diff_full)
    assert diff_full < 1e-7

    # Fast 1st unit cell ERI (for DMFT)
    eri_k2gamma = eri_transform.get_unit_eri(cell, gdf, C_ao_lo=C_ao_lo,
                                             t_reversal_symm=True, symmetry=1)
    eri_k2gamma_no_tr = eri_transform.get_unit_eri(cell, gdf, C_ao_lo=C_ao_lo,
                                                   t_reversal_symm=False,
                                                   symmetry=1)
    print ('Fast 1st unit cell ERI compared to supcell', \
            max_abs(eri_full[:,:nao,:nao,:nao,:nao] - eri_k2gamma))
    assert la.norm(eri_full[:,:nao,:nao,:nao,:nao] - eri_k2gamma) < 1e-7
    assert la.norm(eri_k2gamma_no_tr - eri_k2gamma) < 1e-10

def test_ERI_gdf_uhf():
    cell = gto.M(
        unit = 'B',
        a = [[ 0.,          6.74027466,  6.74027466],
             [ 6.74027466,  0.,          6.74027466],
             [ 6.74027466,  6.74027466,  0.        ]],
        atom = '''H 0 0 0
                  H 1.68506866 1.68506866 1.68506866
                  H 3.37013733 3.37013733 3.37013733''',
        basis = '321g',
        verbose = 1,
        precision = 1e-14,
        charge = 0,
        spin = 1,
    )

    kmesh = [1, 1, 3]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts
    cell.spin = cell.spin*nkpts
    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    #if not os.path.isfile(gdf_fname):
    if True:
        gdf.build()

#    print ('### Test AO ERI transform ###')
#    C_ao_lo = np.zeros((nkpts, nao, nao),dtype=complex)
#    C_ao_lo[:, range(nao), range(nao)] = 1.+0.j
#    _test_ERI(cell, gdf, kpts, C_ao_lo)
#
#    print ('### Test Lowdin AO ERI transform ###')
#    ovlp = np.asarray(cell.pbc_intor('cint1e_ovlp_sph',kpts=kpts))
#    X = np.zeros_like(ovlp)
#    for ki in range(nkpts):
#        X[ki] = la.inv(la.sqrtm(ovlp[ki]))
#    C_ao_lo = X.copy()
#    _test_ERI(cell, gdf, kpts, C_ao_lo)

    print ('### Test IAO+PAO ERI transform ###')
    chkfname = 'hchain.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = scf.KUHF(cell, kpts, exxdiv=None)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-14
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KUHF(cell, kpts, exxdiv=None)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-14
        kmf.chkfile = chkfname
        kmf.kernel()

    C_ao_lo = make_basis.get_C_ao_lo_iao(Lat, kmf, \
            minao='minao', full_return=False)
    _test_ERI(cell, gdf, kpts, C_ao_lo)

if __name__ == "__main__":
    test_ERI_gdf_uhf()
