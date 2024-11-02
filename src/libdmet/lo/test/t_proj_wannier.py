#! /usr/bin/env python

"""
Test projections.
"""

def test_proj_CH4():
    """
    CH4 minimal basis set 9 bands -> 9 wannier (Cs, 4Csp3, 4Hs)
    """
    import os
    import numpy as np

    from pyscf.lib import chkfile
    from pyscf.pbc import gto, scf, df

    from libdmet.system import lattice
    from libdmet.lo import pywannier90
    from libdmet.utils import plot

    cell = gto.Cell()
    cell.atom = '''
     C                  3.17500000    3.17500000    3.17500000
     H                  2.54626556    2.54626556    2.54626556
     H                  3.80373444    3.80373444    2.54626556
     H                  2.54626556    3.80373444    3.80373444
     H                  3.80373444    2.54626556    3.80373444
    '''
    cell.basis = 'sto3g'
    cell.a = np.eye(3) * 6.35
    cell.precision = 1e-13
    cell.verbose = 5
    cell.build()

    kmesh = [1, 1, 1]
    Lat = lattice.Lattice(cell, kmesh)
    nao = Lat.nao
    nkpts = Lat.nkpts
    kpts = Lat.kpts

    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    if not os.path.isfile(gdf_fname):
    #if True:
        gdf.build()

    chkfname = 'CH4.chk'
    if os.path.isfile(chkfname):
    #if False:
        kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        kmf.kernel()

    num_wann = 9
    keywords = \
    '''
    num_iter = 200
    begin projections
    C:l=0;sp3
    H:l=0
    end projections
    '''

    # wannier run
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords=keywords)
    w90.make_win()
    w90.setup()
    A_mat = w90.get_A_mat()
    C_mo_lo = A_mat.transpose(2, 0, 1)
    C_ao_mo = kmf.mo_coeff
    C_ao_lo = np.zeros((nkpts, nao, num_wann), dtype=C_mo_lo.dtype)
    for k in range(nkpts):
        C_ao_lo[k] = np.dot(C_ao_mo[k], C_mo_lo[k])
    plot.plot_orb_k_all(cell, "CH4", C_ao_lo, kpts, nx=80, ny=80, nz=80)

def test_proj_HChain():
    """
    CH4 minimal basis set 9 bands -> 9 wannier (Cs, 4Csp3, 4Hs)
    """
    import os
    import numpy as np

    from pyscf.lib import chkfile
    from pyscf.pbc import gto, scf, df

    from libdmet.system import lattice
    from libdmet.lo import pywannier90
    from libdmet.utils import plot

    cell = lattice.HChain()
    cell.basis = '321g'
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 3]
    Lat = lattice.Lattice(cell, kmesh)
    nao = Lat.nao
    nkpts = Lat.nkpts
    kpts = Lat.kpts

    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    if not os.path.isfile(gdf_fname):
    #if True:
        gdf.build()

    chkfname = 'CH4.chk'
    if os.path.isfile(chkfname):
    #if False:
        kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=None).density_fit()
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        kmf.kernel()

    num_wann = 4
    keywords = \
    '''
    num_iter = 200
    begin projections
    H:sp
    end projections
    '''

    # wannier run
    w90 = pywannier90.W90(kmf, kmesh, num_wann, other_keywords=keywords)
    w90.make_win()
    w90.setup()
    A_mat = w90.get_A_mat()
    C_mo_lo = A_mat.transpose(2, 0, 1)
    C_ao_mo = kmf.mo_coeff
    C_ao_lo = np.zeros((nkpts, nao, num_wann), dtype=C_mo_lo.dtype)
    for k in range(nkpts):
        C_ao_lo[k] = np.dot(C_ao_mo[k], C_mo_lo[k])
    plot.plot_orb_k_all(cell, "HChain", C_ao_lo, kpts, nx=80, ny=80, nz=80)

if __name__ == "__main__":
    test_proj_HChain()
    test_proj_CH4()
