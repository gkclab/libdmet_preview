#!/usr/bin/env python

"""
Test transform GDF.
"""

def compare_h5(fname1, fname2, tol=1e-10):
    import h5py
    import numpy as np
    from libdmet.utils.misc import max_abs

    res = True
    f1 = h5py.File(fname1, 'r')
    f2 = h5py.File(fname2, 'r')
    if f1.keys() != f2.keys():
        print ("keys are different")
        res = False
    else:
        #kptij_1 = np.asarray(f1["j3c-kptij"])
        #kptij_2 = np.asarray(f2["j3c-kptij"])
        #
        #if max_abs(kptij_1 - kptij_2) > tol:
        #    print ("kptij different: ", max_abs(kptij_1 - kptij_2))
        #    res = False

        for k in f1['j3c'].keys():
            for i in f1['j3c'][k].keys():
                eri_1 = np.asarray(f1['j3c'][k][i])
                eri_2 = np.asarray(f2['j3c'][k][i])
                if max_abs(eri_1 - eri_2) > tol:
                    print ("Lpq different: ", max_abs(eri_1 - eri_2))
                    res = False
                    break
            else: # break nested loops
                continue
            break
    f1.close()
    f2.close()
    return res

def test_transform_gdf():
    import os
    import numpy as np
    import scipy.linalg as la
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df

    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.basis_transform import eri_transform
    from libdmet.routine.pbc_helper import get_eri_7d
    from libdmet.utils.misc import max_abs
    from libdmet.utils import logger as log

    log.verbose = "DEBUG2"
    np.set_printoptions(3, linewidth=1000, suppress=True)

    cell = gto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''
    cell.atom = ''' H 5.0      5.0      0.75
                    H 5.0      5.0      2.25 '''
    cell.basis = '321g'
    cell.verbose = 4
    cell.precision = 1e-10
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 3]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts

    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    #if not os.path.isfile(gdf_fname):
    if True:
        gdf.build()

    chkfname = 'hchain.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = scf.KRHF(cell, kpts, exxdiv=None)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-10
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=None)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-10
        kmf.chkfile = chkfname
        kmf.kernel()

    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, \
            kmf, minao='minao', full_return=True)
    C_ao_lo = C_ao_iao

    # 1. compare using time reversal or not
    eri_transform.transform_gdf_to_lo(gdf, C_ao_lo, fname="no_tr_symm.h5", \
            t_reversal_symm=False)
    eri_transform.transform_gdf_to_lo(gdf, C_ao_lo, fname="tr_symm.h5", \
            t_reversal_symm=True)
    assert compare_h5("tr_symm.h5", "no_tr_symm.h5")

    # 2. compare LO based 7d ERI
    eri_7d_ref = gdf.ao2mo_7d(C_ao_lo)

    gdf_lo_fname = 'tr_symm.h5'
    gdf_lo = df.GDF(cell, kpts)
    gdf_lo._cderi_to_save = gdf_lo_fname
    gdf_lo._cderi = gdf_lo_fname
    eri_7d = get_eri_7d(cell, gdf_lo)

    diff_7d_eri = max_abs(eri_7d - eri_7d_ref)
    print ("diff 7d ERI: ", diff_7d_eri)
    assert diff_7d_eri < 1e-10

    # 3. compare LO based HF energy and rdm1
    hcore = kmf.get_hcore()
    ovlp = kmf.get_ovlp()
    rdm1 = kmf.make_rdm1()
    hcore_lo = make_basis.transform_h1_to_lo(hcore, C_ao_lo)
    ovlp_lo = make_basis.transform_h1_to_lo(ovlp, C_ao_lo)
    rdm1_lo = make_basis.transform_rdm1_to_lo(rdm1, C_ao_lo, ovlp)

    kmf_lo = scf.KRHF(cell, kpts, exxdiv=None)
    kmf_lo.get_hcore = lambda *args: hcore_lo
    kmf_lo.get_ovlp  = lambda *args: ovlp_lo
    kmf_lo.with_df = gdf_lo
    kmf_lo.with_df._cderi = gdf_lo_fname
    kmf_lo.conv_tol = 1e-10
    kmf_lo.kernel(rdm1_lo)

    E_mf_ref = kmf.e_tot
    E_mf = kmf_lo.e_tot
    assert abs(E_mf - E_mf_ref) < 1e-10

    rdm1_ao = make_basis.transform_rdm1_to_ao(kmf_lo.make_rdm1(), C_ao_lo)
    assert max_abs(rdm1_ao - rdm1) < 1e-10

if __name__ == "__main__":
    test_transform_gdf()
