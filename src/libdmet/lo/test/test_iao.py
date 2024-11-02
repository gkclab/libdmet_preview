#! /usr/bin/env python

def test_iao_virt():
    """
    Test IAO virtual.
    """
    import os
    import numpy as np
    from pyscf.pbc import scf, gto, df, dft
    from pyscf.lib import chkfile
    from libdmet.system import lattice
    from libdmet.lo import iao
    from libdmet.lo import check_orthonormal, check_orthogonal
    from libdmet.utils import max_abs, plot_orb_k_all

    from libdmet.utils import logger as log
    log.verbose = "DEBUG2"

    #np.set_printoptions(3, linewidth=1000, suppress=False)

    cell = gto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''
    cell.atom = ''' H 5.0      5.0      0.75
                    H 5.0      5.0      2.25 '''
    cell.basis = '321g'
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 3]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = Lat.nao
    nkpts = Lat.nkpts
    exxdiv = None

    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    #if not os.path.isfile(gdf_fname):
    if True:
        gdf.build()

    chkfname = 'hchain.chk'
    #if os.path.isfile(chkfname):
    if False:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = 'gdf_ints.h5'
        kmf.conv_tol = 1e-12
        kmf.max_cycle = 300
        kmf.chkfile = chkfname
        kmf.kernel()

    mo_coeff = np.array(kmf.mo_coeff)
    mo_occ   = np.array(kmf.mo_occ)
    ovlp     = np.array(kmf.get_ovlp())
    C_ao_iao_g = iao.iao(cell, mo_coeff[0, :, :1], minao='minao', kpts=None)
    C_ao_iao_1 = iao.iao(cell, mo_coeff[:, :, :1], minao='minao', kpts=kpts)
    C_ao_iao_2 = iao.iao(cell, mo_coeff[:, :, :1], minao='minao', kpts=kpts,
            mo_coeff_B1=mo_coeff)

    assert max_abs(C_ao_iao_1[0] - C_ao_iao_g) < 1e-12
    assert max_abs(C_ao_iao_1 - C_ao_iao_2) < 1e-12
    C_ao_iao_1 = iao.vec_lowdin(C_ao_iao_1, ovlp)

    C_ao_pao_1 = iao.get_iao_virt(cell, C_ao_iao_1, ovlp, minao='minao', full_virt=False, \
        pmol=None, max_ovlp=False)
    C_ao_pao_1 = iao.vec_lowdin(C_ao_pao_1, ovlp)

    C_ao_pao_2 = iao.get_iao_virt(cell, C_ao_iao_1, ovlp, minao='minao', full_virt=False, \
        pmol=None, max_ovlp=True)
    C_ao_pao_2 = iao.vec_lowdin(C_ao_pao_2, ovlp)

    assert check_orthonormal(C_ao_iao_1, ovlp)
    assert check_orthonormal(C_ao_pao_1, ovlp)
    assert check_orthonormal(C_ao_pao_2, ovlp)

    assert check_orthogonal(C_ao_iao_1, C_ao_pao_1, ovlp)
    assert check_orthogonal(C_ao_iao_1, C_ao_pao_2, ovlp)

    C_ao_iao_1_occ = iao.iao(cell, mo_coeff, minao='minao', kpts=kpts,
            mo_occ=mo_occ*0.5, tol=1e-10)
    C_ao_iao_1_occ_orth = iao.vec_lowdin(C_ao_iao_1_occ, ovlp)

    print (C_ao_iao_1_occ_orth)
    print (C_ao_iao_1)
    print (max_abs(C_ao_iao_1_occ_orth - C_ao_iao_1))

    # test projected with reference AO
    latt_vec = cell.lattice_vectors()
    latt_vec[0,0] = 3.0
    latt_vec[1,1] = 3.0
    C_ao_lo = iao.proj_ref_ao(cell, minao='minao', kpts=kpts)
    C_ao_lo, labels = iao.proj_ref_ao(cell, minao='minao', kpts=kpts, return_labels=True)
    print (labels)
    #plot_orb_k_all(cell, 'iao_val', C_ao_lo, kpts, nx=50, ny=50, \
    #        nz=50, resolution=None, margin=5.0, latt_vec=latt_vec, \
    #        boxorig=[5.0, 5.0, -3.0])

    #from libdmet.basis_transform import transform_rdm1_to_lo
    #rdm1_lo = transform_rdm1_to_lo(kmf.make_rdm1(), C_ao_iao_1_occ_orth, ovlp)
    #print (Lat.k2R(rdm1_lo))
    #rdm1_lo = transform_rdm1_to_lo(kmf.make_rdm1(), C_ao_iao_1, ovlp)
    #print (Lat.k2R(rdm1_lo))

    #print (max_abs(C_ao_iao_1 - C_ao_iao_1_occ))
    ##assert max_abs(C_ao_iao_1 - C_ao_iao_1_occ) < 1e-12

    #from libdmet.routine.mfd import assignocc
    #from libdmet.routine.ftsystem import fermi_smearing_occ, gaussian_smearing_occ
    #
    #mo_occ = assignocc(kmf.mo_energy, cell.nelectron * nkpts * 0.5, 100.0, 0.0, \
    #        f_occ=fermi_smearing_occ)[0]
    #C_ao_iao_1_occ = iao.iao(cell, mo_coeff, minao='minao', kpts=kpts,
    #        mo_occ=mo_occ, tol=1e-12)

    #print (max_abs(C_ao_iao_1 - C_ao_iao_1_occ))
    #print (C_ao_iao_1_occ)
    #print (C_ao_iao_1)
    #print (mo_occ)

    #latt_vec = cell.lattice_vectors()
    #latt_vec[0,0] = 3.0
    #latt_vec[1,1] = 3.0
    #plot_orb_k_all(cell, 'iao_val', C_ao_iao_1_occ, kpts, nx=50, ny=50, \
    #        nz=50, resolution=None, margin=5.0, latt_vec=latt_vec, \
    #        boxorig=[5.0, 5.0, -3.0])
    #exit()

    #latt_vec = cell.lattice_vectors()
    #latt_vec[0,0] = 3.0
    #latt_vec[1,1] = 3.0
    #plot_orb_k_all(cell, 'iao_val', C_ao_iao_1, kpts, nx=50, ny=50, \
    #        nz=50, resolution=None, margin=5.0, latt_vec=latt_vec, \
    #        boxorig=[5.0, 5.0, -3.0])
    #plot_orb_k_all(cell, 'iao_virt', C_ao_pao_2, kpts, nx=50, ny=50, \
    #        nz=50, resolution=None, margin=5.0, latt_vec=latt_vec, \
    #        boxorig=[5.0, 5.0, -3.0])

def test_get_idx_each():
    import os
    import numpy as np
    from libdmet.utils import read_poscar, max_abs
    from libdmet.lo import iao

    pos_file = os.path.dirname(os.path.realpath(__file__)) + "/HBCO.pos"
    cell = read_poscar(pos_file)
    cell.basis = 'def2-svp'
    cell.ecp = 'def2-svp'
    cell.build()
    labels = np.asarray(cell.ao_labels())

    dic_atom = iao.get_idx_each(labels=labels, kind='atom')

    dic_al = iao.get_idx_each(labels=labels, kind='atom l')

    dic_ial = iao.get_idx_each(labels=labels, kind='id atom l')

    dic_anl = iao.get_idx_each(labels=labels, kind='atom nl')

    Hg_ref = np.asarray(cell.search_ao_label("Hg"))
    assert max_abs(dic_atom["Hg"] - Hg_ref) < 1e-10

    Cu1_ref = np.asarray(cell.search_ao_label("Cu1"))
    assert max_abs(dic_atom["Cu1"] - Cu1_ref) < 1e-10

    Cu1_d_ref = np.asarray(cell.search_ao_label("Cu1 .d"))
    assert max_abs(dic_al["Cu1 d"] - Cu1_d_ref) < 1e-10

    Cu1_4d_ref = np.asarray(cell.search_ao_label("Cu1 4d"))
    assert max_abs(dic_anl["Cu1 4d"] - Cu1_4d_ref) < 1e-10

def test_build_pmol_core_val():
    """
    Test build pmol_core_val.
    """
    import os
    import numpy as np
    from libdmet.lo import iao
    from pyscf.pbc import gto

    basis = os.path.dirname(os.path.realpath(__file__)) + "/cc-pvdz"
    minao = os.path.dirname(os.path.realpath(__file__)) + "/ccpvdz-atom-iao"
    basis_core = os.path.dirname(os.path.realpath(__file__)) + "/ccpvdz-atom-iao-core"
    basis_val = os.path.dirname(os.path.realpath(__file__)) + "/ccpvdz-atom-iao-val"

    cell = gto.Cell()
    cell.a = np.eye(3) * 10
    cell.atom = '''Cu 0.0 0.0 0.0
                    O 0.0 0.0 1.5'''
    cell.basis = basis
    cell.pseudo = {"Cu": 'gth-pbe-q19', "O": 'gth-pbe'}
    cell.build()

    pmol_core, pmol_val = iao.build_pmol_core_val(cell, basis_core, basis_val)

    core_labels = pmol_core.ao_labels()
    print ("core labels")
    print (core_labels)

    val_labels = pmol_val.ao_labels()
    print ("valence labels")
    print (val_labels)
    assert val_labels[0] == "0 Cu 4s    "
    assert val_labels[-1] == "1 O 2pz   "
    assert len(val_labels) == 12

if __name__ == "__main__":
    test_build_pmol_core_val()
    test_get_idx_each()
    test_iao_virt()
