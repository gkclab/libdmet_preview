#! /usr/bin/env python

import os
import numpy as np
import scipy.linalg as la

def test_martin_basis():
    from pyscf import gto as molgto
    from pyscf import scf as molscf
    from pyscf import mp as molmp
    from pyscf import ao2mo
    from pyscf import lib
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, dft, gto, df, cc, tools, mp
    from pyscf.pbc.df import df_ao2mo

    from libdmet.utils.misc import max_abs, mdot
    from libdmet.basis_transform import make_basis

    np.set_printoptions(3, linewidth=1000, suppress=False)
    
    eps = 1e-7
    nbands = 5
    kpt_idx = 1 
    NH = 6
    R  = 1.5
    nkz = 11
    exxdiv = 'ewald'
    cell_prec = 1e-12
    kmf_conv_tol = 1e-10
    kmf_max_cycle = 300

    cell = gto.Cell()
    cell.unit  = 'Bohr'
    cell.a     = np.diag([30, 30, NH*R])
    cell.atom  = [['H',(0.0, 0.0, (i-(NH-1)*0.5)*R)] for i in range(NH)]
    #cell.basis = 'aug-ccpvdz'
    cell.basis = 'ccpvdz'
    cell.pseudo  = None
    cell.precision = cell_prec
    cell.verbose = 4
    cell.spin = 0 
    cell.build()

    kpts_scaled_z = np.linspace(0, 0.5, nkz)
    kpts_scaled = np.array([[0.0, 0.0, kz] for kz in kpts_scaled_z]) 
    kpts_abs = np.array(cell.get_abs_kpts(kpts_scaled))
    kpts = kpts_abs[kpt_idx][np.newaxis]
    nkpts = len(kpts)

    nu = (cell.nelectron + cell.spin)//2
    nd = (cell.nelectron + cell.spin)//2
    nb   = cell.nao_nr()
    nbph = nb//NH

    gdf_fname = 'H%s_k%.2f_gdf_ints.h5'%(NH, kpts_scaled_z[kpt_idx])
    chkfname = 'H%s_k%.2f.chk'%(NH, kpts_scaled_z[kpt_idx])

    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    #if not os.path.isfile(gdf_fname):
    if True:
        gdf.build()

    #if os.path.isfile(chkfname):
    if False:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf = molscf.addons.remove_linear_dep_(kmf, threshold=eps)
        kmf.conv_tol = kmf_conv_tol
        kmf.max_cycle = kmf_max_cycle
        data = chkfile.load(chkfname, 'scf')
        kmf.__dict__.update(data)
        hcore = np.load('hcore.npy')
        ovlp = np.load('ovlp.npy')
        kmf.get_hcore = lambda *args: hcore
        kmf.get_ovlp = lambda *args: ovlp
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv)
        kmf.with_df = gdf
        kmf.with_df._cderi = gdf_fname
        kmf = molscf.addons.remove_linear_dep_(kmf, threshold=eps)
        kmf.conv_tol = kmf_conv_tol
        kmf.max_cycle = kmf_max_cycle
        kmf.chkfile = chkfname
        if not os.path.isfile('hcore.npy'):
            hcore = kmf.get_hcore()
        else:
            hcore = np.load('hcore.npy')
        
        if not os.path.isfile('ovlp.npy'):
            ovlp = kmf.get_ovlp()
        else:
            ovlp = np.load('ovlp.npy')

        np.save('hcore.npy', hcore)
        np.save('ovlp.npy', ovlp)
        kmf.get_hcore = lambda *args: hcore
        kmf.get_ovlp = lambda *args: ovlp
        kmf.kernel()


    kmf.mo_coeff = np.asarray(kmf.mo_coeff)
    kmf.mo_occ = np.asarray(kmf.mo_occ)
    fock = kmf.get_fock(h1e=hcore, s1e=ovlp)

    nao, nmo = np.array(kmf.mo_coeff).shape[-2:]
    print ("***************************************")
    print ("nao: ", nao)
    print ("nmo: ", nmo)
    print ("nu: ", nu)
    print ("nd: ", nd)
    print ("***************************************")


    print ("***************************************")
    print ("KMP2 calculation")
    nocc = int(np.round(np.sum(kmf.mo_occ))) // 2
    mo_energy_old = np.copy(kmf.mo_energy)
    madelung = tools.madelung(cell, kpts)
    energy_shift = -0.5*madelung*float(cell.tot_electrons(nkpts)) / nkpts
    from pyscf.pbc.cc.ccsd import _adjust_occ
    kmf.mo_energy = [_adjust_occ(mo_e, nocc, madelung)
            for k, mo_e in enumerate(kmf.mo_energy)]
    mypt = mp.KMP2(kmf)
    e_kmp2, _ = mypt.kernel()
    rdm1_kmp2_mo = np.array(mypt.make_rdm1())[0]

    print ("KMP2 energy (per unit cell) =", mypt.e_tot)
    kmf.mo_energy = mo_energy_old
    print ("***************************************")

    inv, sgn = make_basis.detect_inv_sym(cell)
    C_ao_rao = make_basis.build_Martin_basis(inv, sgn, ovlp[0])

    hcore_rao = mdot(C_ao_rao.conj().T, hcore[0], C_ao_rao)
    ovlp_rao = mdot(C_ao_rao.conj().T, ovlp[0], C_ao_rao)
    fock_rao = mdot(C_ao_rao.conj().T, fock[0], C_ao_rao)

    print ("RAO Imaginary:")
    print ("hcore:", max_abs(hcore_rao.imag))
    print ("ovlp:", max_abs(ovlp_rao.imag))
    print ("fock:", max_abs(fock_rao.imag))
    print ("") 
    assert max_abs(hcore_rao.imag) < 1e-7
    assert max_abs(ovlp_rao.imag) < 1e-7
    assert max_abs(fock_rao.imag) < 1e-7
    hcore_rao = hcore_rao.real
    ovlp_rao = ovlp_rao.real
    fock_rao = fock_rao.real

    mo_energy, C_rao_rmo = kmf.eig(fock_rao[None], ovlp_rao[None])
    mo_occ = kmf.get_occ(mo_energy, C_rao_rmo)
    C_rao_rmo = C_rao_rmo[0]
    C_ao_rmo = C_ao_rao.dot(C_rao_rmo)
    C_ao_rmo_occ = C_ao_rmo[:, :nocc]
    C_ao_rmo_virt = C_ao_rmo[:, nocc:]
    nmo = C_rao_rmo.shape[-1]
    nvirt = nmo - nocc
    C_mo_rmo = mdot(kmf.mo_coeff[0].conj().T, kmf.get_ovlp()[0], C_ao_rmo)

    hcore_rmo = mdot(C_ao_rmo.conj().T, hcore[0], C_ao_rmo)
    ovlp_rmo = mdot(C_ao_rmo.conj().T, ovlp[0], C_ao_rmo)
    fock_rmo = mdot(C_ao_rmo.conj().T, fock[0], C_ao_rmo)
    eri_rmo = df_ao2mo.general(gdf, C_ao_rmo, kpts=kpts)
    eri_ovov = df_ao2mo.general(gdf, (C_ao_rmo_occ, C_ao_rmo_virt, C_ao_rmo_occ, C_ao_rmo_virt), kpts=kpts, compact=False)
    eri_rmo = df_ao2mo.general(gdf, C_ao_rmo, kpts=kpts)

    print ("RMO nmo: ", nmo)
    print ("nocc: ", nocc)
    print ("nvirt: ", nvirt)
    print ("hcore:", max_abs(hcore_rmo.imag))
    print ("ovlp:", max_abs(ovlp_rmo.imag))
    print ("fock:", max_abs(fock_rmo.imag))
    print ("eri_ovov: ", max_abs(eri_ovov.imag))
    print ("eri:", max_abs(eri_rmo.imag))
    print ("")

    hcore_rmo = hcore_rmo.real
    ovlp_rmo = ovlp_rmo.real
    fock_rmo = fock_rmo.real
    eri_rmo = eri_rmo.real
    eri_ovov = eri_ovov.real#.reshape((nocc, nvirt, nocc, nvirt))
    print ("eri_ovov shape", eri_ovov.shape)
    mo_energy, C_rmo_rmo = la.eigh(fock_rmo)

    mol = molgto.M()
    mol.verbose=4
    mol.nelectron = cell.nelectron
    mf = molscf.RHF(mol)
    mf.energy_nuc = lambda *args: cell.energy_nuc() + energy_shift
    mf.get_hcore = lambda *args: hcore_rmo
    mf.get_ovlp = lambda *args: ovlp_rmo
    mf._eri = eri_rmo
    #mf._eri = None

    dm0 = np.zeros(hcore_rmo.shape[-1])
    dm0[:cell.nelectron//2] = 2.0
    dm0 = np.diag(dm0)
    mf.kernel(dm0)

    print (kmf.e_tot)
    print (mf.e_tot)
    print ("mean-field energy diff")
    print (mf.e_tot - kmf.e_tot)
    assert abs(mf.e_tot - kmf.e_tot) < 1e-8

    mypt_mol = molmp.MP2(mf)
    e_molmp2_ref, _ = mypt_mol.kernel()


    mf.mo_energy = mo_energy
    mf.mo_coeff = C_rmo_rmo
    mf.mo_occ = mf.get_occ()

    mf.mo_energy = _adjust_occ(mf.mo_energy, nocc, madelung)

    mypt_mol = molmp.MP2(mf)
    from pyscf.mp import mp2 as molmp2
    eris = molmp2._ChemistsERIs(mypt_mol)
    eris.ovov = eri_ovov

    e_molmp2, t2_molmp2 = molmp2.kernel(mypt_mol, \
            mo_energy=mf.mo_energy, eris=eris)
    rdm1_molmp2 = molmp2.make_rdm1(mypt_mol, t2=t2_molmp2, eris=eris, \
            ao_repr=False)

    print ("MP2 energy from real orbitals")
    print (e_molmp2)
    print ("Energy diff")
    print (e_molmp2 - e_kmp2)
    print (e_molmp2_ref - e_kmp2)
    assert abs(e_molmp2 - e_kmp2) < 1e-6
    assert abs(e_molmp2_ref - e_kmp2) < 1e-6

    rdm1_molmp2_mo = mdot(C_mo_rmo, rdm1_molmp2, C_mo_rmo.conj().T)

    print ("MP2 rdm1 diff from KMP2", max_abs(rdm1_molmp2_mo - rdm1_kmp2_mo))
    assert max_abs(rdm1_molmp2_mo - rdm1_kmp2_mo) < 1e-6

if __name__ == "__main__":
    test_martin_basis()
