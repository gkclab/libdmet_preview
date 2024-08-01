#! /usr/bin/env python

"""
Test eri transform for generalized spin orbitals 
with partial particle-hole transform. with MPI.
"""

from mpi4pyscf.tools import mpi
import os, sys
import numpy as np
import scipy.linalg as la

def t_eri_trans_gso():
    """
    Test ERI transform with particle-hole transformation.
    """
    import libdmet.dmet.Hubbard as dmet
    from libdmet.system import lattice
    from libdmet.basis_transform import make_basis
    from libdmet.basis_transform.eri_transform import get_unit_eri, get_emb_eri
    from libdmet.basis_transform import eri_transform_mpi as eri_transform
    from libdmet.routine import spinless
    from libdmet.utils import logger as log
    from libdmet.utils.misc import max_abs, mdot

    import pyscf
    from pyscf import lib, ao2mo
    from pyscf.pbc.lib import chkfile
    from pyscf.pbc import scf, gto, df, dft
    
    np.set_printoptions(4, linewidth=1000, suppress=True)
    log.verbose = "DEBUG2"

    cell = gto.Cell()
    cell.a = ''' 10.0    0.0     0.0
                 0.0     10.0    0.0
                 0.0     0.0     3.0 '''
    cell.atom = ''' H 5.0      5.0      0.0
                    H 5.0      5.0      1.5 '''
    cell.basis = 'sto3g'
    cell.verbose = 4
    cell.precision = 1e-12
    cell.build(unit='Angstrom')

    kmesh = [1, 1, 10]
    Lat = lattice.Lattice(cell, kmesh)
    kpts = Lat.kpts
    nao = nscsites = Lat.nao
    nso = nao * 2
    nkpts = Lat.nkpts
    exxdiv = None

    # system
    Filling = cell.nelectron / float(Lat.nscsites*2.0)
    restricted = True
    bogoliubov = False
    int_bath = False
    use_hcore_as_emb_ham = False
    H2_fname = "emb_eri_slater.h5"
    nscsites = Lat.nscsites
    Mu = 0.0
    last_dmu = 0.0
    beta = np.inf

    # vcor initialization
    vcor = dmet.VcorLocal(restricted, bogoliubov, nscsites)
    z_mat = np.zeros((2, nscsites, nscsites))
    vcor.assign(z_mat)

    log.section("\nSolving SCF mean-field problem\n")
    gdf_fname = 'gdf_ints.h5'
    gdf = df.GDF(cell, kpts)
    gdf._cderi_to_save = gdf_fname
    gdf._cderi = gdf_fname
    if True:
    #if not os.path.isfile(gdf_fname):
        gdf.build()

    chkfname = 'hchain.chk'
    if False:
    #if os.path.isfile(chkfname):
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
        assert(kmf.converged)

    log.section("\nPre-process, orbital localization and subspace partition\n")
    C_ao_iao, C_ao_iao_val, C_ao_iao_virt = \
            make_basis.get_C_ao_lo_iao(Lat, kmf, minao='minao', full_return=True)

    ncore = 0
    nval = C_ao_iao_val.shape[-1]
    nvirt = cell.nao_nr() - ncore - nval
    Lat.set_val_virt_core(nval, nvirt, ncore)

    C_ao_lo = C_ao_iao
    Lat.set_Ham(kmf, gdf, C_ao_lo, use_hcore_as_emb_ham=use_hcore_as_emb_ham)

    rho, Mu, res = dmet.RHartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)

    smf = scf.KRHF(Lat.bigcell, exxdiv=None).density_fit()
    Cgamma = np.einsum('Rk, kui, Sk -> RuSi', Lat.phase, C_ao_lo, \
            Lat.phase.conj(), optimize=True)
    Cgamma = Cgamma.reshape(nao*nkpts, nao*nkpts)
    H2_ref = smf.with_df.ao2mo(Cgamma, compact=False).reshape([nao*nkpts]*4)
    assert max_abs(H2_ref.imag) < 1e-10
    H2_ref = H2_ref.real
    
    GRho_k = spinless.transform_rdm1_k(res["rho_k"])
    GRho = Lat.k2R(GRho_k)
    basis = spinless.embBasis(Lat, GRho, local=True)
    
    # random to mimic the pairing part of bath
    basis[1:, :, nao*2:] += np.random.random(basis[1:, :, nao*2:].shape) * 0.1

    nbasis = basis.shape[-1]
    basis_Ra, basis_Rb = spinless.separate_basis(basis)
    basis_full = np.asarray((basis_Ra, basis_Rb)).reshape(nkpts*nso, nso*2)
    R_a, R_b = basis_Ra.reshape(nkpts*nao, nso*2), \
               basis_Rb.reshape(nkpts*nao, nso*2)
    basis_k = Lat.R2k_basis(basis) 
    basis_ka, basis_kb = spinless.separate_basis(basis_k)
    basis_spin = np.asarray((basis_ka, basis_kb))
    
    # supercell reference
    GH2_ref, _, _ = spinless.transform_H2_local(H2_ref)
    GH2_ref = spinless.combine_H2(GH2_ref)
    GH2_emb_ref = ao2mo.kernel(GH2_ref, basis_full, compact=False)
    
    # using GDF fast transform
    eri = eri_transform.get_emb_eri_gso(cell, gdf._cderi, gdf.kpts, C_ao_lo=C_ao_lo, 
                                        basis=basis, t_reversal_symm=True,
                                        symmetry=1)
    
    diff_eri = max_abs(eri - GH2_emb_ref)
    print ("diff bewteen transformed and molecular ref: ", diff_eri)
    assert diff_eri < 1e-10

if __name__ == "__main__":
    t_eri_trans_gso()
