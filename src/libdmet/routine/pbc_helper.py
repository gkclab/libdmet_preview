#! /usr/bin/env python

"""
Helper functions for pyscf PBC module.
Including KRHF, KUHF, KGHF, get_eri, get_jk,
conversion between H and GH,
kpoint-symmetry converter, vca potential.

Author:
    Zhihao Cui <zhcui0408@gmail.com>
"""

import copy
import h5py
import numpy as np
from scipy import linalg as la

from pyscf import lib
from pyscf import scf
from pyscf import ao2mo
from pyscf import pbc
from pyscf.pbc import gto as pbcgto
from pyscf.pbc.lib.kpts_helper import get_kconserv
from pyscf.lib import logger as pyscflogger
from pyscf.scf import ghf as mol_ghf
from pyscf.pbc.scf import khf
from pyscf.pbc.df.df_jk import _format_jks
from pyscf.pbc.lib.kpts_helper import (is_zero, gamma_point, member)

from libdmet.solver import scf as scf_helper
from libdmet.utils.misc import mdot, max_abs, add_spin_dim, Iterable
from libdmet.settings import IMAG_DISCARD_TOL
from libdmet.utils import logger as log
from libdmet.routine.pdft_helper import *
from libdmet.routine import kgks

flush = log.flush_for_pyscf
pyscflogger.flush = flush([""])

# ****************************************************************************
# k-point symmetry wrapper.
# ****************************************************************************

def kmf_symm_(kmf_symm):
    """
    Addon function for kmf_symm.
    so that kmf_symm can be used as if using a normal kmf w/o symmetry.
    """
    kpts_symm = kmf_symm.kpts
    if isinstance(kpts_symm, np.ndarray):
        return kmf_symm

    kmf = kmf_symm.to_khf()

    def get_hcore(cell=None, kpts=None):
        kpts = kpts_symm
        hcore_ibz = kmf_symm.get_hcore(cell, kpts)
        hcore_bz = kpts_symm.transform_fock(hcore_ibz)
        return hcore_bz

    def get_ovlp(cell=None, kpts=None):
        kpts = kpts_symm
        ovlp_ibz = kmf_symm.get_ovlp(cell, kpts)
        ovlp_bz = kpts_symm.transform_fock(ovlp_ibz)
        return ovlp_bz

    def get_jk(cell=None, dm_kpts=None, hermi=1, kpts=None, kpts_band=None,
               with_j=True, with_k=True, omega=None, **kwargs):
        if dm_kpts is None:
            dm_kpts = np.asarray(kmf.make_rdm1())
        if dm_kpts.ndim == 3:
            dm_kpts = dm_kpts[kpts_symm.ibz2bz]
        else:
            dm_kpts = dm_kpts[:, kpts_symm.ibz2bz]
        kpts = kpts_symm
        vj_ibz, vk_ibz = kmf_symm.get_jk(cell=cell, dm_kpts=dm_kpts, hermi=hermi,
                                         kpts=kpts, kpts_band=kpts_band,
                                         with_j=with_j, with_k=with_k,
                                         omega=omega, **kwargs)
        vj_bz = kpts_symm.transform_fock(vj_ibz)
        if vk_ibz is None:
            vk_bz = None
        else:
            vk_bz = kpts_symm.transform_fock(vk_ibz)
        return vj_bz, vk_bz

    def get_init_guess(cell=None, key='minao'):
        dm0 = kmf_symm.get_init_guess(cell, key)
        dm0 = kpts_symm.transform_fock(dm0)
        return dm0

    def energy_elec(dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
        if dm_kpts is None:
            dm_kpts = np.asarray(kmf.make_rdm1())
        if dm_kpts.ndim == 3:
            dm_kpts = dm_kpts[kpts_symm.ibz2bz]
        else:
            dm_kpts = dm_kpts[:, kpts_symm.ibz2bz]
        if h1e_kpts is not None:
            h1e_kpts = h1e_kpts[kpts_symm.ibz2bz]
        if vhf_kpts is not None:
            vhf_kpts = np.asarray(vhf_kpts)
            if vhf_kpts.ndim == 3:
                vhf_kpts = vhf_kpts[kpts_symm.ibz2bz]
            else:
                vhf_kpts = vhf_kpts[:, kpts_symm.ibz2bz]
        return kmf_symm.energy_elec(dm_kpts, h1e_kpts, vhf_kpts)

    # save the symmetry version of functions
    kmf.get_hcore_symm = kmf_symm.get_hcore
    kmf.get_ovlp_symm  = kmf_symm.get_ovlp
    kmf.get_jk_symm    = kmf_symm.get_jk
    kmf.get_init_guess_symm = kmf_symm.get_init_guess
    kmf.energy_elec_symm = kmf_symm.energy_elec

    kmf.get_hcore = get_hcore
    kmf.get_ovlp  = get_ovlp
    kmf.get_jk    = get_jk
    kmf.get_init_guess = get_init_guess
    kmf.energy_elec = energy_elec

    kmf.e_tot = kmf_symm.e_tot
    kmf.converged = kmf_symm.converged
    kmf.kpts = kmf.with_df.kpts = kpts_symm.kpts

    return kmf

# ****************************************************************************
# virtual crystal approximation
# ****************************************************************************

def get_veff_vca(mydf, atom_idx, occ, kpts_symm=None):
    """
    Get effective potential from virtual crystal approximation (VCA).

    Args:
        mydf: GDF object contains all atoms including the doped atoms.
        atom_idx: indices of doped atoms.
        occ: the site occupancy of the doped atoms.
             Now only support a float,
             i.e. all doped atom share the same occupancy.
        kpts_symm: a KPoints object, to allow kpoints symmetry.

    Returns:
        veff: shape (nkpts, nao, nao)
    """
    log.info("Get hcore from virtual crystal approximation (VCA).")
    cell = mydf.cell.copy()
    kpts = mydf.kpts
    nkpts = len(kpts)

    if isinstance(occ, Iterable):
        raise NotImplementedError

    charges_old = np.array(cell.atom_charges())[atom_idx]
    atom_doped = [cell.atom_symbol(i) for i in atom_idx]
    charge_doped = charges_old * (occ - 1)

    log.info("%5s %5s %15s %15s", "idx", "atom", "occ", "doped charge")
    for i, atm, chg in zip(atom_idx, atom_doped, charge_doped):
        log.info("%5s %5s %15.8g %15.8g", i, atm, occ, chg)
    log.info("-" * 79)
    log.info("%27s %15.8g", "total (per cell)", np.sum(charge_doped))
    log.info("%27s %15.8g", "total  (lattice)", np.sum(charge_doped) * nkpts)

    # ghost cell
    cell_ghost = pbcgto.Cell()
    a = np.array(cell.lattice_vectors(), copy=True)

    atoms = copy.deepcopy(cell._atom)
    for i, atm in enumerate(atoms):
        if not (i in atom_idx):
            # if atom is not the doped atom, use ghost atom instead
            atoms[i] = ('X-' + atm[0], atm[1])

    pseudo = copy.deepcopy(cell._pseudo)
    pseudo_ghost = {}
    for key, val in pseudo.items():
        if key in atom_doped:
            # only keep the pseudo for non-ghost atoms
            pseudo_ghost[key] = val

    basis = copy.deepcopy(cell._basis)
    basis_ghost = {}
    for key, val in basis.items():
        # all bases should be kept
        if key in atom_doped:
            basis_ghost[key] = val
        else:
            basis_ghost['X-' + key] = val

    cell_ghost.a = a
    cell_ghost.atom = atoms
    cell_ghost.basis = basis_ghost
    cell_ghost.pseudo = pseudo_ghost
    cell_ghost.spin = cell.spin
    cell_ghost.max_memory = cell.max_memory
    cell_ghost.precision = cell.precision
    cell_ghost.verbose = 0
    cell_ghost.build(unit='B', dump_input=False)

    log.debug(0, "atom:\n%s", str(cell_ghost._atom))
    log.debug(0, "basis:\n%s", str(cell_ghost._basis.keys()))
    log.debug(0, "pseudo:\n%s", str(cell_ghost._pseudo.keys()))

    mydf_ghost = mydf.__class__(cell_ghost, mydf.kpts)
    mydf_ghost._cderi = mydf._cderi
    mydf_ghost._cderi_to_save = mydf._cderi_to_save

    if kpts_symm is not None:
        kpts = kpts_symm.kpts_ibz
    if cell_ghost.pseudo:
        vnuc = np.asarray(mydf_ghost.get_pp(kpts))
    else:
        vnuc = np.asarray(mydf_ghost.get_nuc(kpts))

    if len(cell_ghost._ecpbas) > 0:
        raise NotImplementedError
        vnuc += lib.asarray(ecp.ecp_int(cell_ghost, kpts))

    vnuc *= (occ - 1.0)
    return  vnuc

# ****************************************************************************
# KRHF, KUHF
# get_jk for 7d, local and nearest ERI.
# ****************************************************************************

def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
    '''
    Following pyscf.scf.hf.energy_elec()
    Support h1e with spin.
    '''
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)

    h1e_kpts = np.asarray(h1e_kpts)
    if h1e_kpts.ndim == 3:
        h1e_kpts = np.asarray((h1e_kpts, h1e_kpts))
    nkpts = h1e_kpts.shape[-3]
    e1 = 1./nkpts * np.einsum('kij,kji', dm_kpts[0], h1e_kpts[0], optimize=True)
    e1+= 1./nkpts * np.einsum('kij,kji', dm_kpts[1], h1e_kpts[1], optimize=True)
    e_coul = 1./nkpts * np.einsum('kij,kji', dm_kpts[0], vhf_kpts[0], optimize=True) * 0.5
    e_coul+= 1./nkpts * np.einsum('kij,kji', dm_kpts[1], vhf_kpts[1], optimize=True) * 0.5
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e_coul.real
    pyscflogger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
    if abs(e_coul.imag > mf.cell.precision*10):
        log.warn("Coulomb energy has imaginary part %s. "
                    "Coulomb integrals (e-e, e-N) may not converge !",
                    e_coul.imag)
    return (e1+e_coul).real, e_coul.real

class KUHF(pbc.scf.kuhf.KUHF):
    """
    KUHF that supports hcore with spin.
    """
    def init_guess_by_1e(self, cell=None):
        """
        Support h1e with spin dimension.
        """
        if cell is None: cell = self.cell
        h1e = np.asarray(self.get_hcore(cell))
        s1e = self.get_ovlp(cell)
        if h1e.ndim == 3:
            mo_energy, mo_coeff = self.eig((h1e, h1e), s1e)
        else:
            mo_energy, mo_coeff = self.eig(h1e, s1e)
        mo_occ = self.get_occ(mo_energy, mo_coeff)
        dma, dmb = self.make_rdm1(mo_coeff, mo_occ)
        return np.array((dma, dmb))

    energy_elec = energy_elec

def get_eri_7d(cell, xdf, kpts=None, compact=False):
    """
    Get eri 7d in ao basis.
    """
    nao = cell.nao_nr()
    if kpts is None:
        kpts = xdf.kpts
    nkpts = len(kpts)
    eri_7d = np.zeros((nkpts, nkpts, nkpts, nao, nao, nao, nao),
                      dtype=np.complex128)
    kconserv = get_kconserv(cell, kpts)
    for i, kpti in enumerate(kpts):
        for j, kptj in enumerate(kpts):
            for k, kptk in enumerate(kpts):
                l = kconserv[i, j, k]
                kptl = kpts[l]
                eri_7d[i, j, k] = xdf.get_eri((kpti, kptj, kptk, kptl),
                                              compact=compact).reshape((nao,)*4)
    return eri_7d

def get_eri_8d(cell, xdf, kpts=None, compact=False):
    """
    Get eri 8d in ao basis.
    """
    nao = cell.nao_nr()
    if kpts is None:
        kpts = xdf.kpts
    nkpts = len(kpts)
    eri_8d = np.zeros((nkpts, nkpts, nkpts, nkpts, nao, nao, nao, nao),
                      dtype=np.complex128)
    for i, kpti in enumerate(kpts):
        for j, kptj in enumerate(kpts):
            for k, kptk in enumerate(kpts):
                for l, kptl in enumerate(kpts):
                    eri_8d[i, j, k, l] = xdf.get_eri((kpti, kptj, kptk, kptl),
                                                     compact=compact).reshape((nao,)*4)
    return eri_8d

def get_jk_from_eri_7d(eri, dm, with_j=True, with_k=True):
    """
    Get J, K matrix in kpts.
    Assume eri is spinless, 7d.
    """
    eri = np.asarray(eri)
    dm = np.asarray(dm)
    old_shape = dm.shape
    if dm.ndim == 3:
        dm = dm[None]
    spin, nkpts, nao, _ = dm.shape

    if with_j:
        vj = np.zeros((spin, nkpts, nao, nao),
                      dtype=np.result_type(dm.dtype, eri.dtype))
        if with_k: # J and K
            vk = np.zeros((spin, nkpts, nao, nao),
                          dtype=np.result_type(dm.dtype, eri.dtype))
            for s in range(spin):
                for k in range(nkpts):
                    vj[s] += lib.einsum('Rpqrs, qp -> Rrs', eri[k, k], dm[s, k])
                    vk[s] += lib.einsum('Ppqrs, qr -> Pps', eri[:, k, k], dm[s, k])
            vj /= float(nkpts)
            vk /= float(nkpts)
            vj = vj.reshape(old_shape)
            vk = vk.reshape(old_shape)
        else: # only J
            vk = None
            for s in range(spin):
                for k in range(nkpts):
                    vj[s] += lib.einsum('Rpqrs, qp -> Rrs', eri[k, k], dm[s, k])
            vj /= float(nkpts)
            vj = vj.reshape(old_shape)
    else:
        if with_k: # only K
            vj = None
            vk = np.zeros((spin, nkpts, nao, nao),
                          dtype=np.result_type(dm.dtype, eri.dtype))
            for s in range(spin):
                for k in range(nkpts):
                    vk[s] += lib.einsum('Ppqrs, qr -> Pps', eri[:, k, k], dm[s, k])
            vk /= float(nkpts)
            vk = vk.reshape(old_shape)
        else: # no J no K
            vj = vk = None
    return vj, vk

def get_j_from_eri_7d(eri, dm):
    """
    Get J matrix in kpts.
    Assume eri is spinless, 7d.
    """
    return get_jk_from_eri_7d(eri, dm, with_j=True, with_k=False)[0]

def get_k_from_eri_7d(eri, dm):
    """
    Get K matrix in kpts.
    Assume eri is spinless, 7d.
    """
    return get_jk_from_eri_7d(eri, dm, with_j=False, with_k=True)[1]

def get_jk_from_eri_local(eri, dm, eri_symm=4, with_j=True, with_k=True):
    """
    Get J, K matrix in kpts.
    Assume eri is local, spinless.
    """
    dm = np.asarray(dm)
    old_shape = dm.shape
    if dm.ndim == 3:
        dm = dm[None]
    spin, nkpts, nao, _ = dm.shape
    if eri.size == nao**4:
        eri = ao2mo.restore(eri_symm, eri, nao)

    dm_ave = dm.sum(axis=-3) / float(nkpts)
    if max_abs(dm_ave.imag) < IMAG_DISCARD_TOL:
        dm_ave = dm_ave.real
    else:
        log.warn("get_*_from_eri_local assume a real dm_ave, "
                 "now imag = %s", max_abs(dm_ave.imag))

    if with_j:
        vj = np.zeros((spin, nkpts, nao, nao),
                      dtype=np.result_type(dm.dtype, eri.dtype))
        if with_k: # J and K
            vk = np.zeros((spin, nkpts, nao, nao),
                          dtype=np.result_type(dm.dtype, eri.dtype))
            for s in range(spin):
                vj[s], vk[s] = scf.hf.dot_eri_dm(eri, dm_ave[s], hermi=1,
                                                 with_j=with_j, with_k=with_k)
            vj = vj.reshape(old_shape)
            vk = vk.reshape(old_shape)
        else: # only J
            vk = None
            for s in range(spin):
                vj[s], _ = scf.hf.dot_eri_dm(eri, dm_ave[s], hermi=1,
                                             with_j=with_j, with_k=with_k)
            vj = vj.reshape(old_shape)
    else:
        if with_k: # only K
            vj = None
            vk = np.zeros((spin, nkpts, nao, nao),
                          dtype=np.result_type(dm.dtype, eri.dtype))
            for s in range(spin):
                _, vk[s] = scf.hf.dot_eri_dm(eri, dm_ave[s], hermi=1,
                                             with_j=with_j, with_k=with_k)
            vk = vk.reshape(old_shape)
        else: # no J no K
            vj = vk = None
    return vj, vk

def get_j_from_eri_local(eri, dm, eri_symm=4):
    """
    Get J matrix in kpts.
    Assume eri is local, spinless.
    """
    return get_jk_from_eri_local(eri, dm, eri_symm=eri_symm,
                                 with_j=True, with_k=False)[0]

def get_k_from_eri_local(eri, dm, eri_symm=4):
    """
    Get K matrix in kpts.
    Assume eri is local, spinless.
    """
    return get_jk_from_eri_local(eri, dm, eri_symm=eri_symm,
                                 with_j=False, with_k=True)[1]

def get_jk_from_eri_nearest(eri, dm, lattice, with_j=True, with_k=True):
    """
    Get J, K matrix in kpts.
    Assume eri is nearest, spinless, in real space representation,
    shape (nR, nao, nao, nao, nao)
    dm is in kspace.
    """
    dm = np.asarray(dm)
    old_shape = dm.shape
    if dm.ndim == 3:
        dm = dm[None]
    spin, nkpts, nao, _ = dm.shape

    dm_R = lattice.k2R(dm)
    if max_abs(dm_R.imag) < IMAG_DISCARD_TOL:
        dm_R = dm_R.real
    else:
        log.warn("get_*_from_eri_nearest assume a real dm_R, "
                 "now imag = %s", max_abs(dm_R.imag))
    eri_ave = eri.sum(axis=-5)

    if with_j:
        vj = np.zeros((spin, nkpts, nao, nao), dtype=np.complex128)
        if with_k: # J and K
            vk_R = np.zeros((spin, nkpts, nao, nao))
            for s in range(spin):
                vj[s] = lib.einsum('qp, pqrs -> rs', dm_R[s, 0], eri_ave)
                for R in range(nkpts):
                    vk_R[s, R] = lib.einsum('pqrs, qr -> ps', eri[R], dm_R[s, R])
            vk = lattice.R2k(vk_R)
            vj = vj.reshape(old_shape)
            vk = vk.reshape(old_shape)
        else: # only J
            vk = None
            for s in range(spin):
                vj[s] = lib.einsum('qp, pqrs -> rs', dm_R[s, 0], eri_ave)
            vj = vj.reshape(old_shape)
    else:
        if with_k: # only K
            vj = None
            vk_R = np.zeros((spin, nkpts, nao, nao))
            for s in range(spin):
                vj[s] = lib.einsum('qp, pqrs -> rs', dm_R[s, 0], eri_ave)
                for R in range(nkpts):
                    vk_R[s, R] = lib.einsum('pqrs, qr -> ps', eri[R], dm_R[s, R])
            vk = lattice.R2k(vk_R)
            vk = vk.reshape(old_shape)
        else: # no J no K
            vj = vk = None
    return vj, vk

def get_j_from_eri_nearest(eri, dm, lattice):
    """
    Get J matrix in kpts.
    Assume eri is nearest, spinless.
    """
    return get_jk_from_eri_nearest(eri, dm, lattice, with_j=True,
                                   with_k=False)[0]

def get_k_from_eri_nearest(eri, dm, lattice):
    """
    Get K matrix in kpts.
    Assume eri is nearest, spinless.
    """
    return get_jk_from_eri_nearest(eri, dm, lattice, with_j=False,
                                   with_k=True)[1]

def eri_to_gdf(eri, kpts, fname="local_eri.h5", **kwargs):
    """
    Convert a local ERI to GDF form.

    Args:
        eri: local eri, (nao, nao, nao, nao)
        kpts: k-points
        fname: file to store gdf file

    Kwargs:
        kwargs passing to cholesky.

    Returns:
        fname
    """
    from libdmet.utils import cholesky
    assert eri.ndim == 4
    assert not np.iscomplexobj(eri)
    nkpts = len(kpts)
    norb = eri.shape[-1]
    cderi = cholesky.cholesky(eri, **kwargs).kernel()
    naux = cderi.shape[0]

    kptij_lst = [(ki, kpts[j]) for i, ki in enumerate(kpts) for j in range(i+1)]
    kptij_lst = np.asarray(kptij_lst)

    dataname = 'j3c'
    nkptij = len(kptij_lst)

    feri = h5py.File(fname, 'w')
    feri[dataname + '-kptij'] = kptij_lst

    # ZHC NOTE only store the independent part
    feri["cderi"] = cderi.astype(complex)
    feri["cderi_s2"] = lib.pack_tril(cderi.reshape(-1, norb, norb)).astype(complex)
    feri["cderi_s4"] = lib.pack_tril(cderi.reshape(-1, norb, norb))

    for k in range(nkptij):
        kpti, kptj = kptij_lst[k]
        is_real = gamma_point(kptij_lst[k])
        aosym_ks2 = gamma_point(kpti - kptj)
        i, j = member(kpti, kpts)[0], member(kptj, kpts)[0]

        if is_real:
            feri['%s/%d/%d' % (dataname, k, 0)] = h5py.SoftLink('/cderi_s4')
        elif aosym_ks2:
            feri['%s/%d/%d' % (dataname, k, 0)] = h5py.SoftLink('/cderi_s2')
        else:
            feri['%s/%d/%d' % (dataname, k, 0)] = h5py.SoftLink('/cderi')

    feri.close()
    return fname

# ****************************************************************************
# KGHF
# ****************************************************************************

def get_jk_from_eri_local_ghf(eri, dm):
    """
    Get J, K matrix in kpts.
    Assume eri is local, has spin dimension 3.
    dm has shape (nso, nso)

    Args:
        eri: (3,) + nao*4 or nao_pair*2, aa, bb, ab
        dm: (nkpts, nso, nso)

    Returns:
        vj: (nkpts, nso, nso)
        vk: (nkpts, nso, nso)
    """
    dm = np.asarray(dm)
    nkpts, nso, _ = dm.shape
    nao = nso // 2
    eri = np.asarray(eri)
    assert eri.shape[0] == 3

    vj = np.zeros((nkpts, nso, nso), dtype=np.result_type(dm.dtype, eri.dtype))
    vk = np.zeros((nkpts, nso, nso), dtype=np.result_type(dm.dtype, eri.dtype))
    dm_ave = dm.sum(axis=0) / float(nkpts)
    if max_abs(dm_ave.imag) < IMAG_DISCARD_TOL:
        dm_ave = dm_ave.real
    else:
        log.warn("get_*_from_eri_local assume a real dm_ave, "
                 "now imag = %s", max_abs(dm_ave.imag))
    vj[:], vk[:] = scf_helper._get_jk_ghf(dm_ave, eri)
    return vj, vk

def energy_elec_ghf(mf, dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
    '''
    Calculate electronic energy for GHF.

    Args:
        dm_kpts:  (nkpts, nso, nso)
        h1e_kpts: (nkpts, nso, nso)
        vhf_kpts: (nkpts, nso, nso)

    Note:
        No Mu contribution.
    '''
    if dm_kpts  is None: dm_kpts = mf.make_rdm1()
    if h1e_kpts is None: h1e_kpts = mf.get_hcore()
    if vhf_kpts is None: vhf_kpts = mf.get_veff(mf.cell, dm_kpts)

    ovlp = mf.get_ovlp()
    h1e_kpts = np.array(h1e_kpts, copy=True) # copy for subrtact Mu
    nkpts, nso, _ = h1e_kpts.shape
    nao = nso // 2
    if getattr(mf, "Mu", None) is not None:
        h1e_kpts[:, :nao, :nao] += ovlp[:, :nao, :nao] * mf.Mu
        h1e_kpts[:, nao:, nao:] -= ovlp[:, nao:, nao:] * mf.Mu

    e1 = 1./nkpts * np.einsum('kij, kji', dm_kpts, h1e_kpts, optimize=True)
    e_coul = 1./nkpts * np.einsum('kij, kji', dm_kpts, vhf_kpts, optimize=True) * 0.5
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['e2'] = e_coul.real
    pyscflogger.debug(mf, 'E1 = %s  E_coul = %s', e1, e_coul)
    if abs(e_coul.imag > mf.cell.precision*10):
        log.warn(mf, "Coulomb energy has imaginary part %s. "
                    "Coulomb integrals (e-e, e-N) may not converge !",
                    e_coul.imag)
    return (e1+e_coul).real, e_coul.real

class KGHF(pbc.scf.kghf.KGHF):
    """
    KGHF.
    """
    energy_elec = energy_elec_ghf

def smearing_(mf, sigma=None, method='fermi', mu0=None, tol=1e-12,
              fit_spin=False, fix_mu=False):
    """
    Fermi-Dirac or Gaussian smearing.
    This version support Sz for UHF smearing.

    Args:
        mf: kmf object.
        sigma: smearing parameter, ~ 1/beta, unit in Hartree.
        method: fermi or gaussian
        mu0: initial mu
        tol: tolerance for fitting nelec
        fit_spin: if True, will fit each spin channel seprately.
        fix_mu: fix the mu to be mu0.

    Returns:
        mf: modified mf object.
    """
    from pyscf.scf import uhf
    from pyscf.pbc.scf import uhf as pbcuhf
    from pyscf.pbc.scf import kuhf
    from pyscf.scf import ghf
    from pyscf.pbc.scf import ghf as pbcghf
    from pyscf.pbc.scf import kghf
    from libdmet.routine import mfd, ftsystem
    mf_class = mf.__class__
    # ZHC FIXME support istype in the future.
    is_uhf = isinstance(mf, uhf.UHF) or isinstance(mf, pbcuhf.UHF) or isinstance(mf, kuhf.KUHF)
    is_ghf = isinstance(mf, ghf.GHF) or isinstance(mf, pbcghf.GHF) or isinstance(mf, kghf.KGHF)
    is_rhf = (not is_uhf) and (not is_ghf)
    is_khf = isinstance(mf, khf.KSCF)
    if hasattr(mf, "cell"):
        Sz = mf.cell.spin
    else:
        Sz = mf.mol.spin

    def partition_occ(mo_occ, mo_energy_kpts):
        mo_occ_kpts = []
        p1 = 0
        for e in mo_energy_kpts:
            p0, p1 = p1, p1 + e.size
            occ = mo_occ[p0:p1]
            mo_occ_kpts.append(occ)
        return mo_occ_kpts

    def get_occ(mo_energy_kpts=None, mo_coeff_kpts=None):
        """
        Label the occupancies for each orbital for sampled k-points.

        This is a k-point version of scf.hf.SCF.get_occ
        """
        if hasattr(mf, 'kpts') and getattr(mf.kpts, 'kpts_ibz', None) is not None:
            mo_energy_kpts = mf.kpts.transform_mo_energy(mo_energy_kpts)
        mo_occ_kpts = mf_class.get_occ(mf, mo_energy_kpts, mo_coeff_kpts)
        if (mf.sigma == 0) or (not mf.sigma) or (not mf.smearing_method):
            return mo_occ_kpts

        if is_khf:
            nkpts = getattr(mf.kpts, 'nkpts', None)
            if nkpts is None:
                nkpts = len(mf.kpts)
        else:
            nkpts = 1

        # find nelec_target
        if isinstance(mf.mol, pbcgto.Cell):
            nelectron = mf.mol.tot_electrons(nkpts)
        else:
            nelectron = mf.mol.tot_electrons()
        if is_uhf:
            if fit_spin:
                nelec_target = [(nelectron + Sz) * 0.5, (nelectron - Sz) * 0.5]
            else:
                nelec_target = nelectron
        elif is_ghf:
            nelec_target = nelectron
        else:
            nelec_target = nelectron * 0.5

        if mf.smearing_method.lower() == 'fermi': # Fermi-Dirac smearing
            f_occ = ftsystem.fermi_smearing_occ
        else: # Gaussian smearing
            f_occ = ftsystem.gaussian_smearing_occ

        # check whether the mo_energy shapes are matched
        matched = True
        if is_khf:
            if is_uhf:
                shape0 = mo_energy_kpts[0][0].shape
                for s in range(2):
                    for k in range(nkpts):
                        if (mo_energy_kpts[s][k].shape != shape0):
                            matched = False
                            break
            else:
                shape0 = mo_energy_kpts[0].shape
                for k in range(nkpts):
                    if (mo_energy_kpts[k].shape != shape0):
                        matched = False
                        break

        if matched:
            mo_energy = np.asarray(mo_energy_kpts)
        else:
            if is_uhf:
                mo_energy = np.append(np.hstack(mo_energy_kpts[0]),
                                      np.hstack(mo_energy_kpts[1]))
            else:
                mo_energy = np.hstack(mo_energy_kpts)

        # ZHC NOTE tol should not be too small.
        fit_tol = max(min(mf.conv_tol * 0.1, tol), 1e-15)

        mo_occ, mf.mu, nerr = mfd.assignocc(mo_energy, nelec_target, 1.0/mf.sigma,
                                            mf.mu, fit_tol=fit_tol, f_occ=f_occ,
                                            fix_mu=fix_mu)
        mo_occ = mo_occ.reshape(mo_energy.shape)

        # See https://www.vasp.at/vasp-workshop/slides/k-points.pdf
        if mf.smearing_method.lower() == 'fermi':
            f = mo_occ[(mo_occ>0) & (mo_occ<1)]
            mf.entropy = -(f*np.log(f) + (1-f)*np.log(1-f)).sum() / nkpts
        else:
            if is_uhf and fit_spin:
                mf.entropy = (np.exp(-((mo_energy[0]-mf.mu[0])/mf.sigma)**2).sum()
                              / (2*np.sqrt(np.pi)) / nkpts) + \
                             (np.exp(-((mo_energy[1]-mf.mu[1])/mf.sigma)**2).sum()
                              / (2*np.sqrt(np.pi)) / nkpts)
            else:
                mf.entropy = (np.exp(-((mo_energy-mf.mu)/mf.sigma)**2).sum()
                              / (2*np.sqrt(np.pi)) / nkpts)
        if is_rhf:
            mo_occ *= 2
            mf.entropy *= 2

        nelec_now = mo_occ.sum()
        logger.debug(mf, '    Fermi level %s  Sum mo_occ_kpts = %s  should equal nelec = %s',
                     mf.mu, nelec_now, nelectron)
        if (not fix_mu) and abs(nelec_now - nelectron) > fit_tol * 100:
            logger.warn(mf, "Occupancy (nelec_now %s) is not equal to cell.nelectron (%s).",
                        nelec_now, nelectron)
        logger.info(mf, '    sigma = %g  Optimized mu = %s  entropy = %.12g',
                    mf.sigma, mf.mu, mf.entropy)

        # ZHC NOTE different k points may have different mo_coeff shapes
        if not matched:
            if is_uhf:
                nao_tot = mo_occ.size // 2
                mo_occ = (partition_occ(mo_occ[:nao_tot], mo_energy_kpts[0]),
                          partition_occ(mo_occ[nao_tot:], mo_energy_kpts[1]))
            else:
                mo_occ = partition_occ(mo_occ, mo_energy_kpts)

        if hasattr(mf, 'kpts') and getattr(mf.kpts, 'kpts_ibz', None) is not None:
            if is_uhf:
                mo_occ = (mf.kpts.check_mo_occ_symmetry(mo_occ[0]),
                          mf.kpts.check_mo_occ_symmetry(mo_occ[1]))
            else:
                mo_occ = mf.kpts.check_mo_occ_symmetry(mo_occ)
        return mo_occ

    def get_grad_tril(mo_coeff_kpts, mo_occ_kpts, fock):
        if is_khf:
            grad_kpts = []
            for k, mo in enumerate(mo_coeff_kpts):
                f_mo = mdot(mo.T.conj(), fock[k], mo)
                nmo = f_mo.shape[0]
                grad_kpts.append(f_mo[np.tril_indices(nmo, -1)])
            return np.hstack(grad_kpts)
        else:
            f_mo = mdot(mo_coeff_kpts.T.conj(), fock, mo_coeff_kpts)
            nmo = f_mo.shape[0]
            return f_mo[np.tril_indices(nmo, -1)]

    def get_grad(mo_coeff_kpts, mo_occ_kpts, fock=None):
        if (mf.sigma == 0) or (not mf.sigma) or (not mf.smearing_method):
            return mf_class.get_grad(mf, mo_coeff_kpts, mo_occ_kpts, fock)
        if fock is None:
            dm1 = mf.make_rdm1(mo_coeff_kpts, mo_occ_kpts)
            fock = mf.get_hcore() + mf.get_veff(mf.mol, dm1)
        if is_uhf:
            ga = get_grad_tril(mo_coeff_kpts[0], mo_occ_kpts[0], fock[0])
            gb = get_grad_tril(mo_coeff_kpts[1], mo_occ_kpts[1], fock[1])
            return np.hstack((ga,gb))
        else: # rhf and ghf
            return get_grad_tril(mo_coeff_kpts, mo_occ_kpts, fock)

    if is_khf:
        def energy_tot(dm_kpts=None, h1e_kpts=None, vhf_kpts=None):
            e_tot = mf.energy_elec(dm_kpts, h1e_kpts, vhf_kpts)[0] + mf.energy_nuc()
            if (mf.sigma and mf.smearing_method and
                mf.entropy is not None):
                mf.e_free = e_tot - mf.sigma * mf.entropy
                mf.e_zero = e_tot - mf.sigma * mf.entropy * .5
                logger.info(mf, '    Total E(T) = %.15g  Free energy = %.15g  E0 = %.15g',
                            e_tot, mf.e_free, mf.e_zero)
            return e_tot
    else:
        def energy_tot(dm=None, h1e=None, vhf=None):
            e_tot = mf.energy_elec(dm, h1e, vhf)[0] + mf.energy_nuc()
            if (mf.sigma and mf.smearing_method and
                mf.entropy is not None):
                mf.e_free = e_tot - mf.sigma * mf.entropy
                mf.e_zero = e_tot - mf.sigma * mf.entropy * .5
                logger.info(mf, '    Total E(T) = %.15g  Free energy = %.15g  E0 = %.15g',
                            e_tot, mf.e_free, mf.e_zero)
            return e_tot

    mf.sigma = sigma
    mf.smearing_method = method
    mf.entropy = None
    mf.e_free = None
    mf.e_zero = None
    if mu0 is None:
        mf.mu = 0.0
    else:
        mf.mu = mu0
    mf._keys = mf._keys.union(['sigma', 'smearing_method',
                               'entropy', 'e_free', 'e_zero', 'mu'])

    mf.get_occ = get_occ
    mf.energy_tot = energy_tot
    mf.get_grad = get_grad
    return mf

# ****************************************************************************
# P-H routines
# ****************************************************************************

def H2ABD(H, bare_dim):
    """
    Extract H to HA, HB and HD.

    Args:
        H: ndarray to extract
        bare_dim: int, the dimension of H,
                  if no dimension of spin.

    Returns:
        HA, HB, HD

    Note:
        H is copied.
    """
    if H.ndim == bare_dim:
        HA, HB = np.array(H), np.array(H)
        HD = np.zeros_like(HA)
    elif H.ndim == bare_dim + 1:
        if H.shape[0] == 1:
            HA, HB = np.array(H[0]), np.array(H[0])
            HD = np.zeros_like(HA)
        elif H.shape[0] == 2:
            HA, HB = np.array(H)
            HD = np.zeros_like(HA)
        elif H.shape[0] == 3:
            HA, HB, HD = np.array(H)
        else:
            raise ValueError
    else:
        raise ValueError
    return HA, HB, HD

def combine_H1(H):
    """
    Combine H to get GH.

    Args:
        H: (3, nao, nao), aa, bb, ab order

    Returns:
        GH: (nso, nso)
    """
    assert len(H) == 3
    HA, HB, HD = H
    nao = HA.shape[-1]
    nso = nao * 2
    GH = np.empty((nso, nso), dtype=np.result_type(HA.dtype, HB.dtype, HD.dtype))
    GH[:nao, :nao] = HA
    GH[nao:, nao:] = HB
    GH[:nao, nao:] = HD
    GH[nao:, :nao] = HD.conj().T
    return GH

H2GH = combine_H = combine_H1

combine_rdm1 = combine_H1

def combine_H2(H2):
    """
    Combine H2 to get GH2.

    Args:
        H2: (3, nao, nao, nao, nao) or 4-fold, aa, bb, ab order

    Returns:
        GH2: (nso, nso, nso, nso) or (nso_pair, nso_pair)
    """
    assert len(H2) == 3
    H2A, H2B, H2D = H2
    if H2A.ndim == 4:
        nao = H2A.shape[-1]
        nso = nao * 2
        GH2 = np.zeros((nso,)*4, dtype=np.result_type(H2A.dtype, H2B.dtype, H2D.dtype))
        GH2[:nao, :nao, :nao, :nao] = H2A
        GH2[nao:, nao:, nao:, nao:] = H2B
        GH2[:nao, :nao, nao:, nao:] = H2D
        GH2[nao:, nao:, :nao, :nao] = H2D.transpose(3, 2, 1, 0).conj()
    elif H2A.ndim == 2:
        nao_pair = H2A.shape[-1]
        nao = int(np.sqrt(nao_pair * 2))
        nso = nao * 2
        nso_pair = nso * (nso + 1) // 2
        GH2 = np.zeros((nso_pair, nso_pair),
                       dtype=np.result_type(H2A.dtype, H2B.dtype, H2D.dtype))
        tril_idx = np.tril_indices(nso)
        mask = np.isin(tril_idx, np.arange(nao))
        AA  = np.logical_and(*mask)
        BB  = np.logical_not(np.logical_or(*mask))
        GH2[np.ix_(AA, AA)] = H2A
        GH2[np.ix_(BB, BB)] = H2B
        GH2[np.ix_(AA, BB)] = H2D
        GH2[np.ix_(BB, AA)] = H2D.conj().T
    else:
        raise ValueError
    return GH2

def combine_mo_coeff(C):
    """
    Combine C to get GC.

    Args:
        C: ((spin,), nao, nmo)

    Returns:
        GC: (nao*2, nmo*2)
    """
    C = np.asarray(C)
    C = add_spin_dim(C, 2, non_spin_dim=2)
    GC = la.block_diag(C[0], C[1])
    return GC

def combine_mo_coeff_k(C, factor_b=1):
    """
    Combine C to get GC.

    Args:
        C: ((spin,), nkpts, nao, nmo)

    Returns:
        GC: (nkpts, nao*2, nmo*2)
    """
    C = np.asarray(C)
    C = add_spin_dim(C, 2, non_spin_dim=3)
    GC = []
    for k in range(C.shape[-3]):
        if factor_b == 1:
            GC.append(la.block_diag(C[0, k], C[1, k]))
        else:
            GC.append(la.block_diag(C[0, k], C[1, k] * factor_b))
    GC = np.asarray(GC)
    return GC

def separate_H1(GH):
    """
    Separate GH to H.

    Args:
        GH: (nso, nso)

    Returns:
        H: (3, nao, nao)
    """
    nao = GH.shape[-1] // 2
    log.eassert(nao * 2 == GH.shape[-1], "generalized H1 dimension error")
    H = np.zeros((3, nao, nao), dtype=GH.dtype)
    H[0] = GH[:nao, :nao]
    H[1] = GH[nao:, nao:]
    H[2] = GH[:nao, nao:]
    return H

GH2H = separate_H = separate_H1

def combine_H1_k(H_k):
    """
    Combine H_k to get generalized H_k.

    Args:
        H_k:  (3, nkpts, nao, nao), aa, bb, ab order

    Returns:
        GH_k: (nkpts, nso, nso)
    """
    HA, HB, HD = H_k
    nkpts, nao, _ = HA.shape
    nso = nao * 2
    res_dtype = np.result_type(HA.dtype, HB.dtype, HD.dtype)
    GH_k = np.empty((nkpts, nso, nso), dtype=res_dtype)
    for k in range(nkpts):
        GH_k[k, :nao, :nao] = HA[k]
        GH_k[k, nao:, nao:] = HB[k]
        GH_k[k, :nao, nao:] = HD[k]
        GH_k[k, nao:, :nao] = HD[k].conj().T
    return GH_k

H_k2GH_k = combine_H_k = combine_H1_k

def separate_H1_k(GH_k):
    """
    Separate generalized H_k to H_k.

    Args:
        GH_k: (nkpts, nso, nso)

    Returns:
        H_k:  (3, nkpts, nao, nao), aa, bb, ab order
    """
    nkpts, nso, _ = GH_k.shape
    nao = nso // 2
    log.eassert(nao * 2 == nso, "generalized H1 dimension error")
    H_k = np.zeros((3, nkpts, nao, nao), dtype=GH_k.dtype)
    for k in range(nkpts):
        H_k[0, k] = GH_k[k, :nao, :nao]
        H_k[1, k] = GH_k[k, nao:, nao:]
        H_k[2, k] = GH_k[k, :nao, nao:]
    return H_k

GH_k2H_k = separate_H_k = separate_H1_k

def add_H1_loc_to_k(H1_loc, H1_k):
    """
    Add a local H1 to k-space H1.

    Args:
        H1_loc: ((spin,), nao, nao)
        H1_k: ((spin,), nkpts, nao, nao)

    Returns:
        H1_new: ((spin,), nkpts, nao, nao)
    """
    res = np.array(H1_k, copy=True)
    if res.ndim == 3:
        res += H1_loc
    elif res.ndim == 4:
        for s in range(res.shape[0]):
            res[s] += H1_loc[s]
    else:
        raise ValueError
    return res

def transform_H1_local(H1, ovlp=None, C_ao_lo=None, compact=True):
    """
    Transform a local H1 to spinless form.

    Args:
        H1: real ndarray, shape (nao, nao) or (2 or 3, nao, nao)
        ovlp: overlap matrix, (nao, nao).
        C_ao_lo: if not None, ((spin,), nao, nlo), transform to LO basis.

    Returns:
        GH1: (3, nao, nao), new local H1
        GH0: constant energy shift per cell
    """
    log.debug(2, "P-H transform H1 local")
    assert not np.iscomplexobj(H1)
    H1_A, H1_B, H1_D = H2ABD(H1, bare_dim=2)
    GH1 = np.asarray((H1_A, -H1_B.T, H1_D))

    if C_ao_lo is not None:
        C_ao_lo = add_spin_dim(C_ao_lo, 2, non_spin_dim=2)
        C_a, C_b = C_ao_lo
    if ovlp is None:
        if C_ao_lo is None:
            dm = np.eye(GH1.shape[-1])
        else:
            dm = np.dot(C_b, C_b.conj().T)
    else:
        dm = la.inv(ovlp)

    GH0 = np.dot(H1_B, dm).trace()

    if C_ao_lo is not None:
        nlo = C_ao_lo.shape[-1]
        GH1_lo = np.empty((3, nlo, nlo), dtype=GH1.dtype)
        GH1_lo[0] = mdot(C_a.conj().T, GH1[0],C_a)
        GH1_lo[1] = mdot(C_b.conj().T, GH1[1],C_b)
        GH1_lo[2] = mdot(C_a.conj().T, GH1[2],C_b)
        GH1 = GH1_lo

    if not compact:
        GH1 = combine_H1(GH1)
    return GH1, GH0

def transform_H2_local(H2, ovlp=None, C_ao_lo=None, compact=True, hyb=1.0,
                       hyb_j=1.0, ao_repr=False):
    """
    Transform a local H2 to spinless form.

    Args:
        H2: real ndarray, shape (nao, nao, nao, nao) or (nao_pair, nao_pair)
        ovlp: overlap matrix, real, (nao, nao).
        C_ao_lo: real, ((spin,) nao, nao): local orbital for transform.

    Returns:
        GV2: (3, *old_shape), new local H2, aa, bb, ab
        GV1: (3, nao, nao), new local H1
        GV0: constant energy shift per cell
    """
    log.debug(2, "P-H transform H2 local")
    assert not np.iscomplexobj(H2)

    if H2.ndim == 4:
        nao = H2.shape[-1]
        GV2 = np.zeros((3, nao, nao, nao, nao))
    elif H2.ndim == 2:
        nao_pair = H2.shape[-1]
        nao = int(np.sqrt(nao_pair * 2))
        GV2 = np.zeros((3, nao_pair, nao_pair))
    else:
        raise ValueError

    if C_ao_lo is not None:
        C_ao_lo = add_spin_dim(C_ao_lo, 2, non_spin_dim=2)
        C_a, C_b = C_ao_lo
    if ovlp is None:
        if C_ao_lo is None:
            dm = np.eye(nao)
        else:
            dm = np.dot(C_b, C_b.conj().T)
    else:
        dm = la.inv(ovlp)

    #: vj = np.einsum('ijkk -> ij', H2, optimize=True)
    #: vk = np.einsum('ikkj -> ij', H2, optimize=True)
    vj, vk = scf.hf.dot_eri_dm(H2, dm, hermi=1)

    if hyb != 1.0:
        vk *= hyb
    if hyb_j != 1.0:
        vj *= hyb_j

    # H2
    GV2[0] =  H2 # from aa
    GV2[1] =  H2 # from bb
    GV2[2] = -H2 # from ab

    # H1
    log.debug(2, "vj:\n%s", vj)
    log.debug(2, "vk:\n%s", vk)
    GV1 = np.zeros((3, nao, nao))

    # from bb
    GV1[1]  = vk
    GV1[1] -= vj

    # from ab and ba
    GV1[0]  = vj

    # H0 from bb
    GV0_J =  0.5 * (np.dot(vj, dm).trace())
    GV0_K =  0.5 * (np.dot(vk, dm).trace())
    log.debug(2, "GV0_J: %s", GV0_J)
    log.debug(2, "GV0_K: %s", GV0_K)
    GV0 = GV0_J - GV0_K

    if C_ao_lo is not None and (not ao_repr): # ao2lo
        nlo = C_ao_lo.shape[-1]
        if GV2.ndim == 5:
            GV2_lo = np.zeros((3, nlo, nlo, nlo, nlo), dtype=GV2.dtype)
        else:
            nlo_pair = nlo * (nlo + 1) // 2
            GV2_lo = np.zeros((3, nlo_pair, nlo_pair), dtype=GV2.dtype)
        GV2_lo[0] = ao2mo.kernel(GV2[0], (C_a, C_a, C_a, C_a))
        GV2_lo[1] = ao2mo.kernel(GV2[1], (C_b, C_b, C_b, C_b))
        GV2_lo[2] = ao2mo.kernel(GV2[2], (C_a, C_a, C_b, C_b))
        GV2 = GV2_lo

        GV1_lo = np.zeros((3, nlo, nlo), dtype=GV1.dtype)
        GV1_lo[0] = mdot(C_a.conj().T, GV1[0], C_a)
        GV1_lo[1] = mdot(C_b.conj().T, GV1[1], C_b)
        GV1 = GV1_lo

    if not compact:
        GV2 = combine_H2(GV2)
        GV1 = combine_H1(GV1)
    return GV2, GV1, GV0

def transform_rdm1_local(rdm1, ovlp=None, compact=True):
    if rdm1.ndim == 2:
        rdm1 = rdm1[None]
    elif rdm1.ndim == 3:
        rdm1 = rdm1[:, None]
    else:
        raise ValueError
    if ovlp is not None:
        ovlp = ovlp[None]
    Grdm1 = transform_rdm1_k(rdm1, ovlp=ovlp, compact=compact)
    if compact:
        Grdm1 = Grdm1[:, 0]
    else:
        Grdm1 = Grdm1[0]
    return Grdm1

def transform_H1_k(H1, ovlp=None, C_ao_lo=None, compact=True):
    """
    Transform a k-space H1 to spinless form.

    Args:
        H1: complex ndarray, shape ((spin,) nkpts, nao, nao)
              if has spin, can be HA, HB or HA, HB, HD
        ovlp: overlap matrix.

    Returns:
        GH1: (3, nkpts, nao, nao)
        GH0: real constant energy shift per cell
    """
    nkpts = H1.shape[-3]
    nao = H1.shape[-1]
    HA, HB, HD = H2ABD(H1, bare_dim=3)

    if C_ao_lo is not None:
        C_ao_lo = add_spin_dim(C_ao_lo, 2, non_spin_dim=3)
        C_a, C_b = C_ao_lo

    dm = np.zeros((nkpts, nao, nao), dtype=np.complex128)
    if ovlp is None:
        if C_ao_lo is None:
            dm[:, range(nao), range(nao)] = 1.0
        else:
            for k in range(nkpts):
                dm[k] = np.dot(C_b[k], C_b[k].conj().T)
    else:
        for k in range(nkpts):
            dm[k] = la.inv(ovlp[k])

    GH1 = np.zeros((3, nkpts, nao, nao), dtype=H1.dtype)
    GH1[0] =  HA
    #GH1[1] = -HB.transpose(0, 2, 1).conj()
    GH1[1] = -HB
    GH1[2] =  HD

    #GH0 = np.einsum('kii ->', HB, optimize=True)
    GH0 = 0.0
    for k in range(nkpts):
        GH0 += np.trace(np.dot(HB[k], dm[k]))
    if abs(GH0.imag) > IMAG_DISCARD_TOL:
        log.warn("transform_H1_k: GH0 has imaginary part: %s", GH0.imag)
    GH0 = GH0.real / float(nkpts)

    if C_ao_lo is not None:
        nlo = C_b.shape
        GH1_lo = np.empty((3, nkpts, nlo, nlo), dtype=GH1.dtype)
        for k in range(nkpts):
            GH1_lo[0, k] = mdot(C_a[k].conj().T, GH1[0, k], C_a[k])
            GH1_lo[1, k] = mdot(C_b[k].conj().T, GH1[1, k], C_b[k])
            GH1_lo[2, k] = mdot(C_a[k].conj().T, GH1[2, k], C_b[k])
        GH1 = GH1_lo

    if not compact:
        GH1 = combine_H1_k(GH1)
    return GH1, GH0

def transform_rdm1_k(rdm1_k, ovlp=None, compact=False):
    """
    Transform a k-space rdm1 to spinless form.

    Args:
        rdm1_k: complex ndarray, shape ((spin,) nkpts, nao, nao)
                if has spin, can be rdm1_A, rdm1_B (and kappa_AB).
        ovlp_k: overlap matrix. (nkpts, nao, nao)

    Returns:
        GRdm1: if compact, (3, nkpts, nao, nao), same convention as input.
               else, (nkpts, nso, nso)

    Note:
        kappa^AB_{pq} = <a_qbeta a_palpha>
    """
    nkpts = rdm1_k.shape[-3]
    nao = rdm1_k.shape[-1]
    rdm1_A_k, rdm1_B_k, rdm1_AB_k = H2ABD(rdm1_k, bare_dim=3)

    if compact:
        GRdm1 = np.zeros((3, nkpts, nao, nao), dtype=rdm1_k.dtype)
        GRdm1[0]  =  rdm1_A_k
        if ovlp is None:
            #GRdm1[1] -= rdm1_B_k.transpose(0, 2, 1).conj()
            GRdm1[1, :, range(nao), range(nao)] = 1.0
        else:
            for k in range(nkpts):
                GRdm1[1, k] = la.inv(ovlp[k])
        GRdm1[1] -= rdm1_B_k
        GRdm1[2]  =  rdm1_AB_k
    else:
        nso = nao * 2
        GRdm1 = np.zeros((nkpts, nso, nso), dtype=rdm1_k.dtype)
        GRdm1[:, :nao, :nao]  =  rdm1_A_k
        if ovlp is None:
            #GRdm1[:, nao:, nao:] -= rdm1_B_k.transpose(0, 2, 1).conj()
            GRdm1[:, range(nao, nso), range(nao, nso)] = 1.0
        else:
            for k in range(nkpts):
                GRdm1[k, nao:, nao:] = la.inv(ovlp[k])
        GRdm1[:, nao:, nao:] -= rdm1_B_k
        GRdm1[:, :nao, nao:]  =  rdm1_AB_k
        GRdm1[:, nao:, :nao]  =  rdm1_AB_k.conj().transpose(0, 2, 1)
    return GRdm1

def get_GV1_GV0_from_df(mydf, ovlp=None, C_ao_lo=None, compact=True,
                        return_jk=False, return_hf=False, hyb=1.0,
                        hyb_j=1.0, ao_repr=False):
    """
    Generate particle-hole transformed H1 and H0
    from density fitting integrals.

    Args:
        mydf: df object.
        ovlp: (nkpts, nao, nao), AO overlap matrix.
        C_ao_lo: (nkpts, nao, nlo), do the ao2lo during the transform.
        hyb: hybridization factor for vk.

    Returns:
        res_H1: (3, nkpts, nlo, nlo), new H1_k in LO basis.
        res_H0: constant energy shift per cell.
        vj: if return_jk == True
        vk: if return_jk == True
    """
    log.debug(2, "P-H transform H2 DF")
    log.debug(2, "vk scale factor %s", hyb)

    nkpts = len(mydf.kpts)
    nao = mydf.cell.nao_nr()

    if C_ao_lo is not None:
        C_ao_lo = add_spin_dim(C_ao_lo, 2, non_spin_dim=3)
        C_a, C_b = C_ao_lo

    dm = np.zeros((nkpts, nao, nao), dtype=np.complex128)
    if ovlp is None:
        if C_ao_lo is None:
            log.warn("get_GV1_GV0_from_df:\novlp and C_ao_lo are both None, "
                     "using identity for dm ...")
            dm[:, range(nao), range(nao)] = 1.0
        else:
            for k in range(nkpts):
                dm[k] = np.dot(C_b[k], C_b[k].conj().T)
    else:
        for k in range(nkpts):
            dm[k] = la.inv(ovlp[k])

    # vj, vk
    vj_ao, vk_ao = mydf.get_jk(dm, hermi=True)
    if hyb_j != 1.0:
        vj_ao *= hyb_j
    vj_ao = np.asarray((vj_ao, vj_ao))

    GV1_ao = np.zeros((3, nkpts, nao, nao), dtype=np.complex128)
    GV1_ao[0]  = vj_ao[0]
    GV1_ao[1]  = vk_ao * hyb
    GV1_ao[1] -= vj_ao[1]

    GV0_J = 0.0
    GV0_K = 0.0
    for k in range(nkpts):
        GV0_J += np.trace(np.dot(vj_ao[1, k], dm[k]))
        GV0_K += np.trace(np.dot(vk_ao[k], dm[k]))

    GV0_J = GV0_J.real / (nkpts * 2.0)
    GV0_K = GV0_K.real / (nkpts * 2.0)
    GV0 = GV0_J - GV0_K * hyb

    if return_hf:
        GV1_ao_hf = np.zeros((3, nkpts, nao, nao), dtype=np.complex128)
        GV1_ao_hf[0]  = vj_ao[0]
        GV1_ao_hf[1]  = vk_ao
        GV1_ao_hf[1] -= vj_ao[1]
        GV0_hf = GV0_J - GV0_K
        GV1_hf = GV1_ao_hf

    # ao2lo
    if C_ao_lo is not None and (not ao_repr):
        nlo = C_ao_lo.shape[-1]
        vj_lo = np.zeros((2, nkpts, nlo, nlo), dtype=np.complex128)
        vk_lo = np.zeros((nkpts, nlo, nlo), dtype=np.complex128)
        for k in range(nkpts):
            vj_lo[0, k] = mdot(C_a[k].conj().T, vj_ao[0, k], C_a[k])
            vj_lo[1, k] = mdot(C_b[k].conj().T, vj_ao[1, k], C_b[k])
            vk_lo[k] = mdot(C_b[k].conj().T, vk_ao[k], C_b[k])

        GV1_lo = np.zeros((3, nkpts, nlo, nlo), dtype=np.complex128)
        GV1_lo[0]  = vj_lo[0]
        GV1_lo[1]  = vk_lo * hyb
        GV1_lo[1] -= vj_lo[1]

        vj = vj_lo
        vk = vk_lo
        GV1 = GV1_lo

        if return_hf:
            GV1_lo_hf = np.zeros((3, nkpts, nlo, nlo), dtype=np.complex128)
            GV1_lo_hf[0]  = vj_lo[0]
            GV1_lo_hf[1]  = vk_lo
            GV1_lo_hf[1] -= vj_lo[1]
            GV1_hf = GV1_lo_hf
    else:
        vj = vj_ao
        vk = vk_ao
        GV1 = GV1_ao

    # H1 and H0
    log.debug(2, "vj          :\n%s", vj)
    log.debug(2, "vk (scaled) :\n%s", vk * hyb)

    log.debug(2, "GV0_J          : %s", GV0_J)
    log.debug(2, "GV0_K (scaled) : %s", GV0_K * hyb)
    if not compact:
        GV1 = combine_H1_k(GV1)

    if return_hf: # return the non-scaled GV1 and GV0 for HF.
        if not compact:
            GV1_hf = combine_H1_k(GV1_hf)
        return GV1, GV0, GV1_hf, GV0_hf

    if return_jk:
        return GV1, GV0, vj, vk
    else:
        return GV1, GV0

get_H1_H0_from_df = get_GV1_GV0_from_df

def get_jk_ph(mf, cell=None, dm_kpts=None, hermi=0, kpts=None, kpts_band=None,
              with_j=True, with_k=True, **kwargs):
    if cell is None: cell = mf.cell
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if kpts is None: kpts = mf.kpts
    if kpts_band is None: kpts_band = kpts
    nkpts = len(kpts)
    nband = len(kpts_band)

    dm_kpts = np.asarray(dm_kpts)
    nso = dm_kpts.shape[-1]
    nao = nso // 2
    dms = dm_kpts.reshape(-1, nkpts, nso, nso)
    n_dm = dms.shape[0]

    dmaa = dms[:, :, :nao, :nao]
    dmab = dms[:, :, nao:, :nao]
    dmbb = dms[:, :, nao:, nao:]
    dms = np.vstack((dmaa, dmbb, dmab))

    j1, k1 = mf.with_df.get_jk(dms, hermi, kpts, kpts_band, with_j, with_k,
                               exxdiv=mf.exxdiv)
    if with_j:
        j1 = j1.reshape(3, n_dm, nband, nao, nao)
    if with_k:
        k1 = k1.reshape(3, n_dm, nband, nao, nao)

    vj = vk = None
    if with_j:
        vj = np.zeros((n_dm,nband,nso,nso), j1.dtype)
        # ZHC NOTE
        #vj[:,:,:nao,:nao] = vj[:,:,nao:,nao:] = j1[0] + j1[1]
        vj[:, :, :nao, :nao] = j1[0] - j1[1]
        vj[:, :, nao:, nao:] = j1[1] - j1[0]
        vj = _format_jks(vj, dm_kpts, kpts_band, kpts)

    if with_k:
        vk = np.zeros((n_dm,nband,nso,nso), k1.dtype)
        # ZHC NOTE FIXME: I think there is a bug in pyscf.
        vk[:, :, :nao, :nao] = k1[0]
        vk[:, :, nao:, nao:] = k1[1]
        #vk[:,:,:nao,nao:] = k1[2]
        #vk[:,:,nao:,:nao] = k1[2].transpose(0,1,3,2).conj()
        #vk[:, :, :nao, nao:] = -k1[2]
        #vk[:, :, nao:, :nao] = -k1[2].transpose(0,1,3,2).conj()
        vk[:, :, nao:, :nao] = -k1[2]
        vk[:, :, :nao, nao:] = -k1[2].transpose(0, 1, 3, 2).conj()
        vk = _format_jks(vk, dm_kpts, kpts_band, kpts)

    return vj, vk

class KGHFPH(pbc.scf.kghf.KGHF):
    """
    KGHF with P-H transform.
    """
    get_jk = get_jk_ph

    energy_elec = energy_elec_ghf

class KGKSPH(kgks.KGKS):
    """
    KGHF with P-H transform.
    """
    get_jk = get_jk_ph

    get_veff = kgks.get_veff_ph

    energy_elec = kgks.energy_elec

def frac_mu_(mf, nelec, mu0=None, tol=None):
    """
    Addons for HFB methods to assign fractional chemical potential mu.

    Args:
        mf: kmf object.
        nelec: number of electrons (per cell).
        mu0: initial mu.
        tol: tolerance for fitting nelec.

    Returns:
        mf: modified mf object.
    """
    from pyscf.scf import ghf
    from pyscf.pbc.scf import khf
    from libdmet.routine.spinless_helper import mono_fit_2
    mf_class = mf.__class__
    is_ghf = isinstance(mf, ghf.GHF)
    is_khf = isinstance(mf, khf.KSCF)
    if tol is None:
        tol = mf.conv_tol * 0.1
    assert is_ghf

    if is_khf:
        raise NotImplementedError
    else:
        def eig(h, s):
            nso = s.shape[-1]
            nao = nso // 2

            def nelec_cost_fn_brentq(mu):
                h1 = np.array(h, copy=True)
                h1[:nao, :nao] -= s[:nao, :nao] * mu
                h1[nao:, nao:] += s[nao:, nao:] * mu
                e, c = la.eigh(h1, s)
                mo_occ = mf.get_occ(e, c)
                dm = mf.make_rdm1(c, mo_occ)
                rho = np.einsum('pq, qp -> p', dm, s)
                nelec_calc = rho[:nao].sum() + nao - (rho[nao:].sum())
                return nelec_calc

            mu_elec = mono_fit_2(nelec_cost_fn_brentq, nelec, mf.mu_elec, tol,
                                 increase=True)
            h1 = np.array(h, copy=True)
            h1[:nao, :nao] -= s[:nao, :nao] * mu_elec
            h1[nao:, nao:] += s[nao:, nao:] * mu_elec
            e, c = la.eigh(h1, s)
            mf.mu_elec = mu_elec
            return e, c

    if mu0 is None:
        mf.mu_elec = 0.0
    else:
        mf.mu_elec = mu0
    mf._keys = mf._keys.union(['mu_elec'])

    mf.eig = eig
    return mf

def project_dm_nr2nr(cell1, dm1, cell2, kpts=None):
    """
    Project density matrices from cell1 basis to cell2 basis.
    """
    s22 = cell2.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts)
    s21 = pbcgto.intor_cross('int1e_ovlp', cell2, cell1, kpts=kpts)
    if kpts is None or np.shape(kpts) == (3,):  # A single k-point
        p21 = lib.cho_solve(s22, s21, strict_sym_pos=False)
        if isinstance(dm1, np.ndarray) and dm1.ndim == 2:
            dm2 = np.dot(p21, np.dot(dm1, p21.conj().T))
        else:
            dm2 = lib.einsum('pi,nij,qj->npq', p21, dm1, p21.conj())
    else:
        dm1 = np.asarray(dm1)
        nkpts = dm1.shape[-3]
        assert len(kpts) == nkpts
        if dm1.ndim == 3:
            dm2 = []
            for k in range(nkpts):
                p21 = lib.cho_solve(s22[k], s21[k], strict_sym_pos=False)
                dm2.append(np.dot(p21, np.dot(dm1[k], p21.conj().T)))
            dm2 = np.asarray(dm2)
        else:
            spin = dm1.shape[0]
            dm2 = []
            for s in range(spin):
                dm2_k = []
                for k in range(nkpts):
                    p21 = lib.cho_solve(s22[k], s21[k], strict_sym_pos=False)
                    dm2_k.append(np.dot(p21, np.dot(dm1[s, k], p21.conj().T)))
                dm2.append(dm2_k)
            dm2 = np.asarray(dm2)
    return dm2

