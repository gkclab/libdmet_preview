#! /usr/bin/env python

"""
Basic k-basis, R-basis functions, folding of Brillouin zone.

Author:
    Zhihao Cui
"""

import numpy as np
import scipy.linalg as la
from scipy import fft as scifft
import itertools as it

from pyscf import lib
from pyscf.pbc import tools
from pyscf.pbc.lib.kpts_helper import KPT_DIFF_TOL

from libdmet.utils import logger as log
from libdmet.utils.misc import max_abs, format_idx, get_cart_prod_idx
from libdmet.settings import IMAG_DISCARD_TOL

"""
Fractional / Cartesian coords transform.
"""

def Frac2Real(cellsize, coord):
    assert cellsize.ndim == 2 and cellsize.shape[0] == cellsize.shape[1]
    return np.dot(coord, cellsize)

def Real2Frac(cellsize, coord):
    assert cellsize.ndim == 2 and cellsize.shape[0] == cellsize.shape[1]
    return np.dot(coord, la.inv(cellsize))

frac2cart = frac2real = Frac2Real
cart2frac = real2frac = Real2Frac

def get_R_vec(cell, kmesh):
    latt_vec = cell.lattice_vectors()
    R_rel = [np.arange(x) for x in kmesh]
    R_vec_rel = lib.cartesian_prod(R_rel)
    R_vec_abs = np.dot(R_vec_rel, latt_vec)
    return R_vec_abs

def make_kpts_scaled(kmesh):
    """
    Make scaled kpoints that follow the convention of np.fft, 
    should be the same as cell.make_kpts with wrap_around=True.
    """
    ks_each_axis = [scifft.fftfreq(kmesh[d], 1.) for d in range(len(kmesh))]
    scaled_kpts = lib.cartesian_prod(ks_each_axis)
    return scaled_kpts

def round_to_FBZ(kpts, tol=1e-10, wrap_around=True):
    """
    Round fractional kpts to the first Brillouin zone.
    wrap_around == True, [-0.5, 0.5), otherwise [0.0, 1.0).
    """
    kpts_round = kpts - np.floor(kpts) # first to 0.0 - 1.0
    if wrap_around:
        kpts_round[kpts_round > (0.5 - tol)] -= 1.0 # move 0.5 to -0.5
    else:
        kpts_round[kpts_round > (1.0 - tol)] = 0.0 # move 1.0 to 0.0
    return kpts_round

def round_to_FUC(coords, tol=1e-10, wrap_around=False):
    """
    Round fractional coords to the first unit cell.
    """
    return round_to_FBZ(coords, tol=tol, wrap_around=wrap_around)

def kpt_member(kpt, kpts):
    """
    Find the index of a kpt in the kpts (both in fractional kpoints).
    will consider the PBC of Brillouin zone.
    """
    kpts = np.reshape(kpts, (len(kpts), kpt.size))
    dk = kpts - kpt.ravel()
    dk = la.norm(dk - np.round(dk), axis=-1)
    return np.where(dk < KPT_DIFF_TOL)[0]

def get_kmesh(cell, kpts):
    """
    Get kmesh from a set of kpts.
    """
    scaled_k = cell.get_scaled_kpts(kpts).round(8)
    kmesh = [len(np.unique(scaled_k[:, d])) for d in range(scaled_k.shape[-1])]
    return kmesh

"""
Fouier phases and k <-> R transform.
"""

def get_phase(cell, kpts, kmesh=None):
    """
    The unitary transformation that transforms the supercell basis k-mesh
    adapted basis. iRk / np.sqrt(Nk).
    """
    if kmesh is None:
        kmesh = get_kmesh(cell, kpts)
    
    R_vec_abs = get_R_vec(cell, kmesh)
    ncells = len(R_vec_abs)
    phase = np.exp(1.0j * np.einsum('Ru, ku -> Rk', R_vec_abs, kpts))
    phase /= np.sqrt(ncells)  # normalization in supercell

    # R_rel_mesh has to be construct exactly same to the Ts in super_cell function
    scell = tools.super_cell(cell, kmesh)
    return scell, phase

def get_phase_R2k(cell, kpts, kmesh=None):
    """
    exp(-iRk)
    """
    if kmesh is None:
        kmesh = get_kmesh(cell, kpts)
    R_vec_abs = get_R_vec(cell, kmesh)
    ncells = len(R_vec_abs)
    phase = np.exp(-1.0j * np.einsum('Ru, ku -> Rk', R_vec_abs, kpts))
    return phase

def get_phase_k2R(cell, kpts, kmesh=None):
    """
    exp(ikR) / nkpts
    """
    return get_phase_R2k(cell, kpts, kmesh=kmesh).conj().T / len(kpts)

def R2k(dm_R, kmesh):
    """
    transform a one-body operator in stripe, e.g. H1, dm to k space.
    allow additional dim of spin.
    """
    if dm_R.ndim == 3:
        dm_k = FFTtoK(dm_R, kmesh)
    elif dm_R.ndim == 4: # unrestricted, generalized
        dm_k = np.zeros_like(dm_R, dtype=np.complex)
        for s in range(dm_R.shape[0]):
            dm_k[s] = FFTtoK(dm_R[s], kmesh)
    else:
        raise ValueError("unknown shape of dm_R: %s" % str(dm_R.shape))
    return dm_k

def k2R(dm_k, kmesh, tol=IMAG_DISCARD_TOL):
    """
    transform a one-body operator with kpts, e.g. H1, dm to real space (stripe).
    allow additional dim of spin.
    """
    if dm_k.ndim == 3:
        dm_R = FFTtoT(dm_k, kmesh, tol=tol)
    elif dm_k.ndim == 4: # unrestricted, generalized
        dm_R = np.zeros_like(dm_k)
        for s in range(dm_R.shape[0]):
            dm_R[s] = FFTtoT(dm_k[s], kmesh, tol=tol)
    else:
        raise ValueError("unknown shape of dm_k: %s" % str(dm_k.shape))
    if max_abs(dm_R.imag) < tol:
        dm_R = dm_R.real
    return dm_R

def FFTtoK(A, kmesh):
    """
    before FFT ncells * nscsites * nscsites where the first index is cell
    after FFT first index is k-point
    """
    return scifft.fftn(A.reshape(tuple(kmesh) + A.shape[-2:]), \
        axes=range(len(kmesh)), workers=-1).reshape(A.shape)

def FFTtoT(B, kmesh, tol=IMAG_DISCARD_TOL):
    """
    from k space to real space.
    """
    A = scifft.ifftn(B.reshape(tuple(kmesh) + B.shape[-2:]), \
        axes=range(len(kmesh)), workers=-1).reshape(B.shape)
    if max_abs(A.imag) < tol:
        A = A.real
    return A

def R2k_H2(H2, phase_R2k):
    """
    Transform H2 from R to k space.
    Assume H2 is stored as (ncell, )*3 + (nao, )*4
    i.e. PQR0pqrs
    phase_R2k: exp(-iRk)
    """
    nkpts = H2.shape[-5]
    norb = H2.shape[-1]
    H2_k = lib.einsum('XP, PQRpqrs -> XQRpqrs', phase_R2k.T, H2)
    # ZHC FIXME NOTE Here I think the following is correct,
    # however, not equals to the supercell calculation
    # maybe there is difference between H2 and Gamma_2?
    #H2_k = lib.einsum('YQ, XQRpqrs -> XYRpqrs', \
    #        phase_R2k.conj().T/float(nkpts), H2_k)
    H2_k = lib.einsum('YQ, XQRpqrs -> XYRpqrs', phase_R2k.conj().T, H2_k)
    H2_k = lib.einsum('ZR, XYRpqrs -> XYZpqrs', phase_R2k.T, H2_k)
    return H2_k

def k2R_H2(H2, phase_k2R, tol=IMAG_DISCARD_TOL):
    """
    Transform H2 from k to R space.
    Assume H2 is stored as (nkpts, )*3 + (nao, )*4
    i.e. XYZpqrs
    phase_k2R: 1/N exp(ikR)
    """
    nkpts = H2.shape[-5]
    norb = H2.shape[-1]
    H2_R = lib.einsum('PX, XYZpqrs -> PYZpqrs', phase_k2R.T, H2)
    # ZHC FIXME NOTE Here I think the following is correct,
    # however, not equals to the supercell calculation
    # maybe there is difference between H2 and Gamma_2?
    #H2_R = lib.einsum('QY, PYZpqrs -> PQZpqrs', \
    #        phase_k2R.conj().T*float(nkpts), H2_R)
    H2_R = lib.einsum('QY, PYZpqrs -> PQZpqrs', phase_k2R.conj().T, H2_R)
    H2_R = lib.einsum('RZ, PQZpqrs -> PQRpqrs', phase_k2R.T, H2_R)
    if max_abs(H2_R.imag) < tol:
        H2_R = H2_R.real
    return H2_R

"""
Folding BZ from unit cell to supercell.
"""

def find_idx_k_in_K(k_abs, sk_abs, scell, tol=1e-10, wrap_around=True):
    """
    Find the map from kpts in unit cell to kpts in supercell.
    """
    recip_vec_sc = scell.reciprocal_vectors()
    sk_scaled_sc = scell.get_scaled_kpts(sk_abs)
    k_scaled_sc = round_to_FBZ(Real2Frac(recip_vec_sc, k_abs), tol=tol, \
            wrap_around=wrap_around)
    norm_diff = la.norm(k_scaled_sc[:, None] - sk_scaled_sc[None], axis=-1)
    idx = np.where(norm_diff < tol)[1]
    assert len(idx) == len(k_abs)
    return idx

def find_idx_R_vec(R_vec, skmesh, scell, tol=1e-10):
    latt_vec = scell.lattice_vectors()
    R_vec_scaled = Real2Frac(latt_vec, R_vec)
    cart_prod = (R_vec_scaled + tol).astype(int)
    return get_cart_prod_idx(cart_prod, skmesh)

def k2gamma(mo_energy, mo_coeff, mo_occ, phase, make_real=False, \
        lattice=None, ovlp=None, tol_deg=1e-5):
    """
    Convert mf objects with k sampling to ones at Gamma point.

    Args:
        mo_energy: orbital energy, (nkpts, nmo)
        mo_coeff: oribital coeff,  (nkpts, nao, nmo)
        mo_occ: occupancy,         (nkpts, nmo)
        phase: 1/sqrt(N) exp(iRk)
        make_real: whether make the coefficient real
    
    Returns:
        mo_energy_g
        mo_coeff_g
        mo_occ_g
    """
    log.debug(1, "k2gamma: start")
    if make_real:
        assert lattice is not None
        assert ovlp is not None
    mo_energy = np.asarray(mo_energy)
    mo_coeff = np.asarray(mo_coeff)
    mo_occ = np.asarray(mo_occ)
    nkpts, nao, nmo = mo_coeff.shape[-3:]
    nR = phase.shape[0]
    assert nR == nkpts
    
    mo_energy_g = mo_energy.reshape(nkpts*nmo)
    mo_coeff_g = np.einsum('Rk, kum -> Rukm', phase, \
            mo_coeff).reshape(nR*nao, nkpts*nmo)
    mo_occ_g = mo_occ.reshape(nkpts*nmo)
    
    # sort according to orbital energy
    sort_idx = np.argsort(mo_energy_g.ravel(), kind='mergesort')
    mo_energy_g = mo_energy_g[sort_idx]
    mo_coeff_g = mo_coeff_g[:, sort_idx]
    mo_occ_g = mo_occ_g[sort_idx]

    if make_real:
        log.debug(1, "k2gamma: make coeff real")
        if max_abs(mo_coeff_g.imag) < IMAG_DISCARD_TOL:
            log.debug(2, "k2gamma: already real")
            mo_coeff_g = mo_coeff_g.real
        else: # make coeffient real by take linear combination of MOs
            ovlp_k = ovlp
            ovlp_R_sc = lattice.expand(lattice.k2R(ovlp_k))
            energy_deg = np.abs(mo_energy_g[1:] - mo_energy_g[:-1]) < tol_deg
            mask_deg = np.append(False, energy_deg) | np.append(energy_deg, False)
            log.debug(2, "k2gamma: mo_energy_g: \n%s", mo_energy_g)
            log.debug(2, "k2gamma: mask_deg: \n%s", mask_deg)
            if np.any(energy_deg):
                if max_abs(mo_coeff_g[:, ~mask_deg].imag) < IMAG_DISCARD_TOL:
                    # imag comes from the degenerate part
                    shift = np.min(mo_energy_g[mask_deg]) - 0.1
                    fock = np.dot(mo_coeff_g[:, mask_deg] \
                            * (mo_energy_g[mask_deg] - shift), \
                            mo_coeff_g[:, mask_deg].conj().T)
                    assert max_abs(fock.imag) < IMAG_DISCARD_TOL
                    nat_occ, nat_orb = la.eigh(fock.real, ovlp_R_sc, type=2)
                    mo_coeff_g = mo_coeff_g.real
                    mo_coeff_g[:, mask_deg] = \
                            nat_orb[:, nat_occ > IMAG_DISCARD_TOL]
                else:
                    fock = np.dot(mo_coeff_g * mo_energy_g, mo_coeff_g.conj().T)
                    assert max_abs(fock.imag) < IMAG_DISCARD_TOL
                    e, mo_coeff_g = la.eigh(fock.real, ovlp_R_sc, type=2)
            else:
                fock = np.dot(mo_coeff_g * mo_energy_g, mo_coeff_g.conj().T)
                assert max_abs(fock.imag) < IMAG_DISCARD_TOL
                e, mo_coeff_g = la.eigh(fock.real, ovlp_R_sc, type=2)
    
    # add kpt dim
    mo_energy_g = mo_energy_g[None]
    mo_coeff_g = mo_coeff_g[None]
    mo_occ_g = mo_occ_g[None]
    log.debug(1, "k2gamma: end")
    return mo_energy_g, mo_coeff_g, mo_occ_g

def fold_kmf(mo_energy, mo_coeff, mo_occ, latt_0, latt_1, resort=True, \
        tol=1e-10):
    """
    Convert mf objects with k sampling to supercell with smaller kmesh.
    From cell_0 to cell_1.
    When cell_1 is the whole lattice (with only Gamma point),
    it is equivalent to k2gamma.

    Args:
        mo_energy: orbital energy, (nkpts, nmo)
        mo_coeff: oribital coeff,  (nkpts, nao, nmo)
        mo_occ: occupancy,         (nkpts, nmo)
        latt_0: lattice object for kmesh_0
        latt_1: lattice object for kmesh_1
    
    Returns:
        ew, ev, occ in latt_1.
    """
    log.debug(1, "Fold kmf: start")
    cell_0  = latt_0.cell
    kpts_0  = latt_0.kpts
    kmesh_0 = latt_0.kmesh
    phase_0 = latt_0.phase
    cell_1  = latt_1.cell
    kpts_1  = latt_1.kpts
    kmesh_1 = latt_1.kmesh
    phase_1 = latt_1.phase

    idx_k_in_1 = find_idx_k_in_K(kpts_0, kpts_1, cell_1, tol=tol)
    idx_k = np.argsort(idx_k_in_1, kind='mergesort')
    log.debug(2, "idx k: \n%s", idx_k)

    R_vec_0 = get_R_vec(cell_0, kmesh_0)
    idx_R_in_1 = find_idx_R_vec(R_vec_0, kmesh_1, cell_1, tol=tol)
    idx_R = np.argsort(idx_R_in_1, kind='mergesort')
    log.debug(2, "idx R: \n%s", idx_R)
     
    mo_energy = np.asarray(mo_energy)
    mo_coeff = np.asarray(mo_coeff)
    mo_occ = np.asarray(mo_occ)
    nkpts_0, nao_0, nmo_0 = mo_coeff.shape[-3:]
    nkpts_1 = latt_1.nkpts
    nR_1 = nkpts_1
    nao_1 = latt_1.nao
    nmo_1 = nmo_0 * nkpts_0 // nkpts_1

    ew = mo_energy[idx_k].reshape(nkpts_1, nmo_1)
    ev = np.einsum('Rk, kum -> Rukm', phase_0[idx_R][:, idx_k], \
            mo_coeff[idx_k]).reshape(nR_1, nao_1, nkpts_1, nmo_1)
    occ = mo_occ[idx_k].reshape(nkpts_1, nmo_1)
    
    # sort according to orbital energy at each k
    if resort:
        sort_idx = np.argsort(ew, axis=-1, kind='mergesort')
        for k, idx in enumerate(sort_idx):
            ew[k] = ew[k, idx]
            ev[:, :, k]  = ev[:, :, k, idx]
            occ[k] = occ[k, idx]
    
    # absorb the phase factor of supercell
    ev = np.einsum('Rk, Rukm -> kum', phase_1.conj(), ev)
    log.debug(1, "Fold kmf: end")
    return ew, ev, occ

def fold_h1(hcore_0_k, latt_0, latt_1, resort_row=False, resort_col=False, \
        uc2sc=False, tol=1e-10):
    """
    Fold 1-body operator to the supercell.
    
    Args:
        hcore_0_k: h1 in the small cell, with kpts
        latt_0: the small cell lattice
        latt_1: the large cell lattice
        resort_row: resort the row indices according to core, val and virt
        resort_row: resort the column indices according to core, val and virt
        uc2sc: generate transformation matrix from unit cell to supercell
        tol: tolerance for kpts, Rvec equivalence.

    Returns:
        hcore_1_k: the folded object.    
    """
    log.debug(1, "Fold h1 / lo: start")
    hcore_0_k = np.asarray(hcore_0_k)
    cell_0  = latt_0.cell
    nkpts_0, nrow_0, ncol_0 = hcore_0_k.shape 
    kmesh_0 = latt_0.kmesh
    log.eassert(np.prod(kmesh_0) == nkpts_0, 
                "kmesh_0 (%s) != nkpts_0 (%s)", np.prod(kmesh_0), nkpts_0)

    cell_1  = latt_1.cell
    nkpts_1 = latt_1.nkpts
    kmesh_1 = latt_1.kmesh
    ncells_first = nkpts_0 // nkpts_1
    nrow_1 = nrow_0 * ncells_first
    ncol_1 = ncol_0 * ncells_first

    R_vec_0 = get_R_vec(cell_0, kmesh_0)
    idx_R_in_1 = find_idx_R_vec(R_vec_0, kmesh_1, cell_1, tol=tol)
    idx_R = np.argsort(idx_R_in_1, kind='mergesort')
    log.debug(2, "idx R: \n%s", idx_R)
    
    hcore_0_R = latt_0.k2R(hcore_0_k)
    hcore_1_R = np.zeros((nkpts_0, nrow_0, ncells_first, ncol_0), \
            dtype=hcore_0_R.dtype)
    for q, cell_q in enumerate(idx_R[:ncells_first]):
        idx = [latt_0.subtract(cell_p, cell_q) for cell_p in idx_R]
        hcore_1_R[:, :, q] = hcore_0_R[idx]
    hcore_1_R = hcore_1_R.reshape(nkpts_1, nrow_1, ncol_1)
     
    # possible sort for orbitals
    if resort_row:
        ncore_0 = latt_0.ncore
        nval_0  = latt_0.nval
        nvirt_0 = latt_0.nvirt
        cell_row_idx = np.arange(0, nrow_1, nrow_0)[:, None]
        core_row_idx = (np.arange(0, ncore_0) + cell_row_idx).ravel()
        val_row_idx  = (np.arange(ncore_0, ncore_0 + nval_0) + \
                cell_row_idx).ravel()
        virt_row_idx = (np.arange(ncore_0+nval_0, ncore_0+nval_0+nvirt_0) + \
                cell_row_idx).ravel()
        sort_row_idx = np.hstack((core_row_idx, val_row_idx, virt_row_idx))
        hcore_1_R = hcore_1_R[:, sort_row_idx]
     
    if resort_col:
        ncore_0 = latt_0.ncore
        nval_0  = latt_0.nval
        nvirt_0 = latt_0.nvirt
        cell_col_idx = np.arange(0, ncol_1, ncol_0)[:, None]
        core_col_idx = (np.arange(0, ncore_0) + cell_col_idx).ravel()
        val_col_idx  = (np.arange(ncore_0, ncore_0 + nval_0) + \
                cell_col_idx).ravel()
        virt_col_idx = (np.arange(ncore_0+nval_0, ncore_0+nval_0+nvirt_0) + \
                cell_col_idx).ravel()
        sort_col_idx = np.hstack((core_col_idx, val_col_idx, virt_col_idx))
        hcore_1_R = hcore_1_R[:, :, sort_col_idx]

    if uc2sc:
        # reorder to the original cell order
        idx_R_inv = np.argsort(idx_R, kind='mergesort')
        hcore_1_k = latt_0.R2k(hcore_1_R.reshape(nkpts_0, nrow_0, \
                ncol_1)[idx_R_inv])
    else:
        hcore_1_k = latt_1.R2k(hcore_1_R)
    log.debug(1, "Fold h1 / lo: end")
    return hcore_1_k

def fold_lo(C_ao_lo, latt_0, latt_1, resort_row=False, resort_col=True, \
        uc2sc=False, tol=1e-10):
    """
    Fold C_ao_lo to supercell. See fold_h1 for details.
    
    Note:
        Column indices will be resorted according to latt_0. 
        The core, val and virtual are grouped separately.
    """
    return fold_h1(C_ao_lo, latt_0, latt_1, resort_row=resort_row, \
            resort_col=resort_col, uc2sc=uc2sc, tol=tol)

"""
Supercell basis / atom indexing and labels.
Select some coordinates and find the corresponding basis, atom ids 
in the whole lattice.
"""

def search_basis_id_sc(cell, scell, coords, labels, tol=1e-8, \
        return_labels=False):
    """
    Search the basis ids for the atoms at coords, within the supercell.

    Args:
        cell: unit cell
        scell: super cell
        coords: target fractional coordinations
        labels: a list of string, 'id atom nlm'
        tol: tolerance for frac coords

    Returns:
        basis_ids: supercell basis ID corresponds to coords.
    """
    atom_ids = search_atom_id_sc(cell, scell, coords, tol=tol)
    natm = cell.natm
    ncells = scell.natm // natm
    labels_sc = translate_labels(labels, ncells, natm)
    atom_ids_sc = np.array([int(lab.split()[0]) for lab in labels_sc])
    basis_ids = np.where(np.isin(atom_ids_sc, atom_ids))[0]
    log.info("Search basis ID (atom):\n%s", atom_ids)
    log.info("Search basis ID (basis):\n%s", labels_sc[basis_ids])
    if return_labels:
        return basis_ids, labels_sc[basis_ids]
    else:
        return basis_ids

def search_atom_id_sc(cell, scell, coords, tol=1e-8):
    """
    Search atom ids of coords in the supercell.

    Args:
        cell: unit cell
        scell: super cell
        coords: target fractional coordinations
        tol: tolerance

    Returns:
        atom_ids: supercell atom ID corresponds to coords.
    """
    atoms = scell._atom
    names, coords_sc = zip(*atoms)
    names, coords_sc = np.asarray(names), np.asarray(coords_sc)
    coords_sc = Real2Frac(scell.lattice_vectors(), coords_sc)
    coords_sc = round_to_FUC(coords_sc, tol=tol)
    coords_sc = Frac2Real(scell.lattice_vectors(), coords_sc)

    coords = Frac2Real(cell.lattice_vectors(), coords)
    norm_diff = la.norm(coords[:, None] - coords_sc[None], axis=-1)
    atom_ids = np.where(norm_diff < tol)[1]
    log.eassert(len(atom_ids) == len(coords), \
            "len(atom_ids) [%s] != len(coords) [%s]", \
             len(atom_ids), len(coords))
    return atom_ids
    
def translate_labels(labels, ncells, natm):
    """
    Translate AO / LO labels to supercell.
    
    Args:
        labels: a list of string, 'id atom nlm'
        ncells: number of cells in the lattice

    Returns:
        labels_sc: labels in the supercell
    """
    labels_sp = []
    for lab in labels:
        lab_sp = lab.split()
        labels_sp.append(lab_sp)
    
    labels_sc = []
    for R in range(ncells):
        for lab in labels_sp:
            lab_new = " ".join([str(int(lab[0]) + R * natm), *lab[1:]])
            labels_sc.append(lab_new)
    return np.array(labels_sc)
