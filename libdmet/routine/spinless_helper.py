#! /usr/bin/env python

"""
Helper functions for spinless formalism.

Author:
    Zhihao Cui <zhcui0408@gmail.com>
"""

from functools import partial
import numpy as np

from pyscf import ao2mo

from libdmet.system import integral
from libdmet.routine.pbc_helper import *
from libdmet.routine.bcs_helper import mono_fit, mono_fit_2
from libdmet.routine.slater import (init_H2, restore_eri_local,
                                    get_emb_basis_other_cell)
from libdmet.utils import misc
from libdmet.utils.misc import mdot, max_abs, Iterable, format_idx
from libdmet.utils import logger as log
from libdmet.settings import IMAG_DISCARD_TOL

einsum = partial(np.einsum, optimize=True)

# *************************************************************************************************************
# Convenient functions
# *************************************************************************************************************

def separate_basis(basis, copy=False):
    """
    Separate the basis to alpha and beta spins.

    Args:
        basis: (nkpts, nso, nbasis) [can be in R or k space]
        copy: whether to make new array.

    Returns:
        basis_a: (nkpts, nao, nbasis)
        basis_b: (nkpts, nao, nbasis)
    """
    nkpts, nso, nbasis = basis.shape
    nao = nso // 2
    return np.array(basis[:, :nao], copy=copy), \
           np.array(basis[:, nao:], copy=copy)

def transform_spinless_mol(h1, D, h2):
    """
    Transform a Hamiltonian with singlet pairing to spinless form.
    h1 + D c^d c^d + h2

    Args:
        h1 (2, norb, norb), aa, bb, ab
        D  (norb, norb)
        h2 (3, norb, norb, norb, norb), aa, bb, ab

    Returns:
        Ham
    """
    h1_a, h1_b = h1
    h2_aa, h2_bb, h2_ab = h2
    norb = h1_a.shape[-1]

    H0 = 0.0
    H1 = np.zeros((3, norb, norb))
    H2 = np.zeros((3,) + (norb,)*4)

    # transform h1_a, h1_b, D
    H0 += h1_b.trace()
    H1[0] =  h1_a
    H1[1] = -h1_b.T
    H1[2] = D

    # transform h2_aa
    H2[0] = h2_aa

    # transform h2_bb
    H0 += 0.5 * (einsum('iikk->', h2_bb) - einsum('ijji->', h2_bb))
    H1[1] += (einsum('ijki -> jk', h2_bb) - einsum('ijkk -> ij', h2_bb))
    H2[1] = h2_bb

    # transform h2_ab and h2_ba
    H1[0] += einsum('ijkk -> ij', h2_ab)
    H2[2] = -np.swapaxes(h2_ab, -1, -2)

    return integral.Integral(norb, restricted=False, bogoliubov=False,
                             H0=H0, H1={"cd": H1}, H2={"ccdd": H2})

def Ham_compact2full(Ham):
    """
    Convert the compact format of spinless Ham to full format.

    full format:
        H1 has shape (nso, nso)
        H2 has shape (nso, nso, nso, nso)
    """
    norb = Ham.norb
    nso = norb * 2
    H0 = Ham.H0

    H1 = np.zeros((nso, nso))
    H1[:norb, :norb] = Ham.H1["cd"][0]
    H1[norb:, norb:] = Ham.H1["cd"][1]
    if Ham.H1["cd"].shape[0] == 3:
        H1[:norb, norb:] = Ham.H1["cd"][2]
        H1[norb:, :norb] = Ham.H1["cd"][2].conj().T

    H2 = misc.tile_eri(Ham.H2["ccdd"][0], Ham.H2["ccdd"][1], Ham.H2["ccdd"][2])

    return integral.Integral(nso, restricted=False, bogoliubov=False,
                             H0=H0, H1={"cd": H1[None]}, H2={"ccdd": H2[None]})

def Ham_full2compact(Ham):
    """
    Convert the full format of spinless Ham to compact format.

    compact format:
        H1 (3, norb, norb)
        H2 (3,) + (norb,)*4
    """
    nso = Ham.norb
    norb = nso // 2
    H0 = Ham.H0

    H1 = np.zeros((3, norb, norb))
    H1[0] = Ham.H1["cd"][:norb, :norb]
    H1[1] = Ham.H1["cd"][norb:, norb:]
    H1[2] = Ham.H1["cd"][:norb, norb:]

    H2 = np.asarray(misc.untile_eri(Ham.H2["ccdd"]))

    return integral.Integral(norb, restricted=False, bogoliubov=False,
                             H0=H0, H1={"cd": H1}, H2={"ccdd": H2})

def Ham_compact2uhf(Ham, eri_spin=1):
    """
    Convert the compact format of spinless Ham to uhf format.

    uhf format:
        H1 has shape (2, nso, nso)
        H2 has shape (eri_spin, nso, nso, nso, nso)
        eri_spin can be 1 or 3
    """
    norb = Ham.norb
    nso = norb * 2
    H0 = Ham.H0

    H1 = np.zeros((2, nso, nso))
    H1[0, :norb, :norb] = Ham.H1["cd"][0]
    H1[0, norb:, norb:] = Ham.H1["cd"][1]
    H1[0, :norb, norb:] = Ham.H1["cd"][2]
    H1[0, norb:, :norb] = Ham.H1["cd"][2].conj().T

    H2 = np.zeros((eri_spin,) + (nso,)*4)
    H2[0, :norb, :norb, :norb, :norb] = Ham.H2["ccdd"][0]
    H2[0, norb:, norb:, norb:, norb:] = Ham.H2["ccdd"][1]
    H2[0, :norb, :norb, norb:, norb:] = Ham.H2["ccdd"][2]
    H2[0, norb:, norb:, :norb, :norb] = \
            Ham.H2["ccdd"][2].transpose(3, 2, 1, 0).conj()
    return integral.Integral(nso, restricted=False, bogoliubov=False,
                             H0=H0, H1={"cd": H1}, H2={"ccdd": H2})

def Ham_uhf2compact(Ham):
    """
    Convert the uhf format of spinless Ham to compact format.
    """
    nso = Ham.norb
    norb = nso // 2
    H0 = Ham.H0

    H1 = np.zeros((3, norb, norb))
    H1[0] = Ham.H1["cd"][0, :norb, :norb]
    H1[1] = Ham.H1["cd"][0, norb:, norb:]
    H1[2] = Ham.H1["cd"][0, :norb, norb:]

    H2 = np.zeros((3,) + (norb,)*4)
    H2[0] = Ham.H2["ccdd"][0, :norb, :norb, :norb, :norb]
    H2[1] = Ham.H2["ccdd"][0, norb:, norb:, norb:, norb:]
    H2[2] = Ham.H2["ccdd"][0, :norb, :norb, norb:, norb:]

    return integral.Integral(norb, restricted=False, bogoliubov=False,
                             H0=H0, H1={"cd": H1}, H2={"ccdd": H2})

def extract_rdm1(GRho):
    """
    Note:
        Generalized density matrix convention:
        GRho = [[rho_A    k_ab]
                [k_ab^dg  1-rho_B]]
        k_ba = -k_ab.T
    """
    norbs = GRho.shape[0] // 2
    log.eassert(norbs * 2 == GRho.shape[0],
                "generalized density matrix dimension error")
    rhoA = np.array(GRho[:norbs, :norbs], copy=True)
    rhoB = np.eye(norbs) - GRho[norbs:, norbs:].T
    kappaAB = np.array(GRho[:norbs, norbs:], copy=True)
    return rhoA, rhoB, kappaAB

extractRdm = extract_rdm1

def extract_rdm12(GRdm1, GRdm2):
    """
    Extract the rdm1 (in aa, bb, ab order)
    and normal rdm2 (in aaaa, bbbb, aabb order)
    from the generalized GRdm1 and GRdm2.

    Args:
        GRdm1: generalized rdm1, (nso, nso)
        GRdm2: generalized rdm2, (nso, nso, nso, nso)

    Returns:
        rdm1: (3, nao, nao) in aa, bb, ab order.
        rdm2: (3, nao, nao, nao, nao) in aaaa, bbbb, aabb order.
    """
    assert GRdm1.ndim == 2
    assert GRdm2.ndim == 4
    nso = GRdm2.shape[-1]
    nao = nso // 2
    I = np.eye(nao)

    GRdm1_aa = GRdm1[:nao, :nao]
    GRdm1_bb = GRdm1[nao:, nao:]
    GRdm1_ab = GRdm1[:nao, nao:]

    rdm1 = np.zeros((3, nao, nao))
    rdm1[0] = GRdm1_aa
    rdm1[1] = I - GRdm1_bb.T
    rdm1[2] = GRdm1_ab

    rdm2 = np.zeros((3, nao, nao, nao, nao))
    rdm2[0] = GRdm2[:nao, :nao, :nao, :nao]

    rdm2[1]  = einsum('pq, rs -> pqrs', I, I)
    rdm2[1] -= einsum('pq, rs -> pqrs', GRdm1_bb, I)
    rdm2[1] -= einsum('rq, ps -> pqrs', I, I)
    rdm2[1] += einsum('rq, ps -> pqrs', GRdm1_bb, I)
    rdm2[1] += einsum('ps, rq -> pqrs', GRdm1_bb, I)
    rdm2[1] -= einsum('rs, pq -> pqrs', GRdm1_bb, I)
    rdm2[1] += GRdm2[nao:, nao:, nao:, nao:].transpose(3, 2, 1, 0)

    rdm2[2] = einsum('qp, rs -> pqrs', GRdm1_aa, I)
    rdm2[2] -= GRdm2[:nao, :nao, nao:, nao:].transpose(0, 1, 3, 2)
    return rdm1, rdm2

def idx_ao2so(idx_list, nao):
    """
    Given index in atomic orbitals, return index in the spin orbitals.

    Args:
        idx_list: index list.
        nao: number of AO.

    Returns:
        idx_a
        idx_b
    """
    return [idx for idx in idx_list], [idx + nao for idx in idx_list]

def get_H2_mask(nao, neo):
    """
    Get mask for fill the impurity block of GH2 with unit H2.

    Args:
        nao: number of AO.
        neo: number of EO.

    Returns:
        mask_aa: mask of alpha-alpha.
        mask_bb: mask of beta-beta.
        mask_ab: mask of alpha-beta.
        mask_ba: mask of beta-alpha.
    """
    idx_a = np.arange(nao)
    idx_b = np.arange(nao, nao*2)
    tril_idx = np.tril_indices(neo)
    idx_in_a = np.isin(tril_idx, idx_a)
    idx_in_b = np.isin(tril_idx, idx_b)
    mask_a = np.logical_and(*idx_in_a)
    mask_b = np.logical_and(*idx_in_b)
    mask_aa = np.ix_(mask_a, mask_a)
    mask_bb = np.ix_(mask_b, mask_b)
    mask_ab = np.ix_(mask_a, mask_b)
    mask_ba = np.ix_(mask_b, mask_a)
    return mask_aa, mask_bb, mask_ab, mask_ba

def unit2emb(H2_unit, neo):
    """
    Allocate H2_emb and fill the impurity block with H2_unit.
    Assume 4-fold symmetry.

    Args:
        H2_unit: unit ERI, shape (3, nao_pair, nao_pair).
                 aa, bb, ab order.
        neo: number of embedding orbitals.

    Returns:
        H2_emb: embedding ERI, with impurity block filled by H2_unit.
                4-fold symmetry, shape (1, neo_pair, neo_pair).
    """
    assert H2_unit.ndim == 3
    spin_pair = H2_unit.shape[0]
    assert spin_pair == 3
    nao = int(np.sqrt(H2_unit.shape[-1] * 2))
    mask_aa, mask_bb, mask_ab, mask_ba = get_H2_mask(nao, neo)

    H2_emb = init_H2(neo, 4)
    H2_emb[mask_aa] = H2_unit[0]
    H2_emb[mask_bb] = H2_unit[1]
    H2_emb[mask_ab] = H2_unit[2]
    H2_emb[mask_ba] = H2_unit[2].T
    return H2_emb

# *************************************************************************************************************
# DMET functions
# *************************************************************************************************************

def transform_eri_local(basis_Ra, basis_Rb, H2, symm=4):
    """
    Transform the spin local H2 to embedding space.
    Used for interacting bath formalism.

    Args:
        basis_Ra: (ncells, nao, neo)
        basis_Rb: (ncells, nao, neo)
        H2: (3, nao, nao, nao, nao) or (3, nao_pair, nao_pair)

    Returns:
        GH2_emb: (neo,)*4 or (neo_pair, neo_pair)
    """
    ncells, nao, neo = basis_Ra.shape
    neo_pair = neo * (neo + 1) // 2
    H2 = restore_eri_local(H2, nao)

    GH2_emb = np.zeros((neo_pair, neo_pair))
    for i in range(ncells):
        GH2_emb += ao2mo.incore.general(H2[0], (basis_Ra[i], basis_Ra[i],
                                                basis_Ra[i], basis_Ra[i]), compact=True)
        GH2_emb += ao2mo.incore.general(H2[1], (basis_Rb[i], basis_Rb[i],
                                                basis_Rb[i], basis_Rb[i]), compact=True)
        tmp = ao2mo.incore.general(H2[2], (basis_Ra[i], basis_Ra[i],
                                           basis_Rb[i], basis_Rb[i]), compact=True)
        GH2_emb += tmp
        GH2_emb += tmp.T
    GH2_emb = ao2mo.restore(symm, GH2_emb, neo)
    return GH2_emb

def transform_trans_inv_k(basis_ka, basis_kb, H_k):
    """
    Transform a translational invariant quantity from LO to EO,
    using k-basis and k-one-particle quantities.

    Args:
        basis_ka: (nkpts, nao, nbasis)
        basis_kb: (nkpts, nao, nbasis)
        H_k: (2 or 3, nkpts, nao, nao)

    Returns:
        GH_emb: (nbasis, nbasis)
    """
    H_k = np.asarray(H_k)
    assert H_k.ndim == 4
    assert (H_k.shape[0] == 2) or (H_k.shape[0] == 3)
    nkpts, nao, nbasis = basis_ka.shape

    GH_emb = np.zeros((nbasis, nbasis), dtype=np.complex128)
    # aa, bb contribution
    for k in range(nkpts):
        GH_emb += mdot(basis_ka[k].conj().T, H_k[0, k], basis_ka[k])
        GH_emb += mdot(basis_kb[k].conj().T, H_k[1, k], basis_kb[k])
    # ab and ba contribution
    if H_k.shape[0] == 3:
        for k in range(nkpts):
            tmp = mdot(basis_ka[k].conj().T, H_k[2, k], basis_kb[k])
            GH_emb += tmp
            GH_emb += tmp.conj().T
    if max_abs(GH_emb.imag) > IMAG_DISCARD_TOL:
        log.warn("transform_trans_inv_k: has imag part %s", max_abs(GH_emb.imag))
    GH_emb = GH_emb.real / float(nkpts)
    return GH_emb

def transform_local(basis_Ra, basis_Rb, H):
    """
    Transform a local quantity from LO to EO.

    Args:
        basis_Ra: (ncells, nao, nbasis)
        basis_Rb: (ncells, nao, nbasis)
        H: (2 or 3, nao, nao)

    Returns:
        GH_emb: (nbasis, nbasis)
    """
    H = np.asarray(H)
    assert (H.shape[0] == 2) or (H.shape[0] == 3)
    ncells, nao, nbasis = basis_Ra.shape
    res = np.zeros((nbasis, nbasis))
    # aa and bb
    for i in range(ncells):
        res += mdot(basis_Ra[i].conj().T, H[0], basis_Ra[i])
        res += mdot(basis_Rb[i].conj().T, H[1], basis_Rb[i])
    # ab and ba
    if H.shape[0] == 3:
        for i in range(ncells):
            tmp = mdot(basis_Ra[i].conj().T, H[2], basis_Rb[i])
            res += tmp
            res += tmp.conj().T
    return res

def transform_imp(basis_Ra, basis_Rb, H):
    """
    Transform a local quantity from LO to EO,
    only keep the impurity part.

    Args:
        basis_Ra: (ncells, nao, nbasis)
        basis_Rb: (ncells, nao, nbasis)
        H: (2 or 3, nao, nao)

    Returns:
        GH_emb: (nbasis, nbasis)
    """
    H = np.asarray(H)
    assert (H.shape[0] == 2) or (H.shape[0] == 3)
    ncells, nao, nbasis = basis_Ra.shape
    res = np.zeros((nbasis, nbasis))
    # aa and bb
    res += mdot(basis_Ra[0].conj().T, H[0], basis_Ra[0])
    res += mdot(basis_Rb[0].conj().T, H[1], basis_Rb[0])
    # ab and ba
    if H.shape[0] == 3:
        tmp = mdot(basis_Ra[0].conj().T, H[2], basis_Rb[0])
        res += tmp
        res += tmp.conj().T
    return res

# *************************************************************************************************************
# global density matrix
# *************************************************************************************************************

def get_rho_glob_R(basis, lattice, rho_emb, symmetric=True, compact=True,
                   sign=None):
    """
    Get rho_glob in site basis, in stripe shape.
    Use democratic partitioning.
    Average of the IJ blocks from I and from J impurity problem.

    Args:
        basis: C_lo_eo, (ncells, nso, neo), or list of C_so_eo.
        lattice: lattice object, or list of lattices
        rho_emb: rdm1, (neo, neo), or list of rdm1.

    Returns:
        rho_glob_R: global rdm1,
                    if compact (ncells, nso, nso)
                    else       (ncells*nso, ncells*nso).
    """
    if not isinstance(lattice, Iterable):
        basis_col = [basis]
        lattice_col = [lattice]
        rho_emb_col = [rho_emb]
    else:
        basis_col = basis
        lattice_col = lattice
        rho_emb_col = rho_emb

    if sign is None:
        sign = np.ones(len(lattice_col), dtype=int)
    else:
        sign = np.asarray(sign)
        compact = False
    assert len(sign) == len(lattice_col)

    rho_glob = 0.0
    I_idx = 0

    for basis_I, lattice_I, rho_emb_I, sign_I in zip(basis_col, lattice_col,
                                                     rho_emb_col, sign):
        log.debug(0, "Build rdm1_glob, impurity %s, indices: %s, sign: %s",
                  I_idx, format_idx(lattice_I.imp_idx), sign_I)
        basis_I = np.asarray(basis_I)
        ncells, nso, _ = basis_I.shape

        if compact:
            rho_R = np.zeros((ncells * nso, nso))
            for R in range(ncells):
                basis_other = get_emb_basis_other_cell(lattice_I, basis_I, R)
                imp_idx = np.asarray(lattice_I.imp_idx)
                imp_idx = np.hstack((imp_idx, imp_idx + nso // 2))
                imp_idx = np.asarray(imp_idx) + R * nso
                env_idx = np.where(~np.isin(np.arange(ncells * nso), imp_idx))[0]
                tmp = np.isin(np.arange(nso), imp_idx)
                imp_idx_0 = np.where(tmp)[0]
                env_idx_0 = np.where(~tmp)[0]
                imp_env = np.ix_(imp_idx, env_idx_0)
                env_imp = np.ix_(env_idx, imp_idx_0)
                env_env = np.ix_(env_idx, env_idx_0)

                log.debug(3, "rdm1_glob: R %s", R)
                C_R = basis_other.reshape(-1, basis_other.shape[-1])
                rdm1_R = mdot(C_R, rho_emb_I, C_R[:nso].conj().T)
                rdm1_R[imp_env] *= 0.5
                rdm1_R[env_imp] *= 0.5
                rdm1_R[env_env]  = 0.0
                rho_R += rdm1_R
            rho_R = rho_R.reshape(ncells, nso, nso)
        else:
            rho_R = np.zeros((ncells * nso, ncells * nso))
            for R in range(ncells):
                basis_other = get_emb_basis_other_cell(lattice_I, basis_I, R)
                imp_idx = np.asarray(lattice_I.imp_idx)
                imp_idx = np.hstack((imp_idx, imp_idx + nso // 2))
                imp_idx = (np.asarray(imp_idx) + R * nso) % (ncells * nso)
                env_idx = np.where(~np.isin(np.arange(ncells * nso), imp_idx))[0]
                imp_env = np.ix_(imp_idx, env_idx)
                env_imp = np.ix_(env_idx, imp_idx)
                env_env = np.ix_(env_idx, env_idx)

                log.debug(3, "rdm1_glob: R %s", R)
                C_R = basis_other.reshape(-1, basis_other.shape[-1])
                rdm1_R = mdot(C_R, rho_emb_I, C_R.conj().T)
                rdm1_R[imp_env] *= 0.5
                rdm1_R[env_imp] *= 0.5
                rdm1_R[env_env]  = 0.0
                rho_R += rdm1_R

        rho_glob += (rho_R * sign_I)
        I_idx += 1
    return rho_glob

def get_rho_glob_k(basis, lattice, rho_emb, symmetric=True, compact=True,
                   sign=None):
    if sign is not None:
        compact = False

    rho_R = get_rho_glob_R(basis, lattice, rho_emb, symmetric=symmetric,
                           compact=compact, sign=sign)

    if isinstance(lattice, Iterable):
        if compact:
            rho_k = lattice[0].R2k(rho_R)
        else:
            rho_k = lattice[0].R2k(lattice[0].extract_stripe(rho_R))
    else:
        if compact:
            rho_k = lattice.R2k(rho_R)
        else:
            rho_k = lattice.R2k(lattice.extract_stripe(rho_R))
    return rho_k

def get_rho_glob_full(basis, lattice, rho_emb, symmetric=True, compact=True,
                      sign=None):
    """
    Get rho_glob in site basis, in full shape.
    Use democratic partitioning.
    """
    if sign is not None:
        compact = False
    rho_glob_R = get_rho_glob_R(basis, lattice, rho_emb, symmetric=symmetric,
                                compact=compact, sign=sign)
    if compact:
        if isinstance(lattice, Iterable):
            rho_glob_full = lattice[0].expand(rho_glob_R)
        else:
            rho_glob_full = lattice.expand(rho_glob_R)
    else:
        rho_glob_full = rho_glob_R
    return rho_glob_full


