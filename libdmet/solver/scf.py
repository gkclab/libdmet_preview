#! /usr/bin/env python

"""
scf solver for impurity problem.
This module includes:

    mean-field routines:
        - SCF class
            - HF (RHF, UHF, UIHF)
            - HFB
            - GHF (GIHF)
        - _get_veff etc.

    integral transform:
        - ao2mo incore
            - orb-res,   eri-res
            - orb-unres, eri-res
            - orb-unres, eri-unres
        - ao2mo for Ham class
        - restore for Ham class

Author:
    Zhihao Cui <zhcui0408@gmail.com>
    Bo-Xiao Zheng <boxiao.zheng@gmail.com>
"""

import numpy as np
import scipy.linalg as la
from functools import reduce

import pyscf
from pyscf import gto, ao2mo, scf, mp
from pyscf.scf import hf
from pyscf.soscf import newton_ah
from pyscf.mp.mp2 import MP2
from pyscf import lib
import pyscf.lib.logger as pyscflogger

from libdmet.system import integral
from libdmet.solver.mp import UIMP2, GGMP2
from libdmet.utils.misc import mdot, add_spin_dim, max_abs
from libdmet.routine.bcs_helper import extractRdm, extractH1
from libdmet import settings
from libdmet.utils import logger as log
flush = log.flush_for_pyscf

# *********************************************************************
# ao2mo routines
# *********************************************************************

def regularize_coeff(mo_coeffs):
    """
    Given 4 sets of MO coefficients,
    regularize them to the following format:
    [lst0, lst1, lst2, lst3]
    each lst can be a ndarray or [ndarray0, ndarray1]
    """
    mo_lst = []
    for mo in mo_coeffs:
        if isinstance(mo, np.ndarray) and mo.ndim == 2:
            mo_lst.append(mo)
            spin = 1
        else:
            spin = len(mo)
            if spin == 1:
                mo_lst.append(mo[0])
            else:
                mo_lst.append(mo)
    return mo_lst, spin

def incore_transform(eri, c, compact=False):
    """
    Simple 2e integral transformation.
    c is a tuple of coefficients.
    The returned ERI always has a spin dimension.
    """
    log.eassert(len(c) == 4, "Need 4 coefficents for transformation.")
    # regularize the format of coeff
    c, spin = regularize_coeff(c)

    if spin == 1:
        nao = c[0].shape[-2]
        eri_format, spin_dim = integral.get_eri_format(eri, nao)
        if spin_dim == 1:
            eri = eri[0]
        elif spin_dim > 1 or spin_dim < 0:
            raise ValueError
        eri_mo = ao2mo.incore.general(eri, c, compact=compact)
        eri_mo = eri_mo[np.newaxis]
    elif spin == 2:
        nao = c[0][0].shape[-2]
        eri_format, spin_dim = integral.get_eri_format(eri, nao)
        if spin_dim == 0:
            eri_aa = ao2mo.general(eri, (c[0][0], c[1][0], c[2][0], c[3][0]),
                                   compact=compact)
            eri_bb = ao2mo.general(eri, (c[0][1], c[1][1], c[2][1], c[3][1]),
                                   compact=compact)
            eri_ab = ao2mo.general(eri, (c[0][0], c[1][0], c[2][1], c[3][1]),
                                   compact=compact)
        elif spin_dim == 1:
            eri_aa = ao2mo.general(eri[0], (c[0][0], c[1][0], c[2][0], c[3][0]),
                                   compact=compact)
            eri_bb = ao2mo.general(eri[0], (c[0][1], c[1][1], c[2][1], c[3][1]),
                                   compact=compact)
            eri_ab = ao2mo.general(eri[0], (c[0][0], c[1][0], c[2][1], c[3][1]),
                                   compact=compact)
        elif spin_dim == 3: # aa, bb, ab
            eri_aa = ao2mo.general(eri[0], (c[0][0], c[1][0], c[2][0], c[3][0]),
                                   compact=compact)
            eri_bb = ao2mo.general(eri[1], (c[0][1], c[1][1], c[2][1], c[3][1]),
                                   compact=compact)
            eri_ab = ao2mo.general(eri[2], (c[0][0], c[1][0], c[2][1], c[3][1]),
                                   compact=compact)
        else:
            raise ValueError
        eri_mo = np.asarray((eri_aa, eri_bb, eri_ab))
    else:
        raise ValueError("Incorrect spin dimension (%s) of mo_coeff." %spin)
    return eri_mo

def ao2mo_Ham(Ham, C, compact=True, in_place=False):
    """
    Given Ham and mo_coeff C, return MO transformed Ham.
    eri will be convert to 4-fold symmetry.
    """
    norb = Ham.norb
    if Ham.bogoliubov:
        raise NotImplementedError
    if C.ndim == 2:
        C = C[np.newaxis]
    if Ham.restricted:
        h1e = mdot(C[0].conj().T, Ham.H1["cd"][0], C[0])[np.newaxis]
        eri = ao2mo.restore(8, Ham.H2["ccdd"][0], norb)
        eri = ao2mo.full(eri, C[0], compact=compact)[np.newaxis]
    else: # unrestricted case
        assert C.ndim == 3 and C.shape[0] == 2
        norb_pair = norb * (norb + 1) // 2

        # H1
        h1e = np.zeros((2, norb, norb))
        for s in range(2):
            if Ham.H1["cd"].shape[0] == 1:
                h1e[s] = mdot(C[s].conj().T, Ham.H1["cd"][0], C[s])
            else:
                h1e[s] = mdot(C[s].conj().T, Ham.H1["cd"][s], C[s])

        # H2
        if compact:
            eri = np.zeros((3, norb_pair, norb_pair))
        else:
            eri = np.zeros((3, norb * norb, norb * norb))

        if Ham.H2["ccdd"].shape[0] == 1: # res H2
            eri_aa = ao2mo.restore(8, Ham.H2["ccdd"][0], norb)
            eri[0] = ao2mo.full(eri_aa, C[0], compact=compact)

            eri_bb = eri_aa
            eri[1] = ao2mo.full(eri_bb, C[1], compact=compact)

            eri_ab = eri_aa
            eri[2] = ao2mo.general(eri_ab, (C[0], C[0], C[1], C[1]),
                                   compact=compact)
            eri_aa = eri_bb = eri_ab = None
        elif Ham.H2["ccdd"].shape[0] == 3: # unres H2 aa, bb, ab
            eri_aa = ao2mo.restore(8, Ham.H2["ccdd"][0], norb)
            eri[0] = ao2mo.full(eri_aa, C[0], compact=compact)
            eri_aa = None

            eri_bb = ao2mo.restore(8, Ham.H2["ccdd"][1], norb)
            eri[1] = ao2mo.full(eri_bb, C[1], compact=compact)
            eri_bb = None

            eri_ab = ao2mo.restore(4, Ham.H2["ccdd"][2], norb)
            eri[2] = ao2mo.general(eri_ab, (C[0], C[0], C[1], C[1]),
                                   compact=compact)
            eri_ab = None
        else:
            raise ValueError

    if not compact:
        eri = eri.reshape((-1, norb, norb, norb, norb))
    if in_place:
        Ham.H1["cd"] = h1e
        Ham.H2["ccdd"] = eri
        return Ham
    else:
        Ham_mo = integral.Integral(norb, Ham.restricted, Ham.bogoliubov, Ham.H0,
                {"cd": h1e}, {"ccdd": eri})
        return Ham_mo

def restore_Ham(Ham, symm, in_place=True):
    norb = Ham.norb
    if Ham.bogoliubov:
        raise NotImplementedError
    if Ham.restricted:
        eri = ao2mo.restore(symm, Ham.H2["ccdd"][0], norb)[np.newaxis]
    else: # unrestricted case
        norb_pair = norb*(norb+1) // 2
        if Ham.H2["ccdd"].shape[0] == 1: # res H2
            if symm == 8:
                norb_pair_pair = norb_pair*(norb_pair+1) // 2
                eri = np.zeros((1, norb_pair_pair), dtype=np.double)
            elif symm == 4:
                eri = np.zeros((1, norb_pair, norb_pair), dtype=np.double)
            else:
                eri = np.zeros((1, norb, norb, norb, norb), dtype=np.double)
            eri[0] = ao2mo.restore(symm, Ham.H2["ccdd"][0], norb)
        elif Ham.H2["ccdd"].shape[0] == 3: # unres H2
            if symm == 8:
                raise ValueError
            elif symm == 4:
                eri = np.zeros((3, norb_pair, norb_pair), dtype=np.double)
            else:
                eri = np.zeros((3, norb, norb, norb, norb), dtype=np.double)
            eri[0] = ao2mo.restore(symm, Ham.H2["ccdd"][0], norb)
            eri[1] = ao2mo.restore(symm, Ham.H2["ccdd"][1], norb)
            eri[2] = ao2mo.restore(symm, Ham.H2["ccdd"][2], norb)
        else:
            raise ValueError

    if in_place:
        Ham.H2["ccdd"] = eri
        return Ham
    else:
        raise NotImplementedError

# *********************************************************************
# Restricted (RHF) and Unrestricted (integral) Hartree-Fock [U(I)HF]
# *********************************************************************

class RIHF(scf.hf.RHF):
    def __init__(self, mol, DiisDim=12, MaxIter=50, alpha=None):
        scf.hf.RHF.__init__(self, mol)
        self.diis_space = DiisDim
        self.max_cycle = MaxIter
        self.alpha = alpha
        self._keys = self._keys.union(["h1e", "ovlp", "Mu", "alpha"])

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None):
        # Note the incore version, which initializes an _eri array in memory.
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if (not omega and
            (self._eri is not None or mol.incore_anyway or self._is_mem_enough())):
            if self._eri is None:
                self._eri = mol.intor('int2e', aosym='s8')
            vj, vk = scf.hf.dot_eri_dm(self._eri, dm, hermi, with_j, with_k)
        else:
            vj, vk = scf.hf.SCF.get_jk(self, mol, dm, hermi, with_j, with_k, omega)
        if self.alpha is not None:
            vk *= self.alpha
        return vj, vk

def _get_jk(dm, eri, with_j=True, with_k=True):
    """
    Get J and K potential from rdm and ERI.
    For RHF, UHF, UIHF, GGHF.

    Math:
    vj00 = np.tensordot(dm[0], eri[0], ((0,1), (0,1))) # J a from a
    vj11 = np.tensordot(dm[1], eri[1], ((0,1), (0,1))) # J b from b
    vj10 = np.tensordot(dm[0], eri[2], ((0,1), (0,1))) # J b from a
    vj01 = np.tensordot(dm[1], eri[2], ((1,0), (3,2))) # J a from b
    vk00 = np.tensordot(dm[0], eri[0], ((0,1), (0,3))) # K a from a
    vk11 = np.tensordot(dm[1], eri[1], ((0,1), (0,3))) # K b from b
    JK = np.asarray([vj00 + vj01 - vk00, vj11 + vj10 - vk11])

    PySCF dot_eri_dm convention:
    J: ijkl, kl -> ij
    K: ijkl, il -> jk

    Args:
        dm: rdm1, ((spin), nao, nao)
        eri: ERI, can have spin dimension, s1 or s4 or s8.
        with_j: calculate J
        with_k: calculate K

    Returns:
        vj: (spin, nao, nao), or (2, 2, nao, nao) for UIHF.
        vk: (spin, nao, nao)
    """
    dm = np.asarray(dm, dtype=np.double)
    old_shape = dm.shape
    if dm.ndim == 2:
        dm = dm[np.newaxis]
    spin = dm.shape[0]
    nao = dm.shape[-1]
    eri = np.asarray(eri, dtype=np.double)
    eri_format, spin_dim = integral.get_eri_format(eri, nao)
    if spin_dim == 0:
        eri = eri[None]
        spin_dim = 1

    if spin == 1:
        if eri_format == 's1':
            eri = ao2mo.restore(8, eri[0], nao)
        else:
            eri = eri[0]
        vj, vk = hf.dot_eri_dm(eri, dm, hermi=1, with_j=with_j, with_k=with_k)
    else:
        if spin_dim == 1: # UHF
            if eri_format == 's1':
                eri = ao2mo.restore(8, eri[0], nao)
            else:
                eri = eri[0]
            vj, vk = hf.dot_eri_dm(eri, dm, hermi=1, with_j=with_j,
                                   with_k=with_k)
        elif spin_dim == 3: # UIHF
            assert dm.shape[0] == 2
            eri_aa = ao2mo.restore(4, eri[0], nao)
            vj00, vk00 = hf.dot_eri_dm(eri_aa, dm[0], hermi=1, with_j=with_j,
                                       with_k=with_k)
            eri_aa = None

            eri_bb = ao2mo.restore(4, eri[1], nao)
            vj11, vk11 = hf.dot_eri_dm(eri_bb, dm[1], hermi=1, with_j=with_j,
                                       with_k=with_k)
            eri_bb = None

            eri_ab = ao2mo.restore(4, eri[2], nao)
            vj01 = hf.dot_eri_dm(eri_ab, dm[1], hermi=1, with_j=with_j,
                                 with_k=False)[0]
            # NOTE the transpose, ijkl, kl -> ij
            vj10 = hf.dot_eri_dm(eri_ab.T, dm[0], hermi=1, with_j=with_j,
                                 with_k=False)[0]
            eri_ab = None

            # NOTE explicit write down vj, without broadcast
            vj = np.asarray(((vj00, vj11), (vj01, vj10)))
            vk = np.asarray((vk00, vk11))
        else:
            raise ValueError
    return vj, vk

def _get_veff(dm, eri):
    """
    Get HF effective potential from rdm and ERI.
    For RHF and UHF.

    veff shape (spin, nao, nao)
    """
    dm = np.asarray(dm, dtype=np.double)
    if dm.ndim == 2:
        dm = dm[np.newaxis]
    spin = dm.shape[0]
    vj, vk = _get_jk(dm, eri)
    if spin == 1:
        veff = vj - vk * 0.5
    else:
        veff = vj[0] + vj[1] - vk
    return veff

class UIHF(scf.uhf.UHF):
    """
    A routine for unrestricted HF with integrals
    different for two spin species
    """
    def __init__(self, mol, DiisDim=12, MaxIter=50, alpha=None):
        scf.uhf.UHF.__init__(self, mol)
        self._keys = self._keys.union(["h1e", "ovlp", "Mu", "alpha"])
        self.direct_scf = False
        self.diis_space = DiisDim
        self.max_cycle = MaxIter
        self.h1e = None
        self.ovlp = None
        self.Mu = None
        self.alpha = alpha

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True):
        '''Coulomb (J) and exchange (K)

        Args:
            dm : a list of 2D arrays or a list of 3D arrays
                (alpha_dm, beta_dm) or (alpha_dms, beta_dms)
        '''
        # ZHC TODO implement save_mem
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        if self._eri is not None or mol.incore_anyway or self._is_mem_enough():
            if self._eri is None:
                log.error("SCF eri is not initialized.")
                self._eri = mol.intor('int2e', aosym='s8')

            vj, vk = _get_jk(dm, self._eri)
        else:
            log.error("Direct SCF not implemented")
            vj, vk = hf.SCF.get_jk(self, mol, dm, hermi, with_j, with_k)
        if self.alpha is not None:
            vk *= self.alpha
        return vj, vk

    def eig(self, fock, s):
        """
        Allow s has spin dimension.
        """
        s = np.asarray(s)
        if s.ndim == 2:
            s = (s, s)
        e_a, c_a = self._eigh(fock[0], s[0])
        e_b, c_b = self._eigh(fock[1], s[1])
        return np.array((e_a, e_b)), np.array((c_a, c_b))

    def energy_elec(self, dm=None, h1e=None, vhf=None):
        """
        Electronic part of energy.
        Remove the contribution of Mu if exists.
        """
        if dm is None: dm = self.make_rdm1()
        if h1e is None: h1e = self.get_hcore()
        if vhf is None: vhf = self.get_veff(self.mol, dm)
        ovlp = self.get_ovlp()
        h1e_no_mu = np.array(h1e, copy=True)
        if self.Mu is not None:
            nao = h1e_no_mu.shape[-1] // 2
            h1e_no_mu[0, :nao, :nao] += ovlp[:nao, :nao] * self.Mu
            h1e_no_mu[0, nao:, nao:] -= ovlp[nao:, nao:] * self.Mu
        e1 = np.einsum('spq, sqp', h1e_no_mu, dm)
        e_coul = 0.5 * np.einsum('spq, sqp', vhf, dm)
        log.debug(1, "E_coul = %.15f", e_coul)
        return e1 + e_coul, e_coul

    def _finalize(self):
        # ZHC NOTE FIXME
        # how to calculate the ss, s when ovlp is spin dependent?
        ovlp = np.asarray(self.get_ovlp())
        if ovlp.ndim == 2:
            ss, s = self.spin_square()
            if self.converged:
                pyscflogger.note(self, 'converged SCF energy = %.15g  '
                            '<S^2> = %.8g  2S+1 = %.8g', self.e_tot, ss, s)
            else:
                pyscflogger.note(self, 'SCF not converged.')
                pyscflogger.note(self, 'SCF energy = %.15g after %d cycles  '
                            '<S^2> = %.8g  2S+1 = %.8g',
                            self.e_tot, self.max_cycle, ss, s)
        else:
            if self.converged:
                pyscflogger.note(self, 'converged SCF energy = %.15g  ', self.e_tot)
            else:
                pyscflogger.note(self, 'SCF not converged.')
                pyscflogger.note(self, 'SCF energy = %.15g after %d cycles  ',
                            self.e_tot, self.max_cycle)
        return self

    def init_guess_by_1e(self, mol=None, breaksym=False):
        if mol is None: mol = self.mol
        log.debug(0, 'Initial guess from hcore.')
        h1e = self.get_hcore(mol)
        s1e = self.get_ovlp(mol)
        mo_energy, mo_coeff = self.eig(h1e, s1e)
        mo_occ = self.get_occ(mo_energy, mo_coeff)
        dma, dmb = self.make_rdm1(mo_coeff, mo_occ)
        return np.asarray((dma, dmb))

    def get_hcore(self, *args):
        return self.h1e

    def get_ovlp(self, *args):
        return self.ovlp

# *********************************************************************
# Unrestricted Hartree-Fock Bogoliubov (UHFB)
# ZHC TODO implement permutation symmetry for BCS.
# *********************************************************************

def _UHFB_get_grad(mo_coeff, mo_occ, fock_ao):
    '''UHFB orbital gradients

    Args:
        mo_coeff : 2D ndarray
            Obital coefficients
        mo_occ : 1D ndarray
            Orbital occupancy
        fock_ao : 2D ndarray
            Fock matrix in AO representation

    Returns:
        Gradients in MO representation.  It's a num_occ*num_vir vector.
    '''
    occidx = mo_occ > 0
    viridx = ~occidx
    g = reduce(np.dot, (mo_coeff[:,viridx].conj().T, fock_ao,
                           mo_coeff[:,occidx]))
    return g.ravel()

def _get_veff_bcs(rhoA, rhoB, kappaBA, eri):
    """
    get_veff for UHFB.
    Assume no cccd and cccc terms.
    """
    eriA, eriB, eriAB = eri
    vj00 = np.tensordot(rhoA, eriA, ((0,1), (0,1)))
    vj11 = np.tensordot(rhoB, eriB, ((0,1), (0,1)))
    vj10 = np.tensordot(rhoA, eriAB, ((0,1), (0,1)))
    vj01 = np.tensordot(eriAB, rhoB, ((2,3), (0,1)))
    vk00 = np.tensordot(rhoA, eriA, ((0,1), (0,3)))
    vk11 = np.tensordot(rhoB, eriB, ((0,1), (0,3)))
    vl10 = np.tensordot(kappaBA, eriAB, ((1,0), (0,2))) # wrt kappa_ba
    va = vj00 + vj01 - vk00
    vb = vj11 + vj10 - vk11
    # ZHC NOTE should be vl01?
    vd = vl10
    return va, vb, vd

def _get_veff_bcs_full(rhoA, rhoB, kappaBA, eri, eri2, eri4):
    """
    get_veff for UHFB.
    Assume has cccd and cccc terms.
    """
    eriA, eriB, eriAB = eri
    eri2A, eri2B = eri2
    eri4AB = eri4[0]
    vj00 = np.tensordot(rhoA, eriA, ((0,1), (0,1)))
    vj11 = np.tensordot(rhoB, eriB, ((0,1), (0,1)))
    vj10 = np.tensordot(rhoA, eriAB, ((0,1), (0,1)))
    vj01 = np.tensordot(eriAB, rhoB, ((2,3), (0,1)))
    vk00 = np.tensordot(rhoA, eriA, ((0,1), (0,3)))
    vk11 = np.tensordot(rhoB, eriB, ((0,1), (0,3)))
    vl10 = np.tensordot(kappaBA, eriAB, ((1,0), (0,2)))
    vy00 = -np.tensordot(kappaBA, eri2A, ((1,0), (0,2)))
    vy11 = np.tensordot(kappaBA, eri2B, ((0,1), (0,2)))
    vy10 = np.tensordot(rhoA, eri2A, ((0,1), (0,3))) - \
           np.tensordot(rhoB, eri2B, ((0,1), (0,3))).T
    vx10 = np.tensordot(kappaBA, eri4AB, ((1,0), (0,2)))
    va = vj00 + vj01 - vk00 + vy00 + vy00.T
    vb = vj11 + vj10 - vk11 + vy11 + vy11.T
    vd = vl10 + vy10 - vx10
    return va, vb, vd

def _get_veff_bcs_save_mem(rhoA, rhoB, kappaBA, _eri):
    """
    get_veff for UHFB.
    For save_mem == True.
    """
    eri = _eri[0]
    nImp = eri.shape[0]
    rhoAI = rhoA[:nImp, :nImp]
    rhoBI = rhoB[:nImp, :nImp]
    kappaBAI = kappaBA[:nImp, :nImp]
    vj00 = np.tensordot(rhoAI, eri, ((0,1), (0,1)))
    vj11 = np.tensordot(rhoBI, eri, ((0,1), (0,1)))
    vj10 = vj00
    vj01 = vj11
    vk00 = np.tensordot(rhoAI, eri, ((0,1), (0,3)))
    vk11 = np.tensordot(rhoBI, eri, ((0,1), (0,3)))
    vl10 = np.tensordot(kappaBAI, eri, ((1,0), (0,2)))# wrt kappa_ba
    va = vj00 + vj01 - vk00
    vb = vj11 + vj10 - vk11
    vd = vl10
    return va, vb, vd

class UHFB(hf.RHF):
    """
    Main class for UHFB calculations.
    """
    def __init__(self, mol, DiisDim=12, MaxIter=50):
        hf.RHF.__init__(self, mol)
        self._keys = self._keys.union(["h1e", "ovlp", "norb", "Mu"])
        self.direct_scf = False
        self.diis_space = DiisDim
        self.max_cycle = MaxIter
        self.h1e = None
        self.ovlp = None
        hf.get_grad = _UHFB_get_grad

    def get_veff(self, mol, dm, dm_last=0, vhf_last=0, hermi=1):
        assert self._eri is not None
        rhoA, rhoB, kappaBA = extractRdm(dm)

        if settings.save_mem:
            va, vb, vd = _get_veff_bcs_save_mem(rhoA, rhoB, kappaBA,
                                                self._eri["ccdd"])
        elif self._eri["cccd"] is None or (max_abs(self._eri["cccd"]) < 1e-12):
            va, vb, vd = _get_veff_bcs(rhoA, rhoB, kappaBA, self._eri["ccdd"])
        else:
            va, vb, vd = _get_veff_bcs_full(rhoA, rhoB, kappaBA,
                                            self._eri["ccdd"],
                                            self._eri["cccd"],
                                            self._eri["cccc"])

        norb = self.norb
        nv = va.shape[0]
        vhf = np.zeros((norb*2, norb*2))
        # ZHC NOTE norb:norb+nv is for save_mem
        vhf[:nv, :nv] = va
        vhf[norb:norb+nv, norb:norb+nv] = -vb
        vhf[:nv, norb:norb+nv] = vd
        vhf[norb:norb+nv, :nv] = vd.T
        return vhf

    def energy_elec(self, dm, h1e, vhf):
        """
        Electronic part of Hartree-Fock energy,
        for given core hamiltonian and HF potential.
        no chemical potential contribution.
        """
        ovlp = self.get_ovlp()
        rhoA, rhoB, kappaBA = extractRdm(dm)
        HA, HB, DT = extractH1(h1e)
        nao = HA.shape[-1]
        HA += ovlp[:nao, :nao] * self.Mu
        HB += ovlp[nao:, nao:] * self.Mu
        VA, VB, VDT = extractH1(vhf)
        e1 = np.sum(rhoA*HA + rhoB*HB + 2.0 * DT*kappaBA)
        e_coul = 0.5 * np.sum(rhoA*VA + rhoB*VB + 2.0 * VDT*kappaBA)
        return e1 + e_coul, e_coul

    def get_hcore(self, *args):
        return self.h1e

    def get_ovlp(self, *args):
        return self.ovlp

    def get_occ(self, mo_energy=None, mo_coeff=None):
        '''Label the occupancies for each orbital

        Kwargs:
            mo_energy : 1D ndarray
                Obital energies

            mo_coeff : 2D ndarray
                Obital coefficients
        '''
        if mo_energy is None: mo_energy = self.mo_energy
        e_idx = np.argsort(mo_energy, kind='mergesort')
        e_sort = mo_energy[e_idx]
        nmo = mo_energy.size
        mo_occ = np.zeros(nmo)
        nocc = self.mol.nelectron // 2
        mo_occ[e_idx[:nocc]] = 1 # singly occupied

        pyscflogger.info(self, 'HOMO = %.12g  LUMO = %.12g',
                    e_sort[nocc-1], e_sort[nocc])
        if e_sort[nocc-1]+1e-3 > e_sort[nocc]:
            pyscflogger.warn(self, '!! HOMO %.12g == LUMO %.12g',
                        e_sort[nocc-1], e_sort[nocc])
        if self.verbose >= pyscflogger.DEBUG:
            np.set_printoptions(threshold=nmo)
            pyscflogger.debug(self, '  mo_energy = %s', mo_energy)
            np.set_printoptions(threshold=1000)
        return mo_occ

    def canonicalize(self, mo_coeff, mo_occ, fock=None):
        if fock is None:
            dm = self.make_rdm1(mo_coeff, mo_occ)
            fock = self.get_hcore() + self.get_veff(mol, dm)
        coreidx = mo_occ == 1 # singly occupied
        viridx = mo_occ == 0
        openidx = ~(coreidx | viridx)
        mo = np.empty_like(mo_coeff)
        mo_e = np.empty(mo_occ.size)
        for idx in (coreidx, openidx, viridx):
            if np.count_nonzero(idx) > 0:
                orb = mo_coeff[:,idx]
                f1 = reduce(np.dot, (orb.conj().T, fock, orb))
                e, c = la.eigh(f1)
                mo[:,idx] = np.dot(orb, c)
                mo_e[idx] = e
        return mo_e, mo

# *********************************************************************
# Newton Raphson method for UHFB
# *********************************************************************

def gen_g_hop_uhfb(mf, mo_coeff, mo_occ, fock_ao=None, h1e=None,
                  with_symmetry=True):
    """
    UHFB gen_g_hop. Hacked from GHF routine.
    """
    mol = mf.mol
    occidx = np.where(mo_occ==1)[0]
    viridx = np.where(mo_occ==0)[0]
    nocc = len(occidx)
    nvir = len(viridx)
    orbo = mo_coeff[:,occidx]
    orbv = mo_coeff[:,viridx]

    if fock_ao is None:
        dm0 = mf.make_rdm1(mo_coeff, mo_occ)
        fock_ao = mf.get_fock(h1e, dm=dm0)
    fock = reduce(np.dot, (mo_coeff.conj().T, fock_ao, mo_coeff))

    g = fock[viridx[:, None], occidx]

    foo = fock[occidx[:, None], occidx]
    fvv = fock[viridx[:, None], viridx]

    h_diag = fvv.diagonal().real[:, None] - foo.diagonal().real

    def h_op(x):
        x = x.reshape(nvir,nocc)
        x2  = np.einsum('ps,sq -> pq', fvv, x)
        x2 -= np.einsum('ps,rp -> rs', foo, x)

        d1 = reduce(np.dot, (orbv, x, orbo.conj().T))
        dm1 = d1 + d1.conj().T
        # ZHC NOTE
        nmo = mo_occ.shape[0] // 2
        dm1[nmo:, nmo:] += np.eye(nmo)
        v1 = mf.get_veff(mol, dm1)
        x2 += reduce(np.dot, (orbv.conj().T, v1, orbo))
        return x2.ravel()

    return g.reshape(-1), h_op, h_diag.reshape(-1)

def newton(mf):
    '''
    Co-iterative augmented hessian (CIAH) second order SCF solver
    This hack is for UHFB.
    '''
    if isinstance(mf, newton_ah._CIAH_SOSCF):
        return mf
    assert isinstance(mf, hf.SCF)

    class SecondOrderUHFB(mf.__class__, newton_ah._CIAH_SOSCF):
        __init__ = newton_ah._CIAH_SOSCF.__init__
        #dump_flags = newton_ah._CIAH_SOSCF.dump_flags
        build = newton_ah._CIAH_SOSCF.build
        kernel = newton_ah._CIAH_SOSCF.kernel
        gen_g_hop = gen_g_hop_uhfb

        def get_mo_energy(self, fock, s1e, dm):
            return self.eig(fock, s1e)

    return SecondOrderUHFB(mf)

# *********************************************************************
# Generalized (integral) Hartree-Fock (GIHF, GGHF)
# *********************************************************************

def _get_veff_ghf(dm, eri):
    """
    Get HF effective potential from rdm and ERI.
    For GGHF.
    """
    dm = np.asarray(dm, dtype=np.double)
    vj, vk = _get_jk(dm, eri)
    JK = vj[0] - vk[0]
    return JK

def energy_elec_ghf(mf, dm=None, h1e=None, vhf=None):
    '''
    Electronic part of Hartree-Fock energy,
    for given core hamiltonian and HF potential.
    Not include the contribution of Mu.
    '''
    if dm is None: dm = mf.make_rdm1()
    if h1e is None: h1e = mf.get_hcore()
    if vhf is None: vhf = mf.get_veff(mf.mol, dm)
    # remove contribution from Mu
    ovlp = mf.get_ovlp()
    nso = ovlp.shape[-1]
    nao = nso // 2
    h1e_no_mu = np.array(h1e, copy=True)
    if mf.Mu is not None:
        h1e_no_mu[:nao, :nao] += ovlp[:nao, :nao] * mf.Mu
        h1e_no_mu[nao:, nao:] -= ovlp[nao:, nao:] * mf.Mu

    e1 = np.einsum('ij, ji ->', h1e_no_mu, dm)
    e_coul = np.einsum('ij, ji ->', vhf, dm) * .5
    log.debug(1, 'E1 = %s  E_coul = %s', e1, e_coul)
    return (e1+e_coul).real, e_coul

class GGHF(scf.ghf.GHF):
    """
    a routine for generalized HF
    with generalized integrals.
    """
    def __init__(self, mol, DiisDim=12, MaxIter=50, alpha=None):
        scf.ghf.GHF.__init__(self, mol)
        self._keys = self._keys.union(["h1e", "ovlp", "norb", "Mu", "alpha"])
        self.direct_scf = False
        self.diis_space = DiisDim
        self.max_cycle = MaxIter
        self.h1e = None
        self.ovlp = None
        self.Mu = None
        self.alpha = alpha

    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
               omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        dm = np.asarray(dm)
        log.eassert(self._eri is not None, "mf._eri should be intialized.")
        vj, vk = _get_jk(dm, self._eri)
        if self.alpha is not None:
            vk *= self.alpha
        return vj[0], vk[0]

    def _finalize(self):
        try:
            ss, s = self.spin_square()
        except AssertionError:
            # ZHC NOTE FIXME ovlp may not be exactly the same for alpha and beta
            ss = -1.0
            s = -1.0

        if self.converged:
            pyscflogger.note(self, 'converged SCF energy = %.15g  '
                             '<S^2> = %.8g  2S+1 = %.8g', self.e_tot, ss, s)
        else:
            pyscflogger.note(self, 'SCF not converged.')
            pyscflogger.note(self, 'SCF energy = %.15g after %d cycles  '
                             '<S^2> = %.8g  2S+1 = %.8g',
                             self.e_tot, self.max_cycle, ss, s)
        return self

    energy_elec = energy_elec_ghf

def _get_jk_ghf(dm, eri):
    """
    Get J and K potential from rdm and ERI for GIHF.
    pyscf dot_eri_dm convention
    J: ijkl, kl -> ij
    K: ijkl, il -> jk
    Args:
        dm: (nso, nso)
        eri: (3, nao, nao, nao, nao) or s4, assume aa, bb, ab order.
    Return:
        vj (nso, nso)
        vk (nso, nso)
    """
    # split dm to aa, bb, ab
    dm = np.asarray(dm)
    log.eassert(dm.ndim == 2, "density matrix should be (nso, nso)")
    nso = dm.shape[-1]
    nao = nso // 2
    dm_aa = dm[:nao, :nao]
    dm_ab = dm[:nao, nao:]
    dm_bb = dm[nao:, nao:]
    # split eri to aa, bb, ab
    eri = np.asarray(eri)
    log.eassert(eri.shape[0] == 3, "GIHF routine needs aa, bb, ab.")

    vj = np.zeros((nso, nso))
    vk = np.zeros((nso, nso))

    # vj00, vk00
    eri_aa = ao2mo.restore(4, eri[0], nao)
    vj[:nao, :nao], vk[:nao, :nao] = scf.hf.dot_eri_dm(eri_aa, dm_aa, hermi=1)
    eri_aa = None

    # vj11, vk11
    eri_bb = ao2mo.restore(4, eri[1], nao)
    vj[nao:, nao:], vk[nao:, nao:] = scf.hf.dot_eri_dm(eri_bb, dm_bb, hermi=1)
    eri_bb = None

    # vj01, vj10, vk22
    # ZHC NOTE this hermi
    eri_ab = ao2mo.restore(4, eri[2], nao)
    vj01, _ = scf.hf.dot_eri_dm(eri_ab, dm_bb, hermi=1, with_j=True,
                                with_k=False)
    vj10, _ = scf.hf.dot_eri_dm(eri_ab.T, dm_aa, hermi=1, with_j=True,
                                with_k=False)
    _, vk22 = scf.hf.dot_eri_dm(eri_ab, dm_ab, hermi=0, with_j=False,
                                with_k=True)
    vj[:nao, :nao] += vj01
    vj[nao:, nao:] += vj10
    vk[:nao, nao:] = vk22
    vk[nao:, :nao] = vk22.conj().T
    return vj, vk

class GIHF(GGHF):
    """
    a routine for generalized HF
    with integrals different for two spin species.
    """
    def get_jk(self, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
               omega=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        dm = np.asarray(dm)
        log.eassert(self._eri is not None, "mf._eri should be intialized.")
        vj, vk = _get_jk_ghf(dm, self._eri)
        return vj, vk

# *********************************************************************
# Main class for HF (RHF, UHF, UIHF), HFB (UHFB), GHF and MP2 (UMP2)
# *********************************************************************

class SCF(object):
    def __init__(self, tmp="/tmp", newton_ah=True, no_kernel=False,
                 verbose=None, chkname=None):
        self.mf = None
        self.sys_initialized = False
        self.integral_initialized = False
        self.doneHF = False
        self.newton_ah = newton_ah
        self.no_kernel = no_kernel
        self.verbose = verbose
        self.chkname = chkname
        log.debug(0, "Using pyscf version %s", pyscf.__version__)
        if self.newton_ah:
            if log.Level[log.verbose] <= log.Level["RESULT"]:
                pyscflogger.flush = flush([""])
                pyscflogger.flush.addkey("macro X")
            elif log.Level[log.verbose] <= log.Level["INFO"]:
                pyscflogger.flush = flush([""])
                pyscflogger.flush.addkey("macro")
            else:
                pyscflogger.flush = flush([""])
        else:
            if log.Level[log.verbose] <= log.Level["INFO"]:
                pyscflogger.flush = flush([""])
                pyscflogger.flush.addkey("cycle=")
            else:
                pyscflogger.flush = flush([""])

    def set_system(self, nelec, spin, bogoliubov, spinRestricted,
                   max_memory=120000):
        if bogoliubov:
            log.eassert(nelec is None,
                        "nelec cannot be specified when doing BCS calculations")
        self.nelec = nelec
        self.spin = spin
        self.bogoliubov = bogoliubov
        self.spinRestricted = spinRestricted
        self.mol = gto.Mole()
        # ZHC TODO for system with huge ERI, implement outcore routine.
        self.mol.incore_anyway = True
        self.mol.max_memory =  max_memory
        if self.verbose is None:
            if log.Level[log.verbose] >= log.Level["RESULT"]:
                verbose = 4
            else:
                verbose = 2
        else:
            verbose = self.verbose
        self.mol.build(verbose=verbose, dump_input=False)

        self.mol.nelectron = self.nelec
        self.mol.tot_electrons = lambda *args: self.nelec # may use in smearing
        self.mol.spin = self.spin
        self.sys_initialized = True

    def set_integral(self, *args):
        log.eassert(self.sys_initialized, "set_integral() should be used "
                    "after initializing set_system()")
        if len(args) == 1:
            log.eassert(self.bogoliubov == args[0].bogoliubov,
                        "Integral is not consistent with system type")
            self.integral = args[0]
        elif len(args) == 4:
            self.integral = integral.Integral(args[0], self.spinRestricted,
                                              self.bogoliubov, *args[1:])
        else:
            log.error("input either an integral object, or (norb, H0, H1, H2)")
        self.integral_initialized = True
        if self.bogoliubov:
            self.mol.nelectron = self.integral.norb * 2

    def HF(self, DiisDim=12, MaxIter=50, InitGuess=None, tol=1e-10, Mu=None,
           do_diis=True, alpha=None, beta=np.inf, fit_spin=True):
        """
        R(O)HF, U(I)HF routines.
        If Mu is not None and using UHF, Mu will add to h1e[0]
        to mimic the GHF behavior.
        """
        log.eassert(self.sys_initialized and self.integral_initialized,
                    "components for Hartree-Fock (Bogoliubov) calculation"
                    " are not ready\nsys_init = %s\nint_init = %s",
                    self.sys_initialized, self.integral_initialized)
        if self.bogoliubov:
            return self.HFB(0., DiisDim, MaxIter, InitGuess, tol)

        if not self.spinRestricted: # UHF
            log.result("Unrestricted Hartree-Fock with pyscf")
            self.mf = UIHF(self.mol, DiisDim=DiisDim, MaxIter=MaxIter,
                           alpha=alpha)
            self.mf.energy_nuc = lambda *args: self.integral.H0
            # ZHC NOTE copy to avoid influence of adding Mu.
            ovlp = self.integral.ovlp
            self.mf.h1e = np.array(self.integral.H1["cd"], copy=True)
            if len(self.mf.h1e) == 1:
                self.mf.h1e = np.asarray((self.mf.h1e[0], self.mf.h1e[0]))
            if Mu is not None:
                nao = self.integral.norb // 2
                self.mf.h1e[0, :nao, :nao] -= ovlp[:nao, :nao] * Mu
                self.mf.h1e[0, nao:, nao:] += ovlp[nao:, nao:] * Mu
                self.mf.Mu = Mu
            self.mf.ovlp = ovlp
            self.mf._eri = self.integral.H2["ccdd"] #vaa, vbb, vab
            self.mf.conv_tol = tol
            if self.chkname is not None:
                self.mf.chkfile = self.chkname

            if not do_diis:
                self.mf.diis = None

            if self.newton_ah:
                self.mf = self.mf.newton()

            if beta < np.inf:
                from libdmet.routine import pbc_helper as pbc_hp
                self.mf = pbc_hp.smearing_(self.mf, sigma=1.0/beta,
                                           method='fermi', fit_spin=fit_spin)

            if self.no_kernel:
                E = rho = 0.0
            else:
                E = self.mf.kernel(dm0=InitGuess)
                rho = np.asarray(self.mf.make_rdm1())
        else: # RHF
            log.result("Restricted Hartree-Fock with pyscf")
            self.mf = RIHF(self.mol, alpha=alpha)
            self.mf.energy_nuc = lambda *args: self.integral.H0
            self.mf.get_hcore = lambda *args: self.integral.H1["cd"][0]
            self.mf.get_ovlp = lambda *args: self.integral.ovlp
            self.mf._eri = ao2mo.restore(4, self.integral.H2["ccdd"][0],
                                         self.integral.norb)
            self.mf.direct_scf = False
            self.mf.diis_space = DiisDim
            self.mf.max_cycle = MaxIter
            self.mf.conv_tol = tol
            if self.chkname is not None:
                self.mf.chkfile = self.chkname

            if not do_diis:
                self.mf.diis = None

            if self.newton_ah: # RHF w/ newton
                self.mf = self.mf.newton()

            if beta < np.inf:
                from libdmet.routine import pbc_helper as pbc_hp
                self.mf = pbc_hp.smearing_(self.mf, sigma=1.0/beta,
                                           method='fermi', fit_spin=fit_spin)

            if InitGuess is not None and InitGuess.ndim == 3:
                InitGuess = InitGuess[0]
            if self.no_kernel:
                E = rho = 0.0
            else:
                E = self.mf.kernel(dm0=InitGuess)
                # ZHC NOTE the 0.5 factor and additional dim
                rho = np.asarray(self.mf.make_rdm1())[None] * 0.5

        log.result("Hartree-Fock convergence: %s", self.mf.converged)
        log.result("Hartree-Fock energy = %20.12f", E)
        self.doneHF = True
        return E, rho

    def HFB(self, Mu, DiisDim=12, MaxIter=50, InitGuess=None, tol=1e-10,
            do_diis=True):
        log.eassert(self.sys_initialized and self.integral_initialized,
                    "components for Hartree-Fock Bogoliubov calculation"
                    " are not ready\nsys_init = %s\nint_init = %s",
                    self.sys_initialized, self.integral_initialized)

        norb = self.integral.norb
        if not self.spinRestricted:
            log.result("Unrestricted Hartree-Fock-Bogoliubov with pyscf")
            self.mf = UHFB(self.mol, DiisDim=DiisDim, MaxIter=MaxIter)
            self.mf.energy_nuc = lambda *args: self.integral.H0
            ovlp = self.integral.ovlp
            h1e = np.empty((norb*2, norb*2))
            self.mf.Mu = Mu
            self.mf.norb = norb
            h1e[:norb, :norb] = self.integral.H1["cd"][0] - ovlp * Mu
            h1e[norb:, norb:] = -(self.integral.H1["cd"][1] - ovlp * Mu)
            h1e[:norb, norb:] = self.integral.H1["cc"][0]
            h1e[norb:, :norb] = self.integral.H1["cc"][0].T
            self.mf.h1e = h1e
            s1e = np.zeros_like(h1e)
            s1e[:norb, :norb] = ovlp
            s1e[norb:, norb:] = ovlp
            self.mf.ovlp = s1e
            self.mf._eri = self.integral.H2 # we can have cccd and cccc terms
            self.mf.conv_tol = tol
            if self.chkname is not None:
                self.mf.chkfile = self.chkname
            if not do_diis:
                self.mf.diis = None
            if self.newton_ah:
                self.mf = newton(self.mf)
            if InitGuess is not None:
                log.eassert(InitGuess.ndim == 2,
                            "HFB InitGuess should have shape (nso, nso)")
            else:
                # ZHC FIXME the HFB solution can be very different.
                # is this InitGuess reasonable?
                InitGuess = np.eye(norb * 2) * 0.5
            if self.no_kernel:
                E = GRho = 0.0
            else:
                E = self.mf.kernel(dm0=InitGuess)
                GRho = np.asarray(self.mf.make_rdm1())
        else:
            log.error("Restricted Hartree-Fock-Bogoliubov not implemented yet")

        log.result("Hartree-Fock-Bogoliubov convergence: %s", self.mf.converged)
        log.result("Hartree-Fock-Bogoliubov energy = %20.12f", E)
        self.doneHF = True
        return E, GRho

    def GHF(self, Mu, DiisDim=12, MaxIter=50, InitGuess=None, tol=1e-10,
            do_diis=True, alpha=None, beta=np.inf):
        log.eassert(self.sys_initialized and self.integral_initialized,
                    "components for GHF calculation"
                    " are not ready\nsys_init = %s\nint_init = %s",
                    self.sys_initialized, self.integral_initialized)
        norb = self.integral.norb
        if self.mol.nelectron is None:
            self.mol.nelectron = norb
        nelec = self.mol.nelectron

        log.result("Generalized Hartree-Fock with pyscf")
        self.mf = GIHF(self.mol)
        self.mf.diis_space = DiisDim
        self.mf.max_cycle = MaxIter
        self.mf.conv_tol = tol
        if self.chkname is not None:
            self.mf.chkfile = self.chkname

        self.mf.norb = norb
        self.mf.Mu = Mu
        self.mf.energy_nuc = lambda *args: self.integral.H0
        ovlp = self.integral.ovlp
        h1e = np.empty((norb*2, norb*2))
        h1e[:norb, :norb] = self.integral.H1["cd"][0] - ovlp * Mu
        h1e[norb:, norb:] = self.integral.H1["cd"][1] + ovlp * Mu
        h1e[:norb, norb:] = self.integral.H1["cd"][2]
        h1e[norb:, :norb] = self.integral.H1["cd"][2].conj().T
        self.mf.get_hcore = lambda *args: h1e

        s1e = np.zeros_like(h1e)
        s1e[:norb, :norb] = ovlp
        s1e[norb:, norb:] = ovlp
        self.mf.get_ovlp = lambda *args: s1e

        self.mf._eri = self.integral.H2["ccdd"] # should has shape (3, ...)

        if not do_diis:
            self.mf.diis = None

        if self.newton_ah:
            self.mf = self.mf.newton()

        if beta < np.inf:
            from libdmet.routine import pbc_helper as pbc_hp
            self.mf = pbc_hp.smearing_(self.mf, sigma=1.0/beta,
                                       method='fermi', fit_spin=True)

        if InitGuess is not None:
            log.eassert(InitGuess.ndim == 2,
                        "GHF InitGuess should have shape (nso, nso)")
        else:
            # ZHC FIXME the GHF solution can be very different.
            # is this InitGuess reasonable?
            InitGuess = np.eye(norb * 2) * (nelec / (norb*2.0))
        if self.no_kernel:
            E = GRho = 0.0
        else:
            E = self.mf.kernel(dm0=InitGuess)
            GRho = np.asarray(self.mf.make_rdm1())

        log.result("GHF convergence: %s", self.mf.converged)
        log.result("GHF energy = %20.12f", E)
        self.doneHF = True
        return E, GRho

    def GGHF(self, Mu=None, DiisDim=12, MaxIter=50, InitGuess=None, tol=1e-10,
             do_diis=True, alpha=None, beta=np.inf, fit_mu=False,
             nelec_target=None, basis=None, mu_elec=None, conv_tol_grad=None,
             tol_nelec=None, fix_mu=False):
        log.eassert(self.sys_initialized and self.integral_initialized,
                    "components for GHF calculation are not ready"
                    "\nsys_init = %s\nint_init = %s",
                    self.sys_initialized, self.integral_initialized)
        norb = self.integral.norb
        if self.mol.nelectron is None:
            self.mol.nelectron = norb // 2
        nelec = self.mol.nelectron

        log.result("Generalized Hartree-Fock with pyscf")
        if fit_mu:
            log.info("fit chemical potential")
            from libdmet.solver import scf_mu
            if basis is None:
                log.warn("fit chemical potential typically need basis to pass in...")
            self.mf = scf_mu.GGHFpMu(self.mol, alpha=alpha,
                                     nelec_target=nelec_target,
                                     basis=basis, mu_elec=mu_elec,
                                     tol_nelec=tol_nelec)
        else:
            self.mf = GGHF(self.mol, alpha=alpha)

        self.mf.diis_space = DiisDim
        self.mf.max_cycle = MaxIter
        self.mf.conv_tol = tol
        self.mf.conv_tol_grad = conv_tol_grad
        if self.chkname is not None:
            self.mf.chkfile = self.chkname

        h1e = np.array(self.integral.H1["cd"][0], copy=True)
        ovlp = self.integral.ovlp
        nao = norb // 2
        if Mu is not None:
            assert not fit_mu
            log.debug(0, "GGHF: mu = %.5g", Mu)
            h1e[:nao, :nao] -= ovlp[:nao, :nao] * Mu
            h1e[nao:, nao:] += ovlp[nao:, nao:] * Mu

        self.mf.norb = norb
        self.mf.Mu = Mu
        self.mf.energy_nuc = lambda *args: self.integral.H0
        self.mf.get_hcore = lambda *args: h1e
        self.mf.get_ovlp = lambda *args: ovlp
        self.mf._eri = self.integral.H2["ccdd"][0]

        if not do_diis:
            self.mf.diis = None

        if self.newton_ah:
            self.mf = self.mf.newton()

        if beta < np.inf:
            from libdmet.routine import pbc_helper as pbc_hp
            log.debug(0, "smearing: beta: %s , fix_mu: %s", beta, fix_mu)
            self.mf = pbc_hp.smearing_(self.mf, sigma=1.0/beta,
                                       method='fermi', fit_spin=False,
                                       mu0=0.0, fix_mu=fix_mu)

        if InitGuess is not None:
            log.eassert(InitGuess.ndim == 2,
                        "GHF InitGuess should have shape (nso, nso)")
        else:
            # ZHC FIXME the GHF solution can be very different.
            # is this InitGuess reasonable?
            InitGuess = np.eye(norb) * (nelec / float(norb))
            #InitGuess = np.zeros((norb, norb))
        if self.no_kernel:
            E = GRho = 0.0
        else:
            E = self.mf.kernel(dm0=InitGuess)
            GRho = np.asarray(self.mf.make_rdm1())

        log.result("GHF convergence: %s", self.mf.converged)
        log.result("GHF energy = %20.12f", E)
        self.doneHF = True
        return E, GRho

    def MP2(self, mo_energy=None, mo_coeff=None, mo_occ=None, frozen=None):
        if not self.doneHF:
            log.warning("running HF first with default settings")
            self.HF()
        log.check(self.mf.converged, "Hartree-Fock calculation has not converged")
        if not self.spinRestricted:
            log.result("Unrestricted MP2 with pyscf")
            self.mp = UIMP2(self.mf, frozen=frozen, mo_coeff=mo_coeff,
                            mo_occ=mo_occ)
            E, t2 = self.mp.kernel(mo_energy=mo_energy, mo_coeff=mo_coeff)
            rdm1 = self.mp.make_rdm1(ao_repr=True)
        else:
            log.result("Restricted MP2 with pyscf")
            self.mp = MP2(self.mf, frozen=frozen, mo_coeff=mo_coeff,
                          mo_occ=mo_occ)
            E, t2 = self.mp.kernel(mo_energy=mo_energy, mo_coeff=mo_coeff)
            rdm1 = self.mp.make_rdm1(ao_repr=True)
            # ZHC NOTE add spin dim and scaled by 0.5
            rdm1 = rdm1[None] * 0.5
        return E, rdm1

    def GMP2(self, mo_energy=None, mo_coeff=None, mo_occ=None, frozen=None):
        if not self.doneHF:
            log.warning("running HF first with default settings")
            self.HF()
        log.check(self.mf.converged, "Hartree-Fock calculation has not converged")
        log.result("Generalized MP2 with pyscf")
        self.mp = GGMP2(self.mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
        # ZHC NOTE set the nocc since mf may have fractional occupations.
        if sum(self.mp.mo_occ == 0) + sum(self.mp.mo_occ == 1) != len(self.mp.mo_occ):
            nelec = self.mol.nelectron
            if self.mp.frozen is None:
                nocc = nelec
            elif isinstance(self.mp.frozen, (int, np.integer)):
                nocc = nelec - self.mp.frozen
            elif isinstance(self.mp.frozen[0], (int, np.integer)):
                occ_idx = np.zeros(self.mp.mo_occ.shape, dtype=bool)
                occ_idx[:nelec] = True
                occ_idx[list(self.mp.frozen)] = False
                nocc = np.count_nonzero(occ_idx)
            else:
                raise ValueError
            self.mp.nocc = nocc
        E, t2 = self.mp.kernel(mo_energy=mo_energy, mo_coeff=mo_coeff)
        rdm1 = self.mp.make_rdm1(ao_repr=True)
        return E, rdm1

    def get_mo(self):
        log.eassert(self.doneHF, "Hartree-Fock calculation is not done")
        if self.mf.mo_coeff.ndim == 2 and (not self.bogoliubov):
            return np.asarray([self.mf.mo_coeff])
        else:
            return np.asarray(self.mf.mo_coeff)

    def get_mo_energy(self):
        log.eassert(self.doneHF, "Hartree-Fock calculation is not done")
        if self.mf.mo_coeff.ndim == 2 and (not self.bogoliubov):
            return np.asarray([self.mf.mo_energy])
        else:
            return np.asarray(self.mf.mo_energy)

if __name__ == "__main__":
    np.set_printoptions(3, linewidth=1000)
    log.verbose = "DEBUG2"
    Int1e = -np.eye(8, k=1)
    Int1e[0, 7] = -1
    Int1e += Int1e.T
    Int1e = np.asarray([Int1e, Int1e])
    Int2e = np.zeros((3,8,8,8,8))

    for i in range(8):
        Int2e[0,i,i,i,i] = 4.0
        Int2e[1,i,i,i,i] = 4.0
        Int2e[2,i,i,i,i] = 4.0

    myscf = SCF(newton_ah=True)
    #myscf = SCF(newton_ah=False)

    # UHF
    myscf.set_system(8, 0, False, False)
    myscf.set_integral(8, 0, {"cd": Int1e}, {"ccdd": Int2e})
    _, rhoHF = myscf.HF(MaxIter=100, tol=1e-8,
                        InitGuess=(np.diag([1,0,1,0,1,0,1,0.0]),
                                   np.diag([0,1,0,1,0,1,0,1.0]))
                        )
    log.result("HF density matrix:\n%s\n%s", rhoHF[0], rhoHF[1])

    # UHFB
    myscf = SCF(newton_ah=True)
    np.random.seed(8)
    myscf.set_system(None, 0, True, False)
    myscf.set_integral(8, 0, {"cd": Int1e, "cc": np.random.rand(1,8,8) * 0.1},
                       {"ccdd": Int2e, "cccd": None, "cccc": None})
    _, GRhoHFB = myscf.HF(MaxIter=100, tol=1e-3, Mu=2.02,
                          InitGuess=np.diag([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]))
    rhoA, rhoB, kappaBA = extractRdm(GRhoHFB)
    log.result("HFB density matrix:\n%s\n%s\n%s", rhoA, rhoB, -kappaBA.T)
