#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generalized Tailored CCSD.

Authors:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

import os
import numpy as np

from pyscf import lib
from pyscf import ao2mo
from pyscf import scf
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import gintermediates as imd
from pyscf import __config__

from pyscf.fci import direct_spin1
from pyscf.ci.cisd import tn_addrs_signs
from pyscf.cc import gccsd
from libdmet.utils.misc import mdot, max_abs, take_eri
from libdmet.solver.block import Block

MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

einsum = lib.einsum

def get_cisd_vec_cas(mycc, vals, dets):
    ncas      = mycc.ncas
    nocc_cas  = mycc.nocc_cas
    nvir_cas  = mycc.nvir_cas
    nvir_cas2 = nvir_cas*(nvir_cas-1)//2
    nocc_cas2 = nocc_cas*(nocc_cas-1)//2

    solver = mycc.cisolver.cisolver
    if Block.reorder:
        reorderFile = os.path.join(solver.tmpDir, "orbital_reorder.npy")
        reorder = np.load(reorderFile)
        assert reorder.shape[0] == ncas

    c0   = np.zeros(1)
    c1a  = np.zeros((nocc_cas*nvir_cas))
    c2aa = np.zeros((nocc_cas2*nvir_cas2))

    def convert_phase(h_a, val):
        n_perm = 0
        ranka = len(h_a)
        if ranka > 0:
            n_perm += ranka*nocc_cas - sum(h_a) - ranka*(ranka+1)//2
        return val * ( 1 - ( (n_perm & 1) << 1 ) )

    for val, det in zip(vals, dets):
        # hole, ptcl
        h_a, p_a = [], []
        if not Block.reorder:
            for i in range(nocc_cas):
                if det[i] == 0: h_a.append(i)
            for a in range(nocc_cas,ncas,1):
                if det[a] == 1: p_a.append(a-nocc_cas)
        else:
            for i in range(ncas):
                org_i = reorder[i]
                if det[i] == 0 and org_i < nocc_cas:  h_a.append(org_i)
                if det[i] == 1 and org_i >= nocc_cas: p_a.append(org_i-nocc_cas)
            h_a.sort()
            p_a.sort()

        assert len(h_a) == len(p_a)
        if   len(h_a) == 0:
            c0[0] = val
        elif len(h_a) == 1:
            idx_a = nvir_cas * h_a[0] + p_a[0]
            c1a[idx_a] = convert_phase(h_a, val)
        elif len(h_a) == 2:
            h2 = h_a[1]*(h_a[1]-1)//2 + h_a[0]
            p2 = p_a[1]*(p_a[1]-1)//2 + p_a[0]
            idx_a = nvir_cas2 * h2 + p2
            c2aa[idx_a] = convert_phase(h_a, val)
        else:
            continue

    return c0, c1a, c2aa

def get_cas_amps(mycc, eris):
    """
    Get cas space amplitudes.

    Args:
        mycc: cc object.
        eris: eris.

    Returns:
        t1_cas: cas space t1, shape (nocc_cas, nvir_cas)
        t2_cas: cas space t2, shape (nocc_cas, nocc_cas, nvir_cas, nvir_cas)
    """
    ncas = mycc.ncas
    nocc = mycc.nocc
    nocc_cas = mycc.nocc_cas
    nvir_cas = mycc.nvir_cas

    # CI solver
    if mycc.cisolver is None:
        cisolver = direct_spin1.FCI()
        cisolver.verbose = mycc.verbose
        cisolver.max_memory = mycc.max_memory
        cisolver.max_cycle = mycc.max_cycle
        cisolver.conv_tol = mycc.conv_tol * 0.1
    else:
        cisolver = mycc.cisolver

    logger.info(mycc, 'TCCSD CI start.')

    h0 = eris.h0_cas
    h1 = eris.h1_cas
    h2 = eris.h2_cas

    from libdmet.solver.impurity_solver import Block2
    from libdmet.system import integral
    if isinstance(cisolver, Block2):
        Ham = integral.Integral(ncas, False, False, h0, {"cd": h1[None]},
                                {"ccdd": h2[None]})
        spin = 0 if cisolver.cisolver.use_general_spin else nocc_cas
        _, e_fci = cisolver.run(Ham, spin=spin, nelec=nocc_cas)
    else:
        e_fci, fcivec = cisolver.kernel(h1, h2, ncas, (nocc_cas, 0),
                                        ecore=h0, **mycc.ci_args)
    logger.info(mycc, 'TCCSD CI energy: %25.15f', e_fci)

    #C = mycc.mo_cas
    #rdm1 = mdot(C, cisolver.make_rdm1(fcivec, ncas, (nocc_cas, 0)), C.conj().T) + eris.rdm1_core
    #np.save("rdm1_eo.npy", rdm1)

    # FCI/DMRG-MPS -> CISD -> CCSD
    if isinstance(cisolver, Block2):
        # MPS to CI
        ref_str = "1"*nocc_cas + "0"*nvir_cas
        cisolver.cisolver.mps2ci_run(ref_str, tol=1e-9)
        tmpDir = cisolver.cisolver.tmpDir
        civecFile = os.path.join(tmpDir, "sample-vals.npy")
        civec = np.load(civecFile)
        cidetFile = os.path.join(tmpDir, "sample-dets.npy")
        cidet = np.load(cidetFile)
        idx = np.argsort(np.abs(civec))[::-1]
        max_id  = idx[0]
        max_str = cidet[max_id]
        max_vec = civec[max_id]
    else:
        from pyscf.fci import cistring
        max_id = np.unravel_index(np.argmax(np.abs(fcivec)), fcivec.shape)
        max_str = bin(cistring.addr2str(ncas, nocc_cas, max_id[0]))
        max_vec = fcivec[max_id]

    logger.info(mycc, "max fcivec det id: %s", max_id)
    logger.info(mycc, "string: %s", max_str)
    logger.info(mycc, "weight: %s", max_vec)

    if isinstance(cisolver, Block2):
        c0, cis_a, cid_aa = get_cisd_vec_cas(mycc, civec, cidet)
    else:
        t1addrs, t1signs = tn_addrs_signs(ncas, nocc_cas, 1)
        t2addrs, t2signs = tn_addrs_signs(ncas, nocc_cas, 2)

        c0 = fcivec[0, 0]
        logger.info(mycc, 'TCCSD CI reference weight c0: %25.15f', c0)
        cis_a = fcivec[t1addrs, 0] * t1signs
        #logger.info(mycc, "fcivec[t1addrs, 0]\n%s", cis_a)

        cid_aa = fcivec[t2addrs, 0] * t2signs

        # ZHC NOTE
        #idx1 = np.argsort(t1addrs, kind='mergesort')
        #idx2 = np.argsort(t2addrs, kind='mergesort')
        #cis_a = cis_a[idx1]
        #cid_aa = cid_aa[idx2]

    cis_a /= c0
    cid_aa /= c0

    t1_cas = cis_a.reshape(nocc_cas, nvir_cas)
    t2_cas  = ccsd._unpack_4fold(cid_aa, nocc_cas, nvir_cas)
    tmp = np.einsum('ia, jb -> ijab', t1_cas, t1_cas)
    tmp = tmp - tmp.transpose(0, 1, 3, 2)
    t2_cas -= tmp
    return t1_cas, t2_cas

def update_amps(mycc, t1, t2, eris):
    assert isinstance(eris, gccsd._PhysicistsERIs)
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc, nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift

    tau = imd.make_tau(t2, t1, t1)

    Fvv = imd.cc_Fvv(t1, t2, eris)
    Foo = imd.cc_Foo(t1, t2, eris)
    Fov = imd.cc_Fov(t1, t2, eris)

    # Move energy terms to the other side
    Fvv[np.diag_indices(nvir)] -= mo_e_v
    Foo[np.diag_indices(nocc)] -= mo_e_o

    # T1 equation
    t1new  =  einsum('ie,ae->ia', t1, Fvv)
    t1new -=  einsum('ma,mi->ia', t1, Foo)
    t1new +=  einsum('imae,me->ia', t2, Fov)
    t1new -=  einsum('nf,naif->ia', t1, eris.ovov)
    t1new += (-0.5) * einsum('imef,maef->ia', t2, eris.ovvv)
    t1new += (-0.5) * einsum('mnae,mnie->ia', t2, eris.ooov)
    t1new += fov.conj()

    # T2 equation
    Ftmp = Fvv - 0.5 * einsum('mb, me -> be', t1, Fov)
    tmp = einsum('ijae, be -> ijab', t2, Ftmp)
    t2new = tmp - tmp.transpose(0,1,3,2)
    Ftmp = Foo + 0.5 * einsum('je,me->mj', t1, Fov)
    tmp = einsum('imab, mj -> ijab', t2, Ftmp)
    t2new -= tmp - tmp.transpose(1,0,2,3)
    t2new += np.asarray(eris.oovv).conj()

    Woooo = imd.cc_Woooo(t1, t2, eris)
    t2new += einsum('mnab, mnij -> ijab', tau, Woooo * 0.5)
    Woooo = None

    Wvvvv = imd.cc_Wvvvv(t1, t2, eris)
    t2new += 0.5 * einsum('ijef,abef->ijab', tau, Wvvvv)
    Wvvvv = None
    tau = None

    Wovvo = imd.cc_Wovvo(t1, t2, eris)
    tmp = einsum('imae,mbej->ijab', t2, Wovvo)
    Wovvo = None

    tmp += einsum('ie, ma, mbje -> ijab', t1, t1, eris.ovov)
    tmp = tmp - tmp.transpose(1,0,2,3)
    tmp = tmp - tmp.transpose(0,1,3,2)
    t2new += tmp
    tmp = einsum('ie, jeba -> ijab', t1, np.asarray(eris.ovvv).conj())
    t2new += (tmp - tmp.transpose(1,0,2,3))
    tmp = einsum('ma, ijmb -> ijab', t1, np.asarray(eris.ooov).conj())
    t2new -= (tmp - tmp.transpose(0,1,3,2))

    eia = mo_e_o[:,None] - mo_e_v
    t1new /= eia
    for i in range(nocc):
        t2new[i] /= lib.direct_sum('a, jb -> jab', eia[i], eia)

    # TCC
    ncore = mycc.ncore
    nvir_cas = mycc.nvir_cas
    if mycc.t1_cas is None or mycc.t2_cas is None:
        mycc.t1_cas, mycc.t2_cas = mycc.get_cas_amps(eris=eris)
    t1_cas = mycc.t1_cas
    t2_cas = mycc.t2_cas
    t1new[ncore:, :nvir_cas] = t1_cas
    t2new[ncore:, ncore:, :nvir_cas, :nvir_cas] = t2_cas
    return t1new, t2new

class GGTCCSD(gccsd.GCCSD):

    conv_tol = getattr(__config__, 'cc_gccsd_GCCSD_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'cc_gccsd_GCCSD_conv_tol_normt', 1e-6)

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None,
                 ncas=0, nelecas=0, nocc=None):
        """
        ncas: number of cas spin orbitals
        nelecas: number of cas electrons
        """
        assert isinstance(mf, scf.ghf.GHF)
        ccsd.CCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)

        # initialize CAS space
        nmo = self.nmo

        if nocc is None:
            nocc = self.nocc
        else:
            self.nocc = nocc

        ncore = nocc - nelecas
        nocc_cas = nelecas
        nvir_cas = ncas - nocc_cas
        nvir = nmo - nocc

        self.ncas = ncas
        self.nelecas = nelecas
        self.ncore = ncore
        self.nvir = nvir
        self.nocc_cas = nocc_cas
        self.nvir_cas = nvir_cas
        assert 0 <= self.ncas <= self.nmo
        assert 0 <= self.nelecas <= self.ncas
        assert 0 <= self.ncore <= self.nmo
        assert 0 <= self.nocc_cas <= self.ncas
        assert 0 <= self.nvir_cas <= self.ncas

        self.mo_core = self.mo_coeff[:, :ncore]
        self.mo_cas = self.mo_coeff[:, ncore:ncore+ncas]
        self.mo_vir = self.mo_coeff[:, ncore+ncas:]

        self.cisolver = None
        self.ci_args = {"ci0": None, "pspace_size": 1000}
        self.t1_cas = None
        self.t2_cas = None

        self._keys = self._keys.union(["mo_core", "mo_cas", "mo_vir",
                                       "cisolver", "ci_args",
                                       "t1_cas", "t2_cas",
                                       "ncas", "nelecas",
                                       "ncore", "nvir", "nocc_cas",
                                       "nvir_cas"])

    def dump_flags(self, verbose=None):
        gccsd.GCCSD.dump_flags(self, verbose=verbose)
        logger.info(self, 'TCCSD nocc     = %4d, nvir     = %4d, nmo  = %4d',
                    self.nocc, self.nvir, self.nmo)
        logger.info(self, 'TCCSD nocc_cas = %4d, nvir_cas = %4d, ncas = %4d',
                    self.nocc_cas, self.nvir_cas, self.ncas)
        return self

    def init_amps(self, eris=None):
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        if self.t1_cas is None or self.t2_cas is None:
            self.t1_cas, self.t2_cas = self.get_cas_amps(eris=eris)
        mo_e = eris.mo_energy
        nocc = self.nocc
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        eijab = lib.direct_sum('ia, jb -> ijab', eia, eia)
        t1 = eris.fock[:nocc,nocc:] / eia
        eris_oovv = np.array(eris.oovv)
        t2 = eris_oovv / eijab

        # TCC
        ncore = self.ncore
        nvir_cas = self.nvir_cas
        t1_cas = self.t1_cas
        t2_cas = self.t2_cas
        t1[ncore:, :nvir_cas] = t1_cas
        t2[ncore:, ncore:, :nvir_cas, :nvir_cas] = t2_cas

        self.emp2 = 0.25 * einsum('ijab, ijab', t2, eris_oovv.conj()).real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        return self.emp2, t1, t2

    update_amps = update_amps

    get_cas_amps = get_cas_amps

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        '''Ground-state unrestricted (U)CCSD.

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to CCSD.
        '''
        # initialize t1_cas and t2_cas
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        if self.t1_cas is None or self.t2_cas is None:
            self.t1_cas, self.t2_cas = self.get_cas_amps(eris=eris)

        if (t1 is not None) or (t2 is not None):
            # ZHC NOTE
            # if t1, t2 are specified (from restart)
            # still need to fix the amplitudes in the CAS space
            ncore = self.ncore
            nvir_cas = self.nvir_cas
            t1[ncore:, :nvir_cas] = self.t1_cas
            t2[ncore:, ncore:, :nvir_cas, :nvir_cas] = self.t2_cas

        if mbpt2:
            from pyscf.mp import gmp2
            pt = gmp2.GMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
            self.e_corr, self.t2 = pt.kernel(eris=eris)
            nocc, nvir = self.t2.shape[1:3]
            self.t1 = np.zeros((nocc,nvir))
            return self.e_corr, self.t1, self.t2

        # Initialize orbspin so that we can attach it to t1, t2
        if getattr(self.mo_coeff, 'orbspin', None) is None:
            orbspin = scf.ghf.guess_orbspin(self.mo_coeff)
            if not np.any(orbspin == -1):
                self.mo_coeff = lib.tag_array(self.mo_coeff, orbspin=orbspin)

        e_corr, self.t1, self.t2 = ccsd.CCSD.ccsd(self, t1, t2, eris)
        if getattr(eris, 'orbspin', None) is not None:
            self.t1 = lib.tag_array(self.t1, orbspin=eris.orbspin)
            self.t2 = lib.tag_array(self.t2, orbspin=eris.orbspin)
        return e_corr, self.t1, self.t2

    def make_rdm1(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        '''Un-relaxed 1-particle density matrix in MO space'''
        from pyscf.cc import gccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2

        if l1 is None:
            l1, l2 = self.solve_lambda(t1, t2)

        return gccsd_rdm.make_rdm1(self, t1, t2, l1, l2, ao_repr=ao_repr)

    def make_rdm2(self, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
        '''2-particle density matrix in MO space.  The density matrix is
        stored as

        dm2[p,r,q,s] = <p^+ q^+ s r>
        '''
        from pyscf.cc import gccsd_rdm
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if l1 is None: l1 = self.l1
        if l2 is None: l2 = self.l2

        if l1 is None:
            l1, l2 = self.solve_lambda(t1, t2)

        return gccsd_rdm.make_rdm2(self, t1, t2, l1, l2, ao_repr=ao_repr)

    def ao2mo(self, mo_coeff=None):
        nmo = self.nmo
        mem_incore = nmo**4*2 * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore_ghf(self, mo_coeff)
        elif getattr(self._scf, 'with_df', None):
            raise NotImplementedError
        else:
            raise NotImplementedError
            #return _make_eris_outcore(self, mo_coeff)

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None,
                     eris=None, approx_l=True):
        if t1 is None: t1 = self.t1
        if t2 is None: t2 = self.t2
        if eris is None: eris = self.ao2mo(self.mo_coeff)
        nocc = self.nocc
        nvir = self.nmo - nocc

        if approx_l:
            ncore = self.ncore
            nvir_cas = self.nvir_cas

            l1 = np.array(t1, copy=True)
            l1[ncore:, :nvir_cas] = 0.0
            self.l1 = l1

            l2 = np.array(t2, copy=True)
            l2[ncore:, ncore:, :nvir_cas, :nvir_cas] = 0.0
            self.l2 = l2
        else:
            raise NotImplementedError
        return self.l1, self.l2

def _make_eris_incore_ghf(mycc, mo_coeff=None, ao2mofn=None):
    """
    Incore GGCC ERI ao2mo.
    Memory usage is optimized:
    required additional memory ~ vvvv + 1/8 * pppp (normal case is 2 * pppp)
    """
    eris = gccsd._PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nao, nmo = eris.mo_coeff.shape

    if callable(ao2mofn):
        raise NotImplementedError
    else:
        assert eris.mo_coeff.dtype == np.double
        if (nao == nmo) and (max_abs(eris.mo_coeff - np.eye(nmo)) < 1e-13):
            # ZHC NOTE special treatment for OO-CCD,
            # where the ao2mo is not needed for identity mo_coeff.
            eri = mycc._scf._eri
        else:
            eri = ao2mo.kernel(mycc._scf._eri, eris.mo_coeff)

    o = np.arange(0, nocc)
    v = np.arange(nocc, nmo)

    tmp = take_eri(eri, o, o, o, o)
    eris.oooo = tmp.transpose(0, 2, 1, 3) - tmp.transpose(0, 2, 3, 1)
    tmp = None

    tmp = take_eri(eri, o, o, o, v)
    eris.ooov = tmp.transpose(0, 2, 1, 3) - tmp.transpose(2, 0, 1, 3)
    tmp = None

    tmp = take_eri(eri, o, v, o, v)
    eris.oovv = tmp.transpose(0, 2, 1, 3) - tmp.transpose(0, 2, 3, 1)
    tmp = None

    tmp_oovv = take_eri(eri, o, o, v, v)
    tmp_ovvo = take_eri(eri, o, v, v, o)
    eris.ovov = tmp_oovv.transpose(0, 2, 1, 3) - tmp_ovvo.transpose(0, 2, 3, 1)
    eris.ovvo = tmp_ovvo.transpose(0, 2, 1, 3) - tmp_oovv.transpose(0, 2, 3, 1)
    tmp_oovv = tmp_ovvo = None

    tmp = take_eri(eri, o, v, v, v)
    eris.ovvv = tmp.transpose(0, 2, 1, 3) - tmp.transpose(0, 2, 3, 1)
    tmp = None

    tmp = take_eri(eri, v, v, v, v)
    eris.vvvv = tmp.transpose(0, 2, 1, 3) - tmp.transpose(0, 2, 3, 1)
    tmp = None

    # cas hamiltonian in the MO space
    ncore = mycc.ncore
    ncas = mycc.ncas
    dm_core = mycc._scf.make_rdm1(mycc.mo_core, mycc.mo_occ[:ncore])
    hcore = mycc._scf.get_hcore()
    vhf_core = mycc._scf.get_veff(mycc.mol, dm_core)
    e_core = np.einsum('ij, ji -> ', hcore, dm_core, optimize=True) + \
             np.einsum('ij, ji -> ', vhf_core, dm_core, optimize=True) * 0.5 + \
             mycc._scf.energy_nuc()

    mo_cas = mycc.mo_cas
    h1_cas = mdot(mo_cas.conj().T, hcore + vhf_core, mo_cas)
    idx_cas = np.arange(ncore, ncore+ncas)
    h2_cas = take_eri(eri, idx_cas, idx_cas, idx_cas, idx_cas, compact=True)
    eris.h0_cas = e_core
    eris.h1_cas = h1_cas
    eris.h2_cas = h2_cas
    eris.rdm1_core = dm_core
    eris.vhf_core = vhf_core
    return eris

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import cc

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = 'cc-pvdz'
    mol.spin = 0
    mol.build(verbose=4)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.run()
    mf = scf.addons.convert_to_ghf(mf)
    rdm1_mf_ref = mf.make_rdm1()
    E_mf_ref = mf.e_tot

    mycc = cc.GCCSD(mf)
    e_cc_ref, t1, t2 = mycc.kernel()

    from libdmet.system import integral
    from libdmet.solver import scf as scf_hp
    from libdmet.solver import scf_solver
    from libdmet.utils import tile_eri

    nelec = mol.nelectron
    nao = mol.nao_nr()
    nso = nao * 2

    # core
    # O 1s O 2s -> 2 core orbitals (4 spin orbitals), 4 core electrons
    # val
    # O 2p H 1s x 2 -> 5 val orbitals (10 spin orbitals), 6 val electrons
    # virt
    # 34 spin orbitals

    ncas = 10
    nelecas = 6

    e_nuc = mf.energy_nuc()
    hcore = mf.get_hcore()
    ovlp = mf.get_ovlp()
    eri_ao = ao2mo.restore(4, mf._eri, nao)
    eri = tile_eri(eri_ao, eri_ao, eri_ao)
    eri_ao = None

    print (hcore.shape)
    print (ovlp.shape)
    print (eri.shape)

    Ham = integral.Integral(hcore.shape[-1], True, False, e_nuc, {"cd": hcore[None]},
                            {"ccdd": eri[None]}, ovlp=ovlp)


    solver = scf_solver.SCFSolver(ghf=True, tol=1e-10, max_cycle=200,
                                  oomp2=False, tol_normt=1e-6, ci_conv_tol=1e-8,
                                  level_shift=0.1, restart=True, mc_conv_tol=1e-6)

    rdm1_mf, E_mf = solver.run(Ham, nelec=mol.nelectron, dm0=rdm1_mf_ref)

    diff_mf = abs(E_mf - E_mf_ref)
    print ("E_mf : ", E_mf)
    print ("diff to ref : ", diff_mf)
    assert diff_mf < 1e-8

    mf = solver.scfsolver.mf

    mycc = GGTCCSD(mf, ncas=ncas, nelecas=nelecas)
    mycc.conv_tol   = 1e-12
    e_cc, t1, t2 = mycc.kernel()

    print (e_cc)
    e_cc_ref = -0.213484111125395
    diff_cc = abs(e_cc - e_cc_ref)
    print ("e_cc : ", e_cc)
    print ("diff to ref : ", diff_cc)
    assert diff_cc < 1e-8

    from libdmet.solver import impurity_solver
    solver = impurity_solver.Block2(nproc=1, nthread=1, nnode=1, TmpDir="./tmp", \
        SharedDir=None, reorder=True, minM=250, maxM=2000, tol=1e-12, \
        spinAdapted=False, bcs=False, use_general_spin=False, mem=1)

    mycc = GGTCCSD(mf, ncas=ncas, nelecas=nelecas)
    mycc.conv_tol   = 1e-12
    mycc.cisolver   = solver
    e_cc_dmrg, t1, t2 = mycc.kernel()

    print (e_cc_dmrg)
    e_cc_ref = -0.213484111125395
    diff_cc = abs(e_cc_dmrg- e_cc_ref)
    print ("e_cc_dmrg : ", e_cc_dmrg)
    print ("diff to ref : ", diff_cc)
    assert diff_cc < 1e-8

    from libdmet.solver import impurity_solver
    solver = impurity_solver.Block2(nproc=1, nthread=1, nnode=1, TmpDir="./tmp", \
        SharedDir=None, reorder=True, minM=250, maxM=2000, tol=1e-12, \
        spinAdapted=False, bcs=False, use_general_spin=True, mem=1)

    mycc = GGTCCSD(mf, ncas=ncas, nelecas=nelecas)
    mycc.conv_tol   = 1e-12
    mycc.cisolver   = solver
    e_cc_dmrg, t1, t2 = mycc.kernel()

    print (e_cc_dmrg)
    e_cc_ref = -0.213484111125395
    diff_cc = abs(e_cc_dmrg- e_cc_ref)
    print ("e_cc_dmrg : ", e_cc_dmrg)
    print ("diff to ref : ", diff_cc)
    assert diff_cc < 1e-8

