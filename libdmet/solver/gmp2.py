#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
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
GMP2 with chemists' ERI.

Author:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

import copy
import numpy as np
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf import scf
from pyscf import ao2mo
from pyscf.mp import mp2
from pyscf import __config__

from libdmet.utils import max_abs

WITH_T2 = getattr(__config__, 'mp_gmp2_with_t2', True)

def kernel(mp, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2, verbose=None):
    if mo_energy is not None or mo_coeff is not None:
        # For backward compatibility.  In pyscf-1.4 or earlier, mp.frozen is
        # not supported when mo_energy or mo_coeff is given.
        assert(mp.frozen == 0 or mp.frozen is None)

    if eris is None:
        eris = mp.ao2mo(mo_coeff)

    if mo_energy is None:
        mo_energy = eris.mo_energy

    nocc = mp.nocc
    nvir = mp.nmo - nocc
    eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]

    if with_t2:
        t2 = np.empty((nocc, nocc, nvir, nvir), dtype=eris.ovov.dtype)
    else:
        t2 = None

    emp2 = 0
    for i in range(nocc):
        if isinstance(eris.ovov, np.ndarray) and eris.ovov.ndim == 4:
            # When mf._eri is a custom integrals wiht the shape (n,n,n,n), the
            # ovov integrals might be in a 4-index tensor.
            gi = eris.ovov[i]
        else:
            gi = np.asarray(eris.ovov[i*nvir:(i+1)*nvir])

        gi = gi.reshape(nvir,nocc,nvir).transpose(1, 0, 2)
        t2i = gi.conj() / lib.direct_sum('jb+a -> jba', eia, eia[i])
        emp2 += np.einsum('jab, jab', t2i, gi, optimize=True)
        emp2 -= np.einsum('jab, jba', t2i, gi, optimize=True)
        if with_t2:
            t2[i] = t2i
    emp2 *= 0.5
    return emp2.real, t2

# Iteratively solve MP2 if non-canonical HF is provided
def _iterative_kernel(mp, eris, verbose=None, t2=None):
    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mp, verbose)
    
    if t2 is None:
        emp2, t2 = mp.init_amps(eris=eris)
    else:
        log.info('Use customized t2 initial guess')
        emp2 = mp.energy(t2, eris)
    log.info('Init E(MP2) = %.15g', emp2)

    adiis = lib.diis.DIIS(mp)
    adiis.space = mp.diis_space

    conv = False
    for istep in range(mp.max_cycle):
        t2new = mp.update_amps(t2, eris)

        if isinstance(t2new, np.ndarray):
            normt = np.linalg.norm(t2new - t2)
            t2 = None
            t2new = adiis.update(t2new)
        else: # UMP2
            normt = np.linalg.norm([np.linalg.norm(t2new[i] - t2[i])
                                       for i in range(3)])
            t2 = None
            t2shape = [x.shape for x in t2new]
            t2new = np.hstack([x.ravel() for x in t2new])
            t2new = adiis.update(t2new)
            t2new = lib.split_reshape(t2new, t2shape)

        t2, t2new = t2new, None
        emp2, e_last = mp.energy(t2, eris), emp2
        log.info('cycle = %d  E_corr(MP2) = %.15g  dE = %.9g  norm(t2) = %.6g',
                 istep+1, emp2, emp2 - e_last, normt)
        cput1 = log.timer('MP2 iter', *cput1)
        if abs(emp2-e_last) < mp.conv_tol and normt < mp.conv_tol_normt:
            conv = True
            break
    log.timer('MP2', *cput0)
    return conv, emp2, t2

def energy(mp, t2, eris):
    '''MP2 energy'''
    nocc, nvir = t2.shape[1:3]
    eris_ovov = np.asarray(eris.ovov).reshape(nocc, nvir, nocc, nvir)
    emp2  = np.einsum('ijab, iajb', t2, eris_ovov, optimize=True)
    emp2 -= np.einsum('ijab, ibja', t2, eris_ovov, optimize=True)
    emp2 *= 0.5
    return emp2.real

def update_amps(mp, t2, eris):
    '''Update non-canonical MP2 amplitudes'''
    nocc, nvir = t2.shape[1:3]
    fock = eris.fock
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mp.level_shift

    foo = fock[:nocc, :nocc] - np.diag(mo_e_o)
    fvv = fock[nocc:, nocc:] - np.diag(mo_e_v)
    t2new  = lib.einsum('ijac, bc -> ijab', t2, fvv)
    t2new -= lib.einsum('ki, kjab -> ijab', foo, t2)
    t2new = t2new + t2new.transpose(1, 0, 3, 2)

    eris_ovov = np.asarray(eris.ovov).reshape(nocc, nvir, nocc, nvir)
    t2new += eris_ovov.conj().transpose(0, 2, 1, 3)
    eris_ovov = None

    eia = mo_e_o[:, None] - mo_e_v
    for i in range(nocc):
        t2new[i] /= lib.direct_sum('a, jb -> jab', eia[i], eia)
    return t2new

def make_rdm1(mp, t2=None, eris=None, ao_repr=False):
    r'''Spin-traced one-particle density matrix.
    The occupied-virtual orbital response is not included.

    dm1[p,q] = <q_alpha^\dagger p_alpha> + <q_beta^\dagger p_beta>

    The convention of 1-pdm is based on McWeeney's book, Eq (5.4.20).
    The contraction between 1-particle Hamiltonian and rdm1 is
    E = einsum('pq,qp', h1, rdm1)

    Kwargs:
        ao_repr : boolean
            Whether to transfrom 1-particle density matrix to AO
            representation.
    '''
    from pyscf.cc import gccsd_rdm
    doo, dvv = _gamma1_intermediates(mp, t2, eris)
    nocc = doo.shape[0]
    nvir = dvv.shape[0]
    dov = np.zeros((nocc, nvir), dtype=doo.dtype)
    dvo = dov.T
    return gccsd_rdm._make_rdm1(mp, (doo, dov, dvo, dvv), with_frozen=True,
                                ao_repr=ao_repr)

def _gamma1_intermediates(mp, t2=None, eris=None):
    if t2 is None: t2 = mp.t2
    nmo = mp.nmo
    nocc = mp.nocc
    nvir = nmo - nocc
    if t2 is None:
        if eris is None:
            eris = mp.ao2mo()
        mo_energy = eris.mo_energy
        eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
        dtype = eris.ovov.dtype
    else:
        dtype = t2.dtype

    dm1occ = np.zeros((nocc, nocc), dtype=dtype)
    dm1vir = np.zeros((nvir, nvir), dtype=dtype)
    for i in range(nocc):
        if t2 is None:
            gi = np.asarray(eris.ovov[i*nvir:(i+1)*nvir])
            gi = gi.reshape(nvir, nocc, nvir).transpose(1,0,2)
            t2i = gi.conj() / lib.direct_sum('jb+a -> jba', eia, eia[i])
        else:
            t2i = t2[i]
        l2i = t2i.conj()
        dm1vir += lib.einsum('jca, jcb -> ba', l2i, t2i)
        dm1vir -= lib.einsum('jca, jbc -> ba', l2i, t2i)
        dm1occ -= lib.einsum('iab, jab -> ij', l2i, t2i)
        dm1occ += lib.einsum('iab, jba -> ij', l2i, t2i)
    return dm1occ, dm1vir

def make_rdm2(mp, t2=None, eris=None, ao_repr=False):
    r'''
    Spin-traced two-particle density matrix in MO basis

    dm2[p,q,r,s] = \sum_{sigma,tau} <p_sigma^\dagger r_tau^\dagger s_tau q_sigma>

    Note the contraction between ERIs (in Chemist's notation) and rdm2 is
    E = einsum('pqrs,pqrs', eri, rdm2)
    '''
    if t2 is None: t2 = mp.t2
    nmo = nmo0 = mp.nmo
    nocc = nocc0 = mp.nocc
    nvir = nmo - nocc
    if t2 is None:
        if eris is None:
            eris = mp.ao2mo()
        mo_energy = eris.mo_energy
        eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]

    if mp.frozen is not None:
        nmo0 = mp.mo_occ.size
        nocc0 = np.count_nonzero(mp.mo_occ > 0)
        moidx = get_frozen_mask(mp)
        oidx = np.where(moidx & (mp.mo_occ > 0))[0]
        vidx = np.where(moidx & (mp.mo_occ ==0))[0]
    else:
        moidx = oidx = vidx = None

    dm1 = make_rdm1(mp, t2, eris)
    dm1[np.diag_indices(nocc0)] -= 1

    dm2 = np.zeros((nmo0,nmo0,nmo0,nmo0), dtype=dm1.dtype) # Chemist notation
    for i in range(nocc):
        if t2 is None:
            gi = np.asarray(eris.ovov[i*nvir:(i+1)*nvir])
            gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
            t2i = gi.conj() / lib.direct_sum('jb+a -> jba', eia, eia[i])
        else:
            t2i = t2[i]
        # dm2 was computed as dm2[p,q,r,s] = < p^\dagger r^\dagger s q > in the
        # above. Transposing it so that it be contracted with ERIs (in Chemist's
        # notation):
        #   E = einsum('pqrs,pqrs', eri, rdm2)
        dovov = t2i.transpose(1, 0, 2) - t2i.transpose(2, 0, 1)
        if moidx is None:
            dm2[i,nocc:,:nocc,nocc:] = dovov
            dm2[nocc:,i,nocc:,:nocc] = dovov.conj().transpose(0, 2, 1)
        else:
            dm2[oidx[i],vidx[:,None,None],oidx[:,None],vidx] = dovov
            dm2[vidx[:,None,None],oidx[i],vidx[:,None],oidx] = dovov.conj().transpose(0, 2, 1)

    # Be careful with convention of dm1 and dm2
    #   dm1[q,p] = <p^\dagger q>
    #   dm2[p,q,r,s] = < p^\dagger r^\dagger s q >
    #   E = einsum('pq,qp', h1, dm1) + .5 * einsum('pqrs,pqrs', eri, dm2)
    # When adding dm1 contribution, dm1 subscripts need to be flipped
    for i in range(nocc0):
        dm2[i,i,:,:] += dm1.T #* 2
        dm2[:,:,i,i] += dm1.T #* 2
        dm2[:,i,i,:] -= dm1.T
        dm2[i,:,:,i] -= dm1

    for i in range(nocc0):
        for j in range(nocc0):
            dm2[i,i,j,j] += 1
            dm2[i,j,j,i] -= 1

    if ao_repr:
        from pyscf.cc import ccsd_rdm
        dm2 = ccsd_rdm._rdm2_mo2ao(dm2, mp.mo_coeff)
    return dm2

class GGMP2(mp2.MP2):
    diis_space = getattr(__config__, 'mp_gmp2_GGMP2_diis_space', 8)
    
    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        assert(isinstance(mf, scf.ghf.GHF))
        mp2.MP2.__init__(self, mf, frozen, mo_coeff, mo_occ)
    
    def ao2mo(self, mo_coeff=None):
        return _make_eris(self, mo_coeff, verbose=self.verbose)
    
    def kernel(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2,
               t2=None):
        '''
        Args:
            with_t2 : bool
                Whether to generate and hold t2 amplitudes in memory.
        '''
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()

        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        
        if hasattr(self, "get_e_hf"):
            # pyscf 2.1
            self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)
        else:
            self.e_hf = getattr(eris, 'e_hf', None)
            if self.e_hf is None:
                self.e_hf = self._scf.e_tot

        if self._scf.converged:
            self.e_corr, self.t2 = self.init_amps(mo_energy, mo_coeff, eris, with_t2)
        else:
            self.converged, self.e_corr, self.t2 = _iterative_kernel(self, eris, t2=t2)
        
        self.e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        self.e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        self.e_corr = float(self.e_corr)

        self._finalize()
        return self.e_corr, self.t2

    make_rdm1 = make_rdm1
    make_rdm2 = make_rdm2

    def density_fit(self, auxbasis=None, with_df=None):
        from pyscf.mp import dfgmp2
        mymp = dfgmp2.DFGMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
        if with_df is not None:
            mymp.with_df = with_df
        if mymp.with_df.auxbasis != auxbasis:
            mymp.with_df = copy.copy(mymp.with_df)
            mymp.with_df.auxbasis = auxbasis
        return mymp

    def nuc_grad_method(self):
        raise NotImplementedError

    # For non-canonical MP2
    energy = energy
    update_amps = update_amps
    
    def init_amps(self, mo_energy=None, mo_coeff=None, eris=None, with_t2=WITH_T2):
        return kernel(self, mo_energy, mo_coeff, eris, with_t2)

MP2 = GGMP2

def _make_eris(mp, mo_coeff=None, ao2mofn=None, verbose=None):
    log = logger.new_logger(mp, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    eris = mp2._ChemistsERIs()
    eris._common_init_(mp, mo_coeff)
    mo_coeff = eris.mo_coeff
    
    nao = mo_coeff.shape[-2]
    nocc = mp.nocc
    nmo = mp.nmo
    nvir = nmo - nocc
    mem_incore, mem_outcore, mem_basic = mp2._mem_usage(nocc, nvir)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mp.max_memory - mem_now)
    if max_memory < mem_basic:
        log.warn('Not enough memory for integral transformation. '
                 'Available mem %s MB, required mem %s MB',
                 max_memory, mem_basic)

    co = np.asarray(mo_coeff[:,:nocc], order='F')
    cv = np.asarray(mo_coeff[:,nocc:], order='F')
    if (mp.mol.incore_anyway or
        (mp._scf._eri is not None and mem_incore < max_memory)):
        log.debug('transform (ia|jb) incore')
        
        if (nao == nmo) and (max_abs(eris.mo_coeff - np.eye(nmo)) < 1e-13):
            # ZHC NOTE special treatment for OOMP2,
            # where the ao2mo is not needed for identity mo_coeff.
            eri = ao2mo.restore(1, mp._scf._eri, nao).reshape(nao, nao, nao, nao)
            eris.ovov = eri[:nocc, nocc:, :nocc, nocc:]
        else:
            if callable(ao2mofn):
                eris.ovov = ao2mofn((co,cv,co,cv)).reshape(nocc*nvir,nocc*nvir)
            else:
                eris.ovov = ao2mo.general(mp._scf._eri, (co,cv,co,cv))

    elif getattr(mp._scf, 'with_df', None):
        # To handle the PBC or custom 2-electron with 3-index tensor.
        # Call dfmp2.MP2 for efficient DF-MP2 implementation.
        log.warn('DF-HF is found. (ia|jb) is computed based on the DF '
                 '3-tensor integrals.\n'
                 'You can switch to dfmp2.MP2 for better performance')
        log.debug('transform (ia|jb) with_df')
        eris.ovov = mp._scf.with_df.ao2mo((co,cv,co,cv))

    else:
        log.debug('transform (ia|jb) outcore')
        eris.feri = lib.H5TmpFile()
        #ao2mo.outcore.general(mp.mol, (co,cv,co,cv), eris.feri,
        #                      max_memory=max_memory, verbose=log)
        #eris.ovov = eris.feri['eri_mo']
        eris.ovov = mp2._ao2mo_ovov(mp, co, cv, eris.feri, max(2000, max_memory), log)

    log.timer('Integral transformation', *time0)
    return eris

del(WITH_T2)

if __name__ == '__main__':
    import numpy as np
    from pyscf import gto, scf, ao2mo
    from libdmet_solid.system import integral
    from libdmet_solid.solver import scf as scf_hp
    from libdmet_solid.utils.misc import tile_eri
    from libdmet.solver import gmp2
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = 'cc-pvdz'
    mol.verbose = 4
    mol.build()
    mf = scf.RHF(mol).run()
    
    mf = scf.addons.convert_to_ghf(mf)
    dm0 = mf.make_rdm1()
    dm_guess = mf.get_init_guess()
    mf.kernel(dm0=dm0)
    
    ovlp = mf.get_ovlp()
    H0 = mf.energy_nuc()
    H1 = mf.get_hcore()
    H2 = mf._eri
    H2 = ao2mo.restore(4, H2, mol.nao_nr())
    
    H2 = tile_eri(H2, H2, H2)
    Ham = integral.Integral(H1.shape[-1], True, False, H0, {"cd": H1[None]},
                            {"ccdd": H2[None]}, ovlp=ovlp)
    
    scfsolver = scf_hp.SCF(newton_ah=True)
    scfsolver.set_system(mol.nelectron, 0, False, True, max_memory=mol.max_memory)
    scfsolver.set_integral(Ham)
    ehf, rhoHF = scfsolver.GGHF(tol=1e-12, InitGuess=dm0)
    emp2_ref, rdm1_mp2_ref = scfsolver.GMP2()
    
    emp2_ref2 = -0.204019967288338

    mf = scfsolver.mf

    pt = gmp2.GGMP2(mf)
    emp2, t2 = pt.kernel()
    
    diff_1 = abs(emp2 - emp2_ref)
    print("diff to emp2_ref from pyscf")
    print (diff_1)
    assert diff_1 < 1e-9

    diff_2 = abs(emp2 - emp2_ref2)
    print("diff to RMP2 from pyscf")
    print (diff_2)
    assert diff_2 < 1e-9
    
    rdm1 = pt.make_rdm1(ao_repr=True)
    rdm2 = pt.make_rdm2(ao_repr=True)
    
    E_re = np.einsum('pq, qp -> ', mf.get_hcore(), rdm1) + \
           np.einsum('pqrs, pqrs ->', ao2mo.restore(1, mf._eri, mol.nao_nr()*2), rdm2) * 0.5 + \
           mf.energy_nuc()

    diff_3 = abs(E_re - pt.e_tot)
    print ("diff E from rdm12")
    print (diff_3)
    assert diff_3 < 1e-10
    
    diff_rdm1 = np.linalg.norm(rdm1 - rdm1_mp2_ref)
    print ("diff rdm1")
    print (diff_rdm1)
    assert diff_rdm1 < 1e-10
    
    # non canonical MP2
    mf = scf.RHF(mol).run(max_cycle=1)
    mf = scf.addons.convert_to_ghf(mf)
    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mo_energy = mf.mo_energy

    scfsolver = scf_hp.SCF(newton_ah=False, no_kernel=True)
    scfsolver.set_system(mol.nelectron, 0, False, True, max_memory=mol.max_memory)
    scfsolver.set_integral(Ham)
    ehf, rhoHF = scfsolver.GGHF(tol=1e-12, InitGuess=dm_guess, MaxIter=0)
    mf = scfsolver.mf
    mf.mo_coeff = mo_coeff
    mf.mo_occ = mo_occ
    mf.mo_energy = mo_energy

    pt = gmp2.GGMP2(mf)
    pt.conv_tol = 1e-10
    pt.conv_tol_normt = 1e-8
    E, t2 = pt.kernel()

    E_ref = -0.204479916653321
    diff_4 = abs(E - E_ref)
    print ("diff non canonical")
    print (diff_4)
    assert diff_4 < 1e-9
