#! /usr/bin/env python

"""
CC impurity solver.

Author:
    Zhi-Hao Cui
"""

import os
import numpy as np
import scipy.linalg as la
import h5py
import time
from functools import partial

from libdmet.utils import logger as log
from libdmet.system import integral
from libdmet.solver import scf
from libdmet.solver import lgccsd, lgccd
from libdmet.solver.scf import (ao2mo_Ham, restore_Ham, _get_veff,
                                _get_veff_ghf)
from libdmet.solver.ccd import CCD
from libdmet.solver.uccd import UCCD
from libdmet.solver.gccd import GCCD
from libdmet.solver import uccsd_ite
from libdmet.solver import gccsd_ite
from libdmet.basis_transform.make_basis import (transform_rdm1_to_ao_mol,
                                                transform_rdm2_to_ao_mol,
                                                rotate_emb_basis,
                                                find_closest_mo, trans_mo,
                                                get_mo_ovlp)
from libdmet.utils.misc import mdot, max_abs, take_eri
from pyscf.scf import hf
from pyscf import cc
from pyscf import ci
from pyscf import lib
from pyscf import ao2mo
from pyscf.cc import gccsd
from pyscf.cc.uccsd import _ChemistsERIs
from pyscf.cc.gccsd import _PhysicistsERIs
from pyscf.lib import logger


einsum = partial(np.einsum, optimize=True)

# ****************************************************************************
# UCC
# ****************************************************************************

def _make_eris_incore_uhf(mycc, mo_coeff=None, ao2mofn=None):
    """
    Hacked CC make eri function. NOTE the order.
    """
    from libdmet.utils import tril_take_idx
    assert ao2mofn is None
    cput0 = (time.process_time(), time.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    naoa, nmoa = moa.shape
    naob, nmob = mob.shape
    assert naoa == naob

    if len(mycc._scf._eri) == 1:
        eri_ao = [mycc._scf._eri[0], mycc._scf._eri[0], mycc._scf._eri[0]]
    elif len(mycc._scf._eri) == 3:
        eri_ao = mycc._scf._eri
    else:
        raise ValueError("Unknown ERI length %s"%(len(mycc._scf._eri)))
    
    # aa
    o = np.arange(0, nocca)
    v = np.arange(nocca, nmoa)
    
    # ZHC NOTE special treatment for OO-CCD,
    if (naoa == nmoa) and (max_abs(moa - np.eye(nmoa)) < 1e-13):
        eri_aa = ao2mo.restore(4, eri_ao[0], naoa)
    else:
        eri_aa = ao2mo.full(ao2mo.restore(4, eri_ao[0], naoa),
                            moa, compact=True)
    
    eris.oooo = take_eri(eri_aa, o, o, o, o)
    eris.ovoo = take_eri(eri_aa, o, v, o, o)
    eris.ovov = take_eri(eri_aa, o, v, o, v)
    eris.oovv = take_eri(eri_aa, o, o, v, v)
    eris.ovvo = take_eri(eri_aa, o, v, v, o)

    idx1 = tril_take_idx(o, v, compact=False)
    idx2 = tril_take_idx(v, v, compact=True)
    eris.ovvv = eri_aa[np.ix_(idx1, idx2)].reshape(nocca, nvira, nvira*(nvira+1)//2)
    
    eris.vvvv = take_eri(eri_aa, v, v, v, v, compact=True)
    eri_aa = None
    
    # bb
    O = np.arange(0, noccb)
    V = np.arange(noccb, nmob)

    if (naob == nmob) and (max_abs(mob - np.eye(nmob)) < 1e-13):
        eri_bb = ao2mo.restore(4, eri_ao[1], naob)
    else:
        eri_bb = ao2mo.full(ao2mo.restore(4, eri_ao[1], naob),
                            mob, compact=True)

    eris.OOOO = take_eri(eri_bb, O, O, O, O)
    eris.OVOO = take_eri(eri_bb, O, V, O, O)
    eris.OVOV = take_eri(eri_bb, O, V, O, V)
    eris.OOVV = take_eri(eri_bb, O, O, V, V)
    eris.OVVO = take_eri(eri_bb, O, V, V, O)

    idx1 = tril_take_idx(O, V, compact=False)
    idx2 = tril_take_idx(V, V, compact=True)
    eris.OVVV = eri_bb[np.ix_(idx1, idx2)].reshape(noccb, nvirb, nvirb*(nvirb+1)//2)
    
    eris.VVVV = take_eri(eri_bb, V, V, V, V, compact=True)
    eri_bb = None

    # ab
    if ((naoa == naob) and (nmoa == nmob)) and \
       ((naoa == nmoa) and (max_abs(moa - np.eye(nmoa)) < 1e-13)) and \
       ((naob == nmob) and (max_abs(mob - np.eye(nmob)) < 1e-13)):
        eri_ab = ao2mo.restore(4, eri_ao[2], naoa)
    else:
        eri_ab = ao2mo.general(ao2mo.restore(4, eri_ao[2], naoa),
                               (moa, moa, mob, mob), compact=True)
    eri_ao = None

    eris.ooOO = take_eri(eri_ab, o, o, O, O)
    eris.ovOO = take_eri(eri_ab, o, v, O, O)
    eris.ovOV = take_eri(eri_ab, o, v, O, V)
    eris.ooVV = take_eri(eri_ab, o, o, V, V)
    eris.ovVO = take_eri(eri_ab, o, v, V, O)
    
    idx1 = tril_take_idx(o, v, compact=False)
    eris.ovVV = eri_ab[np.ix_(idx1, idx2)].reshape(nocca, nvira, nvirb*(nvirb+1)//2)
    
    eris.vvVV = take_eri(eri_ab, v, v, V, V, compact=True)
    
    # ba
    eri_ba = eri_ab.T
    eri_ab = None

    eris.OVoo = take_eri(eri_ba, O, V, o, o)
    eris.OOvv = take_eri(eri_ba, O, O, v, v)
    eris.OVvo = take_eri(eri_ba, O, V, v, o)
    
    idx1 = tril_take_idx(O, V, compact=False)
    idx2 = tril_take_idx(v, v, compact=True)
    eris.OVvv = eri_ba[np.ix_(idx1, idx2)].reshape(noccb, nvirb, nvira*(nvira+1)//2)
    eri_ba = None
    return eris
    
def ao2mo_uhf(mycc, mo_coeff=None):
    nmoa, nmob = mycc.get_nmo()
    nao = mycc.mo_coeff[0].shape[0]
    nmo_pair = nmoa * (nmoa+1) // 2
    nao_pair = nao * (nao+1) // 2
    mem_incore = (max(nao_pair**2, nmoa**4) + nmo_pair**2) * 8/1e6
    mem_now = lib.current_memory()[0]
    if (mycc._scf._eri is not None and
        (mem_incore+mem_now < mycc.max_memory or mycc.incore_complete)):
        return _make_eris_incore_uhf(mycc, mo_coeff)

    elif getattr(mycc._scf, 'with_df', None):
        log.warn('UCCSD detected DF being used in the HF object. '
                 'MO integrals are computed based on the DF 3-index tensors.\n'
                 'It\'s recommended to use dfccsd.CCSD for the '
                 'DF-CCSD calculations')
        raise NotImplementedError

    else:
        raise NotImplementedError
        return _make_eris_outcore(mycc, mo_coeff)
    
def init_amps_uhf(mycc, eris=None):
    time0 = logger.process_clock(), logger.perf_counter()
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    nocca, noccb = mycc.nocc

    fova = eris.focka[:nocca,nocca:]
    fovb = eris.fockb[:noccb,noccb:]
    mo_ea_o = eris.mo_energy[0][:nocca]
    mo_ea_v = eris.mo_energy[0][nocca:] + mycc.level_shift
    mo_eb_o = eris.mo_energy[1][:noccb]
    mo_eb_v = eris.mo_energy[1][noccb:] + mycc.level_shift
    eia_a = lib.direct_sum('i-a->ia', mo_ea_o, mo_ea_v)
    eia_b = lib.direct_sum('i-a->ia', mo_eb_o, mo_eb_v)
    
    t1a = fova.conj() / eia_a
    t1b = fovb.conj() / eia_b

    eris_ovov = np.asarray(eris.ovov)
    eris_OVOV = np.asarray(eris.OVOV)
    eris_ovOV = np.asarray(eris.ovOV)
    t2aa = eris_ovov.transpose(0,2,1,3) / lib.direct_sum('ia+jb->ijab', eia_a, eia_a)
    t2ab = eris_ovOV.transpose(0,2,1,3) / lib.direct_sum('ia+jb->ijab', eia_a, eia_b)
    t2bb = eris_OVOV.transpose(0,2,1,3) / lib.direct_sum('ia+jb->ijab', eia_b, eia_b)
    t2aa = t2aa - t2aa.transpose(0,1,3,2)
    t2bb = t2bb - t2bb.transpose(0,1,3,2)
    e  =      np.einsum('iJaB,iaJB', t2ab, eris_ovOV)
    e += 0.25*np.einsum('ijab,iajb', t2aa, eris_ovov)
    e -= 0.25*np.einsum('ijab,ibja', t2aa, eris_ovov)
    e += 0.25*np.einsum('ijab,iajb', t2bb, eris_OVOV)
    e -= 0.25*np.einsum('ijab,ibja', t2bb, eris_OVOV)
    mycc.emp2 = e.real
    logger.info(mycc, 'Init t2, MP2 energy = %.15g', mycc.emp2)
    logger.timer(mycc, 'init mp2', *time0)
    return mycc.emp2, (t1a,t1b), (t2aa,t2ab,t2bb)
    
def make_rdm2_uhf(mycc, t1=None, t2=None, l1=None, l2=None, ao_repr=False):
    from libdmet.solver import uccsd_rdm
    if t1 is None: t1 = mycc.t1
    if t2 is None: t2 = mycc.t2
    if l1 is None: l1 = mycc.l1
    if l2 is None: l2 = mycc.l2
    if l1 is None: l1, l2 = mycc.solve_lambda(t1, t2)
    return uccsd_rdm.make_rdm2(mycc, t1, t2, l1, l2, ao_repr=ao_repr)

class UICCSD(cc.uccsd.UCCSD):
    init_amps = init_amps_uhf
    ao2mo = ao2mo_uhf
    make_rdm2 = make_rdm2_uhf

class UICCSDITE(uccsd_ite.UCCSDITE):
    init_amps = init_amps_uhf
    ao2mo = ao2mo_uhf
    make_rdm2 = make_rdm2_uhf

class UICCD(UCCD):
    init_amps = init_amps_uhf
    ao2mo = ao2mo_uhf

# ****************************************************************************
# GCC
# ****************************************************************************

def _make_eris_incore_ghf_direct(mycc, mo_coeff=None, ao2mofn=None):
    """
    Incore GGCC ERI ao2mo. 
    Memory usage is optimized:
    required additional memory ~ vvvv + 1/8 * pppp (normal case is 2 * pppp)
    """
    cput0 = (time.process_time(), time.perf_counter())
    eris = _PhysicistsERIsdirect()
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
        if eri.size == nmo ** 4:
            eri = ao2mo.restore(8, eri, nmo)
        
    #eri = eri.transpose(0,2,1,3) - eri.transpose(0,2,3,1)
    # resverse index # 0 2 1 3 - 0 3 1 2
    o = np.arange(0, nocc)
    v = np.arange(nocc, nmo)

    tmp = take_eri(eri, o, o, o, o)
    #eris.doooo = tmp.transpose(0, 2, 1, 3)
    eris.oooo =  tmp.transpose(0, 2, 1, 3) #- tmp.transpose(0, 2, 3, 1)
    tmp = None

    tmp = take_eri(eri, o, o, o, v)
    #eris.dooov = tmp.transpose(0, 2, 1, 3)
    eris.ooov = tmp.transpose(0, 2, 1, 3) - tmp.transpose(2, 0, 1, 3)
    tmp = None

    tmp = take_eri(eri, o, v, o, v)
    #eris.doovv = tmp.transpose(0, 2, 1, 3)
    eris.oovv = tmp.transpose(0, 2, 1, 3) #- tmp.transpose(0, 2, 3, 1)
    tmp = None
    
    tmp_oovv = take_eri(eri, o, o, v, v)
    tmp_ovvo = take_eri(eri, o, v, v, o)
    #eris.dovov = tmp_oovv.transpose(0, 2, 1, 3)
    eris.ovov = tmp_oovv.transpose(0, 2, 1, 3) #- tmp_ovvo.transpose(0, 2, 3, 1)
    #eris.dovvo = tmp_ovvo.transpose(0, 2, 1, 3)
    eris.ovvo = tmp_ovvo.transpose(0, 2, 1, 3) #- tmp_oovv.transpose(0, 2, 3, 1)
    tmp_oovv = tmp_ovvo = None

    tmp = take_eri(eri, o, v, v, v)
    #eris.dovvv = tmp.transpose(0, 2, 1, 3)
    eris.ovvv = tmp.transpose(0, 2, 1, 3) - tmp.transpose(0, 2, 3, 1)
    tmp = None
    
    tmp = take_eri(eri, v, v, v, v)
    #eris.dvvvv = tmp.transpose(0, 2, 1, 3)
    eris.vvvv = tmp.transpose(0, 2, 1, 3) #- tmp.transpose(0, 2, 3, 1)
    tmp = None
    return eris


def _make_eris_incore_ghf(mycc, mo_coeff=None, ao2mofn=None):
    """
    Incore GGCC ERI ao2mo. 
    Memory usage is optimized:
    required additional memory ~ vvvv + 1/8 * pppp (normal case is 2 * pppp)
    """
    cput0 = (time.process_time(), time.perf_counter())
    eris = _PhysicistsERIs()
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
        if eri.size == nmo ** 4:
            eri = ao2mo.restore(8, eri, nmo)
        
    #eri = eri.transpose(0,2,1,3) - eri.transpose(0,2,3,1)
    # resverse index # 0 2 1 3 - 0 3 1 2
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
    return eris
    
def ao2mo_ghf(mycc, mo_coeff=None):
    nmo = mycc.nmo
    mem_incore = nmo**4*2 * 8/1e6
    mem_now = lib.current_memory()[0]
    if (mycc._scf._eri is not None and
        (mem_incore+mem_now < mycc.max_memory) or mycc.mol.incore_anyway):
        return _make_eris_incore_ghf(mycc, mo_coeff)
    elif getattr(mycc._scf, 'with_df', None):
        raise NotImplementedError
    else:
        raise NotImplementedError
        return _make_eris_outcore(mycc, mo_coeff)

def ao2mo_ghf_direct(mycc, mo_coeff=None):
    nmo = mycc.nmo
    mem_incore = nmo**4*2 * 8/1e6
    mem_now = lib.current_memory()[0]
    if (mycc._scf._eri is not None and
        (mem_incore+mem_now < mycc.max_memory) or mycc.mol.incore_anyway):
        return _make_eris_incore_ghf_direct(mycc, mo_coeff)
    elif getattr(mycc._scf, 'with_df', None):
        raise NotImplementedError
    else:
        raise NotImplementedError
        return _make_eris_outcore(mycc, mo_coeff)

def init_amps_ghf(mycc, eris=None):
    """
    GCCSD initial amplitudes from level-shift MP2.
    """
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    
    nocc = mycc.nocc
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift
    eia = mo_e_o[:, None] - mo_e_v
    t1 = eris.fock[:nocc, nocc:] / eia
    eris_oovv = np.asarray(eris.oovv)
    t2 = np.array(eris_oovv, copy=True).conj()
    emp2 = 0.0
    for i in range(nocc):
        t2[i] /= lib.direct_sum('a, jb -> jab', eia[i], eia)
        emp2 += 0.25 * einsum('jab, jab', t2[i], eris_oovv[i])
    mycc.emp2 = emp2
    log.info('Init t2, MP2 energy = %.15g', mycc.emp2)
    return mycc.emp2, t1, t2

def init_d_amps_ghf(mycc, eris=None):
    """
    GCCSD initial amplitudes from level-shift MP2.
    """
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    
    nocc = mycc.nocc
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift
    eia = mo_e_o[:, None] - mo_e_v
    t1 = eris.fock[:nocc, nocc:] / eia
    #eris_doovv = np.asarray(eris.doovv)
    eris_doovv = np.asarray(eris.oovv)
    t2 = np.array(eris_doovv, copy=True).conj()
    emp2 = 0.0
    for i in range(nocc):
        t2[i] /= lib.direct_sum('a, jb -> jab', eia[i], eia)
        emp2 += 0.25 * einsum('jab, jab', t2[i], eris_doovv[i])
    mycc.emp2 = emp2
    log.info('Init t2, MP2 energy = %.15g', mycc.emp2)
    return mycc.emp2, t1, t2


def analyze_amps(mycc):
    """
    Analyze the blocks of amps, based on spin channels.
    """
    log = logger.new_logger(mycc)
    t1 = mycc.t1
    t2 = mycc.t2
    nvir_a = mycc.nvir_a
    nocc_a = mycc.nocc_a
    assert nocc_a is not None and nvir_a is not None

    log.info('*' * 79)
    log.info('Analyze amps based on spin channels.')
    log.info("Tx %27s %12s %12s",
             "i      a      j      b    ", "max_abs", "la.norm")
    log.info('-' * 79)

    t1T = t1.T
    nvir, nocc = t1T.shape
    dic_o = {'a': slice(0, nocc_a), 'b': slice(nocc_a, nocc)}
    dic_v = {'a': slice(0, nvir_a), 'b': slice(nvir_a, nvir)}
    
    labs_o = {'a': "a (p)", 'b': "b (h)"}
    labs_v = {'a': "a (h)", 'b': "b (p)"}

    for o0 in ['a', 'b']:
        for v0 in ['a', 'b']:
            block = t1T[dic_v[v0], dic_o[o0]]
            norm_max = max_abs(block)
            norm_tot = la.norm(block)
            log.info("T1 %6s %6s %6s %6s %12.6f %12.6f",
                     labs_o[o0], labs_v[v0], "", "", norm_max, norm_tot)
    
    t2T = t2.transpose(2, 3, 0, 1)
        
    dic_o = {'a': slice(0, nocc_a), 'b': slice(nocc_a, nocc)}
    dic_v = {'a': slice(0, nvir_a), 'b': slice(nvir_a, nvir)}
    
    log.info('-' * 79)
    for o0 in ['a', 'b']:
        for v0 in ['a', 'b']:
            for o1 in ['a', 'b']:
                for v1 in ['a', 'b']:
                    block = t2T[dic_v[v0], dic_v[v1], dic_o[o0], dic_o[o1]]
                    norm_max = max_abs(block)
                    norm_tot = la.norm(block)
                    log.info("T2 %6s %6s %6s %6s %12.6f %12.6f",
                             labs_o[o0], labs_v[v0], labs_o[o1], labs_v[v1],
                             norm_max, norm_tot)
    log.info('*' * 79)

class GGCCSD(gccsd.GCCSD):
    init_amps = init_amps_ghf
    ao2mo = ao2mo_ghf

class GGCCSDITE(gccsd_ite.GCCSDITE):
    init_amps = init_amps_ghf
    ao2mo = ao2mo_ghf

class GGCCSDITE_RK(gccsd_ite.GCCSDITE_RK):
    init_amps = init_amps_ghf
    ao2mo = ao2mo_ghf

class GGCCSD_KRYLOV(gccsd_ite.GCCSD_KRYLOV):
    init_amps = init_amps_ghf
    ao2mo = ao2mo_ghf
    analyze_amps = analyze_amps

class LGGCCSD(lgccsd.LGCCSD):
    init_amps = init_amps_ghf
    ao2mo = ao2mo_ghf

class GGCCD(GCCD):
    init_amps = init_amps_ghf
    ao2mo = ao2mo_ghf

class LGGCCD(lgccd.LGCCD):
    init_amps = init_amps_ghf
    ao2mo = ao2mo_ghf

class GGCISD(ci.gcisd.GCISD):
    init_amps = init_amps_ghf
    ao2mo = ao2mo_ghf

try:
    from CC_Variants import (ringccd, ringccsd, ladderccsd, dringccsd, dladderccsd,
                             dringccd, dringccsd_diag, dringccsd_diag_krylov)
    from CC_Variants.dladderccsd import _PhysicistsERIs as _PhysicistsERIsdirect
    
    class RINGGCCD(ringccd.RINGCCD):
        init_amps = init_amps_ghf
        ao2mo = ao2mo_ghf

    class RINGGCCSD(ringccsd.RINGCCSD):
        init_amps = init_amps_ghf
        ao2mo = ao2mo_ghf

    class DRINGGCCSD(dringccsd.DRINGCCSD):
        init_amps = init_d_amps_ghf
        ao2mo = ao2mo_ghf_direct

    class DRINGGCCSD_DIAG(dringccsd_diag.DRINGCCSD):
        init_amps = init_d_amps_ghf
        ao2mo = ao2mo_ghf_direct
    
    class DRINGGCCSD_DIAG_KRYLOV(dringccsd_diag_krylov.DRINGCCSD_KRYLOV):
        init_amps = init_d_amps_ghf
        ao2mo = ao2mo_ghf_direct

    class DRINGGCCD(dringccd.DRINGCCD):
        init_amps = init_d_amps_ghf
        ao2mo = ao2mo_ghf_direct

    class DLADGGCCSD(dladderccsd.DLADGCCSD):
        init_amps = init_d_amps_ghf
        ao2mo = ao2mo_ghf_direct

    class LADGGCCSD(ladderccsd.LADGCCSD):
        init_amps = init_amps_ghf
        ao2mo = ao2mo_ghf

except ImportError:
    pass

# ****************************************************************************
# Main solver
# ****************************************************************************

class CCSD(object):
    def __init__(self, nproc=1, nnode=1, nthread=28, TmpDir="./tmp", SharedDir=None, 
                 restricted=False, Sz=0, bcs=False, ghf=False, tol=1e-7, 
                 tol_normt=1e-5, max_cycle=200, level_shift=0.0, frozen=0, 
                 max_memory=40000, compact_rdm2=False, scf_newton=True,
                 diis_space=8, diis_start_cycle=None, iterative_damping=1.0,
                 linear=False, approx_l=False, alpha=None, beta=np.inf, tcc=False,
                 ite=None, ite_rk=None, ovlp_tol=0.9, use_mpi=False, ring=False, ladder=False,
                 directring=False, directladder=False, directringdoubles=False, directring_diag=False):
        """
        CCSD solver.
        """
        self.nnode = nnode
        self.nproc = nproc
        self.nthread = nthread
        self.use_mpi = ((self.nnode * self.nproc) > 1) or use_mpi

        self.restricted = restricted
        self.max_cycle = max_cycle
        self.max_memory = max_memory
        self.conv_tol = tol
        self.conv_tol_normt = tol_normt
        self.level_shift = level_shift
        self.diis_space = diis_space
        self.diis_start_cycle = diis_start_cycle
        self.iterative_damping = iterative_damping
        self.frozen = frozen
        self.verbose = 5
        self.bcs = bcs
        self.ghf = ghf
        self.Sz = Sz
        self.alpha = alpha
        self.beta = beta
        self.linear = linear
        self.tcc = tcc
        self.ite = ite
        self.ite_rk = ite_rk
        self.approx_l = approx_l
        self.scfsolver = scf.SCF(newton_ah=scf_newton)
        self.ring = ring
        self.ladder = ladder
        self.directring = directring
        self.directladder = directladder
        self.directringdoubles = directringdoubles
        self.directring_diag = directring_diag
        self.onepdm = None
        self.twopdm = None

        self.ovlp_tol = ovlp_tol
        self.optimized = False
    
    def run(self, Ham=None, nelec=None, guess=None, restart=False,
            dump_tl=False, fcc_name="fcc.h5", fcc_name_save=None,
            calc_rdm2=False, Mu=None, **kwargs):
        """
        Main kernel function of the solver.
        NOTE: the spin order for unrestricted H2 is aa, bb, ab.
        
        kwargs:
            remove_h2: if True, will modify the Ham object, set Ham.H2["ccdd"] = None.
        """
        # *********************************************************************
        # 1. sanity check
        # *********************************************************************
        log.info("CC solver: start")
        spin = Ham.H1["cd"].shape[0]
        if spin > 1:
            log.eassert(not self.restricted, "CC solver: spin (%s) > 1 " 
                        "requires unrestricted", spin)
        if nelec is None:
            if self.bcs:
                nelec = Ham.norb * 2
            elif self.ghf:
                nelec = Ham.norb // 2
            else:
                raise ValueError("CC solver: nelec cannot be None "
                                 "for RCC or UCC.")

        nelec_a, nelec_b = (nelec + self.Sz) // 2, (nelec - self.Sz) // 2
        log.eassert(nelec_a >= 0 and nelec_b >=0, "CC solver: " 
                    "nelec_a (%s), nelec_b (%s) should >= 0", nelec_a, nelec_b)
        log.eassert(nelec_a + nelec_b == nelec, "CC solver: " 
                    "nelec_a (%s) + nelec_b (%s) should == nelec (%s)", 
                    nelec_a, nelec_b, nelec)
        
        # *********************************************************************
        # 2. mean-field calculation
        # *********************************************************************
        log.debug(1, "CC solver: mean-field")
        self.scfsolver.set_system(nelec, self.Sz, False, self.restricted, 
                                  max_memory=self.max_memory)
        self.scfsolver.set_integral(Ham)

        dm0 = kwargs.get("dm0", None)
        scf_max_cycle = kwargs.get("scf_max_cycle", 200)
        remove_h2 = kwargs.get("remove_h2", False)
        bcc = kwargs.get("bcc", False)
        bcc_verbose = kwargs.get("bcc_verbose", 2)
        bcc_restart = kwargs.get("bcc_restart", False)
        bcc_tol = kwargs.get("bcc_tol", self.conv_tol_normt * 10.0)
        if bcc and bcc_restart and self.optimized and restart:
            bcc_restart = True
            scf_max_cycle = 1 # need not to do scf
        else:
            bcc_restart = False
        
        if self.ghf:
            E_HF, rhoHF = self.scfsolver.GGHF(tol=min(self.conv_tol*0.1, 1e-10), 
                                              MaxIter=scf_max_cycle,
                                              InitGuess=dm0,
                                              Mu=Mu,
                                              alpha=self.alpha,
                                              beta=self.beta)
        else:
            E_HF, rhoHF = self.scfsolver.HF(tol=min(self.conv_tol*0.1, 1e-10),
                                            MaxIter=scf_max_cycle,
                                            InitGuess=dm0, Mu=Mu,
                                            alpha=self.alpha, beta=self.beta)
        if self.alpha != 1.0:
            # ZHC NOTE alpha is adjusted to 1 after converge mf
            log.info("adjust mf.alpha to 1.0 for CC.")
            self.scfsolver.mf.alpha = 1.0

        log.debug(1, "CC solver: mean-field converged: %s", 
                  self.scfsolver.mf.converged)

        if "mo_energy_custom" in kwargs:
            self.scfsolver.mf.mo_energy = kwargs["mo_energy_custom"]
        if "mo_occ_custom" in kwargs:
            self.scfsolver.mf.mo_occ = kwargs["mo_occ_custom"]
        if "mo_coeff_custom" in kwargs:
            log.info("Use customized MO as reference.")
            self.scfsolver.mf.mo_coeff = kwargs["mo_coeff_custom"]
            self.scfsolver.mf.e_tot = self.scfsolver.mf.energy_tot()
        
        log.debug(2, "CC solver: mean-field rdm1: \n%s", 
                  self.scfsolver.mf.make_rdm1())
        if remove_h2:
            Ham.H2["ccdd"] = None
        
        # *********************************************************************
        # 3. CC kernel
        # *********************************************************************
        if kwargs.get("ccd", False) and self.linear == False and self.ring == False:
            log.info("Using CCD as CC solver.")
            if self.ghf:
                if self.use_mpi:
                    from mpi4pyscf import cc as mpicc
                    self.cisolver = mpicc.gccd.GGCCD(self.scfsolver.mf)
                else:
                    self.cisolver = GGCCD(self.scfsolver.mf)
            elif Ham.restricted:
                self.cisolver = CCD(self.scfsolver.mf)
            else:
                self.cisolver = UICCD(self.scfsolver.mf)
        else:
            if self.ghf:
                if self.linear:
                    if kwargs.get("ccd", True):
                        log.info("Using linearized CCD solver.")
                        self.cisolver = LGGCCD(self.scfsolver.mf)
                    else:
                        log.info("Using linearized CC solver.")
                        self.cisolver = LGGCCSD(self.scfsolver.mf)
                elif self.tcc:
                    if self.use_mpi:
                        from mpi4pyscf.cc import GGTCCSD
                    else:
                        from libdmet.solver.gtccsd import GGTCCSD
                    log.info("Using tailored CC solver.")
                    assert self.approx_l
                    ncas = kwargs.get("ncas")
                    nelecas = kwargs.get("nelecas")
                    self.cisolver = GGTCCSD(self.scfsolver.mf, ncas=ncas,
                                            nelecas=nelecas)
                elif self.ring:
                    if kwargs.get("ccd", False):
                        log.info("Using ring CCSD solver.")
                        self.cisolver = RINGGCCD(self.scfsolver.mf)
                    else:
                        if self.use_mpi:
                            if isinstance(self.ite, str):
                                log.info("Using ring MPI krylov CCSD solver.")
                                from CC_Variants import grccsd_krylov
                                self.cisolver = grccsd_krylov.GGCCSD_KRYLOV(self.scfsolver.mf, 
                                                                    method=self.ite,
                                                                    precond=kwargs.get("precond", 'finv'),
                                                                    inner_m=kwargs.get("inner_m", 10),
                                                                    outer_k=kwargs.get("outer_k", 6),
                                                                    frozen_abab=kwargs.get("frozen_abab", False),
                                                                    frozen_aaaa_bbbb=kwargs.get("frozen_aaaa_bbbb", False),
                                                                    t1_frozen_list=kwargs.get("t1_frozen_list", None),
                                                                    t2_frozen_list=kwargs.get("t2_frozen_list", None),
                                                                    t1_fix_list=kwargs.get("t1_fix_list", None),
                                                                    t2_fix_list=kwargs.get("t2_fix_list", None),
                                                                    nocc_a=kwargs.get("nocc_a", None),
                                                                    nvir_a=kwargs.get("nvir_a", None))
                            else:
                                from CC_Variants import grccsd
                                log.info("Using ring MPI CCSD solver.")
                                self.cisolver = grccsd.GGRCCSD(self.scfsolver.mf,
                                                                remove_h2=True)
                        else:
                            log.info("Using ring CCSD solver.")
                            self.cisolver = RINGGCCSD(self.scfsolver.mf)
                elif self.ladder:
                    if self.use_mpi:
                        if isinstance(self.ite, str):
                            log.info("Using ladder MPI krylov CCSD solver.")
                            from CC_Variants import glccsd_krylov
                            self.cisolver = glccsd_krylov.GGCCSD_KRYLOV(self.scfsolver.mf, 
                                                                method=self.ite,
                                                                precond=kwargs.get("precond", 'finv'),
                                                                inner_m=kwargs.get("inner_m", 10),
                                                                outer_k=kwargs.get("outer_k", 6),
                                                                frozen_abab=kwargs.get("frozen_abab", False),
                                                                frozen_aaaa_bbbb=kwargs.get("frozen_aaaa_bbbb", False),
                                                                t1_frozen_list=kwargs.get("t1_frozen_list", None),
                                                                t2_frozen_list=kwargs.get("t2_frozen_list", None),
                                                                t1_fix_list=kwargs.get("t1_fix_list", None),
                                                                t2_fix_list=kwargs.get("t2_fix_list", None),
                                                                nocc_a=kwargs.get("nocc_a", None),
                                                                nvir_a=kwargs.get("nvir_a", None))
                        else:
                            from CC_Variants import glccsd
                            log.info("Using ladder MPI CCSD solver.")
                            self.cisolver = glccsd.GGLCCSD(self.scfsolver.mf,
                                                                remove_h2=True)
                    else:
                        log.info("Using ladder CCSD solver.")
                        self.cisolver = LADGGCCSD(self.scfsolver.mf)
                elif self.directring:
                    if self.use_mpi:                        
                        if isinstance(self.ite, str):
                            from CC_Variants import gdrccsd_krylov
                            log.info("Using direct ring MPI krylov CCSD solver.")
                            self.cisolver = gdrccsd_krylov.GGCCSD_KRYLOV(self.scfsolver.mf, 
                                                                method=self.ite,
                                                                precond=kwargs.get("precond", 'finv'),
                                                                inner_m=kwargs.get("inner_m", 10),
                                                                outer_k=kwargs.get("outer_k", 6),
                                                                frozen_abab=kwargs.get("frozen_abab", False),
                                                                frozen_aaaa_bbbb=kwargs.get("frozen_aaaa_bbbb", False),
                                                                t1_frozen_list=kwargs.get("t1_frozen_list", None),
                                                                t2_frozen_list=kwargs.get("t2_frozen_list", None),
                                                                t1_fix_list=kwargs.get("t1_fix_list", None),
                                                                t2_fix_list=kwargs.get("t2_fix_list", None),
                                                                nocc_a=kwargs.get("nocc_a", None),
                                                                nvir_a=kwargs.get("nvir_a", None))
                        else:
                            from CC_Variants import gdrccsd
                            log.info("Using direct ring MPI CCSD solver.")
                            self.cisolver = gdrccsd.GGDRCCSD(self.scfsolver.mf,
                                                                remove_h2=True)
                    else:
                        log.info("Using direct ring CCSD solver.")
                        self.cisolver = DRINGGCCSD(self.scfsolver.mf)
                elif self.directring_diag:
                    if isinstance(self.ite, str):
                        log.info("Using direct ring diagonalization CCSD solver with KRYLOV.")
                        self.cisolver = DRINGGCCSD_DIAG_KRYLOV(self.scfsolver.mf)
                    else:
                        log.info("Using direct ring diagonalization CCSD solver.")
                        self.cisolver = DRINGGCCSD_DIAG(self.scfsolver.mf)
                elif self.directringdoubles:
                    log.info("Using direct ring CCD solver.")
                    self.cisolver = DRINGGCCD(self.scfsolver.mf)
                elif self.directladder:
                    if self.use_mpi:
                        if isinstance(self.ite, str):
                            from CC_Variants import gdlccsd_krylov
                            log.info("Using direct ladder MPI krylov CCSD solver.")
                            self.cisolver = gdlccsd_krylov.GGCCSD_KRYLOV(self.scfsolver.mf, 
                                                                method=self.ite,
                                                                precond=kwargs.get("precond", 'finv'),
                                                                inner_m=kwargs.get("inner_m", 10),
                                                                outer_k=kwargs.get("outer_k", 6),
                                                                frozen_abab=kwargs.get("frozen_abab", False),
                                                                frozen_aaaa_bbbb=kwargs.get("frozen_aaaa_bbbb", False),
                                                                t1_frozen_list=kwargs.get("t1_frozen_list", None),
                                                                t2_frozen_list=kwargs.get("t2_frozen_list", None),
                                                                t1_fix_list=kwargs.get("t1_fix_list", None),
                                                                t2_fix_list=kwargs.get("t2_fix_list", None),
                                                                nocc_a=kwargs.get("nocc_a", None),
                                                                nvir_a=kwargs.get("nvir_a", None))
                        else:
                            from CC_Variants import gdlccsd
                            log.info("Using direct ladder MPI CCSD solver.")
                            self.cisolver = gdlccsd.GGDLCCSD(self.scfsolver.mf,
                                                                remove_h2=True)
                    else:
                        log.info("Using direct ladder CCSD solver.")
                        self.cisolver = DLADGGCCSD(self.scfsolver.mf)
                else:
                    if self.use_mpi:
                        from mpi4pyscf import cc as mpicc
                        if isinstance(self.ite, str):
                            self.cisolver = mpicc.GGCCSD_KRYLOV(self.scfsolver.mf, 
                                                                method=self.ite,
                                                                precond=kwargs.get("precond", 'finv'),
                                                                inner_m=kwargs.get("inner_m", 10),
                                                                outer_k=kwargs.get("outer_k", 6),
                                                                frozen_abab=kwargs.get("frozen_abab", False),
                                                                frozen_aaaa_bbbb=kwargs.get("frozen_aaaa_bbbb", False),
                                                                t1_frozen_list=kwargs.get("t1_frozen_list", None),
                                                                t2_frozen_list=kwargs.get("t2_frozen_list", None),
                                                                t1_fix_list=kwargs.get("t1_fix_list", None),
                                                                t2_fix_list=kwargs.get("t2_fix_list", None),
                                                                nocc_a=kwargs.get("nocc_a", None),
                                                                nvir_a=kwargs.get("nvir_a", None))
                        elif self.ite:
                            self.cisolver = mpicc.gccsd.GGCCSD(self.scfsolver.mf,
                                                               remove_h2=True,
                                                               dt=self.ite)
                        elif self.ite_rk:
                            self.cisolver = mpicc.gccsd.GGCCSDITE_RK(self.scfsolver.mf,
                                                                     remove_h2=True,
                                                                     dt=self.ite,
                                                                     rk_order=kwargs.get("rk_order", 4))
                        else:
                            self.cisolver = mpicc.gccsd.GGCCSD(self.scfsolver.mf,
                                                               remove_h2=True)
                    else:
                        if isinstance(self.ite, str):
                            self.cisolver = GGCCSD_KRYLOV(self.scfsolver.mf,
                                                          method=self.ite,
                                                          precond=kwargs.get("precond", 'finv'),
                                                          inner_m=kwargs.get("inner_m", 10),
                                                          outer_k=kwargs.get("outer_k", 6),
                                                          frozen_abab = kwargs.get("frozen_abab", False),
                                                          nocc_a = kwargs.get("nocc_a", None),
                                                          nvir_a = kwargs.get("nvir_a", None))
                        elif self.ite:
                            self.cisolver = GGCCSDITE(self.scfsolver.mf, dt=self.ite)
                        elif self.ite_rk:
                            self.cisolver = GGCCSDITE_RK(self.scfsolver.mf, dt=self.ite)
                        else:
                            self.cisolver = GGCCSD(self.scfsolver.mf)
            elif Ham.restricted:
                self.cisolver = cc.CCSD(self.scfsolver.mf)
            else:
                if self.ite:
                    log.info("Using imaginary time evolution for CC.")
                    self.cisolver = UICCSDITE(self.scfsolver.mf, dt=self.ite)
                else:
                    self.cisolver = UICCSD(self.scfsolver.mf)
        
        self.cisolver.set(frozen=self.frozen)
        
        if self.beta < np.inf:
            # ZHC NOTE overwrite cc.nocc for smearing cases,
            # this allows that eris.fock comes from a smeared dm.
            log.info("adjust cisolver.nocc to integer for CC.")
            frozen = self.cisolver.frozen
            if self.ghf:
                if frozen is None:
                    nocc = nelec
                elif isinstance(frozen, (int, np.integer)):
                    nocc = nelec - frozen
                elif isinstance(frozen[0], (int, np.integer)):
                    occ_idx = np.zeros(self.cisolver.mo_occ.shape, dtype=bool)
                    occ_idx[:nelec] = True
                    occ_idx[list(frozen)] = False
                    nocc = np.count_nonzero(occ_idx)
                else:
                    raise ValueError
            elif Ham.restricted:
                if frozen is None:
                    nocc = nelec // 2
                elif isinstance(frozen, (int, np.integer)):
                    nocc = nelec // 2 - frozen
                elif isinstance(frozen[0], (int, np.integer)):
                    occ_idx = np.zeros(self.cisolver.mo_occ.shape, dtype=bool)
                    occ_idx[:(nelec // 2)] = True
                    occ_idx[list(frozen)] = False
                    nocc = np.count_nonzero(occ_idx)
                else:
                    raise ValueError
            else:
                if frozen is None:
                    nocc = (nelec_a, nelec_b)
                elif isinstance(frozen, (int, np.integer)):
                    nocc = (nelec_a - frozen, nelec_b - frozen)
                elif isinstance(frozen[0], (int, np.integer)):
                    if len(frozen) > 0 and isinstance(frozen[0], (int, np.integer)):
                        # The same frozen orbital indices for alpha and beta orbitals
                        frozen = [frozen, frozen]
                    occ_idx_a = np.zeros(self.cisolver.mo_occ[0].shape, dtype=bool)
                    occ_idx_a[:nelec_a] = True
                    occ_idx_a[list(frozen[0])] = False
                    nocc_a = np.count_nonzero(occ_idx_a)
                    
                    occ_idx_b = np.zeros(self.cisolver.mo_occ[1].shape, dtype=bool)
                    occ_idx_b[:nelec_b] = True
                    occ_idx_b[list(frozen[1])] = False
                    nocc_b = np.count_nonzero(occ_idx_b)

                    nocc = (nocc_a, nocc_b)
                else:
                    raise ValueError
            
            # ZHC NOTE for TCC with smearing, we need to rewrite nocc
            if self.tcc:
                self.cisolver = GGTCCSD(self.scfsolver.mf, ncas=ncas,
                                        nelecas=nelecas, nocc=nocc)
                self.cisolver.set(frozen=self.frozen)
            self.cisolver.nocc = nocc
        
        # ZHC NOTE if dm is customized then this dm will be used to compute fock in CC.
        if "dm_custom" in kwargs:
            self.cisolver.dm = kwargs["dm_custom"]

        self.cisolver.max_cycle = self.max_cycle
        self.cisolver.conv_tol = self.conv_tol
        self.cisolver.conv_tol_normt = self.conv_tol_normt
        self.cisolver.level_shift = self.level_shift
        if self.diis_space <= 0:
            self.cisolver.diis = False
        self.cisolver.diis_space = self.diis_space
        if self.diis_start_cycle is not None:
            self.cisolver.diis_start_cycle = self.diis_start_cycle
        self.cisolver.iterative_damping = self.iterative_damping
        self.cisolver.verbose = self.verbose

        # *********************************************************************
        # 4. solve t1, t2, restart can use the previously saved t1 and t2 
        # *********************************************************************
        if (not self.use_mpi):
            eris = self.cisolver.ao2mo(self.cisolver.mo_coeff)
        
        if restart:
            log.eassert("basis" in kwargs, "restart requires basis passed in")
        if restart and self.optimized:
            t1, t2, l1, l2 = self.load_t12_from_h5(fcc_name, kwargs["basis"], 
                                                   self.scfsolver.mf.mo_coeff, 
                                                   bcc_restart=bcc_restart)
        else:
            if isinstance(guess, str) and guess == 'cisd':
                if self.ghf:
                    myci = GGCISD(self.scfsolver.mf)
                elif Ham.restricted:
                    raise NotImplementedError
                else:
                    raise NotImplementedError
                myci.conv_tol = self.conv_tol
                ecisd, civec = myci.kernel(eris=eris)
                _, t1, t2 = myci.cisdvec_to_amplitudes(civec, myci.nmo, myci.nocc)
                l1 = l2 = None
            elif guess is not None:
                if len(guess) == 2:
                    t1, t2 = guess
                    l1, l2 = None, None
                else:
                    t1, t2, l1, l2 = guess
            else:
                t1, t2, l1, l2 = None, None, None, None

        # solve t1, t2
        log.debug(1, "CC solver: solve t amplitudes")
        if self.use_mpi:
            E_corr, t1, t2 = self.cisolver.kernel()
        else:
            E_corr, t1, t2 = self.cisolver.kernel(t1=t1, t2=t2, eris=eris)
        
        # brueckner CC
        if bcc:
            assert not self.use_mpi
            log.info("Using Brueckner CC.")
            verbose_old = self.cisolver.verbose
            mo_coeff_old = self.cisolver.mo_coeff
            self.cisolver = bcc_loop(self.cisolver, utol=bcc_tol,
                                     verbose=bcc_verbose)
            self.cisolver.verbose = verbose_old
            self.scfsolver.mf.verbose = verbose_old
            self.scfsolver.mf.mo_coeff = self.cisolver.mo_coeff
            self.scfsolver.mf.e_tot = self.cisolver._scf.e_tot
            t1, t2 = self.cisolver.t1, self.cisolver.t2
            eris = self.cisolver.ao2mo(self.cisolver.mo_coeff)
            
            if l1 is not None and l2 is not None:
                # transform l1, l2 to the new BCC basis
                log.info("transform old l1, l2 to new BCC basis...")
                if isinstance(mo_coeff_old, np.ndarray) and mo_coeff_old.ndim == 2:
                    umat = get_mo_ovlp(mo_coeff_old, self.cisolver.mo_coeff,
                                       self.scfsolver.mf.get_ovlp())
                    l1 = transform_l1_to_bo(l1, umat)
                    l2 = transform_l2_to_bo(l2, umat)
                else: # UHF
                    umat = []
                    for s in range(mo_coeff.shape[0]):
                        umat.append(get_mo_ovlp(mo_coeff_old[s],
                                                self.cisolver.mo_coeff[s],
                                                self.scfsolver.mf.get_ovlp()))
                    l1 = transform_l1_to_bo(l1, umat)
                    l2 = transform_l2_to_bo(l2, umat)
        
        # *********************************************************************
        # 5. (T) and solve lambda
        # *********************************************************************
        log.debug(1, "CC solver: solve l amplitudes")
        if kwargs.get("ccsdt", False):
            assert not self.use_mpi
            log.info("CCSD(T) correction")
            # CCSD(T)
            # ZHC NOTE the CCSD(T) lambda does not support fov != 0
            # thus the correctness of BCCD(T) needs to be checked!
            if kwargs.get("ccsdt_energy", False):
                e_t = self.cisolver.ccsd_t(eris=eris)
                E_corr += e_t
                log.info("CCSD(T) E(T): %20.12f", e_t)
            else:
                e_t = 0.0

            lambda_drv, rdm1_drv, rdm2_drv = self._get_ccsdt_drv(eris=eris)
            l1, l2 = self.cisolver.l1, self.cisolver.l2 = \
                    lambda_drv(self.cisolver, eris=eris, t1=t1, t2=t2, 
                               max_cycle=self.max_cycle, 
                               tol=self.conv_tol_normt, 
                               verbose=self.verbose)[1:]
        else:
            if self.use_mpi:
                l1, l2 = self.cisolver.solve_lambda(approx_l=self.approx_l)
            else:
                if self.tcc and self.approx_l:
                    ncore = self.cisolver.ncore
                    nvir_cas = self.cisolver.nvir_cas

                    l1 = np.array(self.cisolver.t1, copy=True)
                    l1[ncore:, :nvir_cas] = 0.0
                    self.cisolver.l1 = l1

                    l2 = np.array(self.cisolver.t2, copy=True)
                    l2[ncore:, ncore:, :nvir_cas, :nvir_cas] = 0.0
                    self.cisolver.l2 = l2
                elif self.linear or self.approx_l:
                    l1 = self.cisolver.l1 = self.cisolver.t1
                    l2 = self.cisolver.l2 = self.cisolver.t2
                else:
                    l1, l2 = self.cisolver.solve_lambda(t1=t1, t2=t2, l1=l1, l2=l2,
                                                        eris=eris)
            rdm1_drv = rdm2_drv = None
            e_t = 0.0
        
        # *********************************************************************
        # 6. collect properties
        # *********************************************************************
        # energy
        E = self.cisolver.e_tot + e_t
        
        if self.beta < np.inf:
            # ZHC NOTE modify the mo_occ since the frozen may need it in rdm
            log.info("adjust mf.mo_occ to integer for CC.")
            if self.ghf:
                self.cisolver.mo_occ = np.zeros_like(self.cisolver.mo_occ)
                self.cisolver.mo_occ[:nelec] = 1.0
            elif Ham.restricted:
                self.cisolver.mo_occ = np.zeros_like(self.cisolver.mo_occ)
                self.cisolver.mo_occ[:nelec//2] = 2.0
            else:
                self.cisolver.mo_occ = np.zeros_like(self.cisolver.mo_occ)
                self.cisolver.mo_occ[0, :nelec_a] = 1.0
                self.cisolver.mo_occ[1, :nelec_b] = 1.0

        # rdm1 and rdm2
        self.make_rdm1(Ham, drv=rdm1_drv)
        if calc_rdm2:
            self.make_rdm2(Ham, drv=rdm2_drv)

        # dump t1, t2, l1, l2, basis, mo_coeff
        if dump_tl or restart:
            if fcc_name_save is None:
                fcc_name_save = fcc_name
            self.save_t12_to_h5(fcc_name_save, kwargs["basis"], self.cisolver.mo_coeff)
        
        if not self.cisolver.converged:
            log.warn("CC solver not converged...")
        self.optimized = True
        
        if kwargs.get("analyze_amps", False):
            self.cisolver.analyze_amps()

        # ZHC NOTE cleanup for MPICC
        if self.use_mpi:
            self.scfsolver.mf = None
            self.cisolver._release_regs()

        return self.onepdm, E
    
    def _get_ccsdt_drv(self, eris=None):
        if self.ghf:
            from pyscf.cc.gccsd_t_lambda import kernel as lambda_drv
            from pyscf.cc import gccsd_t_rdm as rdm_mod
        elif self.restricted:
            from pyscf.cc.ccsd_t_lambda_slow import kernel as lambda_drv
            from pyscf.cc import ccsd_t_rdm_slow as rdm_mod
        else:
            from pyscf.cc.uccsd_t_lambda import kernel as lambda_drv
            from pyscf.cc import uccsd_t_rdm as rdm_mod
        rdm1_drv = partial(rdm_mod.make_rdm1, mycc=self.cisolver, eris=eris)
        rdm2_drv = partial(rdm_mod.make_rdm2, mycc=self.cisolver, eris=eris)
        return lambda_drv, rdm1_drv, rdm2_drv 
    
    def run_dmet_ham(self, Ham, last_aabb=True, save_dmet_ham=False, 
                     dmet_ham_fname='dmet_ham.h5', use_calculated_twopdm=False,
                     **kwargs):
        """
        Run scaled DMET Hamiltonian.
        NOTE: the spin order for H2 is aa, bb, ab, the same as ImpHam.
        """
        if kwargs.get("ccsdt", False) or use_calculated_twopdm:
            return self.run_dmet_ham_slow(Ham, last_aabb=last_aabb, 
                                          save_dmet_ham=save_dmet_ham,
                                          dmet_ham_fname=dmet_ham_fname, 
                                          use_calculated_twopdm=use_calculated_twopdm,
                                          **kwargs)
        
        log.info("CC solver Run DMET Hamiltonian.")
        log.debug(0, "ao2mo for DMET Hamiltonian.")
        Ham = ao2mo_Ham(Ham, self.cisolver.mo_coeff, compact=True, in_place=True)

        if self.ghf:
            H1 = Ham.H1["cd"][0]
            H2 = Ham.H2["ccdd"][0]
            H0 = Ham.H0
            E = exp_val_gccsd(self.cisolver, H1, H2, H0=H0)
        elif Ham.restricted:
            H1 = Ham.H1["cd"][0]
            H2 = Ham.H2["ccdd"][0]
            H0 = Ham.H0
            E = exp_val_rccsd(self.cisolver, H1, H2, H0=H0)
        else:
            # H2 is in aa, bb, ab order
            H1 = Ham.H1["cd"]
            H2 = Ham.H2["ccdd"]
            H0 = Ham.H0
            E = exp_val_uccsd(self.cisolver, H1, H2, H0=H0)
                  
        if save_dmet_ham:
            fdmet_ham = h5py.File(dmet_ham_fname, 'w')
            fdmet_ham['H0'] = Ham.H0
            fdmet_ham['H1'] = H1
            fdmet_ham['H2'] = H2
            fdmet_ham['mo_coeff'] = self.cisolver.mo_coeff
            fdmet_ham.close()
        return E
    
    def run_dmet_ham_slow(self, Ham, last_aabb=True, save_dmet_ham=False, 
                          dmet_ham_fname='dmet_ham.h5',
                          use_calculated_twopdm=False, **kwargs):
        """
        Run scaled DMET Hamiltonian.
        NOTE: the spin order for H2 is aa, bb, ab, the same as ImpHam.
        """
        log.info("CC solver Run DMET Hamiltonian.")
        log.debug(0, "ao2mo for DMET Hamiltonian.")
        Ham = ao2mo_Ham(Ham, self.cisolver.mo_coeff, compact=True,
                        in_place=True)
        Ham = restore_Ham(Ham, 1, in_place=True)
        # calculate rdm2 in aa, bb, ab order
        if use_calculated_twopdm:
            log.info("Using exisiting twopdm in MO basis...")
            assert self.twopdm_mo is not None
        else:
            if kwargs.get("ccsdt", False):
                eris = self.cisolver.ao2mo(self.cisolver.mo_coeff)
                rdm2_drv = self._get_ccsdt_drv(eris=eris)[-1]
            else:
                rdm2_drv = None
            self.make_rdm2(drv=rdm2_drv)

        if self.ghf:
            h1 = Ham.H1["cd"][0]
            h2 = Ham.H2["ccdd"][0]
            r1 = self.onepdm
            r2 = self.twopdm
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape
            
            E1 = einsum('ij, ji', h1, r1)
            E2 = 0.5 * einsum('ijkl, ijkl', h2, r2)
        elif Ham.restricted:
            h1 = Ham.H1["cd"]
            h2 = Ham.H2["ccdd"]
            r1 = self.onepdm_mo
            r2 = self.twopdm_mo
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape
            
            E1 = 2.0 * einsum('ij, ji', h1[0], r1[0])
            E2 = 0.5 * einsum('ijkl, ijkl', h2[0], r2[0])
        else:
            h1 = Ham.H1["cd"]
            # h2 is in aa, bb, ab order
            h2 = Ham.H2["ccdd"]
            r1 = self.onepdm_mo
            # r2 is in aa, bb, ab order
            r2 = self.twopdm_mo
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape
            
            # energy
            E1 = einsum('sij, sji', h1, r1)
            E2_aa = 0.5 * einsum('ijkl, ijkl', h2[0], r2[0])
            E2_bb = 0.5 * einsum('ijkl, ijkl', h2[1], r2[1])
            E2_ab =       einsum('ijkl, ijkl', h2[2], r2[2])
            E2 = E2_aa + E2_bb + E2_ab
        
        E = E1 + E2
        E += Ham.H0
        log.debug(0, "run DMET Hamiltonian:\nE0 = %20.12f, E1 = %20.12f, " 
                "E2 = %20.12f, E = %20.12f", Ham.H0, E1, E2, E)
        
        if save_dmet_ham:
            fdmet_ham = h5py.File(dmet_ham_fname, 'w')
            fdmet_ham['H0'] = Ham.H0
            fdmet_ham['H1'] = h1
            fdmet_ham['H2'] = h2
            fdmet_ham['mo_coeff'] = self.cisolver.mo_coeff
            fdmet_ham.close()
        return E
    
    def make_rdm1(self, Ham=None, drv=None):
        log.debug(1, "CC solver: solve rdm1")
        if drv is None:
            drv = self.cisolver.make_rdm1
        if self.use_mpi:
            onepdm = drv()
        else:
            onepdm = drv(t1=self.cisolver.t1, t2=self.cisolver.t2, 
                         l1=self.cisolver.l1, l2=self.cisolver.l2)

        if self.ghf: # GHF
            self.onepdm_mo = np.asarray(onepdm)
        elif self.restricted:
            self.onepdm_mo = np.asarray(onepdm)[np.newaxis] * 0.5
        else:
            self.onepdm_mo = np.asarray(onepdm)

        # rotate back to the AO basis
        log.debug(1, "CC solver: rotate rdm1 to AO")
        self.onepdm = transform_rdm1_to_ao_mol(self.onepdm_mo, 
                                               self.cisolver.mo_coeff)
        return self.onepdm

    def make_rdm2(self, Ham=None, ao_repr=False, drv=None, with_dm1=True):
        """
        Compute rdm2.
        NOTE: the returned value's spin order for rdm2 is aa, bb, ab.
        """
        log.debug(1, "CC solver: solve rdm2")
        if drv is None:
            drv = self.cisolver.make_rdm2
        if self.use_mpi:
            if with_dm1:
                twopdm_mo = drv()
            else:
                twopdm_mo = drv(with_dm1=with_dm1)
        else:
            if with_dm1:
                twopdm_mo = drv(t1=self.cisolver.t1, t2=self.cisolver.t2, 
                                l1=self.cisolver.l1, l2=self.cisolver.l2)
            else:
                twopdm_mo = drv(t1=self.cisolver.t1, t2=self.cisolver.t2, 
                                l1=self.cisolver.l1, l2=self.cisolver.l2,
                                with_dm1=with_dm1)

        if self.ghf: # GHF
            self.twopdm_mo = twopdm_mo
        elif self.restricted:
            self.twopdm_mo = twopdm_mo[np.newaxis]
        else:
            # NOTE: here is aa, ab, bb order
            self.twopdm_mo = np.asarray(twopdm_mo)
            twopdm_mo = None

        # rotate back to the AO basis
        # NOTE: the transform function use aa, ab, bb order.
        if ao_repr:
            log.debug(1, "CC solver: rotate rdm2 to AO")
            self.twopdm = transform_rdm2_to_ao_mol(self.twopdm_mo, 
                                                   self.cisolver.mo_coeff)
            self.twopdm_mo = None
        else:
            self.twopdm = None
            
        if not self.restricted and not self.ghf:
            if self.twopdm_mo is not None:
                self.twopdm_mo = self.twopdm_mo[[0, 2, 1]]
            if self.twopdm is not None:
                self.twopdm = self.twopdm[[0, 2, 1]]
        return self.twopdm

    def load_t12_from_h5(self, fcc_name, basis_new, mo_coeff_new,
                         bcc_restart=False, ovlp_tol=None):
        """
        Load t1, t2, l1, l2 and rotate to current basis.
        """
        log.debug(1, "CC solver: read previous t and basis")
        if not os.path.isfile(fcc_name):
            log.info("CC solver: read previous t and basis failed, "
                     "file %s does not exist.", fcc_name)
            return None, None, None, None
        if ovlp_tol is None:
            ovlp_tol = self.ovlp_tol
        fcc = h5py.File(fcc_name, 'r')
        basis_old    = np.asarray(fcc['basis'])
        mo_coeff_old = np.asarray(fcc['mo_coeff'])

        if not self.use_mpi:
            if mo_coeff_old.ndim == 2:
                t1_old = np.asarray(fcc['t1'])
                t2_old = np.asarray(fcc['t2'])
                if not (self.linear or self.approx_l):
                    l1_old = np.asarray(fcc['l1'])
                    l2_old = np.asarray(fcc['l2'])
            else:
                spin = mo_coeff_old.shape[0]
                t1_old = [np.asarray(fcc['t1_%s'%s]) for s in range(spin)]
                t2_old = [np.asarray(fcc['t2_%s'%s]) for s in range(spin*(spin+1)//2)]
                if not (self.linear or self.approx_l):
                    l1_old = [np.asarray(fcc['l1_%s'%s]) for s in range(spin)]
                    l2_old = [np.asarray(fcc['l2_%s'%s]) for s in range(spin*(spin+1)//2)]
        fcc.close()
        
        mo_coeff_new = np.asarray(mo_coeff_new)
        if mo_coeff_new.shape != mo_coeff_old.shape:
            log.warn("CC solver: mo_coeff shape changed (%s -> %s).", 
                     mo_coeff_old.shape, mo_coeff_new.shape)
            return None, None, None, None
        nao, nmo = mo_coeff_new.shape[-2:]
        # frozen
        mo_idx = self.cisolver.get_frozen_mask()
        is_same_basis = (basis_new is basis_old) or \
                        (max_abs(basis_new - basis_old) < 1e-12)
        if is_same_basis:
            log.debug(2, "restart with the same basis.")
        else:
            log.debug(2, "restart with the different basis.")
        
        try:
            if bcc_restart:
                # ZHC NOTE
                # restart for a bcc calculation, 
                # new mo is estimated by maximizing the overlap 
                # w.r.t. last calculation
                # self.scfsolver.mf.mo_coeff and
                # self.cisolver.mo_coeff are overwritten
                # t1, t2, l1, l2 are unchanged!

                if mo_coeff_new.ndim == 2: # RHF and GHF
                    basis = basis_new.reshape(-1, nmo)
                    basis_cas_old = basis_old.reshape(-1, nmo).dot(mo_coeff_old[:, mo_idx])
                    mo_coeff_new = find_closest_mo(basis, basis_cas_old,
                                                   return_rotmat=True)[1]
                else: # UHF
                    mo_coeff_new = []
                    for s in range(2):
                        basis = basis_new[s].reshape(-1, nmo)
                        basis_cas_old = basis_old[s].reshape(-1, nmo).dot(mo_coeff_old[s][:, mo_idx[s]])
                        mo_coeff_new.append(find_closest_mo(basis, basis_cas_old,
                                                            return_rotmat=True)[1])
                self.scfsolver.mf.mo_coeff = mo_coeff_new
                self.cisolver.mo_coeff = mo_coeff_new
                t1 = t1_old
                t2 = t2_old
                if not (self.linear or self.approx_l):
                    l1 = l1_old
                    l2 = l2_old

            else: # normal CCSD restart
                if mo_coeff_new.ndim == 2: # RHF and GHF 
                    # umat maximally match the basis, C_old U ~ C_new
                    basis_cas_old = basis_old.reshape(-1, nmo).dot(mo_coeff_old[:, mo_idx])
                    basis_cas_new = basis_new.reshape(-1, nmo).dot(mo_coeff_new[:, mo_idx])
                    mo_ovlp = abs(la.det(np.dot(basis_cas_old.conj().T, basis_cas_new)))
                    umat = find_closest_mo(basis_cas_old, basis_cas_new, return_rotmat=True)[1]
                else: # UHF 
                    umat = []
                    mo_ovlp = 1.0 
                    for s in range(2):
                        basis_cas_old = np.dot(basis_old[s].reshape(-1, nmo), 
                                               mo_coeff_old[s][:, mo_idx[s]])
                        basis_cas_new = np.dot(basis_new[s].reshape(-1, nmo), 
                                               mo_coeff_new[s][:, mo_idx[s]])
                        mo_ovlp = min(mo_ovlp, abs(la.det(np.dot(basis_cas_old.conj().T, basis_cas_new))))
                        umat.append(find_closest_mo(basis_cas_old, basis_cas_new, 
                                                    return_rotmat=True)[1])
                
                log.debug(1, "CC solver: restart ovlp: %.8g", mo_ovlp)
                if mo_ovlp < ovlp_tol:
                    log.debug(1, "CC solver: restart ovlp smaller than ovlp_tol (%.8g), "
                              "not use restart", ovlp_tol)
                    t1 = t2 = l1 = l2 = None
                else:
                    if self.use_mpi:
                        t1 = t2 = l1 = l2 = None
                        self.cisolver.restore_from_h5(fname=fcc_name, umat=umat)
                    else:
                        t1 = transform_t1_to_bo(t1_old, umat)
                        t2 = transform_t2_to_bo(t2_old, umat)
                        if not (self.linear or self.approx_l):
                            l1 = transform_l1_to_bo(l1_old, umat)
                            l2 = transform_l2_to_bo(l2_old, umat)
        except np.linalg.LinAlgError:
            log.warn("SVD error catched during matching basis...")
            t1 = t2 = l1 = l2 = None
        
        if (self.linear or self.approx_l):
            l1 = t1
            l2 = t2
        return t1, t2, l1, l2
    
    def save_t12_to_h5(self, fcc_name, basis_new, mo_coeff_new,
                       mo_occ=None, mo_energy=None):
        """
        Save t1, t2, l1, l2, basis and mo_coeff.
        """
        log.debug(1, "CC solver: dump t and l")
        mo_coeff_new = np.asarray(mo_coeff_new)
        fcc = h5py.File(fcc_name, 'w')
        fcc['mo_coeff'] = mo_coeff_new
        fcc['basis'] = np.asarray(basis_new)
        if mo_occ is None:
            mo_occ = self.scfsolver.mf.mo_occ
        fcc['mo_occ'] = np.asarray(mo_occ)
        if mo_energy is None:
            mo_energy = self.scfsolver.mf.mo_energy
        fcc['mo_energy'] = np.asarray(mo_energy)
        if not self.use_mpi:
            if mo_coeff_new.ndim == 2: 
                fcc['t1'] = np.asarray(self.cisolver.t1)
                fcc['t2'] = np.asarray(self.cisolver.t2)
                if not (self.linear or self.approx_l):
                    fcc['l1'] = np.asarray(self.cisolver.l1)
                    fcc['l2'] = np.asarray(self.cisolver.l2)
            else:
                spin = mo_coeff_new.shape[0]
                for s in range(spin):
                    fcc['t1_%s'%s] = np.asarray(self.cisolver.t1[s])
                    if not (self.linear or self.approx_l):
                        fcc['l1_%s'%s] = np.asarray(self.cisolver.l1[s])
                for s in range(spin*(spin+1)//2):
                    fcc['t2_%s'%s] = np.asarray(self.cisolver.t2[s])
                    if not (self.linear or self.approx_l):
                        fcc['l2_%s'%s] = np.asarray(self.cisolver.l2[s])
        else:
            self.cisolver.save_amps(fname=fcc_name)
        fcc.close()

    def save_rdm_mo(self, rdm_fname='rdm_mo_cc.h5'):
        frdm = h5py.File(rdm_fname, 'w')
        frdm['rdm1'] = np.asarray(self.onepdm_mo)
        frdm['rdm2'] = np.asarray(self.twopdm_mo)
        frdm["mo_coeff"] = np.asarray(self.cisolver.mo_coeff)
        frdm.close()
    
    def load_rdm_mo(self, rdm_fname='rdm_mo_cc.h5'):
        frdm = h5py.File(rdm_fname, 'r')
        rdm1 = np.asarray(frdm["rdm1"])
        rdm2 = np.asarray(frdm["rdm2"])
        mo_coeff = np.asarray(frdm["mo_coeff"])
        frdm.close()
        return rdm1, rdm2, mo_coeff 
    
    def load_dmet_ham(self, dmet_ham_fname='dmet_ham.h5'):
        fdmet_ham = h5py.File(dmet_ham_fname, 'r')
        H1 = np.asarray(fdmet_ham["H1"])
        H2 = np.asarray(fdmet_ham["H2"])
        fdmet_ham.close()
        return H1, H2 
    
    def onepdm(self):
        log.debug(1, "Compute 1pdm")
        return self.onepdm

    def twopdm(self):
        log.debug(1, "Compute 2pdm")
        return self.twopdm

    def cleanup(self):
        pass

def get_umat_from_t1(t1):
    """
    Get rotation matrix, U = exp(t1 - t1^T)
    """
    if isinstance(t1, np.ndarray) and t1.ndim == 2: # RHF GHF
        nocc, nvir = t1.shape
        amat = np.zeros((nocc+nvir, nocc+nvir), dtype=t1.dtype)
        amat[:nocc, -nvir:] = -t1
        amat[-nvir:, :nocc] = t1.conj().T
        umat = la.expm(amat)
    else: # UHF
        spin = len(t1)
        nmo = np.sum(t1[0].shape)
        umat = np.zeros((spin, nmo, nmo), dtype=np.result_type(*t1))
        for s in range(spin):
            nocc, nvir = t1[s].shape
            amat = np.zeros((nmo, nmo), dtype=t1[s].dtype)
            amat[:nocc, -nvir:] = -t1[s]
            amat[-nvir:, :nocc] = t1[s].conj().T
            umat[s] = la.expm(amat)
    return umat

def transform_t1_to_bo(t1, umat):
    """
    Transform t1 to brueckner orbital basis.
    """
    if isinstance(t1, np.ndarray) and t1.ndim == 2: # RHF GHF
        nocc, nvir = t1.shape
        umat_occ = umat[:nocc, :nocc]
        umat_vir = umat[nocc:, nocc:] 
        return mdot(umat_occ.conj().T, t1, umat_vir)
    else: # UHF
        spin = len(t1)
        return [transform_t1_to_bo(t1[s], umat[s]) for s in range(spin)]

def transform_t2_to_bo(t2, umat, umat_b=None):
    """
    Transform t2 to brueckner orbital basis.
    """
    if isinstance(t2, np.ndarray) and t2.ndim == 4: # RHF GHF
        umat_a = umat
        if umat_b is None:
            umat_b = umat_a

        nocc_a, nocc_b, nvir_a, nvir_b = t2.shape
        umat_occ_a = umat_a[:nocc_a, :nocc_a]
        umat_occ_b = umat_b[:nocc_b, :nocc_b]
        umat_vir_a = umat_a[nocc_a:, nocc_a:]
        umat_vir_b = umat_b[nocc_b:, nocc_b:]
        t2_bo = einsum("ijab, iI, jJ, aA, bB -> IJAB", t2, umat_occ_a,
                       umat_occ_b, umat_vir_a, umat_vir_b)
    else: # UHF
        # t2 order: aa, ab, bb
        t2_bo = [None, None, None]
        t2_bo[0] = transform_t2_to_bo(t2[0], umat[0])
        t2_bo[1] = transform_t2_to_bo(t2[1], umat[0], umat_b=umat[1])
        t2_bo[2] = transform_t2_to_bo(t2[2], umat[1])
    return t2_bo

transform_l1_to_bo = transform_t1_to_bo
transform_l2_to_bo = transform_t2_to_bo

def bcc_loop(mycc, u=None, utol=1e-5, max_cycle=10, diis=True, verbose=2):
    """
    Brueckner coupled-cluster wrapper, using an outer-loop algorithm.
    """ 
    def max_abs(x):
        if isinstance(x, np.ndarray):
            if np.iscomplexobj(x):
                return np.abs(x).max()
            else:
                return max(np.max(x), abs(np.min(x)))
        else:
            return np.max([max_abs(xi) for xi in x])

    if u is None:
        u = get_umat_from_t1(mycc.t1)
    mf = mycc._scf
    ovlp = mf.get_ovlp()
    adiis = lib.diis.DIIS()
    # ZHC NOTE need to remember these options:
    frozen = mycc.frozen
    level_shift = mycc.level_shift

    with lib.temporary_env(mf, verbose=verbose):
        e_tot_last = mycc.e_tot
        for i in range(max_cycle):
            mf.mo_coeff = trans_mo(mf.mo_coeff, u)
            if diis:
                mo_coeff_new = adiis.update(mf.mo_coeff)
                u = trans_mo(u, get_mo_ovlp(mf.mo_coeff, mo_coeff_new, ovlp))
                mf.mo_coeff = mo_coeff_new
            mf.e_tot = mf.energy_tot()
            t1 = transform_t1_to_bo(mycc.t1, u)
            t2 = transform_t2_to_bo(mycc.t2, u)
            mycc.__init__(mf)
            mycc.frozen = frozen
            mycc.level_shift = level_shift
            mycc.verbose = verbose
            log.debug(1, "BCC frozen: %s", mycc.frozen)
            log.debug(1, "BCC level_shift: %s", mycc.level_shift)
            mycc.kernel(t1=t1, t2=t2)
            dE = mycc.e_tot - e_tot_last
            e_tot_last = mycc.e_tot
            if not mycc.converged:
                log.warn("CC not converged")
            t1_norm = max_abs(mycc.t1)
            log.info("BCC iter: %4d  E: %20.12f  dE: %12.3e  |t1|: %12.3e", 
                     i, mycc.e_tot, dE, t1_norm)
            if t1_norm < utol:
                break
            u = get_umat_from_t1(mycc.t1)
        else:
            log.warn("BCC: not converged, max_cycle reached.")
    return mycc

# ****************************************************************************
# Expectation value from CCSD rdm, outcore version.
# ****************************************************************************

def exp_val_rccsd(mycc, H1, H2, H0=0.0, rdm2_tmp_fname=None, blksize=None):
    """
    Expectation value of H0, H1 and H2, using an outcore routine.
    H1 and H2 are in MO basis. 
    H2 is real and has 8-fold symmetry, 1, 4, 8-fold array are ok.

    Args:
        mycc: cc object.
        H1: (nmo, nmo)
        H2: (nmo,) * 4, or (nmo_pair, nmo_pair) or (nmo_pair_pair)
        H0: scalar
        rdm2_tmp_fname: if given, will save the rdm2 intermidiates.

    Returns:
        E_tot: the expectation value of H0, H1 and H2.
    """
    log.debug(0, "exp_val_rccsd: start")
    # frozen core
    mo_idx = mycc.get_frozen_mask()
    if not all(mo_idx):
        nocc = np.count_nonzero(mycc.mo_occ > 0)
        core_idx = np.arange(nocc)[~mo_idx[:nocc]]
        act_idx  = np.arange(H1.shape[-1])[mo_idx]
        rdm1_core = np.zeros_like(H1)
        rdm1_core[core_idx, core_idx] = 2.0
        veff_core = _get_veff(rdm1_core, H2)[0]
        E_core = einsum('ij, ji', H1 + veff_core*0.5, rdm1_core)
        
        H0 += E_core
        H1 = (H1 + veff_core)[np.ix_(act_idx, act_idx)]
        H2 = take_eri(H2, act_idx, act_idx, act_idx, act_idx, compact=True)

    f = lib.H5TmpFile(filename=rdm2_tmp_fname, mode='w')
    t1, t2, l1, l2 = mycc.t1, mycc.t2, mycc.l1, mycc.l2
    cc.ccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, f, False)
    nocc, nvir = t1.shape
    norb = nocc + nvir
    
    # E1
    d1 = cc.ccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    E1 =  (einsum('ij, ji', H1[:nocc, :nocc], doo) \
         + einsum('ij, ji', H1[nocc:, :nocc], dov) \
         + einsum('ij, ji', H1[:nocc, nocc:], dvo) \
         + einsum('ij, ji', H1[nocc:, nocc:], dvv) \
         + np.sum(H1[range(nocc), range(nocc)])) * 2
    
    # product part
    rdm1_mo = cc.ccsd_rdm._make_rdm1(mycc, d1, with_frozen=False)
    rdm1_mo[np.diag_indices(nocc)] -= 2.0
    rdm1_hf = np.zeros((norb, norb))
    rdm1_hf[range(nocc), range(nocc)] = 2.0
    veff_hf = _get_veff(rdm1_hf, H2)[0]
    E2_prod = 0.5 * einsum('ij, ji', veff_hf, rdm1_hf + rdm1_mo * 2.0)

    # cumulant part
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    unit = nvir**3 * 6
    if blksize is None:
        blksize = min(nocc, nvir, max(cc.ccsd.BLKMIN, int(max_memory*0.9e6/8/unit)))
    
    log.debug(0, "exp_val_rccsd: current memory: %s MB (max_memory: %s MB)", 
              lib.current_memory()[0], mycc.max_memory)
    log.debug(1, "unit size: %s , blksize: %s", unit, blksize)

    oidx = np.arange(nocc)
    vidx = np.arange(nocc, norb)
    E2_cum = 0.0
    
    for p0, p1 in lib.prange(0, nocc, blksize):
        cidx = oidx[p0:p1]

        # ovov
        eri_ovov = take_eri(H2, cidx, vidx, oidx, vidx)
        E_ovov = einsum('ijkl, ijkl', eri_ovov, f['dovov'][p0:p1])
        E2_cum += E_ovov
        eri_ovov = None

        # oovv
        eri_oovv = take_eri(H2, cidx, oidx, vidx, vidx)
        E_oovv = einsum('ijkl, ijkl', eri_oovv, f['doovv'][p0:p1]) 
        E2_cum += E_oovv
        eri_oovv = None
    
        # ovvo
        eri_ovvo = take_eri(H2, cidx, vidx, vidx, oidx)
        E_ovvo = einsum('ijkl, ijkl', eri_ovvo, f['dovvo'][p0:p1])
        E2_cum += E_ovvo
        eri_ovvo = None
        
        # oooo
        eri_oooo = take_eri(H2, cidx, oidx, oidx, oidx)
        E_oooo = einsum('ijkl, ijkl', eri_oooo, f['doooo'][p0:p1])
        E2_cum += E_oooo
        eri_oooo = None
    
        # ovvv
        eri_ovvv = take_eri(H2, cidx, vidx, vidx, vidx)
        E_ovvv = einsum('ijkl, ijkl', eri_ovvv, f['dovvv'][p0:p1])
        E2_cum += E_ovvv
        eri_ovvv = None
        
        # ooov
        eri_ooov = take_eri(H2, cidx, oidx, oidx, vidx)
        E_ooov = einsum('ijkl, ijkl', eri_ooov, f['dooov'][p0:p1])
        E2_cum += E_ooov
        eri_ooov = None
        
    for p0, p1 in lib.prange(0, nvir, blksize):
        cidx = vidx[p0:p1]
        # vvvv
        eri_vvvv = take_eri(H2, cidx, vidx, vidx, vidx)
        E_vvvv = einsum('ijkl, ijkl', eri_vvvv, f['dvvvv'][p0:p1])
        E2_cum += E_vvvv
        eri_vvvv = None
    
    E2_cum *= 2.0
    E2 = E2_prod + E2_cum
    
    E_tot = H0 + E1 + E2
    log.debug(0, "CC exp_val: E0: %20.12f, E1: %20.12f, E2: %20.12f \n"
              "(prod: %20.12f, cum: %20.12f), E_tot: %20.12f", 
              H0, E1, E2, E2_prod, E2_cum, E_tot)
    return E_tot

def exp_val_uccsd(mycc, H1, H2, H0=0.0, rdm2_tmp_fname=None, blksize=None):
    """
    Expectation value of H0, H1 and H2, using an outcore routine.
    H1 and H2 are in MO basis. 
    H2 is real and has 4-fold symmetry, 1, 4-fold array are ok.

    Args:
        mycc: cc object.
        H1: (2, nmo, nmo)
        H2: (3, nmo, nmo, nmo, nmo) or (3, nmo_pair, nmo_pair),
            aa, bb, ab order.
        H0: scalar
        rdm2_tmp_fname: if given, will save the rdm2 intermidiates.

    Returns:
        E_tot: the expectation value of H0, H1 and H2.
    """
    from libdmet.solver import uccsd_rdm
    log.debug(0, "exp_val_uccsd: start")

    # frozen core
    mo_idx = mycc.get_frozen_mask()
    if not (all(mo_idx[0]) and all(mo_idx[1])):
        nocc = [np.count_nonzero(mycc.mo_occ[s] > 0) for s in range(2)]
        core_idx = [np.arange(nocc[s])[~mo_idx[s][:nocc[s]]] for s in range(2)]
        act_idx  = [np.arange(H1.shape[-1])[mo_idx[s]] for s in range(2)]
        rdm1_core = np.zeros_like(H1)
        for s in range(2):
            rdm1_core[s, core_idx[s], core_idx[s]] = 1.0
        veff_core = _get_veff(rdm1_core, H2)
        E_core = einsum('sij, sji', H1 + veff_core*0.5, rdm1_core)
        
        H0 += E_core
        H1 = [(H1[s] + veff_core[s])[np.ix_(act_idx[s], act_idx[s])] for s in range(2)]
        H2 = [take_eri(H2[0], act_idx[0], act_idx[0], act_idx[0], act_idx[0], compact=True),
              take_eri(H2[1], act_idx[1], act_idx[1], act_idx[1], act_idx[1], compact=True),
              take_eri(H2[2], act_idx[0], act_idx[0], act_idx[1], act_idx[1], compact=True)]

    f = lib.H5TmpFile(filename=rdm2_tmp_fname, mode='w')
    t1, t2, l1, l2 = mycc.t1, mycc.t2, mycc.l1, mycc.l2
    uccsd_rdm._gamma2_outcore(mycc, t1, t2, l1, l2, f, False)

    nocca, nvira = t1[0].shape
    norba = nocca + nvira
    noccb, nvirb = t1[1].shape
    norbb = noccb + nvirb
    
    # E1
    d1 = uccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
    doo, dOO = d1[0]
    dov, dOV = d1[1]
    dvo, dVO = d1[2]
    dvv, dVV = d1[3]
    E1 =   einsum('ij, ji', H1[0][:nocca, :nocca], doo) \
         + einsum('ij, ji', H1[0][nocca:, :nocca], dov) \
         + einsum('ij, ji', H1[0][:nocca, nocca:], dvo) \
         + einsum('ij, ji', H1[0][nocca:, nocca:], dvv) \
         + np.sum(H1[0][range(nocca), range(nocca)])
    E1 +=  einsum('ij, ji', H1[1][:noccb, :noccb], dOO) \
         + einsum('ij, ji', H1[1][noccb:, :noccb], dOV) \
         + einsum('ij, ji', H1[1][:noccb, noccb:], dVO) \
         + einsum('ij, ji', H1[1][noccb:, noccb:], dVV) \
         + np.sum(H1[1][range(noccb), range(noccb)])
    
    # product part
    rdm1_mo = uccsd_rdm._make_rdm1(mycc, d1, with_frozen=False)
    rdm1_mo = list(rdm1_mo)
    rdm1_mo[0][np.diag_indices(nocca)] -= 1.0
    rdm1_mo[1][np.diag_indices(noccb)] -= 1.0

    rdm1_hf = [np.zeros((norba, norba)), np.zeros((norbb, norbb))]
    rdm1_hf[0][range(nocca), range(nocca)] = 1.0
    rdm1_hf[1][range(noccb), range(noccb)] = 1.0
    
    # veff_hf shape can be different for different spin
    vj00, vk00 = hf.dot_eri_dm(H2[0], rdm1_hf[0], hermi=1)
    vj11, vk11 = hf.dot_eri_dm(H2[1], rdm1_hf[1], hermi=1)
    if rdm1_hf[0].shape == rdm1_hf[1].shape:
        vj01 = hf.dot_eri_dm(H2[2], rdm1_hf[1], hermi=1, with_j=True,
                             with_k=False)[0]
        vj10 = hf.dot_eri_dm(H2[2].T, rdm1_hf[0], hermi=1, with_j=True,
                             with_k=False)[0]
    else:
        rdm1_b = rdm1_hf[1] * 2.0
        rdm1_b[range(norbb), range(norbb)] *= 0.5
        rdm1_b = lib.pack_tril(rdm1_b)
        vj01 = np.dot(H2[2], rdm1_b)
        vj01 = lib.unpack_tril(vj01)
        
        rdm1_a = rdm1_hf[0] * 2.0
        rdm1_a[range(norba), range(norba)] *= 0.5
        rdm1_a = lib.pack_tril(rdm1_a)
        vj10 = np.dot(H2[2].T, rdm1_a)
        vj10 = lib.unpack_tril(vj10)
        rdm1_a = rdm1_b = None

    veff_hf = [vj00 + vj01 - vk00, vj11 + vj10 - vk11]
    E2_prod = 0.0
    for s in range(2):
        E2_prod += 0.5 * einsum('ij, ji', veff_hf[s], rdm1_hf[s] + rdm1_mo[s] * 2.0)
    
    # cumulant part
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    unit = max(nvira, nvirb)**3 * 6
    if blksize is None:
        blksize = min(nocca, noccb, nvira, nvirb, max(cc.ccsd.BLKMIN, 
                      int(max_memory*0.9e6/8/unit)))
        if blksize <= 0:
            blksize = 100
    
    log.debug(0, "exp_val_uccsd: current memory: %s MB (max_memory: %s MB)", 
              lib.current_memory()[0], mycc.max_memory)
    log.debug(1, "exp_val_uccsd: unit size: %s , blksize: %s", unit, blksize)
    
    # AAAA part
    E2_aa = 0.0
    oidxa = np.arange(nocca)
    vidxa = np.arange(nocca, norba)
    
    for p0, p1 in lib.prange(0, nocca, blksize):
        cidx = oidxa[p0:p1]

        # ovov
        eri_ovov = take_eri(H2[0], cidx, vidxa, oidxa, vidxa)
        E_ovov = einsum('ijkl, ijkl', eri_ovov, f['dovov'][p0:p1])
        E2_aa += E_ovov * 2
        eri_ovov = None

        # oovv
        eri_oovv = take_eri(H2[0], cidx, oidxa, vidxa, vidxa)
        E_oovv = einsum('ijkl, ijkl', eri_oovv, f['doovv'][p0:p1]) 
        E2_aa += E_oovv * 2
        eri_oovv = None
    
        # ovvo
        eri_ovvo = take_eri(H2[0], cidx, vidxa, vidxa, oidxa)
        E_ovvo = einsum('ijkl, ijkl', eri_ovvo, f['dovvo'][p0:p1])
        E2_aa += E_ovvo * 2
        eri_ovvo = None
        
        # oooo
        eri_oooo = take_eri(H2[0], cidx, oidxa, oidxa, oidxa)
        E_oooo = einsum('ijkl, ijkl', eri_oooo, f['doooo'][p0:p1])
        E2_aa += E_oooo
        eri_oooo = None
    
        # ovvv
        eri_ovvv = take_eri(H2[0], cidx, vidxa, vidxa, vidxa)
        E_ovvv = einsum('ijkl, ijkl', eri_ovvv, f['dovvv'][p0:p1])
        E2_aa += E_ovvv * 4
        eri_ovvv = None
        
        # ooov
        eri_ooov = take_eri(H2[0], cidx, oidxa, oidxa, vidxa)
        E_ooov = einsum('ijkl, ijkl', eri_ooov, f['dooov'][p0:p1])
        E2_aa += E_ooov * 4
        eri_ooov = None
        
    for p0, p1 in lib.prange(0, nvira, blksize):
        cidx = vidxa[p0:p1]
        # vvvv
        eri_vvvv = take_eri(H2[0], cidx, vidxa, vidxa, vidxa)
        E_vvvv = einsum('ijkl, ijkl', eri_vvvv, f['dvvvv'][p0:p1])
        E2_aa += E_vvvv
        eri_vvvv = None

    # BBBB part
    E2_bb = 0.0
    Oidxb = np.arange(noccb)
    Vidxb = np.arange(noccb, norbb)
    
    for p0, p1 in lib.prange(0, noccb, blksize):
        cidx = Oidxb[p0:p1]

        # ovov
        eri_ovov = take_eri(H2[1], cidx, Vidxb, Oidxb, Vidxb)
        E_ovov = einsum('ijkl, ijkl', eri_ovov, f['dOVOV'][p0:p1])
        E2_bb += E_ovov * 2
        eri_ovov = None

        # oovv
        eri_oovv = take_eri(H2[1], cidx, Oidxb, Vidxb, Vidxb)
        E_oovv = einsum('ijkl, ijkl', eri_oovv, f['dOOVV'][p0:p1]) 
        E2_bb += E_oovv * 2
        eri_oovv = None
    
        # ovvo
        eri_ovvo = take_eri(H2[1], cidx, Vidxb, Vidxb, Oidxb)
        E_ovvo = einsum('ijkl, ijkl', eri_ovvo, f['dOVVO'][p0:p1])
        E2_bb += E_ovvo * 2
        eri_ovvo = None
        
        # oooo
        eri_oooo = take_eri(H2[1], cidx, Oidxb, Oidxb, Oidxb)
        E_oooo = einsum('ijkl, ijkl', eri_oooo, f['dOOOO'][p0:p1])
        E2_bb += E_oooo
        eri_oooo = None
    
        # ovvv
        eri_ovvv = take_eri(H2[1], cidx, Vidxb, Vidxb, Vidxb)
        E_ovvv = einsum('ijkl, ijkl', eri_ovvv, f['dOVVV'][p0:p1])
        E2_bb += E_ovvv * 4
        eri_ovvv = None
        
        # ooov
        eri_ooov = take_eri(H2[1], cidx, Oidxb, Oidxb, Vidxb)
        E_ooov = einsum('ijkl, ijkl', eri_ooov, f['dOOOV'][p0:p1])
        E2_bb += E_ooov * 4
        eri_ooov = None
        
    for p0, p1 in lib.prange(0, nvirb, blksize):
        cidx = Vidxb[p0:p1]
        # vvvv
        eri_vvvv = take_eri(H2[1], cidx, Vidxb, Vidxb, Vidxb)
        E_vvvv = einsum('ijkl, ijkl', eri_vvvv, f['dVVVV'][p0:p1])
        E2_bb += E_vvvv
        eri_vvvv = None
    
    # AABB part
    E2_ab = 0.0
    for p0, p1 in lib.prange(0, nocca, blksize):
        cidx = oidxa[p0:p1]

        # ovOV
        eri_ovOV = take_eri(H2[2], cidx, vidxa, Oidxb, Vidxb)
        E_ovOV = einsum('ijkl, ijkl', eri_ovOV, f['dovOV'][p0:p1])
        E2_ab += E_ovOV * 2
        eri_ovOV = None
        
        # ooVV
        eri_ooVV = take_eri(H2[2], cidx, oidxa, Vidxb, Vidxb)
        E_ooVV = einsum('ijkl, ijkl', eri_ooVV, f['dooVV'][p0:p1])
        E2_ab += E_ooVV
        eri_ooVV = None

        # ovVO
        eri_ovVO = take_eri(H2[2], cidx, vidxa, Vidxb, Oidxb)
        E_ovVO = einsum('ijkl, ijkl', eri_ovVO, f['dovVO'][p0:p1])
        E2_ab += E_ovVO * 2
        eri_ovVO = None

        # ooOO
        eri_ooOO = take_eri(H2[2], cidx, oidxa, Oidxb, Oidxb)
        E_ooOO = einsum('ijkl, ijkl', eri_ooOO, f['dooOO'][p0:p1])
        E2_ab += E_ooOO
        eri_ooOO = None
        
        # ovVV
        eri_ovVV = take_eri(H2[2], cidx, vidxa, Vidxb, Vidxb)
        E_ovVV = einsum('ijkl, ijkl', eri_ovVV, f['dovVV'][p0:p1])
        E2_ab += E_ovVV * 2
        eri_ovVV = None
        
        # ooOV
        eri_ooOV = take_eri(H2[2], cidx, oidxa, Oidxb, Vidxb)
        E_ooOV = einsum('ijkl, ijkl', eri_ooOV, f['dooOV'][p0:p1])
        E2_ab += E_ooOV * 2
        eri_ooOV = None
        
        # ovOO
        eri_ovOO = take_eri(H2[2], cidx, vidxa, Oidxb, Oidxb)
        E_ovOO = einsum('ijkl, klij', eri_ovOO, f['dOOov'][:, :, p0:p1])
        E2_ab += E_ovOO * 2
        eri_ovOO = None
    
    for p0, p1 in lib.prange(0, nvira, blksize):
        cidx = vidxa[p0:p1]
        
        # vvOO
        eri_vvOO = take_eri(H2[2], cidx, vidxa, Oidxb, Oidxb)
        E_vvOO = einsum('ijkl, klij', eri_vvOO, f['dOOvv'][:, :, p0:p1])
        E2_ab += E_vvOO
        eri_vvOO = None

        # vvVV
        eri_vvVV = take_eri(H2[2], cidx, vidxa, Vidxb, Vidxb)
        E_vvVV = einsum('ijkl, ijkl', eri_vvVV, f['dvvVV'][p0:p1])
        E2_ab += E_vvVV
        eri_vvVV = None

        # vvOV
        eri_vvOV = take_eri(H2[2], cidx, vidxa, Oidxb, Vidxb)
        E_vvOV = einsum('ijkl, klij', eri_vvOV, f['dOVvv'][:, :, p0:p1])
        E2_ab += E_vvOV * 2
        eri_vvOV = None
    
    E2_cum = 0.5 * (E2_aa + E2_bb) + E2_ab
    E2 = E2_prod + E2_cum
    
    E_tot = H0 + E1 + E2
    log.debug(0, "CC exp_val: E0: %20.12f, E1: %20.12f, E2: %20.12f \n"
              "(prod: %20.12f, cum: %20.12f), E_tot: %20.12f", 
              H0, E1, E2, E2_prod, E2_cum, E_tot)
    return E_tot

def exp_val_gccsd(mycc, H1, H2, H0=0.0, rdm2_tmp_fname=None, blksize=None):
    """
    Expectation value of H0, H1 and H2, using an outcore routine.
    H1 and H2 are in MO basis. 
    H2 is real and has 8-fold symmetry, 1, 4, 8-fold array are ok.

    Args:
        mycc: cc object.
        H1: (nmo, nmo)
        H2: (nmo, nmo, nmo, nmo) or (nmo_pair, nmo_pair) or (nmo_pair_pair).
        H0: scalar
        rdm2_tmp_fname: if given, will save the rdm2 intermidiates.

    Returns:
        E_tot: the expectation value of H0, H1 and H2.
    """
    # ZHC NOTE TODO support outcore routine 
    # currently I save intermidiates to disk and then load
    # also, we may load H2 from disk
    log.debug(0, "exp_val_gccsd: start")
    
    # frozen core
    mo_idx = mycc.get_frozen_mask()
    if not all(mo_idx):
        nocc = np.count_nonzero(mycc.mo_occ > 0)
        core_idx = np.arange(nocc)[~mo_idx[:nocc]]
        act_idx  = np.arange(H1.shape[-1])[mo_idx]
        rdm1_core = np.zeros_like(H1)
        rdm1_core[core_idx, core_idx] = 1.0
        veff_core = _get_veff_ghf(rdm1_core, H2)
        E_core = einsum('ij, ji', H1 + veff_core*0.5, rdm1_core)
        
        H0 += E_core
        H1 = (H1 + veff_core)[np.ix_(act_idx, act_idx)]
        H2 = take_eri(H2, act_idx, act_idx, act_idx, act_idx, compact=True)
    
    f = lib.H5TmpFile(filename=rdm2_tmp_fname, mode='w')
    t1, t2, l1, l2 = mycc.t1, mycc.t2, mycc.l1, mycc.l2
    d2 = cc.gccsd_rdm._gamma2_intermediates(mycc, t1, t2, l1, l2)
    d2 = [np.zeros((0)) if dx is None else dx for dx in d2]
    f["dovov"], f["dvvvv"], f["doooo"], f["doovv"], \
    f["dovvo"], f["dvvov"], f["dovvv"], f["dooov"] = d2
    d2 = None
    
    nocc, nvir = t1.shape
    norb = nocc + nvir

    # E1
    d1 = cc.gccsd_rdm._gamma1_intermediates(mycc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    E1 =   einsum('ij, ji', H1[:nocc, :nocc], doo) \
         + einsum('ij, ji', H1[nocc:, :nocc], dov) \
         + einsum('ij, ji', H1[:nocc, nocc:], dvo) \
         + einsum('ij, ji', H1[nocc:, nocc:], dvv) \
         + np.sum(H1[range(nocc), range(nocc)])
    
    # product part
    rdm1_mo = cc.gccsd_rdm._make_rdm1(mycc, d1, with_frozen=False)
    rdm1_mo = np.asarray(rdm1_mo)
    rdm1_mo[np.diag_indices(nocc)] -= 1.0

    rdm1_hf = np.zeros((norb, norb))
    rdm1_hf[range(nocc), range(nocc)] = 1.0
    
    veff_hf = _get_veff_ghf(rdm1_hf, H2)
    E2_prod = 0.5 * einsum('ij, ji', veff_hf, rdm1_hf + rdm1_mo * 2.0)
    
    # cumulant part
    max_memory = max(0, mycc.max_memory - lib.current_memory()[0])
    unit = nvir**3 * 6
    if blksize is None:
        blksize = min(nocc, nvir, max(cc.ccsd.BLKMIN, int(max_memory*0.9e6/8/unit)))
    
    log.debug(0, "exp_val_gccsd: current memory: %s MB (max_memory: %s MB)", 
              lib.current_memory()[0], mycc.max_memory)
    log.debug(1, "exp_val_gccsd: unit size: %s , blksize: %s", unit, blksize)
    
    E2_cum = 0.0
    oidx = np.arange(nocc)
    vidx = np.arange(nocc, norb)
    
    for p0, p1 in lib.prange(0, nocc, blksize):
        cidx = oidx[p0:p1]
        
        # ovov
        eri_ovov = take_eri(H2, cidx, vidx, oidx, vidx)
        E_ovov = einsum('ijkl, ijkl', eri_ovov, f['dovov'][p0:p1])
        E2_cum += E_ovov * 2
        eri_ovov = None
        
        eri_oovv = take_eri(H2, cidx, oidx, vidx, vidx)
        E_oovv = -einsum('ijkl, iklj', eri_oovv, f['dovvo'][p0:p1])
        E2_cum += E_oovv * 2
        eri_oovv = None
    
        # ovvo
        eri_ovvo = take_eri(H2, cidx, vidx, vidx, oidx)
        E_ovvo = einsum('ijkl, ijkl', eri_ovvo, f['dovvo'][p0:p1])
        E2_cum += E_ovvo * 2
        eri_ovvo = None
        
        # oooo
        eri_oooo = take_eri(H2, cidx, oidx, oidx, oidx)
        E_oooo = einsum('ijkl, ijkl', eri_oooo, f['doooo'][p0:p1])
        E2_cum += E_oooo
        eri_oooo = None
    
        # ovvv
        eri_ovvv = take_eri(H2, cidx, vidx, vidx, vidx)
        E_ovvv = einsum('ijkl, ijkl', eri_ovvv, f['dovvv'][p0:p1])
        E2_cum += E_ovvv * 4
        eri_ovvv = None
        
        # ooov
        eri_ooov = take_eri(H2, cidx, oidx, oidx, vidx)
        E_ooov = einsum('ijkl, ijkl', eri_ooov, f['dooov'][p0:p1])
        E2_cum += E_ooov * 4
        eri_ooov = None
        
    for p0, p1 in lib.prange(0, nvir, blksize):
        cidx = vidx[p0:p1]
        # vvvv
        eri_vvvv = take_eri(H2, cidx, vidx, vidx, vidx)
        E_vvvv = einsum('ijkl, ijkl', eri_vvvv, f['dvvvv'][p0:p1])
        E2_cum += E_vvvv
        eri_vvvv = None
    
    E2_cum *= 0.5
    E2 = E2_prod + E2_cum
    
    E_tot = H0 + E1 + E2
    log.debug(0, "CC exp_val: E0: %20.12f, E1: %20.12f, E2: %20.12f \n"
              "(prod: %20.12f, cum: %20.12f), E_tot: %20.12f", 
              H0, E1, E2, E2_prod, E2_cum, E_tot)
    return E_tot

# ****************************************************************************
# orbital optimized OO-CCSD
# ****************************************************************************

class CCDAsFCISolver(object):
    def __init__(self, restricted=True, ghf=True, max_cycle=200, 
                 level_shift=0.0, conv_tol=1e-7,
                 conv_tol_normt=1e-5, diis_space=8, iterative_damping=1.0, 
                 max_memory=120000, restart=False, verbose=4, fname='mcscf',
                 fcivec=None, approx_l=False, fix_fcivec=False, alpha=None,
                 Mu=None, linear=False, **kwargs):
        """
        CCD as CASSCF solver.
        """
        self.mycc = None
        self.restricted = restricted
        self.ghf = ghf
        self.level_shift = level_shift
        self.max_cycle = max_cycle
        self.conv_tol = conv_tol
        self.conv_tol_normt = conv_tol_normt
        self.diis_space = diis_space
        self.iterative_damping = iterative_damping
        self.max_memory = max_memory
        self.verbose = verbose
        self.approx_l = approx_l
        self.linear = linear

        self.alpha = alpha
        self.Mu = Mu
        
        self.restart = restart
        self.fname = fname
        self.fcivec = fcivec
        self.fix_fcivec = fix_fcivec

    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        if self.ghf:
            from libdmet.solver import scf as scf_hp
            nelec = sum(nelec)
            Ham = integral.Integral(norb, True, False, ecore, {"cd": h1[None]},
                                    {"ccdd": h2[None]}, ovlp=None)
            scfsolver = scf_hp.SCF(newton_ah=False, no_kernel=True,
                                   verbose=self.verbose)
            scfsolver.set_system(nelec, 0, False, True,
                                 max_memory=self.max_memory)
            scfsolver.set_integral(Ham)
            
            scfsolver.GGHF(alpha=self.alpha, Mu=self.Mu)
            fake_hf = scfsolver.mf
            fake_hf.mo_coeff = np.eye(norb)
            fake_hf.mo_occ   = np.zeros(norb)
            fake_hf.mo_occ[:nelec] = 1
            
            if self.linear:
                log.info("LGGCCD used.")
                self.mycc = LGGCCD(fake_hf)
            else:
                self.mycc = GGCCD(fake_hf)
            eris = self.mycc.ao2mo()
            self.mycc.level_shift = self.level_shift
            self.mycc.max_cycle = self.max_cycle
            self.mycc.conv_tol = self.conv_tol
            self.mycc.conv_tol_normt = self.conv_tol_normt
            if self.diis_space <= 0:
                self.mycc.diis = False
            self.mycc.diis_space = self.diis_space
            self.mycc.iterative_damping = self.iterative_damping

            if self.restart:
                tl = self.fcivec
                if tl is not None:
                    t1, t2, l1, l2 = tl.cc_amplitues
                    if os.path.exists("%s_u.npy"%self.fname):
                        log.info("OO-CCD: rotate t1, t2, l1, l2.")
                        u_mat = np.load("%s_u.npy"%self.fname)
                        t1 = transform_t1_to_bo(t1, u_mat)
                        t2 = transform_t2_to_bo(t2, u_mat)
                        if l1 is not None:
                            l1 = transform_l1_to_bo(l1, u_mat)
                        if l2 is not None:
                            l2 = transform_l2_to_bo(l2, u_mat)
                else:
                    t1 = t2 = l1 = l2 = None
            else:
                t1 = t2 = l1 = l2 = None
            
            if (not self.fix_fcivec) or (t1 is None) or (t2 is None):
                e_corr, t1, t2 = self.mycc.kernel(t1=t1, t2=t2, eris=eris)

            if (not self.fix_fcivec) or (l1 is None) or (l2 is None):
                if (not self.approx_l):
                    l1, l2 = self.mycc.solve_lambda(l1, l2, eris=eris)
            
            if self.fix_fcivec:
                self.mycc.e_hf = getattr(eris, 'e_hf', None)
                self.mycc.e_corr = self.mycc.energy(t1=t1, t2=t2, eris=eris)
            
            e_tot = self.mycc.e_tot
            
            if self.restart:
                self.fcivec = CCSDAmplitudesAsCIWfn([t1, t2, l1, l2])
            return e_tot, CCSDAmplitudesAsCIWfn([t1, t2, l1, l2])
        else:
            if self.restricted:
                raise NotImplementedError

            from libdmet.solver import scf as scf_hp
            nelec_a, nelec_b = nelec
            spin = nelec_a - nelec_b
            # ZHC NOTE
            # pyscf's convention of h2 is aa, ab, bb, need to convert
            h1 = np.asarray(h1)
            #h2 = np.asarray((h2[0], h2[2], h2[1]))

            Ham = integral.Integral(norb, False, False, ecore, {"cd": h1},
                                    {"ccdd": h2}, ovlp=None)
            scfsolver = scf_hp.SCF(newton_ah=False, no_kernel=True,
                                   verbose=self.verbose)
            scfsolver.set_system(nelec[0] + nelec[1], nelec[0] - nelec[1],
                                 False, False, max_memory=self.max_memory)
            scfsolver.set_integral(Ham)
            
            scfsolver.HF(alpha=self.alpha, Mu=self.Mu)
            fake_hf = scfsolver.mf
            fake_hf.mo_coeff = np.array((np.eye(norb), np.eye(norb)))
            fake_hf.mo_occ   = np.zeros((2, norb))
            fake_hf.mo_occ[0, :nelec[0]] = 1
            fake_hf.mo_occ[1, :nelec[1]] = 1
            
            if self.linear:
                raise NotImplementedError
            else:
                self.mycc = UICCD(fake_hf)
            eris = self.mycc.ao2mo()
            self.mycc.level_shift = self.level_shift
            self.mycc.max_cycle = self.max_cycle
            self.mycc.conv_tol = self.conv_tol
            self.mycc.conv_tol_normt = self.conv_tol_normt
            if self.diis_space <= 0:
                self.mycc.diis = False
            self.mycc.diis_space = self.diis_space
            self.mycc.iterative_damping = self.iterative_damping

            if self.restart:
                tl = self.fcivec
                if tl is not None:
                    t1, t2, l1, l2 = tl.cc_amplitues
                    if os.path.exists("%s_u_a.npy"%self.fname):
                        log.info("OO-CCD: rotate t1, t2, l1, l2.")
                        u_mat_a = np.load("%s_u_a.npy"%self.fname)
                        u_mat_b = np.load("%s_u_b.npy"%self.fname)
                        u_mat = (u_mat_a, u_mat_b)
                        t1 = transform_t1_to_bo(t1, u_mat)
                        t2 = transform_t2_to_bo(t2, u_mat)
                        if l1 is not None:
                            l1 = transform_l1_to_bo(l1, u_mat)
                        if l2 is not None:
                            l2 = transform_l2_to_bo(l2, u_mat)
                else:
                    t1 = t2 = l1 = l2 = None
            else:
                t1 = t2 = l1 = l2 = None
            
            if (not self.fix_fcivec) or (t1 is None) or (t2 is None):
                e_corr, t1, t2 = self.mycc.kernel(t1=t1, t2=t2, eris=eris)

            if (not self.fix_fcivec) or (l1 is None) or (l2 is None):
                if (not self.approx_l):
                    l1, l2 = self.mycc.solve_lambda(l1, l2, eris=eris)
            
            if self.fix_fcivec:
                self.mycc.e_hf = getattr(eris, 'e_hf', None)
                self.mycc.e_corr = self.mycc.energy(t1=t1, t2=t2, eris=eris)
            
            e_tot = self.mycc.e_tot
            
            if self.restart:
                self.fcivec = CCSDAmplitudesAsCIWfn([t1, t2, l1, l2])
            return e_tot, CCSDAmplitudesAsCIWfn([t1, t2, l1, l2])

    def make_rdm1(self, fake_ci, norb, nelec):
        t1, t2, l1, l2 = fake_ci.cc_amplitues
        if self.approx_l:
            l1 = t1
            l2 = t2
        dm1 = self.mycc.make_rdm1(t1, t2, l1, l2)
        return dm1
    
    make_rdm1s = make_rdm1

    def make_rdm12(self, fake_ci, norb, nelec):
        t1, t2, l1, l2 = fake_ci.cc_amplitues
        if self.approx_l:
            l1 = t1
            l2 = t2
        dm2 = self.mycc.make_rdm2(t1, t2, l1, l2)
        dm1 = self.make_rdm1(fake_ci, norb, nelec)
        return dm1, dm2
    
    make_rdm12s = make_rdm12

    def load_fcivec(self, fname):
        log.debug(1, "CCD solver: read previous t, l.")
        if not os.path.isfile(fname):
            log.info("CCD solver: read previous t, l failed, "
                     "file %s does not exist.", fname)
            return None
        fcc = h5py.File(fname, 'r')
        if "t2" in fcc.keys():
            t1 = np.asarray(fcc['t1'])
            t2 = np.asarray(fcc['t2'])
            if "l2" in fcc.keys():
                l1 = np.asarray(fcc['l1'])
                l2 = np.asarray(fcc['l2'])
            else:
                l1 = l2 = None
        else:
            spin = 2
            t1 = [np.asarray(fcc['t1_%s'%s]) for s in range(spin)]
            t2 = [np.asarray(fcc['t2_%s'%s]) for s in range(spin*(spin+1)//2)]
            if "l2_0" in fcc.keys():
                l1 = [np.asarray(fcc['l1_%s'%s]) for s in range(spin)]
                l2 = [np.asarray(fcc['l2_%s'%s]) for s in range(spin*(spin+1)//2)]
            else:
                l1 = l2 = None
        fcc.close()
        return CCSDAmplitudesAsCIWfn([t1, t2, l1, l2])

    def save_fcivec(self, fname):
        log.debug(1, "CCD solver: dump t1, t2, l1, l2.")
        t1, t2, l1, l2 = self.fcivec.cc_amplitues

        fcc = h5py.File(fname, 'w')
        if isinstance(t2, np.ndarray) and t2.ndim == 4:
            fcc['t1'] = np.asarray(t1)
            fcc['t2'] = np.asarray(t2)
            if l2 is not None:
                fcc['l1'] = np.asarray(l1)
                fcc['l2'] = np.asarray(l2)
        else:
            spin = 2
            for s in range(spin):
                fcc['t1_%s'%s] = np.asarray(t1[s])
            for s in range(spin*(spin+1)//2):
                fcc['t2_%s'%s] = np.asarray(t2[s])
            if l2 is not None:
                for s in range(spin):
                    fcc['l1_%s'%s] = np.asarray(l1[s])
                for s in range(spin*(spin+1)//2):
                    fcc['l2_%s'%s] = np.asarray(l2[s])
        fcc.close()

class CCSDAmplitudesAsCIWfn:
    def __init__(self, cc_amplitues):
        self.cc_amplitues = cc_amplitues

# ****************************************************************************
# reference functions:
# ****************************************************************************

def _make_eris_incore_uhf_ref(mycc, mo_coeff=None, ao2mofn=None):
    """
    Hacked CC make eri function. NOTE the order.
    """
    assert ao2mofn is None
    cput0 = (time.process_time(), time.perf_counter())
    eris = _ChemistsERIs()
    eris._common_init_(mycc, mo_coeff)

    nocca, noccb = mycc.nocc
    nmoa, nmob = mycc.nmo
    nvira, nvirb = nmoa-nocca, nmob-noccb

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    nmoa = moa.shape[1]
    nmob = mob.shape[1]
    
    if len(mycc._scf._eri) == 1:
        eri_ao = [mycc._scf._eri[0], mycc._scf._eri[0], mycc._scf._eri[0]]
    elif len(mycc._scf._eri) == 3:
        eri_ao = mycc._scf._eri
    else:
        raise ValueError("Unknown ERI length %s"%(len(mycc._scf._eri)))
    
    # aa
    eri_aa = ao2mo.restore(1, ao2mo.full(eri_ao[0], moa), nmoa)
    eri_aa = eri_aa.reshape(nmoa,nmoa,nmoa,nmoa)

    eris.oooo = eri_aa[:nocca,:nocca,:nocca,:nocca].copy()
    eris.ovoo = eri_aa[:nocca,nocca:,:nocca,:nocca].copy()
    eris.ovov = eri_aa[:nocca,nocca:,:nocca,nocca:].copy()
    eris.oovv = eri_aa[:nocca,:nocca,nocca:,nocca:].copy()
    eris.ovvo = eri_aa[:nocca,nocca:,nocca:,:nocca].copy()

    ovvv = eri_aa[:nocca, nocca:, nocca:, nocca:].copy()
    ovvv = ovvv.reshape(nocca*nvira, nvira, nvira)
    eris.ovvv = lib.pack_tril(ovvv).reshape(nocca, nvira, nvira*(nvira+1)//2)
    ovvv = None
    
    vvvv = eri_aa[nocca:, nocca:, nocca:, nocca:].copy()
    eri_aa = None
    eris.vvvv = ao2mo.restore(4, vvvv, nvira)
    vvvv = None

    # bb
    eri_bb = ao2mo.restore(1, ao2mo.full(eri_ao[1], mob), nmob)
    eri_bb = eri_bb.reshape(nmob,nmob,nmob,nmob)

    eris.OOOO = eri_bb[:noccb,:noccb,:noccb,:noccb].copy()
    eris.OVOO = eri_bb[:noccb,noccb:,:noccb,:noccb].copy()
    eris.OVOV = eri_bb[:noccb,noccb:,:noccb,noccb:].copy()
    eris.OOVV = eri_bb[:noccb,:noccb,noccb:,noccb:].copy()
    eris.OVVO = eri_bb[:noccb,noccb:,noccb:,:noccb].copy()
    
    OVVV = eri_bb[:noccb, noccb:, noccb:, noccb:].copy()
    OVVV = OVVV.reshape(noccb*nvirb,nvirb,nvirb)
    eris.OVVV = lib.pack_tril(OVVV).reshape(noccb, nvirb, nvirb*(nvirb+1)//2)
    OVVV = None
    
    VVVV = eri_bb[noccb:, noccb:, noccb:, noccb:].copy()
    eri_bb = None
    eris.VVVV = ao2mo.restore(4, VVVV, nvirb)
    VVVV = None

    # ab
    eri_ab = ao2mo.general(eri_ao[2], (moa, moa, mob, mob),
                           compact=False)
    eri_ao = None
    eri_ab = eri_ab.reshape(nmoa,nmoa,nmob,nmob)

    eris.ooOO = eri_ab[:nocca,:nocca,:noccb,:noccb].copy()
    eris.ovOO = eri_ab[:nocca,nocca:,:noccb,:noccb].copy()
    eris.ovOV = eri_ab[:nocca,nocca:,:noccb,noccb:].copy()
    eris.ooVV = eri_ab[:nocca,:nocca,noccb:,noccb:].copy()
    eris.ovVO = eri_ab[:nocca,nocca:,noccb:,:noccb].copy()
    
    ovVV = eri_ab[:nocca, nocca:, noccb:, noccb:].copy()
    ovVV = ovVV.reshape(nocca*nvira, nvirb, nvirb)
    eris.ovVV = lib.pack_tril(ovVV).reshape(nocca, nvira, nvirb*(nvirb+1)//2)
    ovVV = None
    
    vvVV = eri_ab[nocca:, nocca:, noccb:, noccb:].copy()
    vvVV = vvVV.reshape(nvira**2, nvirb**2)
    idxa = np.tril_indices(nvira)
    idxb = np.tril_indices(nvirb)
    eris.vvVV = lib.take_2d(vvVV, idxa[0]*nvira+idxa[1], idxb[0]*nvirb+idxb[1])
    vvVV = None
    
    # ba
    eri_ba = eri_ab.transpose(2,3,0,1)
    eri_ab = None

    #eris.OOoo = eri_ba[:noccb,:noccb,:nocca,:nocca].copy()
    eris.OVoo = eri_ba[:noccb,noccb:,:nocca,:nocca].copy()
    #eris.OVov = eri_ba[:noccb,noccb:,:nocca,nocca:].copy()
    eris.OOvv = eri_ba[:noccb,:noccb,nocca:,nocca:].copy()
    eris.OVvo = eri_ba[:noccb,noccb:,nocca:,:nocca].copy()
    
    OVvv = eri_ba[:noccb,noccb:,nocca:,nocca:].copy()
    #eris.VVvv = eri_ba[noccb:,noccb:,nocca:,nocca:].copy()
    eri_ba = None
        
    OVvv = OVvv.reshape(noccb*nvirb, nvira, nvira)
    eris.OVvv = lib.pack_tril(OVvv).reshape(noccb, nvirb, nvira*(nvira+1)//2)
    OVvv = None
    return eris

def _make_eris_incore_ghf_ref(mycc, mo_coeff=None, ao2mofn=None):
    cput0 = (time.process_time(), time.perf_counter())
    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nao, nmo = eris.mo_coeff.shape

    if callable(ao2mofn):
        eri = ao2mofn(eris.mo_coeff).reshape([nmo]*4)
    else:
        assert eris.mo_coeff.dtype == np.double
        if (nao == nmo) and (max_abs(eris.mo_coeff - np.eye(nmo)) < 1e-13):
            eri = ao2mo.restore(1, mycc._scf._eri, nao)
        else:
            eri = ao2mo.kernel(mycc._scf._eri, eris.mo_coeff) 
            if eri.dtype == np.double:
                eri = ao2mo.restore(1, eri, nmo)

    eri = eri.reshape(nmo,nmo,nmo,nmo)
    eri = eri.transpose(0,2,1,3) - eri.transpose(0,2,3,1)

    eris.oooo = eri[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ooov = eri[:nocc,:nocc,:nocc,nocc:].copy()
    eris.oovv = eri[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovov = eri[:nocc,nocc:,:nocc,nocc:].copy()
    eris.ovvo = eri[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri[nocc:,nocc:,nocc:,nocc:].copy()
    return eris

if __name__ == '__main__':
    pass
