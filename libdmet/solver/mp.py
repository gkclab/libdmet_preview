#! /usr/bin/env python

"""
MP solver for impurity problem:
    UIMP2
    GGMP2
    OOMP2

Author:
    Zhihao Cui <zhcui0408@gmail.com>
"""

import os
import time
import numpy as np
import h5py

from pyscf import ao2mo, mp
from pyscf import lib
from pyscf.lib import logger
from pyscf.mp.ump2 import _ChemistsERIs
from pyscf.mp.gmp2 import _PhysicistsERIs 

from libdmet.system import integral
from libdmet.utils import max_abs
from libdmet.solver import gmp2 as mygmp2
from libdmet.utils import logger as log

# ****************************************************************************
# UIMP2
# ****************************************************************************

def _make_eris_uhf(mp, mo_coeff=None, ao2mofn=None, verbose=None):
    log = logger.new_logger(mp, verbose)
    time0 = (logger.process_clock(), logger.perf_counter())
    # ZHC NOTE for compatibility of old pyscf
    try:
        eris = _ChemistsERIs()
        eris._common_init_(mp, mo_coeff)
    except TypeError:
        eris = _ChemistsERIs(mp, mo_coeff)

    nocca, noccb = mp.get_nocc()
    nmoa, nmob = mp.get_nmo()
    nvira, nvirb = nmoa-nocca, nmob-noccb
    nao = eris.mo_coeff[0].shape[0]
    nmo_pair = nmoa * (nmoa+1) // 2
    nao_pair = nao * (nao+1) // 2
    mem_incore = (nao_pair**2 + nmo_pair**2) * 3 * 8 / 1e6
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mp.max_memory-mem_now)

    moa = eris.mo_coeff[0]
    mob = eris.mo_coeff[1]
    orboa = moa[:,:nocca]
    orbob = mob[:,:noccb]
    orbva = moa[:,nocca:]
    orbvb = mob[:,noccb:]

    if (mp.mol.incore_anyway or
        (mp._scf._eri is not None and mem_incore+mem_now < mp.max_memory)):
        log.debug('transform (ia|jb) incore')
        if callable(ao2mofn):
            raise NotImplementedError
        else:
            # ZHC NOTE
            eri_format, spin_dim = integral.get_eri_format(mp._scf._eri, nao)
            if spin_dim == 0:
                eris.ovov = ao2mo.general(mp._scf._eri, (orboa, orbva, orboa, orbva))
                eris.ovOV = ao2mo.general(mp._scf._eri, (orboa, orbva, orbob, orbvb))
                eris.OVOV = ao2mo.general(mp._scf._eri, (orbob, orbvb, orbob, orbvb))
            elif spin_dim == 1:
                eris.ovov = ao2mo.general(mp._scf._eri[0], (orboa, orbva, orboa, orbva))
                eris.ovOV = ao2mo.general(mp._scf._eri[0], (orboa, orbva, orbob, orbvb))
                eris.OVOV = ao2mo.general(mp._scf._eri[0], (orbob, orbvb, orbob, orbvb))
            else:
                eris.ovov = ao2mo.general(mp._scf._eri[0], (orboa, orbva, orboa, orbva))
                eris.ovOV = ao2mo.general(mp._scf._eri[2], (orboa, orbva, orbob, orbvb))
                eris.OVOV = ao2mo.general(mp._scf._eri[1], (orbob, orbvb, orbob, orbvb))

    elif getattr(mp._scf, 'with_df', None):
        #log.debug('transform (ia|jb) with_df')
        raise NotImplementedError

    else:
        #log.debug('transform (ia|jb) outcore')
        raise NotImplementedError

    time1 = log.timer('Integral transformation', *time0)
    return eris

class UIMP2(mp.ump2.UMP2):
    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return _make_eris_uhf(self, mo_coeff, verbose=self.verbose)

# ****************************************************************************
# GMP2
# ****************************************************************************

def _make_eris_incore_ghf(mp, mo_coeff=None, ao2mofn=None, verbose=None):
    eris = _PhysicistsERIs()
    eris._common_init_(mp, mo_coeff)

    nocc = mp.nocc
    nao, nmo = eris.mo_coeff.shape
    nvir = nmo - nocc
    orbspin = eris.orbspin

    if callable(ao2mofn):
        raise NotImplementedError
    else:
        if (nao == nmo) and (max_abs(eris.mo_coeff - np.eye(nmo)) < 1e-13):
            # ZHC NOTE special treatment for OOMP2,
            # where the ao2mo is not needed for identity mo_coeff.
            eri = ao2mo.restore(1, mp._scf._eri, nao).reshape(nao, nao, nao, nao)
            eri = eri[:nocc, nocc:, :nocc, nocc:]
        else:
            orbo = eris.mo_coeff[:, :nocc]
            orbv = eris.mo_coeff[:, nocc:]
            eri = ao2mo.kernel(mp._scf._eri, (orbo, orbv, orbo, orbv))
            eri = eri.reshape(nocc, nvir, nocc, nvir)
    
    eris.oovv = eri.transpose(0, 2, 1, 3) - eri.transpose(0, 2, 3, 1)
    return eris
    
class GGMP2(mp.gmp2.GMP2):
    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        nmo = self.nmo
        nocc = self.nocc
        nvir = nmo - nocc
        mem_incore = nocc**2*nvir**2*3 * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore_ghf(self, mo_coeff, verbose=self.verbose)
        elif getattr(self._scf, 'with_df', None):
            raise NotImplementedError
        else:
            raise NotImplementedError

# ****************************************************************************
# orbital optimized OO-MP2
# ****************************************************************************

class MP2AsFCISolver(object):
    def __init__(self, ghf=True, max_cycle=200, level_shift=0.0, conv_tol=1e-7,
                 conv_tol_normt=1e-5, diis_space=8, max_memory=120000,
                 verbose=4, restart=False, fname='mcscf', fcivec=None,
                 approx_l=False, fix_fcivec=False, **kwargs):
        self.ghf = ghf
        self.level_shift = level_shift
        self.max_cycle = max_cycle
        self.conv_tol = conv_tol
        self.conv_tol_normt = conv_tol_normt
        self.diis_space = diis_space
        self.max_memory = max_memory
        self.verbose = verbose
        
        self.restart = restart
        self.fname = fname
        self.fcivec = fcivec
        self.fix_fcivec = fix_fcivec
        
    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        if self.ghf:
            from libdmet.solver import scf as scf_hp
            from libdmet.solver.cc import transform_t2_to_bo
            nelec = sum(nelec)
            Ham = integral.Integral(norb, True, False, ecore, {"cd": h1[None]},
                                    {"ccdd": h2[None]}, ovlp=None)
            scfsolver = scf_hp.SCF(newton_ah=False, no_kernel=True,
                                   verbose=self.verbose)
            scfsolver.set_system(nelec, 0, False, True,
                                 max_memory=self.max_memory)
            scfsolver.set_integral(Ham)
            
            scfsolver.GGHF()
            fake_hf = scfsolver.mf
            
            fake_hf.mo_coeff = np.eye(norb)
            fake_hf.mo_occ   = np.zeros(norb)
            fake_hf.mo_occ[:nelec] = 1

            self.cisolver = mygmp2.GGMP2(fake_hf)
            self.cisolver.level_shift = self.level_shift
            self.cisolver.max_cycle = self.max_cycle
            self.cisolver.conv_tol = self.conv_tol
            self.cisolver.conv_tol_normt = self.conv_tol_normt
            self.cisolver.diis_space = self.diis_space
            
            if self.restart:
                t2 = self.fcivec
                if t2 is not None and os.path.exists("%s_u.npy"%self.fname):
                    u_mat = np.load("%s_u.npy"%self.fname)
                    log.info("OO-MP2: rotate t2.")
                    t2 = transform_t2_to_bo(t2, u_mat)
            else:
                t2 = None
            
            e_corr, t2 = self.cisolver.kernel(t2=t2)
            e_tot = self.cisolver.e_tot
            if self.restart:
                self.fcivec = t2
            return e_tot, t2
        else:
            raise NotImplementedError

    def make_rdm1(self, t2, norb, nelec):
        return self.cisolver.make_rdm1(t2)

    def make_rdm12(self, t2, norb, nelec):
        dm1 = self.cisolver.make_rdm1(t2)
        dm2 = self.cisolver.make_rdm2(t2)
        return dm1, dm2
        
    def load_fcivec(self, fname):
        log.debug(1, "MP2 solver: read previous t.")
        if not os.path.isfile(fname):
            log.info("MP2 solver: read previous t failed, "
                     "file %s does not exist.", fname)
            return None
        fcc = h5py.File(fname, 'r')
        if "t2" in fcc.keys():
            t2 = np.asarray(fcc['t2'])
        else:
            spin = 2
            t2 = [np.asarray(fcc['t2_%s'%s]) for s in range(spin*(spin+1)//2)]
        fcc.close()
        return t2

    def save_fcivec(self, fname):
        log.debug(1, "MP2 solver: dump t2.")
        fcc = h5py.File(fname, 'w')
        if isinstance(self.fcivec, np.ndarray) and self.fcivec.ndim == 4:
            fcc['t2'] = np.asarray(self.fcivec)
        else:
            spin = 2
            for s in range(spin*(spin+1)//2):
                fcc['t2_%s'%s] = np.asarray(self.fcivec[s])
        fcc.close()

if __name__ == "__main__":
    from libdmet.solver.scf import SCF
    from libdmet.utils import logger as log
    
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

    # UHF
    myscf.set_system(8, 0, False, False)
    myscf.set_integral(8, 0, {"cd": Int1e}, \
            {"ccdd": Int2e})
    _, rhoHF = myscf.HF(MaxIter=100, tol=1e-8, \
        InitGuess = (
            np.diag([1,0,1,0,1,0,1,0.0]),
            np.diag([0,1,0,1,0,1,0,1.0])
        ))
    log.result("HF density matrix:\n%s\n%s", rhoHF[0], rhoHF[1])
    E_MP2, rdm1_mp2 = myscf.MP2()
    
