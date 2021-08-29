#! /usr/bin/env python

"""
MP solver for impurity problem:
    UIMP2
    GGMP2

Author:
    Zhihao Cui <zhcui0408@gmail.com>
"""

import time
import numpy as np

from pyscf import ao2mo, mp
from pyscf import lib
from pyscf.lib import logger
from pyscf.mp.ump2 import _ChemistsERIs
from pyscf.mp.gmp2 import _PhysicistsERIs 

from libdmet.system import integral

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
    # ZHC NOTE the memory requirement.
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

if __name__ == "__main__":
    from libdmet.utils import logger as log 
    from libdmet.solver.scf import SCF
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
     
