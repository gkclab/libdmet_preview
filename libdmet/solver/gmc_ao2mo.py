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

'''
MO integrals for GCASSCF methods
'''

import numpy as np
from pyscf import lib
from pyscf import ao2mo
from pyscf.ao2mo import _ao2mo

def trans_e1_incore(eri_ao, mo, ncore, ncas):
    nmo = mo.shape[-1]
    nocc = ncore + ncas
    nvir = nmo - nocc

    AAPP = AApp = APPA = IAPCV = APcv = None
    jC_PP = jC_pp = kC_PP = ICVCV = None
    
    if ncore == 0 and nvir == 0:
        # ZHC NOTE special optimization for OO-MP2 and OO-CCD case.
        eria = ao2mo.kernel(eri_ao, mo, compact=True)
        aapp = ao2mo.restore(1, eria, nmo)
        eria = None
        appa = aapp
        Iapcv = np.zeros((ncas, nmo, ncore, nmo-ncore))
        aaPP = None
        apPA = None
        apCV = None

        jc_pp = np.zeros((ncore, nmo, nmo))
        jc_PP = None
        kc_pp = np.zeros((ncore,nmo,nmo))
        Icvcv = np.zeros((ncore,nmo-ncore,ncore,nmo-ncore))
        cvCV = None
    else:
        eria = ao2mo.incore.half_e1(eri_ao, (mo[:, :nocc], mo), compact=False)
        load_buf = lambda bufid: eria[bufid*nmo:bufid*nmo+nmo]#.copy()
        aapp, aaPP, appa, apPA, Iapcv, apCV = \
                _trans_aapp_(mo, ncore, ncas, load_buf)
        jc_pp, jc_PP, kc_pp, Icvcv, cvCV = \
                _trans_cvcv_(mo, ncore, ncas, load_buf)

    jkcpp = jc_pp - kc_pp
    jkcPP = None
    return jkcpp, jkcPP, jC_pp, jc_PP, \
            aapp, aaPP, AApp, AAPP, appa, apPA, APPA, \
            Iapcv, IAPCV, apCV, APcv, Icvcv, ICVCV, cvCV

def _trans_aapp_(mo, ncore, ncas, fload, ao_loc=None):
    nmo = mo.shape[-1]
    nocc = ncore + ncas
    klshape = (0, nmo, 0, nmo)
    
    aaPP = None
    apPA = None
    apCV = None
    japcv = np.empty((ncas, nmo, ncore, nmo-ncore))
    aapp = np.empty((ncas, ncas, nmo, nmo))
    appa = np.empty((ncas, nmo, nmo, ncas))
    ppp = np.empty((nmo,nmo,nmo))
    for i in range(ncas):
        buf = _ao2mo.nr_e2(fload(ncore+i), mo, klshape,
                           aosym='s4', mosym='s2', ao_loc=ao_loc)
        lib.unpack_tril(buf, out=ppp)
        aapp[i] = ppp[ncore:nocc]
        appa[i] = ppp[:,:,ncore:nocc]
        japcv[i] = ppp[:,:ncore,ncore:] * 2 \
                 - ppp[:ncore,:,ncore:].transpose(1,0,2) \
                 - ppp[ncore:,:ncore,:].transpose(2,1,0)
        
    return aapp, aaPP, appa, apPA, japcv, apCV

def _trans_cvcv_(mo, ncore, ncas, fload, ao_loc=None):
    nmo = mo.shape[-1]
    jc_pp = np.empty((ncore, nmo, nmo))
    jc_PP = None
    kc_pp = np.empty((ncore,nmo,nmo))
    jcvcv = np.zeros((ncore,nmo-ncore,ncore,nmo-ncore))
    cvCV = None
    vcp = np.empty((nmo-ncore,ncore,nmo))
    cpp = np.empty((ncore,nmo,nmo))
    for i in range(ncore):
        buf = np.array(fload(i), copy=True)

        klshape = (0, ncore, 0, nmo)
        _ao2mo.nr_e2(buf[ncore:nmo], mo, klshape,
                      aosym='s4', mosym='s1', out=vcp, ao_loc=ao_loc)
        kc_pp[i,ncore:] = vcp[:,i]

        klshape = (0, nmo, 0, nmo)
        _ao2mo.nr_e2(buf[:ncore], mo, klshape,
                     aosym='s4', mosym='s2', out=buf[:ncore],
                     ao_loc=ao_loc)
        lib.unpack_tril(buf[:ncore], out=cpp)
        jc_pp[i] = cpp[i]
        kc_pp[i,:ncore] = cpp[:,i]

        jcvcv[i] = vcp[:,:,ncore:] * 2 \
                 - vcp[:,:,ncore:].transpose(2,1,0) \
                 - cpp[:,ncore:,ncore:].transpose(1,0,2)

    return jc_pp, jc_PP, kc_pp, jcvcv, cvCV

class _ERIS(object):
    def __init__(self, casscf, mo, method='incore'):
        mol = casscf.mol
        ncore = self.ncore = casscf.ncore
        ncas  = self.ncas  = casscf.ncas
        
        ncore = ncore * 2
        ncas  = ncas * 2

        nmo = mo.shape[-1]
        mem_incore, mem_outcore, mem_basic = _mem_usage(ncore, ncas, nmo)
        mem_now = lib.current_memory()[0]

        eri = casscf._scf._eri
        if (method == 'incore' and eri is not None and
            ((mem_incore+mem_now) < casscf.max_memory*.9) or
            mol.incore_anyway):
            if eri is None:
                eri = mol.intor('int2e', aosym='s8')
            (self.jkcpp, self.jkcPP, self.jC_pp, self.jc_PP,
             self.aapp, self.aaPP, self.AApp, self.AAPP,
             self.appa, self.apPA, self.APPA,
             self.Iapcv, self.IAPCV, self.apCV, self.APcv,
             self.Icvcv, self.ICVCV, self.cvCV) = \
                    trans_e1_incore(eri, mo, ncore, ncas)
            self.vhf_c = np.einsum('ipq->pq', self.jkcpp, optimize=True)
        else:
            raise NotImplementedError

def _mem_usage(ncore, ncas, nmo):
    outcore = basic = ncas**2*nmo**2*2 * 8/1e6
    incore = outcore + (ncore+ncas)*nmo**3*4/1e6
    return incore, outcore, basic

if __name__ == '__main__':
    pass
