#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
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
#
# Author: Zhi-Hao Cui <zhcui0408@gmail.com>
#         Qiming Sun <osirpt.sun@gmail.com>
#         Xing Zhang
#

'''
GCASSCF (CASSCF with genralized spin orbitals)
1-step optimization algorithm
'''

import sys

import copy
from functools import reduce, partial
import numpy as np
from scipy import linalg as la
import pyscf.gto
import pyscf.scf
from pyscf import lib
from pyscf.lib import logger
from pyscf.mcscf.mc1step import expmat, rotate_orb_cc
from pyscf.mcscf import chkfile
from pyscf import __config__

from libdmet.solver import gcasci
from libdmet.solver import gmc_ao2mo

einsum = partial(np.einsum, optimize=True)


def gen_g_hop_vo(casscf, mo, u, casdm1s, casdm2s, eris):
    ncas = casscf.ncas * 2
    ncore = casscf.ncore * 2
    nocc = ncas + ncore
    nmo = casscf.mo_coeff.shape[-1]
    if ncore > 0 or nmo > ncas:
        return gen_g_hop_big_cas(casscf, mo, u, casdm1s, casdm2s, eris)

    nocc_cas = sum(casscf.nelecas)
    nvir = nmo - nocc_cas
    dm1 = casdm1s
    ################# gradient #################
    hcore = casscf.get_hcore()
    h1e_mo = reduce(np.dot, (mo.T, hcore, mo))

    g = np.dot(h1e_mo, dm1)
    g += lib.einsum("tuvw,vwuq->qt", casdm2s, eris.aapp)

    def gorb_update(u, fcivec):
        r0 = casscf.pack_uniq_var(u)
        return g_orb + h_op(r0)

    ############## hessian, diagonal ###########
    # part1
    #hdm2apap_vovo  = einsum('pijq,pabq->aibj', casdm2s[:,:nocc_cas,:nocc_cas], eris.appa[:,nocc_cas:,nocc_cas:])
    #hdm2apap_vovo -= einsum('pibq,pajq->aibj', casdm2s[:,:nocc_cas,nocc_cas:], eris.appa[:,nocc_cas:,:nocc_cas])
    #hdm2apap_vovo += einsum('ipjq,pabq->aibj', casdm2s[:nocc_cas,:,:nocc_cas], eris.appa[:,nocc_cas:,nocc_cas:])
    #hdm2apap_vovo -= einsum('ipbq,pajq->aibj', casdm2s[:nocc_cas,:,nocc_cas:], eris.appa[:,nocc_cas:,:nocc_cas])

    #hdm2apap_vovo += einsum('pabq,pijq->aibj', casdm2s[:,nocc_cas:,nocc_cas:], eris.appa[:,:nocc_cas,:nocc_cas])
    #hdm2apap_vovo -= einsum('pajq,pibq->aibj', casdm2s[:,nocc_cas:,:nocc_cas], eris.appa[:,:nocc_cas,nocc_cas:])
    #hdm2apap_vovo += einsum('apbq,pijq->aibj', casdm2s[nocc_cas:,:,nocc_cas:], eris.appa[:,:nocc_cas,:nocc_cas])
    #hdm2apap_vovo -= einsum('apjq,pibq->aibj', casdm2s[nocc_cas:,:,:nocc_cas], eris.appa[:,:nocc_cas,nocc_cas:])

    #hdm2_vovo  = einsum('abpq,pqij->aibj', casdm2s[nocc_cas:,nocc_cas:], eris.aapp[:,:,:nocc_cas,:nocc_cas])
    #hdm2_vovo -= einsum('ajpq,pqib->aibj', casdm2s[nocc_cas:,:nocc_cas], eris.aapp[:,:,:nocc_cas,nocc_cas:])
    #hdm2_vovo += einsum('ijpq,pqab->aibj', casdm2s[:nocc_cas,:nocc_cas], eris.aapp[:,:,nocc_cas:,nocc_cas:])
    #hdm2_vovo -= einsum('ibpq,pqaj->aibj', casdm2s[:nocc_cas,nocc_cas:], eris.aapp[:,:,nocc_cas:,:nocc_cas])

    #hess_vovo = hdm2_vovo + hdm2apap_vovo

    # part7
    # h_diag[0] ~ alpha-alpha
    h_diag = einsum('ii,jj->ij', h1e_mo, dm1) - h1e_mo * dm1
    h_diag = h_diag + h_diag.T

    # part8
    g_diag = g.diagonal()
    h_diag -= g_diag + g_diag.reshape(-1,1)

    # part5 and part6 diag
    v_diag_vo  = lib.einsum('aapq,pqii->ai', casdm2s[nocc_cas:,nocc_cas:], eris.aapp[:,:,:nocc_cas,:nocc_cas])
    v_diag_vo += lib.einsum('paaq,piiq->ai', casdm2s[:,nocc_cas:,nocc_cas:], eris.appa[:,:nocc_cas,:nocc_cas])
    v_diag_vo += lib.einsum('apaq,piiq->ai', casdm2s[nocc_cas:,:,nocc_cas:], eris.appa[:,:nocc_cas,:nocc_cas])

    v_diag_ov  = lib.einsum('iipq,pqaa->ia', casdm2s[:nocc_cas,:nocc_cas], eris.aapp[:,:,nocc_cas:,nocc_cas:])
    v_diag_ov += lib.einsum('piiq,paaq->ia', casdm2s[:,:nocc_cas,:nocc_cas], eris.appa[:,nocc_cas:,nocc_cas:])
    v_diag_ov += lib.einsum('ipiq,paaq->ia', casdm2s[:nocc_cas,:,:nocc_cas], eris.appa[:,nocc_cas:,nocc_cas:]) 

    h_diag[nocc_cas:,:nocc_cas] += v_diag_vo + v_diag_ov.T

    g_orb = casscf.pack_uniq_var(g-g.T)
    h_diag = casscf.pack_uniq_var(h_diag)

    def h_op(x):
        x1a = casscf.unpack_uniq_var(x)
        x1a_vo = x1a[nocc_cas:, :nocc_cas]

        # NOTE x2a has shape (nvir, nocc_cas)
        x2a = np.zeros((nvir, nocc_cas))
        # part7
        #x2a = reduce(np.dot, (h1e_mo, x1a, dm1))
        x2a += einsum('ap,pq,qi->ai', h1e_mo[nocc_cas:], x1a, dm1[:,:nocc_cas])
        x2a -= einsum('ip,pq,qa->ai', h1e_mo[:nocc_cas], x1a, dm1[:,nocc_cas:])

        #x2a -= np.dot(g.T, x1a)
        x2a -= np.dot(g[nocc_cas:, nocc_cas:].T, x1a_vo)
        x2a -= np.dot(x1a_vo, g[:nocc_cas, :nocc_cas])

        # part1
        #x2a += einsum('aibj,bj->ai', hess_vovo, x1a_vo)

        tmp  = lib.einsum('pabq,bj->pajq', eris.appa[:,nocc_cas:,nocc_cas:], x1a_vo)
        x2a += lib.einsum('pijq,pajq->ai', casdm2s[:,:nocc_cas,:nocc_cas], tmp)
        x2a += lib.einsum('ipjq,pajq->ai', casdm2s[:nocc_cas,:,:nocc_cas], tmp)
 
        tmp  = lib.einsum('pajq,bj->pabq', eris.appa[:,nocc_cas:,:nocc_cas], x1a_vo)
        x2a -= lib.einsum('pibq,pabq->ai', casdm2s[:,:nocc_cas,nocc_cas:], tmp)
        x2a -= lib.einsum('ipbq,pabq->ai', casdm2s[:nocc_cas,:,nocc_cas:], tmp)

        tmp  = lib.einsum('pijq,bj->pibq', eris.appa[:,:nocc_cas,:nocc_cas], x1a_vo)
        x2a += lib.einsum('pabq,pibq->ai', casdm2s[:,nocc_cas:,nocc_cas:], tmp)
        x2a += lib.einsum('apbq,pibq->ai', casdm2s[nocc_cas:,:,nocc_cas:], tmp)

        tmp  = lib.einsum('pibq,bj->pijq', eris.appa[:,:nocc_cas,nocc_cas:], x1a_vo)
        x2a -= lib.einsum('pajq,pijq->ai', casdm2s[:,nocc_cas:,:nocc_cas], tmp)
        x2a -= lib.einsum('apjq,pijq->ai', casdm2s[nocc_cas:,:,:nocc_cas], tmp)

        tmp  = lib.einsum('pqij,bj->pqib', eris.aapp[:,:,:nocc_cas,:nocc_cas], x1a_vo)
        x2a += lib.einsum('abpq,pqib->ai', casdm2s[nocc_cas:,nocc_cas:], tmp)
        tmp  = lib.einsum('pqib,bj->pqij', eris.aapp[:,:,:nocc_cas,nocc_cas:], x1a_vo)
        x2a -= lib.einsum('ajpq,pqij->ai', casdm2s[nocc_cas:,:nocc_cas], tmp)
        tmp  = lib.einsum('pqab,bj->pqaj', eris.aapp[:,:,nocc_cas:,nocc_cas:], x1a_vo)
        x2a += lib.einsum('ijpq,pqaj->ai', casdm2s[:nocc_cas,:nocc_cas], tmp)
        tmp  = lib.einsum('pqaj,bj->pqab', eris.aapp[:,:,nocc_cas:,:nocc_cas], x1a_vo)
        x2a -= lib.einsum('ibpq,pqab->ai', casdm2s[:nocc_cas,nocc_cas:], tmp)

        return x2a.ravel()
    return g_orb, gorb_update, h_op, h_diag


def gen_g_hop(casscf, mo, u, casdm1s, casdm2s, eris):
    """
    Gradients, hessian operator and hessian diagonal.
    """
    ncas = casscf.ncas * 2
    ncore = casscf.ncore * 2
    nocc = ncas + ncore
    nmo = casscf.mo_coeff.shape[-1]

    dm1 = np.zeros((nmo, nmo))
    idx = np.arange(ncore)
    dm1[idx, idx] = 1
    dm1[ncore:nocc, ncore:nocc] = casdm1s

    # part2, part3
    vhf_c = eris.vhf_c
    vhf_ca = (vhf_c +
              lib.einsum('uvpq,uv->pq', eris.aapp, casdm1s) -
              lib.einsum('upqv,uv->pq', eris.appa, casdm1s))

    ################# gradient #################
    hdm2 = einsum('tuvw,vwpq->tupq', casdm2s, eris.aapp)

    hcore = casscf.get_hcore()
    h1e_mo = reduce(np.dot, (mo.T, hcore, mo))
    g = np.dot(h1e_mo, dm1)

    g[:, :ncore] += vhf_ca[:, :ncore]
    g[:, ncore:nocc] += \
            einsum('vuuq->qv', hdm2[:,:,ncore:nocc]) + \
            np.dot(vhf_c[:,ncore:nocc], casdm1s)

    def gorb_update(u, fcivec):
        r0 = casscf.pack_uniq_var(u)
        return g_orb + h_op(r0)

    ############## hessian, diagonal ###########
    # part1
    tmp = casdm2s.transpose(1,2,0,3) + casdm2s.transpose(0,2,1,3)
    hdm2apap = einsum('uvtw,tpqw->upvq', tmp, eris.appa)
    hdm2apap += hdm2.transpose(0,2,1,3)
    hdm2 = hdm2apap

    # part7
    # h_diag[0] ~ alpha-alpha
    h_diag = einsum('ii,jj->ij', h1e_mo, dm1) - h1e_mo * dm1
    h_diag = h_diag + h_diag.T

    # part8
    idx = np.arange(nmo)
    g_diag = g.diagonal()
    h_diag -= g_diag + g_diag.reshape(-1,1)
    h_diag[idx,idx] += g_diag * 2

    # part2, part3
    v_diag = vhf_ca.diagonal() # (pr|kl) * e(sq,lk)
    h_diag[:,:ncore] += v_diag.reshape(-1,1)
    h_diag[:ncore] += v_diag
    idx = np.arange(ncore)
    # (V_{qr} delta_{ps} + V_{ps} delta_{qr}) delta_{pr} delta_{sq}
    h_diag[idx,idx] -= v_diag[:ncore] * 2

    # V_{pr} e_{sq}
    tmp = einsum('ii,jj->ij', vhf_c, casdm1s)
    h_diag[:,ncore:nocc] += tmp
    h_diag[ncore:nocc,:] += tmp.T
    tmp = -vhf_c[ncore:nocc,ncore:nocc] * casdm1s
    h_diag[ncore:nocc,ncore:nocc] += tmp + tmp.T

    # part4
    # (qp|rs)-(pr|sq) rp in core
    tmp = -einsum('cpp->cp', eris.jkcpp)
    # (qp|sr) - (qr|sp) rp in core => 0
    #h_diag[:ncore,:] += tmp
    #h_diag[:,:ncore] += tmp.T
    #h_diag[:ncore,:ncore] -= tmp[:,:ncore] * 2
    h_diag[:ncore, ncore:] += tmp[:, ncore:]
    h_diag[ncore:, :ncore] += tmp[:, ncore:].T

    # part5 and part6 diag
    #+(qr|kp) e_s^k  p in core, sk in active
    #+(qr|sl) e_l^p  s in core, pl in active
    #-(qj|sr) e_j^p  s in core, jp in active
    #-(qp|kr) e_s^k  p in core, sk in active
    #+(qj|rs) e_j^p  s in core, jp in active
    #+(qp|rl) e_l^s  p in core, ls in active
    #-(qs|rl) e_l^p  s in core, lp in active
    #-(qj|rp) e_j^s  p in core, js in active
    jkcaa = eris.jkcpp[:,ncore:nocc,ncore:nocc]
    tmp = 2 * einsum('jik,ik->ji', jkcaa, casdm1s)
    h_diag[:ncore,ncore:nocc] += tmp
    h_diag[ncore:nocc,:ncore] += tmp.T

    v_diag = einsum('ijij->ij', hdm2)
    h_diag[ncore:nocc,:] += v_diag
    h_diag[:,ncore:nocc] += v_diag.T

    g_orb = casscf.pack_uniq_var(g-g.T)
    h_diag = casscf.pack_uniq_var(h_diag)

    def h_op(x):
        x1a = casscf.unpack_uniq_var(x)
        xa_cu = x1a[:ncore,ncore:]
        xa_av = x1a[ncore:nocc,nocc:]
        xa_ac = x1a[ncore:nocc,:ncore]

        # part7
        x2a = reduce(np.dot, (h1e_mo, x1a, dm1))
        # part8, the hessian gives
        #x2a -= np.dot(g[0], x1a)
        #x2b -= np.dot(g[1], x1b)
        # it may ruin the hermitian of hessian unless g == g.T. So symmetrize it
        # x_{pq} -= g_{pr} \delta_{qs} x_{rs} * .5
        # x_{rs} -= g_{rp} \delta_{sq} x_{pq} * .5
        x2a -= np.dot(g.T, x1a)
        # part2
        x2a[:ncore] += np.dot(xa_cu, vhf_ca[ncore:])

        # part3
        x2a[ncore:nocc] += reduce(np.dot, (casdm1s, xa_av, vhf_c[nocc:])) + \
                           reduce(np.dot, (casdm1s, xa_ac, vhf_c[:ncore]))

        # part1
        x2a[ncore:nocc] += einsum('upvr,vr->up', hdm2apap, x1a[ncore:nocc])

        # part4, part5, part6
        if ncore > 0:
            va, vc = casscf.update_jk_in_ah(mo, x1a, casdm1s, eris)
            x2a[ncore:nocc] += va
            x2a[:ncore,ncore:] += vc

        x2a = x2a - x2a.T
        return casscf.pack_uniq_var(x2a)
    return g_orb, gorb_update, h_op, h_diag

def gen_g_hop_big_cas(casscf, mo, u, casdm1s, casdm2s, eris):
    """
    Gradients, hessian operator and hessian diagonal, for big CAS space.
    """
    ncas = casscf.ncas * 2
    ncore = casscf.ncore * 2
    nocc = ncas + ncore
    nmo = casscf.mo_coeff.shape[-1]

    dm1 = np.zeros((nmo, nmo))
    idx = np.arange(ncore)
    dm1[idx, idx] = 1
    dm1[ncore:nocc, ncore:nocc] = casdm1s

    # part2, part3
    vhf_c = eris.vhf_c
    vhf_ca = (vhf_c +
              lib.einsum('uvpq,uv->pq', eris.aapp, casdm1s) -
              lib.einsum('upqv,uv->pq', eris.appa, casdm1s))

    ################# gradient #################
    # ZHC NOTE
    #hdm2 = einsum('tuvw,vwpq->tupq', casdm2s, eris.aapp)
    # ZHC NOTE

    hcore = casscf.get_hcore()
    h1e_mo = reduce(np.dot, (mo.T, hcore, mo))
    g = np.dot(h1e_mo, dm1)

    g[:, :ncore] += vhf_ca[:, :ncore]
    
    # ZHC NOTE
    #g[:, ncore:nocc] += \
    #        einsum('vuuq->qv', hdm2[:,:,ncore:nocc]) + \
    #        np.dot(vhf_c[:,ncore:nocc], casdm1s)
    g[:, ncore:nocc] += np.dot(vhf_c[:,ncore:nocc], casdm1s)
    # O(N5)
    g[:, ncore:nocc] += lib.einsum('tuvw, vwuq -> qt', casdm2s,
                                   eris.aapp[:, :, ncore:nocc])
    # ZHC NOTE

    def gorb_update(u, fcivec):
        r0 = casscf.pack_uniq_var(u)
        return g_orb + h_op(r0)

    ############## hessian, diagonal ###########
    # part1
    # ZHC NOTE
    #tmp = casdm2s.transpose(1,2,0,3) + casdm2s.transpose(0,2,1,3)
    #hdm2apap = einsum('uvtw,tpqw->upvq', tmp, eris.appa)
    #hdm2apap += hdm2.transpose(0,2,1,3)
    #hdm2 = hdm2apap
    # ZHC NOTE

    # part7
    # h_diag[0] ~ alpha-alpha
    h_diag = einsum('ii,jj->ij', h1e_mo, dm1) - h1e_mo * dm1
    h_diag = h_diag + h_diag.T

    # part8
    idx = np.arange(nmo)
    g_diag = g.diagonal()
    h_diag -= g_diag + g_diag.reshape(-1,1)
    h_diag[idx,idx] += g_diag * 2

    # part2, part3
    v_diag = vhf_ca.diagonal() # (pr|kl) * e(sq,lk)
    h_diag[:,:ncore] += v_diag.reshape(-1,1)
    h_diag[:ncore] += v_diag
    idx = np.arange(ncore)
    # (V_{qr} delta_{ps} + V_{ps} delta_{qr}) delta_{pr} delta_{sq}
    h_diag[idx,idx] -= v_diag[:ncore] * 2

    # V_{pr} e_{sq}
    tmp = einsum('ii,jj->ij', vhf_c, casdm1s)
    h_diag[:,ncore:nocc] += tmp
    h_diag[ncore:nocc,:] += tmp.T
    tmp = -vhf_c[ncore:nocc,ncore:nocc] * casdm1s
    h_diag[ncore:nocc,ncore:nocc] += tmp + tmp.T

    # part4
    # (qp|rs)-(pr|sq) rp in core
    tmp = -einsum('cpp->cp', eris.jkcpp)
    # (qp|sr) - (qr|sp) rp in core => 0
    #h_diag[:ncore,:] += tmp
    #h_diag[:,:ncore] += tmp.T
    #h_diag[:ncore,:ncore] -= tmp[:,:ncore] * 2
    h_diag[:ncore, ncore:] += tmp[:, ncore:]
    h_diag[ncore:, :ncore] += tmp[:, ncore:].T

    # part5 and part6 diag
    #+(qr|kp) e_s^k  p in core, sk in active
    #+(qr|sl) e_l^p  s in core, pl in active
    #-(qj|sr) e_j^p  s in core, jp in active
    #-(qp|kr) e_s^k  p in core, sk in active
    #+(qj|rs) e_j^p  s in core, jp in active
    #+(qp|rl) e_l^s  p in core, ls in active
    #-(qs|rl) e_l^p  s in core, lp in active
    #-(qj|rp) e_j^s  p in core, js in active
    jkcaa = eris.jkcpp[:,ncore:nocc,ncore:nocc]
    tmp = 2 * einsum('jik,ik->ji', jkcaa, casdm1s)
    h_diag[:ncore,ncore:nocc] += tmp
    h_diag[ncore:nocc,:ncore] += tmp.T

    # ZHC NOTE
    #v_diag = einsum('ijij->ij', hdm2)
    # O(N4)
    #v_diag = lib.einsum('uutw, tppw -> up', tmp_dm2, eris.appa) + \
    #         lib.einsum('ttvw, vwpp -> tp', casdm2s, eris.aapp)
    v_diag = lib.einsum('tuuw, tppw -> up', casdm2s, eris.appa) + \
             lib.einsum('utuw, tppw -> up', casdm2s, eris.appa) + \
             lib.einsum('uutw, twpp -> up', casdm2s, eris.aapp)
    # ZHC NOTE

    h_diag[ncore:nocc,:] += v_diag
    h_diag[:,ncore:nocc] += v_diag.T

    g_orb = casscf.pack_uniq_var(g-g.T)
    h_diag = casscf.pack_uniq_var(h_diag)
    
    def h_op(x):
        x1a = casscf.unpack_uniq_var(x)
        xa_cu = x1a[:ncore,ncore:]
        xa_av = x1a[ncore:nocc,nocc:]
        xa_ac = x1a[ncore:nocc,:ncore]

        # part7
        x2a = reduce(np.dot, (h1e_mo, x1a, dm1))
        # part8, the hessian gives
        #x2a -= np.dot(g[0], x1a)
        #x2b -= np.dot(g[1], x1b)
        # it may ruin the hermitian of hessian unless g == g.T. So symmetrize it
        # x_{pq} -= g_{pr} \delta_{qs} x_{rs} * .5
        # x_{rs} -= g_{rp} \delta_{sq} x_{pq} * .5
        x2a -= np.dot(g.T, x1a)
        # part2
        x2a[:ncore] += np.dot(xa_cu, vhf_ca[ncore:])

        # part3
        x2a[ncore:nocc] += reduce(np.dot, (casdm1s, xa_av, vhf_c[nocc:])) + \
                           reduce(np.dot, (casdm1s, xa_ac, vhf_c[:ncore]))
        
        # part1
        # ZHC NOTE
        #x2a[ncore:nocc] += einsum('upvr,vr->up', hdm2apap, x1a[ncore:nocc])
        # O (N5)
        tmp = lib.einsum('tpqw, vq -> tpvw', eris.appa, x1a[ncore:nocc])
        #x2a[ncore:nocc] += lib.einsum('uvtw, tpvw -> up', tmp_dm2, tmp)
        x2a[ncore:nocc] += lib.einsum('tuvw, tpvw -> up', casdm2s, tmp)
        x2a[ncore:nocc] += lib.einsum('utvw, tpvw -> up', casdm2s, tmp)
        tmp = None
        tmp = lib.einsum('twpq, vq -> twpv', eris.aapp, x1a[ncore:nocc])
        x2a[ncore:nocc] += lib.einsum('uvtw, twpv -> up', casdm2s, tmp)
        tmp = None
        # ZHC NOTE

        # part4, part5, part6
        if ncore > 0:
            va, vc = casscf.update_jk_in_ah(mo, x1a, casdm1s, eris)
            x2a[ncore:nocc] += va
            x2a[:ncore,ncore:] += vc

        x2a = x2a - x2a.T
        return casscf.pack_uniq_var(x2a)
    return g_orb, gorb_update, h_op, h_diag

def kernel(casscf, mo_coeff, tol=1e-7, conv_tol_grad=None,
           ci0=None, callback=None, verbose=logger.NOTE, dump_chk=True):
    '''quasi-newton CASSCF optimization driver
    '''
    log = logger.new_logger(casscf, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start 1-step CASSCF')
    if callback is None:
        callback = casscf.callback

    mo = mo_coeff
    nmo = mo_coeff.shape[-1]
    ncore = casscf.ncore
    ncas = casscf.ncas
    nocc = ncore + ncas

    eris = casscf.ao2mo(mo)
    e_tot, e_cas, fcivec = casscf.casci(mo, ci0, eris, log, locals())

    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(tol)
        logger.info(casscf, 'Set conv_tol_grad to %g', conv_tol_grad)
    conv_tol_ddm = conv_tol_grad * 3
    conv = False
    totmicro = totinner = 0
    norm_gorb = norm_gci = -1
    de, elast = e_tot, e_tot
    r0 = None

    t1m = log.timer('Initializing 1-step CASSCF', *cput0)
    casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, ncas*2, casscf.nelecas)
    norm_ddm = 1e2
    casdm1_prev = casdm1_last = casdm1
    t3m = t2m = log.timer('CAS DM', *t1m)
    imacro = 0
    dr0 = None
    while not conv and imacro < casscf.max_cycle_macro:
        imacro += 1
        max_cycle_micro = casscf.micro_cycle_scheduler(locals())
        max_stepsize = casscf.max_stepsize_scheduler(locals())
        imicro = 0
        rota = casscf.rotate_orb_cc(mo, lambda:fcivec, lambda:casdm1, lambda:casdm2,
                                    eris, r0, conv_tol_grad*.3, max_stepsize, log)
        for u, g_orb, njk, r0 in rota:
            imicro += 1
            norm_gorb = np.linalg.norm(g_orb)
            if imicro == 1:
                norm_gorb0 = norm_gorb
            norm_t = np.linalg.norm(u-np.eye(nmo))
            t3m = log.timer('orbital rotation', *t3m)
            if imicro >= max_cycle_micro:
                log.debug('micro %d  |u-1|=%5.3g  |g[o]|=%5.3g',
                          imicro, norm_t, norm_gorb)
                break

            casdm1, casdm2, gci, fcivec = \
                    casscf.update_casdm(mo, u, fcivec, e_cas, eris)
            norm_ddm = np.linalg.norm(casdm1 - casdm1_last)
            norm_ddm_micro = np.linalg.norm(casdm1 - casdm1_prev)
            casdm1_prev = casdm1
            t3m = log.timer('update CAS DM', *t3m)
            if isinstance(gci, np.ndarray):
                norm_gci = np.linalg.norm(gci)
                log.debug('micro %d  |u-1|=%5.3g  |g[o]|=%5.3g  |g[c]|=%5.3g  |ddm|=%5.3g',
                          imicro, norm_t, norm_gorb, norm_gci, norm_ddm)
            else:
                norm_gci = None
                log.debug('micro %d  |u-1|=%5.3g  |g[o]|=%5.3g  |g[c]|=%s  |ddm|=%5.3g',
                          imicro, norm_t, norm_gorb, norm_gci, norm_ddm)

            if callable(callback):
                callback(locals())

            t3m = log.timer('micro iter %d'%imicro, *t3m)
            if (norm_t < conv_tol_grad or
                (norm_gorb < conv_tol_grad*.5 and
                 (norm_ddm < conv_tol_ddm*.4 or norm_ddm_micro < conv_tol_ddm*.4))):
                break

        rota.close()
        rota = None

        totmicro += imicro
        totinner += njk

        eris = None
        # keep u, g_orb in locals() so that they can be accessed by callback
        u = u.copy()
        g_orb = g_orb.copy()
        mo = casscf.rotate_mo(mo, u, log)
        eris = casscf.ao2mo(mo)
        t2m = log.timer('update eri', *t3m)

        e_tot, e_cas, fcivec = casscf.casci(mo, fcivec, eris, log, locals())
        casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, ncas*2, casscf.nelecas)
        norm_ddm = np.linalg.norm(casdm1 - casdm1_last)
        casdm1_prev = casdm1_last = casdm1
        log.timer('CASCI solver', *t2m)
        t3m = t2m = t1m = log.timer('macro iter %d'%imacro, *t1m)

        de, elast = e_tot - elast, e_tot
        if (abs(de) < tol
            and (norm_gorb0 < conv_tol_grad and norm_ddm < conv_tol_ddm)):
            conv = True

        if dump_chk:
            casscf.dump_chk(locals())

        if callable(callback):
            callback(locals())

    if conv:
        log.info('1-step CASSCF converged in %d macro (%d JK %d micro) steps',
                 imacro, totinner, totmicro)
    else:
        log.info('1-step CASSCF not converged, %d macro (%d JK %d micro) steps',
                 imacro, totinner, totmicro)

    if casscf.canonicalization:
        log.info('CASSCF canonicalization')
        mo, fcivec, mo_energy = \
                casscf.canonicalize(mo, fcivec, eris, casscf.sorting_mo_energy,
                                    casscf.natorb, casdm1, log)
        if casscf.natorb and dump_chk: # dump_chk may save casdm1
            occ, gcas = casscf._eig(-casdm1, ncore*2, nocc*2)
            casdm1 = np.diag(-occ)
    else:
        if casscf.natorb:
            # FIXME (pyscf-2.0): Whether to transform natural orbitals in
            # active space when this flag is enabled?
            log.warn('The attribute natorb of mcscf object affects only the '
                     'orbital canonicalization.\n'
                     'If you would like to get natural orbitals in active space '
                     'without touching core and external orbitals, an explicit '
                     'call to mc.cas_natorb_() is required')
        mo_energy = None

    if dump_chk:
        casscf.dump_chk(locals())

    log.timer('1-step CASSCF', *cput0)
    return conv, e_tot, e_cas, fcivec, mo, mo_energy

class GCASSCF(gcasci.GCASCI):
# the max orbital rotation and CI increment, prefer small step size
    max_stepsize = getattr(__config__, 'mcscf_mc1step_CASSCF_max_stepsize', .02)
    max_cycle_macro = getattr(__config__, 'mcscf_mc1step_CASSCF_max_cycle_macro', 50)
    max_cycle_micro = getattr(__config__, 'mcscf_mc1step_CASSCF_max_cycle_micro', 4)
    conv_tol = getattr(__config__, 'mcscf_mc1step_CASSCF_conv_tol', 1e-7)
    conv_tol_grad = getattr(__config__, 'mcscf_mc1step_CASSCF_conv_tol_grad', None)
    # for augmented hessian
    ah_level_shift = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_level_shift', 1e-8)
    ah_conv_tol = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_conv_tol', 1e-12)
    ah_max_cycle = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_max_cycle', 30)
    ah_lindep = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_lindep', 1e-14)
# * ah_start_tol and ah_start_cycle control the start point to use AH step.
#   In function rotate_orb_cc, the orbital rotation is carried out with the
#   approximate aug_hessian step after a few davidson updates of the AH eigen
#   problem.  Reducing ah_start_tol or increasing ah_start_cycle will delay
#   the start point of orbital rotation.
# * We can do early ah_start since it only affect the first few iterations.
#   The start tol will be reduced when approach the convergence point.
# * Be careful with the SYMMETRY BROKEN caused by ah_start_tol/ah_start_cycle.
#   ah_start_tol/ah_start_cycle actually approximates the hessian to reduce
#   the J/K evaluation required by AH.  When the system symmetry is higher
#   than the one given by mol.symmetry/mol.groupname,  symmetry broken might
#   occur due to this approximation,  e.g.  with the default ah_start_tol,
#   C2 (16o, 8e) under D2h symmetry might break the degeneracy between
#   pi_x, pi_y orbitals since pi_x, pi_y belong to different irreps.  It can
#   be fixed by increasing the accuracy of AH solver, e.g.
#               ah_start_tol = 1e-8;  ah_conv_tol = 1e-10
# * Classic AH can be simulated by setting eg
#               ah_start_tol = 1e-7
#               max_stepsize = 1.5
#               ah_grad_trust_region = 1e6
# ah_grad_trust_region allow gradients being increased in AH optimization
    ah_start_tol = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_start_tol', 2.5)
    ah_start_cycle = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_start_cycle', 3)
    ah_grad_trust_region = getattr(__config__, 'mcscf_mc1step_CASSCF_ah_grad_trust_region', 3.0)

    internal_rotation = getattr(__config__, 'mcscf_mc1step_CASSCF_internal_rotation', False)
    internal_rotation_vo = getattr(__config__, 'mcscf_mc1step_CASSCF_internal_rotation_vo', False)
    ci_response_space = getattr(__config__, 'mcscf_mc1step_CASSCF_ci_response_space', 4)
    ci_grad_trust_region = getattr(__config__, 'mcscf_mc1step_CASSCF_ci_grad_trust_region', 3.0)
    with_dep4 = getattr(__config__, 'mcscf_mc1step_CASSCF_with_dep4', False)
    chk_ci = getattr(__config__, 'mcscf_mc1step_CASSCF_chk_ci', False)
    kf_interval = getattr(__config__, 'mcscf_mc1step_CASSCF_kf_interval', 4)
    kf_trust_region = getattr(__config__, 'mcscf_mc1step_CASSCF_kf_trust_region', 3.0)

    ao2mo_level = getattr(__config__, 'mcscf_mc1step_CASSCF_ao2mo_level', 2)
    natorb = getattr(__config__, 'mcscf_mc1step_CASSCF_natorb', False)
    canonicalization = getattr(__config__, 'mcscf_mc1step_CASSCF_canonicalization', True)
    sorting_mo_energy = getattr(__config__, 'mcscf_mc1step_CASSCF_sorting_mo_energy', False)
    scale_restoration = getattr(__config__, 'mcscf_mc1step_CASSCF_scale_restoration', 0.5)

    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None, frozen=None):
        gcasci.GCASCI.__init__(self, mf_or_mol, ncas, nelecas, ncore)
        self.frozen = frozen

        self.callback = None
        self.chkfile = self._scf.chkfile

        self.fcisolver.max_cycle = getattr(__config__,
                                           'mcscf_gmc1step_GCASSCF_fcisolver_max_cycle', 50)
        self.fcisolver.conv_tol = getattr(__config__,
                                          'mcscf_gmc1step_GCASSCF_fcisolver_conv_tol', 1e-8)

##################################################
# don't modify the following attributes, they are not input options
        self.e_tot = None
        self.e_cas = None
        self.ci = None
        self.mo_coeff = self._scf.mo_coeff
        self.converged = False
        self._max_stepsize = None

        keys = set(('max_stepsize', 'max_cycle_macro', 'max_cycle_micro',
                    'conv_tol', 'conv_tol_grad', 'ah_level_shift',
                    'ah_conv_tol', 'ah_max_cycle', 'ah_lindep',
                    'ah_start_tol', 'ah_start_cycle', 'ah_grad_trust_region',
                    'internal_rotation', 'internal_rotation_vo','ci_response_space',
                    'with_dep4', 'chk_ci',
                    'kf_interval', 'kf_trust_region', 'fcisolver_max_cycle',
                    'fcisolver_conv_tol', 'natorb', 'canonicalization',
                    'sorting_mo_energy'))
        self._keys = set(self.__dict__.keys()).union(keys)
    
    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        ncore = self.ncore
        ncas = self.ncas
        nvir = self.mo_coeff.shape[-1] // 2 - ncore - ncas
        log.info('CAS (%de+%de, %do), ncore = %d, nvir = %d',
                 self.nelecas[0], self.nelecas[1], ncas, ncore, nvir)
        if self.frozen is not None:
            log.info('frozen orbitals %s', str(self.frozen))
        log.info('max_cycle_macro = %d', self.max_cycle_macro)
        log.info('max_cycle_micro = %d', self.max_cycle_micro)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_grad = %s', self.conv_tol_grad)
        log.info('orbital rotation max_stepsize = %g', self.max_stepsize)
        log.info('augmented hessian ah_max_cycle = %d', self.ah_max_cycle)
        log.info('augmented hessian ah_conv_tol = %g', self.ah_conv_tol)
        log.info('augmented hessian ah_linear dependence = %g', self.ah_lindep)
        log.info('augmented hessian ah_level shift = %g', self.ah_level_shift)
        log.info('augmented hessian ah_start_tol = %g', self.ah_start_tol)
        log.info('augmented hessian ah_start_cycle = %d', self.ah_start_cycle)
        log.info('augmented hessian ah_grad_trust_region = %g', self.ah_grad_trust_region)
        log.info('kf_trust_region = %g', self.kf_trust_region)
        log.info('kf_interval = %d', self.kf_interval)
        log.info('ci_response_space = %d', self.ci_response_space)
        log.info('ci_grad_trust_region = %d', self.ci_grad_trust_region)
        log.info('with_dep4 %d', self.with_dep4)
        log.info('natorb = %s', self.natorb)
        log.info('canonicalization = %s', self.canonicalization)
        log.info('sorting_mo_energy = %s', self.sorting_mo_energy)
        log.info('ao2mo_level = %d', self.ao2mo_level)
        log.info('chkfile = %s', self.chkfile)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        log.info('internal_rotation = %s', self.internal_rotation)
        log.info('internal_rotation_vo = %s', self.internal_rotation_vo)
        if getattr(self.fcisolver, 'dump_flags', None):
            self.fcisolver.dump_flags(self.verbose)
        if self.mo_coeff is None:
            log.error('Orbitals for CASCI are not specified. The relevant SCF '
                      'object may not be initialized.')

        if (getattr(self._scf, 'with_solvent', None) and
            not getattr(self, 'with_solvent', None)):
            log.warn('''Solvent model %s was found at SCF level but not applied to the CASSCF object.
The SCF solvent model will not be applied to the current CASSCF calculation.
To enable the solvent model for CASSCF, the following code needs to be called
        from pyscf import solvent
        mc = mcscf.CASSCF(...)
        mc = solvent.ddCOSMO(mc)
''',
                     self._scf.with_solvent.__class__)
        return self

    def kernel(self, mo_coeff=None, ci0=None, callback=None, _kern=kernel):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if callback is None: callback = self.callback

        self.check_sanity()
        self.dump_flags()

        self.converged, self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy = \
                _kern(self, mo_coeff,
                      tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
                      ci0=ci0, callback=callback, verbose=self.verbose)
        logger.note(self, 'GCASSCF energy = %.15g', self.e_tot)
        #if self.verbose >= logger.INFO:
        #    self.analyze(mo_coeff, self.ci, verbose=self.verbose)
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff, self.mo_energy

    def mc1step(self, mo_coeff=None, ci0=None, callback=None):
        return self.kernel(mo_coeff, ci0, callback)

    def mc2step(self, mo_coeff=None, ci0=None, callback=None):
        from libdmet.solver import gmc2step
        return self.kernel(mo_coeff, ci0, callback, gmc2step.kernel)

    def get_h2eff(self, mo_coeff=None):
        '''Computing active space two-particle Hamiltonian.
        '''
        return self.get_h2cas(mo_coeff)

    def get_h2cas(self, mo_coeff=None):
        return gcasci.GCASCI.ao2mo(self, mo_coeff)

    def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
        if eris is None:
            fcasci = copy.copy(self)
            fcasci.ao2mo = self.get_h2cas
        else:
            fcasci = _fake_h_for_fast_casci(self, mo_coeff, eris)

        log = logger.new_logger(self, verbose)

        e_tot, e_cas, fcivec = gcasci.kernel(fcasci, mo_coeff, ci0, log)
        
        if envs is not None and log.verbose >= logger.INFO:
            log.debug('CAS space CI energy = %.15g', e_cas)

            if 'imicro' in envs:  # Within CASSCF iteration
                log.info('macro iter %d (%d JK  %d micro), '
                         'GCASSCF E = %.15g  dE = %.8g',
                         envs['imacro'], envs['njk'], envs['imicro'],
                         e_tot, e_tot-envs['elast'])
                if 'norm_gci' in envs:
                    log.info('               |grad[o]|=%5.3g  '
                             '|grad[c]|= %s  |ddm|=%5.3g',
                             envs['norm_gorb0'],
                             envs['norm_gci'], envs['norm_ddm'])
                else:
                    log.info('               |grad[o]|=%5.3g  |ddm|=%5.3g',
                             envs['norm_gorb0'], envs['norm_ddm'])
            else:  # Initialization step
                log.info('GCASCI E = %.15g', e_tot)
        return e_tot, e_cas, fcivec
    
    def uniq_var_indices(self, nmo, ncore, ncas, frozen):
        nocc = ncore + ncas
        mask = np.zeros((nmo, nmo),dtype=bool)
        # ZHC NOTE cas-core, vir-core, vir-cas rotation
        mask[ncore*2:nocc*2, :ncore*2] = True
        mask[nocc*2:, :nocc*2] = True
        if self.internal_rotation_vo:
            # cas-v to cas-o
            nocc_cas = sum(self.nelecas) + ncore*2
            mask[nocc_cas:nocc*2, ncore*2:nocc_cas] = True
        elif self.internal_rotation:
            # cas-cas
            mask[ncore*2:nocc*2, ncore*2:nocc*2][np.tril_indices(ncas*2, -1)] = True
        if frozen is not None:
            if isinstance(frozen, (int, np.integer)):
                mask[:frozen] = mask[:, :frozen] = False
            else:
                frozen = np.asarray(frozen)
                mask[frozen] = mask[:, frozen] = False
        return mask

    def pack_uniq_var(self, mat):
        nmo = self.mo_coeff.shape[-1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        return mat[idx]

    # to anti symmetric matrix
    def unpack_uniq_var(self, v):
        nmo = self.mo_coeff.shape[-1]
        idx = self.uniq_var_indices(nmo, self.ncore, self.ncas, self.frozen)
        mat = np.zeros((nmo,nmo))
        mat[idx] = v
        return mat - mat.T

    def update_rotate_matrix(self, dx, u0=1):
        dr = self.unpack_uniq_var(dx)
        return np.dot(u0, expmat(dr))

    gen_g_hop = gen_g_hop
    rotate_orb_cc = rotate_orb_cc

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        return gmc_ao2mo._ERIS(self, mo_coeff)
    
    def update_jk_in_ah(self, mo, r, casdm1s, eris):
        ncas = self.ncas
        ncore = self.ncore
        nocc = ncas + ncore
        ra = r
        
        vhf3ca = lib.einsum('srqp, sr -> qp', eris.Icvcv, ra[:ncore*2, ncore*2:])
        vhf3aa = lib.einsum('kpsr, sr -> kp', eris.Iapcv, ra[:ncore*2, ncore*2:])

        dm4 = np.dot(casdm1s, ra[ncore*2:nocc*2])
        vhf4a = lib.einsum('krqp, kr -> qp', eris.Iapcv, dm4)

        va = np.dot(casdm1s, vhf3aa)
        vc = vhf3ca + vhf4a
        return va, vc

    def update_casdm(self, mo, u, fcivec, e_cas, eris):
        ecore, h1cas, h2cas = self.approx_cas_integral(mo, u, eris)
        ci1, g = self.solve_approx_ci(h1cas, h2cas, fcivec, ecore, e_cas)
        casdm1, casdm2 = self.fcisolver.make_rdm12(ci1, self.ncas*2, self.nelecas)
        return casdm1, casdm2, g, ci1

    def approx_cas_integral(self, mo, u, eris):
        ncas = self.ncas
        ncore = self.ncore
        nocc = ncas + ncore
        nmo = mo.shape[-1]
        rmat = u - np.eye(nmo)
        mocas = mo[:, ncore*2:nocc*2]

        hcore = self.get_hcore()
        h1effa = reduce(np.dot, (rmat[:, :nocc*2].T, mo.T, hcore, mo[:, :nocc*2]))
        h1effa = h1effa + h1effa.T

        aapc = eris.aapp[:, :, :, :ncore*2]
        apca = eris.appa[:, :, :ncore*2, :]
        jka = einsum('iup -> up', eris.jkcpp[:, :nocc*2])
        v1a = (einsum('up, pv -> uv', jka[ncore*2:], rmat[:, ncore*2:nocc*2]) +
               einsum('uvpi, pi -> uv', aapc-apca.transpose(0, 3, 1, 2), rmat[:, :ncore*2]))
        h1casa = (h1effa[ncore*2:, ncore*2:] + (v1a + v1a.T) +
                  reduce(np.dot, (mocas.T, hcore, mocas)) +
                  eris.vhf_c[ncore*2:nocc*2, ncore*2:nocc*2])
        h1cas = h1casa

        aaap = eris.aapp[:, :, ncore*2:nocc*2, :]
        aaaa = einsum('tuvp, pw -> tuvw', aaap, rmat[:, ncore*2:nocc*2])
        aaaa = aaaa + aaaa.transpose(0, 1, 3, 2)
        aaaa = aaaa + aaaa.transpose(2, 3, 0, 1)
        aaaa += aaap[:, :, :, ncore*2:nocc*2]

        # pure core response
        ecore = (h1effa[:ncore*2].trace() +
                 einsum('jp,pj->', jka[:ncore*2], rmat[:, :ncore*2])*2)

        return ecore, h1cas, aaaa
    
    def solve_approx_ci(self, h1, h2, ci0, ecore, e_cas, envs={}):
        ''' Solve CI eigenvalue/response problem approximately
        '''
        ncas = self.ncas
        nelecas = self.nelecas
        if 'norm_gorb' in envs:
            tol = max(self.conv_tol, envs['norm_gorb']**2*.1)
        else:
            tol = None
        if getattr(self.fcisolver, 'approx_kernel', None):
            fn = self.fcisolver.approx_kernel
            e, ci1 = fn(h1, h2, ncas*2, nelecas, ecore=ecore, ci0=ci0,
                        tol=tol, max_memory=self.max_memory)
            return ci1, None
        elif not (getattr(self.fcisolver, 'contract_2e', None) and
                  getattr(self.fcisolver, 'absorb_h1e', None)):
            fn = self.fcisolver.kernel
            e, ci1 = fn(h1, h2, ncas*2, nelecas, ecore=ecore, ci0=ci0,
                        tol=tol, max_memory=self.max_memory,
                        max_cycle=self.ci_response_space)
            return ci1, None

        h2eff = self.fcisolver.absorb_h1e(h1, h2, ncas*2, nelecas, .5)

        # Be careful with the symmetry adapted contract_2e function. When the
        # symmetry adapted FCI solver is used, the symmetry of ci0 may be
        # different to fcisolver.wfnsym. This function may output 0.
        if getattr(self.fcisolver, 'guess_wfnsym', None):
            wfnsym = self.fcisolver.guess_wfnsym(self.ncas*2, self.nelecas, ci0)
        else:
            wfnsym = None

        def contract_2e(c):
            if wfnsym is None:
                hc = self.fcisolver.contract_2e(h2eff, c, ncas*2, nelecas)
            else:
                with lib.temporary_env(self.fcisolver, wfnsym=wfnsym):
                    hc = self.fcisolver.contract_2e(h2eff, c, ncas*2, nelecas)
            return hc.ravel()

        hc = contract_2e(ci0)
        g = hc - (e_cas-ecore) * ci0.ravel()

        if self.ci_response_space > 7:
            logger.debug(self, 'CI step by full response')
            # full response
            max_memory = max(400, self.max_memory-lib.current_memory()[0])
            e, ci1 = self.fcisolver.kernel(h1, h2, ncas*2, nelecas, ecore=ecore,
                                           ci0=ci0, tol=tol, max_memory=max_memory)
        else:
            nd = self.ci_response_space
            xs = [ci0.ravel()]
            ax = [hc]
            heff = np.empty((nd,nd))
            seff = np.empty((nd,nd))
            heff[0,0] = np.dot(xs[0], ax[0])
            seff[0,0] = 1
            for i in range(1, nd):
                dx = ax[i-1] - xs[i-1] * e_cas
                if np.linalg.norm(dx) < 1e-6:
                    break
                xs.append(dx)
                ax.append(contract_2e(xs[i]))
                for j in range(i+1):
                    heff[i,j] = heff[j,i] = np.dot(xs[i], ax[j])
                    seff[i,j] = seff[j,i] = np.dot(xs[i], xs[j])
            nd = len(xs)
            e, v = lib.safe_eigh(heff[:nd,:nd], seff[:nd,:nd])[:2]
            ci1 = xs[0] * v[0,0]
            for i in range(1, nd):
                ci1 += xs[i] * v[i,0]
        return ci1, g

    def dump_chk(self, envs):
        if not self.chkfile:
            return self

        if getattr(self.fcisolver, 'nevpt_intermediate', None):
            civec = None
        elif self.chk_ci:
            civec = envs['fcivec']
        else:
            civec = None
        ncore = self.ncore
        nocc = ncore + self.ncas
        if 'mo' in envs:
            mo_coeff = envs['mo']
        else:
            mo_coeff = envs['mo_coeff']
        mo_occ = np.zeros(mo_coeff.shape[-1])
        mo_occ[:ncore*2] = 1
        if self.natorb:
            occ = self._eig(-envs['casdm1'], ncore*2, nocc*2)[0]
            mo_occ[ncore*2:nocc*2] = -occ
        else:
            mo_occ[ncore*2:nocc*2] = envs['casdm1'].diagonal()
# Note: mo_energy in active space =/= F_{ii}  (F is general Fock)
        if 'mo_energy' in envs:
            mo_energy = envs['mo_energy']
        else:
            mo_energy = 'None'
        chkfile.dump_mcscf(self, self.chkfile, 'mcscf', envs['e_tot'],
                           mo_coeff, ncore, self.ncas, mo_occ,
                           mo_energy, envs['e_cas'], civec, envs['casdm1'],
                           overwrite_mol=False)
        return self

    def rotate_mo(self, mo, u, log=None):
        '''Rotate orbitals with the given unitary matrix'''
        mo = np.dot(mo, u)
        if log is not None and log.verbose >= logger.DEBUG:
            ncore = self.ncore
            ncas = self.ncas
            nocc = ncore + ncas
            s = reduce(np.dot, (mo[:, ncore*2:nocc*2].T, self._scf.get_ovlp(),
                                self.mo_coeff[:, ncore*2:nocc*2]))
            log.debug('Active space overlap to initial guess, SVD = %s',
                      np.linalg.svd(s)[1])
            log.debug('Active space overlap to last step, SVD = %s',
                      np.linalg.svd(u[ncore*2:nocc*2, ncore*2:nocc*2])[1])
        return mo

    def micro_cycle_scheduler(self, envs):
        #log_norm_ddm = np.log(envs['norm_ddm'])
        #return max(self.max_cycle_micro, int(self.max_cycle_micro-1-log_norm_ddm))
        return self.max_cycle_micro

    def max_stepsize_scheduler(self, envs):
        if self._max_stepsize is None:
            self._max_stepsize = self.max_stepsize
        if envs['de'] > self.conv_tol:  # Avoid total energy increasing
            self._max_stepsize *= .5
            logger.debug(self, 'set max_stepsize to %g', self._max_stepsize)
        else:
            self._max_stepsize = np.sqrt(self.max_stepsize*self.max_stepsize)
        return self._max_stepsize

    @property
    def max_orb_stepsize(self):  # pragma: no cover
        return self.max_stepsize
    @max_orb_stepsize.setter
    def max_orb_stepsize(self, x):  # pragma: no cover
        sys.stderr.write('WARN: Attribute "max_orb_stepsize" was replaced by "max_stepsize"\n')
        self.max_stepsize = x

CASSCF = GCASSCF

# to avoid calculating AO integrals
def _fake_h_for_fast_casci(casscf, mo, eris):
    mc = copy.copy(casscf)
    mc.mo_coeff = mo
    ncore = casscf.ncore
    nocc = ncore + casscf.ncas

    mo_core = mo[:, :ncore*2]
    mo_cas  = mo[:, ncore*2:nocc*2]
    core_dm = np.dot(mo_core, mo_core.T)
    hcore = casscf.get_hcore()
    energy_core  = casscf.energy_nuc()
    energy_core += einsum('ij, ji', core_dm, hcore)
    energy_core += eris.vhf_c[:ncore*2, :ncore*2].trace() * 0.5
    h1eff  = reduce(np.dot, (mo_cas.T, hcore, mo_cas))
    h1eff += eris.vhf_c[ncore*2:nocc*2, ncore*2:nocc*2]
    mc.get_h1eff = lambda *args: (h1eff, energy_core)

    eri_cas = eris.aapp[:, :, ncore*2:nocc*2, ncore*2:nocc*2].copy()
    mc.get_h2eff = lambda *args: eri_cas
    return mc

class GCASSCFBigCAS(GCASSCF):
    """
    GCASSCF with big CAS space, using a O(N5) algorithm.
    """
    gen_g_hop = gen_g_hop_big_cas

class GCASSCFVO(GCASSCF):
    """
    GCASSCF with only VO block MO response
    """
    gen_g_hop = gen_g_hop_vo


if __name__ == '__main__':
    from pyscf import gto, scf, ao2mo
    from pyscf import mcscf
    
    np.set_printoptions(4, linewidth=1000, suppress=True)
    np.random.seed(10086)

    mol = gto.Mole()
    mol.verbose = 4
    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]

    mol.basis = {'H': 'sto-3g',
                 'O': '6-31g',}
    mol.incore_anyway = True
    mol.build()
    
    # reference RCASSCF
    from pyscf.mcscf import CASSCF, UCASSCF
    mf = scf.RHF(mol)
    mf.kernel()
    
    #mf = scf.addons.convert_to_uhf(mf)
    mc = CASSCF(mf, 4, 4)
    mc.internal_rotation = True
    mc.kernel()

    # GCAS
    #mf = scf.GHF(mol)
    mf = scf.addons.convert_to_ghf(mf)
    dm0 = mf.make_rdm1()
    mf.kernel(dm0=dm0)
    
    ovlp = mf.get_ovlp()
    H0 = mf.energy_nuc()
    H1 = mf.get_hcore()
    H2 = mf._eri
    H2 = ao2mo.restore(4, H2, mol.nao_nr())
    from libdmet_solid.system import integral
    from libdmet_solid.solver import scf
    from libdmet_solid.utils.misc import tile_eri
    H2 = tile_eri(H2, H2, H2)
    dm0 = mf.make_rdm1()

    Ham = integral.Integral(H1.shape[-1], True, False, H0, {"cd": H1[None]},
                            {"ccdd": H2[None]}, ovlp=ovlp)

    scfsolver = scf.SCF(newton_ah=True)
    scfsolver.set_system(mol.nelectron, 0, False, True, max_memory=mol.max_memory)
    scfsolver.set_integral(Ham)
    ehf, rhoHF = scfsolver.GGHF(tol=1e-12, InitGuess=dm0)
    
    mf = scfsolver.mf
    mc = GCASSCF(mf, 4, (4, 0))
    mc.internal_rotation = True
    mc.kernel()
