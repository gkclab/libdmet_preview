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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Zhi-Hao Cui <zhcui0408@gmail.com>
#

'''
UCASSCF (CASSCF without spin-degeneracy between alpha and beta orbitals)
1-step optimization algorithm
'''

import sys

import copy
from functools import reduce, partial
import numpy as np
import pyscf.gto
import pyscf.scf
from pyscf import lib
from pyscf.lib import logger
from pyscf.mcscf import ucasci
from pyscf.mcscf.mc1step import expmat, rotate_orb_cc
from pyscf.mcscf import chkfile
from pyscf import __config__

from libdmet.solver import umc_ao2mo

einsum = partial(np.einsum, optimize=True)

#FIXME:  when the number of core orbitals are different for alpha and beta,
# the convergence are very unstable and slow

# gradients, hessian operator and hessian diagonal
def gen_g_hop(casscf, mo, u, casdm1s, casdm2s, eris):
    ncas = casscf.ncas
    ncore = casscf.ncore
    nocc = (ncas + ncore[0], ncas + ncore[1])
    nmo = casscf.mo_coeff[0].shape[1]

    dm1 = np.zeros((2,nmo,nmo))
    idx = np.arange(ncore[0])
    dm1[0,idx,idx] = 1
    idx = np.arange(ncore[1])
    dm1[1,idx,idx] = 1
    dm1[0,ncore[0]:nocc[0],ncore[0]:nocc[0]] = casdm1s[0]
    dm1[1,ncore[1]:nocc[1],ncore[1]:nocc[1]] = casdm1s[1]

    # part2, part3
    vhf_c = eris.vhf_c
    vhf_ca = ((vhf_c[0] +
               einsum('uvpq,uv->pq', eris.aapp, casdm1s[0]) -
               einsum('upqv,uv->pq', eris.appa, casdm1s[0]) +
               einsum('uvpq,uv->pq', eris.AApp, casdm1s[1])),
              (vhf_c[1] +
               einsum('uvpq,uv->pq', eris.aaPP, casdm1s[0]) +
               einsum('uvpq,uv->pq', eris.AAPP, casdm1s[1]) -
               einsum('upqv,uv->pq', eris.APPA, casdm1s[1])),)

    ################# gradient #################
    hdm2 = [(einsum('tuvw,vwpq->tupq', casdm2s[0], eris.aapp) +
             einsum('tuvw,vwpq->tupq', casdm2s[1], eris.AApp)),
            (einsum('vwtu,vwpq->tupq', casdm2s[1], eris.aaPP) +
             einsum('tuvw,vwpq->tupq', casdm2s[2], eris.AAPP))]

    hcore = casscf.get_hcore()
    h1e_mo = (reduce(np.dot, (mo[0].T, hcore[0], mo[0])),
              reduce(np.dot, (mo[1].T, hcore[1], mo[1])))
    g = [np.dot(h1e_mo[0], dm1[0]),
         np.dot(h1e_mo[1], dm1[1])]

    def gpart(m):
        g[m][:,:ncore[m]] += vhf_ca[m][:,:ncore[m]]
        g[m][:,ncore[m]:nocc[m]] += \
                einsum('vuuq->qv', hdm2[m][:,:,ncore[m]:nocc[m]]) \
              + np.dot(vhf_c[m][:,ncore[m]:nocc[m]], casdm1s[m])
    gpart(0)
    gpart(1)

    def gorb_update(u, fcivec):
        r0 = casscf.pack_uniq_var(u)
        return g_orb + h_op(r0)

    ############## hessian, diagonal ###########
    # part1
    tmp = casdm2s[0].transpose(1,2,0,3) + casdm2s[0].transpose(0,2,1,3)
    hdm2apap = einsum('uvtw,tpqw->upvq', tmp, eris.appa)
    hdm2apap += hdm2[0].transpose(0,2,1,3)
    hdm2[0] = hdm2apap

    tmp = casdm2s[1].transpose(1,2,0,3) + casdm2s[1].transpose(0,2,1,3)
    # (jp|RK) *[e(jq,SK) + e(jq,LS)] => qSpR
    hdm2apAP = einsum('uvtw,tpqw->upvq', tmp, eris.apPA)
    # (JP|rk) *[e(sk,JQ) + e(ls,JQ)] => QsPr
    #hdm2APap = hdm2apAP.transpose(2,3,0,1)

    tmp = casdm2s[2].transpose(1,2,0,3) + casdm2s[2].transpose(0,2,1,3)
    hdm2APAP = einsum('uvtw,tpqw->upvq', tmp, eris.APPA)
    hdm2APAP += hdm2[1].transpose(0,2,1,3)
    hdm2[1] = hdm2APAP

    # part7
    # h_diag[0] ~ alpha-alpha
    h_diag = [einsum('ii,jj->ij', h1e_mo[0], dm1[0]) - h1e_mo[0] * dm1[0],
              einsum('ii,jj->ij', h1e_mo[1], dm1[1]) - h1e_mo[1] * dm1[1]]
    h_diag[0] = h_diag[0] + h_diag[0].T
    h_diag[1] = h_diag[1] + h_diag[1].T

    # part8
    idx = np.arange(nmo)
    g_diag = g[0].diagonal()
    h_diag[0] -= g_diag + g_diag.reshape(-1,1)
    h_diag[0][idx,idx] += g_diag * 2
    g_diag = g[1].diagonal()
    h_diag[1] -= g_diag + g_diag.reshape(-1,1)
    h_diag[1][idx,idx] += g_diag * 2

    # part2, part3
    def fpart2(m):
        v_diag = vhf_ca[m].diagonal() # (pr|kl) * e(sq,lk)
        h_diag[m][:,:ncore[m]] += v_diag.reshape(-1,1)
        h_diag[m][:ncore[m]] += v_diag
        idx = np.arange(ncore[m])
        # (V_{qr} delta_{ps} + V_{ps} delta_{qr}) delta_{pr} delta_{sq}
        h_diag[m][idx,idx] -= v_diag[:ncore[m]] * 2
    fpart2(0)
    fpart2(1)

    def fpart3(m):
        # V_{pr} e_{sq}
        tmp = einsum('ii,jj->ij', vhf_c[m], casdm1s[m])
        h_diag[m][:,ncore[m]:nocc[m]] += tmp
        h_diag[m][ncore[m]:nocc[m],:] += tmp.T
        tmp = -vhf_c[m][ncore[m]:nocc[m],ncore[m]:nocc[m]] * casdm1s[m]
        h_diag[m][ncore[m]:nocc[m],ncore[m]:nocc[m]] += tmp + tmp.T
    fpart3(0)
    fpart3(1)

    # part4
    def fpart4(jkcpp, m):
        # (qp|rs)-(pr|sq) rp in core
        tmp = -einsum('cpp->cp', jkcpp)
        # (qp|sr) - (qr|sp) rp in core => 0
        h_diag[m][:ncore[m],:] += tmp
        h_diag[m][:,:ncore[m]] += tmp.T
        h_diag[m][:ncore[m],:ncore[m]] -= tmp[:,:ncore[m]] * 2
    fpart4(eris.jkcpp, 0)
    fpart4(eris.jkcPP, 1)

    # part5 and part6 diag
    #+(qr|kp) e_s^k  p in core, sk in active
    #+(qr|sl) e_l^p  s in core, pl in active
    #-(qj|sr) e_j^p  s in core, jp in active
    #-(qp|kr) e_s^k  p in core, sk in active
    #+(qj|rs) e_j^p  s in core, jp in active
    #+(qp|rl) e_l^s  p in core, ls in active
    #-(qs|rl) e_l^p  s in core, lp in active
    #-(qj|rp) e_j^s  p in core, js in active
    def fpart5(jkcpp, m):
        jkcaa = jkcpp[:,ncore[m]:nocc[m],ncore[m]:nocc[m]]
        tmp = -2 * einsum('jik,ik->ji', jkcaa, casdm1s[m])
        h_diag[m][:ncore[m],ncore[m]:nocc[m]] -= tmp
        h_diag[m][ncore[m]:nocc[m],:ncore[m]] -= tmp.T
    fpart5(eris.jkcpp, 0)
    fpart5(eris.jkcPP, 1)

    def fpart1(m):
        v_diag = einsum('ijij->ij', hdm2[m])
        h_diag[m][ncore[m]:nocc[m],:] += v_diag
        h_diag[m][:,ncore[m]:nocc[m]] += v_diag.T
    fpart1(0)
    fpart1(1)

    g_orb = casscf.pack_uniq_var((g[0]-g[0].T, g[1]-g[1].T))
    h_diag = casscf.pack_uniq_var(h_diag)

    def h_op(x):
        x1a, x1b = casscf.unpack_uniq_var(x)
        xa_cu = x1a[:ncore[0],ncore[0]:]
        xa_av = x1a[ncore[0]:nocc[0],nocc[0]:]
        xa_ac = x1a[ncore[0]:nocc[0],:ncore[0]]
        xb_cu = x1b[:ncore[1],ncore[1]:]
        xb_av = x1b[ncore[1]:nocc[1],nocc[1]:]
        xb_ac = x1b[ncore[1]:nocc[1],:ncore[1]]

        # part7
        x2a = reduce(np.dot, (h1e_mo[0], x1a, dm1[0]))
        x2b = reduce(np.dot, (h1e_mo[1], x1b, dm1[1]))
        # part8, the hessian gives
        #x2a -= np.dot(g[0], x1a)
        #x2b -= np.dot(g[1], x1b)
        # it may ruin the hermitian of hessian unless g == g.T. So symmetrize it
        # x_{pq} -= g_{pr} \delta_{qs} x_{rs} * .5
        # x_{rs} -= g_{rp} \delta_{sq} x_{pq} * .5
        x2a -= np.dot(g[0].T, x1a)
        x2b -= np.dot(g[1].T, x1b)
        # part2
        x2a[:ncore[0]] += np.dot(xa_cu, vhf_ca[0][ncore[0]:])
        x2b[:ncore[1]] += np.dot(xb_cu, vhf_ca[1][ncore[1]:])

        # part3
        def fpart3(m, x2, x_av, x_ac):
            x2[ncore[m]:nocc[m]] += reduce(np.dot, (casdm1s[m], x_av, vhf_c[m][nocc[m]:])) \
                                  + reduce(np.dot, (casdm1s[m], x_ac, vhf_c[m][:ncore[m]]))
        fpart3(0, x2a, xa_av, xa_ac)
        fpart3(1, x2b, xb_av, xb_ac)

        # part1
        x2a[ncore[0]:nocc[0]] += einsum('upvr,vr->up', hdm2apap, x1a[ncore[0]:nocc[0]])
        x2a[ncore[0]:nocc[0]] += einsum('upvr,vr->up', hdm2apAP, x1b[ncore[1]:nocc[1]])
        x2b[ncore[1]:nocc[1]] += einsum('vrup,vr->up', hdm2apAP, x1a[ncore[0]:nocc[0]])
        x2b[ncore[1]:nocc[1]] += einsum('upvr,vr->up', hdm2APAP, x1b[ncore[1]:nocc[1]])

        # part4, part5, part6
        if ncore[0] > 0 or ncore[1] > 0:
            va, vc = casscf.update_jk_in_ah(mo, (x1a,x1b), casdm1s, eris)
            x2a[ncore[0]:nocc[0]] += va[0]
            x2b[ncore[1]:nocc[1]] += va[1]
            x2a[:ncore[0],ncore[0]:] += vc[0]
            x2b[:ncore[1],ncore[1]:] += vc[1]

        x2a = x2a - x2a.T
        x2b = x2b - x2b.T
        return casscf.pack_uniq_var((x2a,x2b))
    return g_orb, gorb_update, h_op, h_diag

def gen_g_hop_big_cas(casscf, mo, u, casdm1s, casdm2s, eris):
    ncas = casscf.ncas
    ncore = casscf.ncore
    nocc = (ncas + ncore[0], ncas + ncore[1])
    nmo = casscf.mo_coeff[0].shape[1]

    dm1 = np.zeros((2,nmo,nmo))
    idx = np.arange(ncore[0])
    dm1[0,idx,idx] = 1
    idx = np.arange(ncore[1])
    dm1[1,idx,idx] = 1
    dm1[0,ncore[0]:nocc[0],ncore[0]:nocc[0]] = casdm1s[0]
    dm1[1,ncore[1]:nocc[1],ncore[1]:nocc[1]] = casdm1s[1]

    # part2, part3
    vhf_c = eris.vhf_c
    vhf_ca = ((vhf_c[0] +
               lib.einsum('uvpq,uv->pq', eris.aapp, casdm1s[0]) -
               lib.einsum('upqv,uv->pq', eris.appa, casdm1s[0]) +
               lib.einsum('uvpq,uv->pq', eris.AApp, casdm1s[1])),
              (vhf_c[1] +
               lib.einsum('uvpq,uv->pq', eris.aaPP, casdm1s[0]) +
               lib.einsum('uvpq,uv->pq', eris.AAPP, casdm1s[1]) -
               lib.einsum('upqv,uv->pq', eris.APPA, casdm1s[1])),)

    ################# gradient #################
    # ZHC NOTE
    #hdm2 = [(einsum('tuvw,vwpq->tupq', casdm2s[0], eris.aapp) +
    #         einsum('tuvw,vwpq->tupq', casdm2s[1], eris.AApp)),
    #        (einsum('vwtu,vwpq->tupq', casdm2s[1], eris.aaPP) +
    #         einsum('tuvw,vwpq->tupq', casdm2s[2], eris.AAPP))]
    # ZHC NOTE

    hcore = casscf.get_hcore()
    h1e_mo = (reduce(np.dot, (mo[0].T, hcore[0], mo[0])),
              reduce(np.dot, (mo[1].T, hcore[1], mo[1])))
    g = [np.dot(h1e_mo[0], dm1[0]),
         np.dot(h1e_mo[1], dm1[1])]

    def gpart(m):
        g[m][:,:ncore[m]] += vhf_ca[m][:,:ncore[m]]
        #g[m][:,ncore[m]:nocc[m]] += \
        #        einsum('vuuq->qv', hdm2[m][:,:,ncore[m]:nocc[m]]) \
        #      + np.dot(vhf_c[m][:,ncore[m]:nocc[m]], casdm1s[m])
        g[m][:,ncore[m]:nocc[m]] += np.dot(vhf_c[m][:,ncore[m]:nocc[m]], casdm1s[m])
        if m == 0:
            #g[m][:,ncore[m]:nocc[m]] += \
            #        einsum('vuuq->qv', hdm2[m][:,:,ncore[m]:nocc[m]])
            g[m][:,ncore[m]:nocc[m]] += \
                    lib.einsum('tuvw, vwuq -> qt', casdm2s[0], eris.aapp)
            g[m][:,ncore[m]:nocc[m]] += \
                    lib.einsum('tuvw, vwuq -> qt', casdm2s[1], eris.AApp)
        else:
            g[m][:,ncore[m]:nocc[m]] += \
                    lib.einsum('vwtu, vwuq -> qt', casdm2s[1], eris.aaPP)
            g[m][:,ncore[m]:nocc[m]] += \
                    lib.einsum('tuvw, vwuq -> qt', casdm2s[2], eris.AAPP)

    gpart(0)
    gpart(1)

    def gorb_update(u, fcivec):
        r0 = casscf.pack_uniq_var(u)
        return g_orb + h_op(r0)

    ############## hessian, diagonal ###########
    # part1
#    tmp = casdm2s[0].transpose(1,2,0,3) + casdm2s[0].transpose(0,2,1,3)
#    hdm2apap = einsum('uvtw,tpqw->upvq', tmp, eris.appa)
#    hdm2apap += hdm2[0].transpose(0,2,1,3)
#    hdm2[0] = hdm2apap
#
#    tmp = casdm2s[1].transpose(1,2,0,3) + casdm2s[1].transpose(0,2,1,3)
#    # (jp|RK) *[e(jq,SK) + e(jq,LS)] => qSpR
#    hdm2apAP = einsum('uvtw,tpqw->upvq', tmp, eris.apPA)
#    # (JP|rk) *[e(sk,JQ) + e(ls,JQ)] => QsPr
#    #hdm2APap = hdm2apAP.transpose(2,3,0,1)
#
#    tmp = casdm2s[2].transpose(1,2,0,3) + casdm2s[2].transpose(0,2,1,3)
#    hdm2APAP = einsum('uvtw,tpqw->upvq', tmp, eris.APPA)
#    hdm2APAP += hdm2[1].transpose(0,2,1,3)
#    hdm2[1] = hdm2APAP

    # part7
    # h_diag[0] ~ alpha-alpha
    h_diag = [einsum('ii,jj->ij', h1e_mo[0], dm1[0]) - h1e_mo[0] * dm1[0],
              einsum('ii,jj->ij', h1e_mo[1], dm1[1]) - h1e_mo[1] * dm1[1]]
    h_diag[0] = h_diag[0] + h_diag[0].T
    h_diag[1] = h_diag[1] + h_diag[1].T

    # part8
    idx = np.arange(nmo)
    g_diag = g[0].diagonal()
    h_diag[0] -= g_diag + g_diag.reshape(-1,1)
    h_diag[0][idx,idx] += g_diag * 2
    g_diag = g[1].diagonal()
    h_diag[1] -= g_diag + g_diag.reshape(-1,1)
    h_diag[1][idx,idx] += g_diag * 2

    # part2, part3
    def fpart2(m):
        v_diag = vhf_ca[m].diagonal() # (pr|kl) * e(sq,lk)
        h_diag[m][:,:ncore[m]] += v_diag.reshape(-1,1)
        h_diag[m][:ncore[m]] += v_diag
        idx = np.arange(ncore[m])
        # (V_{qr} delta_{ps} + V_{ps} delta_{qr}) delta_{pr} delta_{sq}
        h_diag[m][idx,idx] -= v_diag[:ncore[m]] * 2
    fpart2(0)
    fpart2(1)

    def fpart3(m):
        # V_{pr} e_{sq}
        tmp = einsum('ii,jj->ij', vhf_c[m], casdm1s[m])
        h_diag[m][:,ncore[m]:nocc[m]] += tmp
        h_diag[m][ncore[m]:nocc[m],:] += tmp.T
        tmp = -vhf_c[m][ncore[m]:nocc[m],ncore[m]:nocc[m]] * casdm1s[m]
        h_diag[m][ncore[m]:nocc[m],ncore[m]:nocc[m]] += tmp + tmp.T
    fpart3(0)
    fpart3(1)

    # part4
    def fpart4(jkcpp, m):
        # (qp|rs)-(pr|sq) rp in core
        tmp = -einsum('cpp->cp', jkcpp)
        # (qp|sr) - (qr|sp) rp in core => 0
        h_diag[m][:ncore[m],:] += tmp
        h_diag[m][:,:ncore[m]] += tmp.T
        h_diag[m][:ncore[m],:ncore[m]] -= tmp[:,:ncore[m]] * 2
    fpart4(eris.jkcpp, 0)
    fpart4(eris.jkcPP, 1)

    # part5 and part6 diag
    #+(qr|kp) e_s^k  p in core, sk in active
    #+(qr|sl) e_l^p  s in core, pl in active
    #-(qj|sr) e_j^p  s in core, jp in active
    #-(qp|kr) e_s^k  p in core, sk in active
    #+(qj|rs) e_j^p  s in core, jp in active
    #+(qp|rl) e_l^s  p in core, ls in active
    #-(qs|rl) e_l^p  s in core, lp in active
    #-(qj|rp) e_j^s  p in core, js in active
    def fpart5(jkcpp, m):
        jkcaa = jkcpp[:,ncore[m]:nocc[m],ncore[m]:nocc[m]]
        tmp = -2 * einsum('jik,ik->ji', jkcaa, casdm1s[m])
        h_diag[m][:ncore[m],ncore[m]:nocc[m]] -= tmp
        h_diag[m][ncore[m]:nocc[m],:ncore[m]] -= tmp.T
    fpart5(eris.jkcpp, 0)
    fpart5(eris.jkcPP, 1)

    def fpart1(m):
        #v_diag = einsum('ijij->ij', hdm2[m])
        if m == 0:
            v_diag  = lib.einsum('tuuw, tppw -> up', casdm2s[0], eris.appa)
            v_diag += lib.einsum('utuw, tppw -> up', casdm2s[0], eris.appa)
            v_diag += lib.einsum('uuvw, vwpp -> up', casdm2s[0], eris.aapp)
            v_diag += lib.einsum('uuvw, vwpp -> up', casdm2s[1], eris.AApp)
        else:
            v_diag  = lib.einsum('tuuw, tppw -> up', casdm2s[2], eris.APPA)
            v_diag += lib.einsum('utuw, tppw -> up', casdm2s[2], eris.APPA)
            v_diag += lib.einsum('uuvw, vwpp -> up', casdm2s[2], eris.AAPP)
            v_diag += lib.einsum('vwuu, vwpp -> up', casdm2s[1], eris.aaPP)
    
        h_diag[m][ncore[m]:nocc[m],:] += v_diag
        h_diag[m][:,ncore[m]:nocc[m]] += v_diag.T
    fpart1(0)
    fpart1(1)

    g_orb = casscf.pack_uniq_var((g[0]-g[0].T, g[1]-g[1].T))
    h_diag = casscf.pack_uniq_var(h_diag)

    def h_op(x):
        x1a, x1b = casscf.unpack_uniq_var(x)
        xa_cu = x1a[:ncore[0],ncore[0]:]
        xa_av = x1a[ncore[0]:nocc[0],nocc[0]:]
        xa_ac = x1a[ncore[0]:nocc[0],:ncore[0]]
        xb_cu = x1b[:ncore[1],ncore[1]:]
        xb_av = x1b[ncore[1]:nocc[1],nocc[1]:]
        xb_ac = x1b[ncore[1]:nocc[1],:ncore[1]]

        # part7
        x2a = reduce(np.dot, (h1e_mo[0], x1a, dm1[0]))
        x2b = reduce(np.dot, (h1e_mo[1], x1b, dm1[1]))
        # part8, the hessian gives
        #x2a -= np.dot(g[0], x1a)
        #x2b -= np.dot(g[1], x1b)
        # it may ruin the hermitian of hessian unless g == g.T. So symmetrize it
        # x_{pq} -= g_{pr} \delta_{qs} x_{rs} * .5
        # x_{rs} -= g_{rp} \delta_{sq} x_{pq} * .5
        x2a -= np.dot(g[0].T, x1a)
        x2b -= np.dot(g[1].T, x1b)
        # part2
        x2a[:ncore[0]] += np.dot(xa_cu, vhf_ca[0][ncore[0]:])
        x2b[:ncore[1]] += np.dot(xb_cu, vhf_ca[1][ncore[1]:])

        # part3
        def fpart3(m, x2, x_av, x_ac):
            x2[ncore[m]:nocc[m]] += reduce(np.dot, (casdm1s[m], x_av, vhf_c[m][nocc[m]:])) \
                                  + reduce(np.dot, (casdm1s[m], x_ac, vhf_c[m][:ncore[m]]))
        fpart3(0, x2a, xa_av, xa_ac)
        fpart3(1, x2b, xb_av, xb_ac)

        # part1
        # ZHC NOTE
        #x2a[ncore[0]:nocc[0]] += lib.einsum('upvr,vr->up', hdm2apap, x1a[ncore[0]:nocc[0]])
        #x2a[ncore[0]:nocc[0]] += lib.einsum('upvr,vr->up', hdm2apAP, x1b[ncore[1]:nocc[1]])
        #x2b[ncore[1]:nocc[1]] += lib.einsum('vrup,vr->up', hdm2apAP, x1a[ncore[0]:nocc[0]])
        #x2b[ncore[1]:nocc[1]] += lib.einsum('upvr,vr->up', hdm2APAP, x1b[ncore[1]:nocc[1]])
    
        # AA
        tmp = lib.einsum('tpqw, vq -> tpvw', eris.appa, x1a[ncore[0]:nocc[0]])
        x2a[ncore[0]:nocc[0]] += lib.einsum('tuvw, tpvw -> up', casdm2s[0], tmp)
        x2a[ncore[0]:nocc[0]] += lib.einsum('utvw, tpvw -> up', casdm2s[0], tmp)
        tmp = None
        
        tmp = lib.einsum('twpq, vq -> twpv', eris.aapp, x1a[ncore[0]:nocc[0]])
        x2a[ncore[0]:nocc[0]] += lib.einsum('uvtw, twpv -> up', casdm2s[0], tmp)
        tmp = None
        
        tmp = lib.einsum('twpq, vq -> twpv', eris.AApp, x1a[ncore[0]:nocc[0]])
        x2a[ncore[0]:nocc[0]] += lib.einsum('uvtw, twpv -> up', casdm2s[1], tmp)
        tmp = None
        
        # BB
        tmp = lib.einsum('tpqw, vq -> tpvw', eris.APPA, x1b[ncore[1]:nocc[1]])
        x2b[ncore[1]:nocc[1]] += lib.einsum('tuvw, tpvw -> up', casdm2s[2], tmp)
        x2b[ncore[1]:nocc[1]] += lib.einsum('utvw, tpvw -> up', casdm2s[2], tmp)
        tmp = None
        
        tmp = lib.einsum('twpq, vq -> twpv', eris.AAPP, x1b[ncore[1]:nocc[1]])
        x2b[ncore[1]:nocc[1]] += lib.einsum('uvtw, twpv -> up', casdm2s[2], tmp)
        tmp = None
        
        tmp = lib.einsum('vwpq, uq -> vwpu', eris.aaPP, x1b[ncore[1]:nocc[1]])
        x2b[ncore[1]:nocc[1]] += lib.einsum('vwtu, vwpu -> tp', casdm2s[1], tmp)
        tmp = None
        
        # AB
        tmp = lib.einsum('tpqw, vq -> tpvw', eris.apPA, x1b[ncore[1]:nocc[1]])
        x2a[ncore[0]:nocc[0]] += lib.einsum('tuvw, tpvw -> up', casdm2s[1], tmp)
        x2a[ncore[0]:nocc[0]] += lib.einsum('utvw, tpvw -> up', casdm2s[1], tmp)
        tmp = None

        tmp = lib.einsum('tpqw, up -> tuqw', eris.apPA, x1a[ncore[0]:nocc[0]])
        x2b[ncore[1]:nocc[1]] += lib.einsum('tuvw, tuqw -> vq', casdm2s[1], tmp)
        x2b[ncore[1]:nocc[1]] += lib.einsum('utvw, tuqw -> vq', casdm2s[1], tmp)
        tmp = None
        
        # part4, part5, part6
        if ncore[0] > 0 or ncore[1] > 0:
            va, vc = casscf.update_jk_in_ah(mo, (x1a,x1b), casdm1s, eris)
            x2a[ncore[0]:nocc[0]] += va[0]
            x2b[ncore[1]:nocc[1]] += va[1]
            x2a[:ncore[0],ncore[0]:] += vc[0]
            x2b[:ncore[1],ncore[1]:] += vc[1]

        x2a = x2a - x2a.T
        x2b = x2b - x2b.T
        return casscf.pack_uniq_var((x2a,x2b))
    return g_orb, gorb_update, h_op, h_diag

def kernel(casscf, mo_coeff, tol=1e-7, conv_tol_grad=None,
           ci0=None, callback=None, verbose=None, dump_chk=True):
    if verbose is None:
        verbose = casscf.verbose
    log = logger.Logger(casscf.stdout, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start 1-step CASSCF')

    mo = mo_coeff
    nmo = mo[0].shape[1]
    #TODO: lazy evaluate eris, to leave enough memory for FCI solver
    eris = casscf.ao2mo(mo)
    e_tot, e_cas, fcivec = casscf.casci(mo, ci0, eris, log, locals())
    if casscf.ncas == nmo and not casscf.internal_rotation:
        return True, e_tot, e_cas, fcivec, mo

    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(tol)
        logger.info(casscf, 'Set conv_tol_grad to %g', conv_tol_grad)
    conv_tol_ddm = conv_tol_grad * 3
    conv = False
    totmicro = totinner = 0
    norm_gorb = norm_gci = 0
    de, elast = e_tot, e_tot
    r0 = None

    t1m = log.timer('Initializing 1-step CASSCF', *cput0)
    casdm1, casdm2 = casscf.fcisolver.make_rdm12s(fcivec, casscf.ncas, casscf.nelecas)
    norm_ddm = 1e2
    casdm1_last = casdm1
    t3m = t2m = log.timer('CAS DM', *t1m)
    imacro = 0
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
            if imicro >= max_cycle_micro:
                log.debug('micro %d  |u-1|=%5.3g  |g[o]|=%5.3g  ',
                          imicro, norm_t, norm_gorb)
                break

            casdm1, casdm2, gci, fcivec = casscf.update_casdm(mo, u, fcivec, e_cas, eris)
            norm_ddm = (np.linalg.norm(casdm1[0] - casdm1_last[0]) +
                        np.linalg.norm(casdm1[1] - casdm1_last[1]))
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
            if (norm_t < 1e-4 or
                (norm_gorb < conv_tol_grad*.5 and norm_ddm < conv_tol_ddm*.4)):
                break

        rota.close()
        rota = None

        totmicro += imicro
        totinner += njk

        eris = None
        u = copy.copy(u)
        g_orb = copy.copy(g_orb)
        mo = casscf.rotate_mo(mo, u, log)
        eris = casscf.ao2mo(mo)
        t2m = log.timer('update eri', *t3m)

        e_tot, e_cas, fcivec = casscf.casci(mo, fcivec, eris, log, locals())
        casdm1, casdm2 = casscf.fcisolver.make_rdm12s(fcivec, casscf.ncas, casscf.nelecas)
        norm_ddm = (np.linalg.norm(casdm1[0] - casdm1_last[0]) +
                    np.linalg.norm(casdm1[1] - casdm1_last[1]))
        casdm1_last = casdm1
        log.timer('CASCI solver', *t2m)
        t2m = t1m = log.timer('macro iter %d'%imacro, *t1m)

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
                 imacro+1, totinner, totmicro)
    else:
        log.info('1-step CASSCF not converged, %d macro (%d JK %d micro) steps',
                 imacro+1, totinner, totmicro)
    log.timer('1-step CASSCF', *cput0)
    return conv, e_tot, e_cas, fcivec, mo

class UCASSCF(ucasci.UCASCI):
    max_stepsize = getattr(__config__, 'mcscf_umc1step_UCASSCF_max_stepsize', .02)
    max_cycle_macro = getattr(__config__, 'mcscf_umc1step_UCASSCF_max_cycle_macro', 50)
    max_cycle_micro = getattr(__config__, 'mcscf_umc1step_UCASSCF_max_cycle_micro', 4)
    conv_tol = getattr(__config__, 'mcscf_umc1step_UCASSCF_conv_tol', 1e-7)
    conv_tol_grad = getattr(__config__, 'mcscf_umc1step_UCASSCF_conv_tol_grad', None)
    # for augmented hessian
    ah_level_shift = getattr(__config__, 'mcscf_umc1step_UCASSCF_ah_level_shift', 1e-8)
    ah_conv_tol = getattr(__config__, 'mcscf_umc1step_UCASSCF_ah_conv_tol', 1e-12)
    ah_max_cycle = getattr(__config__, 'mcscf_umc1step_UCASSCF_ah_max_cycle', 30)
    ah_lindep = getattr(__config__, 'mcscf_umc1step_UCASSCF_ah_lindep', 1e-14)
    ah_start_tol = getattr(__config__, 'mcscf_umc1step_UCASSCF_ah_start_tol', 2.5)
    ah_start_cycle = getattr(__config__, 'mcscf_umc1step_UCASSCF_ah_start_cycle', 3)
    ah_grad_trust_region = getattr(__config__, 'mcscf_umc1step_UCASSCF_ah_grad_trust_region', 3.0)

    internal_rotation = getattr(__config__, 'mcscf_umc1step_UCASSCF_internal_rotation', False)
    internal_rotation_vo = getattr(__config__, 'mcscf_mc1step_UCASSCF_internal_rotation_vo', False)
    ci_response_space = getattr(__config__, 'mcscf_umc1step_UCASSCF_ci_response_space', 4)
    with_dep4 = getattr(__config__, 'mcscf_umc1step_UCASSCF_with_dep4', False)
    chk_ci = getattr(__config__, 'mcscf_umc1step_UCASSCF_chk_ci', False)
    kf_interval = getattr(__config__, 'mcscf_umc1step_UCASSCF_kf_interval', 4)
    kf_trust_region = getattr(__config__, 'mcscf_umc1step_UCASSCF_kf_trust_region', 3.0)

    natorb = getattr(__config__, 'mcscf_umc1step_UCASSCF_natorb', False)
    #canonicalization = getattr(__config__, 'mcscf_umc1step_UCASSCF_canonicalization', True)
    #sorting_mo_energy = getattr(__config__, 'mcscf_umc1step_UCASSCF_sorting_mo_energy', False)
    
    # pyscf's eri's convention, aa, ab, bb
    eri_ab_mid = True

    def __init__(self, mf_or_mol, ncas, nelecas, ncore=None, frozen=None):
        ucasci.UCASCI.__init__(self, mf_or_mol, ncas, nelecas, ncore)
        self.frozen = frozen

        self.callback = None
        self.chkfile = self._scf.chkfile

        self.fcisolver.max_cycle = getattr(__config__,
                                           'mcscf_umc1step_UCASSCF_fcisolver_max_cycle', 50)
        self.fcisolver.conv_tol = getattr(__config__,
                                          'mcscf_umc1step_UCASSCF_fcisolver_conv_tol', 1e-8)

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
                    'internal_rotation', 'internal_rotation_vo', 'ci_response_space',
                    'with_dep4', 'chk_ci',
                    'kf_interval', 'kf_trust_region', 'fcisolver_max_cycle',
                    'fcisolver_conv_tol', 'natorb', 'canonicalization',
                    'sorting_mo_energy'))
        self._keys = set(self.__dict__.keys()).union(keys)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** UHF-CASSCF flags ********')
        nmo = self.mo_coeff[0].shape[1]
        ncore = self.ncore
        ncas = self.ncas
        nvir_alpha = nmo - ncore[0] - ncas
        nvir_beta  = nmo - ncore[1]  - ncas
        log.info('CAS (%de+%de, %do), ncore = [%d+%d], nvir = [%d+%d]',
                 self.nelecas[0], self.nelecas[1], ncas,
                 ncore[0], ncore[1], nvir_alpha, nvir_beta)
        if ncore[0] != ncore[1]:
            log.warn('converge might be slow since num alpha core %d != num beta core %d',
                     ncore[0], ncore[1])
        if self.frozen is not None:
            log.info('frozen orbitals %s', str(self.frozen))
        log.info('max. macro cycles = %d', self.max_cycle_macro)
        log.info('max. micro cycles = %d', self.max_cycle_micro)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_grad = %s', self.conv_tol_grad)
        log.info('max. orb step = %g', self.max_stepsize)
        log.info('augmented hessian max_cycle = %d', self.ah_max_cycle)
        log.info('augmented hessian conv_tol = %g', self.ah_conv_tol)
        log.info('augmented hessian linear dependence = %g', self.ah_lindep)
        log.info('augmented hessian level shift = %g', self.ah_level_shift)
        log.info('augmented hessian start_tol = %g', self.ah_start_tol)
        log.info('augmented hessian start_cycle = %d', self.ah_start_cycle)
        log.info('augmented hessian grad_trust_region = %g', self.ah_grad_trust_region)
        log.info('kf_trust_region = %g', self.kf_trust_region)
        log.info('kf_interval = %d', self.kf_interval)
        log.info('ci_response_space = %d', self.ci_response_space)
        #log.info('diis = %s', self.diis)
        log.info('chkfile = %s', self.chkfile)
        #log.info('natorb = %s', self.natorb)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, pyscf.lib.current_memory()[0])
        log.info('internal_rotation = %s', self.internal_rotation)
        log.info('internal_rotation_vo = %s', self.internal_rotation_vo)
        try:
            self.fcisolver.dump_flags(self.verbose)
        except AttributeError:
            pass
    
    def get_hcore(self, mol=None):
        # ZHC NOTE allow spin dependent hcore
        hcore = self._scf.get_hcore(mol)
        if isinstance(hcore, np.ndarray) and hcore.ndim == 2:
            hcore = (hcore, hcore)
        else:
            assert len(hcore) == 2
        return hcore

    def kernel(self, mo_coeff=None, ci0=None, callback=None, _kern=kernel):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if callback is None: callback = self.callback

        self.check_sanity()
        self.dump_flags()

        self.converged, self.e_tot, self.e_cas, self.ci, self.mo_coeff = \
                _kern(self, mo_coeff,
                      tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
                      ci0=ci0, callback=callback, verbose=self.verbose)
        logger.note(self, 'UCASSCF energy = %.15g', self.e_tot)
        #if self.verbose >= logger.INFO:
        #    self.analyze(mo_coeff, self.ci, verbose=self.verbose)
        self._finalize()
        return self.e_tot, self.e_cas, self.ci, self.mo_coeff

    def mc1step(self, mo_coeff=None, ci0=None, callback=None):
        return self.kernel(mo_coeff, ci0, callback)

    def mc2step(self, mo_coeff=None, ci0=None, callback=None):
        from libdmet.solver import umc2step
        return self.kernel(mo_coeff, ci0, callback, umc2step.kernel)

    def get_h2eff(self, mo_coeff=None):
        '''Computing active space two-particle Hamiltonian.
        '''
        return self.get_h2cas(mo_coeff)
    def get_h2cas(self, mo_coeff=None):
        return ucasci.UCASCI.ao2mo(self, mo_coeff)

    def casci(self, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
        if eris is None:
            fcasci = copy.copy(self)
            fcasci.ao2mo = self.get_h2cas
        else:
            fcasci = _fake_h_for_fast_casci(self, mo_coeff, eris,
                                            eri_ab_mid=self.eri_ab_mid)

        log = logger.new_logger(self, verbose)

        e_tot, e_cas, fcivec = ucasci.kernel(fcasci, mo_coeff, ci0, log)
        if envs is not None and log.verbose >= logger.INFO:
            log.debug('CAS space CI energy = %.15g', e_cas)

            if 'imicro' in envs:  # Within CASSCF iteration
                log.info('macro iter %d (%d JK  %d micro), '
                         'UCASSCF E = %.15g  dE = %.8g',
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
                log.info('UCASCI E = %.15g', e_tot)
        return e_tot, e_cas, fcivec

    def uniq_var_indices(self, nmo, ncore, ncas, frozen, nelecas=None):
        nocc = ncore + ncas
        mask = np.zeros((nmo, nmo),dtype=bool)
        mask[ncore:nocc, :ncore] = True
        mask[nocc:, :nocc] = True
        
        if self.internal_rotation_vo:
            # cas-v to cas-o
            nocc_cas = nelecas + ncore
            mask[nocc_cas:nocc, ncore:nocc_cas] = True
        elif self.internal_rotation:
            # cas-cas
            mask[ncore:nocc, ncore:nocc][np.tril_indices(ncas, -1)] = True

        if frozen is not None:
            if isinstance(frozen, (int, np.integer)):
                mask[:frozen] = mask[:,:frozen] = False
            else:
                mask[frozen] = mask[:,frozen] = False
        return mask

    def pack_uniq_var(self, mat):
        nmo = self.mo_coeff[0].shape[1]
        ncore = self.ncore
        ncas = self.ncas
        idxa = self.uniq_var_indices(nmo, ncore[0], ncas, self.frozen, nelecas=self.nelecas[0])
        idxb = self.uniq_var_indices(nmo, ncore[1], ncas, self.frozen, nelecas=self.nelecas[1])
        return np.hstack((mat[0][idxa], mat[1][idxb]))

    # to anti symmetric matrix
    def unpack_uniq_var(self, v):
        nmo = self.mo_coeff[0].shape[1]
        ncore = self.ncore
        ncas = self.ncas
        idx = np.empty((2,nmo,nmo), dtype=bool)
        idx[0] = self.uniq_var_indices(nmo, ncore[0], ncas, self.frozen, nelecas=self.nelecas[0])
        idx[1] = self.uniq_var_indices(nmo, ncore[1], ncas, self.frozen, nelecas=self.nelecas[1])
        mat = np.zeros((2,nmo,nmo))
        mat[idx] = v
        mat[0] = mat[0] - mat[0].T
        mat[1] = mat[1] - mat[1].T
        return mat

    def update_rotate_matrix(self, dx, u0=1):
        if isinstance(u0, int) and u0 == 1:
            u0 = (1,1)
        dr = self.unpack_uniq_var(dx)
        ua = np.dot(u0[0], expmat(dr[0]))
        ub = np.dot(u0[1], expmat(dr[1]))
        return (ua, ub)

    def gen_g_hop(self, *args):
        return gen_g_hop(self, *args)

    rotate_orb_cc = rotate_orb_cc

    def ao2mo(self, mo_coeff=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
#        nmo = mo[0].shape[1]
#        ncore = self.ncore
#        ncas = self.ncas
#        nocc = (ncas + ncore[0], ncas + ncore[1])
#        eriaa = pyscf.ao2mo.incore.full(self._scf._eri, mo[0])
#        eriab = pyscf.ao2mo.incore.general(self._scf._eri, (mo[0],mo[0],mo[1],mo[1]))
#        eribb = pyscf.ao2mo.incore.full(self._scf._eri, mo[1])
#        eriaa = pyscf.ao2mo.restore(1, eriaa, nmo)
#        eriab = pyscf.ao2mo.restore(1, eriab, nmo)
#        eribb = pyscf.ao2mo.restore(1, eribb, nmo)
#        eris = lambda:None
#        eris.jkcpp = einsum('iipq->ipq', eriaa[:ncore[0],:ncore[0],:,:]) \
#                   - einsum('ipqi->ipq', eriaa[:ncore[0],:,:,:ncore[0]])
#        eris.jkcPP = einsum('iipq->ipq', eribb[:ncore[1],:ncore[1],:,:]) \
#                   - einsum('ipqi->ipq', eribb[:ncore[1],:,:,:ncore[1]])
#        eris.jC_pp = einsum('pqii->pq', eriab[:,:,:ncore[1],:ncore[1]])
#        eris.jc_PP = einsum('iipq->pq', eriab[:ncore[0],:ncore[0],:,:])
#        eris.aapp = np.copy(eriaa[ncore[0]:nocc[0],ncore[0]:nocc[0],:,:])
#        eris.aaPP = np.copy(eriab[ncore[0]:nocc[0],ncore[0]:nocc[0],:,:])
#        eris.AApp = np.copy(eriab[:,:,ncore[1]:nocc[1],ncore[1]:nocc[1]].transpose(2,3,0,1))
#        eris.AAPP = np.copy(eribb[ncore[1]:nocc[1],ncore[1]:nocc[1],:,:])
#        eris.appa = np.copy(eriaa[ncore[0]:nocc[0],:,:,ncore[0]:nocc[0]])
#        eris.apPA = np.copy(eriab[ncore[0]:nocc[0],:,:,ncore[1]:nocc[1]])
#        eris.APPA = np.copy(eribb[ncore[1]:nocc[1],:,:,ncore[1]:nocc[1]])
#
#        eris.cvCV = np.copy(eriab[:ncore[0],ncore[0]:,:ncore[1],ncore[1]:])
#        eris.Icvcv = eriaa[:ncore[0],ncore[0]:,:ncore[0],ncore[0]:] * 2\
#                   - eriaa[:ncore[0],:ncore[0],ncore[0]:,ncore[0]:].transpose(0,3,1,2) \
#                   - eriaa[:ncore[0],ncore[0]:,:ncore[0],ncore[0]:].transpose(0,3,2,1)
#        eris.ICVCV = eribb[:ncore[1],ncore[1]:,:ncore[1],ncore[1]:] * 2\
#                   - eribb[:ncore[1],:ncore[1],ncore[1]:,ncore[1]:].transpose(0,3,1,2) \
#                   - eribb[:ncore[1],ncore[1]:,:ncore[1],ncore[1]:].transpose(0,3,2,1)
#
#        eris.Iapcv = eriaa[ncore[0]:nocc[0],:,:ncore[0],ncore[0]:] * 2 \
#                   - eriaa[:,ncore[0]:,:ncore[0],ncore[0]:nocc[0]].transpose(3,0,2,1) \
#                   - eriaa[:,:ncore[0],ncore[0]:,ncore[0]:nocc[0]].transpose(3,0,1,2)
#        eris.IAPCV = eribb[ncore[1]:nocc[1],:,:ncore[1],ncore[1]:] * 2 \
#                   - eribb[:,ncore[1]:,:ncore[1],ncore[1]:nocc[1]].transpose(3,0,2,1) \
#                   - eribb[:,:ncore[1],ncore[1]:,ncore[1]:nocc[1]].transpose(3,0,1,2)
#        eris.apCV = np.copy(eriab[ncore[0]:nocc[0],:,:ncore[1],ncore[1]:])
#        eris.APcv = np.copy(eriab[:ncore[0],ncore[0]:,ncore[1]:nocc[1],:].transpose(2,3,0,1))
#        return eris
        return umc_ao2mo._ERIS(self, mo_coeff)

    def update_jk_in_ah(self, mo, r, casdm1s, eris):
        ncas = self.ncas
        ncore = self.ncore
        nocc = (ncas + ncore[0], ncas + ncore[1])
        ra, rb = r
        vhf3ca = einsum('srqp,sr->qp', eris.Icvcv, ra[:ncore[0],ncore[0]:])
        vhf3ca += einsum('qpsr,sr->qp', eris.cvCV, rb[:ncore[1],ncore[1]:]) * 2
        vhf3cb = einsum('srqp,sr->qp', eris.ICVCV, rb[:ncore[1],ncore[1]:])
        vhf3cb += einsum('srqp,sr->qp', eris.cvCV, ra[:ncore[0],ncore[0]:]) * 2

        vhf3aa = einsum('kpsr,sr->kp', eris.Iapcv, ra[:ncore[0],ncore[0]:])
        vhf3aa += einsum('kpsr,sr->kp', eris.apCV, rb[:ncore[1],ncore[1]:]) * 2
        vhf3ab = einsum('kpsr,sr->kp', eris.IAPCV, rb[:ncore[1],ncore[1]:])
        vhf3ab += einsum('kpsr,sr->kp', eris.APcv, ra[:ncore[0],ncore[0]:]) * 2

        dm4 = (np.dot(casdm1s[0], ra[ncore[0]:nocc[0]]),
               np.dot(casdm1s[1], rb[ncore[1]:nocc[1]]))
        vhf4a = einsum('krqp,kr->qp', eris.Iapcv, dm4[0])
        vhf4a += einsum('krqp,kr->qp', eris.APcv, dm4[1]) * 2
        vhf4b = einsum('krqp,kr->qp', eris.IAPCV, dm4[1])
        vhf4b += einsum('krqp,kr->qp', eris.apCV, dm4[0]) * 2

        va = (np.dot(casdm1s[0], vhf3aa), np.dot(casdm1s[1], vhf3ab))
        vc = (vhf3ca + vhf4a, vhf3cb + vhf4b)
        return va, vc

    def update_casdm(self, mo, u, fcivec, e_cas, eris):

        ecore, h1cas, h2cas = self.approx_cas_integral(mo, u, eris)

        ci1, g = self.solve_approx_ci(h1cas, h2cas, fcivec, ecore, e_cas)
        casdm1, casdm2 = self.fcisolver.make_rdm12s(ci1, self.ncas, self.nelecas)
        return casdm1, casdm2, g, ci1

    def approx_cas_integral(self, mo, u, eris):
        ncas = self.ncas
        ncore = self.ncore
        nocc = (ncas + ncore[0], ncas + ncore[1])
        nmo = mo[0].shape[1]
        rmat = u - np.eye(nmo)
        mocas = (mo[0][:,ncore[0]:nocc[0]], mo[1][:,ncore[1]:nocc[1]])

        hcore = self.get_hcore()
        h1effa = reduce(np.dot, (rmat[0][:,:nocc[0]].T, mo[0].T,
                                    hcore[0], mo[0][:,:nocc[0]]))
        h1effb = reduce(np.dot, (rmat[1][:,:nocc[1]].T, mo[1].T,
                                    hcore[1], mo[1][:,:nocc[1]]))
        h1effa = h1effa + h1effa.T
        h1effb = h1effb + h1effb.T

        aapc = eris.aapp[:,:,:,:ncore[0]]
        aaPC = eris.aaPP[:,:,:,:ncore[1]]
        AApc = eris.AApp[:,:,:,:ncore[0]]
        AAPC = eris.AAPP[:,:,:,:ncore[1]]
        apca = eris.appa[:,:,:ncore[0],:]
        APCA = eris.APPA[:,:,:ncore[1],:]
        jka = einsum('iup->up', eris.jkcpp[:,:nocc[0]]) + eris.jC_pp[:nocc[0]]
        v1a = (einsum('up,pv->uv', jka[ncore[0]:], rmat[0][:,ncore[0]:nocc[0]]) +
               einsum('uvpi,pi->uv', aapc-apca.transpose(0,3,1,2), rmat[0][:,:ncore[0]]) +
               einsum('uvpi,pi->uv', aaPC, rmat[1][:,:ncore[1]]))
        jkb = einsum('iup->up', eris.jkcPP[:,:nocc[1]]) + eris.jc_PP[:nocc[1]]
        v1b = (einsum('up,pv->uv', jkb[ncore[1]:], rmat[1][:,ncore[1]:nocc[1]]) +
               einsum('uvpi,pi->uv', AApc, rmat[0][:,:ncore[0]]) +
               einsum('uvpi,pi->uv', AAPC-APCA.transpose(0,3,1,2), rmat[1][:,:ncore[1]]))
        h1casa = (h1effa[ncore[0]:,ncore[0]:] + (v1a + v1a.T) +
                  reduce(np.dot, (mocas[0].T, hcore[0], mocas[0])) +
                  eris.vhf_c[0][ncore[0]:nocc[0],ncore[0]:nocc[0]])
        h1casb = (h1effb[ncore[1]:,ncore[1]:] + (v1b + v1b.T) +
                  reduce(np.dot, (mocas[1].T, hcore[1], mocas[1])) +
                  eris.vhf_c[1][ncore[1]:nocc[1],ncore[1]:nocc[1]])
        h1cas = (h1casa, h1casb)

        aaap = eris.aapp[:,:,ncore[0]:nocc[0],:]
        aaAP = eris.aaPP[:,:,ncore[1]:nocc[1],:]
        AAap = eris.AApp[:,:,ncore[1]:nocc[1],:]
        AAAP = eris.AAPP[:,:,ncore[1]:nocc[1],:]
        aaaa = einsum('tuvp,pw->tuvw', aaap, rmat[0][:,ncore[0]:nocc[0]])
        aaaa = aaaa + aaaa.transpose(0,1,3,2)
        aaaa = aaaa + aaaa.transpose(2,3,0,1)
        aaaa += aaap[:,:,:,ncore[0]:nocc[0]]
        AAAA = einsum('tuvp,pw->tuvw', AAAP, rmat[1][:,ncore[1]:nocc[1]])
        AAAA = AAAA + AAAA.transpose(0,1,3,2)
        AAAA = AAAA + AAAA.transpose(2,3,0,1)
        AAAA += AAAP[:,:,:,ncore[1]:nocc[1]]
        tmp = (einsum('vwtp,pu->tuvw', AAap, rmat[0][:,ncore[0]:nocc[0]]),
               einsum('tuvp,pw->tuvw', aaAP, rmat[1][:,ncore[1]:nocc[1]]))
        aaAA = (tmp[0] + tmp[0].transpose(1,0,2,3) +
                tmp[1] + tmp[1].transpose(0,1,3,2))
        aaAA += aaAP[:,:,:,ncore[1]:nocc[1]]

        # pure core response
        ecore = (h1effa[:ncore[0]].trace() + h1effb[:ncore[1]].trace() +
                 einsum('jp,pj->', jka[:ncore[0]], rmat[0][:,:ncore[0]])*2 +
                 einsum('jp,pj->', jkb[:ncore[1]], rmat[1][:,:ncore[1]])*2)

        return ecore, h1cas, (aaaa, aaAA, AAAA)

    def solve_approx_ci(self, h1, h2, ci0, ecore, e_cas):
        ''' Solve CI eigenvalue/response problem approximately
        '''
        ncas = self.ncas
        nelecas = self.nelecas

        if getattr(self.fcisolver, 'approx_kernel', None):
            ci1 = self.fcisolver.approx_kernel(h1, h2, ncas, nelecas, ci0=ci0)[1]
            return ci1, None

        h2eff = self.fcisolver.absorb_h1e(h1, h2, ncas, nelecas, .5)
        hc = self.fcisolver.contract_2e(h2eff, ci0, ncas, nelecas).ravel()

        g = hc - (e_cas-ecore) * ci0.ravel()
        if self.ci_response_space > 6:
            logger.debug(self, 'CI step by full response')
            # full response
            e, ci1 = self.fcisolver.kernel(h1, h2, ncas, nelecas, ci0=ci0,
                                           max_memory=self.max_memory)
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
                if np.linalg.norm(dx) < 1e-3:
                    break
                xs.append(dx)
                ax.append(self.fcisolver.contract_2e(h2eff, xs[i], ncas,
                                                     nelecas).ravel())
                for j in range(i+1):
                    heff[i,j] = heff[j,i] = np.dot(xs[i], ax[j])
                    seff[i,j] = seff[j,i] = np.dot(xs[i], xs[j])
            nd = len(xs)
            e, v = pyscf.lib.safe_eigh(heff[:nd,:nd], seff[:nd,:nd])[:2]
            ci1 = 0
            for i in range(nd):
                ci1 += xs[i] * v[i,0]
        return ci1, g

    def dump_chk(self, envs):
        if not self.chkfile:
            return self

        if self.chk_ci:
            civec = envs['fcivec']
        else:
            civec = None
        ncore = self.ncore
        ncas = self.ncas
        nocca = ncore[0] + ncas
        noccb = ncore[1] + ncas
        if 'mo' in envs:
            mo_coeff = envs['mo']
        else:
            mo_coeff = envs['mo']
        mo_occ = np.zeros((2,envs['mo'][0].shape[1]))
        mo_occ[0,:ncore[0]] = 1
        mo_occ[1,:ncore[1]] = 1
        if self.natorb:
            occa, ucas = self._eig(-envs['casdm1'][0], ncore[0], nocca)
            occb, ucas = self._eig(-envs['casdm1'][1], ncore[1], noccb)
            mo_occ[0,ncore[0]:nocca] = -occa
            mo_occ[1,ncore[1]:noccb] = -occb
        else:
            mo_occ[0,ncore[0]:nocca] = envs['casdm1'][0].diagonal()
            mo_occ[1,ncore[1]:noccb] = envs['casdm1'][1].diagonal()
        mo_energy = 'None'

        chkfile.dump_mcscf(self, self.chkfile, 'mcscf', envs['e_tot'],
                           mo_coeff, ncore, ncas, mo_occ,
                           mo_energy, envs['e_cas'], civec, envs['casdm1'],
                           overwrite_mol=False)
        return self

    def rotate_mo(self, mo, u, log=None):
        '''Rotate orbitals with the given unitary matrix'''
        mo_a = np.dot(mo[0], u[0])
        mo_b = np.dot(mo[1], u[1])
        if log is not None and log.verbose >= logger.DEBUG:
            ncore = self.ncore[0]
            ncas = self.ncas
            nocc = ncore + ncas
            s = reduce(np.dot, (mo_a[:,ncore:nocc].T, self._scf.get_ovlp(),
                                   self.mo_coeff[0][:,ncore:nocc]))
            log.debug('Alpha active space overlap to initial guess, SVD = %s',
                      np.linalg.svd(s)[1])
            log.debug('Alpha active space overlap to last step, SVD = %s',
                      np.linalg.svd(u[0][ncore:nocc,ncore:nocc])[1])
        return mo_a, mo_b

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

CASSCF = UCASSCF

class UCASSCFBigCAS(UCASSCF):
    """
    UCASSCF with big CAS space, using a O(N5) algorithm.
    """
    gen_g_hop = gen_g_hop_big_cas

# to avoid calculating AO integrals
def _fake_h_for_fast_casci(casscf, mo, eris, eri_ab_mid=True):
    mc = copy.copy(casscf)
    mc.mo_coeff = mo
    # vhf for core density matrix
    s = mc._scf.get_ovlp()
    if isinstance(s, np.ndarray) and s.ndim == 2:
        mo_inv = (np.dot(mo[0].T, s), np.dot(mo[1].T, s))
    elif len(s) == 2:
        mo_inv = (np.dot(mo[0].T, s[0]), np.dot(mo[1].T, s[1]))
    else:
        raise ValueError
    vjk =(einsum('ipq->pq', eris.jkcpp) + eris.jC_pp,
          einsum('ipq->pq', eris.jkcPP) + eris.jc_PP)

    vhf =(reduce(np.dot, (mo_inv[0].T, vjk[0], mo_inv[0])),
          reduce(np.dot, (mo_inv[1].T, vjk[1], mo_inv[1])))
    mc.get_veff = lambda *args: vhf

    ncas = casscf.ncas
    ncore = casscf.ncore
    nocc = (ncas + ncore[0], ncas + ncore[1])
    if eri_ab_mid:
        eri_cas = np.asarray((eris.aapp[:,:,ncore[0]:nocc[0],ncore[0]:nocc[0]],
                              eris.aaPP[:,:,ncore[1]:nocc[1],ncore[1]:nocc[1]],
                              eris.AAPP[:,:,ncore[1]:nocc[1],ncore[1]:nocc[1]]))
    else: # libdmet's convention aa, bb, ab
        eri_cas = np.asarray((eris.aapp[:,:,ncore[0]:nocc[0],ncore[0]:nocc[0]],
                              eris.AAPP[:,:,ncore[1]:nocc[1],ncore[1]:nocc[1]],
                              eris.aaPP[:,:,ncore[1]:nocc[1],ncore[1]:nocc[1]]))
    mc.get_h2eff = lambda *args: eri_cas
    return mc

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.mcscf import addons

    mol = gto.Mole()
    mol.verbose = 0
    mol.output = None#"out_h2o"
    mol.atom = [
        ['H', ( 1.,-1.    , 0.   )],
        ['H', ( 0.,-1.    ,-1.   )],
        ['H', ( 1.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-1.   )],
        ['H', ( 0.,-0.5   ,-0.   )],
        ['H', ( 0.,-0.    ,-1.   )],
        ['H', ( 1.,-0.5   , 0.   )],
        ['H', ( 0., 1.    , 1.   )],
    ]

    mol.basis = {'H': 'sto-3g'}
    mol.charge = 1
    mol.spin = 1
    mol.build()

    m = scf.UHF(mol)
    ehf = m.scf()
    mc = UCASSCF(m, 4, (2,1))
    #mo = m.mo_coeff
    mo = addons.sort_mo(mc, m.mo_coeff, [(3,4,5,6),(3,4,6,7)], 1)
    emc = kernel(mc, mo, verbose=4)[1]
    print(ehf, emc, emc-ehf)
    print(emc - -2.9782774463926618)


    mol.atom = [
        ['O', ( 0., 0.    , 0.   )],
        ['H', ( 0., -0.757, 0.587)],
        ['H', ( 0., 0.757 , 0.587)],]
    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.symmetry = 1
    mol.charge = 1
    mol.spin = 1
    mol.build()

    m = scf.UHF(mol)
    ehf = m.scf()
    mc = UCASSCF(m, 4, (2,1))
    mc.verbose = 4
    emc = mc.mc1step()[0]
    print(ehf, emc, emc-ehf)
    print(emc - -75.5644202701263, emc - -75.573930418500652,
          emc - -75.574137883405612, emc - -75.648547447838951)


    mc = UCASSCF(m, 4, (2,1))
    mc.verbose = 4
    mo = mc.sort_mo((3,4,6,7))
    emc = mc.mc1step(mo)[0]
    print(ehf, emc, emc-ehf)
    print(emc - -75.5644202701263, emc - -75.573930418500652,
          emc - -75.574137883405612, emc - -75.648547447838951)

