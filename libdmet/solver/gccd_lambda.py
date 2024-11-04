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
#
# Author: Zhi-Hao Cui <zhcui0408@gmail.com>
#

import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.cc import ccsd_lambda

einsum = lib.einsum

def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           max_cycle=50, tol=1e-8, verbose=logger.INFO):
    if eris is None: eris = mycc.ao2mo()
    return ccsd_lambda.kernel(mycc, eris, t1, t2, l1, l2, max_cycle, tol,
                              verbose, make_intermediates, update_lambda)

# l2, t2 as ijab
def make_intermediates(mycc, t1, t2, eris):
    nocc, nvir = t1.shape
    foo = eris.fock[:nocc,:nocc]
    fov = eris.fock[:nocc,nocc:]
    fvo = eris.fock[nocc:,:nocc]
    fvv = eris.fock[nocc:,nocc:]

    class _IMDS: pass
    imds = _IMDS()
    imds.ftmp = lib.H5TmpFile()
    dtype = numpy.result_type(t2, eris.vvvv).char
    imds.woooo = imds.ftmp.create_dataset('woooo', (nocc,nocc,nocc,nocc), dtype)
    imds.wovvo = imds.ftmp.create_dataset('wovvo', (nocc,nvir,nvir,nocc), dtype)
    imds.wovoo = imds.ftmp.create_dataset('wovoo', (nocc,nvir,nocc,nocc), dtype)
    imds.wvvvo = imds.ftmp.create_dataset('wvvvo', (nvir,nvir,nvir,nocc), dtype)

    tau = t2

    v1  = einsum('jkca,jkbc->ba', eris.oovv, tau) * .5
    v1 += fvv

    v2  = einsum('ikbc,jkbc->ij', eris.oovv, tau) * .5
    v2 += foo

    woooo  = einsum('ijcd,klcd->ijkl', eris.oovv, tau) * 0.25
    woooo += numpy.asarray(eris.oooo) * 0.5
    imds.woooo[:] = woooo
    woooo = None

    v4  = einsum('ljdb, klcd -> jcbk', eris.oovv, t2)
    v4 += numpy.asarray(eris.ovvo)

    v5  = fvo + numpy.einsum('kc,jkbc->bj', fov, t2, optimize=True)
    v5 -= einsum('kljc, klbc -> bj', eris.ooov, t2) * .5
    v5 += einsum('kbdc, jkcd -> bj', eris.ovvv, t2) * .5

    w3  = v5 #+ numpy.einsum('jcbk,jb->ck', v4, t1, optimize=True)
    #w3 += numpy.einsum('cb,jb->cj', v1, t1, optimize=True)
    #w3 -= numpy.einsum('jk,jb->bk', v2, t1, optimize=True)

    #wovvo  = numpy.einsum('ljdb,lc,kd->jcbk', eris.oovv, t1, -t1, optimize=True)
    wovvo = v4
    #wovvo -= einsum('ljkb,lc->jcbk', eris.ooov, t1)
    #wovvo += einsum('jcbd,kd->jcbk', eris.ovvv, t1)
    imds.wovvo[:] = wovvo
    wovvo = None
    v4 = None

    wovoo  = einsum('icdb, jkdb -> icjk', eris.ovvv, tau) * .25
    wovoo += numpy.einsum('jkic -> icjk', numpy.asarray(eris.ooov).conj(), optimize=True) * .5
    #wovoo += einsum('icbk, jb -> icjk', v4, t1)
    wovoo -= einsum('lijb, klcb -> icjk', eris.ooov, t2)
    imds.wovoo[:] = wovoo
    wovoo = None

    #wvvvo = einsum('jcak,jb->bcak', v4, t1)
    wvvvo = einsum('jlka,jlbc->bcak', eris.ooov, tau) * .25
    wvvvo -= numpy.einsum('jacb->bcaj', numpy.asarray(eris.ovvv).conj(), optimize=True) * .5
    wvvvo += einsum('kbad,jkcd->bcaj', eris.ovvv, t2)
    imds.wvvvo[:] = wvvvo
    wvvvo = None

    imds.v1 = v1
    imds.v2 = v2
    imds.w3 = w3
    imds.ftmp.flush()
    return imds

# update L1, L2
def update_lambda(mycc, t1, t2, l1, l2, eris, imds):
    time0 = logger.process_clock(), logger.perf_counter()
    log = logger.Logger(mycc.stdout, mycc.verbose)
    nocc, nvir = t1.shape
    fov = eris.fock[:nocc,nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + mycc.level_shift
    v1 = imds.v1 - numpy.diag(mo_e_v)
    v2 = imds.v2 - numpy.diag(mo_e_o)

    #l1new = numpy.array(fov, copy=True)
    l1new = numpy.zeros_like(l1)

    mba = einsum('klca, klcb -> ba', l2, t2) * .5
    mij = einsum('kicd, kjcd -> ij', l2, t2) * .5

    oovv = numpy.asarray(eris.oovv)
    #tau  = numpy.einsum('ia, jb -> ijab', t1, t1 * 2.0, optimize=True)
    tau = t2
    tmp = einsum('ijcd, klcd -> ijkl', l2, tau)
    tau = None
    m3 = einsum('klab,ijkl->ijab', l2, numpy.asarray(imds.woooo))
    m3 += einsum('klab,ijkl->ijab', oovv, tmp) * .25
    tmp = None

    #tmp = einsum('ijcd,kd -> ijck', l2, t1)
    #m3 -= einsum('kcba,ijck->ijab', eris.ovvv, tmp)
    #tmp = None
    m3 += einsum('ijcd,cdab->ijab', l2, eris.vvvv) * .5

    l2new = numpy.array(oovv, copy=True)
    l2new += m3
    #l1new += einsum('ijab,jb->ia', m3, t1)
    m3 = None

    fov1 = fov #+ einsum('kjcb,kc->jb', oovv, t1)
    #tmp  = einsum('ia,jb->ijab', l1, fov1)
    tmp = einsum('kica,jcbk->ijab', l2, numpy.asarray(imds.wovvo))
    tmp = tmp - tmp.transpose(1,0,2,3)
    l2new += tmp - tmp.transpose(0,1,3,2)
    tmp = None

    #tmp  = einsum('ka, ijkb -> ijab', l1, eris.ooov)
    tmp = einsum('ijca, cb -> ijab', l2, v1)
    tmp1vv = mba #+ einsum('ka, kb -> ba', l1, t1)
    tmp += einsum('ca, ijcb -> ijab', tmp1vv, oovv)
    l2new -= tmp - tmp.transpose(0,1,3,2)
    tmp = None

    #tmp  = einsum('ic, jcba -> jiba', l1, eris.ovvv)
    tmp = einsum('kiab, jk -> ijab', l2, v2)
    tmp1oo = mij #+ einsum('ic,kc->ik', l1, t1)
    tmp -= einsum('ik,kjab->ijab', tmp1oo, oovv)
    l2new += tmp - tmp.transpose(1,0,2,3)
    tmp = None

    #l1new += einsum('jb, ibaj -> ia', l1, eris.ovvo)
    #l1new += einsum('ib, ba -> ia', l1, v1)
    #l1new -= einsum('ja, ij -> ia', l1, v2)
    #l1new -= einsum('kjca, icjk -> ia', l2, imds.wovoo)
    #l1new -= einsum('ikbc, bcak -> ia', l2, imds.wvvvo)
    #l1new += einsum('jiba, bj -> ia', l2, imds.w3)
    #tmp =(t1 + einsum('kc,kjcb->jb', l1, t2)
    #      - einsum('bd,jd->jb', tmp1vv, t1)
    #      - einsum('lj,lb->jb', mij, t1))
    #l1new += numpy.einsum('jiba, jb -> ia', oovv, tmp, optimize=True)
    #l1new += numpy.einsum('icab, bc -> ia', eris.ovvv, tmp1vv, optimize=True)
    #l1new -= numpy.einsum('jika, kj -> ia', eris.ooov, tmp1oo, optimize=True)
    #tmp = fov - einsum('kjba, jb -> ka', oovv, t1)
    oovv = None
    #l1new -= numpy.einsum('ik, ka -> ia', mij, tmp, optimize=True)
    #l1new -= numpy.einsum('ca, ic -> ia', mba, tmp, optimize=True)

    eia = lib.direct_sum('i-j->ij', mo_e_o, mo_e_v)
    #l1new /= eia
    for i in range(nocc):
        l2new[i] /= lib.direct_sum('a, jb -> jab', eia[i], eia)

    time0 = log.timer_debug1('update l1 l2', *time0)
    return l1new, l2new

if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import gccsd

    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]
    mol.basis = '631g'
    mol.spin = 0
    mol.build()
    mf = scf.RHF(mol).run()
    mf0 = mf
    mf = scf.addons.convert_to_ghf(mf)
    mycc = gccsd.GCCSD(mf)
    eris = mycc.ao2mo()
    mycc.kernel(eris=eris)
    l1, l2 = mycc.solve_lambda(mycc.t1, mycc.t2, eris=eris)
    l1 = mycc.spin2spatial(l1, mycc.mo_coeff.orbspin)
    l2 = mycc.spin2spatial(l2, mycc.mo_coeff.orbspin)
    print(lib.finger(l1[0]) --0.0030030170069977758)
    print(lib.finger(l1[1]) --0.0030030170069977758)
    print(lib.finger(l2[0]) --0.041444910588788492 )
    print(lib.finger(l2[1]) - 0.1077575086912813   )
    print(lib.finger(l2[2]) --0.041444910588788492 )
    print(abs(l2[1]-l2[1].transpose(1,0,2,3)-l2[0]).max())
    print(abs(l2[1]-l2[1].transpose(0,1,3,2)-l2[0]).max())

    from pyscf.cc import ccsd
    mycc0 = ccsd.CCSD(mf0)
    eris0 = mycc0.ao2mo()
    mycc0.kernel(eris=eris0)
    t1 = mycc0.t1
    t2 = mycc0.t2
    imds = ccsd_lambda.make_intermediates(mycc0, t1, t2, eris0)
    l1, l2 = ccsd_lambda.update_lambda(mycc0, t1, t2, t1, t2, eris0, imds)
    l1ref, l2ref = ccsd_lambda.update_lambda(mycc0, t1, t2, l1, l2, eris0, imds)
    t1 = mycc.spatial2spin(t1, mycc.mo_coeff.orbspin)
    t2 = mycc.spatial2spin(t2, mycc.mo_coeff.orbspin)
    l1 = mycc.spatial2spin(l1, mycc.mo_coeff.orbspin)
    l2 = mycc.spatial2spin(l2, mycc.mo_coeff.orbspin)
    imds = make_intermediates(mycc, t1, t2, eris)
    l1, l2 = update_lambda(mycc, t1, t2, l1, l2, eris, imds)
    l1 = mycc.spin2spatial(l1, mycc.mo_coeff.orbspin)
    l2 = mycc.spin2spatial(l2, mycc.mo_coeff.orbspin)
    print(abs(l1[0]-l1ref).max())
    print(abs(l2[1]-l2ref).max())
