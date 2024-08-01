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
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Jun Yang
#         Zhi-Hao Cui <zhcui0408@gmail.com>
#

import numpy as np
from pyscf import lib
from pyscf.cc.uccsd_rdm import (_gamma1_intermediates, make_rdm1,
                                _make_rdm1, _dm2ab_mo2ao)

einsum = lib.einsum

# gamma2 intermediates in Chemist's notation
def _gamma2_outcore(cc, t1, t2, l1, l2, h5fobj, compress_vvvv=False):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2

    tauaa  = np.einsum('ia, jb -> ijab', t1a * 2.0, t1a)
    tauaa += t2aa
    tauab  = np.einsum('ia, jb -> ijab', t1a, t1b)
    tauab += t2ab
    taubb  = np.einsum('ia, jb -> ijab', t1b * 2.0, t1b)
    taubb += t2bb

    miajb  = einsum('ikac, kjcb -> iajb', l2aa, t2aa)
    miajb += einsum('ikac, jkbc -> iajb', l2ab, t2ab)

    miaJB  = einsum('ikac, kjcb -> iajb', l2aa, t2ab)
    miaJB += einsum('ikac, kjcb -> iajb', l2ab, t2bb)

    mIAjb  = einsum('kica, jkbc -> iajb', l2bb, t2ab)
    mIAjb += einsum('kica, kjcb -> iajb', l2ab, t2aa)

    mIAJB  = einsum('ikac, kjcb -> iajb', l2bb, t2bb)
    mIAJB += einsum('kica, kjcb -> iajb', l2ab, t2ab)

    miAjB = einsum('ikca, jkcb -> iajb', l2ab, t2ab)

    mIaJb = einsum('kiac, kjbc -> iajb', l2ab, t2ab)

    # oovv
    goovv = (l2aa.conj() + tauaa) * 0.25
    goOvV = (l2ab.conj() + tauab) * 0.5
    gOOVV = (l2bb.conj() + taubb) * 0.25

    tmpa  = einsum('kc,kica->ia', l1a, t2aa)
    tmpa += einsum('kc,ikac->ia', l1b, t2ab)
    tmpb  = einsum('kc,kica->ia', l1b, t2bb)
    tmpb += einsum('kc,kica->ia', l1a, t2ab)
    goovv += einsum('ia,jb->ijab', tmpa, t1a)
    goOvV += einsum('ia,jb->ijab', tmpa * 0.5, t1b)
    goOvV += einsum('ia,jb->jiba', tmpb * 0.5, t1a)
    gOOVV += einsum('ia,jb->ijab', tmpb, t1b)

    tmpa = einsum('kc,kb->cb', l1a, t1a)
    tmpb = einsum('kc,kb->cb', l1b, t1b)
    goovv += einsum('cb,ijca->ijab', tmpa * 0.5, t2aa)
    goOvV -= einsum('cb,ijac->ijab', tmpb * 0.5, t2ab)
    goOvV -= einsum('cb,jica->jiba', tmpa * 0.5, t2ab)
    gOOVV += einsum('cb,ijca->ijab', tmpb * 0.5, t2bb)
    tmpa = einsum('kc,jc->kj', l1a, t1a)
    tmpb = einsum('kc,jc->kj', l1b, t1b)
    goovv += einsum('kiab,kj->ijab', tauaa, tmpa * 0.5)
    goOvV -= einsum('ikab,kj->ijab', tauab , tmpb * 0.5)
    goOvV -= einsum('kiba,kj->jiba', tauab , tmpa * 0.5)
    gOOVV += einsum('kiab,kj->ijab', taubb, tmpb * 0.5)

    tmpa  = np.einsum('ldjd->lj', miajb)
    tmpa += np.einsum('ldjd->lj', miAjB)
    tmpb  = np.einsum('ldjd->lj', mIAJB)
    tmpb += np.einsum('ldjd->lj', mIaJb)
    goovv -= einsum('lj,liba->ijab', tmpa * 0.25, tauaa)
    goOvV -= einsum('lj,ilab->ijab', tmpb * 0.25, tauab)
    goOvV -= einsum('lj,liba->jiba', tmpa * 0.25, tauab)
    gOOVV -= einsum('lj,liba->ijab', tmpb * 0.25, taubb)
    tmpa  = np.einsum('ldlb->db', miajb)
    tmpa += np.einsum('ldlb->db', mIaJb)
    tmpb  = np.einsum('ldlb->db', mIAJB)
    tmpb += np.einsum('ldlb->db', miAjB)
    goovv -= einsum('db,jida->ijab', tmpa * 0.25, tauaa)
    goOvV -= einsum('db,ijad->ijab', tmpb * 0.25, tauab)
    goOvV -= einsum('db,jida->jiba', tmpa * 0.25, tauab)
    gOOVV -= einsum('db,jida->ijab', tmpb * 0.25, taubb)

    goovv -= einsum('ldia,ljbd->ijab', miajb, tauaa) * .5
    goovv += einsum('LDia,jLbD->ijab', mIAjb, t2ab ) * .5
    gOOVV -= einsum('ldia,ljbd->ijab', mIAJB, taubb) * .5
    gOOVV += einsum('ldia,ljdb->ijab', miaJB, t2ab ) * .5
    goOvV -= einsum('LDia,LJBD->iJaB', mIAjb, taubb) * .25
    goOvV += einsum('ldia,lJdB->iJaB', miajb, t2ab ) * .25
    goOvV -= einsum('ldIA,ljbd->jIbA', miaJB, tauaa) * .25
    goOvV += einsum('LDIA,jLbD->jIbA', mIAJB, t2ab ) * .25
    goOvV += einsum('lDiA,lJbD->iJbA', miAjB, tauab) * .5
    goOvV += einsum('LdIa,jd,LB->jIaB', mIaJb, t1a, t1b * 0.5)

    tmpaa = einsum('klcd,ijcd->ijkl', l2aa, tauaa) * .25**2
    goovv += einsum('ijkl,klab->ijab', tmpaa, tauaa)
    tmpaa = None

    tmpabab = einsum('kLcD,iJcD->iJkL', l2ab, tauab) * .5
    goOvV += einsum('ijkl,klab->ijab', tmpabab, tauab)
    tmpabab = None

    tmpbb = einsum('klcd,ijcd->ijkl', l2bb, taubb) * .25**2
    gOOVV += einsum('ijkl,klab->ijab', tmpbb, taubb)
    tmpbb = None

    # ovov, OVOV, ovOV
    goovv = goovv.conj()
    dovov = goovv.transpose(0,2,1,3) - goovv.transpose(0,3,1,2)
    dovov =(dovov + dovov.transpose(2,3,0,1)) * .5
    h5fobj['dovov'] = dovov
    dovov = goovv = None

    gOOVV = gOOVV.conj()
    dOVOV = gOOVV.transpose(0,2,1,3) - gOOVV.transpose(0,3,1,2)
    dOVOV =(dOVOV + dOVOV.transpose(2,3,0,1)) * .5
    h5fobj['dOVOV'] = dOVOV
    dOVOV = gOOVV = None

    goOvV = goOvV.conj()
    h5fobj['dovOV'] = goOvV.transpose(0,2,1,3)
    goOvV = None

    dOVov = None

    # vvvv
    gvvvv = einsum('ijab,ijcd->abcd', tauaa, l2aa * 0.125)
    dvvvv = gvvvv.transpose(0,2,1,3) - gvvvv.transpose(0,3,1,2)
    dvvvv = dvvvv + dvvvv.transpose(1,0,3,2).conj()
    h5fobj['dvvvv'] = dvvvv
    dvvvv = None

    # ovvv
    tmpa  = np.einsum('kakb->ab', miajb) * .25
    tmpa += np.einsum('kakb->ab', mIaJb) * .25
    tmpb  = np.einsum('kakb->ab', mIAJB) * .25
    tmpb += np.einsum('kakb->ab', miAjB) * .25

    govvv = einsum('ja,ijcb->iacb', .25 * l1a, tauaa)
    govvv += einsum('bcad,id->iabc', gvvvv, t1a)
    gvvvv = None
    govvv += einsum('ab,ic->iacb', tmpa, t1a)
    govvv += einsum('kaib,kc->iabc', miajb, .5 * t1a)
    govvv = govvv.conj()
    govvv += einsum('ijbc,ja->iabc', l2aa, .25*t1a)
    dovvv = govvv.transpose(0,2,1,3) - govvv.transpose(0,3,1,2)
    govvv = None
    h5fobj['dovvv'] = dovvv
    dovvv = None

    # vvVV
    gvVvV = einsum('ijab,ijcd->abcd', tauab, l2ab * 0.25)
    dvvVV = gvVvV.transpose(0,2,1,3) * 2
    dvvVV = dvvVV + dvvVV.transpose(1,0,3,2).conj()
    h5fobj['dvvVV'] = dvvVV
    dvvVV = None

    # ovVV
    goVvV = einsum('ja,ijcb->iacb', .5  * l1b, tauab)
    goVvV -= einsum('bcda,id->iabc', gvVvV, t1a * 2.0)
    goVvV += einsum('ab,ic->iacb', tmpb, t1a)
    goVvV += einsum('KAib,KC->iAbC', mIAjb, 0.5 * t1b)
    goVvV -= einsum('kAiB,kc->iAcB', miAjB, 0.5 * t1a)
    goVvV = goVvV.conj()
    goVvV += einsum('iJbC,JA->iAbC', l2ab, 0.5 *t1b)
    h5fobj['dovVV'] = goVvV.transpose(0,2,1,3)
    goVvV = None

    # VVVV
    gVVVV = einsum('ijab,ijcd->abcd', taubb, l2bb * 0.125)
    dVVVV = gVVVV.transpose(0,2,1,3) - gVVVV.transpose(0,3,1,2)
    dVVVV = dVVVV + dVVVV.transpose(1,0,3,2).conj()
    h5fobj['dVVVV'] = dVVVV
    dVVVV = None

    # OVVV
    gOVVV = einsum('ja,ijcb->iacb', .25 * l1b, taubb)
    gOVVV += einsum('bcad,id->iabc', gVVVV, t1b)
    gVVVV = None
    gOVVV += einsum('ab,ic->iacb', tmpb, t1b)
    gOVVV += einsum('kaib,kc->iabc', mIAJB, .5 * t1b)
    gOVVV = gOVVV.conj()
    gOVVV += einsum('ijbc,ja->iabc', l2bb, .25*t1b)
    dOVVV = gOVVV.transpose(0,2,1,3) - gOVVV.transpose(0,3,1,2)
    gOVVV = None
    h5fobj['dOVVV'] = dOVVV
    dOVVV = None

    # OVvv
    gOvVv = einsum('ja,jibc->iacb', .5  * l1a, tauab)
    gOvVv -= einsum('cbad,id->iabc', gvVvV, t1b * 2.0)
    gOvVv += einsum('ab,ic->iacb', tmpa, t1b)
    gOvVv += einsum('kaIB,kc->IaBc', miaJB, .5 * t1a)
    gOvVv -= einsum('KaIb,KC->IaCb', mIaJb, .5 * t1b)
    gOvVv = gOvVv.conj()
    gOvVv += einsum('jIcB,ja->IaBc', l2ab, .5 *t1a)
    dOVvv = gOvVv.transpose(0,2,1,3)
    gOvVv = None
    h5fobj['dOVvv'] = dOVvv
    dOVvv = None

    # oooo
    goooo = einsum('ijab,klab->ijkl', l2aa, tauaa) * .125
    doooo = goooo.transpose(0,2,1,3) - goooo.transpose(0,3,1,2)
    doooo = doooo + doooo.transpose(1,0,3,2).conj()
    h5fobj['doooo'] = doooo
    doooo = None

    # ooov
    tmpa  = np.einsum('icjc->ij', miajb) * .25
    tmpa += np.einsum('icjc->ij', miAjB) * .25
    tmpb  = np.einsum('icjc->ij', mIAJB) * .25
    tmpb += np.einsum('icjc->ij', mIaJb) * .25

    gooov = einsum('jkba,ib->jkia', tauaa, -0.25 * l1a)
    gooov += einsum('iljk,la->jkia', goooo, t1a)
    goooo = None
    gooov -= einsum('ij,ka->jkia', tmpa, t1a)
    gooov += einsum('icja,kc->jkia', miajb, .5 * t1a)
    gooov = gooov.conj()
    gooov += einsum('jkab,ib->jkia', l2aa, .25*t1a)
    dooov = gooov.transpose(0,2,1,3) - gooov.transpose(1,2,0,3)
    h5fobj['dooov'] = dooov
    dooov = None

    # ooOO
    goOoO = einsum('ijab,klab->ijkl', l2ab, tauab) * .25
    dooOO = goOoO.transpose(0,2,1,3) * 2
    dooOO = dooOO + dooOO.transpose(1,0,3,2).conj()
    h5fobj['dooOO'] = dooOO
    dooOO = None

    # ooOV
    goOoV = einsum('jkba,ib->jkia', tauab, -0.5  * l1a)
    goOoV += einsum('iljk,la->jkia', goOoO, t1b * 2.0)
    goOoV -= einsum('ij,ka->jkia', tmpa, t1b)
    goOoV += einsum('icja,kc->jkia', miAjB, .5 * t1b)
    goOoV -= einsum('icJA,kc->kJiA', miaJB, .5 * t1a)
    goOoV = goOoV.conj()
    goOoV -= einsum('jkba,ib->jkia', l2ab, .5 *t1a)
    dooOV = goOoV.transpose(0,2,1,3)
    h5fobj['dooOV'] = dooOV
    dooOV = goOoV = None

    # OOov
    gOoOv = einsum('kjab,ib->jkia', tauab, -0.5  * l1b)
    gOoOv += einsum('likj,la->jkia', goOoO, t1a * 2.0)
    goOoO = None
    gOoOv -= einsum('ij,ka->jkia', tmpb, t1a)
    gOoOv += einsum('icja,kc->jkia', mIaJb, .5 * t1a)
    gOoOv -= einsum('ICja,KC->KjIa', mIAjb, .5 * t1b)
    gOoOv = gOoOv.conj()
    gOoOv -= einsum('kjab,ib->jkia', l2ab, .5 *t1b)
    dOOov = gOoOv.transpose(0,2,1,3)
    h5fobj['dOOov'] = dOOov
    dOOov = gOoOv = None

    # OOOO
    gOOOO = einsum('ijab,klab->ijkl', l2bb, taubb) * .125
    dOOOO = gOOOO.transpose(0,2,1,3) - gOOOO.transpose(0,3,1,2)
    dOOOO = dOOOO + dOOOO.transpose(1,0,3,2).conj()
    h5fobj['dOOOO'] = dOOOO
    dOOOO = None

    # OOOV
    gOOOV = einsum('jkba,ib->jkia', taubb, -0.25 * l1b)
    gOOOV += einsum('iljk,la->jkia', gOOOO, t1b)
    gOOOO = None
    gOOOV -= einsum('ij,ka->jkia', tmpb, t1b)
    gOOOV += einsum('icja,kc->jkia', mIAJB, .5 * t1b)
    gOOOV = gOOOV.conj()
    gOOOV += einsum('jkab,ib->jkia', l2bb, .25*t1b)
    dOOOV = gOOOV.transpose(0,2,1,3) - gOOOV.transpose(1,2,0,3)
    gOOOV = None
    h5fobj['dOOOV'] = dOOOV
    dOOOV = None

    # ovvo
    # iajb -> ibaj, miajb
    govvo  = miajb.transpose(0, 3, 1, 2)
    govvo += np.einsum('ia, jb -> ibaj', l1a, t1a)
    govvo -= einsum('ikac,jc,kb->ibaj', l2aa, t1a, t1a)
    dovvo = govvo.transpose(0,2,1,3)
    dovvo = (dovvo + dovvo.transpose(3,2,1,0).conj()) * .5
    h5fobj['dovvo'] = dovvo
    doovv = -dovvo.transpose(0,3,2,1)
    h5fobj['doovv'] = doovv
    doovv = dovvo = None
    dvvov = None

    # ovVO, OVvo
    # iajb->ibaj, miaJB
    goVvO  = miaJB.transpose(0, 3, 1, 2)
    goVvO += np.einsum('ia, jb -> ibaj', l1a, t1b)
    goVvO -= einsum('iKaC,JC,KB->iBaJ', l2ab, t1b, t1b)
    dovVO = goVvO.transpose(0,2,1,3)
    goVvO = None

    # iajb->ibaj, mIAjb
    gOvVo  = mIAjb.transpose(0, 3, 1, 2)
    gOvVo += np.einsum('ia, jb -> ibaj', l1b, t1a)
    gOvVo -= einsum('kIcA, jc, kb -> IbAj', l2ab, t1a, t1a)
    dOVvo = gOvVo.transpose(0,2,1,3)
    gOvVo = None
    dovVO = (dovVO + dOVvo.transpose(3,2,1,0).conj()) * .5
    dOVvo =  dovVO.transpose(3,2,1,0).conj()
    h5fobj['dovVO'] = dovVO
    h5fobj['dOVvo'] = dOVvo
    dovVO = dOVvo = None

    # OVVO, OOVV
    # iajb->ibaj, mIAJB
    gOVVO  = mIAJB.transpose(0, 3, 1, 2)
    gOVVO += np.einsum('ia, jb -> ibaj', l1b, t1b)
    gOVVO -= einsum('ikac,jc,kb->ibaj', l2bb, t1b, t1b)
    dOVVO = gOVVO.transpose(0,2,1,3)
    dOVVO = (dOVVO + dOVVO.transpose(3,2,1,0).conj()) * .5
    dOOVV = -dOVVO.transpose(0,3,2,1)
    h5fobj['dOVVO'] = dOVVO
    h5fobj['dOOVV'] = dOOVV
    gOVVO = None
    dOVVO = None
    dOOVV = None

    # ooVV
    # iajb->ibja, miAjB
    goVoV  = miAjB.transpose(0, 3, 2, 1)
    goVoV += einsum('iKcA,jc,KB->iBjA', l2ab, t1a, t1b)
    dooVV = goVoV.transpose(0,2,1,3)
    dooVV = (dooVV + dooVV.transpose(1,0,3,2).conj()) * (-0.5)
    h5fobj['dooVV'] = dooVV
    dooVV = goVoV = None

    # OOvv
    # iajb->ibja, mIaJb
    gOvOv  = mIaJb.transpose(0, 3, 2, 1)
    gOvOv += einsum('kIaC,JC,kb->IbJa', l2ab, t1b, t1a)
    dOOvv = gOvOv.transpose(0,2,1,3)
    dOOvv = (dOOvv + dOOvv.transpose(1,0,3,2).conj()) * (-0.5)
    h5fobj['dOOvv'] = dOOvv
    dOOvv = gOOvv = None

    #dVVOV = None
    #dvvOV = None
    #dVVvv = None
    #dOOoo = None
    #dVVov = None

    if compress_vvvv:
        raise NotImplementedError
    return ((h5fobj['dovov'], h5fobj['dovOV'], None, h5fobj['dOVOV']),
            (h5fobj['dvvvv'], h5fobj['dvvVV'], None, h5fobj['dVVVV']),
            (h5fobj['doooo'], h5fobj['dooOO'], None, h5fobj['dOOOO']),
            (h5fobj['doovv'], h5fobj['dooVV'], h5fobj['dOOvv'], h5fobj['dOOVV']),
            (h5fobj['dovvo'], h5fobj['dovVO'], h5fobj['dOVvo'], h5fobj['dOVVO']),
            (None, None, None, None),
            (h5fobj['dovvv'], h5fobj['dovVV'], h5fobj['dOVvv'], h5fobj['dOVVV']),
            (h5fobj['dooov'], h5fobj['dooOV'], h5fobj['dOOov'], h5fobj['dOOOV']))

def _gamma2_intermediates(mycc, t1, t2, l1, l2, compress_vvvv=False):
    f = lib.H5TmpFile()
    _gamma2_outcore(mycc, t1, t2, l1, l2, f, compress_vvvv)

    d2 = ((f['dovov'][:], f['dovOV'][:], None, f['dOVOV'][:]),
          (f['dvvvv'][:], f['dvvVV'][:], None, f['dVVVV'][:]),
          (f['doooo'][:], f['dooOO'][:], None, f['dOOOO'][:]),
          (f['doovv'][:], f['dooVV'][:], f['dOOvv'][:], f['dOOVV'][:]),
          (f['dovvo'][:], f['dovVO'][:], f['dOVvo'][:], f['dOVVO'][:]),
          (None, None, None, None),
          (f['dovvv'][:], f['dovVV'][:], f['dOVvv'][:], f['dOVVV'][:]),
          (f['dooov'][:], f['dooOV'][:], f['dOOov'][:], f['dOOOV'][:]))
    return d2

# spin-orbital rdm2 in Chemist's notation
def make_rdm2(mycc, t1, t2, l1, l2, ao_repr=False):
    r'''
    Two-particle spin density matrices dm2aa, dm2ab, dm2bb in MO basis

    dm2aa[p,q,r,s] = <q_alpha^\dagger s_alpha^\dagger r_alpha p_alpha>
    dm2ab[p,q,r,s] = <q_alpha^\dagger s_beta^\dagger r_beta p_alpha>
    dm2bb[p,q,r,s] = <q_beta^\dagger s_beta^\dagger r_beta p_beta>

    (p,q correspond to one particle and r,s correspond to another particle)
    Two-particle density matrix should be contracted to integrals with the
    pattern below to compute energy

    E = np.einsum('pqrs,pqrs', eri_aa, dm2_aa)
    E+= np.einsum('pqrs,pqrs', eri_ab, dm2_ab)
    E+= np.einsum('pqrs,rspq', eri_ba, dm2_ab)
    E+= np.einsum('pqrs,pqrs', eri_bb, dm2_bb)

    where eri_aa[p,q,r,s] = (p_alpha q_alpha | r_alpha s_alpha )
    eri_ab[p,q,r,s] = ( p_alpha q_alpha | r_beta s_beta )
    eri_ba[p,q,r,s] = ( p_beta q_beta | r_alpha s_alpha )
    eri_bb[p,q,r,s] = ( p_beta q_beta | r_beta s_beta )
    '''
    d1 = _gamma1_intermediates(mycc, t1, t2, l1, l2)
    d2 = _gamma2_intermediates(mycc, t1, t2, l1, l2)
    return _make_rdm2(mycc, d1, d2, with_dm1=True, with_frozen=True,
                      ao_repr=ao_repr)

def _make_rdm2(mycc, d1, d2, with_dm1=True, with_frozen=True, ao_repr=False):
    dovov, dovOV, dOVov, dOVOV = d2[0]
    dvvvv, dvvVV, dVVvv, dVVVV = d2[1]
    doooo, dooOO, dOOoo, dOOOO = d2[2]
    doovv, dooVV, dOOvv, dOOVV = d2[3]
    dovvo, dovVO, dOVvo, dOVVO = d2[4]
    dvvov, dvvOV, dVVov, dVVOV = d2[5]
    dovvv, dovVV, dOVvv, dOVVV = d2[6]
    dooov, dooOV, dOOov, dOOOV = d2[7]
    nocca, nvira, noccb, nvirb = dovOV.shape
    nmoa = nocca + nvira
    nmob = noccb + nvirb

    dm2aa = np.empty((nmoa,nmoa,nmoa,nmoa), dtype=doovv.dtype)
    dm2ab = np.empty((nmoa,nmoa,nmob,nmob), dtype=doovv.dtype)
    dm2bb = np.empty((nmob,nmob,nmob,nmob), dtype=doovv.dtype)

# dm2aa
    dovov = np.asarray(dovov)
    dm2aa[:nocca,nocca:,:nocca,nocca:] = dovov
    dm2aa[nocca:,:nocca,nocca:,:nocca] = dm2aa[:nocca,nocca:,:nocca,nocca:].transpose(1,0,3,2).conj()
    dovov = None

    #assert(abs(doovv+dovvo.transpose(0,3,2,1)).max() == 0)
    dovvo = np.asarray(dovvo)
    dm2aa[:nocca,:nocca,nocca:,nocca:] =-dovvo.transpose(0,3,2,1)
    dm2aa[nocca:,nocca:,:nocca,:nocca] = dm2aa[:nocca,:nocca,nocca:,nocca:].transpose(2,3,0,1)
    dm2aa[:nocca,nocca:,nocca:,:nocca] = dovvo
    dm2aa[nocca:,:nocca,:nocca,nocca:] = dm2aa[:nocca,nocca:,nocca:,:nocca].transpose(1,0,3,2).conj()
    dovvo = None

    if len(dvvvv.shape) == 2:
        dvvvv = ao2mo.restore(1, dvvvv, nvira)
    dm2aa[nocca:,nocca:,nocca:,nocca:] = dvvvv
    dm2aa[:nocca,:nocca,:nocca,:nocca] = doooo

    dovvv = np.asarray(dovvv)
    dm2aa[:nocca,nocca:,nocca:,nocca:] = dovvv
    dm2aa[nocca:,nocca:,:nocca,nocca:] = dovvv.transpose(2,3,0,1)
    dm2aa[nocca:,nocca:,nocca:,:nocca] = dovvv.transpose(3,2,1,0).conj()
    dm2aa[nocca:,:nocca,nocca:,nocca:] = dovvv.transpose(1,0,3,2).conj()
    dovvv = None

    dooov = np.asarray(dooov)
    dm2aa[:nocca,:nocca,:nocca,nocca:] = dooov
    dm2aa[:nocca,nocca:,:nocca,:nocca] = dooov.transpose(2,3,0,1)
    dm2aa[:nocca,:nocca,nocca:,:nocca] = dooov.transpose(1,0,3,2).conj()
    dm2aa[nocca:,:nocca,:nocca,:nocca] = dooov.transpose(3,2,1,0).conj()
    dooov = None

# dm2bb
    dOVOV = np.asarray(dOVOV)
    dm2bb[:noccb,noccb:,:noccb,noccb:] = dOVOV
    dm2bb[noccb:,:noccb,noccb:,:noccb] = dm2bb[:noccb,noccb:,:noccb,noccb:].transpose(1,0,3,2).conj()
    dOVOV = None

    dOVVO = np.asarray(dOVVO)
    dm2bb[:noccb,:noccb,noccb:,noccb:] =-dOVVO.transpose(0,3,2,1)
    dm2bb[noccb:,noccb:,:noccb,:noccb] = dm2bb[:noccb,:noccb,noccb:,noccb:].transpose(2,3,0,1)
    dm2bb[:noccb,noccb:,noccb:,:noccb] = dOVVO
    dm2bb[noccb:,:noccb,:noccb,noccb:] = dm2bb[:noccb,noccb:,noccb:,:noccb].transpose(1,0,3,2).conj()
    dOVVO = None

    if len(dVVVV.shape) == 2:
        dVVVV = ao2mo.restore(1, dVVVV, nvirb)
    dm2bb[noccb:,noccb:,noccb:,noccb:] = dVVVV
    dm2bb[:noccb,:noccb,:noccb,:noccb] = dOOOO

    dOVVV = np.asarray(dOVVV)
    dm2bb[:noccb,noccb:,noccb:,noccb:] = dOVVV
    dm2bb[noccb:,noccb:,:noccb,noccb:] = dOVVV.transpose(2,3,0,1)
    dm2bb[noccb:,noccb:,noccb:,:noccb] = dOVVV.transpose(3,2,1,0).conj()
    dm2bb[noccb:,:noccb,noccb:,noccb:] = dOVVV.transpose(1,0,3,2).conj()
    dOVVV = None

    dOOOV = np.asarray(dOOOV)
    dm2bb[:noccb,:noccb,:noccb,noccb:] = dOOOV
    dm2bb[:noccb,noccb:,:noccb,:noccb] = dOOOV.transpose(2,3,0,1)
    dm2bb[:noccb,:noccb,noccb:,:noccb] = dOOOV.transpose(1,0,3,2).conj()
    dm2bb[noccb:,:noccb,:noccb,:noccb] = dOOOV.transpose(3,2,1,0).conj()
    dOOOV = None

# dm2ab
    dovOV = np.asarray(dovOV)
    dm2ab[:nocca,nocca:,:noccb,noccb:] = dovOV
    dm2ab[nocca:,:nocca,noccb:,:noccb] = dm2ab[:nocca,nocca:,:noccb,noccb:].transpose(1,0,3,2).conj()
    dovOV = None

    dovVO = np.asarray(dovVO)
    dm2ab[:nocca,:nocca,noccb:,noccb:] = dooVV
    dm2ab[nocca:,nocca:,:noccb,:noccb] = dOOvv.transpose(2,3,0,1)
    dm2ab[:nocca,nocca:,noccb:,:noccb] = dovVO
    dm2ab[nocca:,:nocca,:noccb,noccb:] = dovVO.transpose(1,0,3,2).conj()
    dovVO = None

    if len(dvvVV.shape) == 2:
        idxa = np.tril_indices(nvira)
        dvvVV1 = lib.unpack_tril(dvvVV)
        dvvVV = np.empty((nvira,nvira,nvirb,nvirb))
        dvvVV[idxa] = dvvVV1
        dvvVV[idxa[1],idxa[0]] = dvvVV1
        dvvVV1 = None
    dm2ab[nocca:,nocca:,noccb:,noccb:] = dvvVV
    dm2ab[:nocca,:nocca,:noccb,:noccb] = dooOO

    dovVV = np.asarray(dovVV)
    dm2ab[:nocca,nocca:,noccb:,noccb:] = dovVV
    dm2ab[nocca:,nocca:,:noccb,noccb:] = dOVvv.transpose(2,3,0,1)
    dm2ab[nocca:,nocca:,noccb:,:noccb] = dOVvv.transpose(3,2,1,0).conj()
    dm2ab[nocca:,:nocca,noccb:,noccb:] = dovVV.transpose(1,0,3,2).conj()
    dovVV = None

    dooOV = np.asarray(dooOV)
    dm2ab[:nocca,:nocca,:noccb,noccb:] = dooOV
    dm2ab[:nocca,nocca:,:noccb,:noccb] = dOOov.transpose(2,3,0,1)
    dm2ab[:nocca,:nocca,noccb:,:noccb] = dooOV.transpose(1,0,3,2).conj()
    dm2ab[nocca:,:nocca,:noccb,:noccb] = dOOov.transpose(3,2,1,0).conj()
    dooOV = None

    if with_frozen and mycc.frozen is not None:
        nmoa0 = dm2aa.shape[0]
        nmob0 = dm2bb.shape[0]
        nmoa = mycc.mo_occ[0].size
        nmob = mycc.mo_occ[1].size
        nocca = np.count_nonzero(mycc.mo_occ[0] > 0)
        noccb = np.count_nonzero(mycc.mo_occ[1] > 0)

        rdm2aa = np.zeros((nmoa,nmoa,nmoa,nmoa), dtype=dm2aa.dtype)
        rdm2ab = np.zeros((nmoa,nmoa,nmob,nmob), dtype=dm2ab.dtype)
        rdm2bb = np.zeros((nmob,nmob,nmob,nmob), dtype=dm2bb.dtype)
        moidxa, moidxb = mycc.get_frozen_mask()
        moidxa = np.where(moidxa)[0]
        moidxb = np.where(moidxb)[0]
        idxa = (moidxa.reshape(-1,1) * nmoa + moidxa).ravel()
        idxb = (moidxb.reshape(-1,1) * nmob + moidxb).ravel()
        lib.takebak_2d(rdm2aa.reshape(nmoa**2,nmoa**2),
                       dm2aa.reshape(nmoa0**2,nmoa0**2), idxa, idxa)
        lib.takebak_2d(rdm2bb.reshape(nmob**2,nmob**2),
                       dm2bb.reshape(nmob0**2,nmob0**2), idxb, idxb)
        lib.takebak_2d(rdm2ab.reshape(nmoa**2,nmob**2),
                       dm2ab.reshape(nmoa0**2,nmob0**2), idxa, idxb)
        dm2aa, dm2ab, dm2bb = rdm2aa, rdm2ab, rdm2bb

    if with_dm1:
        dm1a, dm1b = _make_rdm1(mycc, d1, with_frozen=True)
        dm1a[np.diag_indices(nocca)] -= 1
        dm1b[np.diag_indices(noccb)] -= 1

        for i in range(nocca):
            dm2aa[i,i,:,:] += dm1a
            dm2aa[:,:,i,i] += dm1a
            dm2aa[:,i,i,:] -= dm1a
            dm2aa[i,:,:,i] -= dm1a.T
            dm2ab[i,i,:,:] += dm1b
        for i in range(noccb):
            dm2bb[i,i,:,:] += dm1b
            dm2bb[:,:,i,i] += dm1b
            dm2bb[:,i,i,:] -= dm1b
            dm2bb[i,:,:,i] -= dm1b.T
            dm2ab[:,:,i,i] += dm1a

        for i in range(nocca):
            for j in range(nocca):
                dm2aa[i,i,j,j] += 1
                dm2aa[i,j,j,i] -= 1
        for i in range(noccb):
            for j in range(noccb):
                dm2bb[i,i,j,j] += 1
                dm2bb[i,j,j,i] -= 1
        for i in range(nocca):
            for j in range(noccb):
                dm2ab[i,i,j,j] += 1

    dm2aa = dm2aa.transpose(1,0,3,2)
    dm2ab = dm2ab.transpose(1,0,3,2)
    dm2bb = dm2bb.transpose(1,0,3,2)

    if ao_repr:
        from pyscf.cc import ccsd_rdm
        dm2aa = ccsd_rdm._rdm2_mo2ao(dm2aa, mycc.mo_coeff[0])
        dm2bb = ccsd_rdm._rdm2_mo2ao(dm2bb, mycc.mo_coeff[1])
        dm2ab = _dm2ab_mo2ao(dm2ab, mycc.mo_coeff[0], mycc.mo_coeff[1])
    return dm2aa, dm2ab, dm2bb

