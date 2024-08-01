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
#         Qiming Sun <osirpt.sun@gmail.com>
#

'''
GCASSCF (CASSCF with genralized spin orbitals)
2-step optimization algorithm
'''


import numpy as np
import copy
import pyscf.lib.logger as logger

def kernel(casscf, mo_coeff, tol=1e-7, conv_tol_grad=None,
           ci0=None, callback=None, verbose=None, dump_chk=True):
    if verbose is None:
        verbose = casscf.verbose
    log = logger.Logger(casscf.stdout, verbose)
    cput0 = (logger.process_clock(), logger.perf_counter())
    log.debug('Start 2-step CASSCF')

    mo = mo_coeff
    nmo = mo.shape[-1]
    ncore = casscf.ncore
    eris = casscf.ao2mo(mo)
    e_tot, e_cas, fcivec = casscf.casci(mo, ci0, eris, log, locals())
    if casscf.ncas * 2 == nmo and \
            not (casscf.internal_rotation or casscf.internal_rotation_vo):
        return True, e_tot, e_cas, fcivec, mo, None

    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(tol)
        logger.info(casscf, 'Set conv_tol_grad to %g', conv_tol_grad)
    conv_tol_ddm = conv_tol_grad * 3
    conv = False
    de, elast = e_tot, e_tot
    totmicro = totinner = 0
    casdm1 = 0
    r0 = None

    t2m = t1m = log.timer('Initializing 2-step CASSCF', *cput0)
    imacro = 0
    while not conv and imacro < casscf.max_cycle_macro:
        imacro += 1
        njk = 0
        t3m = t2m
        casdm1_old = casdm1
        casdm1, casdm2 = casscf.fcisolver.make_rdm12(fcivec, casscf.ncas*2, casscf.nelecas)
        norm_ddm = np.linalg.norm(casdm1 - casdm1_old)
        t3m = log.timer('update CAS DM', *t3m)
        max_cycle_micro = 1 # casscf.micro_cycle_scheduler(locals())
        max_stepsize = casscf.max_stepsize_scheduler(locals())
        for imicro in range(max_cycle_micro):
            rota = casscf.rotate_orb_cc(mo, lambda:fcivec, lambda:casdm1, lambda:casdm2,
                                        eris, r0, conv_tol_grad*.3, max_stepsize, log)
            u, g_orb, njk1, r0 = next(rota)
            rota.close()
            njk += njk1
            norm_t = np.linalg.norm(u-np.eye(nmo))
            norm_gorb = np.linalg.norm(g_orb)
            if imicro == 0:
                norm_gorb0 = norm_gorb
            t3m = log.timer('orbital rotation', *t3m)

            eris = None
            u = copy.copy(u)
            g_orb = copy.copy(g_orb)
            mo = casscf.rotate_mo(mo, u, log)
            eris = casscf.ao2mo(mo)
            t3m = log.timer('update eri', *t3m)

            log.debug('micro %d  |u-1|=%5.3g  |g[o]|=%5.3g  |dm1|=%5.3g',
                      imicro, norm_t, norm_gorb, norm_ddm)

            if callable(callback):
                callback(locals())

            t2m = log.timer('micro iter %d'%imicro, *t2m)
            if norm_t < 1e-4 or norm_gorb < conv_tol_grad*.5:
                break

        totinner += njk
        totmicro += imicro+1

        e_tot, e_cas, fcivec = casscf.casci(mo, fcivec, eris, log, locals())
        log.timer('CASCI solver', *t3m)
        t2m = t1m = log.timer('macro iter %d'%imacro, *t1m)

        de, elast = e_tot - elast, e_tot
        if (abs(de) < tol and
            norm_gorb < conv_tol_grad and norm_ddm < conv_tol_ddm):
            conv = True

        if dump_chk:
            casscf.dump_chk(locals())

        if callable(callback):
            callback(locals())

    if conv:
        log.info('2-step CASSCF converged in %d macro (%d JK %d micro) steps',
                 imacro+1, totinner, totmicro)
    else:
        log.info('2-step CASSCF not converged, %d macro (%d JK %d micro) steps',
                 imacro+1, totinner, totmicro)
    log.timer('2-step CASSCF', *cput0)
    return conv, e_tot, e_cas, fcivec, mo, None


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
    import os
    from pyscf.lib import chkfile
    chkfname = "test.chk"

    mf = scf.RHF(mol)
    mf.chkfile = chkfname
    if os.path.exists(chkfname):
        data = chkfile.load(chkfname, 'scf')
        mf.__dict__.update(data)
    else:
        mf.kernel()
    
    mf = scf.addons.convert_to_uhf(mf)
    mc = UCASSCF(mf, 4, 4)
    #mc.internal_rotation = True
    mc.mc2step()

    # GCAS
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
    from libdmet.solver.gmc1step import GCASSCF
    mc = GCASSCF(mf, 4, (4,0))
    #mc.internal_rotation = True
    
    mc.mc2step()

