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
#

'''
Non-relativistic Generalized Kohn-Sham for periodic systems with k-point sampling

See Also:
    pyscf.pbc.dft.rks.py : Non-relativistic Restricted Kohn-Sham for periodic
                           systems at a single k-point
'''

import numpy as np
from scipy import linalg as la

from pyscf import lib
from pyscf.lib import logger
from pyscf.pbc.scf import kghf
from pyscf.pbc.dft import gen_grid
from pyscf.pbc.dft import rks
from pyscf.pbc.dft import multigrid
from pyscf import __config__

def get_veff(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
             kpts=None, kpts_band=None):
    """
    Coulomb + XC functional for GKS.  See pyscf/pbc/dft/gks.py
    :func:`get_veff` fore more details.
    """
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    t0 = (logger.process_clock(), logger.perf_counter())

    omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
    hybrid = abs(hyb) > 1e-10

    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        n, exc, vxc = multigrid.nr_gks(ks.with_df, ks.xc, dm, hermi,
                                       kpts, kpts_band,
                                       with_j=True, return_j=False)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
        return vxc
    
    # ndim = 3 : dm.shape = (nkpts, nso, nso)
    ground_state = (dm.ndim == 3 and kpts_band is None)
    
    assert hermi == 1
    dm = np.asarray(dm)
    nso = dm.shape[-1]
    nao = nso // 2
    dm_a = dm[..., :nao, :nao]
    dm_b = dm[..., nao:, nao:]
    
    if ks.grids.non0tab is None:
        ks.grids.build(with_non0tab=True)
        if (isinstance(ks.grids, gen_grid.BeckeGrids) and
            ks.small_rho_cutoff > 1e-20 and ground_state):
            # ZHC NOTE
            ks.grids = rks.prune_small_rho_grids_(ks, cell, dm, ks.grids, kpts)
        t0 = logger.timer(ks, 'setting up grids', *t0)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        n, exc, vxc = ks._numint.nr_uks(cell, ks.grids, ks.xc, (dm_a, dm_b), 0,
                                        kpts, kpts_band)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
    
    vxc = [np.asarray(la.block_diag(*vxc[:, k]), dtype=np.result_type(dm, vxc))
           for k in range(len(kpts))]
    vxc = np.asarray(vxc)

    weight = 1./len(kpts)

    if not hybrid:
        vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
        vxc = vxc + vj
    else:
        if getattr(ks.with_df, '_j_only', False):  # for GDF and MDF
            ks.with_df._j_only = False
        vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
        vk *= hyb
        if abs(omega) > 1e-10:
            vklr = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
            vklr *= (alpha - hyb)
            vk += vklr
        vxc = vxc + vj - vk

        if ground_state:
            exc -= np.einsum('Kij, Kji', dm, vk).real * .5 * weight

    if ground_state:
        ecoul = np.einsum('Kij, Kji', dm, vj).real * .5 * weight
    else:
        ecoul = None

    vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    return vxc

def get_veff_ph(ks, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1,
                kpts=None, kpts_band=None, skip_jk=False):
    """
    Coulomb + XC functional for GKS, with P-H transform.
    """
    if cell is None: cell = ks.cell
    if dm is None: dm = ks.make_rdm1()
    if kpts is None: kpts = ks.kpts
    t0 = (logger.process_clock(), logger.perf_counter())

    omega, alpha, hyb = ks._numint.rsh_and_hybrid_coeff(ks.xc, spin=cell.spin)
    hybrid = abs(hyb) > 1e-10
    
    assert hermi == 1
    dm = np.asarray(dm)
    nso = dm.shape[-1]
    nao = nso // 2
    
    dm_ks = np.array(dm, copy=True)
    nkpts = dm_ks.shape[-3]
    ovlp = ks.get_ovlp()
    for k in range(nkpts):
        dm_ks[k, nao:, nao:] = la.inv(ovlp[k, nao:, nao:]) - dm_ks[k, nao:, nao:]
    dm_a = dm_ks[..., :nao, :nao]
    dm_b = dm_ks[..., nao:, nao:]

    if not hybrid and isinstance(ks.with_df, multigrid.MultiGridFFTDF):
        n, exc, vxc = multigrid.nr_gks(ks.with_df, ks.xc, dm_ks, hermi,
                                       kpts, kpts_band,
                                       with_j=True, return_j=False)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        vxc[..., nao:, nao:] *= -1
        t0 = logger.timer(ks, 'vxc', *t0)
        return vxc
    
    # ndim = 3 : dm.shape = (nkpts, nso, nso)
    ground_state = (dm.ndim == 3 and kpts_band is None)
    
    if ks.grids.non0tab is None:
        ks.grids.build(with_non0tab=True)
        if (isinstance(ks.grids, gen_grid.BeckeGrids) and
            ks.small_rho_cutoff > 1e-20 and ground_state):
            # ZHC NOTE
            ks.grids = rks.prune_small_rho_grids_(ks, cell, dm_ks, ks.grids, kpts)
        t0 = logger.timer(ks, 'setting up grids', *t0)

    if hermi == 2:  # because rho = 0
        n, exc, vxc = (0,0), 0, 0
    else:
        n, exc, vxc = ks._numint.nr_uks(cell, ks.grids, ks.xc, (dm_a, dm_b),
                                        hermi=hermi, kpts=kpts, kpts_band=kpts_band)
        logger.debug(ks, 'nelec by numeric integration = %s', n)
        t0 = logger.timer(ks, 'vxc', *t0)
    
    vxc = [np.asarray(la.block_diag(vxc[0, k], -vxc[1, k]), dtype=np.result_type(dm, vxc))
           for k in range(len(kpts))]
    vxc = np.asarray(vxc)
    weight = 1./len(kpts)
    
    if not skip_jk:
        if not hybrid:
            vj = ks.get_j(cell, dm, hermi, kpts, kpts_band)
            vxc = vxc + vj
        else:
            if getattr(ks.with_df, '_j_only', False):  # for GDF and MDF
                ks.with_df._j_only = False
            vj, vk = ks.get_jk(cell, dm, hermi, kpts, kpts_band)
            vk *= hyb
            if abs(omega) > 1e-10:
                vklr = ks.get_k(cell, dm, hermi, kpts, kpts_band, omega=omega)
                vklr *= (alpha - hyb)
                vk += vklr
            vxc = vxc + vj - vk

            if ground_state:
                exc -= np.einsum('Kij, Kji', dm, vk).real * .5 * weight

        if ground_state:
            ecoul = np.einsum('Kij, Kji', dm, vj).real * .5 * weight
        else:
            ecoul = None
        
        vxc = lib.tag_array(vxc, ecoul=ecoul, exc=exc, vj=None, vk=None)
    else:
        vxc = lib.tag_array(vxc, ecoul=0.0, exc=exc, vj=None, vk=None)

    return vxc

def energy_elec(mf, dm_kpts=None, h1e_kpts=None, vhf=None):
    if h1e_kpts is None: h1e_kpts = mf.get_hcore(mf.cell, mf.kpts)
    if dm_kpts is None: dm_kpts = mf.make_rdm1()
    if vhf is None or getattr(vhf, 'ecoul', None) is None:
        vhf = mf.get_veff(mf.cell, dm_kpts)
    
    h1e_kpts = np.array(h1e_kpts, copy=True) # copy for subrtact Mu
    nkpts, nso, _ = h1e_kpts.shape
    nao = nso // 2
    ovlp = mf.get_ovlp()
    if getattr(mf, "Mu", None) is not None:
        h1e_kpts[:, :nao, :nao] += ovlp[:, :nao, :nao] * mf.Mu
        h1e_kpts[:, nao:, nao:] -= ovlp[:, nao:, nao:] * mf.Mu

    weight = 1./len(h1e_kpts)
    e1 = weight * (np.einsum('kij, kji', h1e_kpts, dm_kpts))
    tot_e = e1 + vhf.ecoul + vhf.exc
    mf.scf_summary['e1'] = e1.real
    mf.scf_summary['coul'] = vhf.ecoul.real
    mf.scf_summary['exc'] = vhf.exc.real
    logger.debug(mf, 'E1 = %s  Ecoul = %s  Exc = %s', e1, vhf.ecoul, vhf.exc)
    return tot_e.real, vhf.ecoul + vhf.exc

#@lib.with_doc(kghf.get_rho.__doc__)
def get_rho(mf, dm=None, grids=None, kpts=None):
    from pyscf.pbc.dft import krks
    if dm is None:
        dm = mf.make_rdm1()
    dm = np.asarray(dm)
    nso = dm.shape[-1]
    nao = nso // 2
    dm_a = dm[..., :nao, :nao]
    dm_b = dm[..., nao:, nao:]
    return krks.get_rho(mf, dm_a+dm_b, grids, kpts)

class KGKS(kghf.KGHF, rks.KohnShamDFT):
    """
    GKS class adapted for PBCs with k-point sampling.
    """
    def __init__(self, cell, kpts=np.zeros((1,3)), xc='LDA,VWN',
                 exxdiv=getattr(__config__, 'pbc_scf_SCF_exxdiv', 'ewald')):
        kghf.KGHF.__init__(self, cell, kpts, exxdiv=exxdiv)
        rks.KohnShamDFT.__init__(self, xc)

    def dump_flags(self, verbose=None):
        kghf.KGHF.dump_flags(self, verbose)
        rks.KohnShamDFT.dump_flags(self, verbose)
        return self

    get_veff = get_veff
    energy_elec = energy_elec
    get_rho = get_rho

    density_fit = rks._patch_df_beckegrids(kghf.KGHF.density_fit)
    mix_density_fit = rks._patch_df_beckegrids(kghf.KGHF.mix_density_fit)
    def nuc_grad_method(self):
        from pyscf.pbc.grad import kgks
        return kgks.Gradients(self)

if __name__ == '__main__':
    from pyscf.pbc import gto
    cell = gto.Cell()
    cell.unit = 'A'
    cell.atom = 'C 0.,  0.,  0.; C 0.8917,  0.8917,  0.8917'
    cell.a = '''0.      1.7834  1.7834
                1.7834  0.      1.7834
                1.7834  1.7834  0.    '''

    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.verbose = 7
    cell.output = '/dev/null'
    cell.build()
    mf = KGKS(cell, cell.make_kpts([2,1,1]))
    print(mf.kernel())
