#! /usr/bin/env python

"""
CASCI solver for GSO formalism.

Author:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

import numpy as np
import scipy.linalg as la

from libdmet.system import integral
from libdmet.solver import block, scf
from libdmet.solver.scf import _get_veff_ghf
from libdmet.solver.dmrgci import (CASCI, gaopt, momopt, reorder,
                                   split_localize, check_sanity_cas,
                                   match_cas_basis, orbital_reorder_subspace) 
from libdmet.utils import logger as log
from libdmet.utils.misc import mdot, Iterable

def cas_from_1pdm(rho, ncas, nelecas, nelec, tol=0.3):
    assert nelecas <= nelec
    natocc, natorb = la.eigh(rho)
    log.debug(1, "Natural orbital occupations:\n%s", natocc)
    norbs = natocc.shape[0] # spin orbitals
    check_sanity_cas(norbs//2, nelec, ncas, nelecas) 
     
    ncore = (nelec - nelecas) // 2
    nvirt = norbs // 2 - ncore - ncas
    
    log.info("ncore = %d nvirt = %d ncas = %d", ncore, nvirt, ncas)
    log.info("core orbital occupation cut-off = %20.12f", 
             natocc[-ncore*2] if ncore > 0 else 1)
    log.info("virt orbital occupation cut-off = %20.12f", 
             natocc[nvirt*2-1] if nvirt > 0 else 0)
    
    if ncore == 0:
        casocc = natocc[nvirt*2:]
    else:
        casocc = natocc[nvirt*2:-ncore*2]
    _nvirt = np.sum(casocc < tol)
    _ncore = np.sum(casocc > (1.0 - tol))
    _npart = np.sum((casocc >= tol) * (casocc <= (1.0 - tol)))
    log.info("In CAS:\n"
             "Occupied (n > %.2g): %d\n""Virtual  (n < %.2g): %d\n"
             "Partial Occupied: %d\n", (1.0 - tol), _ncore, tol, _nvirt, _npart)
    
    core = natorb[:, norbs-ncore*2:]
    cas  = natorb[:, range(norbs-ncore*2-1, nvirt*2-1, -1)]
    virt = natorb[:, :nvirt*2]
    return core, cas, virt, (_ncore, _npart, _nvirt)

def cas_from_energy(mo, mo_energy, ncas, nelecas, nelec):
    assert nelecas <= nelec
    log.debug(1, "Orbital energies:\n%s", mo_energy)
    norbs = mo_energy.shape[0]
    check_sanity_cas(norbs//2, nelec, ncas, nelecas) 
    
    ncore = (nelec - nelecas) // 2
    nvirt = norbs // 2 - ncore - ncas
    
    log.info("ncore = %d nvirt = %d ncas = %d", ncore, nvirt, ncas)
    log.info("core orbital energy cut-off = %20.12f", 
             mo_energy[ncore*2-1] if ncore > 0 else np.inf)
    log.info("virt orbital eneryg cut-off = %20.12f", 
             mo_energy[-nvirt*2] if nvirt > 0 else -np.inf)
    
    if nvirt == 0:
        casenergy = mo_energy[ncore*2:]
    else:
        casenergy = mo_energy[ncore*2:-nvirt*2]
    mu = 0.5 * (casenergy[nelecas-1] + casenergy[nelecas])
    log.debug(0, "HF gap = %20.12f", casenergy[nelecas] - casenergy[nelecas-1])
    _nvirt = np.sum(casenergy > mu+1e-4)
    _ncore = np.sum(casenergy < mu-1e-4)
    _npart = np.sum((casenergy >= mu-1e-4) * (casenergy <= mu+1e-4))
    log.info("In CAS:\n"
             "Occupied (e<mu): %d\n""Virtual  (e>mu): %d\n"
             "Partial Occupied: %d\n", _ncore, _nvirt, _npart)
    
    core = mo[:, :ncore*2]
    cas  = mo[:, ncore*2:norbs-nvirt*2]
    virt = mo[:, norbs-nvirt*2:]
    return core, cas, virt, (_ncore, _npart, _nvirt)

def get_orbs(casci, Ham, guess, nelec, return_rdm1=False, scf_tol=1e-12,
             order=None, scf_max_cycle=100, beta=np.inf):
    """
    Returns:
        core, cas, virt, info, (rho0): no spin dim.
    """
    # ZHC TODO support for order.
    # nelec, spin, bogo, res
    casci.scfsolver.set_system(nelec, nelec, False, False)
    casci.scfsolver.set_integral(Ham)
    
    E_HF, rhoHF = casci.scfsolver.GGHF(tol=scf_tol, MaxIter=scf_max_cycle,
                                       InitGuess=guess, beta=beta)

    if casci.MP2natorb:
        E_MP2, rhoMP2 = casci.scfsolver.GMP2()
        log.result("MP2 energy = %20.12f", E_HF + E_MP2)

    rho0 = rhoHF
    if casci.MP2natorb:
        core, cas, virt, casinfo = cas_from_1pdm(rhoMP2, casci.ncas,
                                                 casci.nelecas, nelec)
    else:
        mo = casci.scfsolver.get_mo()[0]
        mo_energy = casci.scfsolver.get_mo_energy()[0]
        core, cas, virt, casinfo = cas_from_energy(mo, mo_energy, casci.ncas, 
                                                   casci.nelecas, nelec)
    
    if return_rdm1:
        return core, cas, virt, casinfo, rho0
    else:
        return core, cas, virt, casinfo

def buildCASHamiltonian(Ham, core, cas, rdm1_core=None):
    assert core.ndim == 2 and cas.ndim == 2
    if rdm1_core is None:
        rdm1_core = np.dot(core, core.conj().T)
    # zero-energy
    H0 = Ham.H0
    # core-core one-body
    H0 = H0 + np.einsum('pq, qp ->', rdm1_core, Ham.H1["cd"][0])
    # core-fock
    v = _get_veff_ghf(rdm1_core, Ham.H2["ccdd"][0])
    # core-core two-body, note the half factor for HF energy
    H0 += 0.5 * np.einsum('pq, qp ->', rdm1_core, v)
    H1 = {"cd": mdot(cas.conj().T, Ham.H1["cd"][0] + v, cas)[None]}
    
    nao = Ham.H1["cd"].shape[-1]
    eri_format, spin_dim = integral.get_eri_format(Ham.H2["ccdd"], nao)
    norb = cas.shape[-1]
    H2 = {"ccdd": scf.incore_transform(Ham.H2["ccdd"], (cas, cas, cas, cas), 
                                       compact=(eri_format != 's1'))}
    return integral.Integral(norb, True, False, H0, H1, H2)

class GSOCASCI(CASCI):
    
    def run(self, Ham, ci_args={}, guess=None, nelec=None, basis=None, 
            mom_tol=0.7, ham_only=False, hf_occ=True, order=None, 
            orbs=None, warmup_occ=False, **kwargs): 
        """
        Main kernel of CASCI.

        Args:
            Ham: hamiltonian.
            ci_args: additional args for cisolver, 
                     a list or dict for ci solver, or None.
            guess: dm0 used for scfsolver.
            nelec: number of total electrons, if None, will use half-filled, 
                   i.e. Ham.norb // 2
            basis: C_lo_eo, (ncells, nlo*2, neo), used for basis matching.
            mom_tol: tolerance for MOM reorder quality.
            ham_only: only return casHam and (cas, core, rdm1_core) without run.
            hf_occ: can be an array of elements 2, 1, 0. if True, will use HF occupancy.
            order: order used to reorder the natural orbitals.
            orbs: a tuple of (core, cas, virt, casinfo, dm).

        Returns:
            rdm1: in EO basis, same basis as Ham.
            E: energy directly from cisolver.
        """
        if nelec is None: 
            nelec = Ham.norb // 2

        # construct cas orbitals and casHam
        if orbs is None:
            scf_max_cycle = kwargs.get("scf_max_cycle", 100)
            beta = kwargs.get("beta", np.inf)
            core, cas, virt, casinfo, dm = self.get_orbs(Ham, guess, nelec,
                                                         return_rdm1=True,
                                                         order=order,
                                                         scf_max_cycle=scf_max_cycle,
                                                         beta=beta)
            rdm1_core = kwargs.get("rdm1_core", np.dot(core, core.conj().T))
        else:
            # core, cas, virt can be specified manually
            log.info("CASCI: use customized core, cas, virt")
            if len(orbs) == 3:
                core, cas, virt = orbs
                casinfo = None
                dm = None
            else:
                core, cas, virt, casinfo, dm = orbs
            
            # sanity check
            norbs = Ham.norb
            ncore = (nelec - self.nelecas) // 2
            ncas  = self.ncas
            nvirt = norbs // 2 - ncore - ncas
            log.eassert(core.shape[-1] == ncore * 2, 
                        "core shape %s does not match ncore (%s)", 
                        core.shape, ncore)
            log.eassert(cas.shape[-1] == ncas * 2, 
                        "cas shape %s does not match ncas (%s)", 
                        cas.shape, ncas)
            log.eassert(virt.shape[-1] == nvirt * 2, 
                        "virt shape %s does not match nvirt (%s)", 
                        virt.shape, nvirt)
            nelecas = self.nelecas
            check_sanity_cas(norbs, nelec, ncas, nelecas)
            
            rdm1_core = kwargs.get("rdm1_core", np.dot(core, core.conj().T))
            
            if dm is None:
                dm  = np.dot(cas[:, :nelecas], cas[:, :nelecas].conj().T)
                dm += rdm1_core
            if casinfo is None:
                casinfo = (nelecas, 0, ncas * 2 - nelecas)
        
        log.info("core orb shape: %s", str(core.shape))
        log.info("cas  orb shape: %s", str(cas.shape))
        log.info("virt orb shape: %s", str(virt.shape))

        # ZHC NOTE casinfo can be manually set
        if self.dyn_corr_method is not None:
            log.eassert(kwargs.get("casinfo", None) is not None, 
                        "dyn_corr_method needs casinfo in run().")
        if kwargs.get("casinfo", None):
            casinfo = kwargs.get("casinfo")
            log.info("You are using casinfo from input... %s", casinfo)
        
        # check the basis shape are the same.
        basis_shape_unchanged = (self.basis_old is not None) and \
                                (self.basis_old.shape == basis.shape)
        # match cas orbital first.
        # and update C_eo_mo_old and basis_old.
        if basis_shape_unchanged:
            u_mat, diff = match_cas_basis(basis[None], cas[None],
                                          self.basis_old[None], 
                                          self.C_eo_mo_old[None], [casinfo])
            u_mat = u_mat[0]
            cas = np.dot(cas, u_mat)
        else:
            if self.basis_old is not None:
                log.warn("CASCI solver: basis shape changes [from %s to %s],\n"
                         "this is usually due to the truncation of bath. \n"
                         "Make sure the nelec (%s) is set correctly!", 
                         str(self.basis_old.shape), str(basis.shape), nelec)
        if basis is not None:
            self.basis_old = np.array(basis)
        self.C_eo_mo_old = np.array(cas, copy=True)
        casHam = buildCASHamiltonian(Ham, core, cas, rdm1_core=rdm1_core)
        
        if isinstance(hf_occ, str):
            hf_occ = hf_occ
        elif isinstance(hf_occ, Iterable):
            hf_occ = np.asarray(hf_occ, dtype=int)
            assert len(hf_occ) == self.ncas * 2
        elif hf_occ == True:
            hf_occ = []
            for i in range(self.ncas * 2):
                hf_occ.append(int(i < self.nelecas))
            hf_occ = np.asarray(hf_occ, dtype=int)
        
        # split localization
        # and save the C_mo_lmo_old for the next step guess.
        if self.splitloc:
            # check whether the basis shape changes
            if basis_shape_unchanged:
                loc_guess = self.C_mo_lmo_old[None]
            else:
                loc_guess = self.loc_guess
            match_basis_ghf = kwargs.get("match_basis_ghf", False)
            cas, C_mo_lmo, C_mo_lmo_no_reorder = \
                    split_localize(cas[None], [casinfo], casHam,
                                   basis=basis[None], Ham_eo=Ham,
                                   method=self.loc_method, 
                                   guess=loc_guess, 
                                   tol=self.loc_conv_tol, 
                                   return_Ham=False,
                                   match_basis_ghf=match_basis_ghf)[1:]
            cas = cas[0]
            C_mo_lmo = C_mo_lmo[0]
            C_mo_lmo_no_reorder = C_mo_lmo_no_reorder[0]
            if match_basis_ghf:
                pair_idx = dict(casHam.pair_idx)
                if not isinstance(hf_occ, str): # hf_occ should be reordered
                    order_pair = []
                    for idx in casHam.pair_idx:
                        order_pair.extend([idx[0], idx[1] + self.nelecas])
                    hf_occ = hf_occ[order_pair]
            else:
                pair_idx = None

            casHam = None
            casHam = buildCASHamiltonian(Ham, core, cas, rdm1_core=rdm1_core)
            self.C_mo_lmo_old = C_mo_lmo_no_reorder.copy()
        else:
            pair_idx = None

        # MOM reorder compared with old basis
        restart_flag = ci_args.get("restart", False)
        if self.mom_reorder:
            log.eassert(basis is not None, 
                        "maximum overlap method (MOM) requires embedding basis")
            if self.C_lo_lmo_old is None or (not basis_shape_unchanged):
                if pair_idx is None:
                    order = gaopt(casHam, tmp=self.tmpDir, nproc=self.gaopt_nthreads,
                                  fiedler=self.fiedler)
                else:
                    order = gaopt(casHam, tmp=self.tmpDir, nproc=self.gaopt_nthreads,
                                  fiedler=self.fiedler,
                                  select_idx=np.arange(casHam.norb)[::2])
                    
                    order = np.asarray(order) * 2
                    order_new = []
                    for o in order:
                        order_new.extend([o, o+1])
                    order = np.asarray(order_new)
                if "restart" in ci_args:
                    ci_args["restart"] = False
            else:
                # define C_lo_lmo
                C_lo_lmo = np.tensordot(basis, cas, (-1, 0))
                # C_lo_lmo and self.C_lo_lmo_old are both in
                # atomic representation now
                order, q = momopt(self.C_lo_lmo_old[None], C_lo_lmo[None])
                # if the quality of mom is too bad, we reorder the orbitals
                # using genetic algorithm
                if q < mom_tol:
                    if pair_idx is None:
                        order = gaopt(casHam, tmp=self.tmpDir, nproc=self.gaopt_nthreads,
                                      fiedler=self.fiedler)
                    else:
                        order = gaopt(casHam, tmp=self.tmpDir, nproc=self.gaopt_nthreads,
                                      fiedler=self.fiedler,
                                      select_idx=np.arange(casHam.norb)[::2])
                        
                        order = np.asarray(order) * 2
                        order_new = []
                        for o in order:
                            order_new.extend([o, o+1])
                        order = np.asarray(order_new)

                    if "restart" in ci_args:
                        ci_args["restart"] = False
            
            if self.dyn_corr_method:
                # ZHC NOTE reorder according to the subspace
                order = orbital_reorder_subspace(order, casinfo[0], casinfo[-1])
            
            log.info("Orbital order: %s", order)
            # reorder casHam and cas
            casHam, cas = reorder(order, casHam, cas[None])
            cas = cas[0]
            # store cas in atomic basis
            self.C_lo_lmo_old = np.tensordot(basis, cas, (-1, 0))
            # reorder hf_occ for DMRG
            if isinstance(hf_occ, Iterable) and (not isinstance(hf_occ, str)):
                hf_occ = hf_occ[order]
        
        # save cas orbital and core for interacting bath
        self.core = core
        self.cas  = cas
        self.virt = virt

        # check if solver needs restart and basis
        if ci_args.get("restart", False):
            ci_args["basis"] = np.tensordot(basis, self.cas, (-1, 0))
        
        # hf_occ for DMRG solver
        log.debug(1, "hf_occ: %s", hf_occ)
        ci_args["hf_occ"] = hf_occ
        if isinstance(warmup_occ, Iterable):
            warmup_occ = np.einsum('pi, p, pi -> i', 
                                   self.cas.conj(), warmup_occ, self.cas,
                                   optimize=True)
            ci_args["occ"] = warmup_occ
        elif warmup_occ:
            warmup_occ = hf_occ
            ci_args["occ"] = warmup_occ
        log.debug(1, "warmup_occ: %s", warmup_occ)

        # casinfo for DMRG MRCI
        if self.dyn_corr_method is not None:
            ci_args["dyn_corr_method"] = self.dyn_corr_method
            ci_args["casinfo"] = casinfo
        
        # initial dm0 guess for possible addtional mf in the cas solver.
        # assume ovlp is indentity.
        dm0_cas = mdot(self.cas.conj().T, dm - rdm1_core, self.cas)
        if "dm0" not in ci_args:
            ci_args["dm0"] = dm0_cas
            
        np.save("%s_core.npy"   %(self.name), self.core)
        np.save("%s_cas.npy"    %(self.name), self.cas)
        np.save("%s_virt.npy"   %(self.name), self.virt)
        np.save("%s_hf_occ.npy" %(self.name), hf_occ)
        casHam.save("%s_casHam.h5" %(self.name))
        
        if ham_only:
            # only build Ham without run.
            return casHam, (self.cas, self.core, rdm1_core, hf_occ)
        else:
            # run CI solver, and rotate back dm, 
            rdm1_cas, E = self.cisolver.run(casHam, nelec=self.nelecas,
                                            spin=self.nelecas, **ci_args)
            # uhf-type rdm1
            if rdm1_cas.ndim == 3:
                rdm1_cas = rdm1_cas.sum(axis=0)
            
            rdm1 = mdot(cas, rdm1_cas, cas.conj().T) + rdm1_core
            
            # remove "dm0" for possible reuse
            ci_args.pop("dm0", None)
            # resume the old restart flag
            if "restart" in ci_args:
                ci_args["restart"] = restart_flag
            return rdm1, E
    
    get_orbs = get_orbs

    buildCASHamiltonian = buildCASHamiltonian

GSODmrgCI = GSOCASCI
