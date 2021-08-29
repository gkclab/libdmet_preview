#! /usr/bin/env python

"""
Mean-field routines.

Only implement non-scf routines
    restricted / unrestricted
    Slater / BCS
    thermal occupation
    kscf helper functions

Author:
    Boxiao Zheng
    Zhihao Cui
"""

import numpy as np
import scipy.linalg as la
import itertools as it

from pyscf import lib
from pyscf import scf
from pyscf import ao2mo
from pyscf import pbc
from pyscf.pbc.lib.kpts_helper import get_kconserv
from pyscf.lib import logger as pyscflogger

from libdmet.utils.misc import mdot, max_abs, add_spin_dim, Iterable
from libdmet.routine import ftsystem
from libdmet.settings import IMAG_DISCARD_TOL 
from libdmet.routine.pbc_helper import *
from libdmet.utils import logger as log

flush = log.flush_for_pyscf
pyscflogger.flush = flush([""])

def DiagRHF(Fock, vcor, **kwargs):
    if Fock.ndim == 3:
        Fock = Fock[np.newaxis]
    ncells = Fock.shape[-3]
    nscsites = Fock.shape[-1]
    ew = np.empty((ncells, nscsites))
    ev = np.empty((ncells, nscsites, nscsites), dtype=np.complex128)
    if vcor is None:
        for i in range(ncells):
            ew[i], ev[i] = la.eigh(Fock[0, i])
    else:
        for i in range(ncells):
            ew[i], ev[i] = la.eigh(Fock[0, i] + vcor.get(i, True)[0])
    return ew, ev

def DiagRHF_symm(Fock, vcor, lattice, **kwargs):
    if Fock.ndim == 3:
        Fock = Fock[np.newaxis]
    ncells = Fock.shape[-3]
    nscsites = Fock.shape[-1]
    ew = np.empty((ncells, nscsites))
    ev = np.empty((ncells, nscsites, nscsites), dtype=np.complex128)
    
    computed = set()
    for i in range(ncells):
        neg_i = lattice.cell_pos2idx(-lattice.cell_idx2pos(i))
        if neg_i in computed:
            ew[i], ev[i] = ew[neg_i], ev[neg_i].conj()
        else:
            if vcor is None:
                ew[i], ev[i] = la.eigh(Fock[0, i])
            else:
                ew[i], ev[i] = la.eigh(Fock[0, i] + vcor.get(i, True)[0])
            computed.add(i)
    return ew, ev

def DiagUHF(Fock, vcor, **kwargs):
    if Fock.ndim == 3:
        Fock = np.asarray((Fock, Fock))
    ncells = Fock.shape[-3]
    nscsites = Fock.shape[-1]
    ew = np.empty((2, ncells, nscsites))
    ev = np.empty((2, ncells, nscsites, nscsites), dtype=np.complex128)
    if vcor is None:
        for i in range(ncells):
            ew[0][i], ev[0][i] = la.eigh(Fock[0, i])
            ew[1][i], ev[1][i] = la.eigh(Fock[1, i])
    else:
        for i in range(ncells):
            ew[0][i], ev[0][i] = la.eigh(Fock[0, i] + vcor.get(i, True)[0])
            ew[1][i], ev[1][i] = la.eigh(Fock[1, i] + vcor.get(i, True)[1])
    return ew, ev

def DiagUHF_symm(Fock, vcor, lattice, **kwargs):
    if Fock.ndim == 3:
        Fock = np.asarray((Fock, Fock))
    ncells = Fock.shape[-3]
    nscsites = Fock.shape[-1]
    ew = np.empty((2, ncells, nscsites))
    ev = np.empty((2, ncells, nscsites, nscsites), dtype=np.complex128)
    
    computed = set()
    for i in range(ncells):
        neg_i = lattice.cell_pos2idx(-lattice.cell_idx2pos(i))
        if neg_i in computed:
            ew[0][i], ev[0][i] = ew[0][neg_i], ev[0][neg_i].conj()
            ew[1][i], ev[1][i] = ew[1][neg_i], ev[1][neg_i].conj()
        else:
            if vcor is None:
                ew[0][i], ev[0][i] = la.eigh(Fock[0, i])
                ew[1][i], ev[1][i] = la.eigh(Fock[1, i])
            else:
                ew[0][i], ev[0][i] = la.eigh(Fock[0, i] + vcor.get(i, True)[0])
                ew[1][i], ev[1][i] = la.eigh(Fock[1, i] + vcor.get(i, True)[1])
            computed.add(i)
    return ew, ev

def DiagHF_scf(lattice, vcor, filling, restricted, mu0=None, beta=np.inf, \
        ires=False, **kwargs):
    """
    RHF and UHF diagonalization routine for lattice problem, doing self-consistency.
    Passing a dm0 in kwargs for initial guess.
    Using newton if newton_ah == True.

    See HF for more details.
    """
    # ZHC FIXME when use smearing should we return e_zero?
    nao   = lattice.nscsites
    kpts  = lattice.kpts
    nkpts = lattice.nkpts

    if isinstance(filling, Iterable):
        assert len(filling) == 2
        nelec = []
        nelec_per_cell = []
        for s in range(len(filling)):
            nelec_s, nelec_per_cell_s = check_nelec(nao * nkpts * filling[s], nkpts)
            nelec.append(nelec_s)
            nelec_per_cell.append(nelec_per_cell_s)
        nelec_per_cell = np.sum(nelec_per_cell)
        Sz = nelec[0] - nelec[1]
    else: 
        # total number of electron
        nelec, nelec_per_cell = check_nelec(nao * nkpts * 2 * filling, nkpts)
        Sz = 0

    cell = lattice.cell
    cell.nelectron = nelec_per_cell
    cell.spin = Sz
    if restricted:
        log.info("Restricted Hartree-Fock using pyscf")
        kmf = pbc.scf.KRHF(cell, kpts, exxdiv=None)
    else:
        log.info("Unrestricted Hartree-Fock using pyscf")
        kmf = KUHF(cell, kpts, exxdiv=None)
    
    if lattice.H2_format == "local":
        eri = lattice.getH2(kspace=False)
        def get_jk(cell=None, dm_kpts=None, hermi=1, kpts=None, kpts_band=None,
                   with_j=True, with_k=True, omega=None, **kwargs):
            if cell is None: cell = kmf.cell
            if kpts is None: kpts = kmf.kpts
            if dm_kpts is None: dm_kpts = kmf.make_rdm1()
            vj, vk = get_jk_from_eri_local(eri, dm_kpts, eri_symm=4)
            return vj, vk
    elif lattice.H2_format == "nearest":
        eri = lattice.getH2(kspace=False)
        def get_jk(cell=None, dm_kpts=None, hermi=1, kpts=None, kpts_band=None,
                   with_j=True, with_k=True, omega=None, **kwargs):
            if cell is None: cell = kmf.cell
            if kpts is None: kpts = kmf.kpts
            if dm_kpts is None: dm_kpts = kmf.make_rdm1()
            vj, vk = get_jk_from_eri_nearest(eri, dm_kpts, lattice)
            return vj, vk
    else:
        raise NotImplementedError
    
    hcore = lattice.getH1(kspace=True)
    if hcore.ndim == 3:
        if restricted:
            hcore = hcore[None].copy()
        else:
            hcore = np.array((hcore, hcore))
    if vcor is not None:
        for s in range(hcore.shape[0]):
            hcore[s] += vcor.get(kspace=True)[s]

    if restricted:
        kmf.get_hcore = lambda *args: hcore[0]
    else:
        kmf.get_hcore = lambda *args: hcore
    kmf.get_ovlp = lambda *args: lattice.get_ovlp(kspace=True)
    kmf.get_jk = get_jk    
    kmf.with_df.verbose = 3
    
    # ***********************
    # smearing and finite T
    # ***********************
    fix_mu = kwargs.get("fix_mu", False)
    if beta < np.inf:
        if fix_mu:
            assert mu0 is not None
            kmf = pbc.scf.addons.smearing_(kmf, sigma=1.0/beta, \
                    method='fermi', mu0=mu0)
        else:
            # ZHC NOTE smearing_ here is able to treat tuple of mu.
            kmf = smearing_(kmf, sigma=1.0/beta, method='fermi')
     
    # ***********************
    # initial guess and run
    # ***********************
    if kwargs.get("newton_ah", False):
        kmf = kmf.newton()
    if "verbose" in kwargs:
        kmf.verbose = kwargs["verbose"]
    if "conv_tol" in kwargs:
        kmf.conv_tol = kwargs["conv_tol"]
    if "max_cycle" in kwargs:
        kmf.max_cycle = kwargs["max_cycle"]
    dm0 = kwargs.get("dm0", None)
    kmf.kernel(dm0=dm0)
    if not kmf.converged:
        log.warn("kmf is not converged!")
    lattice.kmf_lo = kmf
    
    ew = np.asarray(kmf.mo_energy)
    ev = np.asarray(kmf.mo_coeff)
    if restricted:
        ew = ew[None]
        ev = ev[None]
    return ew, ev

def HF(lattice, vcor, filling, restricted, mu0=None, beta=np.inf, ires=False, \
        scf=False, use_hcore=None, **kwargs):
    """
    RHF and UHF routine for lattice problem.

    Args:
        lattice: lattice class, should have hamiltonian.
        vcor: correlation potential.
        filling: nelec per nso, can be a float or (float, float).
        restricted: RHF if True, UHF if False
        mu0: initial guess of quasiparticle mu, used when beta != 0
        beta: inverse temperature
        ires: if True, return a dict with keys:
            E0, E (has vcor contribution), e, coef, ew_occ, rho_k, gap, nerr
        scf: if True, perform self-consistency, usually need dm0 as guess
        kwargs:
            symm: [False], consider k and -k symmetry during diagonalization
            fix_mu: [False], fix quasiparticle mu
            dm0: [None], initial guess of dm, used for scf option
    Returns:
        rho: (spin, ncells, nao, nao)
        mu: qusiparticle fermi level
        E: total energy per cell, include vcor contribution
        (ires): dict, more results
    """
    # ***********************
    # initialize Hamiltonian
    # ***********************
    log.eassert(beta >= 0, "beta cannot be negative")
    if use_hcore is None:
        use_hcore = lattice.use_hcore_as_emb_ham
    if use_hcore:
        Fock  = lattice.getH1(kspace=True)
        FockT = H1T = lattice.getH1(kspace=False)
    else:
        Fock  = lattice.getFock(kspace=True)
        FockT = lattice.getFock(kspace=False)
        H1T   = lattice.getH1(kspace=False)
    
    # ***********************
    # diagonalization
    # ***********************
    if scf: # doing scf with pyscf
        ew, ev = DiagHF_scf(lattice, vcor, filling, restricted, mu0=mu0, \
                beta=beta, ires=ires, **kwargs)
        Fock = lattice.kmf_lo.get_fock()
        FockT = lattice.k2R(Fock)
    else:
        if restricted:
            log.info("Restricted Hartree-Fock")
            if kwargs.get("symm", False):
                eig_func = DiagRHF_symm
            else:
                eig_func = DiagRHF
            ew, ev = eig_func(Fock, vcor, lattice=lattice)
            ew = ew[np.newaxis]
            ev = ev[np.newaxis]
        else:
            log.info("Unrestricted Hartree-Fock")
            if kwargs.get("symm", False):
                eig_func = DiagUHF_symm
            else:
                eig_func = DiagUHF
            ew, ev = eig_func(Fock, vcor, lattice=lattice)
    
    # ***********************
    # occupancy
    # ***********************
    fix_mu = kwargs.get("fix_mu", False)
    if isinstance(filling, Iterable): # allow different filling in each spin
        nelec = [ew.size * filling[0] * 0.5, ew.size * filling[1] * 0.5]
        nelec[0], nelec[1] = check_nelec(nelec[0], None)[0], check_nelec(nelec[1], None)[0]
        ew_sorted = [np.sort(ew[s], axis=None, kind='mergesort') for s in range(2)]
        if mu0 is None:
            if nelec[0] <= 0: # 0 electron
                mu0_a = ew_sorted[0][0]
            elif nelec[0] >= len(ew_sorted[0]): # all occupied
                mu0_a = ew_sorted[0][-1]
            else:
                mu0_a = 0.5 * (ew_sorted[0][nelec[0] - 1] + ew_sorted[0][nelec[0]])
            
            if nelec[1] <= 0: # 0 electron
                mu0_b = ew_sorted[1][0]
            elif nelec[1] >= len(ew_sorted[1]): # all occupied
                mu0_b = ew_sorted[1][-1]
            else:
                mu0_b = 0.5 * (ew_sorted[1][nelec[1] - 1] + ew_sorted[1][nelec[1]])
            mu0 = [mu0_a, mu0_b]
    else:
        nelec = ew.size * filling # rhf: per spin  uhf: total nelec
        nelec = check_nelec(nelec, None)[0]
        ew_sorted = np.sort(ew, axis=None, kind='mergesort')
        if mu0 is None:
            if nelec <= 0: # 0 electron
                mu0 = ew_sorted[0]
            elif nelec >= len(ew_sorted): # all occupied
                mu0 = ew_sorted[-1]
            else:
                mu0 = 0.5 * (ew_sorted[nelec - 1] + ew_sorted[nelec])
    ewocc, mu, nerr = assignocc(ew, nelec, beta, mu0, fix_mu=fix_mu)
     
    # ***********************
    # density matrix
    # ***********************
    rho = np.empty_like(ev)
    rhoT = np.empty_like(rho)
    spin, nkpts, _, _ = rho.shape
    for s in range(spin):  # spin
        for k in range(nkpts): # kpoints
            rho[s, k] = mdot(ev[s, k]*ewocc[s, k], ev[s, k].conj().T)
        rhoT[s] = lattice.FFTtoT(rho[s])
    if max_abs(rhoT.imag) < IMAG_DISCARD_TOL:
        rhoT = rhoT.real

    # ***********************
    # energy, homo, lumo
    # ***********************
    # make FockT and H1T has spin label
    FockT = add_spin_dim(FockT, spin)
    H1T = add_spin_dim(H1T, spin)

    if vcor.islocal():
        vcorT = vcor.get(0, kspace=False)
    else:
        vcorT = np.array([vcor.get(i, kspace=False) for i in range(lattice.ncells)])
    
    if spin == 1:
        E0 = np.sum((FockT + H1T) * rhoT) + lattice.getH0()
        if vcor.islocal():
            E = E0 + np.sum(vcorT[0] * rhoT[0, 0])
        else:
            E = E0 + np.sum(vcorT[:, 0] * rhoT[0, 0])
    else:
        E0 = 0.5 * np.sum((FockT + H1T) * rhoT) + lattice.getH0()
        if vcor.islocal():
            E = E0 + 0.5 * np.sum(vcorT[0] * rhoT[0, 0] + vcorT[1] * rhoT[1, 0])
        else:
            E = E0 + 0.5 * np.sum(vcorT[:, 0] * rhoT[0] + vcorT[:, 1] * rhoT[1])
    
    if ires:
        if isinstance(mu, Iterable):
            homo_idx_a = max(np.searchsorted(ew_sorted[0], mu[0], side='right') - 1, 0)
            lumo_idx_a = min(np.searchsorted(ew_sorted[0], mu[0], side='left'), len(ew_sorted[0]) - 1)
            homo_a = ew_sorted[0][homo_idx_a]
            lumo_a = ew_sorted[0][lumo_idx_a]
            
            homo_idx_b = max(np.searchsorted(ew_sorted[1], mu[1], side='right') - 1, 0)
            lumo_idx_b = min(np.searchsorted(ew_sorted[1], mu[1], side='left'), len(ew_sorted[1]) - 1)
            homo_b = ew_sorted[1][homo_idx_b]
            lumo_b = ew_sorted[1][lumo_idx_b]

            gap = np.array((lumo_a - homo_a, lumo_b - homo_b))
        else:
            homo_idx = max(np.searchsorted(ew_sorted, mu, side='right') - 1, 0)
            lumo_idx = min(np.searchsorted(ew_sorted, mu, side='left'), len(ew_sorted) - 1)
            homo = ew_sorted[homo_idx]
            lumo = ew_sorted[lumo_idx]
            gap = lumo - homo
        res = {"gap": gap, "e": ew, "coef": ev, "nerr": nerr, \
                "rho_k": rho, "E0": E0, "E": E, "mo_occ": ewocc}
        return rhoT, mu, E, res
    else:
        return rhoT, mu, E

def DiagBdG(Fock, vcor, mu, **kwargs):
    if Fock.ndim == 3:
        Fock = np.asarray((Fock, Fock))
    nkpts = Fock.shape[-3]
    nscsites = Fock.shape[-1]
    ew = np.empty((nkpts, nscsites*2))
    ev = np.empty((nkpts, nscsites*2, nscsites*2), dtype = np.complex128)
    temp = np.empty((nscsites*2, nscsites*2), dtype = np.complex128)

    for i in range(nkpts):
        temp[:nscsites, :nscsites] = Fock[0, i] + vcor.get(i, True)[0] - \
                mu * np.eye(nscsites)
        # ZHC FIXME NOTE should be there a transpose?
        temp[nscsites:, nscsites:] = -Fock[1, i] - vcor.get(i, True)[1] + \
                mu * np.eye(nscsites)
        #temp[nscsites:, nscsites:] = -(Fock[1, i] + vcor.get(i, True)[1]).T + \
        #        mu * np.eye(nscsites)
        temp[:nscsites, nscsites:] = vcor.get(i, True)[2]
        temp[nscsites:, :nscsites] = vcor.get(i, True)[2].conj().T
        ew[i], ev[i] = la.eigh(temp)
    return ew, ev

def DiagBdGsymm(Fock, vcor, mu, lattice, **kwargs):
    if Fock.ndim == 3:
        Fock = np.asarray((Fock, Fock))
    nkpts = Fock.shape[-3]
    nscsites = Fock.shape[-1]
    ew = np.empty((nkpts, nscsites*2))
    ev = np.empty((nkpts, nscsites*2, nscsites*2), dtype = np.complex128)
    temp = np.empty((nscsites * 2, nscsites * 2), dtype = np.complex128)

    computed = set()
    for i in range(nkpts):
        neg_i = lattice.cell_pos2idx(-lattice.cell_idx2pos(i))
        if neg_i in computed:
            ew[i], ev[i] = ew[neg_i], ev[neg_i].conj()
        else:
            temp[:nscsites, :nscsites] = Fock[0, i] + vcor.get(i, True)[0] - \
                    mu * np.eye(nscsites)
            # ZHC FIXME NOTE should be there a transpose?
            temp[nscsites:, nscsites:] = -Fock[1, i] - vcor.get(i, True)[1] \
                    + mu * np.eye(nscsites)
            #temp[nscsites:, nscsites:] = \
            #        -(Fock[1, i] + vcor.get(i, True)[1]).T + \
            #        mu * np.eye(nscsites)
            temp[:nscsites, nscsites:] = vcor.get(i, True)[2]
            temp[nscsites:, :nscsites] = vcor.get(i, True)[2].conj().T
            ew[i], ev[i] = la.eigh(temp)
            computed.add(i)
    return ew, ev

def HFB(lattice, vcor, restricted, mu=0.0, beta=np.inf, fix_mu=False, \
        ires=False, use_hcore=None, **kwargs):
    """
    HFB routine for lattice problem.
    Args:
    ------
        ires: more results, e.g. total energy, gap, ew, ev
    """
    # ***********************
    # initialize Hamiltonian
    # ***********************
    log.eassert(beta >= 0, "beta cannot be negative")
    if use_hcore is None:
        use_hcore = lattice.use_hcore_as_emb_ham
    if use_hcore:
        Fock  = lattice.getH1(kspace=True)
        FockT = H1T = lattice.getH1(kspace=False)
    else:
        Fock  = lattice.getFock(kspace=True)
        FockT = lattice.getFock(kspace=False)
        H1T   = lattice.getH1(kspace=False)

    # ***********************
    # diagonalization
    # ***********************
    if restricted:
        log.error("restricted Hartree-Fock-Bogoliubov not implemented")
    else:
        log.debug(1, "unrestricted Hartree-Fock-Bogoliubov")
        if kwargs.get("symm", False):
            # use inversion symmetry F(k) = F(-k)^*
            log.debug(2, "Symmetrized diagonalization used.")
            ew, ev = DiagBdGsymm(Fock, vcor, mu, lattice)
        else:
            log.debug(2, "Symmetrized diagonalization not used.")
            ew, ev = DiagBdG(Fock, vcor, mu)

    # ***********************
    # occupancy and rdm1
    # ***********************
    ew_sorted = np.sort(ew, axis=None, kind='mergesort')
    GRho = np.empty_like(ev) # k-space
    GRhoT = np.empty_like(GRho) # real space
    mu_ref = 0.0
    if beta == np.inf: # 0 T
        ewocc = 1 * (ew < mu_ref)
        nocc = np.sum(ewocc)
        log.check(nocc*2 == ew.size, \
                "number of negative and positive modes are not equal," + \
                "the difference is %d, " + \
                "this means total spin on lattice is nonzero", \
                nocc*2 - ew.size)
        for k in range(GRho.shape[0]): # kpoints
            nocc = np.sum(ewocc[k])
            GRho[k] = np.dot(ev[k,:,:nocc], ev[k,:,:nocc].T.conj())
    else:
        if not fix_mu:
            target_density = 0.5
            mu_ref = ftsystem.find_mu_by_density(target_density, \
                    ew_sorted, beta, mu0=mu_ref)
        ewocc = ftsystem.fermi_smearing_occ(mu_ref, ew, beta)
        nocc = np.sum(ewocc)
        log.check(abs(nocc/float(ew.size) - 0.5) < 1e-8, \
                "number of negative and positive modes are not equal," + \
                "the difference is %15.6f, " + \
                "this means total spin on lattice is nonzero", \
                nocc*2 - ew.size)
        for k in range(GRho.shape[0]): # kpoints
            GRho[k] = (ev[k]*ewocc[k]).dot(ev[k].conj().T)
    
    GRhoT = lattice.FFTtoT(GRho)
    if max_abs(GRhoT.imag) < IMAG_DISCARD_TOL:
        GRhoT = GRhoT.real
    
    # ***********************
    # energy, homo, lumo
    # ***********************
    # make FockT and H1T has spin label
    FockT = add_spin_dim(FockT, 2)
    H1T = add_spin_dim(H1T, 2)
    
    if vcor.islocal():
        vcorT = vcor.get(0, kspace=False)
    else:
        vcorT = np.asarray([vcor.get(i, kspace=False) for i in
                range(lattice.ncells)])

    from libdmet.routine.bcs_helper import extractRdm
    rhoTA, rhoTB, kappaTBA = \
            np.swapaxes(np.asarray([extractRdm(x) for x in GRhoT]), 0, 1)
    for c in range(1, rhoTB.shape[0]):
        rhoTB[c] -= np.eye(rhoTB.shape[1])

    n = np.trace(rhoTA[0]) + np.trace(rhoTB[0])
    E = 0.5 * np.sum((FockT[0]+H1T[0])*rhoTA + (FockT[1]+H1T[1])*rhoTB) \
            + lattice.getH0()
    if vcor.islocal():
        E += 0.5 * np.sum(vcorT[0] * rhoTA[0] + vcorT[1] * rhoTB[0] + \
                2 * vcorT[2] * kappaTBA[0])
    else:
        E += 0.5 * np.sum(vcorT[:,0] * rhoTA + vcorT[:,1] * rhoTB + \
                2 * vcorT[:,2] * kappaTBA)

    if ires:
        homo_idx = max(np.searchsorted(ew_sorted, mu_ref, side='right') - 1, 0)
        lumo_idx = min(np.searchsorted(ew_sorted, mu_ref, side='left'), len(ew_sorted) - 1)
        homo = ew_sorted[homo_idx]
        lumo = ew_sorted[lumo_idx]
        res = {"gap": lumo - homo, "e": ew, "coef": ev, "E": E, "rho_k": GRho}
        return GRhoT, n, E, res
    else:
        return GRhoT, n, E

def check_nelec(nelec, ncells=None, tol=1e-5):
    """
    Round off the nelec to its nearest integer.

    Args:
        nelec: number of electrons for the whole lattice.
        ncells: number of cells, if not None, will check nelec / ncells.

    Returns:
        nelec: rounded nelec, int.
        nelec_per_cell: number of elecrtons per cell, float.
    """
    nelec_round = int(np.round(nelec))
    if abs(nelec - nelec_round) > tol:
        log.warn("HF: nelec is rounded to integer nelec = %d (original %.2f)", \
                nelec_round, nelec)
    nelec = nelec_round
    if ncells is None:
        nelec_per_cell = None
    else:
        nelec_per_cell = nelec / float(ncells)
        if abs(nelec_per_cell - np.round(nelec_per_cell)) > tol:
            log.warn("HF: nelec per cell (%.5f) is not an integer.", nelec_per_cell) 
        else:
            nelec_per_cell = int(np.round(nelec_per_cell))
    return nelec, nelec_per_cell

def assignocc(ew, nelec, beta, mu0, fix_mu=False, thr_deg=1e-6, Sz=None, 
              fit_tol=1e-12, f_occ=ftsystem.fermi_smearing_occ):
    """
    Assign the occupation number of a mean-field.
    nelec is per spin for RHF, total for UHF. 
    """
    ew = np.asarray(ew)
    if (Sz is None) and (not isinstance(nelec, Iterable)):
        ew_sorted = np.sort(ew, axis=None, kind='mergesort')
        if beta < np.inf:
            if fix_mu:
                mu = mu0
            else:
                mu = ftsystem.find_mu(nelec, ew_sorted, beta, mu0=mu0, 
                                      tol=fit_tol, f_occ=f_occ)
            ewocc = f_occ(mu, ew, beta)
            nerr = abs(np.sum(ewocc) - nelec)
        else: # zero T
            nelec = check_nelec(nelec, None)[0]   
            if np.sum(ew < mu0-thr_deg) <= nelec and np.sum(ew <= mu0 + thr_deg) >= nelec:
                # we prefer not to change mu
                mu = mu0
            else:
                mu = 0.5 * (ew_sorted[nelec-1] + ew_sorted[nelec])

            ewocc = 1.0 * (ew < mu - thr_deg)
            nremain_elec = nelec - np.sum(ewocc)
            if nremain_elec > 0:
                # fractional occupation
                remain_orb = np.logical_and(ew <= mu + thr_deg, ew >= mu - thr_deg)
                nremain_orb = np.sum(remain_orb)
                log.warn("degenerate HOMO-LUMO, assign fractional occupation\n"
                         "%d electrons assigned to %d orbitals", nremain_elec, nremain_orb)
                ewocc += (float(nremain_elec) / nremain_orb) * remain_orb
            nerr = 0.0
    else: # allow specify Sz
        spin = ew.shape[0]
        assert spin == 2
        if not isinstance(nelec, Iterable):
            nelec = [(nelec + Sz) * 0.5, (nelec - Sz) * 0.5]
        if not isinstance(mu0, Iterable):
            mu0 = [mu0 for s in range(spin)]
        ewocc = np.empty_like(ew)
        mu    = np.zeros((spin,))
        nerr  = np.zeros((spin,))
        ewocc[0], mu[0], nerr[0] = assignocc(ew[0], nelec[0], beta, mu0[0], 
                                             fix_mu=fix_mu, thr_deg=thr_deg,
                                             fit_tol=fit_tol, f_occ=f_occ)
        ewocc[1], mu[1], nerr[1] = assignocc(ew[1], nelec[1], beta, mu0[1], 
                                             fix_mu=fix_mu, thr_deg=thr_deg,
                                             fit_tol=fit_tol, f_occ=f_occ)
    return ewocc, mu, nerr
