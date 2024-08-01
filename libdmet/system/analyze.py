#! /usr/bin/env python

"""
Analyze functions for kLO, kMO and rdm1, chemical bond.

Author:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

from pyscf import gto
from pyscf.lib import logger as pyscf_logger
from pyscf.scf.hf import mulliken_pop as mulliken_pop_rhf
from pyscf.data.nist import BOHR

from libdmet.utils.misc import add_spin_dim
from libdmet.system.fourier import *

def check_lo(lattice, C_ao_lo, kpts=None, ovlp=None, tol=1e-10):
    """
    Check whether a set of k-dependent local orbitals,
    whether has imaginary part or violates time reversal symmetry.
    If ovlp is not None, check the orthonormality as well.
    """
    log.info("-----------------------------------------------------------")
    log.info("Check the reality and time reversal symm of orbitals.")
    if kpts is None:
        kpts = lattice.kpts
    nkpts = len(kpts)
    kpts_scaled = lattice.cell.get_scaled_kpts(kpts)
    kpts_round = round_to_FBZ(kpts_scaled, tol=tol)
    C_ao_lo = np.asarray(C_ao_lo)
    
    # check time reversal symmetry
    diff_k_mk = 0.0
    weights = np.ones(nkpts, dtype=int)
    for i, ki in enumerate(kpts_round):
        if weights[i] == 1:
            for j in range(i+1, nkpts):
                sum_ij = ki + kpts_round[j]
                sum_ij -= np.round(sum_ij)
                if max_abs(sum_ij) < tol:
                    if C_ao_lo.ndim == 3:
                        diff_k_mk = max(diff_k_mk, 
                                        max_abs(C_ao_lo[j] - C_ao_lo[i].conj()))
                    else:
                        for s in range(C_ao_lo.shape[0]):
                            diff_k_mk = max(diff_k_mk,
                                            max_abs(C_ao_lo[s, j] - C_ao_lo[s, i].conj()))
                    weights[i] = 2
                    weights[j] = 0
                    break
    log.info("Maximal difference between k and -k orbitals: %.2e", diff_k_mk)
    
    # check imaginary
    C_ao_lo_R = lattice.k2R(C_ao_lo)
    imag_norm = max_abs(C_ao_lo_R.imag)
    log.info("Imaginary part of orbitals: %.2e", imag_norm)
    
    if ovlp is not None:
        from libdmet.lo.lowdin import check_orthonormal
        log.info("Orthonormal: %s", check_orthonormal(C_ao_lo, ovlp, tol=tol))
    log.info("-----------------------------------------------------------")
    return imag_norm, diff_k_mk

def symmetrize_lo(lattice, C_ao_lo, kpts=None, tol=1e-10, real_first=False):
    """
    Check whether a set of k-dependent local orbitals,
    whether has imaginary part or violates time reversal symmetry.
    """
    log.info("-----------------------------------------------------------")
    log.info("Impose the reality and time reversal symm of orbitals.")
    if kpts is None:
        kpts = lattice.kpts
    nkpts = len(kpts)
    kpts_scaled = lattice.cell.get_scaled_kpts(kpts)
    kpts_round = round_to_FBZ(kpts_scaled, tol=tol)
    C_ao_lo_symm = np.array(C_ao_lo, copy=True)
    
    if real_first:
        # enforce reality 
        C_ao_lo_symm_R = lattice.k2R(C_ao_lo_symm)
        imag_norm = max_abs(C_ao_lo_symm_R.imag)
        C_ao_lo_symm = lattice.R2k(C_ao_lo_symm_R.real)
        log.info("Imaginary part of orbitals: %s", imag_norm)
    
    # enforce time reversal symmetry
    diff_k_mk = 0.0
    weights = np.ones(nkpts, dtype=int)
    for i, ki in enumerate(kpts_round):
        if weights[i] == 1:
            for j in range(i+1, nkpts):
                sum_ij = ki + kpts_round[j]
                sum_ij -= np.round(sum_ij)
                if max_abs(sum_ij) < tol:
                    if C_ao_lo_symm.ndim == 3:
                        diff_k_mk = max(diff_k_mk,
                                        max_abs(C_ao_lo_symm[j] - C_ao_lo_symm[i].conj()))
                        C_ao_lo_symm[j] = C_ao_lo_symm[i].conj()
                    else:
                        for s in range(C_ao_lo_symm.shape[0]):
                            diff_k_mk = max(diff_k_mk, 
                                            max_abs(C_ao_lo_symm[s, j] - C_ao_lo_symm[s, i].conj()))
                            C_ao_lo_symm[s, j] = C_ao_lo_symm[s, i].conj()
                    weights[i] = 2
                    weights[j] = 0
                    break
    log.info("Maximal difference between k and -k orbitals: %.2e", diff_k_mk)
    
    if not real_first:
        # enforce reality 
        C_ao_lo_symm_R = lattice.k2R(C_ao_lo_symm)
        imag_norm = max_abs(C_ao_lo_symm_R.imag)
        C_ao_lo_symm = lattice.R2k(C_ao_lo_symm_R.real)
        log.info("Imaginary part of orbitals: %.2e", imag_norm)
    log.info("-----------------------------------------------------------")
    return C_ao_lo_symm

def symmetrize_kmf(lattice, kmf, tol=1e-10):
    """
    Symmetrize kmf with time reversal symmetry.
    """
    from pyscf.scf import uhf
    from pyscf.pbc.scf import uhf as pbcuhf
    from pyscf.pbc.scf import kuhf
    # ZHC FIXME support istype in the future.
    is_uhf = isinstance(kmf, uhf.UHF) or isinstance(kmf, pbcuhf.UHF) or isinstance(kmf, kuhf.KUHF)
    kpts = lattice.kpts
    nkpts = len(kpts)
    kpts_scaled = lattice.cell.get_scaled_kpts(kpts)
    kpts_round = round_to_FBZ(kpts_scaled, tol=tol)
    weights = np.ones(nkpts, dtype=int)
    for i, ki in enumerate(kpts_round):
        if weights[i] == 1:
            for j in range(i+1, nkpts):
                sum_ij = ki + kpts_round[j]
                sum_ij -= np.round(sum_ij)
                if max_abs(sum_ij) < tol:
                    if not is_uhf:
                        kmf.mo_coeff[i]  = kmf.mo_coeff[j].conj()
                        kmf.mo_energy[i] = kmf.mo_energy[j]
                        kmf.mo_occ[i] = kmf.mo_occ[j]
                    else:
                        for s in range(2):
                            kmf.mo_coeff[s][i]  = kmf.mo_coeff[s][j].conj()
                            kmf.mo_energy[s][i] = kmf.mo_energy[s][j]
                            kmf.mo_occ[s][i] = kmf.mo_occ[s][j]
                    weights[i] = 2
                    weights[j] = 0
                    break
    return kmf

def analyze(lattice, kmf, C_ao_lo=None, labels=None, 
            verbose=pyscf_logger.DEBUG, rdm1_lo_R0=None, method='meta-lowdin',
            pre_orth_ao='ANO'):
    """
    Analyze the given SCF object:  print orbital energies, occupancies;
    print orbital coefficients; Mulliken population analysis; Dipole moment

    Args:
        lattice: lattice object.
        kmf: kscf object.
        C_ao_lo: shape ((spin,), nkpts, nao, nlo), 
                 if None, meta Lowdin is used.
        labels: LO labels, list of strings.
        verbose: verbose level.
        rdm1_lo_R0: analyze a given rdm1_lo at R0.
        method: default LO method.
    """
    from libdmet.lo import lowdin
    kmf.dump_scf_summary(verbose)
    ovlp = np.asarray(kmf.get_ovlp())
    
    if rdm1_lo_R0 is None:
        mo_occ   = kmf.mo_occ
        mo_coeff = kmf.mo_coeff
        rdm1 = np.asarray(kmf.make_rdm1(mo_coeff, mo_occ))
    else:
        rdm1 = None

    if C_ao_lo is None:
        C_ao_lo = lowdin.lowdin_k(kmf, method=method, s=ovlp, pre_orth_ao=pre_orth_ao)
    return mulliken_lo(lattice, rdm1, ovlp, C_ao_lo=C_ao_lo, labels=labels, 
                       verbose=verbose, rdm1_lo_R0=rdm1_lo_R0)

def mulliken_lo(lattice, rdm1, ovlp, C_ao_lo, labels, 
                verbose=pyscf_logger.DEBUG, rdm1_lo_R0=None, rdm1_lo_R0_2=None):
    """
    A modified Mulliken population analysis, based on given LO.
    """
    from libdmet.basis_transform import make_basis
    if isinstance(lattice, gto.Mole):
        cell = lattice
    else:
        cell = lattice.cell.copy()
    log = pyscf_logger.new_logger(cell, verbose)
    C_ao_lo = np.asarray(C_ao_lo)
    
    if rdm1_lo_R0 is None:
        rdm1_lo = lattice.k2R(make_basis.transform_rdm1_to_lo(rdm1, C_ao_lo, ovlp))
        if rdm1_lo.ndim == 3: # RHF
            rdm1_lo_R0 = rdm1_lo[0]
        else:
            rdm1_lo_R0 = rdm1_lo[:, 0]
    rdm1_lo_R0 = np.asarray(rdm1_lo_R0)
    nlo = rdm1_lo_R0.shape[-1]
    
    if labels is None:
        idx_to_ao_labels = np.arange(nlo)
    else:
        # IAO indices need to resort according to each atom.
        atom_ids = [int(lab.split()[0]) for lab in labels]
        idx_to_ao_labels = np.argsort(atom_ids, kind='mergesort')
        
        labels_ao = [labels[idx] for idx in idx_to_ao_labels]
        # a hack, only keep atom id
        labels_ao_no_fmt = [(int(lab.split()[0]),) for lab in labels_ao] 

        def ao_labels(fmt=True):
            if fmt:
                return labels_ao
            else:
                return labels_ao_no_fmt
        cell.ao_labels = ao_labels

    # resort rdm1_lo_R0 according to LO labels, 
    # so that the order is the same as AOs.
    mesh = np.ix_(idx_to_ao_labels, idx_to_ao_labels) 
    if rdm1_lo_R0.ndim == 2: # RHF
        rdm1_lo_R0 = rdm1_lo_R0[mesh]
    else:
        if rdm1_lo_R0.shape[0] == 1: # RHF
            rdm1_lo_R0 = rdm1_lo_R0[0][mesh]
        else: # UHF
            rdm1_lo_R0 = np.asarray([rdm1_lo_R0[s][mesh] 
                                     for s in range(rdm1_lo_R0.shape[0])])

    log.note(' ** Mulliken pop on LOs **')
    if rdm1_lo_R0_2 is None:
        if rdm1_lo_R0.ndim == 2:
            return mulliken_pop_rhf(cell, rdm1_lo_R0, np.eye(nlo), log)
        else:
            return mulliken_pop_uhf(cell, rdm1_lo_R0, np.eye(nlo), log)
    else:
        if rdm1_lo_R0_2.ndim == 2: # RHF
            rdm1_lo_R0_2 = rdm1_lo_R0_2[mesh]
        else:
            if rdm1_lo_R0_2.shape[0] == 1: # RHF
                rdm1_lo_R0_2 = rdm1_lo_R0_2[0][mesh]
            else: # UHF
                rdm1_lo_R0_2 = np.asarray([rdm1_lo_R0_2[s][mesh] 
                                           for s in range(rdm1_lo_R0_2.shape[0])])
        compare_density(cell, rdm1_lo_R0, rdm1_lo_R0_2, np.eye(nlo))
        return None, None

def mulliken_lo_R0(lattice, rdm1_lo_R0, rdm1_lo_R0_2=None, labels=None):
    return mulliken_lo(lattice, None, None, None, labels, 
                       rdm1_lo_R0=rdm1_lo_R0, rdm1_lo_R0_2=rdm1_lo_R0_2)

def mulliken_pop_uhf(mol, dm, s=None, verbose=pyscf_logger.DEBUG):
    """
    Mulliken population analysis, UHF case.
    Include local magnetic moment.
    """
    if s is None: s = hf.get_ovlp(mol)
    log = pyscf_logger.new_logger(mol, verbose)
    if isinstance(dm, np.ndarray) and dm.ndim == 2:
        dm = np.array((dm*.5, dm*.5))
    pop_a = np.einsum('ij,ji->i', dm[0], s).real
    pop_b = np.einsum('ij,ji->i', dm[1], s).real

    log.info(' ** Mulliken pop            alpha | beta      %12s **'%("magnetism"))
    for i, s in enumerate(mol.ao_labels()):
        log.info('pop of  %-14s %10.5f | %-10.5f  %10.5f',
                 s.strip(), pop_a[i], pop_b[i], pop_a[i] - pop_b[i])
    log.info('In total               %10.5f | %-10.5f  %10.5f', 
             sum(pop_a), sum(pop_b), sum(pop_a) - sum(pop_b))

    log.note(' ** Mulliken atomic charges    ( Nelec_alpha | Nelec_beta )'
            ' %12s **'%("magnetism"))
    nelec_a = np.zeros(mol.natm)
    nelec_b = np.zeros(mol.natm)
    for i, s in enumerate(mol.ao_labels(fmt=None)):
        nelec_a[s[0]] += pop_a[i]
        nelec_b[s[0]] += pop_b[i]
    chg = mol.atom_charges() - (nelec_a + nelec_b)
    for ia in range(mol.natm):
        symb = mol.atom_symbol(ia)
        log.note('charge of %4d%s =   %10.5f  (  %10.5f   %10.5f )   %10.5f',
                 ia, symb, chg[ia], nelec_a[ia], nelec_b[ia], nelec_a[ia] - nelec_b[ia])
    return (pop_a,pop_b), chg

def compare_density(mol, rdm_1, rdm_2, s=None, verbose=pyscf_logger.DEBUG):
    r"""
    Compare two density matrices.
    """
    log = pyscf_logger.new_logger(mol, verbose)
    if s is None:
        s = get_ovlp(mol)
    rdm_1 = np.asarray(rdm_1)
    rdm_2 = np.asarray(rdm_2)
    assert rdm_1.shape == rdm_2.shape
    if rdm_1.ndim == 2: # RHF
        pop1 = np.einsum('ij,ji->i', rdm_1, s).real
        pop2 = np.einsum('ij,ji->i', rdm_2, s).real
        log.info(' ** Mulliken pop   %10s  %10s **', "sys1", "sys2")
        for i, s in enumerate(mol.ao_labels()):
            log.info('pop of  %s %10.5f  %10.5f', s, pop1[i], pop2[i])

        log.note(' ** Mulliken atomic charges  **')
        chg1 = np.zeros(mol.natm)
        chg2 = np.zeros(mol.natm)
        for i, s in enumerate(mol.ao_labels(fmt=None)):
            chg1[s[0]] += pop1[i]
            chg2[s[0]] += pop2[i]
        chg1 = mol.atom_charges() - chg1
        chg2 = mol.atom_charges() - chg2
        for ia in range(mol.natm):
            symb = mol.atom_symbol(ia)
            log.note('charge of  %d%s =   %10.5f  %10.5f', ia, symb, 
                     chg1[ia], chg2[ia])
    else: # ROHF, UHF
        pop1_a = np.einsum('ij,ji->i', rdm_1[0], s).real
        pop1_b = np.einsum('ij,ji->i', rdm_1[1], s).real
        pop2_a = np.einsum('ij,ji->i', rdm_2[0], s).real
        pop2_b = np.einsum('ij,ji->i', rdm_2[1], s).real
        log.info(" ** Mulliken pop        alpha | beta           "
                "  alpha | beta")
        for i, s in enumerate(mol.ao_labels()):
            log.info('pop of  %s %10.5f | %-10.5f  %10.5f | %-10.5f',
                     s, pop1_a[i], pop1_b[i], pop2_a[i], pop2_b[i])
        log.info('In total           %10.5f | %-10.5f  %10.5f | %-10.5f', 
                 sum(pop1_a), sum(pop1_b), sum(pop2_a), sum(pop2_b))

        log.note(' ** Mulliken atomic charges   ( Nelec_alpha | Nelec_beta )'
                '     charges   ( Nelec_alpha | Nelec_beta ) **')
        nelec1_a = np.zeros(mol.natm)
        nelec1_b = np.zeros(mol.natm)
        nelec2_a = np.zeros(mol.natm)
        nelec2_b = np.zeros(mol.natm)
        for i, s in enumerate(mol.ao_labels(fmt=None)):
            nelec1_a[s[0]] += pop1_a[i]
            nelec1_b[s[0]] += pop1_b[i]
            nelec2_a[s[0]] += pop2_a[i]
            nelec2_b[s[0]] += pop2_b[i]
        chg1 = mol.atom_charges() - (nelec1_a + nelec1_b)
        chg2 = mol.atom_charges() - (nelec2_a + nelec2_b)
        for ia in range(mol.natm):
            symb = mol.atom_symbol(ia)
            log.note('charge of  %d%s =   %10.5f  (  %10.5f   %10.5f )'
                     '   %10.5f  (  %10.5f   %10.5f )',
                     ia, symb, chg1[ia], nelec1_a[ia], nelec1_b[ia], 
                               chg2[ia], nelec2_a[ia], nelec2_b[ia])

def analyze_kmo(kmf, C_ao_lo=None, lo_labels=None, C_lo_mo=None, num_max=4, 
                k_loop_first=True, mo_print_list=None, kpts_print_list=None, 
                nmo_print=None, nkpts_print=None, pre_orth_ao='ANO'):
    """
    Analyze k-MO at each k point for each MO.

    Args:
        kmf: KSCF object.
        C_ao_lo: C_ao_lo
        ovlp: AO overlap
        lo_labels: can be list or dict
        C_lo_mo: directly give C_lo_mo
        num_max: largest num_max component will be printed.
        k_loop_first: print the results at each kpts.
        mo_print_list: a list of MO indices to print.
        kpts_print_list: a list of kpt indices to print.
        nmo_print: number of MO to print
        nkpts_print: number of kpts to print.
        mo_energy: if provide, will print the energy range of a band.

    Returns:
        order: the order of lo component.
        C_lo_mo_abs: (spin, nkpts, nlo_grouped, nmo)
    """
    from libdmet.lo import lowdin
    from libdmet.basis_transform import make_basis 
    log.info("Analyze k-MO")
    log.info("-" * 79)
    
    mo_coeff = np.asarray(kmf.mo_coeff)
    if mo_coeff.ndim == 3:
        mo_coeff = mo_coeff[None]
    spin, nkpts, nao, nmo = mo_coeff.shape
    mo_energy = np.asarray(kmf.mo_energy).reshape(spin, nkpts, nmo)
    mo_energy_min = np.amin(mo_energy, axis=1)
    mo_energy_max = np.amax(mo_energy, axis=1)
    ovlp = np.asarray(kmf.get_ovlp())

    if C_ao_lo is None:
        C_ao_lo = lowdin.lowdin_k(kmf, method='meta_lowdin', s=ovlp, 
                                  pre_orth_ao=pre_orth_ao)
    C_ao_lo = np.asarray(C_ao_lo)
    C_ao_lo = add_spin_dim(C_ao_lo, spin)
    nlo = C_ao_lo.shape[-1]
    if C_lo_mo is None:
        C_lo_mo = make_basis.get_mo_ovlp_k(C_ao_lo, mo_coeff, ovlp)
    C_lo_mo_abs = np.abs(C_lo_mo) ** 2 * 100.0
    if lo_labels is None:
        lo_labels = kmf.cell.ao_labels()
    
    if nmo_print is None:
        nmo_print = nmo
    else:
        nmo_print = min(nmo_print, nmo)
    if nkpts_print is None:
        nkpts_print = nkpts
    else:
        nkpts_print = min(nkpts_print, nkpts)

    if mo_print_list is None:
        mo_print_list = range(nmo_print)
    if kpts_print_list is None:
        kpts_print_list = range(nkpts_print)

    if isinstance(lo_labels, dict):
        lo_keys = list(lo_labels.keys())
        nlo_grouped = len(lo_keys)
        C_lo_mo_abs_grouped = np.empty((spin, nkpts, nlo_grouped, nmo))
        for l in range(nlo_grouped):
            C_lo_mo_abs_grouped[:, :, l] = \
                    C_lo_mo_abs[:, :, lo_labels[lo_keys[l]]].sum(axis=2)
        C_lo_mo_abs = C_lo_mo_abs_grouped
    else:
        assert len(lo_labels) == nlo
        lo_keys = np.asarray(lo_labels)

    order = np.argsort(C_lo_mo_abs, axis=2, kind='mergesort')[:, :, ::-1]
    for s in range(spin):
        log.info("spin sector: %s", s)
        if k_loop_first:
            for k in kpts_print_list:
                log.info(" kpt: %4s", k)
                for m in mo_print_list:
                    idx = order[s, k, :num_max, m]
                    string = "".join(["%15s (%5.1f) "%(lo_keys[id].strip(), 
                                      C_lo_mo_abs[s, k, id, m]) for id in idx])
                    log.info("   MO  %4s : %s", m, string)
        else:
            for m in mo_print_list:
                log.info(" MO:  %4s    E: [%12.6f, %12.6f]", m,
                         mo_energy_min[s, m], mo_energy_max[s, m])
                for k in kpts_print_list:
                    idx = order[s, k, :num_max, m]
                    string = "".join(["%15s (%5.1f) "%(lo_keys[id].strip(), 
                                      C_lo_mo_abs[s, k, id, m]) for id in idx])
                    log.info("   kpt %4s : %s", k, string)
        log.info("-" * 79)
    return (order, C_lo_mo_abs)

def analyze_cas(mf, basis, lo_labels, num_max=4, mo_print_list=None, 
                nmo_print=None, mo_coeff=None, mo_energy=None, mo_occ=None, 
                sum_R=True):
    """
    Analyze CAS-MO.

    Args:
        mo_coeff: C_eo_mo
        basis: C_lo_eo, (spin, ncells, nlo, neo)
        lo_labels: can be list or dict
        num_max: largest num_max component will be printed.
        mo_print_list: a list of MO indices to print.
        nmo_print: number of MO to print
        mo_energy: if provide, will print the energy.

    Returns:
        order: the order of lo component.
        C_lo_mo_abs: percentage of orbital characters,
                     (spin, nlo_group, nmo).
    """
    log.info("Analyze CAS-MO")
    log.info("-" * 79)
    
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_energy is None:
        mo_energy = mf.mo_energy
    if mo_occ is None:
        mo_occ = mf.mo_occ

    mo_coeff = np.asarray(mo_coeff)
    if mo_coeff.ndim == 2:
        mo_coeff = mo_coeff[None]
    mo_energy = np.asarray(mo_energy)
    if mo_energy.ndim == 1:
        mo_energy = mo_energy[None]
    mo_occ = np.asarray(mo_occ)
    if mo_occ.ndim == 1:
        mo_occ = mo_occ[None]
    
    spin, neo, nmo = mo_coeff.shape
    basis = np.asarray(basis)
    
    C_lo_mo = np.einsum('sRpi, sij -> sRpj', basis, mo_coeff)
    if sum_R:
        C_lo_mo = np.sum(np.abs(C_lo_mo), axis=-3) # sum over R
    else:
        C_lo_mo = np.abs(C_lo_mo).reshape(spin, -1, nmo)

    for s in range(C_lo_mo.shape[0]):
        C_lo_mo[s] /= C_lo_mo[s].sum(axis=-2) # normalize over p
    C_lo_mo *= 100.0
    C_lo_mo_abs = C_lo_mo # spj
    nlo = C_lo_mo_abs.shape[-2]

    if nmo_print is None:
        nmo_print = nmo
    else:
        nmo_print = min(nmo_print, nmo)
    if mo_print_list is None:
        mo_print_list = range(nmo_print)

    if isinstance(lo_labels, dict):
        lo_keys = list(lo_labels.keys())
        nlo_grouped = len(lo_keys)
        C_lo_mo_abs_grouped = np.empty((spin, nlo_grouped, nmo))
        for l in range(nlo_grouped):
            C_lo_mo_abs_grouped[:, l] = \
                    C_lo_mo_abs[:, lo_labels[lo_keys[l]]].sum(axis=-2)
        C_lo_mo_abs = C_lo_mo_abs_grouped
    else:
        assert len(lo_labels) == nlo
        lo_keys = np.asarray(lo_labels)

    order = np.argsort(C_lo_mo_abs, axis=-2, kind='mergesort')[:, ::-1]
    for s in range(spin):
        log.info("spin sector: %s", s)
        for m in mo_print_list:
            idx = order[s, :num_max, m]
            string = "".join(["%15s (%5.1f) "%(lo_keys[i].strip(),
                              C_lo_mo_abs[s, i, m]) for i in idx])
            log.info("  MO %4s [%10.5f, %4.2f] : %s", m, 
                     mo_energy[s, m], mo_occ[s, m], string)
    log.info("-" * 79)
    return (order, C_lo_mo_abs)

def get_symm_orb(mol, idx, perm_idx=None, tol=1e-6, ignore_empty_irep=True):
    """
    Get symmetrized orbitals with selected indices.

    Args:
        mol: cluster with symmetry.
        idx: selected indices.
        perm: permutation indices for AO axis, useful for find IAO symmetry.
        tol: tolerance for removing symmetrized orbitals.
        ignore_empty_irep: ignore the empty irreps.

    Returns:
        symm_orb: a list of symmetrized orbitals.
    """
    log.info("top symmetry: %s",  mol.topgroup)
    log.info("real symmetry: %s", mol.groupname)
    log.info("selected indices: %s", format_idx(idx))
    log.debug(0, "labels:\n%s", np.array(mol.ao_labels())[idx])
    
    if perm_idx is None:
        perm_idx = np.arange(len(idx))
    log.info("perm indices: %s", format_idx(perm_idx))
    log.debug(0, "labels after permutation:\n%s", 
              np.array(mol.ao_labels())[idx][perm_idx])
    
    norb_tot = 0
    nirep = 0
    irep_sizes = []
    symm_orb = []
    for i in range(len(mol.symm_orb)):
        tmp = mol.symm_orb[i][idx]
        norm = la.norm(tmp, axis=0)
        idx_non_zero = norm > tol
        tmp = tmp[:, idx_non_zero]
        
        if tmp.size > 0:
            orbs = tmp[:, [0]]
            for j in range(1, tmp.shape[-1]):
                res = orbs.T.dot(tmp[:, [j]])
                val = la.svd(res)[1][0]
                if val < tol:
                    orbs = np.hstack((orbs, tmp[:, [j]] / la.norm(tmp[:, [j]])))
            symm_orb.append(orbs[perm_idx])
            norb_tot += orbs.shape[-1]
            nirep += 1
            irep_sizes.append(orbs.shape[-1])
        else:
            orbs = tmp
            if not ignore_empty_irep:
                symm_orb.append(orbs)
                nirep += 1
                irep_sizes.append(orbs.shape[-1])
    
    log.info("nirep: %s", nirep)
    log.info("irep sizes: \n%s", np.array(irep_sizes))
    log.eassert(norb_tot == len(idx), "norb_tot (%d) != len(idx) (%d) ",
                norb_tot, len(idx))
    return symm_orb

def get_bond_pairs(mol, length_range=[0.0, 2.0], unit='A', allow_pbc=True,
                   nimgs=[1, 1, 1], bond_type=None, triu=True):
    """
    Get all bond pairs.
    """
    if (getattr(mol, 'pbc_intor', None) is not None) and allow_pbc:
        coords_0 = mol.atom_coords()
        if unit == 'A':
            coords_0 = coords_0 * BOHR
            a = mol.lattice_vectors() * BOHR
        Ts = lib.cartesian_prod((np.arange(-nimgs[0], nimgs[0]+1),
                                 np.arange(-nimgs[1], nimgs[1]+1),
                                 np.arange(-nimgs[2], nimgs[2]+1)))
        Ls = np.dot(Ts, a)
        diff = np.inf
        for L in Ls:
            coords = L + coords_0
            diff = np.minimum(la.norm(coords - coords_0[:, None], axis=2), diff)
    else:
        coords = mol.atom_coords()
        if unit == 'A':
            coords = coords * BOHR
        diff = la.norm(coords - coords[:, None], axis=2)
    
    if triu:
        idx = np.triu_indices(diff.shape[-1], 1)
    else:
        idx = tuple(lib.cartesian_prod((np.arange(diff.shape[-1]), 
                                        np.arange(diff.shape[-1]))).T)
    dis = diff[idx]
    keep = (dis >= length_range[0]) & (dis <= length_range[1])
    pairs = np.asarray(idx).T[keep]
    dis = dis[keep]

    if bond_type is not None:
        pairs_new = []
        dis_new   = []
        for i, (pair, d) in enumerate(zip(pairs, dis)):
            atom_0 = mol._atom[pair[0]][0]
            atom_1 = mol._atom[pair[1]][0]
            if ((atom_0, atom_1) in bond_type) or \
               ((atom_1, atom_0) in bond_type):
                pairs_new.append(pair)
                dis_new.append(d)
        pairs = pairs_new
        dis   = dis_new

    return pairs, dis

def get_bond_order_all(mol, rdm1, ovlp, length_range=[0.0, 2.0], order_tol=0.01, 
                       bond_type=None, labels=None, unit='A'):
    """
    Mayer bond order. 
    Compute all bonds within length_range, and stronger than order_tol.

    Args:
        mol: Mole or Cell object.
        rdm1: density matrix, float64, shape (nao, nao)
        ovlp: overlap matrix, float64, shape (nao, nao)
        length_range: limits of bond length
        order_tol: the tolerance of bond order, only bigger ones are shown
        bond_type: constrains on type of bonds, list or set of tuples, e.g., [(C, C), (C, H)]
    """
    rdm1 = np.asarray(rdm1)
    ovlp = np.asarray(ovlp)
    assert ovlp.ndim == 2
    if rdm1.ndim == 2:
        PS = np.dot(rdm1, ovlp)
    elif rdm1.ndim == 3:
        assert rdm1.shape[0] == 2
        PS = [np.dot(rdm1[0], ovlp), np.dot(rdm1[1], ovlp)]
    else:
        raise ValueError("unknown rdm1 shape %s"%(str(rdm1.shape)))

    pairs, dis = get_bond_pairs(mol, length_range=length_range, unit=unit)
    if labels is None:
        aoind = mol.aoslice_by_atom()
        offset = [slice(*x[2:]) for x in aoind]
    else:
        from libdmet.lo import ibo
        offset = ibo.get_offset(labels)
    
    log.info("\n" + "-" * 79)
    log.info("Bond analysis")
    log.info("-" * 79)
    log.info("%5s         bond        %10s  %10s"%("index", "length", "order"))
    idx = 0
    pairs_new = []
    dis_new = []
    orders = []
    for i, (pair, d) in enumerate(zip(pairs, dis)):
        atom_0 = mol._atom[pair[0]][0]
        atom_1 = mol._atom[pair[1]][0]
        if (bond_type is not None) and (not (atom_0, atom_1) in bond_type) \
                and (not (atom_1, atom_0) in bond_type):
            continue
        idx_0 = offset[pair[0]]
        idx_1 = offset[pair[1]]

        order = get_bond_order(rdm1, ovlp, idx_0, idx_1, PS=PS)
        if order > order_tol:
            log.info("%5d %4s %2s --%4s %2s   %10.3f  %10.3f"
                     %(idx, pair[0], atom_0, pair[1], atom_1, d, order))
            pairs_new.append(pair)
            dis_new.append(d)
            orders.append(order)
            idx += 1
    log.info("-" * 79 + "\n")
    pairs = np.asarray(pairs_new)
    dis = np.asarray(dis_new)
    orders = np.asarray(orders)
    return pairs, dis, orders

def get_bond_order(rdm1, ovlp, idx_0, idx_1, PS=None):
    """
    Mayer bond order.
    """
    rdm1 = np.asarray(rdm1)
    ovlp = np.asarray(ovlp)
    assert ovlp.ndim == 2
    
    if isinstance(idx_0, slice) and isinstance(idx_1, slice):
        idx_ba = (idx_1, idx_0)
        idx_ab = (idx_0, idx_1)
    else:
        idx_ba = np.ix_(idx_1, idx_0)
        idx_ab = np.ix_(idx_0, idx_1)

    if rdm1.ndim == 2:
        if PS is None:
            PS = np.dot(rdm1, ovlp)
        bond_order = np.einsum('ba, ab ->', PS[idx_ba], PS[idx_ab], optimize=True)
    elif rdm1.ndim == 3:
        assert rdm1.shape[0] == 2
        if PS is None:
            PS_a = np.dot(rdm1[0], ovlp)
            PS_b = np.dot(rdm1[1], ovlp)
        else:
            PS_a, PS_b = PS
        bond_order_a = 2 * np.einsum('ba, ab ->', PS_a[idx_ba], PS_a[idx_ab], optimize=True)
        bond_order_b = 2 * np.einsum('ba, ab ->', PS_b[idx_ba], PS_b[idx_ab], optimize=True)
        bond_order = bond_order_a + bond_order_b
    else:
        raise ValueError
    return bond_order
