#! /usr/bin/env python

"""
BCS-DMRGCI solver.

Author:
    Bo-Xiao Zheng
    Zhi-Hao Cui
"""

from copy import deepcopy
import numpy as np
import scipy.linalg as la
from scipy import optimize as opt

from libdmet.solver import block, scf
from libdmet.system import integral
from libdmet.utils import logger as log
from libdmet.utils.misc import mdot
from libdmet.routine.localizer import Localizer
from libdmet.routine.bcs_helper import extractRdm, basisToCanonical, basisToSpin
from libdmet.integral.integral_emb_casci import transform
from libdmet.integral.integral_emb_casci_save_mem import transform as transform_save_mem
from libdmet.integral.integral_localize import transform as transform_local
from libdmet.integral.integral_localize_qp import transform as transform_local_qp
from libdmet.solver.dmrgci import gaopt, momopt
from libdmet import settings

def get_BCS_mo(scfsolver, Ham, guess, max_memory=40000, Mu=None):
    """
    Get mo_coeff and mo_energy from HFB.
    """
    #                         nelec, spin, bogoliubov, restricted, max_memory
    scfsolver.set_system(None,    0,       True,      False, max_memory=max_memory)
    scfsolver.set_integral(Ham)
    if Mu is None:
        Mu = 0.0
    E_HFB, GRho_HFB = scfsolver.HFB(Mu=Mu, tol=1e-10, MaxIter=150, InitGuess=guess)
    return scfsolver.get_mo(), scfsolver.get_mo_energy()

def get_qps(ncas, algo="nelec", **kwargs):
    """
    Get a function that returns core, cas and casinfo.
    """
    if algo == "nelec":
        log.eassert("nelecas" in kwargs, "number of electrons has to be specified")
        return lambda mo, mo_e, *args: get_qps_nelec(ncas, kwargs["nelecas"], mo, mo_e)
    elif algo == "energy":
        return lambda mo, mo_e, *args: get_qps_energy(ncas, mo, mo_e)
    elif algo == "local":
        log.eassert("nocc" in kwargs, "number of core orbitals must be specified")
        log.eassert("PAOidx" in kwargs, "candidate PAO indices must be specified")
        nPAO = ncas - kwargs["nocc"]
        log.eassert(len(kwargs["PAOidx"]) >= nPAO, "number of PAO cannot be larger than all PAO candidates")
        return lambda mo, mo_e, nImp, Ham: get_qps_local(ncas, nPAO, kwargs["PAOidx"], mo, mo_e, nImp, Ham)
    else:
        raise ValueError

def proj_virtual(mo, idx, n):
    # nao: selected AO's to be projected
    # nmo: number of mo's used in the projection
    nao = len(idx)
    nmo = mo.shape[-1]
    assert(nao <= nmo)
    # the overlap matrix between virtuals and atomic
    # orbitals to be projected
    p = np.empty((nmo, len(idx)))
    for i, orbidx in enumerate(idx):
        # <c|i> is simply mo(i,c), so
        p[:, i] = mo[orbidx].T
    # svd: u defines a rotation of molecular orbitals,
    u, s, _ = la.svd(p)
    log.debug(1, "projection singular values:\n%s" % s)
    if n < nao:
        log.warning("using incomplete projected virtual space is an experimental feature,"
                    " use with caution")
        log.debug(0, "current projected virtual space cut-off = %10.4f", s[n])
    else:
        log.debug(0, "no cut-off in projected virtual space")
        if s[nao - 1] < 1e-8:
            log.warning("projected virtual space is likely to be linear dependent, with"
                    "smallest singular value = %10.4f\ncut-off is suggested", s[nao-1])

    # apply rotation
    mo1 = np.dot(mo, u)
    # return core and active
    return mo1[:, n:], mo1[:, :n]

def get_qps_local(ncas, nPAO, PAOidx, mo, mo_energy, nImp, ImpHam):
    norb = mo_energy.size // 2

    log.info("ncore = %d ncas = %d", norb-ncas, ncas)
    # number of electrons per spin
    nelecas = ncas - nPAO
    mo_v = [np.sum(mo[:norb, i]**2) for i in range(norb*2)]
    mo_v_ord = np.argsort(mo_v, kind='mergesort')
    mo_a, mo_b = mo_v_ord[norb:], mo_v_ord[:norb]

    # ordered so that those close to the fermi surface come first
    # AO: e < 0, v > 0.5
    AO = sorted(filter(lambda i: i < norb, mo_a))[::-1]
    # AV: e > 0, v > 0.5
    AV = sorted(filter(lambda i: i >= norb, mo_a))
    # BO: e > 0, v < 0.5
    BO = sorted(filter(lambda i: i >= norb, mo_b))
    # BV: e < 0, v < 0.5
    BV = sorted(filter(lambda i: i < norb, mo_b))[::-1]

    # now extract coefficents
    cAO, cAV = mo[:, AO], mo[:, AV]
    cBO, cBV = np.vstack((mo[norb:, BO], mo[:norb, BO])), np.vstack((mo[norb:, BV], mo[:norb, BV]))

    # transform 2eInt (ccdd part only) to occupied basis
    assert ImpHam.H2["cccd"] is None or la.norm(ImpHam.H2["cccd"]) < 1e-10
    assert ImpHam.H2["cccc"] is None or la.norm(ImpHam.H2["cccc"]) < 1e-10
    if settings.save_mem:
        w = transform_local_qp(cAO[:nImp], cBO[:nImp],
                               cAO[norb:norb+nImp], cBO[norb:norb+nImp], ImpHam.H2["ccdd"][0])
    else:
        assert la.norm(ImpHam.H2["ccdd"][0] - ImpHam.H2["ccdd"][2]) < 1e-10
        assert la.norm(ImpHam.H2["ccdd"][1] - ImpHam.H2["ccdd"][2]) < 1e-10
        w = transform_local_qp(cAO[:norb], cBO[:norb],
                               cAO[norb:], cBO[norb:], ImpHam.H2["ccdd"][2])
    # now localize the occupied space
    locA, locB = Localizer(w[0]), Localizer(w[1])
    locA.optimize()
    locB.optimize()
    # rotate
    cAOloc, cBOloc = np.dot(cAO, locA.coefs.T), np.dot(cBO, locB.coefs.T)
    # compute the indicators
    # self coulomb repulsion
    fA = np.asarray([locA.Int2e[i,i,i,i] for i in range(len(AO))])
    fB = np.asarray([locB.Int2e[i,i,i,i] for i in range(len(BO))])
    # weight on impurity
    wA = np.sum(cAOloc[:nImp]**2, 0) + np.sum(cAOloc[norb:norb+nImp]**2, 0)
    wB = np.sum(cBOloc[:nImp]**2, 0) + np.sum(cBOloc[norb:norb+nImp]**2, 0)
    ordA, ordB = np.argsort(wA, kind='mergesort'), np.argsort(wB, kind='mergesort')
    log.debug(1, "Spin A:\nCoulomb repulsion: \n%s\nImpurity weight: \n%s", fA[ordA], wA[ordA])
    log.debug(1, "Spin B:\nCoulomb repulsion: \n%s\nImpurity weight: \n%s", fB[ordB], wB[ordB])
    # take out core and cas from occupied set
    coreOA, casOA = cAOloc[:, ordA[:len(AO)-nelecas]], cAOloc[:, ordA[len(AO)-nelecas:]]
    coreOB, casOB = cBOloc[:, ordB[:len(BO)-nelecas]], cBOloc[:, ordB[len(BO)-nelecas:]]
    # handle virtual orbitals
    coreVA, casVA = proj_virtual(cAV, PAOidx, nPAO)
    coreVB, casVB = proj_virtual(cBV, PAOidx, nPAO)
    core = np.asarray([
        np.hstack((coreOA, np.vstack((coreVB[norb:], coreVB[:norb])))),
        np.hstack((coreOB, np.vstack((coreVA[norb:], coreVA[:norb]))))
    ])
    cas = np.asarray([
        np.hstack((casOA, casVA)),
        np.hstack((casOB, casVB)),
    ])
    casinfo = (
        (nelecas, 0, nPAO),
        (nelecas, 0, nPAO)
    )
    return core, cas, casinfo

def get_qps_nelec(ncas, nelec, mo, mo_energy):
    """
    Define CAS space based on number of electrons.
    ncas is number of spatial CAS orbital.
    """
    norb = mo_energy.size // 2

    log.info("ncore = %d ncas = %d", norb-ncas, ncas)
    # number of electrons per spin
    nelecas = nelec // 2
    mo_v = [np.sum(mo[:norb, i]**2) for i in range(norb*2)]
    mo_v_ord = np.argsort(mo_v, kind='mergesort')
    # if v > 0.5, classify as alpha modes, otherwise beta modes
    mo_a, mo_b = mo_v_ord[norb:], mo_v_ord[:norb]

    # ordered so that those close to the fermi surface come first
    # AO: e < 0, v > 0.5
    AO = sorted(filter(lambda i: i < norb, mo_a))[::-1]
    # AV: e > 0, v > 0.5
    AV = sorted(filter(lambda i: i >= norb, mo_a))
    # BO: e > 0, v < 0.5
    BO = sorted(filter(lambda i: i >= norb, mo_b))
    # BV: e < 0, v < 0.5
    BV = sorted(filter(lambda i: i < norb, mo_b))[::-1]

    # divide into cas and core
    casA_idx = AO[:nelecas][::-1] + AV[:ncas - nelecas]
    casB_idx = BO[:nelecas][::-1] + BV[:ncas - nelecas]
    coreA_idx = AO[nelecas:] + BV[ncas-nelecas:]
    coreB_idx = BO[nelecas:] + AV[ncas-nelecas:]

    # now seriously classify casA and casB into occ, partial and virt
    casA_occ, casA_part, casA_virt = [], [], []
    # according to energy and particle character
    for idx in casA_idx:
        if mo_v[idx] > 0.7 and mo_energy[idx] < -1e-4:
            casA_occ.append(idx)
        elif mo_v[idx] > 0.7 and mo_energy[idx] > 1e-4:
            casA_virt.append(idx)
        else:
            casA_part.append(idx)

    casB_occ, casB_part, casB_virt = [], [], []
    for idx in casB_idx:
        if mo_v[idx] < 0.3 and mo_energy[idx] > 1e-4:
            casB_occ.append(idx)
        elif mo_v[idx] < 0.3 and mo_energy[idx] < -1e-4:
            casB_virt.append(idx)
        else:
            casB_part.append(idx)

    # extract cas from mo
    casA = mo[:, casA_occ+casA_part+casA_virt]
    casB = np.vstack((
        mo[norb:, casB_occ+casB_part+casB_virt],
        mo[:norb, casB_occ+casB_part+casB_virt]))
    casinfo = (
        (len(casA_occ), len(casA_part), len(casA_virt)),
        (len(casB_occ), len(casB_part), len(casB_virt))
    )
    # extract core
    coreA = mo[:, coreA_idx]
    coreB = np.vstack((
        mo[norb:, coreB_idx],
        mo[:norb, coreB_idx]))
    return np.asarray([coreA, coreB]), np.asarray([casA, casB]), casinfo

def get_qps_energy(ncas, mo, mo_energy):
    """
    Define CAS space based on MO energy.
    ncas is number of spatial CAS orbital.
    """
    norb = mo_energy.size // 2
    log.info("ncore = %d ncas = %d", norb-ncas, ncas)
    log.info("core orbital energy cut-off = %20.12f",
             max(mo_energy[norb-ncas-1], mo_energy[norb+ncas])
             if ncas < norb else float("Inf"))
    log.info("active orbital energy cut-off = %20.12f",
             max(mo_energy[norb-ncas], mo_energy[norb+ncas-1]))

    # generate core
    core = np.empty((2, norb*2, norb - ncas))
    # alpha : first norb-ncas modes
    core[0] = mo[:, :(norb - ncas)] # VA, UB
    # beta : last norb-ncas modes
    core[1, :norb] = mo[norb:, (norb + ncas):] # VB
    core[1, norb:] = mo[:norb, (norb + ncas):] # UA

    # cas 2 ncas modes
    cas_temp = mo[:, (norb - ncas):(norb + ncas)]
    cas_energy = mo_energy[(norb - ncas):(norb + ncas)]
    cas = [{"o": [], "v": [], "p": []}, {"o": [], "v": [], "p": []}]

    cas_v = [la.norm(cas_temp[:norb, i])**2 for i in range(ncas*2)]
    order = np.argsort(cas_v, kind='mergesort')
    for idx in order[ncas:]: # alpha
        if cas_energy[idx] < -1e-4:
            cas[0]['o'].append(cas_temp[:, idx])
        elif cas_energy[idx] > 1e-4:
            cas[0]['v'].append(cas_temp[:, idx])
        else:
            cas[0]['p'].append(cas_temp[:, idx])
    for idx in order[:ncas]: # beta
        if cas_energy[idx] < -1e-4:
            cas[1]['v'].append(cas_temp[list(range(norb, norb*2)) + list(range(norb)), idx])
        elif cas_energy[idx] > 1e-4:
            cas[1]['o'].append(cas_temp[list(range(norb, norb*2)) + list(range(norb)), idx])
        else:
            cas[1]['p'].append(cas_temp[list(range(norb, norb*2)) + list(range(norb)), idx])
    casinfo = [(len(cas[i]['o']), len(cas[i]['p']), len(cas[i]['v'])) for i in range(2)]
    cas = np.asarray([(np.asarray(cas[i]['o'] + cas[i]['p'] + cas[i]['v']).T) for i in range(2)])

    for s in range(2):
        log.info("In CAS (spin %d):\n"
                 "Occupied (e<mu): %d\n""Virtual  (e>mu): %d\n"
                 "Partial Occupied: %d\n", s, casinfo[s][0],
                 casinfo[s][2], casinfo[s][1])
    return core, np.asarray(cas), casinfo

def buildCASHamiltonian(Ham, core, cas, Mu=None):
    """
    Construct CAS Hamiltonian based on Ham, core and cas orbitals.
    """
    norb = Ham.norb
    cVA, cVB, cUA, cUB = core[0, :norb], core[1, :norb], core[1, norb:], core[0, norb:]
    cRhoA = np.dot(cVA, cVA.T)
    cRhoB = np.dot(cVB, cVB.T)
    cKappaBA = np.dot(cUB, cVA.T)

    # Mu term if exists
    if Mu is not None:
        Ham = deepcopy(Ham)
        Ham.H1["cd"][0] -= np.eye(norb) * Mu
        Ham.H1["cd"][1] -= np.eye(norb) * Mu

    # zero-energy
    _H0 = Ham.H0

    # core-core energy
    _H0 += np.sum(cRhoA * Ham.H1["cd"][0] + cRhoB * Ham.H1["cd"][1] + \
            2 * cKappaBA.T * Ham.H1["cc"][0])

    # core-fock
    if settings.save_mem:
        _v = np.asarray(scf._get_veff_bcs_save_mem(cRhoA, cRhoB, cKappaBA, \
                Ham.H2["ccdd"]))
        nv = _v.shape[1]
        v = np.zeros((3, norb, norb))
        v[:, :nv, :nv] = _v
    elif Ham.H2["cccd"] is None or la.norm(Ham.H2["cccd"]) == 0:
        v = np.asarray(scf._get_veff_bcs(cRhoA, cRhoB, cKappaBA, Ham.H2["ccdd"]))
    else:
        v = np.asarray(scf._get_veff_bcs_full(cRhoA, cRhoB, cKappaBA, Ham.H2["ccdd"], Ham.H2["cccd"], Ham.H2["cccc"]))

    # core-core two-body
    _H0 += 0.5 * np.sum(cRhoA * v[0] + cRhoB * v[1] + 2 * cKappaBA.T * v[2])
    VA, VB, UA, UB = cas[0,:norb], cas[1,:norb], cas[1,norb:], cas[0,norb:]

    if settings.save_mem:
        H0, CD, CC, CCDD, CCCD, CCCC = transform_save_mem(VA, VB, UA, \
                UB, _H0, Ham.H1["cd"][0] + v[0], Ham.H1["cd"][1] + v[1], \
                Ham.H1["cc"][0] + v[2], Ham.H2["ccdd"][0])
    else:
        H0, CD, CC, CCDD, CCCD, CCCC = transform(VA, VB, UA, UB, _H0, \
            Ham.H1["cd"][0] + v[0], Ham.H1["cd"][1] + v[1], Ham.H1["cc"][0] + \
            v[2], Ham.H2["ccdd"][0], Ham.H2["ccdd"][1], Ham.H2["ccdd"][2])
    return integral.Integral(cas.shape[2], False, True, H0, {"cd": CD, "cc": CC}, \
            {"ccdd": CCDD, "cccd": CCCD, "cccc": CCCC}), _H0

def rotateHam(rotmat, Ham):
    H0, CD, CC, CCDD, CCCD, CCCC = transform_local(rotmat[0], rotmat[1], Ham.H0, \
            Ham.H1["cd"][0], Ham.H1["cd"][1], Ham.H1["cc"][0], Ham.H2["ccdd"][0], \
            Ham.H2["ccdd"][1], Ham.H2["ccdd"][2], Ham.H2["cccd"][0], \
            Ham.H2["cccd"][1], Ham.H2["cccc"][0])
    return integral.Integral(Ham.norb, False, True, H0, {"cd": CD, "cc": CC}, \
            {"ccdd": CCDD, "cccd": CCCD, "cccc": CCCC})

def split_localize(orbs, info, Ham, basis = None):
    """
    Split localize the orbitals using ER method.
    """
    spin = 2
    norbs = Ham.H1["cd"].shape[1]
    localorbs = np.empty_like(orbs)
    rotmat = np.zeros_like(Ham.H1["cd"])
    for s in range(spin):
        occ, part, virt = info[s]
        if occ > 0:
            localizer = Localizer(Ham.H2["ccdd"][s, \
                    :occ, :occ, :occ, :occ])
            log.info("Localization: Spin %d, occupied", s)
            localizer.optimize()
            occ_coefs = localizer.coefs.T
            localorbs[s, :, :occ] = np.dot(orbs[s,:,:occ], occ_coefs)
            rotmat[s, :occ, :occ] = occ_coefs
        if virt > 0:
            localizer = Localizer(Ham.H2["ccdd"][s, \
                    -virt:, -virt:, -virt:, -virt:])
            log.info("Localization: Spin %d, virtual", s)
            localizer.optimize()
            virt_coefs = localizer.coefs.T
            localorbs[s, :, -virt:] = np.dot(orbs[s, :, -virt:], virt_coefs)
            rotmat[s, -virt:, -virt:] = virt_coefs
        if part > 0:
            localizer = Localizer(Ham.H2["ccdd"][s, occ:norbs-virt, \
                occ:norbs-virt, occ:norbs-virt, occ:norbs-virt])
            log.info("Localization: Spin %d, partially occupied:", s)
            localizer.optimize()
            part_coefs = localizer.coefs.T
            localorbs[s, :, occ:norbs-virt] = \
                    np.dot(orbs[s,:,occ:norbs-virt], part_coefs)
            rotmat[s, occ:norbs-virt, occ:norbs-virt] = part_coefs
    if basis is not None:
        # match alpha, beta basis
        # localorbs contain v and u parts with respect to embedding quasiparticles
        localbasis = basisToSpin(np.tensordot(basisToCanonical(basis), \
                basisToCanonical(localorbs), (2, 0)))
        nscsites = basis.shape[2] // 2
        localbasis0 = np.sqrt(localbasis[0,:,:nscsites]**2 + localbasis[0,:,nscsites:]**2)
        localbasis1 = np.sqrt(localbasis[1,:,:nscsites]**2 + localbasis[1,:,nscsites:]**2)
        ovlp = np.tensordot(localbasis0, localbasis1, ((0,1), (0,1)))
        ovlp_sq = ovlp ** 2

        idx1, idx2 = opt.linear_sum_assignment(1.0 - ovlp_sq)
        indices = list(zip(idx1, idx2))
        vals = [ovlp_sq[idx] for idx in indices]
        log.debug(1, "Quasiparticle pairs and their overlap:")
        for i in range(norbs):
            log.debug(1, "(%2d, %2d) -> %12.6f", indices[i][0], indices[i][1], vals[i])
        log.info("Match localized quasiparticles: max %5.2f min %5.2f ave %5.2f", \
                np.max(vals), np.min(vals), np.average(vals))
        # update localorbs and rotmat
        orderb = [idx[1] for idx in indices]
        localorbs[1] = localorbs[1][:, orderb]
        rotmat[1] = rotmat[1][:, orderb]

        localbasis[1] = localbasis[1][:,:,orderb]
        # make spin up and down basis have the same sign, i.e.
        # inner product larger than 1
        for i in range(norbs):
            if np.sum(localbasis[0,:,:,i] * localbasis[1,:,:,i]) < 0:
                localorbs[1,:,i] *= -1.
                rotmat[1,:,i] *= -1

    HamLocal = rotateHam(rotmat, Ham)
    return HamLocal, localorbs, rotmat

def momopt(old_basis, new_basis):
    norb = old_basis.shape[2] // 2
    old_basis1 = np.sqrt(old_basis[:,:,:norb] ** 2 + old_basis[:,:,norb:] ** 2)
    new_basis1 = np.sqrt(new_basis[:,:,:norb] ** 2 + new_basis[:,:,norb:] ** 2)
    # use Hungarian algorithm to match the basis
    ovlp = 0.5 * np.tensordot(np.abs(old_basis1), np.abs(new_basis1), ((0,1,2), (0,1,2)))
    ovlp_sq = ovlp ** 2

    idx1, idx2 = opt.linear_sum_assignment(1.0 - ovlp_sq)
    indices = list(zip(idx1, idx2))
    vals = [ovlp_sq[idx] for idx in indices]
    log.info("MOM reorder quality: max %5.2f min %5.2f ave %5.2f",
             np.max(vals), np.min(vals), np.average(vals))

    reorder = [idx[1] for idx in indices]
    return reorder, np.average(vals)

def reorder(order, Ham, orbs, rot = None):
    # order 4 1 3 2 means 4 to 1, 1 to 2, 3 to 3, 2 to 4
    # reorder in place
    orbs = orbs[:, :, order]
    Ham.H1["cd"] = Ham.H1["cd"][:, order, :]
    Ham.H1["cd"] = Ham.H1["cd"][:, :, order]
    Ham.H1["cc"] = Ham.H1["cc"][:, order, :]
    Ham.H1["cc"] = Ham.H1["cc"][:, :, order]
    Ham.H2["ccdd"] = Ham.H2["ccdd"][:, order, :, :, :]
    Ham.H2["ccdd"] = Ham.H2["ccdd"][:, :, order, :, :]
    Ham.H2["ccdd"] = Ham.H2["ccdd"][:, :, :, order, :]
    Ham.H2["ccdd"] = Ham.H2["ccdd"][:, :, :, :, order]
    Ham.H2["cccd"] = Ham.H2["cccd"][:, order, :, :, :]
    Ham.H2["cccd"] = Ham.H2["cccd"][:, :, order, :, :]
    Ham.H2["cccd"] = Ham.H2["cccd"][:, :, :, order, :]
    Ham.H2["cccd"] = Ham.H2["cccd"][:, :, :, :, order]
    Ham.H2["cccc"] = Ham.H2["cccc"][:, order, :, :, :]
    Ham.H2["cccc"] = Ham.H2["cccc"][:, :, order, :, :]
    Ham.H2["cccc"] = Ham.H2["cccc"][:, :, :, order, :]
    Ham.H2["cccc"] = Ham.H2["cccc"][:, :, :, :, order]
    if rot is not None:
        rot = rot[:, :, order]
        return Ham, orbs, rot
    else:
        return Ham, orbs

class BCSDmrgCI(object):
    def __init__(self, ncas, splitloc=False, cisolver=None, mom_reorder=True,
                 algo="nelec", tmpDir="./tmp", **kwargs):
        """
        BCS DMRG-CI solver.

        Additional required keywords
            - nelec: nelecas
            - energy: none
            - local: nocc, PAOidx
        """
        assert algo in ["nelec", "energy", "local"]
        self.get_qps = get_qps(ncas, algo, **kwargs)

        self.ncas = ncas
        self.splitloc = splitloc
        log.eassert(cisolver is not None, "No default ci solver is available"
                    " with CASCI, you have to use Block")
        self.cisolver = cisolver
        self.scfsolver = scf.SCF()

        # reorder scheme for restart block calculations
        if mom_reorder:
            if block.Block.reorder:
                log.warning("Using maximal overlap method (MOM) to reorder localized "
                            "orbitals, turning off Block reorder option")
                block.Block.reorder = False

        self.mom_reorder = mom_reorder
        self.localized_cas = None
        self.cas = None
        self.core = None
        self.tmpDir = tmpDir

    def run(self, Ham, ci_args={}, guess=None, basis=None, similar=False,
            scf=True, Mu=None):
        # ci_args is a list or dict for ci solver, or None
        if scf:
            mo, mo_energy = get_BCS_mo(self.scfsolver, Ham, guess, Mu=Mu)
        else:
            mo, mo_energy = guess
        if basis is not None:
            nmodes = np.sum(mo_energy < 0)
            GRhoHFB = np.dot(mo[:, :nmodes], mo[:, :nmodes].T)
            nImp = basis.shape[-2] // 2
            nbasis = basis.shape[-1]
            nA_HFB, nB_HFB = np.sum(np.diag(GRhoHFB)[:nImp]), np.sum(1. - np.diag(GRhoHFB)[nbasis:nbasis+nImp])
            log.info("nelec (HFB) = %20.12f" % (nA_HFB + nB_HFB))
            core, cas, casinfo = self.get_qps(mo, mo_energy, basis.shape[-2] // 2, Ham)
        else:
            core, cas, casinfo = self.get_qps(mo, mo_energy, None, Ham)
        coreGRho = np.dot(core[0], core[0].T)
        casHam, _ = buildCASHamiltonian(Ham, core, cas, Mu=Mu)

        if self.splitloc:
            casHam, cas, _ = split_localize(cas, casinfo, casHam, basis=basis)

        if self.mom_reorder:
            log.eassert(basis is not None, "maximum overlap method (MOM) requires embedding basis")
            if self.localized_cas is None:
                order = gaopt(casHam, tmp = self.tmpDir)
            else:
                # define cas_basis
                cas_basis = basisToSpin(np.tensordot(basisToCanonical(basis),
                                        basisToCanonical(cas), (2,0)))
                # cas_basis and self.localized_cas are both in
                # atomic representation now
                order, q = momopt(self.localized_cas, cas_basis)
                # if the quality of mom is too bad, we reorder the orbitals
                # using genetic algorithm
                # FIXME seems larger q is a better choice
                if q < 0.7:
                    order = gaopt(casHam, tmp=self.tmpDir)

            log.info("Orbital order: %s", order)
            # reorder casHam and cas
            casHam, cas = reorder(order, casHam, cas)
            # store cas in atomic basis
            self.localized_cas = basisToSpin(np.tensordot(basisToCanonical(basis),
                                             basisToCanonical(cas), (2,0)))

        # save cas orbital and core for interacting bath
        self.cas = cas
        self.core = core

        casGRho, E = self.cisolver.run(casHam, **ci_args)
        cas1 = basisToCanonical(cas)
        GRho = mdot(cas1, casGRho, cas1.T) + coreGRho

        # remove Mu contribution to E
        if Mu is not None:
            norb = Ham.norb
            E += Mu * (np.diag(GRho)[:norb].sum() + norb - np.diag(GRho)[norb:].sum())
        return GRho, E

    def cleanup(self):
        self.cisolver.cleanup()

    def run_dmet_ham(self, Ham, ci_args = {}, **kwargs):
        """
        Solve dmet hamiltonian with fixed wavefunction, but scaled Hamiltonian.
        """
        # ci_args is a list or dict for ci solver, or None
        spin = Ham.H1["cd"].shape[0]
        log.eassert(spin == 2, "spin-restricted CASCI solver is not implemented")
        core = self.core
        cas = self.cas
        #coreRho = np.asarray([np.dot(core[0], core[0].T), \
        #        np.dot(core[1], core[1].T)])
        casHam = buildCASHamiltonian(Ham, core, cas)[0]

        #casRho, E = self.cisolver.run_dmet_ham(casHam, nelec = self.nelecas, **ci_args)
        #rho = np.asarray([mdot(cas[0], casRho[0], cas[0].T), \
        #        mdot(cas[1], casRho[1], cas[1].T)]) + coreRho
        #return rho, E
        #E = self.cisolver.evaluate(casHam.H0, casHam.H1, casHam.H2, op = "dmet ham")
        E = self.cisolver.run_dmet_ham(casHam, **ci_args)
        return E
