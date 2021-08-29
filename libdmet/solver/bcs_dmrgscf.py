import time
import numpy as np
import scipy.linalg as la
import pyscf
from pyscf.mcscf import mc1step_uhf, mc1step
from libdmet.utils.misc import mdot
from libdmet.solver import bcs_dmrgci, block
from libdmet.solver.scf import pyscflogger, incore_transform
from libdmet.routine.bcs_helper import extractRdm, combineRdm, \
        basisToCanonical, basisToSpin
from libdmet.utils import logger as log
import copy

def g_atomic_to_canonical(g, mo):
    norb = mo.shape[1] // 2
    v_A, v_B, u_A, u_B = mo[0, :norb], mo[1, :norb], \
            mo[1, norb:], mo[0, norb:]
    g_A, g_B, g_D = g
    val001 = np.dot(np.dot(v_B.T, g_D.T), u_A)
    g_b = 1.0*val001.T + \
            -1.0*np.dot(np.dot(u_A.T, g_A.T), u_A) + \
            -1.0*val001 + \
            1.0*np.dot(np.dot(v_B.T, g_B), v_B)
    val002 = np.dot(v_A.T, g_A)
    val003 = np.dot(u_B.T, g_B.T)
    val004 = np.dot(v_A.T, g_D)
    g_d = -1.0*np.dot(val003, v_B) + \
            1.0*np.dot(val002, u_A) + \
            1.0*np.dot(val004, v_B) + \
            -1.0*np.dot(np.dot(u_B.T, g_D.T), u_A)
    val005 = np.dot(val004, u_B)
    g_a = -1.0*np.dot(val003, u_B) + \
            1.0*val005 + \
            1.0*np.dot(val002, v_A) + \
            -1.0*val005.T
    return (g_a, g_b, g_d)

def g_canonical_to_atomic(g, mo):
    norb = mo.shape[1] // 2
    v_A, v_B, u_A, u_B = mo[0, :norb].T, mo[1, :norb].T, \
            mo[0, norb:].T, mo[1, norb:].T
    g_A, g_B, g_D = g
    val001 = np.dot(np.dot(v_B.T, g_D.T), u_A)
    g_b = 1.0*val001.T + \
            -1.0*np.dot(np.dot(u_A.T, g_A.T), u_A) + \
            -1.0*val001 + \
            1.0*np.dot(np.dot(v_B.T, g_B), v_B)
    val002 = np.dot(v_A.T, g_A)
    val003 = np.dot(u_B.T, g_B.T)
    val004 = np.dot(v_A.T, g_D)
    g_d = -1.0*np.dot(val003, v_B) + \
            1.0*np.dot(val002, u_A) + \
            1.0*np.dot(val004, v_B) + \
            -1.0*np.dot(np.dot(u_B.T, g_D.T), u_A)
    val005 = np.dot(val004, u_B)
    g_a = -1.0*np.dot(val003, u_B) + \
            1.0*val005 + \
            1.0*np.dot(val002, v_A) + \
            -1.0*val005.T
    return (g_a, g_b, g_d)

def gen_g_hop(casscf, mo, u, casdm1s, casdm2s, eris):
    log.eassert(u == 1, "")
    ncas = casscf.ncas
    ncore = mo.shape[2] - ncas

    # get density matrices in "atomic" basis
    dm1, dm2 = casscf.make_rdm12s(mo_coeff = mo, casdm1 = casdm1s, \
            casdm2 = casdm2s)
    # and Hamiltonian
    ham = casscf.integral
    # conditions of integrals to satisfy
    log.eassert(ham.H2["cccd"] is None or la.norm(ham.H2["cccd"]) == 0, "")
    log.eassert(ham.H2["cccc"] is None or la.norm(ham.H2["cccc"]) == 0, "")
    log.eassert(la.norm(ham.H2["ccdd"][0] - ham.H2["ccdd"][2]) < 1e-10, "")
    log.eassert(la.norm(ham.H2["ccdd"][1] - ham.H2["ccdd"][2]) < 1e-10, "")
    
    (h_A, h_B), D = ham.H1["cd"], ham.H1["cc"][0]
    w = ham.H2["ccdd"][2]
    (rho_A, rho_B), kappaBA = dm1[0], -dm1[1].T
    (Gamma_aa, Gamma_bb, Gamma_ab), (Gamma_2a, Gamma_2b), Gamma_4 = dm2

    from libdmet.casscf.g_hop_atomic import g_hop_atomic    
    gorb_atomic, h_diag_atomic, hop_atomic = g_hop_atomic(h_A, h_B, D, w, \
            rho_A, rho_B, kappaBA, Gamma_aa, Gamma_bb, Gamma_ab, Gamma_2a, \
            Gamma_2b, Gamma_4)

    gorb = casscf.pack_uniq_var(g_atomic_to_canonical(gorb_atomic, mo))

    def hop(x):
        # transform x to atomic basis
        x_atomic = g_canonical_to_atomic(casscf.unpack_uniq_var(x), mo)
        # then act on x
        hx = hop_atomic(*x_atomic)
        # transform back
        return casscf.pack_uniq_var(g_atomic_to_canonical(hx, mo))
    
    def fhdiag(x, e):
        x_atomic = g_canonical_to_atomic(casscf.unpack_uniq_var(x), mo)
        hdiagd = [h-e for h in h_diag_atomic]
        for h in hdiagd:
            h[abs(h)<1e-8] = 1e-8
        x1 = [x0 / h0 for x0, h0 in zip(x_atomic, hdiagd)]
        return casscf.pack_uniq_var(g_atomic_to_canonical(tuple(x1), mo))

    return gorb, lambda *args: gorb, hop, fhdiag

def kernel(casscf, mo_coeff, tol=1e-7, conv_tol_grad=None, macro=50, micro=3,
           ci0=None, callback=None, verbose=None,
           dump_chk=True, dump_chk_ci=False):
    if verbose is None:
        verbose = casscf.verbose
    log = pyscflogger.Logger(casscf.stdout, verbose)
    cput0 = (pyscflogger.process_clock(), pyscflogger.perf_counter())
    log.debug('Start 1-step CASSCF')

    mo = mo_coeff
    nmo = mo[0].shape[1]
    ncore = casscf.ncore
    ncas = casscf.ncas
    #TODO: lazy evaluate eris, to leave enough memory for FCI solver
    eris = None
    e_tot, e_ci, fcivec = casscf.casci(mo, ci0, eris)
    log.info('CASCI E = %.15g', e_tot)
    if ncas == nmo:
        return True, e_tot, e_ci, fcivec, mo

    if conv_tol_grad is None:
        conv_tol_grad = np.sqrt(tol*.1)
        pyscflogger.info(casscf, 'Set conv_tol_grad to %g', conv_tol_grad)
    conv_tol_ddm = conv_tol_grad * 3
    max_cycle_micro = micro
    conv = False
    totmicro = totinner = 0
    norm_gorb = norm_gci = 0
    elast = e_tot
    r0 = None

    t1m = log.timer('Initializing 1-step CASSCF', *cput0)
    casdm1, casdm2 = casscf.fcisolver.make_rdm12s(fcivec, ncas, casscf.nelecas)
    norm_ddm = 1e2
    casdm1_last = casdm1
    t3m = t2m = log.timer('CAS DM', *t1m)
    for imacro in range(macro):
        if casscf.dynamic_micro_step:
            max_cycle_micro = max(micro, int(micro-2-np.log(norm_ddm)))
        imicro = 0
        rota = casscf.rotate_orb_cc(mo, lambda:casdm1, lambda:casdm2,
                                    eris, r0, conv_tol_grad, log)
        for u, g_orb, njk in rota:
            imicro += 1
            norm_gorb = la.norm(g_orb)
            if imicro == 1:
                norm_gorb0 = norm_gorb
            norm_t = la.norm(u-np.eye(nmo*2))
            if imicro == max_cycle_micro:
                log.debug('micro %d  |u-1|= %4.3g  |g[o]|= %4.3g  ',
                          imicro, norm_t, norm_gorb)
                break

            casdm1, casdm2, gci, fcivec = casscf.update_casdm(mo, u, fcivec, e_ci, eris)
            if isinstance(gci, np.ndarray):
                norm_gci = la.norm(gci)
            else:
                norm_gci = -1
            norm_ddm =(la.norm(casdm1[0] - casdm1_last[0])
                     + la.norm(casdm1[1] - casdm1_last[1]))
            t3m = log.timer('update CAS DM', *t3m)
            log.debug('micro %d  |u-1|= %4.3g  |g[o]|= %4.3g  ' \
                      '|g[c]|= %4.3g  |ddm|= %4.3g',
                      imicro, norm_t, norm_gorb, norm_gci, norm_ddm)

            if callable(callback):
                callback(locals())

            t3m = log.timer('micro iter %d'%imicro, *t3m)
            if (norm_t < 1e-4 or
                (norm_gorb < conv_tol_grad*.8 and norm_ddm < conv_tol_ddm)):
                break

        rota.close()

        totmicro += imicro
        totinner += njk

        from libdmet.routine.bcs_helper import basisToCanonical, basisToSpin
        mo = basisToSpin(np.dot(basisToCanonical(mo), u))

        eris = None
        t2m = log.timer('update eri', *t3m)

        elast = e_tot
        e_tot, e_ci, fcivec = casscf.casci(mo, fcivec, eris)
        casdm1, casdm2 = casscf.fcisolver.make_rdm12s(fcivec, ncas, casscf.nelecas)
        norm_ddm =(la.norm(casdm1[0] - casdm1_last[0])
                 + la.norm(casdm1[1] - casdm1_last[1]))
        casdm1_last = casdm1
        log.debug('CAS space CI energy = %.15g', e_ci)
        log.timer('CASCI solver', *t2m)
        log.info('macro iter %d (%d JK  %d micro), CASSCF E = %.15g  dE = %.8g',
                 imacro, njk, imicro, e_tot, e_tot-elast)
        log.info('               |grad[o]|= %4.3g  |grad[c]|= %4.3g  |ddm|= %4.3g',
                 norm_gorb0, norm_gci, norm_ddm)
        t2m = t1m = log.timer('macro iter %d'%imacro, *t1m)

        if (abs(e_tot - elast) < tol
            and (norm_gorb0 < conv_tol_grad and norm_ddm < conv_tol_ddm)):
            conv = True

        if dump_chk:
            casscf.dump_chk(locals())

        if conv: break

    if conv:
        log.info('1-step CASSCF converged in %d macro (%d JK %d micro) steps',
                 imacro+1, totinner, totmicro)
    else:
        log.info('1-step CASSCF not converged, %d macro (%d JK %d micro) steps',
                 imacro+1, totinner, totmicro)
    log.timer('1-step CASSCF', *cput0)
    return conv, e_tot, e_ci, fcivec, mo

def proj_GRho(orbs, GRho):
    from libdmet.routine.bcs_helper import swapSpin
    spin = 2
    ncas = orbs.shape[2]
    norb = orbs.shape[1] // 2

    proj_rhoA = mdot(orbs[0].T, GRho, orbs[0])
    proj_rhoB = mdot(orbs[1].T, swapSpin(GRho), orbs[1])
    poccA, porbA = la.eigh(proj_rhoA)
    poccB, porbB = la.eigh(proj_rhoB)

    infoA = (np.sum(poccA > 0.7), np.sum((poccA < 0.3) *(poccA > 0.7)), \
            np.sum(poccA < 0.3))
    infoB = (np.sum(poccB > 0.7), np.sum((poccB < 0.3) *(poccB > 0.7)), \
            np.sum(poccB < 0.3))
    proj_orbs_rot = np.asarray([porbA[:,::-1], porbB[:,::-1]])
    proj_orbs = np.asarray(list(map(np.dot, orbs, proj_orbs_rot)))
    return proj_orbs, proj_orbs_rot, [infoA, infoB]

def make_rdm1s(solver, casscf):
    rhoA, rhoB, kappaBA = extractRdm(solver.onepdm())
    kappaAB = -kappaBA.T
    rot = casscf.rot
    return np.asarray([mdot(rot[0], rhoA, rot[0].T), \
            mdot(rot[1], rhoB, rot[1].T)]), mdot(rot[0], kappaAB, rot[1].T)

def make_rdm12s(solver, casscf):
    rhoA, rhoB, kappaBA = extractRdm(solver.onepdm())
    kappaAB = -kappaBA.T
    gamma0, gamma2, gamma4 = solver.twopdm()
    rot = casscf.rot
    rdm1s_cas = (np.asarray([mdot(rot[0], rhoA, rot[0].T), \
            mdot(rot[1], rhoB, rot[1].T)]), mdot(rot[0], kappaAB, rot[1].T))
    rotInv = np.asarray([rot[0].T, rot[1].T])
    gamma0_cas = incore_transform(gamma0, (rotInv,) * 4) # AA, BB, AB
    gamma2_casA = np.tensordot(rotInv[0], gamma2[0], (0,0))
    gamma2_casA = np.tensordot(rotInv[0], gamma2_casA, (0,1))
    gamma2_casA = np.tensordot(gamma2_casA, rotInv[0], (3,0))
    gamma2_casA = -np.transpose(np.tensordot(gamma2_casA, rotInv[1], (2,0)), (0,1,3,2))
    gamma2_casB = np.tensordot(rotInv[1], gamma2[1], (0,0))
    gamma2_casB = np.tensordot(rotInv[1], gamma2_casB, (0,1))
    gamma2_casB = np.tensordot(gamma2_casB, rotInv[1], (3,0))
    gamma2_casB = -np.transpose(np.tensordot(gamma2_casB, rotInv[0], (2,0)), (0,1,3,2))
    gamma4_cas = np.tensordot(rotInv[0], gamma4[0], (0,0))
    gamma4_cas = np.tensordot(rotInv[0], gamma4_cas, (0,1))
    gamma4_cas = np.tensordot(gamma4_cas, rotInv[1], (3,0))
    gamma4_cas = np.tensordot(gamma4_cas, rotInv[1], (2,0))
    rdm2s_cas = (gamma0_cas, np.asarray([gamma2_casA, gamma2_casB]), \
            gamma4_cas[np.newaxis])
    return rdm1s_cas, rdm2s_cas

class BCS_DMRGSCF(mc1step_uhf.CASSCF):
    def __init__(self, mf, ncas, norb, Ham, fcisolver, nelecas = None, \
            frozen = [], splitloc = True, mom_reorder = True, \
            TmpDir = "/tmp"):
        mc1step_uhf.CASSCF.__init__(self, mf, ncas, norb*2, \
                (norb-ncas, norb-ncas), frozen)
        if log.Level[log.verbose] >= log.Level["DEBUG1"]:
            self.verbose = 5
        elif log.Level[log.verbose] >= log.Level["RESULT"]:
            self.verbose = 4
        else:
            self.verbose = 2
        if log.Level[log.verbose] <= log.Level["INFO"]:
            pyscflogger.flush.addkey("macro iter")
            pyscflogger.flush.addkey("1-step CASSCF")
        else:
            pyscflogger.flush.keywords = set([""])

        self.fcisolver = fcisolver
        self.splitloc = splitloc
        # reorder scheme for restart block calculations
        if mom_reorder:
            if block.Block.reorder:
                log.warning("Using maximal overlap method (MOM) to reorder localized "\
                        "orbitals, turning off Block reorder option")
                block.Block.reorder = False
        self.mom_reorder = mom_reorder
        self.tmpDir = TmpDir
        self.rot = None
        self.basis = None
        self.localized_cas = None
        mo_coefs = mf.mo_coeff
        self.GRho =mdot(mo_coefs[:, :norb], mo_coefs[:, :norb].T)

        def no_ao2mo(self, *args):
            log.warning("ao2mo called")
            return
        self.ao2mo = no_ao2mo
        self.fcisolver.make_rdm1s = lambda *args: \
                make_rdm1s(self.fcisolver, self)
        self.fcisolver.make_rdm12s = lambda *args: \
                make_rdm12s(self.fcisolver, self)

        self.integral = Ham

    def refresh(self, mf, ncas, norb, Ham, nelecas = None, frozen = []):
        fcisolver = copy.copy(self.fcisolver)
        mc1step_uhf.CASSCF.__init__(self, mf, ncas, norb*2, \
                (norb-ncas, norb-ncas), frozen)
        self.fcisolver = fcisolver
        self.rot = None
        self.basis = None
        mo_coefs = mf.mo_coeff
        self.GRho = mdot(mo_coefs[:, :norb], mo_coefs[:, :norb].T)

        def no_ao2mo(self, *args):
            log.warning("ao2mo called")
            return
        self.ao2mo = no_ao2mo
        self.integral = Ham
        self.converged = False

    def mc1step(self, *args, **kwargs):
        if 'basis' in kwargs.keys():
            self.basis = kwargs["basis"]
            kwargs.pop('basis')
        return mc1step_uhf.CASSCF.mc1step(self, *args, **kwargs)

    def mc2step(self, *args, **kwargs):
        log.error("mc2step is not implemented for bcs-casscf")

    def make_rdm1s(self, mo_coeff = None, ncas = None, casdm1 = None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ncas is None:
            ncas = self.ncas
        if casdm1 is None:
            casRho, casKappaAB = self.fcisolver.make_rdm1s()
        else:
            casRho, casKappaAB = casdm1
        ncore = mo_coeff.shape[2] - ncas
        coreGRho = np.dot(mo_coeff[0, :, :ncore], mo_coeff[0, :, :ncore].T)
        cas1 = basisToCanonical(mo_coeff[:,:,ncore:])
        casGRho = combineRdm(casRho[0], casRho[1], casKappaAB)
        rhoA, rhoB, kappaBA = extractRdm(mdot(cas1, casGRho, cas1.T) + coreGRho)
        return np.asarray([rhoA, rhoB]), -kappaBA.T

    def make_rdm12s(self, mo_coeff = None, ncas = None, casdm1 = None, casdm2 = None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if ncas is None:
            ncas = self.ncas
        if casdm2 is None or casdm2 is None:
            (casRho, casKappaAB), (casGamma0, casGamma2, casGamma4) = \
                    self.fcisolver.make_rdm12s()
        else:
            casRho, casKappaAB = casdm1
            casGamma0, casGamma2, casGamma4 = casdm2

        norb = mo_coeff.shape[1] // 2
        ncore = mo_coeff.shape[2] - ncas
        vA, vB, uA, uB = mo_coeff[0, :norb], mo_coeff[1, :norb], \
                mo_coeff[1, norb:], mo_coeff[0, norb:]
        vc_A, va_A = vA[:, :ncore], vA[:, ncore:]
        vc_B, va_B = vB[:, :ncore], vB[:, ncore:]
        uc_A, ua_A = uA[:, :ncore], uA[:, ncore:]
        uc_B, ua_B = uB[:, :ncore], uB[:, ncore:]

        from libdmet.casscf.pdm_transform import cas_pdm_transform
        return cas_pdm_transform(va_A, va_B, ua_A, ua_B, vc_A, vc_B, uc_A, uc_B, \
                casRho[0], casRho[1], -casKappaAB.T, casGamma0[0], casGamma0[1], \
                casGamma0[2], casGamma2[0], casGamma2[1], casGamma4[0])

    def kernel(self, mo_coeff=None, ci0=None, macro=None, micro=None,
               callback=None, _kern=kernel):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        else:
            self.mo_coeff = mo_coeff
        if macro is None: macro = self.max_cycle_macro
        if micro is None: micro = self.max_cycle_micro
        if callback is None: callback = self.callback

        if self.verbose > pyscflogger.QUIET:
            pyscf.gto.mole.check_sanity(self, self._keys, self.stdout)

        self.mol.check_sanity(self)
        self.dump_flags()

        self.converged, self.e_tot, e_cas, self.ci, self.mo_coeff = \
                _kern(self, mo_coeff,
                      tol=self.conv_tol, conv_tol_grad=self.conv_tol_grad,
                      macro=macro, micro=micro,
                      ci0=ci0, callback=callback, verbose=self.verbose)
        pyscflogger.note(self, 'CASSCF energy = %.15g', self.e_tot)
        return self.e_tot, e_cas, self.ci, self.mo_coeff

    def casci(self, mo_coeff, ci0 = None, eris = None):
        log.eassert(eris is None, "there might be a bug")

        ncas = self.ncas
        ncore = self.ncore[0]
        core, cas = mo_coeff[:, :, :ncore], mo_coeff[:, :, ncore:]
        casHam, H0 = bcs_dmrgci.buildCASHamiltonian(self.integral, core, cas)
        self.casHam = casHam
        if self.splitloc:
            # rotate cas quasiparticles
            split_cas, split_rot, casinfo = proj_GRho(cas, self.GRho)
            casHam = bcs_dmrgci.rotateHam(split_rot, casHam)
            casHam, cas_local, rot = bcs_dmrgci.split_localize(split_cas, casinfo, \
                    casHam, basis = self.basis)
            rot = np.asarray(list(map(np.dot, split_rot, rot)))
        else:
            rot = np.asarray([np.eye(ncas), np.eye(ncas)])

        if self.mom_reorder:
            log.eassert(self.basis is not None, \
                    "maximum overlap method (MOM) requires embedding basis")
            if self.localized_cas is None:
                order = bcs_dmrgci.gaopt(casHam, tmp = self.tmpDir)
            else:
                # define cas_basis
                cas_basis = basisToSpin(np.tensordot(basisToCanonical(self.basis), \
                        basisToCanonical(cas), (2,0)))
                order, q = bcs_dmrgci.momopt(self.localized_cas, cas_basis)
                if q < 0.7:
                    order = bcs_dmrgci.gaopt(casHam, tmp = self.tmpDir)

            log.info("Orbital order: %s", order)
            # reorder casHam and cas
            casHam, cas_local, rot = bcs_dmrgci.reorder(order, casHam, cas_local, rot)
            # store cas in atomic basis
            self.localized_cas = basisToSpin(np.tensordot(basisToCanonical(self.basis), \
                    basisToCanonical(cas_local), (2,0)))

        self.rot = rot

        # solve
        cas_local_GRho, E = self.fcisolver.run(casHam)
        # update self.GRho
        (rhoA, rhoB), kappaAB = self.make_rdm1s()
        self.GRho = np.asarray(combineRdm(rhoA, rhoB, kappaAB))

        return E, E-H0, None

    def gen_g_hop(self, mo, u, casdm1, casdm2, eris):
        if not self.exact_integral or (isinstance(u, int) and u == 1):
            return gen_g_hop(self, mo, u, casdm1, casdm2, eris)
        else:
            mo1 = basisToSpin(np.dot(basisToCanonical(mo), u))
            gorb, _, h_op, h_diag = gen_g_hop(self, mo1, 1, casdm1, casdm2, None)
            return gorb, lambda *args: gorb, h_op, h_diag

    def pack_uniq_var(self, xs):
        ncas = self.ncas
        ncore = self.mo_coeff.shape[2] - ncas
        length = ncas * ncore * 4 + ncore * ncore
        x = np.empty(length)
        xa, xb, xd = xs
        x[:ncas*ncore] = xa[ncore:, :ncore].ravel()
        x[ncas*ncore:2*ncas*ncore] = xb[ncore:, :ncore].ravel()
        x[2*ncas*ncore:3*ncas*ncore] = xd[ncore:, :ncore].ravel()
        x[3*ncas*ncore:4*ncas*ncore] = xd[:ncore, ncore:].ravel()
        x[4*ncas*ncore:] = xd[:ncore, :ncore].ravel()
        return x

    def unpack_uniq_var(self, x):
        # the order is xa_AC, xb_AC, xd_AC, xd_CA, xd_CC
        ncas = self.ncas
        ncore = self.mo_coeff.shape[2] - ncas
        length = ncas * ncore * 4 + ncore * ncore
        log.eassert(len(x) == length, "wrong length for the vector")
        xa_AC = x[:ncas*ncore].reshape((ncas, ncore))
        xb_AC = x[ncas*ncore:2*ncas*ncore].reshape((ncas, ncore))
        xd_AC = x[2*ncas*ncore:3*ncas*ncore].reshape((ncas, ncore))
        xd_CA = x[3*ncas*ncore:4*ncas*ncore].reshape((ncore, ncas))
        xd_CC = x[4*ncas*ncore:].reshape((ncore, ncore))
        xa = np.zeros((ncas+ncore, ncas+ncore))
        xb = np.zeros((ncas+ncore, ncas+ncore))
        xd = np.zeros((ncas+ncore, ncas+ncore))
        xa[ncore:, :ncore] = xa_AC
        xa[:ncore, ncore:] = -xa_AC.T
        xb[ncore:, :ncore] = xb_AC
        xb[:ncore, ncore:] = -xb_AC.T
        xd[ncore:, :ncore] = xd_AC
        xd[:ncore, ncore:] = xd_CA
        xd[:ncore, :ncore] = xd_CC
        return (xa, xb, xd)

    def update_rotate_matrix(self, dx, u0 = 1):
        dr = self.unpack_uniq_var(dx)
        K = np.vstack((
            np.hstack((   dr[0], dr[2])),
            np.hstack((-dr[2].T, dr[1]))
        ))
        # negative sign needed for exp(-K) on the left
        # on the right, we need its transpose, which is
        # exp(K) since the matrix is unitary
        return np.dot(u0, mc1step.expmat(K))

    def update_casdm(self, mo, u, fcivec, e_ci, eris):
        log.eassert(eris is None, "there might be a bug")
        ncas = self.ncas
        ncore = self.ncore[0]
        mo1 = basisToSpin(np.dot(basisToCanonical(mo), u))
        core, cas = mo1[:,:,:ncore], mo1[:,:,ncore:]
        # using exact integral
        log.eassert(self.exact_integral, "Only exact integral is implemented")
        casHam, H0 = bcs_dmrgci.buildCASHamiltonian(self.integral, core, cas)
        if self.splitloc:
            split_cas, split_rot, casinfo = proj_GRho(cas, self.GRho)
            casHam = bcs_dmrgci.rotateHam(split_rot, casHam)
            casHam, cas_local, rot = bcs_dmrgci.split_localize(split_cas, casinfo, \
                    casHam, basis = self.basis)
            rot = np.asarray(list(map(np.dot, split_rot, rot)))
        else:
            rot = np.asarray([np.eye(ncas), np.eye(ncas)])

        if self.mom_reorder:
            log.eassert(self.basis is not None, \
                    "maximum overlap method (MOM) requires embedding basis")
            cas_basis = basisToSpin(np.tensordot(basisToCanonical(self.basis), \
                    basisToCanonical(cas), (2,0)))
            order, q = bcs_dmrgci.momopt(self.localized_cas, cas_basis)
            log.check(q > 0.7, "MOM quality is not very good, "
                    "approximate CI solver may give bad results! (q = %.2f)", q)
            log.debug(1, "Orbital order: %s", order)
            casHam, cas_local, rot = bcs_dmrgci.reorder(order, \
                    casHam, cas_local, rot)
            # do not update self.localized cas

        self.rot = rot

        # solve dmrg for a few steps, 2 onedot with noise and 2 onedot without noise
        schedule = block.Schedule(maxiter = 4)
        M = self.fcisolver.maxM
        tol = self.fcisolver.schedule.sweeptol * 0.1
        schedule.gen_custom(arrayM = [self.fcisolver.maxM, self.fcisolver.maxM], \
                arraySweep = [0, 2], arrayTol = [tol, tol], arrayNoise = [tol, 0], \
                twodot_to_onedot = 0)

        # solve
        cas_local_GRho, E = self.fcisolver.run(casHam, schedule = schedule)
        casdm1, casdm2 = self.fcisolver.make_rdm12s()

        return casdm1, casdm2, None, None

    def dump_chk(*args):
        pass
