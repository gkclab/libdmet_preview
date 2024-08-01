import numpy as np
import scipy.linalg as la
import pyscf
from pyscf.mcscf import mc1step_uhf, casci_uhf
from libdmet.utils.misc import mdot
from libdmet.utils import logger as log
from libdmet.solver import dmrgci, block
from libdmet.solver.scf import SCF, incore_transform, pyscflogger
from libdmet.system import integral
import copy

class CASSCF(mc1step_uhf.CASSCF):
    def __init__(self, mf, ncas, nelecas, ncore = None, frozen = []):
        mc1step_uhf.CASSCF.__init__(self, mf, ncas, nelecas, ncore, frozen)
        casci_uhf.CASCI.get_hcore = lambda *args: mf.h1e
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

    def refresh(self, mf, ncas, nelecas, ncore = None, frozen = []):
        self.__init__(mf, ncas, nelecas, ncore, frozen)
        self.converged = False

    def ao2mo(self, mo):
        nmo = mo[0].shape[1]
        ncore = self.ncore
        ncas = self.ncas
        nocc = (ncas + ncore[0], ncas + ncore[1])
        eriaa, eribb, eriab = incore_transform(self._scf._eri, (mo, mo, mo, mo))
        eris = lambda:None
        eris.jkcpp = np.einsum('iipq->ipq', eriaa[:ncore[0],:ncore[0],:,:]) \
                   - np.einsum('ipqi->ipq', eriaa[:ncore[0],:,:,:ncore[0]])
        eris.jkcPP = np.einsum('iipq->ipq', eribb[:ncore[1],:ncore[1],:,:]) \
                   - np.einsum('ipqi->ipq', eribb[:ncore[1],:,:,:ncore[1]])
        eris.jC_pp = np.einsum('pqii->pq', eriab[:,:,:ncore[1],:ncore[1]])
        eris.jc_PP = np.einsum('iipq->pq', eriab[:ncore[0],:ncore[0],:,:])
        eris.aapp = np.copy(eriaa[ncore[0]:nocc[0],ncore[0]:nocc[0],:,:])
        eris.aaPP = np.copy(eriab[ncore[0]:nocc[0],ncore[0]:nocc[0],:,:])
        eris.AApp = np.copy(eriab[:,:,ncore[1]:nocc[1],ncore[1]:nocc[1]].transpose(2,3,0,1))
        eris.AAPP = np.copy(eribb[ncore[1]:nocc[1],ncore[1]:nocc[1],:,:])
        eris.appa = np.copy(eriaa[ncore[0]:nocc[0],:,:,ncore[0]:nocc[0]])
        eris.apPA = np.copy(eriab[ncore[0]:nocc[0],:,:,ncore[1]:nocc[1]])
        eris.APPA = np.copy(eribb[ncore[1]:nocc[1],:,:,ncore[1]:nocc[1]])

        eris.cvCV = np.copy(eriab[:ncore[0],ncore[0]:,:ncore[1],ncore[1]:])
        eris.Icvcv = eriaa[:ncore[0],ncore[0]:,:ncore[0],ncore[0]:] * 2\
                   - eriaa[:ncore[0],:ncore[0],ncore[0]:,ncore[0]:].transpose(0,3,1,2) \
                   - eriaa[:ncore[0],ncore[0]:,:ncore[0],ncore[0]:].transpose(0,3,2,1)
        eris.ICVCV = eribb[:ncore[1],ncore[1]:,:ncore[1],ncore[1]:] * 2\
                   - eribb[:ncore[1],:ncore[1],ncore[1]:,ncore[1]:].transpose(0,3,1,2) \
                   - eribb[:ncore[1],ncore[1]:,:ncore[1],ncore[1]:].transpose(0,3,2,1)

        eris.Iapcv = eriaa[ncore[0]:nocc[0],:,:ncore[0],ncore[0]:] * 2 \
                   - eriaa[:,ncore[0]:,:ncore[0],ncore[0]:nocc[0]].transpose(3,0,2,1) \
                   - eriaa[:,:ncore[0],ncore[0]:,ncore[0]:nocc[0]].transpose(3,0,1,2)
        eris.IAPCV = eribb[ncore[1]:nocc[1],:,:ncore[1],ncore[1]:] * 2 \
                   - eribb[:,ncore[1]:,:ncore[1],ncore[1]:nocc[1]].transpose(3,0,2,1) \
                   - eribb[:,:ncore[1],ncore[1]:,ncore[1]:nocc[1]].transpose(3,0,1,2)
        eris.apCV = np.copy(eriab[ncore[0]:nocc[0],:,:ncore[1],ncore[1]:])
        eris.APcv = np.copy(eriab[:ncore[0],ncore[0]:,ncore[1]:nocc[1],:].transpose(2,3,0,1))
        eris.vhf_c = (np.einsum('ipq->pq', eris.jkcpp) + eris.jC_pp,
                    np.einsum('ipq->pq', eris.jkcPP) + eris.jc_PP)

        return eris

    def approx_cas_integral(self, mo, u, eris):
        if self.exact_integral:
            # new mo
            mo1 = list(map(np.dot, mo, u))
            # h1e
            h1cas, ecore = self.h1e_for_cas(mo1)
            _, ecore0 = self.h1e_for_cas(mo)
            ecore -= ecore0
            # h2e
            _, cas, _ = casci_uhf.extract_orbs(mo1, self.ncas, \
                    self.nelecas, self.ncore)
            eriaa, eribb, eriab = incore_transform(self._scf._eri, \
                    (cas, cas, cas, cas))
            return ecore, h1cas, (eriaa, eriab, eribb)
        else:
            return mc1step_uhf.CASSCF.approx_cas_integral(self, mo, u, eris)

    def gen_g_hop(self, mo, u, casdm1, casdm2, eris):
        if not self.exact_integral or (isinstance(u, int) and u == 1):
            return mc1step_uhf.CASSCF.gen_g_hop(self, mo, u, casdm1, casdm2, eris)
        else:
            mo1 = list(map(np.dot, mo, u))
            eris1 = self.ao2mo(mo1)
            gorb, _, h_op, h_diag = mc1step_uhf.CASSCF.gen_g_hop(self, \
                    mo1, 1, casdm1, casdm2, eris1)
            return gorb, lambda *args: gorb, h_op, h_diag

def proj_rho(orbs, rho):
    spin = rho.shape[0]
    proj_orbs = [None, None]
    proj_orbs_rot = [None, None]
    info = [None, None]

    for s in range(spin):
        proj_rho = mdot(orbs[s].T, rho[s], orbs[s])
        pocc, porb = la.eigh(proj_rho)
        _nvirt = np.sum(pocc < 0.3)
        _ncore = np.sum(pocc > 0.7)
        _npart = np.sum((pocc >= 0.3) * (pocc <= 0.7))
        proj_orbs_rot[s] = porb[:, ::-1]
        proj_orbs[s] = np.dot(orbs[s], porb[:, ::-1])
        info[s] = (_ncore, _npart, _nvirt)
    proj_orbs = np.asarray(proj_orbs)
    proj_orbs_rot = np.asarray(proj_orbs_rot)
    return proj_orbs, proj_orbs_rot, info

def make_rdm1s(solver, casscf):
    rdm1s = solver.onepdm()
    rot = casscf.rot
    return np.asarray([mdot(rot[0], rdm1s[0], rot[0].T), \
            mdot(rot[1], rdm1s[1], rot[1].T)])

def make_rdm12s(solver, casscf):
    rdm1s = solver.onepdm()
    rdm2s = solver.twopdm()
    rot = casscf.rot
    rdm1s_cas = np.asarray([mdot(rot[0], rdm1s[0], rot[0].T), \
            mdot(rot[1], rdm1s[1], rot[1].T)])
    rotInv = np.asarray([rot[0].T, rot[1].T])
    rdm2s_cas = incore_transform(rdm2s, (rotInv,) * 4)
    rdm2s_cas = rdm2s_cas[[0,2,1]]
    return rdm1s_cas, rdm2s_cas

class DMRGSCF(CASSCF):
    def __init__(self, mf, ncas, nelecas, fcisolver, ncore = None, \
            frozen = [], splitloc = True, mom_reorder = True, TmpDir = "/tmp"):
        CASSCF.__init__(self, mf, ncas, nelecas, ncore, frozen)
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
        hf_coefs = mf.mo_coeff
        hf_occs = mf.mo_occ
        self.rho = np.asarray([
            mdot(hf_coefs[0], np.diag(hf_occs[0]), hf_coefs[0].T),
            mdot(hf_coefs[1], np.diag(hf_occs[1]), hf_coefs[1].T)
        ])
        self.fcisolver.make_rdm1s = lambda *args: \
                make_rdm1s(self.fcisolver, self)
        self.fcisolver.make_rdm12s = lambda *args: \
                make_rdm12s(self.fcisolver, self)

    def refresh(self, mf, ncas, nelecas, ncore = None, frozen = []):
        fcisolver = copy.copy(self.fcisolver)
        CASSCF.__init__(self, mf, ncas, nelecas, ncore, frozen)
        self.fcisolver = fcisolver
        self.rot = None
        self.basis = None
        hf_coefs = mf.mo_coeff
        hf_occs = mf.mo_occ
        self.rho = np.asarray([
            mdot(hf_coefs[0], np.diag(hf_occs[0]), hf_coefs[0].T),
            mdot(hf_coefs[1], np.diag(hf_occs[1]), hf_coefs[1].T)
        ])
        self.converged = False

    def mc1step(self, *args, **kwargs):
        if 'basis' in kwargs.keys():
            self.basis = kwargs["basis"]
            kwargs.pop('basis')
        return CASSCF.mc1step(self, *args, **kwargs)

    def mc2step(self, *args, **kwargs):
        if 'basis' in kwargs.keys():
            self.basis = kwargs["basis"]
            kwargs.pop('basis')
        return CASSCF.mc2step(self, *args, **kwargs)

    def casci(self, mo_coeff, ci0 = None, eris = None):
        if eris is None:
            fcasci = self
        else:
            fcasci = mc1step_uhf._fake_h_for_fast_casci(self, mo_coeff, eris)
            vhf = fcasci.get_veff()
            fcasci.get_veff = lambda *args: np.asarray(vhf)
            eri_cas = fcasci.ao2mo()
            fcasci.ao2mo = lambda *args: (eri_cas[0], eri_cas[2], eri_cas[1])

        # adapted from UHF-CASCI kernel
        ncas = self.ncas
        nelecas = self.nelecas
        ncore = self.ncore

        core, cas, virt = casci_uhf.extract_orbs(mo_coeff, ncas, nelecas, ncore)

        # Hamiltonian
        H1, H0 = fcasci.h1e_for_cas(mo_coeff)
        H2 = fcasci.ao2mo(cas)

        if self.splitloc:
            # rotate cas orbitals
            split_cas, split_rot, casinfo = proj_rho(cas, self.rho)
            # Transform into split_cas basis
            casHam = integral.Integral(ncas, False, False, 0., \
                    {"cd": np.asarray(
                            [mdot(split_rot[0].T, H1[0], split_rot[0]),
                            mdot(split_rot[1].T, H1[1], split_rot[1])]
                    )}, \
                    {"ccdd": incore_transform(H2, (split_rot,)*4)})
            casHam, cas_local, rot = dmrgci.split_localize(split_cas, casinfo, \
                    casHam, basis = self.basis)
            rot = np.asarray(list(map(np.dot, split_rot, rot)))
        else:
            casHam = integral.Integral(ncas, False, False, 0., \
                    {"cd": np.asarray(H1)}, {"ccdd": np.asarray(H2)})
            rot = np.asarray([np.eye(ncas), np.eye(ncas)])

        if self.mom_reorder:
            log.eassert(self.basis is not None, \
                    "maximum overlap method (MOM) requires embedding basis")
            if self.localized_cas is None:
                order = dmrgci.gaopt(casHam, tmp = self.tmpDir)
            else:
                cas_basis = np.asarray([
                    np.tensordot(self.basis[0], cas_local[0], (2,0)),
                    np.tensordot(self.basis[1], cas_local[1], (2,0))
                ])
                order, q = dmrgci.momopt(self.localized_cas, cas_basis)
                if q < 0.7:
                    order = dmrgci.gaopt(casHam, tmp = self.tmpDir)
            log.info("Orbital order: %s", order)
            casHam, cas_local, rot = dmrgci.reorder(order, casHam, cas_local, rot)
            self.localized_cas = np.asarray([
                np.tensordot(self.basis[0], cas_local[0], (2,0)),
                np.tensordot(self.basis[1], cas_local[1], (2,0))
            ])

        self.rot = rot

        # solve
        cas_local_1pdm, E = self.fcisolver.run(casHam, nelec = np.sum(self.nelecas))

        # update self.rho
        self.rho = np.asarray(self.make_rdm1s())

        return E+H0, E, None

    def update_casdm(self, mo, u, fcivec, e_ci, eris):
        ecore, h1cas, h2cas = self.approx_cas_integral(mo, u, eris)
        h2cas = (h2cas[0], h2cas[2], h2cas[1])
        self.solve_approx_ci(h1cas, h2cas, mo, u)
        casdm1, casdm2 = self.fcisolver.make_rdm12s()

        return casdm1, casdm2, None, None

    def solve_approx_ci(self, h1, h2, mo, u):
        # solve CI problem approximately
        # by doing only a few sweeps
        log.debug(0, "Solve CI problem approximately with BLOCK")

        ncas = self.ncas
        nelecas = self.nelecas
        ncore = self.ncore

        # get rotated mo_coef
        mo1 = np.asarray([np.dot(mo[0], u[0]), np.dot(mo[1], u[1])])
        core, cas, virt = casci_uhf.extract_orbs(mo1, ncas, nelecas, ncore)

        # mo rotated to occupied and virtual subsets, then split localize
        if self.splitloc:
            # rotate cas orbitals
            split_cas, split_rot, casinfo = proj_rho(cas, self.rho)
            casHam = integral.Integral(ncas, False, False, 0., \
                    {"cd": np.asarray([
                        mdot(split_rot[0].T, h1[0], split_rot[0]),
                        mdot(split_rot[1].T, h1[1], split_rot[1])
                    ])}, \
                    {"ccdd": incore_transform(h2, (split_rot,)*4)})
            casHam, cas_local, rot = dmrgci.split_localize(split_cas, casinfo, \
                    casHam, basis = self.basis)
            rot[0] = np.dot(split_rot[0], rot[0])
            rot[1] = np.dot(split_rot[1], rot[1])
        else:
            casHam = integral.Integral(ncas, False, False, 0., \
                    {"cd": np.asarray(h1)}, {"ccdd": np.asarray(h2)})
            rot = np.asarray([np.eye(ncas), np.eye(ncas)])

        # mo reorder to match exact CI step
        if self.mom_reorder:
            log.eassert(self.basis is not None, \
                    "maximum overlap method (MOM) requires embedding basis")
            cas_basis = np.asarray([
                np.tensordot(self.basis[0], cas_local[0], (2,0)),
                np.tensordot(self.basis[1], cas_local[1], (2,0))
            ])
            order, q = dmrgci.momopt(self.localized_cas, cas_basis)
            log.check(q > 0.7, "MOM quality is not very good, "
                    "approximate CI solver may give bad results! (q = %.2f)", q)
            log.debug(1, "Orbital order: %s", order)
            casHam, cas_local, rot = dmrgci.reorder(order, casHam, cas_local, rot)
            # do not update self.localized cas

        self.rot = rot

        # solve dmrg for a few steps, 2 onedot with noise and 2 onedot without noise
        schedule = block.Schedule(maxiter = 4)
        M = self.fcisolver.maxM
        tol = self.fcisolver.schedule.sweeptol * 0.1
        schedule.gen_custom(arrayM = [self.fcisolver.maxM, self.fcisolver.maxM], \
                arraySweep = [0, 2], arrayTol = [tol, tol], arrayNoise = [tol, 0], \
                twodot_to_onedot = 0)

        cas_local_1pdm, E = self.fcisolver.run(casHam, \
                nelec = np.sum(self.nelecas), schedule = schedule)

        return None, None

if __name__ == "__main__":
    log.verbose = "INFO"
    Int1e = -np.eye(12, k = 1)
    Int1e[0, 11] = -1
    Int1e += Int1e.T
    Int1e = np.asarray([Int1e, Int1e])
    Int2e = np.zeros((3,12,12,12,12))

    for i in range(12):
        Int2e[0,i,i,i,i] = 1
        Int2e[1,i,i,i,i] = 1
        Int2e[2,i,i,i,i] = 1

    scf = SCF()
    scf.set_system(12, 0, False, False)
    scf.set_integral(12, 0, {"cd": Int1e}, {"ccdd": Int2e})
    ehf, rhoHF = scf.HF(MaxIter = 100, tol = 1e-3, \
        InitGuess = (np.diag([0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0]), \
        np.diag([0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5])))
    mc = CASSCF(scf.mf, 8, (4,4))
    emc = mc.mc1step()[0]
    print (mc.make_rdm1s())
    print (ehf, emc, emc-ehf)
