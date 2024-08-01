#! /usr/bin/env python

"""
HF / MP2 / OO-MP2 / OO-CCD solver.

Author:
    Zhi-Hao Cui <zhcui0408@gmail.com>
"""

import os
import numpy as np
import scipy.linalg as la

from libdmet.basis_transform import make_basis
from libdmet.solver import scf, mp, cc, umc1step, gmc1step
from libdmet.utils import logger as log

class SCFSolver(object):
    def __init__(self, nproc=1, nthread=1, nnode=1, TmpDir="./tmp", SharedDir=None,
                 restricted=False, Sz=0, bcs=False, ghf=False, tol=1e-10,
                 max_cycle=200,  max_memory=40000, scf_newton=True,
                 mp2=False, oomp2=False, ooccd=False,
                 mc_conv_tol=1e-4, mc_max_cycle=20,
                 ci_conv_tol=None, ci_diis_space=8,
                 tol_normt=1e-5, level_shift=0.0, frozen=0, restart=False,
                 approx_l=False, fix_fcivec=False, use_mpi=False, verbose=4,
                 alpha=None, beta=np.inf, linear=False, conv_tol_grad=None,
                 tol_nelec=None, fix_mu=False, **kwargs):
        """
        HF / MP2 / OO-MP2 / OO-CCD solver.
        """
        self.restricted = restricted
        self.max_cycle = max_cycle
        self.max_memory = max_memory
        self.conv_tol = tol
        self.conv_tol_grad = conv_tol_grad
        self.tol_nelec = tol_nelec
        if ci_conv_tol is None:
            self.ci_conv_tol = self.conv_tol*100
        else:
            self.ci_conv_tol = ci_conv_tol
        self.ci_diis_space = ci_diis_space
        self.mc_conv_tol = mc_conv_tol
        self.mc_max_cycle = mc_max_cycle
        self.conv_tol_normt = tol_normt
        self.level_shift = level_shift
        self.frozen = frozen

        self.verbose = 5
        self.bcs = bcs
        self.ghf = ghf
        self.Sz = Sz

        self.restart = restart
        self.C_lo_co = None # for restart

        self.alpha = alpha # scaled exact exchange
        self.beta = beta
        self.fix_mu = fix_mu
        self.mp2 = mp2 # MP2 solver
        self.oomp2 = oomp2 # orbital optimized MP2 solver
        self.ooccd = ooccd # orbital optimized CCD solver
        self.approx_l = approx_l
        self.linear = linear
        self.fix_fcivec = fix_fcivec

        self.scfsolver = scf.SCF(newton_ah=scf_newton)
        self.onepdm = None
        self.twopdm = None

        # MPI related args for OO-CCD
        self.nnode = nnode
        self.nproc = nproc
        self.nthread = nthread
        self.use_mpi = use_mpi
        self.verbose = verbose

    def run(self, Ham=None, nelec=None, restart=False, calc_rdm2=False,
            Mu=None, fit_mu=False, nelec_target=None, **kwargs):
        """
        Main function of the solver.
        NOTE: the spin order for H2 is aa, bb, ab.
        """
        log.info("HF solver Run")
        spin = Ham.H1["cd"].shape[0]

        if spin > 1:
            assert not self.restricted
        if nelec is None:
            if self.bcs:
                nelec = Ham.norb * 2
            elif self.ghf:
                nelec = Ham.norb // 2
            else:
                nelec = Ham.norb

        nelec_a = (nelec + self.Sz) // 2
        nelec_b = (nelec - self.Sz) // 2

        assert (nelec_a >= 0) and (nelec_b >=0) and (nelec_a + nelec_b == nelec)

        log.debug(1, "HF solver: mean-field")
        dm0 = kwargs.get("dm0", None)
        fname = kwargs.get("fname", "mcscf")
        fcc_name = kwargs.get("fcc_name", "fcc")
        restart = self.restart or restart
        basis = kwargs.get("basis", None)
        if basis is None:
            if restart:
                log.warn("restart requires basis passed in ...")
        else:
            basis = np.asarray(basis)
            if basis.ndim == 4:
                basis = basis.reshape(basis.shape[0], -1, basis.shape[-1])
            else:
                basis = basis.reshape(-1, basis.shape[-1])

        scf_max_cycle = kwargs.get("scf_max_cycle", 200)

        self.scfsolver.set_system(nelec, self.Sz, False, self.restricted,
                                  max_memory=self.max_memory)
        self.scfsolver.set_integral(Ham)

        if self.ghf:
            if scf_max_cycle < 0:
                # ZHC NOTE special case that no SCF is wanted
                assert dm0 is not None
                self.scfsolver.no_kernel = True
                E_HF, rhoHF = self.scfsolver.GGHF(tol=self.conv_tol,
                                                  MaxIter=scf_max_cycle,
                                                  InitGuess=dm0,
                                                  alpha=self.alpha,
                                                  beta=self.beta,
                                                  Mu=Mu,
                                                  fit_mu=fit_mu,
                                                  nelec_target=nelec_target,
                                                  basis=kwargs.get("basis", None),
                                                  mu_elec=kwargs.get("mu_elec", None),
                                                  conv_tol_grad=self.conv_tol_grad,
                                                  tol_nelec=self.tol_nelec,
                                                  fix_mu=self.fix_mu)
                # orthogonalize ao dm0
                s = self.scfsolver.mf.get_ovlp()
                e, v = la.eigh(s)
                idx = e > 1e-15
                s_half =  np.dot(v[:, idx] * np.sqrt(e[idx]), v[:,idx].conj().T)
                s_half_inv =  np.dot(v[:, idx] / np.sqrt(e[idx]), v[:,idx].conj().T)
                dm0_lo = np.dot(s_half, np.dot(dm0, s_half))

                natocc, natorb = la.eigh(dm0_lo)
                natorb = np.dot(s_half_inv, natorb)
                natocc = natocc[::-1]
                natorb = natorb[:, ::-1]
                mo_occ = np.zeros_like(natocc)
                mo_occ[:nelec] = 1.0

                self.scfsolver.mf.mo_coeff = natorb
                self.scfsolver.mf.mo_occ = mo_occ
            else:
                E_HF, rhoHF = self.scfsolver.GGHF(tol=self.conv_tol,
                                                  MaxIter=scf_max_cycle,
                                                  InitGuess=dm0,
                                                  alpha=self.alpha,
                                                  beta=self.beta,
                                                  Mu=Mu,
                                                  fit_mu=fit_mu,
                                                  nelec_target=nelec_target,
                                                  basis=kwargs.get("basis", None),
                                                  mu_elec=kwargs.get("mu_elec", None),
                                                  conv_tol_grad=self.conv_tol_grad,
                                                  tol_nelec=self.tol_nelec,
                                                  fix_mu=self.fix_mu)
        else:
            E_HF, rhoHF = self.scfsolver.HF(tol=self.conv_tol,
                                            MaxIter=scf_max_cycle,
                                            InitGuess=dm0,
                                            alpha=self.alpha,
                                            beta=self.beta)
        log.debug(1, "HF solver: mean-field converged: %s",
                  self.scfsolver.mf.converged)

        if "mo_energy_custom" in kwargs:
            self.scfsolver.mf.mo_energy = kwargs["mo_energy_custom"]
        if "mo_occ_custom" in kwargs:
            self.scfsolver.mf.mo_occ = kwargs["mo_occ_custom"]
        if "mo_coeff_custom" in kwargs:
            log.info("Use customized MO as reference.")
            self.scfsolver.mf.mo_coeff = kwargs["mo_coeff_custom"]
            self.scfsolver.mf.e_tot = self.scfsolver.mf.energy_tot()
            rhoHF = np.asarray(self.scfsolver.mf.make_rdm1())
            if (not self.ghf) and self.restricted:
                rhoHF *= 0.5

        if self.oomp2 or self.ooccd:
            if self.oomp2:
                fcisolver = mp.MP2AsFCISolver
            else:
                if self.use_mpi:
                    from libdmet.solver import mpicc
                    fcisolver = mpicc.MPIGCCDAsFCISolver
                else:
                    fcisolver = cc.CCDAsFCISolver
            if self.ghf:
                mf = self.scfsolver.mf
                norb = mf.mo_coeff.shape[-1]
                mc = gmc1step.GCASSCFBigCAS(mf, norb//2, nelec)
                #mc = gmc1step.GCASSCFVO(mf, norb//2, nelec)
                # ZHC NOTE only the vo rotation is needed
                #mc.internal_rotation = True
                mc.internal_rotation_vo = True
                mc.conv_tol = self.mc_conv_tol
                mc.max_cycle_macro = self.mc_max_cycle

                mc.fcisolver = fcisolver(ghf=True,
                                         level_shift=self.level_shift,
                                         max_memory=self.max_memory,
                                         max_cycle=self.max_cycle,
                                         conv_tol=self.ci_conv_tol,
                                         conv_tol_normt=self.conv_tol_normt,
                                         diis_space=self.ci_diis_space,
                                         restart=restart,
                                         approx_l=self.approx_l,
                                         fix_fcivec=self.fix_fcivec,
                                         nproc=self.nproc, nthread=self.nthread,
                                         nnode=self.nnode, verbose=self.verbose,
                                         fcc_name=fcc_name, linear=self.linear)
                if restart:
                    if self.C_lo_co is not None:
                        if self.C_lo_co.shape[-1] != norb:
                            log.warn("CAS orbital number changed %s -> %s",
                                     self.C_lo_co.shape[-1], norb)
                            mo_coeff = None
                        else:
                            log.info("restart from the previous calculations...")
                            # estimate new CAS orbitals:
                            mo_coeff = make_basis.find_closest_mo(basis, self.C_lo_co,
                                                                  return_rotmat=True)[1]
                            # since MO is already closest, directly use old t2:
                            mc.fcisolver.fcivec = mc.fcisolver.load_fcivec("%s_fcivec.h5"%fname)
                            mc.fcisolver.optimized = True

                        # u should be removed at the beginning
                        if os.path.exists(fname+"_u.npy"):
                            os.remove(fname+"_u.npy")
                    else:
                        mo_coeff = None
                    # save u for restart within the CASSCF
                    def save_u(envs):
                        u = envs['u']
                        ncore = mc.ncore * 2
                        ncas = mc.ncas * 2
                        nocc = ncore + ncas
                        u_cas = u[ncore:nocc, ncore:nocc]
                        np.save(fname+"_u.npy", u_cas)
                    mc.callback = save_u
                else:
                    mo_coeff = None

                E = mc.mc2step(mo_coeff)[0]
                self.onepdm = mc.make_rdm1()

                if restart and (basis is not None):
                    # save C_lo_co for restart
                    self.C_lo_co = np.dot(basis, mc.mo_coeff)
                    np.save("C_eo_co.npy", mc.mo_coeff)
                    # save fcivec for restart
                    mc.fcisolver.save_fcivec("%s_fcivec.h5"%fname)
            else:
                if self.restricted:
                    raise NotImplementedError

                mf = self.scfsolver.mf
                norb = mf.mo_coeff.shape[-1]
                mc = umc1step.UCASSCFBigCAS(mf, norb, (nelec_a, nelec_b))
                mc.internal_rotation_vo = True
                mc.eri_ab_mid = False # aa, bb, ab order for eri
                mc.conv_tol = self.mc_conv_tol
                mc.max_cycle_macro = self.mc_max_cycle

                mc.fcisolver = fcisolver(restricted=self.restricted,
                                         ghf=False,
                                         level_shift=self.level_shift,
                                         max_memory=self.max_memory,
                                         max_cycle=self.max_cycle,
                                         conv_tol=self.ci_conv_tol,
                                         conv_tol_normt=self.conv_tol_normt,
                                         diis_space=self.ci_diis_space,
                                         restart=restart,
                                         approx_l=self.approx_l,
                                         fix_fcivec=self.fix_fcivec,
                                         nproc=self.nproc, nthread=self.nthread,
                                         nnode=self.nnode, verbose=self.verbose,
                                         fcc_name=fcc_name, linear=self.linear)
                if restart:
                    if self.C_lo_co is not None:
                        if self.C_lo_co.shape[-1] != norb:
                            log.warn("CAS orbital number changed %s -> %s",
                                     self.C_lo_co.shape[-1], norb)
                            mo_coeff = None
                        else:
                            log.info("restart from the previous calculations...")
                            # estimate new CAS orbitals:
                            mo_coeff_a = make_basis.find_closest_mo(basis[0], self.C_lo_co[0],
                                                                    return_rotmat=True)[1]
                            mo_coeff_b = make_basis.find_closest_mo(basis[1], self.C_lo_co[1],
                                                                    return_rotmat=True)[1]
                            mo_coeff = np.array((mo_coeff_a, mo_coeff_b))
                            # since MO is already closest, directly use old t2:
                            mc.fcisolver.fcivec = mc.fcisolver.load_fcivec("%s_fcivec.h5"%fname)
                            mc.fcisolver.optimized = True

                        # u should be removed at the beginning
                        if os.path.exists(fname+"_u_a.npy"):
                            os.remove(fname+"_u_a.npy")
                        if os.path.exists(fname+"_u_b.npy"):
                            os.remove(fname+"_u_b.npy")
                    else:
                        mo_coeff = None
                    # save u for restart within the CASSCF
                    def save_u(envs):
                        u = envs['u']
                        ncore_a, ncore_b = mc.ncore
                        ncas = mc.ncas
                        nocc_a = ncore_a + ncas
                        nocc_b = ncore_b + ncas
                        u_cas_a = u[0][ncore_a:nocc_a, ncore_a:nocc_a]
                        u_cas_b = u[1][ncore_b:nocc_b, ncore_b:nocc_b]
                        np.save(fname+"_u_a.npy", u_cas_a)
                        np.save(fname+"_u_b.npy", u_cas_b)
                    mc.callback = save_u
                else:
                    mo_coeff = None

                E = mc.mc2step(mo_coeff)[0]
                self.onepdm = np.asarray(mc.make_rdm1s())

                if restart and (basis is not None):
                    # save C_lo_co for restart
                    self.C_lo_co = np.einsum('spm, smn -> spn', basis, mc.mo_coeff)
                    np.save("C_eo_co.npy", mc.mo_coeff)
                    # save fcivec for restart
                    mc.fcisolver.save_fcivec("%s_fcivec.h5"%fname)
        elif self.mp2:
            if self.ghf:
                E, self.onepdm = self.scfsolver.GMP2()
            else:
                E, self.onepdm = self.scfsolver.MP2()
        else: # HF
            self.onepdm = rhoHF
            E = E_HF
        return self.onepdm, E

    def run_dmet_ham(self, Ham, last_aabb=True, save_dmet_ham=False,
                     dmet_ham_fname='dmet_ham.h5', use_calculated_twopdm=False,
                     ao_repr=False, **kwargs):
        """
        Run scaled DMET Hamiltonian.
        NOTE: the spin order for H2 is aa, bb, ab, the same as ImpHam.
        """
        log.info("mf solver Run DMET Hamiltonian.")
        if not ao_repr:
            log.info("Use MO representation.")
            Ham = scf.ao2mo_Ham(Ham, self.scfsolver.mf.mo_coeff, compact=True, in_place=True)
        Ham = scf.restore_Ham(Ham, 1, in_place=True)

        # calculate rdm2 in aa, bb, ab order
        if use_calculated_twopdm:
            log.info("Using exisiting twopdm in MO basis...")
        else:
            self.make_rdm2(ao_repr=ao_repr)

        if ao_repr:
            r1 = self.onepdm
            r2 = self.twopdm
        else:
            r1 = self.onepdm_mo
            r2 = self.twopdm_mo

        if self.ghf:
            h1 = Ham.H1["cd"][0]
            h2 = Ham.H2["ccdd"][0]
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape

            # energy
            E1 = np.einsum('pq, qp', h1, r1)
            E2 = np.einsum('pqrs, pqrs', h2, r2) * 0.5
            E = E1 + E2
        elif Ham.restricted:
            h1 = Ham.H1["cd"]
            h2 = Ham.H2["ccdd"]
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape

            # energy
            E1 = np.einsum('pq, qp', h1[0], r1[0]) * 2.0
            E2 = np.einsum('pqrs, pqrs', h2[0], r2[0]) * 0.5
            E = E1 + E2
        else:
            h1 = Ham.H1["cd"]
            # h2 is in aa, bb, ab order
            h2 = Ham.H2["ccdd"]
            # r2 is in aa, bb, ab order
            assert h1.shape == r1.shape
            assert h2.shape == r2.shape

            # energy
            E1 = np.einsum('spq, sqp', h1, r1)
            E2_aa = 0.5 * np.einsum('pqrs, pqrs', r2[0], h2[0])
            E2_bb = 0.5 * np.einsum('pqrs, pqrs', r2[1], h2[1])
            E2_ab = np.einsum('pqrs, pqrs', r2[2], h2[2])
            E = E1 + E2_aa + E2_bb + E2_ab
        E += Ham.H0

        if save_dmet_ham:
            fdmet_ham = h5py.File(dmet_ham_fname, 'w')
            fdmet_ham['H1'] = h1
            fdmet_ham['H2'] = h2
            fdmet_ham.close()

        return E

    def make_rdm1(self):
        return self.onepdm

    def make_rdm2(self, ao_repr=True):
        """
        Compute rdm2.
        NOTE: the returned value's spin order for H2 is aa, bb, ab.
        """
        if self.ooccd:
            raise NotImplementedError
        elif self.oomp2:
            raise NotImplementedError
        elif self.mp2:
            raise NotImplementedError
        else:
            if ao_repr:
                if self.ghf:
                    # GHF rdm2
                    self.twopdm = \
                            np.einsum('qp, sr -> pqrs', self.onepdm, self.onepdm) -\
                            np.einsum('sp, qr -> pqrs', self.onepdm, self.onepdm)
                elif self.restricted:
                    # RHF spin-traced rdm2
                    self.twopdm = \
                            (4.0 * np.einsum('qp, sr -> pqrs', self.onepdm[0],
                                             self.onepdm[0]) - \
                             2.0 * np.einsum('sp, qr -> pqrs', self.onepdm[0],
                                             self.onepdm[0]))[None]
                else:
                    # UHF rdm2, in aa, bb, ab order
                    rdm2_aa = np.einsum('qp, sr -> pqrs', self.onepdm[0],
                                        self.onepdm[0]) - \
                              np.einsum('sp, qr -> pqrs', self.onepdm[0],
                                        self.onepdm[0])
                    rdm2_bb = np.einsum('qp, sr -> pqrs', self.onepdm[1],
                                        self.onepdm[1]) - \
                              np.einsum('sp, qr -> pqrs', self.onepdm[1],
                                        self.onepdm[1])
                    rdm2_ab = np.einsum('qp, sr -> pqrs', self.onepdm[0],
                                        self.onepdm[1])
                    self.twopdm = np.asarray((rdm2_aa, rdm2_bb, rdm2_ab))
                return self.twopdm
            else:
                if self.ghf:
                    # GHF rdm2
                    onepdm = np.diag(self.scfsolver.mf.mo_occ)
                    self.onepdm_mo = onepdm
                    self.twopdm_mo = np.einsum('qp, sr -> pqrs', onepdm, onepdm) -\
                                     np.einsum('sp, qr -> pqrs', onepdm, onepdm)
                elif self.restricted:
                    # RHF spin-traced rdm2
                    onepdm = np.diag(self.scfsolver.mf.mo_occ)
                    self.onepdm_mo = (onepdm * 0.5)[None]
                    self.twopdm_mo = (np.einsum('qp, sr -> pqrs', onepdm, onepdm) -\
                                      0.5 * np.einsum('sp, qr -> pqrs', onepdm,
                                                      onepdm))[None]
                else:
                    # UHF rdm2, in aa, bb, ab order
                    onepdm_a = np.diag(self.scfsolver.mf.mo_occ[0])
                    onepdm_b = np.diag(self.scfsolver.mf.mo_occ[1])
                    self.onepdm_mo = np.asarray((onepdm_a, onepdm_b))
                    rdm2_aa = np.einsum('qp, sr -> pqrs', onepdm_a, onepdm_a) - \
                              np.einsum('sp, qr -> pqrs', onepdm_a, onepdm_a)
                    rdm2_bb = np.einsum('qp, sr -> pqrs', onepdm_b, onepdm_b) - \
                              np.einsum('sp, qr -> pqrs', onepdm_b, onepdm_b)
                    rdm2_ab = np.einsum('qp, sr -> pqrs', onepdm_a, onepdm_b)
                    self.twopdm_mo = np.asarray((rdm2_aa, rdm2_bb, rdm2_ab))
                return self.twopdm_mo

    def onepdm(self):
        log.debug(1, "Compute 1pdm")
        return self.onepdm

    def twopdm(self):
        log.debug(1, "Compute 2pdm")
        return self.twopdm
