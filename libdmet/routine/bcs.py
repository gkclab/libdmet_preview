#! /usr/bin/env python

"""
BCS formalism of DMET.
"""

import numpy as np
import scipy.linalg as la
import itertools as it
from copy import deepcopy
from math import sqrt

from libdmet.routine.bcs_helper import *
from libdmet.routine.slater import MatSqrt, orthonormalizeBasis
from libdmet.routine.mfd import assignocc, HFB
from libdmet.routine.fit import minimize
from libdmet.utils import logger as log
from libdmet.system import integral
from libdmet.utils.misc import mdot, find, max_abs
from libdmet import settings
from libdmet.routine import localizer
from libdmet.routine import ftsystem
from libdmet.routine import slater_helper

def embBasis(lattice, GRho, local=True, **kwargs):
    if local:
        return __embBasis_proj(lattice, GRho, **kwargs)
    else:
        return __embBasis_phsymm(lattice, GRho, **kwargs)

get_emb_basis = embBasis

def __embBasis_proj(lattice, GRho, **kwargs):
    ncells = lattice.ncells
    nscsites = lattice.nscsites
    nval = lattice.nval
    if "sites" in kwargs:
        Imps = kwargs["sites"]
        ImpIdx = Imps + [i+nscsites for i in Imps]
        EnvIdx = [i for i in range(2*nscsites*ncells) if i not in ImpIdx]
        nImp = len(Imps)
        basis = np.zeros((2, ncells*nscsites*2, nImp*2))
        GRhoEnvImp = np.delete(GRho.reshape(nscsites*ncells*2, nscsites*2)[:, ImpIdx], ImpIdx, 0)
        B, sigma, vt = la.svd(GRhoImpEnv, full_matrices=False)
        localize_bath = kwargs.get("localize_bath", None)
        if localize_bath is not None:
            log.eassert(lattice.is_model == True, \
                    "Only model is currently supported for localization of bath.")
            B = localizer.localize_bath(B, method=localize_bath)
        log.debug(1, "bath orbitals\n%s", B)

        basis[np.ix_([0], Imps, range(nImp))] = np.eye(nImp)
        basis[np.ix_([1], Imps, range(nImp))] = np.eye(nImp)
        BathIdxV = list(range(nscsites-nImp))
        BathIdxU = list(range(nscsites-nImp, 2*(nscsites-nImp)))
        for i in range(ncells-1):
            BathIdxV += list(range(2*(nscsites-nImp)+nscsites*i*2, \
                    2*(nscsites-nImp)+nscsites*(2*i+1)))
            BathIdxU += list(range(2*(nscsites-nImp)+nscsites*(2*i+1), \
                    2*(nscsites-nImp)+nscsites*(2*i+2)))
        EnvIdxV = [EnvIdx[i] for i in BathIdxV]
        EnvIdxU = [EnvIdx[i] for i in BathIdxU]
        w = np.diag(np.dot(B[BathIdxV].T, B[BathIdxV]))
        order = np.argsort(w, kind='mergesort')[::-1]
        w1 = w[order]
        orderA, orderB = order[:nImp], order[nImp:]
        wA, wB = w1[:nImp], 1.0-w1[nImp:]
        log.debug(0, "particle character:\nspin A max %.2f min %.2f mean %.2f"
                "\nspin B max %.2f min %.2f mean %.2f", np.max(wA), np.min(wA), \
                np.average(wA), np.max(wB), np.min(wB), np.average(wB))
        basis[np.ix_([0], EnvIdx, range(nImp, nImp*2))] = B[:, orderA]
        basis[np.ix_([1], EnvIdxV, range(nImp, nImp*2))] = B[np.ix_(BathIdxU, orderB)]
        basis[np.ix_([1], EnvIdxU, range(nImp, nImp*2))] = B[np.ix_(BathIdxV, orderB)]
        basis = basis.reshape((2, ncells, nscsites*2, nImp*2))
    else:
        basis = np.zeros((2, ncells, nscsites*2, nscsites+nval))
        val_idx = lattice.val_idx + [i+nscsites for i in lattice.val_idx]
        GRhoEnvImp = GRho[1:].reshape(nscsites*(ncells-1)*2, nscsites*2)[:, val_idx]
        u, sigma, vt = la.svd(GRhoEnvImp, full_matrices=False)
        log.debug(0, "Zero singular values number: %s", np.sum(np.abs(sigma) < 1e-8))
        log.debug(1, "Singular values:\n%s", sigma)
         
        B = u.reshape((ncells-1, nscsites*2, nval*2))
        localize_bath = kwargs.get("localize_bath", None)
        if localize_bath is not None:
            log.eassert(lattice.is_model == True, \
                    "Only model is currently supported for localization of bath.")
            B = localizer.localize_bath(B, method=localize_bath)
        
        basis[0, 0, :nscsites, :nscsites] = np.eye(nscsites)
        basis[1, 0, :nscsites, :nscsites] = np.eye(nscsites)
        w = np.diag(np.tensordot(B[:,:nscsites], B[:,:nscsites], axes = ((0,1),(0,1))))
        order = np.argsort(w, kind='mergesort')[::-1]
        w1 = w[order]
        orderA, orderB = order[:nval], order[nval:]
        wA, wB = w1[:nval], 1.0-w1[nval:]
        log.debug(0, "particle character:\nspin A max %.2f min %.2f mean %.2f"
                "\nspin B max %.2f min %.2f mean %.2f", np.max(wA), np.min(wA), \
                np.average(wA), np.max(wB), np.min(wB), np.average(wB))
        basis[0, 1:, :, nscsites:] = B[:,:,orderA]
        basis[1, 1:, :nscsites, nscsites:], basis[1, 1:, nscsites:, nscsites:] = \
                B[:, nscsites:, orderB], B[:, :nscsites, orderB]
        log.info("Bath coupling strength\n%s\n%s", sigma[orderA], sigma[orderB])
    if kwargs.get("only_return_bath", False):
        return B
    else:
        return basis

def __embBasis_phsymm(lattice, GRho, **kwargs):
    """
    BCS bath from quasiparticle embedding.
    Phys. Rev. B, 93, 035126 (2016).
    """
    if "sites" in kwargs:
        log.error('keyword "sites" not supported.')
    log.eassert(lattice.nval == lattice.nscsites, \
            "Non-local bath does not support truncation.")
    ncells = lattice.ncells
    nscsites = lattice.nscsites
    basis = np.empty((2, ncells, nscsites*2, nscsites*2))
    # particle part -> alpha spin
    A1 = MatSqrt(GRho[0])
    BA1 = np.dot(GRho, la.inv(A1.T))
    BA1 = orthonormalizeBasis(BA1)
    basis[0] = BA1
    
    # hole part -> beta spin
    GRho_h = -GRho
    GRho_h[0] += np.eye(nscsites*2)
    A2 = MatSqrt(GRho_h[0])
    BA2 = np.dot(GRho_h, la.inv(A2.T))
    BA2 = orthonormalizeBasis(BA2)
    basis[1, :, :nscsites], basis[1, :, nscsites:] = \
            BA2[:, nscsites:], BA2[:, :nscsites]
    return basis

def embHam(lattice, basis, vcor, mu, local=True, **kwargs):
    # First transform two-body, since we need ERI to calculate JK_emb for interacting bath.
    log.info("Two-body part")
    Int2e, Int1e_from2e, H0_from2e = \
            __embHam2e(lattice, basis, vcor, local, **kwargs)
    log.info("One-body part")
    (Int1e, H0_from1e), (Int1e_energy, H0_energy_from1e) = \
            __embHam1e(lattice, basis, vcor, mu, Int2e, **kwargs)
    
    nbasis = basis.shape[-1]
    H0 = H0_from1e + H0_from2e
    H0_energy = H0_energy_from1e + H0_from2e
    if Int1e_from2e is not None:
        Int1e["cd"] += Int1e_from2e["cd"]
        Int1e["cc"] += Int1e_from2e["cc"]
        Int1e_energy["cd"] += Int1e_from2e["cd"]
        Int1e_energy["cc"] += Int1e_from2e["cc"]
    return integral.Integral(nbasis, False, True, H0, Int1e, Int2e), \
            (Int1e_energy, H0_energy)

get_emb_Ham = embHam

def __embHam2e(lattice, basis, vcor, local, int_bath=False, \
        last_aabb=True, **kwargs):
    nscsites = lattice.nscsites
    nbasis = basis.shape[-1]
    eri_symmetry = lattice.eri_symmetry 
    max_memory = kwargs.get("max_memory", lattice.cell.max_memory)
    
    if lattice.is_model:
        LatH2 = lattice.getH2(compact=False, kspace=False)
        if settings.save_mem:
            if local:
                return {"ccdd": LatH2[np.newaxis], "cccd": None, "cccc": None}, \
                        None, 0.
            else:
                log.warning("Basis nonlocal, ignoring memory saving option")
                settings.save_mem = False

        if kwargs.get("mmap", False):
            log.debug(0, "Use memory map for 2-electron integral")
            ccdd = np.memmap(NamedTemporaryFile(dir=TmpDir), dtype=float, \
                    mode='w+', shape=(3, nbasis, nbasis, nbasis, nbasis))
            cccd = np.memmap(NamedTemporaryFile(dir=TmpDir), dtype=float, \
                mode='w+', shape=(2, nbasis, nbasis, nbasis, nbasis))
            cccc = np.memmap(NamedTemporaryFile(dir=TmpDir), dtype=float, \
                mode='w+', shape=(1, nbasis, nbasis, nbasis, nbasis))
        else:
            ccdd = np.zeros((3, nbasis, nbasis, nbasis, nbasis))
            cccd = np.zeros((2, nbasis, nbasis, nbasis, nbasis))
            cccc = np.zeros((1, nbasis, nbasis, nbasis, nbasis))
        log.info("H2 memory allocated size = %d MB", \
                ccdd.size * 2 * 8. / 1024 / 1024)
         
        if local:
            if "sites" in kwargs:
                log.eassert(Lat.H2_format == "local", "only local H2 \
                        is supported for sites keywords")
                Imps = kwargs["sites"]
                nImp = len(Imps)
                mask = np.ix_(Imps, Imps, Imps, Imps)
                for i in range(ccdd.shape[0]):
                    ccdd[i, :nImp, :nImp, :nImp, :nImp] = LatH2[mask]
                return {"ccdd": ccdd, "cccd": cccd, "cccc": cccc}, \
                        None, 0.0
            else:
                for s in range(2):
                    log.eassert(max_abs(basis[s, 0, :nscsites, :nscsites] - np.eye(nscsites)) \
                            < 1e-10, "the embedding basis is not local")
                # ZHC TODO add interacting bath
                if lattice.H2_format == "local":
                    if int_bath:
                        raise NotImplementedError
                        H2 = transform_eri_local(basis, lattice, LatH2) 
                    else:
                        for i in range(ccdd.shape[0]):
                            ccdd[i, :nscsites, :nscsites, :nscsites, :nscsites] = LatH2
                elif lattice.H2_format == "nearest":
                    if int_bath:
                        raise NotImplementedError
                    else:
                        for i in range(H2.shape[0]):
                            H2[i, :nscsites, :nscsites, :nscsites, :nscsites] = LatH2[0]
                elif lattice.H2_format == "full":
                    if int_bath:
                        raise NotImplementedError
                    else:
                        for i in range(H2.shape[0]):
                            H2[i, :nscsites, :nscsites, :nscsites, :nscsites] = LatH2[0, 0, 0]
                else:
                    raise ValueError
                return {"ccdd": ccdd, "cccd": cccd, "cccc": cccc}, None, 0.
        else: # non-local basis (quasi-particle embedding)
            log.warn("Nonlocal basis is in test and only models \
                    are supported.")
            from libdmet.integral.integral_nonlocal_emb import transform
            VA, VB, UA, UB = separate_basis(basis)
            if lattice.H2_format == 'local':
                if int_bath:
                    raise NotImplementedError
                else:
                    H01, cd1, cc1, ccdd1, cccd1, cccc1 = \
                            transform(VA[0], VB[0], UA[0], UB[0], LatH2)
            elif lattice.H2_format == 'nearest':
                if int_bath:
                    raise NotImplementedError
                else:
                    H01, cd1, cc1, ccdd1, cccd1, cccc1 = \
                            transform(VA[0], VB[0], UA[0], UB[0], LatH2[0])
            elif lattice.H2_format == 'full':
                raise NotImplementedError
            else:
                raise ValueError
            # ZHC FIXME the definition of UA and UB
            return {"ccdd": ccdd1, "cccd": cccd1, "cccc": cccc1}, {"cd": cd1, "cc": cc1}, H01
    else: # ab initio
        raise NotImplementedError

def __embHam1e(lattice, basis, vcor, mu, H2_emb, \
        int_bath=False, add_vcor=False, **kwargs):
    log.eassert(vcor.islocal(), "nonlocal correlation potential cannot be treated in this routine")
    ncells = lattice.ncells
    nscsites = lattice.nscsites
    nbasis = basis.shape[-1]

    fock_R = lattice.getFock(kspace=False)
    hcore_R = lattice.getH1(kspace=False)
    ImpJK = lattice.getImpJK()
    
    spin = 2
    H0 = 0.
    H1 = {"cd": np.empty((2, nbasis, nbasis)), "cc": np.empty((1, nbasis, nbasis))}
    H0energy = 0.
    H1energy = {"cd": np.empty((2, nbasis, nbasis)), "cc": np.empty((1, nbasis, nbasis))}
    
    log.debug(1, "transform hcore")
    H1["cd"], H1["cc"][0], H0 = \
            transform_trans_inv_sparse(basis, lattice, hcore_R)

    if int_bath:
        raise NotImplementedError
    else:
        add_vcor = True
        if lattice.use_hcore_as_emb_ham:
            log.debug(1, "Use hcore as embedding Hamiltonian.")
            # NIB and use hcore, do not include JK_core in energy.
            lattice.JK_core = None
        else: # NIB but use fock as embedding hamiltonian.
            raise NotImplementedError
    
    if add_vcor:
        # then add Vcor, only in environment; and -mu*I in impurity and environment
        # add it everywhere then subtract impurity part
        log.debug(1, "transform Vcor")
        v = deepcopy(vcor.get())
        v[0] -= mu * np.eye(nscsites)
        v[1] -= mu * np.eye(nscsites)
        tempCD, tempCC, tempH0 = transform_local(basis, lattice, v)
        H1["cd"] += tempCD
        H1["cc"][0] += tempCC
        H0 += tempH0

        if not "fitting" in kwargs or not kwargs["fitting"]:
            # for fitting purpose, we need H1 with vcor on impurity
            tempCD, tempCC, tempH0 = transform_imp(basis, lattice, vcor.get())
            H1["cd"] -= tempCD
            H1["cc"][0] -= tempCC
            H0 -= tempH0

        # subtract impurity Fock if necessary
        # i.e. rho_kl[2(ij||kl)-(il||jk)] where i,j,k,l are all impurity
        if ImpJK is not None:
            log.debug(1, "transform impurity JK")
            tempCD, tempCC, tempH0 = transform_imp(basis, lattice, ImpJK)
            H1["cd"] -= tempCD
            H1["cc"][0] -= tempCC
            H0 -= tempH0

    log.debug(1, "transform native H1")
    H1energy["cd"], H1energy["cc"][0], H0energy = transform_imp_env(basis, lattice, hcore_R)
    return (H1, H0), (H1energy, H0energy)

def foldRho(GRho, Lat, basis, thr=1e-7):
    # ZHC NOTE be careful of "sites" version
    ncells = Lat.ncells
    nscsites = Lat.nscsites
    nbasis = basis.shape[-1]
    basisCanonical = np.empty((ncells, nscsites*2, nbasis*2))
    basisCanonical[:, :,:nbasis] = basis[0] # (VA, UB)^T
    basisCanonical[:, :nscsites, nbasis:] = basis[1, :, nscsites:] # UA
    basisCanonical[:, nscsites:, nbasis:] = basis[1, :, :nscsites] # VB
    res = np.zeros((nbasis*2, nbasis*2))
    mask_basis = set(find(True, map(lambda a: la.norm(a) > thr, basisCanonical)))
    mask_GRho = set(find(True, map(lambda a: la.norm(a) > thr, GRho)))
    if len(mask_GRho) < len(mask_basis):
        for Hidx in mask_GRho:
            for i in mask_basis:
                #j = Lat.add(i, Hidx)
                j = Lat.subtract(i, Hidx) # J = I - X -> X = I - J
                if j in mask_basis:
                    res += mdot(basisCanonical[i].T, GRho[Hidx], basisCanonical[j])
    else:
        for i, j in it.product(mask_basis, repeat=2):
            Hidx = Lat.subtract(i, j)
            if Hidx in mask_GRho:
                res += mdot(basisCanonical[i].T, GRho[Hidx], basisCanonical[j])
    return res

def foldRho_k(GRho_k, basis_k):
    return slater_helper.transform_trans_inv_k(basis_k, GRho_k)

def addDiag(v, scalar):
    rep = v.get()
    nscsites = rep.shape[1]
    rep[0] += np.eye(nscsites) * scalar
    rep[1] += np.eye(nscsites) * scalar
    v.assign(rep)
    return v

def FitVcorEmb(GRho, lattice, basis, vcor, mu, beta=np.inf, \
        MaxIter=300, CG_check=False, BFGS=False, diff_criterion=None, \
        imp_fit=False, fit_idx=None, **kwargs):
    """
    Fitting the correlation potential in the embedding space.
    
    Analytic gradient for 0 T:
        # dGRho_ij / dV_ij, where V corresponds to terms in the
        # embedding generalized density matrix
        #c_jln = np.einsum("jn,ln->jln", evocc, evocc)
        #c_ikm = np.einsum("im,km->ikm", evvirt, evvirt)
        #e_mn = 1. / (-ewvirt.reshape((-1,1)) + ewocc)
        #dGRho_dV = np.swapaxes(np.tensordot(np.tensordot(c_ikm, e_mn, \
        #        axes = (2,0)), c_jln, axes = (2,2)), 1, 2)
        #dGRho_dV += np.swapaxes(np.swapaxes(dGRho_dV, 0, 1), 2, 3)
        #dnorm_dV = np.tensordot(GRho1 - GRho, dGRho_dV, \
        #        axes = ((0,1), (0,1))) / val / sqrt(2.)
    """
    param_begin = vcor.param.copy()
    nscsites = lattice.nscsites
    nbasis = basis.shape[-1]
    mu0 = kwargs.get("mu0", 0.0)
    fix_mu = kwargs.get("fix_mu", True)
    log.eassert(imp_fit == False and fit_idx is None, \
            "Only imp+bath fit is supported.")

    if lattice.use_hcore_as_emb_ham:
        fock_R = lattice.getH1(kspace=False)
    else:
        fock_R = lattice.getFock(kspace=False)
    (embHA, embHB), embD, _ = transform_trans_inv_sparse(basis, \
            lattice, fock_R)
    embH = np.empty((nbasis*2, nbasis*2))
    embH[:nbasis, :nbasis] = embHA
    embH[nbasis:, nbasis:] = -embHB
    embH[:nbasis, nbasis:] = embD
    embH[nbasis:, :nbasis] = embD.T

    # now compute dV/dparam (will be used in gradient)
    dV_dparam = np.empty((vcor.length(), nbasis*2, nbasis*2))
    for ip in range(vcor.length()):
        (dA_dV, dB_dV), dD_dV, _ = \
                transform_local(basis, lattice, vcor.gradient()[ip])
        dV_dparam[ip, :nbasis, :nbasis] = dA_dV
        dV_dparam[ip, nbasis:, nbasis:] = -dB_dV
        dV_dparam[ip, :nbasis, nbasis:] = dD_dV
        dV_dparam[ip, nbasis:, :nbasis] = dD_dV.T

    vcor_zero = deepcopy(vcor)
    vcor_zero.update(np.zeros(vcor_zero.length()))
    v0 = vcor_zero.get()
    v0[0] -= mu * np.eye(nscsites)
    v0[1] -= mu * np.eye(nscsites)
    (A0, B0), D0, _ = \
            transform_local(basis, lattice, v0)

    def Vemb_param(param):
        V = np.tensordot(param, dV_dparam, axes = (0, 0))
        V[:nbasis, :nbasis] += A0
        # add contribution of chemical potential
        V[nbasis:, nbasis:] -= B0
        V[:nbasis, nbasis:] += D0
        V[nbasis:, :nbasis] += D0.T
        return V
    
    if beta == np.inf:
        def errfunc(param):
            embHeff = embH + Vemb_param(param)
            ew, ev = la.eigh(embHeff)
            occ = 1 * (ew < 0.)
            GRho1 = np.dot(ev*occ, ev.T)
            return la.norm(GRho1 - GRho) / sqrt(2.)

        def gradfunc(param):
            embHeff = embH + Vemb_param(param)
            ew, ev = la.eigh(embHeff)
            occ = 1 * (ew < 0.)
            GRho1 = np.dot(ev*occ, ev.T)
            val = la.norm(GRho1 - GRho)
            ewocc, ewvirt = ew[:nbasis], ew[nbasis:]
            evocc, evvirt = ev[:, :nbasis], ev[:, nbasis:]
            
            e_mn = 1. / (-ewvirt.reshape((-1,1)) + ewocc)
            temp_mn = mdot(evvirt.T, GRho1 - GRho, evocc) \
                    * e_mn / (val * sqrt(2.))
            dnorm_dV = mdot(evvirt, temp_mn, evocc.T)
            dnorm_dV += dnorm_dV.T
            return np.tensordot(dV_dparam, dnorm_dV, axes=((1,2), (0,1)))
    else: # finite T
        def errfunc(param):
            embHeff = embH + Vemb_param(param)
            mo_energy, mo_coeff, mo_occ, mu_ref = \
                    ftsystem.kernel(embHeff, embHeff.shape[-1]//2, \
                    beta, mu0=mu0, fix_mu=fix_mu)
            GRho1 = ftsystem.make_rdm1(mo_coeff, mo_occ)
            return la.norm((GRho1 - GRho)) / sqrt(2.)
        
        def gradfunc(param):
            embHeff = embH + Vemb_param(param)
            mo_energy, mo_coeff, mo_occ, mu_ref = \
                    ftsystem.kernel(embHeff, embHeff.shape[-1]//2, \
                    beta, mu0=mu0, fix_mu=fix_mu)
            GRho1 = ftsystem.make_rdm1(mo_coeff, mo_occ)
            dGRho = GRho1 - GRho
            val = la.norm(dGRho)
            dw_dv = ftsystem.get_dw_dv(mo_energy, mo_coeff, dGRho, \
                    mu_ref, beta, fix_mu=fix_mu)
            dw_dparam = dV_dparam.reshape(dV_dparam.shape[0], -1).dot(dw_dv.ravel()) \
                    / (2.0 * val * sqrt(2.))
            return dw_dparam
    
    err_begin = errfunc(vcor.param)
    if beta == np.inf:
        log.info("Using analytic gradient for 0 T")
    else:
        log.info("Using analytic gradient for finite T, beta = %s", beta)
        if imp_fit:
            log.warn("Finite T grad for impurity fitting is not implemented." +\
                    "Use numerical gradient.")
            gradfunc = None
    if kwargs.get("test_grad", False):
        from libdmet.routine.slater import test_grad
        test_grad(vcor, errfunc, gradfunc, dx=1e-4)
        test_grad(vcor, errfunc, gradfunc, dx=1e-5)
    
    param, err_end, pattern, gnorm_res = minimize(errfunc, vcor.param, \
            MaxIter, gradfunc, **kwargs)
    vcor.update(param)
    log.info("Minimizer converge pattern: %d ", pattern)
    log.info("Current function value: %15.8f", err_end)
    log.info("Norm of gradients: %s", gnorm_res)
    log.info("Norm diff of x: %15.8f", max_abs(param - param_begin))

    if CG_check and (pattern == 0 or gnorm_res > 1.0e-4):
        log.info("Check with optimizer in Scipy...")
        from scipy import optimize as opt
        param_new = param.copy()
        gtol = max(5.0e-5, gnorm_res * 0.1)
        gtol = min(gtol, 1.0e-2)
        if BFGS:
            log.info("BFGS used in check")
            method = 'BFGS'
        else:
            log.info("CG used in check")
            method = 'CG'
        min_result = opt.minimize(errfunc, param_new, \
                method=method, jac=gradfunc ,\
                options={'maxiter': len(param_new)*10, \
                'disp': True, 'gtol': gtol})
        param_new_2 = min_result.x
        log.info("CG Final Diff: %s", min_result.fun) 
        log.info("Converged: %s", min_result.status)
        log.info("Jacobian: %s", max_abs(min_result.jac))
        if(not min_result.success):
            log.warn("Minimization unsuccessful. Message:\n%s", min_result.message)
        
        gnorm_new = max_abs(min_result.jac)
        diff_CG_old = max_abs(param_new_2 - param_new)
        log.info("max diff in x between %s and old: %s", method, diff_CG_old)
        if diff_criterion is None:
            if pattern == 0:
                diff_criterion = 2.0
            else:
                diff_criterion = 1.0
        if (gnorm_new < gnorm_res * 0.9) and (min_result.fun < err_end) \
                and (diff_CG_old < diff_criterion):
            log.info("New result used")
            vcor.update(param_new_2)
            err_end = min_result.fun
        else:
            log.info("Old result used")
            vcor.update(param_new)
    else:
        log.info("Old result used")
    return vcor, err_begin, err_end
        
def FitVcorFull(GRho, lattice, basis, vcor, mu, beta=np.inf, MaxIter=20, method='CG', \
        ytol=1e-7, gtol=1e-2, **kwargs):
    nbasis = basis.shape[-1]
    verbose = log.verbose

    def callback(param):
        vcor.update(param)
        log.verbose = "RESULT"
        GRhoTRef, _, _ = HFB(lattice, vcor, False, mu=mu, beta=beta)
        log.verbose = verbose
        GRho1Ref = foldRho(GRhoTRef, lattice, basis, thr=1e-8)
        return {"GRhoTRef": GRhoTRef, "GRho1Ref": GRho1Ref}

    def errfunc(param, ref = None):
        vcor.update(param)
        log.verbose = "RESULT"
        GRhoT, _, _ = HFB(lattice, vcor, False, mu=mu, beta=beta)
        log.verbose = verbose
        if ref is None:
            GRho1 = foldRho(GRhoT, lattice, basis, thr=1e-8)
        else:
            GRho1 = foldRho(GRhoT - ref["GRhoTRef"], lattice, \
                    basis, thr=1e-8) + ref["GRho1Ref"]
        return la.norm(GRho1 - GRho) / sqrt(2.)

    err_begin = errfunc(vcor.param)
    param, err_end, pattern, gnorm_res = minimize(errfunc, \
            vcor.param, MaxIter, callback = callback, \
            method=method, ytol=ytol, gtol=gtol, **kwargs)
    vcor.update(param)
    return vcor, err_begin, err_end

def FitVcorFullK(GRho, lattice, basis, vcor, mu, MaxIter, **kwargs):
    nscsites = lattice.nscsites

    def costfunc(param, v = False):
        vcor.update(param)
        verbose = log.verbose
        log.verbose = "RESULT"
        GRhoT, _, _ = HFB(lattice, vcor, False, mu = mu, beta = np.inf)
        log.verbose = verbose

        tempRdm = [extractRdm(x) for x in GRhoT]
        rhoAT = np.asarray([rhoA for (rhoA, rhoB, kappaBA) in tempRdm])
        rhoBT = np.asarray([rhoB for (rhoA, rhoB, kappaBA) in tempRdm])
        kappaBA0 = tempRdm[0][2]

        kinetic = np.sum((rhoAT+rhoBT) * lattice.getFock(kspace = False))

        rhoA, rhoB, kappaBA = extractRdm(GRho)
        rhoAImp = rhoA[:nscsites, :nscsites]
        rhoBImp = rhoB[:nscsites, :nscsites]
        kappaBAImp = kappaBA[:nscsites, :nscsites]

        constraint = np.sum((vcor.get()[0]-mu*np.eye(nscsites)) * (rhoAT[0] - rhoAImp)) + \
                np.sum((vcor.get()[1]-mu*np.eye(nscsites)) * (rhoBT[0] - rhoBImp)) + \
                np.sum(vcor.get()[2] * (kappaBA0 - kappaBAImp).T) + \
                np.sum(vcor.get()[2].T * (kappaBA0 - kappaBAImp))

        if v:
            return kinetic, constraint
        else:
            return -(kinetic + constraint)

    def grad(param):
        vcor.update(param)
        verbose = log.verbose
        log.verbose = "RESULT"
        GRhoT, _, _ = HFB(lattice, vcor, False, mu = mu, beta = np.inf)
        log.verbose = verbose

        rhoA0, rhoB0, kappaBA0 = extractRdm(GRhoT[0])
        rhoA, rhoB, kappaBA = extractRdm(GRho)

        dRho = np.asarray([rhoA0 - rhoA[:nscsites, :nscsites], \
                rhoB0 - rhoB[:nscsites, :nscsites], \
                2 * (kappaBA0.T - kappaBA.T[:nscsites, :nscsites])])
        return -np.tensordot(vcor.gradient(), dRho, axes = ((1,2,3), (0,1,2)))

    from scipy.optimize import minimize
    ke_begin, c_begin = costfunc(vcor.param, v = True)
    log.info("begin: \nkinetic energy = %20.12f    constraint = %20.12f", ke_begin, c_begin)
    param = minimize(costfunc, vcor.param, jac = grad).x
    ke_end, c_end = costfunc(param, v = True)
    log.info("end: \nkinetic energy = %20.12f    constraint = %20.12f", ke_end, c_end)

    vcor.update(param)
    return vcor, c_begin, c_end

def FitVcorTwoStep(GRho, lattice, basis, vcor, mu, beta=np.inf, \
        MaxIter1=300, MaxIter2=0, kinetic=False, \
        CG_check=False, BFGS=False, serial=True, \
        method='CG', ytol=1e-7, gtol=5e-3, **kwargs):
    """
    Main wrapper for correlation potential fitting.
    """
    vcor_new = deepcopy(vcor)
    log.result("Using two-step vcor fitting")
    err_begin = None
    if kinetic:
        log.check(MaxIter1 > 0, \
                "Embedding fitting with kinetic energy minimization does not work!\n"
                "Skipping Embedding fitting")
        if MaxIter2 == 0:
            log.warning("Setting MaxIter2 to 1")
            MaxIter2 = 1
        vcor_new, err_begin, err_end = FitVcorFullK(GRho, lattice, basis, vcor_new, \
                    mu, MaxIter = MaxIter2)
    else:
        if MaxIter1 > 0:
            log.info("Impurity model stage max %d steps", MaxIter1)
            log.info("Finite temperature used in fitting? beta = %15.6f ", beta)
            vcor_new, err_begin1, err_end1 = FitVcorEmb(GRho, lattice, basis, vcor_new, \
                    mu, beta=beta, MaxIter=MaxIter1, CG_check=CG_check, serial=serial, \
                    BFGS=BFGS, method=method, ytol=ytol, gtol=gtol, **kwargs)
            log.info("Embedding Stage:\nbegin %20.12f    end %20.12f" % (err_begin1, err_end1))
        if MaxIter2 > 0:
            log.info("Full lattice stage  max %d steps", MaxIter2)
            vcor_new, err_begin2, err_end2 = FitVcorFull(GRho, lattice, basis, vcor_new, \
                    mu, MaxIter=MaxIter2, beta=beta, method=method, ytol=ytol, gtol=gtol)
            log.info("Full Lattice Stage:\nbegin %20.12f    end %20.12f" % (err_begin2, err_end2))
        if MaxIter1 > 0:
            err_begin = err_begin1
        else:
            err_begin = err_begin2
        if MaxIter2 > 0:
            err_end = err_end2
        else:
            err_end = err_end1

    log.result("residue (begin) = %20.12f", err_begin)
    log.result("residue (end)   = %20.12f", err_end)
    return vcor_new, err_end

def transformResults(GRhoEmb, E, lattice, basis, ImpHam, \
        H_energy, mu, **kwargs):
    """
    Give last_dmu for energy evaluation.
    """
    # ZHC FIXME nval for bath truncation.
    VA, VB, UA, UB = separate_basis(basis)
    nscsites = basis.shape[-2] // 2
    nbasis = basis.shape[-1]
    R = np.empty((nscsites*2, nbasis*2))
    R[:nscsites, :nbasis] = VA[0]
    R[nscsites:, :nbasis] = UB[0]
    R[:nscsites, nbasis:] = UA[0]
    R[nscsites:, nbasis:] = VB[0]
    GRhoImp = mdot(R, GRhoEmb, R.T)
    occs = np.diag(GRhoImp)
    nelec = np.sum(occs[:nscsites]) - np.sum(occs[nscsites:]) + nscsites
    if E is not None:
        # The energy is from defination of Edmet.
        # Efrag = E1 + E2
        # where E1 = partial Tr(rho, H1), 
        # H1 should not include contribution from Mu and last_dmu
        # E2 = E_solver - <psi | himp| psi>, psi is the wavefunction
        last_dmu = kwargs["last_dmu"]
        rhoA, rhoB, kappaBA = extractRdm(GRhoEmb)
        E2 = E - np.sum(ImpHam.H1["cd"][0]*rhoA + ImpHam.H1["cd"][1]*rhoB) - \
                2.0*np.sum(ImpHam.H1["cc"][0]*kappaBA.T) - ImpHam.H0

        # remove the contribution of last_dmu
        from libdmet.dmet.HubbardBCS import apply_dmu
        ImpHam_no_last_dmu = apply_dmu(lattice, deepcopy(ImpHam), basis, -last_dmu)
        H1_scaled = deepcopy(ImpHam_no_last_dmu.H1)
        
        # add back the global mu 
        v = np.zeros((3, nscsites, nscsites))
        v[0] = mu * np.eye(nscsites)
        v[1] = mu * np.eye(nscsites)
        tempCD, tempCC, tempH0 = transform_local(basis, lattice, v)
        H1_scaled["cd"] += tempCD
        H1_scaled["cc"][0] += tempCC

        # scale by the number of imp indices
        limp = lattice.limp 
        H1_scaled["cd"][0][:limp, limp:] *= 0.5
        H1_scaled["cd"][0][limp:, :limp] *= 0.5
        H1_scaled["cd"][0][limp:, limp:] = 0.0
        H1_scaled["cd"][1][:limp, limp:] *= 0.5
        H1_scaled["cd"][1][limp:, :limp] *= 0.5
        H1_scaled["cd"][1][limp:, limp:] = 0.0
        H1_scaled["cc"][0][:limp, limp:] *= 0.5
        H1_scaled["cc"][0][limp:, :limp] *= 0.5
        H1_scaled["cc"][0][limp:, limp:] = 0.0

        E1 = np.sum(H1_scaled["cd"][0]*rhoA + H1_scaled["cd"][1]*rhoB) + \
                2.0 * np.sum(H1_scaled["cc"][0]*kappaBA.T)
        Efrag = E1 + E2
    else:
        Efrag = None
    return GRhoImp, Efrag, nelec


# ****************************
# Functions in test
# ****************************

#def FitVcorEmb_QC(GRho_list, lattice, basis_list, vcor_list, mu, Imp_list, MaxIter = 300, CG_check = False, **kwargs):
#    """
#    QC version
#    """
#    nImp = len(Imp_list)
#    param_all_begin = np.hstack((vcor_list[i].param for i in range(nImp)))
#    vcor_list_copy = deepcopy(vcor_list)
#
#    embH_list = []
#    dV_dparam_list = []
#    ABD_list = []
#
#    for i in range(nImp):
#        GRho = GRho_list[i]
#        basis = basis_list[i]
#        vcor = vcor_list[i]
#        Imp = Imp_list[i]
#
#        param_begin = vcor.param.copy()
#        #nscsites = lattice.nscsites
#        limp = len(Imp)
#        nbasis = basis.shape[-1]
#        (embHA, embHB), embD, _ = transform_trans_inv_sparse(basis, lattice, \
#                lattice.getFock(kspace = False))
#        embH = np.empty((nbasis*2, nbasis*2))
#        embH[:nbasis, :nbasis] = embHA
#        embH[nbasis:, nbasis:] = -embHB
#        embH[:nbasis, nbasis:] = embD
#        embH[nbasis:, :nbasis] = embD.T
#        
#        # now compute dV/dparam (will be used in gradient)
#        dV_dparam = np.empty((vcor.length(), nbasis*2, nbasis*2))
#        for ip in range(vcor.length()):
#            (dA_dV, dB_dV), dD_dV, _ = \
#                    transform_local(basis, lattice, vcor.gradient()[ip])
#            dV_dparam[ip, :nbasis, :nbasis] = dA_dV
#            dV_dparam[ip, nbasis:, nbasis:] = -dB_dV
#            dV_dparam[ip, :nbasis, nbasis:] = dD_dV
#            dV_dparam[ip, nbasis:, :nbasis] = dD_dV.T
#        
#        vcor_zero = deepcopy(vcor)
#        vcor_zero.update(np.zeros(vcor_zero.length()))
#        v0 = vcor_zero.get()
#        #v0[0] -= mu * np.eye(nscsites)
#        #v0[1] -= mu * np.eye(nscsites)
#        v0[0] -= mu * np.eye(limp)
#        v0[1] -= mu * np.eye(limp)
#        (A0, B0), D0, _ = \
#                transform_local(basis, lattice, v0)
#        
#        embH_list.append(embH.copy())
#        dV_dparam_list.append(dV_dparam.copy())
#        ABD_list.append((A0.copy, B0.copy(), D0.copy()))
#        
#    def vcor_list_update(param_all):
#        start = 0
#        for i in range(nImp):
#            end = start + len(vcor_list[i].param)
#            param = param_all[start:end]
#            vcor_list[i].update(param)
#            start = end
#
#    def Vemb_param(param, dV_dparam, ABD):
#        A0, B0, D0 = ABD
#        V = np.tensordot(param, dV_dparam, axes = (0, 0))
#        # add contribution of chemical potential # ZHC NOTE
#        V[:nbasis, :nbasis] += A0
#        V[nbasis:, nbasis:] -= B0
#        V[:nbasis, nbasis:] += D0
#        V[nbasis:, :nbasis] += D0.T
#        return V
#
#    def errfunc(param_all):
#        res = 0.0
#        start = 0
#        for i in range(nImp):
#            end = start + len(vcor_list[i].param)
#            param = param_all[start:end]
#            #vcor_list[i].update(param)
#            start = end
#            
#            embHeff = embH_list[i] + Vemb_param(param, dV_dparam_list[i], ABD_list[i])
#            ew, ev = la.eigh(embHeff)
#            occ = 1 * (ew < 0.)
#            GRho1 = mdot(ev, np.diag(occ), ev.T)
#            res += la.norm(GRho_list[i] - GRho1) / sqrt(2.)
#        return res
#    
#    def gradfunc(param_all):
#        res = 0.0
#        start = 0
#        grad_list = []
#        for i in range(nImp):
#            
#            GRho = GRho_list[i]
#            embH = embH_list[i]
#            dV_dparam = dV_dparam_list[i]
#            ABD = ABD_list[i]
#            basis = basis_list[i]
#            nbasis = basis.shape[-1]
#
#            end = start + len(vcor_list[i].param)
#            param = param_all[start:end]
#            #vcor_list[i].update(param)
#            start = end
#            embHeff = embH + Vemb_param(param, dV_dparam, ABD)
#            ew, ev = la.eigh(embHeff)
#            nocc = 1 * (ew < 0.)
#            GRho1 = mdot(ev, np.diag(nocc), ev.T)
#            val = la.norm(GRho - GRho1)
#       
#            ewocc, ewvirt = ew[:nbasis], ew[nbasis:]
#            evocc, evvirt = ev[:, :nbasis], ev[:, nbasis:]
#            
#            e_mn = 1. / (-ewvirt.reshape((-1,1)) + ewocc)
#            temp_mn = mdot(evvirt.T, GRho1 - GRho, evocc) * e_mn / val / sqrt(2.)
#            dnorm_dV = mdot(evvirt, temp_mn, evocc.T)
#            dnorm_dV += dnorm_dV.T
#            grad_i = np.tensordot(dV_dparam, dnorm_dV, axes = ((1,2), (0,1)))
#            grad_list.append(grad_i)
#        return np.hstack(grad_list)
#
#    err_begin = errfunc(param_all_begin)
#    log.info("Using analytic gradient")
#    param_all, err_end, pattern, gnorm_res = minimize(errfunc, vcor.param, MaxIter, gradfunc, **kwargs)
#    
#    # ZHC NOTE
#    #gnorm_res = max_abs(gradfunc(param_all))
#    vcor_list_update(param_all)
#    
#    log.info("Minimizer converge pattern: %d ", pattern)
#    log.info("Current function value: %15.8f", err_end)
#    log.info("Norm of gradients: %15.8f", gnorm_res)
#    log.info("Norm diff of x: %15.8f", (max_abs(param_all - param_all_begin)))
#    
#    if CG_check and (pattern == 0 or gnorm_res > 1.0e-4):
#        
#        log.info("Not converge in Bo-Xiao's minimizer, try mixed solver in scipy...")
#
#        param_new = param_all.copy()
#        gtol = max(5.0e-5, gnorm_res * 0.1)
#        gtol = min(gtol, 1.0e-2)
#
#        from scipy import optimize as opt
#        min_result = opt.minimize(errfunc, param_new, method = 'CG', jac = gradfunc ,\
#                options={'maxiter': 10 * len(param_new), 'disp': True, 'gtol': gtol})
#        param_new_2 = min_result.x
#    
#        log.info("CG Final Diff: %s, Converged: %s, Jacobian: %s", \
#                min_result.fun, min_result.status, max_abs(min_result.jac))      
#        if(not min_result.success):
#            log.warn("Minimization unsuccessful. Message: %s", min_result.message)
#    
#        gnorm_new = max_abs(min_result.jac)
#        diff_CG_old = max_abs(param_new_2 - param_new)
#        log.info("max diff in x between CG and old: %s", diff_CG_old)
#        if (gnorm_new < gnorm_res * 0.5) and (min_result.fun < err_end) and (diff_CG_old < 1.0):
#            log.info("CG result used")
#            vcor_list_update(param_new_2)
#            err_end = min_result.fun
#        else:
#            log.info("old result used")
#            vcor_list_update(param_new)
#    else:
#        log.info("old result used")
#    
#    return vcor_list, err_begin, err_end
