#! /usr/bin/env python

"""
Module of DMET with generalized spin orbital formalism.
"""

from libdmet.dmet import Hubbard
from libdmet.dmet.Hubbard import *
from libdmet.routine.mfd import GHF
from libdmet.routine import spinless
from libdmet.routine.spinless_helper import (mono_fit, mono_fit_2, extractRdm, 
                                             transform_imp, separate_basis, 
                                             transform_H1_k, transform_H2_local, 
                                             transform_rdm1_k, transform_local)

def GHartreeFock(Lat, v, filling, mu0_elec, beta=np.inf, fix_mu=False, 
                 thrnelec=1e-8, **kwargs):
    """
    GHF wrapper.
    If filling is None, no fitting on mu. this mu is for real particle.
    mu0_elec is guess for real particle mu.
    fix_mu controls whether the quasiparticle mu is fitted.
    """
    if beta == np.inf:
        log.info("Using 0 T on lattice, beta = %s", beta)
    else:
        log.info("Using finite T on lattice, beta = %15.6f ", beta)
        log.info("Using fixed quasi particle fermi_level = 0.0 ? %s", fix_mu)

    if filling is None:
        mu = mu0_elec
    else:
        # fit mu to get correct filling
        log.info("chemical potential fitting, target = %20.12f", filling)
        log.info("before fitting, mu = %20.12f", mu0_elec)
        fn = lambda mu: GHF(Lat, v, False, mu=mu, beta=beta, fix_mu=fix_mu, 
                            ires=False, **kwargs)[1] / (Lat.nscsites * 2.0)
        #mu = mono_fit(fn, filling, mu0_elec, thrnelec, increase=True)
        mu = mono_fit_2(fn, filling, mu0_elec, thrnelec, increase=True)
        fn_opt = fn(mu)
        log.info("after fitting, mu = %20.12f", mu)
        log.info("after fitting, f(x) = %20.12f", fn_opt)
    rho, n, E, res = GHF(Lat, v, False, mu=mu, beta=beta, fix_mu=fix_mu, 
                         ires=True, **kwargs)

    rhoA, rhoB, kappaAB = extractRdm(rho[0])
    if filling is None:
        log.result("Local density matrix (mean-field): alpha, beta and pairing"
                   "\n%s\n%s\n%s", rhoA, rhoB, kappaAB)
        log.result("nelec per cell (mean-field) = %20.12f", n)
        log.result("Energy per cell (mean-field) = %20.12f", E)
        log.result("Gap (mean-field) = %20.12f" % res["gap"])
        
        # analyze the results:
        labels = kwargs.get("labels", None)
        if labels is not None:
            Lat.mulliken_lo_R0(np.asarray((rhoA, rhoB)), labels=labels)

    if kwargs.get("full_return", False): 
        return rho, mu, res
    else:
        return rho, mu

def transformResults(GRhoEmb, E, lattice, basis, ImpHam, H_energy, 
                     mu, int_bath=False, fit_ghf=False, **kwargs):
    nscsites = basis.shape[-2] // 2
    GRhoImp, Efrag, nelec = spinless.transformResults(GRhoEmb, E, lattice,
                                                      basis, ImpHam, H_energy,
                                                      mu, fit_ghf=fit_ghf, 
                                                      **kwargs)
    log.debug(1, "impurity generalized density matrix:\n%s", GRhoImp)

    if Efrag is None:
        return nelec / nscsites
    else:
        if int_bath:
            # explicitly take out last_dmu, avoid duplicates in args
            last_dmu = kwargs.pop("last_dmu")
            Efrag = spinless.get_E_dmet(basis, lattice, ImpHam, **kwargs)
        log.result("Local density matrix (impurity): alpha, beta and pairing")
        rhoA, rhoB, kappaAB = extractRdm(GRhoImp)
        log.result("%s", rhoA)
        log.result("%s", rhoB)
        log.result("%s", kappaAB)
        log.result("nelec per cell (impurity) = %20.12f", nelec)
        log.result("Energy per cell (impurity) = %20.12f", Efrag)
        
        # analyze the results:
        labels = kwargs.get("labels", None)
        if labels is not None:
            lattice.mulliken_lo_R0(np.asarray((rhoA, rhoB)), labels=labels)

        return GRhoImp, Efrag / nscsites, nelec / nscsites

def Hubbard_transformResults(GRhoEmb, E, basis, ImpHam, H_energy, lattice=None,
                             fit_ghf=False, **kwargs):
    return transformResults(GRhoEmb, E, lattice, basis, ImpHam, H_energy, 0.0,
                            fit_ghf=fit_ghf, **kwargs)

# ZHC TODO consider not to change the module.
Hubbard.transformResults = Hubbard_transformResults

def ConstructImpHam(Lat, GRho, v, mu, matching=True, local=True, **kwargs):
    log.result("Making embedding basis")
    basis = spinless.embBasis(Lat, GRho, local=local, **kwargs)
    log.result("Constructing impurity Hamiltonian")
    ImpHam, _ = spinless.embHam(Lat, basis, v, mu, local=local, **kwargs)
    return ImpHam, None, basis

def apply_dmu(lattice, ImpHam, basis, dmu, fit_ghf=False, **kwargs):
    """
    Args:
        fit_ghf: used for fitting the particle number in the embedding space.
    """
    basis_Ra, basis_Rb = separate_basis(basis)
    if fit_ghf:
        nao = basis_Ra.shape[-2]
        mu_mat = np.zeros((2, nao, nao))
        np.fill_diagonal(mu_mat[0], -dmu)
        np.fill_diagonal(mu_mat[1],  dmu)
        ImpHam.H1["cd"] += transform_local(basis_Ra, basis_Rb, mu_mat)
    else:
        nao = lattice.nao
        # imp_idx can be part of indices in the reference cell
        dmu_idx = kwargs.get("dmu_idx", lattice.imp_idx)
        
        mu_mat = np.zeros((2, nao, nao))
        mu_mat[0][dmu_idx, dmu_idx] = -dmu
        mu_mat[1][dmu_idx, dmu_idx] =  dmu
        
        ImpHam.H1["cd"] += transform_imp(basis_Ra, basis_Rb, mu_mat)
    return ImpHam

Hubbard.apply_dmu = apply_dmu

def AFInitGuess(ImpSize, U, Filling, polar=None, rand=0.01, subA=None, 
                subB=None, bogo_res=False, d_wave=False, trace_zero=False):
    return Hubbard.AFInitGuess(ImpSize, U, Filling, polar, True, rand, 
                               subA=subA, subB=subB, bogo_res=bogo_res,
                               d_wave=d_wave, trace_zero=trace_zero)

addDiag = spinless.addDiag

FitVcor = spinless.FitVcorTwoStep

foldRho_k = spinless.foldRho_k

keep_vcor_trace_fixed = spinless.keep_vcor_trace_fixed

unit2emb = spinless.unit2emb

if __name__ == '__main__':
    pass
