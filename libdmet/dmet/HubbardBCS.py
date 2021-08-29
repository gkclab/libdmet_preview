#! /usr/bin/env python

from libdmet.dmet import Hubbard
from libdmet.dmet.Hubbard import *
from libdmet.routine import bcs
from libdmet.routine.mfd import HFB
from libdmet.routine.bcs_helper import mono_fit, extractRdm, transform_imp

def HartreeFockBogoliubov(Lat, v, filling, mu0, beta=np.inf, fix_mu=False, \
        thrnelec=1e-6, **kwargs):
    """
    HFB wrapper.
    If filling is None, no fitting on mu. this mu is for real particle.
    fix_mu controls whether the quasiparticle mu is fitted.
    """
    if beta == np.inf:
        log.info("Using 0 T on lattice, beta = %s", beta)
    else:
        log.info("Using finite T on lattice, beta = %15.6f ", beta)
        log.info("Using fixed quasi particle fermi_level = 0.0 ? %s", fix_mu)

    if filling is None:
        mu = mu0
    else:
        # fit mu to get correct filling
        log.info("chemical potential fitting, target = %20.12f", filling)
        log.info("before fitting, mu = %20.12f", mu0)
        fn = lambda mu: HFB(Lat, v, False, mu=mu, beta=beta, fix_mu=fix_mu, \
                ires=False, **kwargs)[1] / 2. / Lat.nscsites
        mu = mono_fit(fn, filling, mu0, thrnelec, increase=True)
        fn_opt = fn(mu)
        log.info("after fitting, mu = %20.12f", mu)
        log.info("after fitting, f(x) = %20.12f", fn_opt)
    rho, n, E, res = HFB(Lat, v, False, mu=mu, beta=beta, fix_mu=fix_mu, \
            ires=True, **kwargs)
    rhoA, rhoB, kappaBA = extractRdm(rho[0])
    if filling is None:
        log.result("Local density matrix (mean-field): alpha, beta and pairing"
                "\n%s\n%s\n%s", rhoA, rhoB, kappaBA.T)
        log.result("nelec per cell (mean-field) = %20.12f", n)
        log.result("Energy per cell (mean-field) = %20.12f", E)
        log.result("Gap (mean-field) = %20.12f" % res["gap"])
    if kwargs.get("full_return", False): 
        return rho, mu, res
    else:
        return rho, mu

def transformResults(GRhoEmb, E, lattice, basis, ImpHam, H_energy, \
        mu, int_bath=False, **kwargs):
    nscsites = basis.shape[-2] // 2
    GRhoImp, Efrag, nelec = bcs.transformResults(GRhoEmb, E, lattice, \
            basis, ImpHam, H_energy, mu, **kwargs)
    log.debug(1, "impurity generalized density matrix:\n%s", GRhoImp)

    if Efrag is None:
        return nelec/nscsites
    else:
        if int_bath:
            raise NotImplementedError
            # explicitly take out last_dmu, avoid duplicates in args
            last_dmu = kwargs.pop("last_dmu")
            Efrag = bcs.get_E_dmet(basis, lattice, ImpHam, last_dmu, **kwargs)
        log.result("Local density matrix (impurity): alpha, beta and pairing")
        rhoA, rhoB, kappaBA = extractRdm(GRhoImp)
        log.result("%s", rhoA)
        log.result("%s", rhoB)
        log.result("%s", -kappaBA.T)
        log.result("nelec per cell (impurity) = %20.12f", nelec)
        log.result("Energy per cell (impurity) = %20.12f", Efrag)
        return GRhoImp, Efrag/nscsites, nelec/nscsites

def transformResults_new(GRhoEmb, E, lattice, basis, ImpHam, H_energy, \
        last_dmu, mu, int_bath=False, **kwargs):
    log.warn("transformResults_new is being deprecated.\nUse transformResults instead.")
    kwargs["last_dmu"] = last_dmu 
    return transformResults(GRhoEmb, E, lattice, basis, ImpHam, H_energy, \
        mu, int_bath=int_bath, **kwargs)

Hubbard.transformResults = lambda GRhoEmb, E, basis, ImpHam, H_energy: \
      transformResults(GRhoEmb, E, None, basis, ImpHam, H_energy, 0.)

def ConstructImpHam(Lat, GRho, v, mu, matching=True, local=True, **kwargs):
    log.result("Making embedding basis")
    basis = bcs.embBasis(Lat, GRho, local=local, **kwargs)
    if matching:
        log.result("Rotate bath orbitals to match alpha and beta basis")
        nbasis = basis.shape[-1]
        if local:
            basis[:, :, :, nbasis//2:] = basisMatching(basis[:, :, :, nbasis//2:])
        else:
            basis = basisMatching(basis)
    log.result("Constructing impurity Hamiltonian")
    ImpHam, (H1e, H0e) = bcs.embHam(Lat, basis, v, mu, local=local, **kwargs)
    return ImpHam, (H1e, H0e), basis

def apply_dmu(lattice, ImpHam, basis, dmu):
    nscsites = lattice.nscsites
    nbasis = basis.shape[-1]
    tempCD, tempCC, tempH0 = transform_imp(basis, lattice, dmu * np.eye(nscsites))
    ImpHam.H1["cd"] -= tempCD
    ImpHam.H1["cc"] -= tempCC
    ImpHam.H0 -= tempH0
    return ImpHam

Hubbard.apply_dmu = apply_dmu

def AFInitGuess(ImpSize, U, Filling, polar=None, rand=0.01, subA=None, \
        subB=None, bogo_res=False):
    return Hubbard.AFInitGuess(ImpSize, U, Filling, polar, True, rand, \
            subA=subA, subB=subB, bogo_res=bogo_res)

def get_tiled_vcor(vcor_small, imp_size_small, imp_size_big, rand=0.0, U_Filling=None):
    import itertools as it
    from libdmet.system.lattice import SquareLattice
    Lat = SquareLattice(*(imp_size_big + imp_size_small))
    cell_dict = Lat.celldict 
    nscsites_big = np.prod(imp_size_big) 
    num_cells = len(Lat.cells)
    sites = np.array(list(it.product(*map(range, imp_size_big))))
    
    # compute idx of each small cell basis in the big cell
    cell_idx = [[] for i in range(num_cells)]
    for i, site_i in enumerate(sites):
        cell_idx[cell_dict[tuple(np.floor(site_i / imp_size_small).astype(int))]].append(i)
    
    # assign the vcor_small to correct place of vcor_big
    vcor_mat_small = vcor_small.get()
    vcor_mat_big = np.zeros((3, nscsites_big, nscsites_big), dtype = vcor_mat_small.dtype)
    vcor_mat_big[2] = (np.random.rand(nscsites_big, nscsites_big) - 0.5) * rand

    for i, cell_i in enumerate(cell_idx):
        idx = np.ix_(cell_i, cell_i)
        vcor_mat_big[0][idx] = vcor_mat_small[0]
        vcor_mat_big[1][idx] = vcor_mat_small[1]
        vcor_mat_big[2][idx] = vcor_mat_small[2]

    if U_Filling is not None:
        log.info("use default U, Filling to generate diagonal elements of vcor")
        U, Filling = U_Filling
    else:
        U, Filling = 0.0, 0.5
    
    vcor_big = AFInitGuess(imp_size_big, U, Filling, rand = 0.0)
    if U_Filling is not None:
        vcor_mat_big_default = vcor_big.get()
        np.fill_diagonal(vcor_mat_big[0], vcor_mat_big_default[0].diagonal())
        np.fill_diagonal(vcor_mat_big[1], vcor_mat_big_default[1].diagonal())
    vcor_big.assign(vcor_mat_big)
    return vcor_big

def restart_from_dmet_iter(vcor0, f_name = './dmet_iter.npy'):
    Mu, last_dmu, vcor_param = np.load(f_name)
    vcor0.update(vcor_param)
    return vcor0, Mu, last_dmu

def restart_from_hdf5():
    pass

def restart_mu_record():
    pass

addDiag = bcs.addDiag

FitVcor = bcs.FitVcorTwoStep

foldRho = bcs.foldRho

foldRho_k = bcs.foldRho_k


# **********************************************************
# test functions
# **********************************************************

def transformResults_QC(GRhoEmb_list, E_list, lattice, basis_list, ImpHam_list, H_energy_list, dmu):
    GRhoImp_list = []
    Efrag_list = []
    nelec_list = []
    for GRhoEmb, E, basis, ImpHam, H_energy \
            in zip(GRhoEmb_list, E_list, basis_list, ImpHam_list, H_energy_list):
        nscsites = basis.shape[-2] // 2
        GRhoImp, Efrag, nelec = bcs.transformResults(GRhoEmb, E, lattice, \
                basis, ImpHam, H_energy, dmu)
        log.debug(1, "impurity generalized density matrix:\n%s", GRhoImp)

        if Efrag is None:
            nelec_list.append(nelec)
            #return nelec/nscsites
        else:
            log.result("Local density matrix (impurity): alpha, beta and pairing")
            rhoA, rhoB, kappaBA = extractRdm(GRhoImp)
            log.result("%s", rhoA)
            log.result("%s", rhoB)
            log.result("%s", -kappaBA.T)
            log.result("nelec (impurity) = %20.12f", nelec)
            log.result("Energy (impurity) = %20.12f", Efrag)

            GRhoImp_list.append(GRhoImp)
            Efrag_list.append(Efrag)
            nelec_list.append(nelec)

            #return GRhoImp, Efrag/nscsites, nelec/nscsites
    if Efrag is None:
        return nelec_list
    else:
        return GRhoImp_list, Efrag_list, nelec_list

def transformResults_new_QC(GRhoEmb_list, E_list, lattice, basis_list, ImpHam_list, H_energy_list, last_dmu, Mu):
    GRhoImp_list = []
    Efrag_list = []
    nelec_list = []
    for GRhoEmb, E, basis, ImpHam, H_energy \
            in zip(GRhoEmb_list, E_list, basis_list, ImpHam_list, H_energy_list):
        nscsites = basis.shape[-2] // 2
        #GRhoImp, Efrag, nelec = bcs.transformResults(GRhoEmb, E, lattice, \
        #        basis, ImpHam, H_energy, dmu)
        GRhoImp, Efrag, nelec = bcs.transformResults_new(GRhoEmb, E, lattice, \
                basis, ImpHam, H_energy, last_dmu, Mu)
        log.debug(1, "impurity generalized density matrix:\n%s", GRhoImp)

        if Efrag is None:
            nelec_list.append(nelec)
        else:
            log.result("Local density matrix (impurity): alpha, beta and pairing")
            rhoA, rhoB, kappaBA = extractRdm(GRhoImp)
            log.result("%s", rhoA)
            log.result("%s", rhoB)
            log.result("%s", -kappaBA.T)
            log.result("nelec (impurity) = %20.12f", nelec)
            log.result("Energy (impurity) = %20.12f", Efrag)

            GRhoImp_list.append(GRhoImp)
            Efrag_list.append(Efrag)
            nelec_list.append(nelec)
    
    if Efrag is None:
        return nelec_list
    else:
        return GRhoImp_list, Efrag_list, nelec_list

Hubbard.transformResults_QC = lambda GRhoEmb_list, E_list, basis_list, ImpHam_list, H_energy_list: \
      transformResults_QC(GRhoEmb_list, E_list, None, basis_list, ImpHam_list, H_energy_list, 0.)


if __name__ == '__main__':
    imp_size_big = (2, 4)
    imp_size_small = (2, 2)
    np.set_printoptions(4, linewidth = 1000, suppress = True)
    vcor = AFInitGuess(imp_size_small, 8.0, 0.5, rand = 0.0)
    print (get_tiled_vcor(vcor, imp_size_small, imp_size_big, rand = 0.001).get())

