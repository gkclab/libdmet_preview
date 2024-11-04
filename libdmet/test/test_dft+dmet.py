#!/usr/bin/env python

'''
Test DFT + DMET.
'''

import os, sys
import numpy as np
import scipy.linalg as la

from pyscf import lib, fci, ao2mo
from pyscf.pbc.lib import chkfile
from pyscf.pbc import scf, gto, df, dft, cc

from libdmet.system import lattice
from libdmet.basis_transform import make_basis
from libdmet.basis_transform import eri_transform
from libdmet.utils.misc import mdot, max_abs
from libdmet.routine import slater

from libdmet.utils import logger as log
import libdmet.dmet.Hubbard as dmet

log.verbose = "DEBUG1"
np.set_printoptions(4, linewidth=1000, suppress=True)

### ************************************************************
### System settings
### ************************************************************

cell = gto.Cell()
cell.a = ''' 10.0    0.0     0.0
             0.0     10.0    0.0
             0.0     0.0     3.0 '''
cell.atom = ''' H 5.0      5.0      0.75
                H 5.0      5.0      2.25 '''
cell.basis = '321g'
cell.verbose = 4
cell.precision = 1e-10
cell.build(unit='Angstrom')

kmesh = [1, 1, 3]
Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts
exxdiv = None

### ************************************************************
### DMET settings
### ************************************************************

# system
Filling = cell.nelectron / float(Lat.nscsites*2.0)
restricted = True
bogoliubov = False
int_bath = False
nscsites = Lat.nscsites
Mu = 0.0
last_dmu = 0.0
beta = np.inf

# DMET SCF control
MaxIter = 100
u_tol = 1.0e-6
E_tol = 1.0e-6
iter_tol = 4

# DIIS
adiis = lib.diis.DIIS()
adiis.space = 4
diis_start = 4
dc = dmet.FDiisContext(adiis.space)
trace_start = 3

# solver and mu fit
FCI = dmet.impurity_solver.FCI(restricted=restricted, tol=1e-9)
solver = FCI
nelec_tol = 5.0e-6
delta = 0.01
step = 0.1
load_frecord = False

# vcor fit
imp_fit = False
emb_fit_iter = 1000 # embedding fitting
full_fit_iter = 0

# vcor initialization
vcor = dmet.vcor_zeros(restricted, bogoliubov, nscsites)

### ************************************************************
### SCF Mean-field calculation
### ************************************************************

log.section("\nSolving SCF mean-field problem\n")

gdf_fname = 'gdf_ints.h5'
gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = gdf_fname
#if not os.path.isfile(gdf_fname):
if True:
    gdf.build()

dft_dc = True
#dft_dc = False

chkfname = 'hchain.chk'
#if os.path.isfile(chkfname):
if False:
    if dft_dc:
        kmf = dft.KRKS(cell, kpts).density_fit()
        kmf.xc = 'pbe'
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit()
    kmf.exxdiv = exxdiv
    kmf.with_df = gdf
    kmf.with_df._cderi = 'gdf_ints.h5'
    kmf.conv_tol = 1e-10
    kmf.max_cycle = 300
    data = chkfile.load(chkfname, 'scf')
    kmf.__dict__.update(data)
else:
    if dft_dc:
        kmf = dft.KRKS(cell, kpts).density_fit()
        kmf.xc = 'pbe'
    else:
        kmf = scf.KRHF(cell, kpts, exxdiv=exxdiv).density_fit()
    kmf.exxdiv = exxdiv
    kmf.with_df = gdf
    kmf.with_df._cderi = 'gdf_ints.h5'
    kmf.conv_tol = 1e-10
    kmf.max_cycle = 300
    kmf.chkfile = chkfname
    kmf.kernel()

### ************************************************************
### Pre-processing, LO and subspace partition
### ************************************************************

log.section("\nPre-process, orbital localization and subspace partition\n")
C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao='minao', full_return=True)
C_ao_iao = Lat.symmetrize_lo(C_ao_iao)
C_ao_lo = C_ao_iao

ncore = 0
nval = C_ao_iao_val.shape[-1]
nvirt = cell.nao_nr() - ncore - nval
Lat.set_val_virt_core(nval, nvirt, ncore)
nemb = Lat.nao + Lat.nval
Lat.set_Ham(kmf, gdf, C_ao_lo, eri_symmetry=4)

vj_k = np.asarray(kmf.get_j())
veff_k = np.asarray(kmf.get_veff())
vxc_k = veff_k - vj_k
vxc_0 = slater.transform_h1(vxc_k, C_ao_lo[None])

# unit ERI
eri_unit = eri_transform.get_unit_eri(cell, gdf, C_ao_lo, symmetry=4)
eri_emb = dmet.unit2emb(eri_unit, nemb)
rdm1_lo_R0 = Lat.rdm1_lo_R[0, 0]
rdm1_lo_sc = Lat.expand(Lat.rdm1_lo_R)
rdm1_ao_sc = Lat.expand(Lat.k2R(Lat.rdm1_ao_k))
ovlp_ao_sc = Lat.expand(Lat.k2R(Lat.ovlp_ao_k))

scell = Lat.bigcell
C_ao_lo0 = Lat.k2R_basis(C_ao_lo)
C_ao_lo0_full = C_ao_lo0.reshape((nkpts*nscsites, -1))

if dft_dc:
    # DFT double counting:
    # J_loc
    from pyscf import scf as mol_scf
    vj_unit = mol_scf.hf.dot_eri_dm(eri_unit[0], rdm1_lo_R0, hermi=1, with_j=True, with_k=False)[0]
    vhf_j, vhf_k = kmf.get_jk()
    vhf = np.asarray(vhf_j) - 0.5 * np.asarray(vhf_k)
    fock_hf_ao_k = Lat.hcore_ao_k + vhf
    fock_hf_lo_k = make_basis.transform_h1_to_lo(fock_hf_ao_k, C_ao_lo)

    # vxc_loc
    rdm1_sc = np.dot(C_ao_lo0_full, np.dot(rdm1_lo_R0, C_ao_lo0_full.conj().T))
    kmf_sc = dft.RKS(scell).density_fit()
    kmf_sc.xc = kmf.xc
    kmf_sc.exxdiv = kmf.exxdiv
    n, exc, vxc = kmf_sc._numint.nr_rks(scell, kmf_sc.grids, kmf_sc.xc, rdm1_sc)
    vxc_unit = np.dot(np.dot(C_ao_lo0_full.conj().T, vxc), C_ao_lo0_full)

    vxc_unit = vxc_0
    JK_imp_hf = slater._get_veff(rdm1_lo_R0, eri_unit)[0]
    JK_imp = vj_unit + vxc_unit
else:
    # HF double counting:
    fock_hf_lo_k = Lat.fock_lo_k
    JK_imp = slater._get_veff(rdm1_lo_R0, eri_unit)[0]
    JK_imp_hf = JK_imp

Lat.JK_imp = JK_imp

### ************************************************************
### DMET procedure
### ************************************************************

# DMET main loop
E_old = 0.0
conv = False
history = dmet.IterHistory()
dVcor_per_ele = None
if load_frecord:
    dmet.SolveImpHam_with_fitting.load("./frecord")

for iter in range(MaxIter):
    log.section("\nDMET Iteration %d\n", iter)

    log.section("\nsolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    log.result("Mu (guess) = %20.12f", Mu)
    rho, Mu, res = dmet.RHartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)

    log.section("\nconstructing impurity problem\n")
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, matching=True, \
            int_bath=int_bath, H2_given=eri_emb)
    ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
    basis_k = Lat.R2k_basis(basis)

    log.section("\nsolving impurity problem\n")
    solver_args = {"nelec": Lat.nval * 2, "dm0": dmet.foldRho_k(res["rho_k"], basis_k)*2.0}

    rhoEmb, EnergyEmb, ImpHam, dmu = \
        dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
        solver_args=solver_args, thrnelec=nelec_tol, delta=delta, step=step)
    dmet.SolveImpHam_with_fitting.save("./frecord")
    last_dmu += dmu
    rhoImp, EnergyImp, nelecImp = \
            dmet.transformResults(rhoEmb, EnergyEmb, basis, ImpHam, H1e, \
            lattice=Lat, last_dmu=last_dmu, int_bath=int_bath, \
            solver=solver, solver_args=solver_args, add_vcor_to_E=False, vcor=vcor)
    log.result("last_dmu = %20.12f", last_dmu)

    h1eff = slater.get_H1_dmet(Lat, basis, Lat.hcore_lo_k, fock_hf_lo_k,
            JK_imp_hf)
    E1 = np.sum(h1eff * rhoEmb * (2.0 / basis.shape[0]))
    E2 = EnergyEmb - np.sum(ImpHam.H1["cd"] * rhoEmb) * (2.0 / basis.shape[0]) - ImpHam.H0
    EnergyImp = E1 + E2
    log.result("E(DMET) = %20.12f", EnergyImp)

    dump_res_iter = np.array([Mu, last_dmu, vcor.param, rhoEmb, basis, rhoImp, \
            C_ao_lo, rho, Lat.getFock(kspace=False)], dtype=object)
    np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter, allow_pickle=True)

    log.section("\nfitting correlation potential\n")
    vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, \
            vcor, beta, Filling, MaxIter1=emb_fit_iter, MaxIter2=full_fit_iter,
            method='SD', imp_fit=imp_fit, ytol=1e-8, gtol=1e-4)

    if iter >= trace_start:
        # to avoid spiral increase of vcor and mu
        log.result("Keep trace of vcor unchanged")
        ddiagV = np.average(np.diagonal(\
                (vcor_new.get()-vcor.get())[:2], 0, 1, 2))
        vcor_new = dmet.addDiag(vcor_new, -ddiagV)

    dVcor_per_ele = la.norm(vcor_new.param - vcor.param) / (len(vcor.param))
    dE = EnergyImp - E_old
    E_old = EnergyImp

    if iter >= diis_start:
        pvcor = adiis.update(vcor_new.param)
        dc.nDim = adiis.get_num_vec()
    else:
        pvcor = vcor_new.param

    dVcor_per_ele = la.norm(pvcor - vcor.param) / (len(vcor.param))
    vcor.update(pvcor)
    log.result("trace of vcor: %20.12f ", np.sum(np.diagonal((vcor.get())[:2], 0, 1, 2)))

    history.update(EnergyImp, err, nelecImp, dVcor_per_ele, dc)
    history.write_table()

    if dVcor_per_ele < u_tol and abs(dE) < E_tol and iter > iter_tol :
        conv = True
        break

# compare with FCI
scell = Lat.bigcell
scell.verbose = 4
norb = scell.nao_nr()
mf = scf.RHF(scell, exxdiv=None).density_fit()
dm0 = Lat.expand(Lat.k2R(kmf.make_rdm1())).reshape(norb, norb)
mf.kernel(dm0)

C = mf.mo_coeff
h1e = C.conj().T.dot(mf.get_hcore()).dot(C)
eri = ao2mo.incore.full(mf._eri, mf.mo_coeff)
cisolver = fci.direct_spin1.FCI()
cisolver.max_cycle = 500
cisolver.conv_tol = 1e-12
nelec_a = nelec_b = scell.nelectron // 2
e, fcivec = cisolver.kernel(h1e, eri, norb, (nelec_a, nelec_b))

rdm1 = cisolver.make_rdm1(fcivec, norb, (nelec_a,nelec_b))
C_mo_lo = mdot(C.conj().T, ovlp_ao_sc, C_ao_lo0_full)
rdm1_0 = np.dot(np.dot(C_mo_lo.conj().T, rdm1), C_mo_lo)

print ("FCI rdm1 on impurity")
print (rdm1_0 * 0.5)
print ("rdm1 diff")
print (la.norm(rdm1_0 * 0.5 - rhoImp))

print ("fci e")
print (e / nkpts)
print ("E diff")
print (EnergyImp - e / nkpts)

