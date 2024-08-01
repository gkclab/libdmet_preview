#!/usr/bin/env python

"""
DMET for NiO in FM phase with gth-dzvp basis.
"""

import os, sys
import numpy as np
import scipy.linalg as la

from pyscf import lib, fci, ao2mo
from pyscf.pbc.lib import chkfile
from pyscf.pbc import scf, gto, df, dft, cc, tools

from libdmet.system import lattice
from libdmet.basis_transform import make_basis
from libdmet.basis_transform import eri_transform
from libdmet.lo.iao import reference_mol, get_labels, get_idx
from libdmet.basis_transform.make_basis import symmetrize_kmf
from libdmet.utils.misc import max_abs, mdot, kdot, read_poscar

import libdmet.utils.logger as log
import libdmet.dmet.Hubbard as dmet

log.verbose = "DEBUG1"
np.set_printoptions(4, linewidth=1000, suppress=True)

max_memory = 110000
exxdiv = None

### ************************************************************
### System settings
### ************************************************************

cell = read_poscar("./NiO-AFM-417")
cell.spin = 0
cell.basis   = 'gth-dzvp-molopt-sr'
cell.pseudo  = 'gth-pade'
cell.verbose = 5
cell.precision = 1e-12
cell.max_memory = max_memory
kmesh = [2, 2, 2]
cell.spin = 4 * np.prod(kmesh)
cell.build()

Ni3d = cell.search_ao_label("Ni 3d.*")
Ni0_3d = Ni3d[[2, 4]]
Ni1_3d = Ni3d[[2+5, 4+5]]

Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts

minao = 'gth-szv-molopt-sr'
pmol = reference_mol(cell, minao=minao)
ncore = 0
nval = pmol.nao_nr()
nvirt = cell.nao_nr() - ncore - nval
Lat.set_val_virt_core(nval, nvirt, ncore)

kmf_conv_tol = 1e-11
kmf_max_cycle = 100

gdf_fname = '../03-dmet-nio-afm/gdf_ints_222.h5'
chkfname = './nio_222.chk'

### ************************************************************
### DMET settings
### ************************************************************

# system
#Filling = [(cell.nelectron + 4) / float(Lat.nscsites*2.0),
#           (cell.nelectron - 4) / float(Lat.nscsites*2.0)]
Filling = (cell.nelectron) / float(Lat.nscsites*2.0)
restricted = False
bogoliubov = False
int_bath = True
nscsites = Lat.nscsites
Mu = 0.0
last_dmu = 0.0
#beta = np.inf
beta = 1000.0

# DMET SCF control
MaxIter = 100
u_tol = 1.0e-4
E_tol = 5.0e-5
iter_tol = 4

# DIIS
adiis = lib.diis.DIIS()
adiis.space = 4
diis_start = 4
dc = dmet.FDiisContext(adiis.space)

# solver and mu fit
ncas = nscsites + nval
nelecas = (Lat.ncore + Lat.nval) * 2
cisolver = dmet.impurity_solver.CCSD(restricted=restricted, tol=5e-7, tol_normt=5e-5, max_memory=max_memory)
solver = cisolver
nelec_tol = 1.0e-5
delta = 0.01
step = 0.1
load_frecord = False

# vcor fit
imp_fit = False
emb_fit_iter = 2000 # embedding fitting
full_fit_iter = 0
ytol = 1e-7
gtol = 1e-3
CG_check = False

### ************************************************************
### SCF Mean-field calculation
### ************************************************************

log.section("\nSolving SCF mean-field problem\n")

# idx of orbitals
aoind = cell.aoslice_by_atom()
ao_labels = cell.ao_labels()

idx_Ni0 = list(range(aoind[0][2], aoind[0][3]))
idx_Ni1 = list(range(aoind[1][2], aoind[1][3]))
idx_O0 = list(range(aoind[2][2], aoind[2][3]))
idx_O1 = list(range(aoind[3][2], aoind[3][3]))

labels, B2_labels, virt_labels = get_labels(cell, minao=minao)
iao_Ni0 = get_idx(labels, 0)
iao_Ni1 = get_idx(labels, 1)
iao_O0 = get_idx(labels, 2)
iao_O1 = get_idx(labels, 3)

iao_Ni0_val = get_idx(B2_labels, 0)
iao_Ni1_val = get_idx(B2_labels, 1)
iao_O0_val = get_idx(B2_labels, 2)
iao_O1_val = get_idx(B2_labels, 3)

iao_Ni0_virt = get_idx(virt_labels, 0, offset=len(B2_labels))
iao_Ni1_virt = get_idx(virt_labels, 1, offset=len(B2_labels))
iao_O0_virt = get_idx(virt_labels, 2, offset=len(B2_labels))
iao_O1_virt = get_idx(virt_labels, 3, offset=len(B2_labels))

iao_Ni0_3d = (5, 6, 7, 8, 9) # 3d
iao_Ni1_3d = (15, 16, 17, 18, 19) # 3d
iao_Ni0_t2g = (5, 6, 8)
iao_Ni0_eg = (7, 9)
iao_Ni1_t2g = (15, 16, 18)
iao_Ni1_eg = (17, 19)

sl_Ni0 = slice(aoind[0][2], aoind[0][3])
sl_Ni1 = slice(aoind[1][2], aoind[1][3])
sl_O0 = slice(aoind[2][2], aoind[2][3])
sl_O1 = slice(aoind[3][2], aoind[3][3])
sl = [[sl_Ni0, sl_O0], [sl_Ni1, sl_O1]]

gdf = df.GDF(cell, kpts)
gdf.mesh = np.asarray([18,18,18])
gdf._cderi_to_save = gdf_fname
gdf.auxbasis = df.aug_etb(cell, beta=2.3)
gdf.linear_dep_threshold = 0.
if not os.path.isfile(gdf_fname):
    gdf.build()

chkfname = chkfname
#if False:
if os.path.isfile(chkfname):
    kmf = scf.KUHF(cell, kpts, exxdiv=exxdiv).density_fit()
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    data = chkfile.load(chkfname, 'scf')
    kmf.__dict__.update(data)
else:
    kmf = scf.KUHF(cell, kpts, exxdiv=exxdiv).density_fit()
    kmf.with_df = gdf
    kmf.with_df._cderi = gdf_fname
    kmf.conv_tol = 1e-11
    kmf.chkfile = chkfname
    aoind = cell.aoslice_by_atom()

    dm = kmf.get_init_guess()
    dm[1] = dm[0]

    dm[0, :, Ni0_3d, Ni0_3d] *= 2.0
    dm[0, :, Ni1_3d, Ni1_3d] *= 2.0
    dm[1, :, Ni0_3d, Ni0_3d] = 0.0
    dm[1, :, Ni1_3d, Ni1_3d] = 0.0

    kmf.max_cycle = 50

    Lat.mulliken_lo_R0(Lat.k2R(dm)[:, 0], labels=cell.ao_labels())

    #from libdmet.routine import pbc_helper as pbc_hp
    #kmf = pbc_hp.smearing_(kmf, sigma=(1.0/beta))

    kmf.kernel(dm0=dm)

log.result("kmf electronic energy: %20.12f", kmf.e_tot - kmf.energy_nuc())

### ************************************************************
### Pre-processing, LO and subspace partition
### ************************************************************

log.section("\nPre-process, orbital localization and subspace partition\n")
# IAO
from libdmet.lo.iao import reference_mol, get_labels, get_idx_each
S_ao_ao = kmf.get_ovlp()
C_ao_iao, C_ao_iao_val, C_ao_iao_virt = make_basis.get_C_ao_lo_iao(Lat, kmf, minao=minao, full_return=True)
lo_labels = get_labels(cell, minao=minao)[0]

assert(nval == C_ao_iao_val.shape[-1])

# use IAO
C_ao_lo = C_ao_iao
Lat.set_Ham(kmf, gdf, C_ao_lo, eri_symmetry=4)

# vcor initialization
idx_range = list(range(ncore, ncore+nval))
vcor = dmet.VcorLocal_new(restricted, bogoliubov, nscsites, idx_range=idx_range)
vcor.update(np.zeros(vcor.length()))

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

    log.section("\nSolving mean-field problem\n")
    log.result("Vcor =\n%s", vcor.get())
    log.result("Mu (guess) = %s", Mu)

    rho, Mu, res = dmet.HartreeFock(Lat, vcor, Filling, Mu, beta=beta, ires=True, symm=True)
    rho = rho.real
    rho_k = Lat.R2k(rho)

    Lat.mulliken_lo_R0(rho[:, 0], labels=np.asarray(lo_labels))

    print ("mf rho alpha")
    dm_Ni0_3d_a = rho[0,0][np.ix_(iao_Ni0_3d, iao_Ni0_3d)]
    dm_Ni1_3d_a = rho[0,0][np.ix_(iao_Ni1_3d, iao_Ni1_3d)]
    print (dm_Ni0_3d_a)
    print (dm_Ni1_3d_a)

    print ("mf rho beta")
    dm_Ni0_3d_b = rho[1,0][np.ix_(iao_Ni0_3d, iao_Ni0_3d)]
    dm_Ni1_3d_b = rho[1,0][np.ix_(iao_Ni1_3d, iao_Ni1_3d)]
    print (dm_Ni0_3d_b)
    print (dm_Ni1_3d_b)

    log.section("\nConstructing impurity problem\n")
    ImpHam, H1e, basis = dmet.ConstructImpHam(Lat, rho, vcor, int_bath=int_bath, max_memory=max_memory)
    ImpHam = dmet.apply_dmu(Lat, ImpHam, basis, last_dmu)
    basis_k = Lat.R2k_basis(basis)

    log.section("\nSolving impurity problem\n")
    solver_args = {"nelec": (Lat.ncore+Lat.nval)*2, "dm0": dmet.foldRho_k(rho_k, basis_k)}

    rhoEmb, EnergyEmb, ImpHam, dmu = \
        dmet.SolveImpHam_with_fitting(Lat, Filling, ImpHam, basis, solver, \
        solver_args=solver_args, thrnelec=nelec_tol, \
        delta=delta, step=step)
    dmet.SolveImpHam_with_fitting.save("./frecord")
    last_dmu += dmu
    rhoImp, EnergyImp, nelecImp = \
        dmet.transformResults_new(rhoEmb, EnergyEmb, Lat, basis, ImpHam, H1e, last_dmu, \
        int_bath=int_bath, solver=solver, solver_args=solver_args,
        rebuild_veff=True)
    EnergyImp *= nscsites
    log.result("last_dmu = %20.12f", last_dmu)
    log.result("E(DMET) = %20.12f", EnergyImp)

    Lat.mulliken_lo_R0(rhoImp, labels=np.asarray(lo_labels))

    print ("alpha")
    dm_Ni0_3d_a = rhoImp[0][np.ix_(iao_Ni0_3d, iao_Ni0_3d)]
    dm_Ni1_3d_a = rhoImp[0][np.ix_(iao_Ni1_3d, iao_Ni1_3d)]
    print (dm_Ni0_3d_a)
    print (dm_Ni1_3d_a)

    print ("beta")
    dm_Ni0_3d_b = rhoImp[1][np.ix_(iao_Ni0_3d, iao_Ni0_3d)]
    dm_Ni1_3d_b = rhoImp[1][np.ix_(iao_Ni1_3d, iao_Ni1_3d)]
    print (dm_Ni0_3d_b)
    print (dm_Ni1_3d_b)

    dm_Ni0_a = rhoImp[0][np.ix_(iao_Ni0, iao_Ni0)]
    dm_Ni0_b = rhoImp[1][np.ix_(iao_Ni0, iao_Ni0)]

    print ("magnetic order")
    print (dm_Ni0_a.trace() + dm_Ni0_b.trace())
    print (dm_Ni0_a.trace() - dm_Ni0_b.trace())

    # DUMP results:
    dump_res_iter = np.array([Mu, last_dmu, vcor.param, rhoEmb, basis, rhoImp, C_ao_lo, \
            Lat.getFock(kspace=False), rho], dtype=object)
    np.save('./dmet_iter_%s.npy'%(iter), dump_res_iter)

    log.section("\nfitting correlation potential\n")
    vcor_new, err = dmet.FitVcor(rhoEmb, Lat, basis, \
            vcor, beta, Filling, MaxIter1=emb_fit_iter, MaxIter2=full_fit_iter, method='CG', \
            imp_fit=imp_fit, ytol=ytol, gtol=gtol, CG_check=CG_check)

    dVcor_per_ele = np.max(np.abs(vcor_new.param - vcor.param))
    dE = EnergyImp - E_old
    E_old = EnergyImp

    if iter >= diis_start:
        pvcor = adiis.update(vcor_new.param)
        dc.nDim = adiis.get_num_vec()
    else:
        pvcor = vcor_new.param

    dVcor_per_ele = np.max(np.abs(pvcor - vcor.param))
    vcor.update(pvcor)
    log.result("Trace of vcor: %20.12f ", np.sum(np.diagonal((vcor.get())[:2], 0, 1, 2)))

    history.update(EnergyImp, err, nelecImp, dVcor_per_ele, dc)
    history.write_table()

    if dVcor_per_ele < u_tol and abs(dE) < E_tol and iter > iter_tol :
        conv = True
        break

if conv:
    log.result("DMET converge.")
else:
    log.result("DMET does not converge.")

