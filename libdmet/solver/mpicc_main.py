#! /usr/bin/env python

"""
MPI-CC impurity solver.

Usage: mpirun -np 2 mpicc_main.py input.conf

Author:
    Junjie Yang
    Zhi-Hao Cui
"""

import sys
import h5py
import numpy as np

from libdmet.system import integral
from libdmet.solver import scf as scf_hp
from libdmet.utils import logger as log

from mpi4pyscf import cc as mpicc

def main(gcc_inp_file):
    """
    GCCSD / GCCD main.

    Args:
        gcc_inp_file: a dictionary which contains the `mpicc.GGCCSD` parameters
    """
    import pickle as p

    log.info("loading gcc setting from %s", gcc_inp_file)
    with open(gcc_inp_file, "rb") as f:
        gcc_inp_dict = p.load(f)

    # input integrals
    int_h5_file    = gcc_inp_dict.get('int_h5_file', None)
    assert int_h5_file is not None

    # input electron number
    nelec          = gcc_inp_dict.get("nelec",      None)
    assert nelec is not None

    # scf options
    verbose        = gcc_inp_dict.get("verbose",        4)
    spin           = gcc_inp_dict.get("spin",           0)
    alpha          = gcc_inp_dict.get("alpha",       None)
    beta           = gcc_inp_dict.get("beta",      np.inf)
    scf_newton     = gcc_inp_dict.get("scf_newton",  True)
    max_memory     = gcc_inp_dict.get("max_memory",  1000)
    scf_max_cycle  = gcc_inp_dict.get("scf_max_cycle", 50)
    no_kernel      = gcc_inp_dict.get("no_kernel",  False)
    dm0            = gcc_inp_dict.get("dm0",         None)
    if isinstance(dm0, str):
        dm0 = np.load(dm0)

    # cc options
    ccd            = gcc_inp_dict.get("ccd",           False)
    restart        = gcc_inp_dict.get("restart",       False) # restart
    umat           = gcc_inp_dict.get("umat",           None) # orbital rotation for old amps
    if isinstance(umat, str):
        umat = np.load(umat)
    mo_coeff_custom  = gcc_inp_dict.get("mo_coeff_custom", None)
    mo_energy_custom = gcc_inp_dict.get("mo_energy_custom", None)
    mo_occ_custom    = gcc_inp_dict.get("mo_occ_custom", None)

    approx_l       = gcc_inp_dict.get("approx_l",      False) # use t1 and t2 as lambda
    ao_repr        = gcc_inp_dict.get("ao_repr",       False)
    conv_tol       = gcc_inp_dict.get("conv_tol",       1e-6)
    conv_tol_normt = gcc_inp_dict.get("conv_tol_normt", 1e-5)
    max_cycle      = gcc_inp_dict.get("max_cycle",       200)
    level_shift    = gcc_inp_dict.get("level_shift",     0.0)
    diis_space     = gcc_inp_dict.get("diis_space",        8)
    frozen         = gcc_inp_dict.get("frozen",            0)
    fcc_name       = gcc_inp_dict.get("fcc_name",      "fcc")
    calc_rdm2      = gcc_inp_dict.get("calc_rdm2",     False)

    # output
    e_file         = gcc_inp_dict.get("e_file",       "e_%s.npy"%fcc_name)
    rdm_file       = gcc_inp_dict.get("rdm_file",    "rdm_%s.h5"%fcc_name)

    Ham = integral.load(int_h5_file)
    norb = Ham.norb
    scfsolver = scf_hp.SCF(newton_ah=scf_newton, no_kernel=no_kernel,
                           verbose=verbose)
    scfsolver.set_system(nelec, spin, False, True, max_memory=max_memory)
    scfsolver.set_integral(Ham)

    scf_conv_tol  = min(conv_tol*0.1, 1e-10)

    e_hf, rdm1_hf = scfsolver.GGHF(tol=scf_conv_tol, MaxIter=scf_max_cycle,
                                   InitGuess=dm0,
                                   alpha=alpha,
                                   beta=beta)
    mf = scfsolver.mf

    if alpha != 1.0:
        # ZHC NOTE alpha is adjusted to 1 after converge mf
        log.info("adjust mf.alpha to 1.0 for CC.")
        mf.alpha = 1.0
    if no_kernel:
        mf.mo_coeff = np.eye(norb)
        mf.mo_occ   = np.zeros(norb)
        mf.mo_occ[:nelec] = 1

    if mo_energy_custom is not None:
        mf.mo_energy = mo_energy_custom
    if mo_occ_custom is not None:
        mf.mo_occ = mo_occ_custom
    if mo_coeff_custom is not None:
        log.info("Use customized MO as reference.")
        mf.mo_coeff = mo_coeff_custom
        mf.e_tot = mf.energy_tot()

    if ccd:
        drv = mpicc.GGCCD
    else:
        drv = mpicc.GGCCSD
    mycc = drv(mf)
    mycc.conv_tol       = conv_tol
    mycc.conv_tol_normt = conv_tol_normt
    mycc.max_cycle      = max_cycle
    mycc.level_shift    = level_shift
    mycc.diis_space     = diis_space
    mycc.set(frozen=frozen)
    mycc.remove_h2 = True

    if beta < np.inf:
        # ZHC NOTE overwrite cc.nocc for smearing cases,
        # this allows that eris.fock comes from a smeared dm.
        log.info("adjust cisolver.nocc to integer for CC.")
        frozen = mycc.frozen
        if frozen is None:
            nocc = nelec
        elif isinstance(frozen, (int, np.integer)):
            nocc = nelec - frozen
        elif isinstance(frozen[0], (int, np.integer)):
            occ_idx = np.zeros(mycc.mo_occ.shape, dtype=bool)
            occ_idx[:nelec] = True
            occ_idx[list(frozen)] = False
            nocc = np.count_nonzero(occ_idx)
        else:
            raise ValueError
        mycc.nocc = nocc

    if restart:
        mycc.restore_from_h5(fname=fcc_name, umat=umat)

    e_gcc, t1_gcc, t2_gcc = mycc.kernel()
    l1_gcc, l2_gcc        = mycc.solve_lambda(approx_l=approx_l)

    mycc.save_amps(fname=fcc_name)

    if beta < np.inf:
        # ZHC NOTE modify the mo_occ since the frozen may need it in rdm
        log.info("adjust mf.mo_occ to integer for CC.")
        mycc.mo_occ = np.zeros_like(mycc.mo_occ)
        mycc.mo_occ[:nelec] = 1.0

    rdm1_gcc = mycc.make_rdm1(ao_repr=ao_repr)
    if calc_rdm2:
        rdm2_gcc = mycc.make_rdm2(ao_repr=ao_repr)

    np.save(e_file,       e_gcc)
    frdm = h5py.File(rdm_file, 'w')
    frdm["rdm1"] = np.asarray(rdm1_gcc)
    if calc_rdm2:
        frdm["rdm2"] = np.asarray(rdm2_gcc)
    frdm.close()

if __name__ == "__main__":
    gcc_inp_file = sys.argv[1]
    main(gcc_inp_file)
