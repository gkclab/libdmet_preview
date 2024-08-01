#! /usr/bin/env python

"""
CC impurity solver.

Author:
    Junjie Yang
    Zhi-Hao Cui
"""

import os
import gc
import subprocess as sub
from tempfile import mkdtemp
import h5py
import numpy as np
from functools import partial

from libdmet.utils import logger as log
from libdmet.solver import cc
from libdmet.system import integral

try:
    from libdmet.solver import settings
except ImportError:
    import sys
    sys.stderr.write('settings.py not found.  Please create %s\n'
                     % os.path.join(os.path.dirname(__file__), 'settings.py'))
    raise ImportError('settings.py not found')

einsum = partial(np.einsum, optimize=True)

class MPISolver(object):
    """
    Abstract Base Class for system call MPISolver
    """
    def __init__(self, nproc=None, nnode=None, nthread=None):
        self.nproc   = 1
        self.nnode   = 1
        self.nthread = 1

        if nproc is not None:
            self.nproc   = nproc
        if nnode is not None:
            self.nnode   = nnode
        if nthread is not None:
            self.nthread = nthread

        self.exec_path     = None
        self.int_format    = None
        self.basic_files   = None
        self.restart_files = None
        self.temp_files    = None
        self.name          = None

        self.tmp_dir       = None
        self.tmp_shared    = None
        
        self.env_slurm          = False
        self.use_global_scratch = False
        self.is_restart         = False
        self.is_optimized       = False

        self.count        = 0

    @property
    def mpi_pernode(self):
        return ["mpirun", "--bind-to", "core", "--map-by", 
                "ppr:%s:node:pe=%s"%(self.nproc, self.nthread)]

    def set_nproc(self, nproc, nnode=1):
        self.nproc   = nproc
        self.nnode   = nnode
        self.nthread = 1
        log.info("MPI4PySCF interface running with %d nodes,"
                 " %d processes per node, %d threads per process",
                 self.nnode, self.nproc, self.nthread)
        log.info("MPI4PySCF running on nodes:\n%s", 
                 sub.check_output(self.mpi_pernode + ["hostname"])\
                 .decode('utf-8').replace("\n", "\t"))

    def _call(self):
        inp_file_name = "%s.inp.%03d" %(self.name, self.count)
        out_file_name = "%s.out.%03d" %(self.name, self.count)

        exec_path     = self.exec_path
        inp_file_path = os.path.join(self.tmp_dir, inp_file_name)
        out_file_path = os.path.join(self.tmp_dir, out_file_name)

        log.info("%s call No. %d", self.name, self.count)
        log.debug(0, "Input file  = %s", out_file_path)
        log.debug(0, "Output file = %s", out_file_path)

        with open(out_file_path, "w", buffering=1) as f:
            if self.env_slurm:
                log.debug(1, "slurm environment, srun used.")
                cmd = ["srun", "python", exec_path, inp_file_path]
            else:
                log.debug(1, "normal environment, mpirun used.")
                cmd = [*self.mpi_pernode, "python", exec_path, inp_file_path]

            log.debug(1, " ".join(cmd))
            sub.check_call(cmd, stdout=f)

        self.count += 1
        return out_file_path

    def create_tmp(self, tmp="./tmp", share_dir=None):
        prefix = self.name + "-"
        sub.check_call(["mkdir", "-p", tmp])
        self.tmp_dir = mkdtemp(prefix=prefix, dir=tmp)
        log.info("%s working dir %s", self.name, self.tmp_dir)

    def broadcast(self):
        files = self.basic_files

        assert files is not None
        if self.is_restart and not self.is_optimized:
            files += self.restart_files

        for f in files:
            if not self.use_global_scratch:
                sub.check_call(self.mpi_pernode + ["cp", 
                               os.path.join(self.tmp_shared, f), self.tmp_dir])
            else:
                sub.check_call(["cp", 
                                os.path.join(self.tmp_shared, f), self.tmp_dir])

class GCCSD(MPISolver, cc.CCSD):
    """
    MPI4PySCF interface calls GCCSD solver
    """
    def __init__(self, nproc=None, nnode=None, nthread=None, 
                 restricted=False, bcs=False, ghf=True, Sz=0,    
                 tol=1e-7, tol_normt=1e-5, max_cycle=200,
                 level_shift=0.0, frozen=0, max_memory=40000,
                 scf_newton=True, diis_space=8, iterative_damping=1.0,
                 linear=False, ccd=False,
                 approx_l=False, alpha=None, beta=np.inf, 
                 tcc=False, ite=False, ovlp_tol=0.9, 
                 ao_repr=True, fcc_name='fcc', remove_h2=False):
    
        cc.CCSD.__init__(self, restricted=restricted, Sz=Sz, bcs=bcs,
                         ghf=ghf, tol=tol, tol_normt=tol_normt, 
                         max_cycle=max_cycle,
                         level_shift=level_shift,
                         frozen=frozen,
                         max_memory=max_memory, 
                         compact_rdm2=False, 
                         scf_newton=scf_newton,
                         diis_space=diis_space,
                         iterative_damping=iterative_damping,
                         linear=False,
                         approx_l=approx_l,
                         alpha=alpha,
                         beta=beta,
                         tcc=tcc,
                         ite=ite,
                         ovlp_tol=ovlp_tol
                         )
        
        self.ccd = ccd
        self.ao_repr = ao_repr
        self.fcc_name = fcc_name
        self.scf_newton = scf_newton
         
        self._restart       = None
        self._scf_max_cycle = None
        self._nelec         = None
        self._spin          = None
        self._dm0           = None

        self._init_mpi_solver = False

        MPISolver.__init__(self, nproc=nproc, nnode=nnode, nthread=nthread)
        
        self.env_slurm          = False
        self.use_global_scratch = False
        self.is_restart         = False
        self.is_optimized       = self.optimized 
        self.remove_h2 = remove_h2

        self.exec_path     = settings.MPI_GCCSD_PATH
        self.int_format    = "h5"
        self.basic_files   =  ["mpi-gcc.inp.*", "*.h5"]
        self.restart_files = []
        self.temp_files    = []
        self.name          = "mpi-gcc"
        
        self.e_tot = None
        self.rdm1  = None

        log.debug(0, "Using %s version %s", self.name, self.exec_path)

    def run(self, Ham=None, nelec=None,
            guess=None,    restart=False,
            dump_tl=False, fcc_name="fcc", calc_rdm2=False,
            **kwargs):
        """
        Main kernel function of the solver.
        """
        # ZHC NOTE: self.env_slurm is set to False
        # ZHC TODO: implement BCC
        is_bcc = kwargs.get("bcc", False)

        assert not self.env_slurm
        assert not is_bcc

        if not self._init_mpi_solver:
            self.create_tmp(tmp="./tmp", share_dir="")

        log.info("MPI4PySCF CC solver: start")
        spin = Ham.H1["cd"].shape[0]
        if spin > 1:
            log.eassert(not self.restricted, "MPI4PySCF solver: spin (%s) > 1 " 
                        "requires unrestricted", spin)

        if nelec is None:
            if self.bcs:
                nelec = Ham.norb * 2
            elif self.ghf:
                nelec = Ham.norb // 2
            else:
                raise ValueError("MPI4PySCF CC solver: nelec cannot be None.")

        nelec_a, nelec_b = (nelec + self.Sz) // 2, (nelec - self.Sz) // 2

        log.eassert(nelec_a >= 0 and nelec_b >=0, "MPI4PySCF CC solver: " 
                    "nelec_a (%s), nelec_b (%s) should >= 0", nelec_a, nelec_b)
        log.eassert(nelec_a + nelec_b == nelec, "MPI4PySCF CC solver: " 
                    "nelec_a (%s) + nelec_b (%s) should == nelec (%s)", 
                    nelec_a, nelec_b, nelec)
        
        self._nelec = nelec_a + nelec_b
        self._spin  = self.Sz
        
        # customized mo_coeff
        if "mo_energy_custom" in kwargs:
            self._mo_energy_custom = kwargs["mo_energy_custom"]
        if "mo_occ_custom" in kwargs:
            self._mo_occ_custom = kwargs["mo_occ_custom"]
        if "mo_coeff_custom" in kwargs:
            log.info("Use customized MO as reference.")
            self._mo_coeff_custom = kwargs["mo_coeff_custom"]

        start_path = self.tmp_dir
        inp_file_name = "%s.inp.%03d" %(self.name, self.count)
        inp_file_path = os.path.join(start_path, inp_file_name)

        dm0 = kwargs.get("dm0", None)
        if dm0 is not None:
            dm0_file = os.path.join(start_path, "dm0.npy")
            np.save(dm0_file, dm0)
        else:
            dm0_file = None
        self._dm0 = dm0_file
        
        umat = kwargs.get("umat", None)
        if umat is not None:
            umat_file = os.path.join(start_path, "umat.npy")
            np.save(umat_file, umat)
        else:
            umat_file = None
        self._umat = umat_file
        
        self._no_kernel = False
        self._calc_rdm2 = calc_rdm2

        self._scf_max_cycle = kwargs.get("scf_max_cycle", 100)
        self._restart       = kwargs.get("restart",     False)
        gcc_dict = self._cc_inp(start_path)
        self._gcc_dict = gcc_dict

        import pickle as p
        with open(inp_file_path, "wb") as f:
            p.dump(gcc_dict, f)

        int_file_path = gcc_dict["int_h5_file"]
        Ham.save(int_file_path)
        if self.remove_h2:
            Ham.H2 = None
            gc.collect()

        self._call()

        self.e_tot    = np.load(gcc_dict["e_file"])
        frdm = h5py.File(gcc_dict["rdm_file"], 'r')
        self.rdm1 = np.asarray(frdm["rdm1"])
        frdm.close()
        
        return self.rdm1, self.e_tot

    def _cc_inp(self, start_path):
        # ZHC NOTE here we should only keep 1 file for int and rdm to save space
        #int_file_path = os.path.join(start_path,  "int."+self.int_format)
        int_file_path = "./int." + self.int_format
        e_file        = os.path.join(start_path,    "e_gcc.npy")
        rdm_file      = "./rdm_gcc.h5"

        cc_inp_dict =  {'int_h5_file': int_file_path,

                        'nelec'      : self._nelec,

                        'verbose'       : self.verbose,
                        'spin'          : self._spin,
                        'alpha'         : self.alpha,
                        'beta'          : self.beta,
                        'scf_newton'    : self.scf_newton,
                        'max_memory'    : self.max_memory,
                        'scf_max_cycle' : self._scf_max_cycle,
                        'dm0'           : self._dm0,

                        'ccd'             : self.ccd,
                        'approx_l'        : self.approx_l,
                        'restart'         : self._restart,
                        'umat'            : self._umat,
                        'mo_coeff_custom' : self._mo_coeff_custom,
                        'mo_energy_custom': self._mo_energy_custom,
                        'mo_occ_custom'   : self._mo_occ_custom,

                        'conv_tol'       : self.conv_tol,
                        'conv_tol_normt' : self.conv_tol_normt,
                        'max_cycle'      : self.max_cycle,
                        'level_shift'    : self.level_shift,
                        'diis_space'     : self.diis_space,
                        'frozen'         : self.frozen,
                        'fcc_name'       : self.fcc_name,
                        'calc_rdm2'      : self._calc_rdm2,
                        'ao_repr'        : self.ao_repr,
                        'no_kernel'      : self._no_kernel,
                        
                        'e_file'         :e_file, 
                        'rdm_file'      :rdm_file, 
                        }
        
        log.debug(0, "MPI4PySCF GCCSD input dict:")
        for ii in cc_inp_dict:
            if isinstance(cc_inp_dict[ii], np.ndarray):
                log.debug(0, "%20s = \n%s"%(ii, cc_inp_dict[ii]))
            else:
                log.debug(0, "%20s = %s"%(ii, cc_inp_dict[ii]))
        return cc_inp_dict

    def make_rdm1(self, Ham=None, drv=None):
        if self.rdm1 is None:
            frdm = h5py.File(self._gcc_dict["rdm_file"], 'r')
            self.rdm1 = np.asarray(frdm["rdm1"])
            frdm.close()
        return self.rdm1

    def make_rdm2(self, Ham=None, ao_repr=False, drv=None, with_dm1=True):
        frdm = h5py.File(self._gcc_dict["rdm_file"], 'r')
        rdm2 = np.asarray(frdm["rdm2"])
        frdm.close()
        return rdm2

class MPIGCCDAsFCISolver(GCCSD):
    """
    MPIGCCD as FCI solver for OO-CCD.
    """
    def __init__(self, ghf=True, max_cycle=200, level_shift=0.0, conv_tol=1e-7,
                 conv_tol_normt=1e-5, diis_space=8, max_memory=120000,
                 restart=False, verbose=4, fname='mcscf', fcivec="fcc",
                 approx_l=False, fix_fcivec=False, nproc=1, nthread=1, nnode=1,
                 frozen=0, fcc_name='fcc', remove_h2=False, **kwargs):
        self.mycc = None
        self.ghf = ghf
        self.level_shift = level_shift
        self.max_cycle = max_cycle
        self.conv_tol = conv_tol
        self.conv_tol_normt = conv_tol_normt
        self.diis_space = diis_space
        self.max_memory = max_memory
        self.verbose = verbose
        self.scf_newton = False
        self.ccd = True
        self.ao_repr = False
        self.approx_l = approx_l
        self.frozen = frozen
        
        self.restart = restart
        self.fname = fname
        self.fcivec = fcivec
        self.fcc_name = fcc_name
        self.fix_fcivec = fix_fcivec
        
        self.nnode = nnode
        self.nproc = nproc
        self.nthread = nthread

        MPISolver.__init__(self, nproc=nproc, nnode=nnode, nthread=nthread)
        
        self.env_slurm          = False
        self.use_global_scratch = False

        self.exec_path     = settings.MPI_GCCSD_PATH
        self.int_format    = "h5"
        self.basic_files   =  ["mpi-gcc.inp.*", "*.h5"]
        self.restart_files = []
        self.temp_files    = []
        self.name          = "mpi-gcc"
        
        self.remove_h2 = remove_h2
        self._init_mpi_solver = False
        self.optimized = False
        log.debug(0, "Using %s version %s", self.name, self.exec_path)

    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0, **kwargs):
        if self.ghf:
            from libdmet.solver import scf as scf_hp
            
            # first get the start path
            if not self._init_mpi_solver:
                self.create_tmp(tmp="./tmp", share_dir="")
            
            start_path = self.tmp_dir
            
            inp_file_name = "%s.inp.%03d" %(self.name, self.count)
            inp_file_path = os.path.join(start_path, inp_file_name)
            
            # no scf is need
            self._no_kernel = True
            self._dm0 = None 
            self._scf_max_cycle = kwargs.get("scf_max_cycle", 100)
            
            self._nelec = sum(nelec)
            self._spin  = 0
            self._calc_rdm2 = True
            
            if self.restart and self.optimized:
                if os.path.exists("%s_u.npy"%self.fname):
                    self._umat = np.load("%s_u.npy"%self.fname)
                else:
                    self._umat = None
                self._restart = True
            else:
                self._umat = None
                self._restart = False
            
            # dump input
            gcc_dict = self._cc_inp(start_path)
            self._gcc_dict = gcc_dict
            import pickle as p
            with open(inp_file_path, "wb") as f:
                p.dump(gcc_dict, f)
            
            # dump integral
            Ham = integral.Integral(norb, True, False, ecore, {"cd": h1[None]},
                                    {"ccdd": h2[None]}, ovlp=None)
            int_file_path = gcc_dict["int_h5_file"]
            Ham.save(int_file_path)
            if self.remove_h2:
                Ham.H2 = None
                gc.collect()

            self._call()

            self.e_tot    = np.load(gcc_dict["e_file"])
            frdm = h5py.File(gcc_dict["rdm_file"], 'r')
            self.rdm1 = np.asarray(frdm["rdm1"])
            self.rdm2 = np.asarray(frdm["rdm2"])
            frdm.close()
            
            if self.restart:
                self.optimized = True
            return self.e_tot, self.fcivec
        else:
            raise NotImplementedError
    
    def make_rdm1(self, fake_ci, norb, nelec):
        if self.rdm1 is None:
            frdm = h5py.File(self._gcc_dict["rdm_file"], 'r')
            self.rdm1 = np.asarray(frdm["rdm1"])
            frdm.close()
        return self.rdm1

    def make_rdm12(self, fake_ci, norb, nelec):
        if self.rdm1 is None or self.rdm2 is None:
            frdm = h5py.File(self._gcc_dict["rdm_file"], 'r')
            self.rdm1 = np.asarray(frdm["rdm1"])
            self.rdm2 = np.asarray(frdm["rdm2"])
            frdm.close()
        return self.rdm1, self.rdm2
    
    def load_fcivec(self, fname):
        """
        need not to load here.
        """
        return self.fcivec

    def save_fcivec(self, fname):
        pass

if __name__ == '__main__':
    pass
