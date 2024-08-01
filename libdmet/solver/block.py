#! /usr/bin/env python

"""
Interface for Block, Stackblock and Block2.

Author:
    Bo-Xiao Zheng
    Zhi-Hao Cui
    Huanchen Zhai
"""

import os
import subprocess as sub
from tempfile import mkdtemp
from copy import deepcopy
import numpy as np
from libdmet.utils import logger as log
from libdmet.system import integral
from libdmet.utils.misc import grep, readlines_find, Iterable

try:
    from libdmet.solver import settings
except ImportError:
    import sys
    sys.stderr.write('settings.py not found.  Please create %s\n'
                     % os.path.join(os.path.dirname(__file__), 'settings.py'))
    raise ImportError('settings.py not found')

class Schedule(object):
    def __init__(self, maxiter=35, sweeptol=1e-6, sweep_per_M=5):
        self.initialized = False
        self.twodot_to_onedot = None
        self.maxiter = maxiter
        self.sweeptol = sweeptol
        self.sweep_per_M = sweep_per_M

    def gen_initial(self, minM, maxM, sweep_per_M=None):
        if sweep_per_M is None:
            sweep_per_M = self.sweep_per_M
        defaultM = [250, 400, 800, 1500, 2500, 3500, 5000]
        log.debug(1, "Generate default schedule with startM = %d maxM = %d, maxiter = %d", 
                  minM, maxM, self.maxiter)

        self.arrayM = [minM] + [M for M in defaultM if M > minM and M < maxM] + [maxM]
        self.arraySweep = list(range(0, sweep_per_M * len(self.arrayM), sweep_per_M))
        self.arrayTol = [min(1e-4, self.sweeptol * 0.1 * 10.**i) for i in range(len(self.arrayM))][::-1]
        self.arrayNoise = deepcopy(self.arrayTol)

        self.arrayM.append(maxM)
        self.arraySweep.append(self.arraySweep[-1] + sweep_per_M)
        self.arrayTol.append(self.arrayTol[-1])
        self.arrayNoise.append(0.0)
        self.arrayNoise = np.maximum(np.asarray(self.arrayNoise) * 10.0, 1e-5)
        self.arrayNoise[-1] = 0.0
        self.arrayTol[:-1] = np.maximum(self.arrayTol[:-1], 1e-6)

        self.twodot_to_onedot = self.arraySweep[-1] + sweep_per_M
        if self.twodot_to_onedot + sweep_per_M > self.maxiter:
            log.warning("only %d onedot iterations\nmodify maxiter to %d", 
                        self.maxiter - self.twodot_to_onedot, self.twodot_to_onedot + sweep_per_M)
            self.maxiter = self.twodot_to_onedot + sweep_per_M

        log.debug(2, "bond dimension  " + " %8d" * len(self.arrayM), *self.arrayM)
        log.debug(2, "at sweeps       " + " %8d" * len(self.arraySweep), *self.arraySweep)
        log.debug(2, "Davidson tols   " + " %8.0e" * len(self.arrayTol), *self.arrayTol)
        log.debug(2, "add noise       " + " %8.0e" * len(self.arrayNoise), *self.arrayNoise)
        log.debug(2, "twodot_to_onedot %8d", self.twodot_to_onedot)
        log.debug(2, "maxiter          %8d", self.maxiter)

        self.initialized = True

    def gen_restart(self, M):
        log.debug(1, "Generate default schedule with restart calculation M = %d, "
                  "maxiter = %d", M, self.maxiter)
        self.arrayM     = [M, M, M]
        self.arraySweep = [0, 1, 3]
        self.arrayTol   = [self.sweeptol, self.sweeptol * 0.1, self.sweeptol * 0.1]
        self.arrayNoise = [self.sweeptol, self.sweeptol * 0.1, 0.0]

        self.twodot_to_onedot = self.arraySweep[-1] + 3
        if self.twodot_to_onedot + 3 > self.maxiter:
            log.warning("only %d onedot iterations\nmodify maxiter to %d", 
                        self.maxiter - self.twodot_to_onedot, self.twodot_to_onedot + 3)
            self.maxiter = self.twodot_to_onedot + 3

        log.debug(2, "bond dimension  " + " %8d" * len(self.arrayM), *self.arrayM)
        log.debug(2, "at sweeps       " + " %8d" * len(self.arraySweep), *self.arraySweep)
        log.debug(2, "Davidson tols   " + " %8.0e" * len(self.arrayTol), *self.arrayTol)
        log.debug(2, "add noise       " + " %8.0e" * len(self.arrayNoise), *self.arrayNoise)
        log.debug(2, "twodot_to_onedot %8d", self.twodot_to_onedot)
        log.debug(2, "maxiter          %8d", self.maxiter)

        self.initialized = True

    def gen_extrapolate(self, M):
        log.debug(1, "Generate default schedule for truncation error extrapolation M = %d", M)
        self.arrayM = [M]
        self.arraySweep = [0]
        self.arrayTol = [self.sweeptol * 0.1]
        self.arrayNoise = [0]

        log.debug(2, "bond dimension  " + " %d" * len(self.arrayM), *self.arrayM)
        log.debug(2, "at sweeps       " + " %d" * len(self.arraySweep), *self.arraySweep)
        log.debug(2, "Davidson tols   " + " %.0e" * len(self.arrayTol), *self.arrayTol)
        log.debug(2, "add noise       " + " %.0e" * len(self.arrayNoise), *self.arrayNoise)

        self.twodot_to_onedot = 0
        self.maxiter = 2

        log.debug(2, "twodot_to_onedot %d", self.twodot_to_onedot)
        log.debug(2, "maxiter          %d", self.maxiter)

        self.initialized = True

    def gen_custom(self, arrayM, arraySweep, arrayTol, arrayNoise, twodot_to_onedot=None):
        log.debug(1, "Generate custom schedule")
        nstep = len(arrayM)
        log.eassert(len(arraySweep) == nstep and len(arrayTol) == nstep and 
                    len(arrayNoise) == nstep, "The lengths of input arrays are not consistent.")

        self.arrayM, self.arraySweep, self.arrayTol, self.arrayNoise = \
            arrayM, arraySweep, arrayTol, arrayNoise

        log.debug(2, "bond dimension  " + " %d" * len(self.arrayM), *self.arrayM)
        log.debug(2, "at sweeps       " + " %d" * len(self.arraySweep), *self.arraySweep)
        log.debug(2, "Davidson tols   " + " %.0e" * len(self.arrayTol), *self.arrayTol)
        log.debug(2, "add noise       " + " %.0e" * len(self.arrayNoise), *self.arrayNoise)

        if twodot_to_onedot is None:
            self.twodot_to_onedot = self.arraySweep[-1] + 2
        else:
            self.twodot_to_onedot = twodot_to_onedot

        log.debug(2, "twodot_to_onedot %d", self.twodot_to_onedot)
        log.debug(2, "maxiter          %d", self.maxiter)

        if self.arraySweep[-1] + 2 > self.maxiter:
            log.warning("maxiter smaller than scheduled number of sweeps\nmodify maxiter to %d", 
                        self.arraySweep[-1]+2)
            self.maxiter = self.arraySweep[-1] + 2
        self.initialized = True

    def gen_fixwave(self, M):
        log.debug(1, "Generate default schedule for fixed wavefunction calculation, M = %d", M)
        self.arrayM = [M]
        self.arraySweep = [0]
        self.arrayTol = [1e6]
        self.arrayNoise = [0]

        log.debug(2, "bond dimension  " + " %d" * len(self.arrayM), *self.arrayM)
        log.debug(2, "at sweeps       " + " %d" * len(self.arraySweep), *self.arraySweep)
        log.debug(2, "Davidson tols   " + " %.0e" * len(self.arrayTol), *self.arrayTol)
        log.debug(2, "add noise       " + " %.0e" * len(self.arrayNoise), *self.arrayNoise)

        self.twodot_to_onedot = 0
        self.maxiter = 1

        log.debug(2, "twodot_to_onedot %d", self.twodot_to_onedot)
        log.debug(2, "maxiter          %d", self.maxiter)

        self.initialized = True

    def get_schedule(self):
        log.eassert(self.initialized, "DMRG schedule has not been generated.")
        text = ["", "schedule"]
        nstep = len(self.arrayM)
        text += ["%d %d %.0e %.0e" % (self.arraySweep[n], self.arrayM[n], 
                 self.arrayTol[n], self.arrayNoise[n]) for n in range(nstep)]
        text.append("end")
        text.append("")
        text.append("maxiter %d" % self.maxiter)
        if self.twodot_to_onedot <= 0:
            text.append("onedot")
        elif self.twodot_to_onedot >= self.maxiter:
            text.append("twodot")
        else:
            text.append("twodot_to_onedot %d" % self.twodot_to_onedot)
        text.append("sweep_tol %.0e" % self.sweeptol)
        text.append("")
        text = "\n".join(text)
        log.debug(2, "Generated schedule in configuration file")
        log.debug(1, text)
        return text

def read1pdm(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    nsites = int(lines[0])
    pdm = np.zeros((nsites, nsites))

    for line in lines[1:]:
        tokens = line.split()
        pdm[int(tokens[0]), int(tokens[1])] = float(tokens[2])
    return pdm

def read2pdm(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    nsites = int(lines[0])
    pdm = np.zeros((nsites, nsites, nsites, nsites))

    for line in lines[1:]:
        tokens = line.split()
        # rdm2_block[i, j, k, l] = <i+ j+ k l>
        # rdm2_pyscf[i, j, k, l] = <i+ k+ l j>
        # k -> j, l -> k, j -> l
        # rdm2_pyscf[i, l, j, k] = <i+ j+ k l> = rdm2_block[i, j, k, l]
        # rdm2_pyscf = rdm2_block.transpose(0, 3, 1, 2)
        pdm[int(tokens[0]), int(tokens[3]), int(tokens[1]), int(tokens[2])] = float(tokens[4])
    return pdm

def read2pdm_bcs(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    nsites = int(lines[0])
    pdm = np.zeros((nsites, nsites, nsites, nsites))

    for line in lines[1:]:
        tokens = line.split()
        pdm[int(tokens[0]), int(tokens[1]), int(tokens[2]), int(tokens[3])] = float(tokens[4])
    return pdm

def read1pdm_bin(filename, norb, raw_data=False):
    """
    Read spin orbital rdm1 from binary file.

    Args:
        filename: file name, should be *.bin
        norb: number of spatial orbitals.
        raw_data: if True, will return (nso, nso) array.

    Returns:
        rdm1: (2, norb, norb) if raw_data = False
    """
    log.info("Block: read 1pdm by bin")
    onepdm_size = (norb*2)**2 * 8

    with open(filename, 'rb') as f:
        temp_bin = f.read()[-onepdm_size:]
    temp = np.frombuffer(temp_bin, dtype=np.float64).reshape(norb*2, norb*2)
    
    if raw_data:
        rdm1 = temp
    else:
        rdm1 = np.empty((2, norb, norb))
        rdm1[0] = temp[ ::2,  ::2]
        rdm1[1] = temp[1::2, 1::2]
    return rdm1

def read2pdm_bin(filename, norb, raw_data=False):
    """
    Read spin orbital rdm2 from binary file.

    Args:
        filename: file name, should be *.bin
        norb: number of spatial orbitals.
        raw_data: if True, will return (nso, nso, nso, nso) array.

    Returns:
        rdm2: (3, norb, norb, norb, norb), aa, bb, ab order if raw_data = False
    """
    log.info("Block: read 2pdm by bin")
    twopdm_size = (norb*2)**4 * 8

    with open(filename, 'rb') as f:
        temp_bin = f.read()[-twopdm_size:]
    temp = np.frombuffer(temp_bin, dtype=np.float64).reshape(norb*2, norb*2, norb*2, norb*2)
    temp = temp.transpose(0, 3, 1, 2) # 'pqrs -> psqr'
    if raw_data:
        rdm2 = temp
    else:
        rdm2 = np.empty((3, norb, norb, norb, norb))
        rdm2[0] = temp[ ::2,  ::2,  ::2,  ::2]  # aa
        rdm2[1] = temp[1::2, 1::2, 1::2, 1::2]  # bb
        rdm2[2] = temp[ ::2,  ::2, 1::2, 1::2]  # ab
    return rdm2

class Block(object):
    """
    Interface to Block
    """
    execPath = settings.BLOCKPATH
    nproc = 1
    nnode = 1
    nthread = 1
    intFormat = "FCIDUMP"
    reorder = True

    dyn_corr_method = None
    use_general_spin = False
    big_site = False
    casinfo = None

    # these should not be changed
    basicFiles = ["dmrg.conf.*", "FCIDUMP"]
    restartFiles = ["RestartReorder.dat", "Rotation*", "StateInfo*", "statefile*", "wave*"]
    tempFiles = ["Spin*", "Overlap*", "dmrg.e", "spatial*", "onepdm.*", "twopdm.*", "pairmat.*", 
                 "dmrg.out.*", "RI*"]
    #env_slurm = "SLURM_JOBID" in os.environ
    env_slurm = False
    name = "Block"
    
    @property
    def mpipernode(self):
        return ["mpirun", "--bind-to", "core", "--map-by", 
                "ppr:%s:node:pe=%s"%(self.nproc, self.nthread)]
    
    def set_nproc(self, nproc, nnode=1):
        self.nproc = nproc
        self.nnode = nnode
        self.nthread = 1
        log.info("Block interface  running with %d nodes,"
                 " %d processes per node, %d threads per process", 
                 self.nnode, self.nproc, self.nthread)
        log.info("Block running on nodes:\n%s", 
                 sub.check_output(self.mpipernode + ["hostname"])\
                 .decode('utf-8').replace("\n", "\t"))

    def __init__(self):
        self.sys_initialized = False
        self.schedule_initialized = False
        self.integral_initialized = False
        self.optimized = False
        self.count = 0

        self.warmup_method = "local_4site"
        self.outputlevel = 0
        self.restart = False
        self.use_global_scratch = True
        self.fixwave = False
        self.hf_occ = 'integral'

        log.debug(0, "Using %s version %s", type(self).name, type(self).execPath)

    def createTmp(self, tmp="./tmp", shared=None):
        sub.check_call(["mkdir", "-p", tmp])
        self.tmpDir = mkdtemp(prefix=type(self).name, dir=tmp)
        log.info("%s working dir %s", type(self).name, self.tmpDir)
        if type(self).nnode > 1:
            log.eassert(shared is not None, "when running on multiple nodes," 
                        " a shared tmporary folder is required")
            sub.check_call(["mkdir", "-p", shared])
            self.tmpShared = mkdtemp(prefix=type(self).name, dir=shared)
            sub.check_call(self.mpipernode + ["mkdir", "-p", self.tmpDir])
            log.info("%s shared dir %s", type(self).name, self.tmpShared)

    def set_system(self, nelec, spin, spinAdapted, bogoliubov, spinRestricted):
        self.nelec = nelec
        self.spin = spin
        log.fassert(not (spinAdapted and bogoliubov), 
                    "Bogoliubov calculation with spin adaption is not implemented")
        self.spinAdapted = spinAdapted
        self.spinRestricted = spinRestricted
        self.bogoliubov = bogoliubov
        self.sys_initialized = True

    def set_integral(self, *args):
        log.eassert(self.sys_initialized, "set_integral() should be used after"
                    " initializing set_system()")
        if len(args) == 1:
            # a single integral object
            self.integral = args[0]
        elif len(args) == 4:
            # norb, H0, H1, H2
            self.integral = integral.Integral(args[0], self.spinRestricted, 
                                              self.bogoliubov, *args[1:])
        else:
            log.error("input either an integral object, or (norb, H0, H1, H2)")
        self.integral_initialized = True

    def set_schedule(self, schedule):
        self.schedule = schedule
        self.schedule_initialized = True

    def write_conf(self, f):
        f.write("nelec %d\n" % self.nelec)
        if self.use_general_spin:
            f.write("spin 0\n")
        else:
            f.write("spin %d\n" % self.spin)
        if isinstance(self.hf_occ, str):
            f.write("hf_occ %s\n" %(self.hf_occ))
        else:
            f.write("hf_occ ")
            for o in self.hf_occ:
                f.write("%s " %(o))
            f.write("\n")
        f.write(self.schedule.get_schedule())
        f.write("orbitals %s\n" % os.path.join(self.tmpDir, self.intFormat))
        f.write("warmup %s\n" % self.warmup_method)
        f.write("nroots 1\n")
        f.write("outputlevel %d\n" % self.outputlevel)
        f.write("prefix %s\n" % self.tmpDir)
        if self.fixwave:
            f.write("fullrestart\n")
            self.fixwave = False
        else:
            if self.restart or self.optimized:
                f.write("fullrestart\n")
        if self.bogoliubov:
            f.write("bogoliubov\n")
        if not self.spinAdapted:
            f.write("nonspinadapted\n")
        if not Block.reorder:
            f.write("noreorder\n")

        # ZHC NOTE MRCI
        if self.dyn_corr_method is not None:
            f.write("%s %d %d %d\n"%(self.dyn_corr_method, *self.casinfo))
        if self.use_general_spin:
            f.write("use_general_spin\n")
        if self.big_site:
            f.write("big_site fock\n")
            assert not self.spinAdapted
            assert not self.use_general_spin

    def copy_restartfile(self, src, cleanup=True):
        files = type(self).restartFiles
        if type(self).nnode == 1:
            startPath = self.tmpDir
        else:
            startPath = self.tmpShared
        for f in files:
            sub.check_call(self.mpipernode + ["cp", os.path.join(src, f), startPath])
        if cleanup:
            sub.check_call(["rm", "-rf", src])
        self.restart = True

    def save_restartfile(self, des, cleanup=True):
        # the des has to be created before calling this method
        # recommanded using mkdtemp(prefix = "BLOCK_RESTART", dir = path_to_storage)
        files = type(self).restartFiles
        for f in files:
            sub.check_call(["cp", os.path.join(self.tmpDir, f), des])
        if cleanup:
            self.cleanup()

    def broadcast(self):
        files = type(self).basicFiles
        if self.restart and not self.optimized:
            files += type(self).restartFiles

        for f in files:
            if not self.use_global_scratch:
                sub.check_call(self.mpipernode + ["cp", 
                    os.path.join(self.tmpShared, f), self.tmpDir])
            else:
                sub.check_call(["cp", 
                    os.path.join(self.tmpShared, f), self.tmpDir])


    def callBlock(self):
        outputfile = os.path.join(self.tmpDir, "dmrg.out.%03d" % self.count)
        log.info("%s call No. %d", type(self).name, self.count)
        log.debug(0, "Written to file %s", outputfile)
        with open(outputfile, "w", buffering=1) as f:
            if type(self).env_slurm:
                log.debug(1, "slurm environment, srun used.")
                cmd = ["srun", os.path.join(type(self).execPath, "block.spin_adapted"), 
                       os.path.join(self.tmpDir, "dmrg.conf.%03d" % self.count)]
            else:
                log.debug(1, "normal environment, mpirun used.")
                cmd = [*self.mpipernode, 
                       os.path.join(type(self).execPath, "block.spin_adapted"),
                       os.path.join(self.tmpDir, "dmrg.conf.%03d" % self.count)]
            log.debug(1, " ".join(cmd))
            sub.check_call(cmd, stdout=f)

        log.result("%s sweep summary", type(self).name)
        log.result(grep("Sweep Energy", outputfile))
        self.count += 1
        return outputfile

    def callOH(self):
        outputfile = os.path.join(self.tmpDir, "dmrg.out.%03d" % self.count)
        log.info("OH call No. %d", self.count)
        log.debug(0, "Written to file %s", outputfile)
        with open(outputfile, "w", buffering=1) as f:
            if type(self).env_slurm:
                cmd = ["srun", 
                       os.path.join(type(self).execPath, "OH"), os.path.join(self.tmpDir,
                       "dmrg.conf.%03d" % self.count)]
            else:
                cmd = [*self.mpipernode, 
                       os.path.join(type(self).execPath, "OH"), os.path.join(self.tmpDir, 
                       "dmrg.conf.%03d" % self.count)]
            log.debug(1, " ".join(cmd))
            sub.check_call(cmd, stdout=f)
        self.count += 1

    def extractE(self, text):
        results = []
        lines = [s.split() for s in text.split('\n')[-2:]]
        keys = ["Weight"]
        for key in keys:
            place = [tokens.index(key) for tokens in lines]
            results.append(np.average([float(tokens[idx+2]) for tokens, idx in zip(lines, place)]))

        lines = [s.split() for s in text.split('\n')[-1:]]
        keys = ["Energy"]
        for key in keys:
            place = [tokens.index(key) for tokens in lines]
            results.append(np.average([float(tokens[idx+2]) for tokens, idx in zip(lines, place)]))
        return tuple(results)

    def onepdm(self):
        norb = self.integral.norb
        if self.spinRestricted:
            rho = read1pdm(os.path.join(self.tmpDir, "/node0/spatial_onepdm.0.0.txt")) * 0.5
            rho = rho.reshape((1, norb, norb))
        else:
            rho0 = read1pdm(os.path.join(self.tmpDir, "/node0/onepdm.0.0.txt"))
            rho = np.empty((2, norb, norb))
            rho[0] = rho0[::2, ::2]
            rho[1] = rho0[1::2, 1::2]
        if self.bogoliubov:
            kappa = read1pdm(os.path.join(self.tmpDir, "/node0/spatial_pairmat.0.0.txt"))
            if self.spinRestricted:
                kappa = (kappa + kappa.T) * 0.5
            GRho = np.zeros((norb*2, norb*2))
            GRho[:norb, :norb] = rho[0]
            GRho[norb:, :norb] = -kappa.T
            GRho[:norb, norb:] = -kappa
            if self.spinRestricted:
                GRho[norb:, norb:] = np.eye(norb) - rho[0]
            else:
                GRho[norb:, norb:] = np.eye(norb) - rho[1]
            return GRho
        else:
            return rho

    def twopdm(self, computed=False):
        log.eassert(self.optimized, "2pdm is computed using restart")
        #log.eassert(not self.bogoliubov, \
        #        "2pdm with non particle number conservation is not implemented in BLOCK")

        if not computed:
            log.debug(0, "Run %s with restart_twopdm", type(self).name)

            if type(self).nnode == 1:
                startPath = self.tmpDir
            else:
                startPath = self.tmpShared

            # copy configure file and add a line
            sub.check_call(["cp", os.path.join(startPath, "dmrg.conf.%03d" % (self.count-1)),
                            os.path.join(startPath, "dmrg.conf.%03d" % self.count)])

            with open(os.path.join(startPath, "dmrg.conf.%03d" % self.count), "a") as f:
                f.write("restart_twopdm\n")

            if type(self).nnode > 1:
                self.broadcast()

            self.callBlock()

        norb = self.integral.norb
        if self.spinRestricted:
            log.eassert(not self.bogoliubov, "2pdm with Bogoliubov Hamiltonian is"
                        " only implemented for non spinadapted case")
            # read spatial twopdm
            gamma0 = read2pdm(os.path.join(self.tmpDir, "spatial_twopdm.0.0.txt")) * 0.5
            # gamma_ijkl=0.25*sum_{s,t}<c_is c_jt d_kt d_ls>
            gamma0 = gamma0.reshape((1, norb, norb, norb, norb))
        else:
            temp = read2pdm(os.path.join(self.tmpDir, "twopdm.0.0.txt"))
            gamma0 = np.empty((3, norb, norb, norb, norb))
            gamma0[0] = temp[::2,::2,::2,::2] # alpha-alpha
            gamma0[1] = temp[1::2,1::2,1::2,1::2] # beta-beta
            gamma0[2] = temp[::2,::2,1::2,1::2] # alpha-beta
        if self.bogoliubov:
            temp = read2pdm_bcs(os.path.join(self.tmpDir, "cccdpdm.0.0.txt"))
            gamma2 = np.empty((2, norb, norb, norb, norb))
            gamma2[0] = temp[::2, ::2, 1::2, ::2]
            gamma2[1] = temp[1::2, 1::2, ::2, 1::2]
            temp = read2pdm_bcs(os.path.join(self.tmpDir, "ccccpdm.0.0.txt"))
            gamma4 = np.empty((1, norb, norb, norb, norb))
            gamma4[0] = temp[::2, ::2, 1::2, 1::2]
            return (gamma0, gamma2, gamma4)
        else:
            return gamma0

    def just_run(self, onepdm=True, dry_run=False):
        log.debug(0, "Run %s", type(self).name)

        if type(self).nnode == 1:
            startPath = self.tmpDir
        else:
            startPath = self.tmpShared

        configFile = os.path.join(startPath, "dmrg.conf.%03d" % self.count)
        with open(configFile, "w") as f:
            self.write_conf(f)
            if onepdm:
                f.write("onepdm\n")
        
        intFile = os.path.join(startPath, self.intFormat)
        integral.dump(intFile, self.integral, type(self).intFormat)
        if type(self).nnode > 1:
            self.broadcast()

        if not dry_run:
            outputfile = self.callBlock()
            truncation, energy = self.extractE(grep("Sweep Energy", outputfile))

            if onepdm:
                return truncation, energy, self.onepdm()
            else:
                return truncation, energy, None
        else:
            return None, None, None

    def optimize(self, onepdm=True):
        log.eassert(self.sys_initialized and self.integral_initialized and self.schedule_initialized,
                    "components for optimization are not ready\nsys_init = %s\nint_init = %s\n"
                    "schedule_init = %s", self.sys_initialized, self.integral_initialized,
                    self.schedule_initialized)

        log.info("Run %s to optimize wavefunction", type(self).name)
        results = self.just_run(onepdm, dry_run=False)
        self.optimized = True
        return results

    def restart_optimize(self, onepdm=True, M=None):
        log.eassert(self.optimized, "No wavefunction available")

        if M is None:
            M = self.schedule.arrayM[-1]
        self.schedule.gen_restart(M = M)

        log.info("Run BLOCK to optimize wavefunction (restart)")
        return self.just_run(onepdm, dry_run = False)

    def extrapolate(self, Ms, onepdm=True):
        log.eassert(self.sys_initialized and self.integral_initialized, 
                    "components for optimization are not ready\nsys_init = %s\nint_init = %s",
                    self.sys_initialized, self.integral_initialized)
        results = []
        if not self.optimized or self.restart:
            self.schedule = Schedule()
            self.schedule.gen_initial(Ms[0]//2, Ms[0])
            self.schedule_initialized = True
            results.append(self.optimize(onepdm=onepdm))
        else:
            results.append(self.restart_optimize(self, onepdm=onepdm, M=Ms[0]))
        for M in Ms[1:]:
            self.schedule.gen_extrapolate(M)
            results.append(self.just_run(onepdm=onepdm, dry_run=False))

    def evaluate(self, H0, H1, H2, op="unknown operator"):
        log.eassert(self.optimized, "No wavefunction available")
        self.set_integral(self.integral.norb, H0, H1, H2)

        log.info("Run OH to evaluate expectation value of %s", op)

        if type(self).nnode == 1:
            startPath = self.tmpDir
        else:
            startPath = self.tmpShared

        # just copy configure file
        sub.check_call(["cp", os.path.join(startPath, "dmrg.conf.%03d" % (self.count-1)),
                        os.path.join(startPath, "dmrg.conf.%03d" % self.count)])
        with open(os.path.join(startPath, "dmrg.conf.%03d" % self.count), "a") as f:
            f.write("fullrestart\n")

        intFile = os.path.join(startPath, self.intFormat)
        integral.dump(intFile, self.integral, Block.intFormat)
        if type(self).nnode > 1:
            self.broadcast()
        self.callOH()

        outputfile = os.path.join(self.tmpDir, "dmrg.out.%03d" % (self.count-1))
        h = float(grep("helement", outputfile).split()[-1])
        log.debug(1, "operator evaluated: %20.12f" % h)
        return h

    def cleanup(self, keep_restart=False):
        if keep_restart:
            for filename in type(self).tempFiles:
                sub.check_call(self.mpipernode + ["rm", "-rf", os.path.join(self.tmpDir, filename)])
        else:
            sub.check_call(self.mpipernode + ["rm", "-rf", self.tmpDir])
            if type(self).nnode > 1:
                sub.check_call(["rm", "-rf", self.tmpShared])
            self.optimized = False

class StackBlock(Block):
    """
    Interface to StackBlock
    """
    
    execPath = settings.BLOCKPATH
    nproc = 1
    nnode = 1
    nthread = 28
    
    dyn_corr_method = None
    use_general_spin = False
    big_site = False
    casinfo = None

    # File names
    restartFiles = ["node0/RestartReorder.dat", "node0/Rotation*", "node0/StateInfo*", "node0/statefile*", "node0/wave*"]
    # tempFiles
    tempFiles = ["node*/Block-*", "node0/dmrg.e", "node0/spatial*", "onepdm.*", "pairmat.*", "dmrg.out.*"]
    name = "StackBlock"
    
    def set_nproc(self, nproc, nthread=28, nnode=1):
        self.nproc = nproc
        self.nnode = nnode
        self.nthread = nthread
        log.info("StackBlock interface  running with %d nodes,"
                 " %d processes per node, %d threads per process", 
                 self.nnode, self.nproc, self.nthread)
        log.info("StackBlock running on nodes:\n%s", 
                 sub.check_output(self.mpipernode + ["hostname"])\
                 .decode('utf-8').replace("\n", "\t"))

    def __init__(self):
        Block.__init__(self)
        self.outputlevel = 2
        self.mem = 80

    def write_conf(self, f):
        Block.write_conf(self, f)
        f.write("num_thrds %d\n" % type(self).nthread)
        if settings.use_prebuild:
            f.write("prebuild\n")
        #adding memory spec
        f.write("mem %s g\n"%(str(self.mem)))

    def callBlock(self):
        sub.check_call(" ".join(["export", "OMP_NUM_THREADS=%d"%self.nthread]), 
                       shell=True)
        return Block.callBlock(self)

    def callOH(self):
        # ZHC NOTE FIXME check BCS case.
        outputfile = os.path.join(self.tmpDir, "dmrg.out.%03d" % self.count)
        log.info("OH call No. %d", self.count)
        log.debug(0, "Written to file %s", outputfile)
        wavenum_file = os.path.join(self.tmpDir, "wavenum")
        fwavenum = open(wavenum_file, "w")
        fwavenum.write("0 \n")
        fwavenum.close()
        with open(outputfile, "w", buffering=1) as f:
            if type(self).env_slurm:
                cmd = ["srun", os.path.join(type(self).execPath, "OH"),
                       os.path.join(self.tmpDir, "dmrg.conf.%03d" % self.count),
                       os.path.join(self.tmpDir, "wavenum")]
            else:
                cmd = [*self.mpipernode, os.path.join(type(self).execPath, "OH"),
                       os.path.join(self.tmpDir, "dmrg.conf.%03d" % self.count),
                       os.path.join(self.tmpDir, "wavenum")]
            log.debug(1, " ".join(cmd))
            sub.check_call(cmd, stdout=f)

        self.count += 1

    def onepdm(self):
        norb = self.integral.norb
        if self.spinRestricted:
            #rho = read1pdm(os.path.join(self.tmpDir, "node0/spatial_onepdm.0.0.txt")) * 0.5
            #rho = rho.reshape((1, norb, norb))
            rho = read1pdm_bin(os.path.join(self.tmpDir, "node0/onepdm.0.0.bin"),
                               norb, raw_data=False)
            rho = rho.sum(axis=0)[None] * 0.5
        else:
            #rho0 = read1pdm(os.path.join(self.tmpDir, "node0/onepdm.0.0.txt"))
            #rho = np.empty((2, norb, norb))
            #rho[0] = rho0[::2, ::2]
            #rho[1] = rho0[1::2, 1::2]
            rho = read1pdm_bin(os.path.join(self.tmpDir, "node0/onepdm.0.0.bin"),
                               norb, raw_data=False)
        if self.bogoliubov:
            kappa = read1pdm(os.path.join(self.tmpDir, "node0/spatial_pairmat.0.0.txt"))
            if self.spinRestricted:
                kappa = (kappa + kappa.T) * 0.5
            GRho = np.zeros((norb*2, norb*2))
            GRho[:norb, :norb] = rho[0]
            GRho[norb:, :norb] = -kappa.T
            GRho[:norb, norb:] = -kappa
            if self.spinRestricted:
                GRho[norb:, norb:] = np.eye(norb) - rho[0]
            else:
                GRho[norb:, norb:] = np.eye(norb) - rho[1]
            return GRho
        else:
            return rho

    def twopdm(self, computed = False):
        log.eassert(not self.bogoliubov, 
                    "2pdm with non particle number conservation is not implemented in StackBlock")
        if not computed:
            log.debug(0, "Run %s with restart_twopdm", type(self).name)

            if type(self).nnode == 1:
                startPath = self.tmpDir
            else:
                startPath = self.tmpShared

            # copy configure file and add a line
            sub.check_call(["cp", os.path.join(startPath, "dmrg.conf.%03d" % (self.count-1)),
                            os.path.join(startPath, "dmrg.conf.%03d" % self.count)])

            with open(os.path.join(startPath, "dmrg.conf.%03d" % self.count), "a") as f:
                f.write("restart_twopdm\n")

            if type(self).nnode > 1:
                self.broadcast()

            self.callBlock()

        norb = self.integral.norb

        if self.spinRestricted:
            # read spatial twopdm
            gamma0 = read2pdm(os.path.join(self.tmpDir, "node0/spatial_twopdm.0.0.txt")) * 0.5
            # gamma_ijkl=0.25*sum_{s,t}<c_is c_jt d_kt d_ls>
            gamma0 = gamma0.reshape((1, norb, norb, norb, norb))
        else:
            #temp = read2pdm(os.path.join(self.tmpDir, "node0/twopdm.0.0.txt"))
            #gamma0 = np.empty((3, norb, norb, norb, norb))
            #gamma0[0] = temp[::2,::2,::2,::2]     # alpha-alpha
            #gamma0[1] = temp[1::2,1::2,1::2,1::2] # beta-beta
            #gamma0[2] = temp[::2,::2,1::2,1::2]   # alpha-beta
            gamma0 = read2pdm_bin(os.path.join(self.tmpDir, "node0/twopdm.0.0.bin"), norb, raw_data=False)
        return gamma0

    def evaluate(self, H0, H1, H2, op="unknown operator"):
        log.eassert(self.optimized, "No wavefunction available")
        self.set_integral(self.integral.norb, H0, H1, H2)

        log.info("Run OH to evaluate expectation value of %s", op)

        if type(self).nnode == 1:
            startPath = self.tmpDir
        else:
            startPath = self.tmpShared

        # just copy configure file
        sub.check_call(["cp", os.path.join(startPath, "dmrg.conf.%03d" % (self.count-1)),
                        os.path.join(startPath, "dmrg.conf.%03d" % self.count)])
        with open(os.path.join(startPath, "dmrg.conf.%03d" % self.count), "a") as f:
            f.write("fullrestart\n")

        intFile = os.path.join(startPath, self.intFormat)
        integral.dump(intFile, self.integral, Block.intFormat)
        if type(self).nnode > 1:
            self.broadcast()
        self.callOH()

        outputfile = os.path.join(self.tmpDir, "dmrg.out.%03d" % (self.count-1))
        # ZHC NOTE check OH is correct.
        #h = float(grep("helement", outputfile).split()[-1])
        lines, line_num = readlines_find("printing hamiltonian", outputfile)
        h = float(lines[line_num[-1] + 1].split()[0])
        log.debug(1, "operator evaluated: %20.12f" % h)
        return h

class Block2(StackBlock):
    """
    Interface to block2.
    """

    execPath = settings.BLOCK2PATH
    nproc = 1
    nnode = 1
    nthread = 28
    #intFormat = "FCIDUMP"
    intFormat = "HDF5"
    reorder = False
    
    dyn_corr_method = None
    use_general_spin = False
    big_site = False
    casinfo = None

    # memory, disk
    min_mpo_mem = 'auto'
    integral_tol = 1e-12
    fp_cps_cutoff = 1e-16
    cutoff = None # SVD cutoff
    
    # init guess
    occ = None
    cbias = 0.2
    
    # single_prec
    single_prec = False
    integral_rescale = None

    # parallel
    conn_centers = None
    davidson_soft_max_iter = 50

    # TCC realted
    mps2ci = False
    sample_tol = None
    sample_reference = None
    sample_phase = None

    # these should not be changed
    basicFiles = ["dmrg.conf.*", intFormat]
    restartFiles = []
    tempFiles = []
    #env_slurm = "SLURM_JOBID" in os.environ
    env_slurm = False
    name = "Block2"
    
    @property
    def mpipernode(self):
        return ["mpirun", "--bind-to", "core", "--map-by", 
                "ppr:%s:node:pe=%s"%(self.nproc, self.nthread)]
    
    def set_nproc(self, nproc, nthread=28, nnode=1):
        self.nproc = nproc
        self.nnode = nnode
        self.nthread = nthread
        log.info("Block2 interface  running with %d nodes,"
                 " %d processes per node, %d threads per process", 
                 self.nnode, self.nproc, self.nthread)
        log.info("Block2 running on nodes:\n%s", 
                 sub.check_output(self.mpipernode + ["hostname"])\
                 .decode('utf-8').replace("\n", "\t"))
    
    def write_conf(self, f):
        # ZHC NOTE block2 does not use keyword hf_occ
        if isinstance(self.hf_occ, str) and self.hf_occ == 'integral':
            self.hf_occ = 'None'
        
        if self.single_prec:
            f.write("single_prec\n")
            assert self.integral_rescale is not None
            f.write("integral_rescale %.15g\n" % self.integral_rescale)
            self.schedule.arrayTol = np.maximum(self.schedule.arrayTol, 5e-6)
            self.schedule.arrayNoise[:-1] = np.maximum(self.schedule.arrayNoise[:-1], 5e-5)
        if self.conn_centers is not None:
            if isinstance(self.conn_centers, Iterable):
                string = " ".join(["%d"%center for center in self.conn_centers])
                f.write("conn_centers %s \n" % string)
            else:
                f.write("conn_centers auto %d \n" % self.conn_centers)
            f.write("davidson_soft_max_iter %d \n" % self.davidson_soft_max_iter)
            # parallel over site does not support onedot. use twodot:
            self.schedule.twodot_to_onedot = np.inf

        Block.write_conf(self, f)
        f.write("num_thrds %d\n" % type(self).nthread)
        f.write("mem %s g\n"%(str(self.mem)))
        f.write("min_mpo_mem %s\n" % self.min_mpo_mem)
        f.write("integral_tol %g\n" % self.integral_tol)
        f.write("fp_cps_cutoff %g\n" % self.fp_cps_cutoff)
        if self.cutoff is not None:
            f.write("cutoff %g \n" % self.cutoff)

        if self.occ is not None:
            f.write("occ ")
            for o in self.occ:
                f.write("%g " %(o))
            f.write("\n")
            f.write("cbias %s \n" % self.cbias)
        if self.mps2ci:
            f.write("restart_sample %g\n" % self.sample_tol)
            if self.sample_reference is not None:
                f.write("sample_reference %s\n" % self.sample_reference)
            if self.sample_phase is not None:
                f.write("sample_phase %s\n" % self.sample_phase)

    def mps2ci_run(self, ref_str, tol=1e-5):
        self.mps2ci = True
        self.sample_tol = tol
        self.sample_reference = "2 " + ref_str 
        if Block.reorder:
            reorderFile = os.path.join(self.tmpDir, "orbital_reorder.npy")
            reorder_orb = np.load(reorderFile)
            self.sample_phase = ""
            for idx in reorder_orb:
                self.sample_phase += "%d " % idx
            self.sample_reference = "2 " 
            for i in range(len(ref_str)):
                self.sample_reference += "%s" % ref_str[reorder_orb[i]] 
        self.just_run(onepdm=False, dry_run=False) 

    def just_run(self, onepdm=True, dry_run=False):
        log.debug(0, "Run %s", type(self).name)
        startPath = self.tmpDir

        configFile = os.path.join(startPath, "dmrg.conf.%03d" % self.count)
        with open(configFile, "w") as f:
            self.write_conf(f)
            if onepdm:
                f.write("onepdm\n")
        
        intFile = os.path.join(startPath, self.intFormat)
        integral.dump(intFile, self.integral, type(self).intFormat)

        if not dry_run:
            outputfile = self.callBlock()
            energy = np.load(os.path.join(startPath, "E_dmrg.npy"))
            truncation = np.load(os.path.join(startPath, "discarded_weights.npy"))[-1]

            if onepdm:
                return truncation, energy, self.onepdm()
            else:
                return truncation, energy, None
        else:
            return None, None, None
    
    def callBlock(self):
        outputfile = os.path.join(self.tmpDir, "dmrg.out.%03d" % self.count)
        log.info("%s call No. %d", self.name, self.count)
        log.debug(0, "Written to file %s", outputfile)
        with open(outputfile, "w", buffering=1) as f:
            #if type(self).env_slurm:
            if False:
                log.debug(1, "slurm environment, srun used.")
                cmd = ["srun", "python",
                       os.path.join(self.execPath, "block2main"), 
                       os.path.join(self.tmpDir, "dmrg.conf.%03d" % self.count)]
            else:
                log.debug(1, "normal environment, mpirun used.")
                cmd = [*self.mpipernode, "python",
                       os.path.join(self.execPath, "block2main"), 
                       os.path.join(self.tmpDir, "dmrg.conf.%03d" % self.count)]
            log.debug(1, " ".join(cmd))
            sub.check_call(cmd, stdout=f)
        bond_dims = np.load(os.path.join(self.tmpDir, "bond_dims.npy"))
        sweep_energies = np.load(os.path.join(self.tmpDir, "sweep_energies.npy"))[:, 0]
        discarded_weights = np.load(os.path.join(self.tmpDir, "discarded_weights.npy"))
        
        log.result("%8s %20s %20s", "M", "discarded_weight", "energy")
        for M, delta, E in zip(bond_dims, discarded_weights, sweep_energies):
            log.result("%8d %20.10g %20.10g", M, delta, E)
        self.count += 1
        return outputfile

    def callOH(self):
        outputfile = os.path.join(self.tmpDir, "dmrg.out.%03d" % self.count)
        log.info("OH call No. %d", self.count)
        log.debug(0, "Written to file %s", outputfile)
        with open(outputfile, "w", buffering=1) as f:
            #if type(self).env_slurm:
            if False:
                log.debug(1, "slurm environment, srun used.")
                cmd = ["srun", "python",
                       os.path.join(self.execPath, "block2main"), 
                       os.path.join(self.tmpDir, "dmrg.conf.%03d" % self.count)]
            else:
                log.debug(1, "normal environment, mpirun used.")
                cmd = [*self.mpipernode, "python",
                       os.path.join(self.execPath, "block2main"), 
                       os.path.join(self.tmpDir, "dmrg.conf.%03d" % self.count)]
            log.debug(1, " ".join(cmd))
            sub.check_call(cmd, stdout=f)
        self.count += 1

    def onepdm(self):
        rho = np.load(os.path.join(self.tmpDir, "1pdm.npy"))
        if self.spinRestricted:
            rho = np.load(os.path.join(self.tmpDir, "1pdm.npy"))
            rho = rho.sum(axis=0)[None] * 0.5
        if self.bogoliubov:
            raise NotImplementedError 
        # save rdm1
        sub.check_call(["cp", os.path.join(self.tmpDir, "1pdm.npy"),
                        os.path.join(self.tmpDir, "1pdm_%03d.npy" % self.count)])
        return rho

    def twopdm(self, computed = False):
        raise NotImplementedError 

    def evaluate(self, H0, H1, H2, op="unknown operator"):
        log.eassert(self.optimized, "No wavefunction available")
        self.set_integral(H1["cd"][0].shape[-1], H0, H1, H2)

        log.info("Run OH to evaluate expectation value of %s", op)
        startPath = self.tmpDir

        sub.check_call(["cp", os.path.join(startPath, "dmrg.conf.%03d" % (self.count-1)),
                        os.path.join(startPath, "dmrg.conf.%03d" % self.count)])
        
        with open(os.path.join(startPath, "dmrg.conf.%03d" % self.count), "r") as f:
            lines = f.readlines()
        
        with open(os.path.join(startPath, "dmrg.conf.%03d" % self.count), "a") as f:
            for line in lines:
                if "restart_oh" in line:
                    break
            else:
                f.write("restart_oh\n")

        intFile = os.path.join(startPath, self.intFormat)
        integral.dump(intFile, self.integral, self.intFormat)
        self.callOH()
        
        h = np.load(os.path.join(self.tmpDir, "E_oh.npy"))
        log.debug(1, "operator evaluated: %20.12f" % h)
        return h
