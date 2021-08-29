import numpy as np
import scipy.linalg as la
import itertools as it
import os
from tempfile import mkdtemp
import copy
import subprocess as sub
from libdmet.utils import logger as log
from libdmet.utils.misc import grep

def dumpH1(filename, H1):
    # only cd terms
    spin = H1.shape[0]
    log.eassert(spin == 2, "not implemented for restricted calculations")
    nsites = H1.shape[1]
    eta = 1e-12
    nH1 = [np.sum(abs(H1[s]) > eta) for s in range(spin)]
    log.check(nH1[0] == nH1[1], "number of nonzero elements in alpha hopping matrix"
            "does not equal to that in beta hopping matrix!")
    with open(filename, "w") as f:
        f.write("%d\n" % nsites)
        for s in range(spin):
            f.write("%d\n" % nH1[s])
            for i, j in it.product(range(nsites), repeat = 2):
                if abs(H1[s,i,j]) > eta:
                    f.write("%5d %5d   ( %s, %s )\n" \
                            % (i, j, H1[s,i,j].real.__repr__(), H1[s,i,j].imag.__repr__()))
                    # we need the number to be exact

def dumpH2(filename, H2):
    if isinstance(H2, np.ndarray):
        nsites = H2.shape[1]
        U = H2[2,0,0,0,0]
        for i in range(nsites//2):
            log.eassert(H2[0,i,i,i,i] == U and H2[1,i,i,i,i] == U \
                    and H2[2,i,i,i,i] == U, \
                    "This is not Hubbard impurity Hamiltonian")
            i1 = i + nsites//2
            log.eassert(H2[0,i1,i1,i1,i1] == 0 and H2[1,i1,i1,i1,i1] == 0 \
                    and H2[2,i1,i1,i1,i1] == 0, \
                    "This is not Hubbard impurity Hamiltonian")
        nimp = nsites // 2
    else:
        U = H2["U"]
        nimp = H2["nsites"]
    
    with open(filename, "w") as f:
        f.write("%.12f # U\n" % U)
        f.write("%d # Nup\n" % nimp)
        f.write("%d # Ndn\n" % nimp)
        f.write("%d # h0flag: 0 up=dn, 1 up=conj(dn)\n" % 0)

def dumpOptions(filename, settings):
    text = "\n".join([
        "%.12f  #dt each slice of imagine time",
        "%d  #decomposit 1.d s;2.d c;3.c s;4.c c for different kinds of decomposition",
        # 1 for Hubbard with positive U
        "%d  #background 0. mean field 1. dynamic background walker",
        "%.2f  #bg_cap cap the background, bg_cap<0 means do not cap",
        "%d  #therm_sweep number of therm",
        "%d  #meas_sweep of measurement",
        "%d  #mgs_step number of steps when we do modified GS",
        "%d  #init_phi_aux_flag:0 phi and aux from code,1 aux from code read phi,2 read aux and phi",
        # initial guess
        "%d  #timeslice_size for the length of the beta",
        "%d  #blk_size the size of the block",
        "%d  #meas_step how often to measure.",
        "%d  #meas_skip_len, skip meas_skip_len*dt in left and right projection",
        "%d  #meas_type 0: Only measure energy 1:  Measure all quantities.",
        "%d  #variance_flag: 0 measure normally, 1. skip exp(-dt*H) to for variance problem.",
        "%d  #wf_flag: 0 up=dn, 1 up=conj(dn) only for init_phi_aux_flag=0",
        "%d  #aux_flag: 0 initial random AF, 1 initial CP AF, only init_phi_aux_flag!=2",
        "%d  #seed: 0 will random set the seed, otherwise use the value"
    ]) + "\n"

    params = (settings["dt"], settings["decomp"], 1, settings["bg_cap"], settings["therm_sweep"], \
            settings["meas_sweep"], settings["mgs_step"], 0, \
            int(settings["beta"]/settings["dt"]), \
            int(settings["blksize"]/settings["dt"]), \
            int(settings["meas_interval"]/settings["dt"]), \
            int(settings["meas_skip"]/settings["dt"])-1, \
            1 if settings["onepdm"] else 0, 0, 0, 1, settings["seed"])

    with open(filename, "w") as f:
        f.write(text % params)


def read1pdm(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    lines_sp = [l.split() for l in lines]
    rho = np.asarray([float(tokens[0])+float(tokens[1])*1.j for tokens in lines_sp])
    if la.norm(rho.imag) < 1e-8:
        rho = rho.real
    drho = np.asarray([float(tokens[4]) for tokens in lines_sp])
    return rho, drho

class AFQMC(object):
    
    execPath = os.path.join(os.environ["HOME"], "dev/afqmc")
    nproc = 1
    nnode = 1
    basicFiles = ["latt_param.dat", "method_param.dat", "model_param.dat"]
    env_slurm = "SLURM_JOBID" in os.environ
    settings = {
        "dt": 0.01,
        "decomp": 1, # use 3 for large U
        "beta": 50,
        "blksize": 0.1,
        "meas_interval": 0.5,
        "meas_skip": 10,
        "therm_sweep": 10,
        "meas_sweep": 200,
        "bg_cap": 3.6,
        "mgs_step": 10,
        "seed": 96384297,
    }

    def __init__(self, nproc, nnode = 1, TmpDir = "/tmp"):
        AFQMC.nproc = nproc
        AFQMC.nnode = nnode
        sub.check_call(["mkdir", "-p", TmpDir])
        self.tmpDir = mkdtemp(prefix = "AFQMC", dir = TmpDir)
        log.info("AFQMC working dir %s", self.tmpDir)
        log.info("running with %d nodes, %d processors per node", \
            AFQMC.nnode, AFQMC.nproc)
        self.count = 0

    def run(self, Ham, onepdm = True, similar = False):
        log.info("run AFQMC")
        log.info("parameters:")
        log.info("dt        = %9.4f", AFQMC.settings["dt"])
        log.info("beta      = %9.2f", AFQMC.settings["beta"])
        log.info("meas_int  = %9.4f", AFQMC.settings["meas_interval"])
        log.info("meas_skip = %9d", AFQMC.settings["meas_skip"])
        log.info("sweeps    = %3d, %4d", AFQMC.settings["therm_sweep"], AFQMC.settings["meas_sweep"])
        self.norbs = Ham.norb
        settings = copy.copy(AFQMC.settings)
        settings["onepdm"] = onepdm
        # clear the directory
        for f in os.listdir(self.tmpDir):
            if f.endswith(".dat"):
                os.remove(os.path.join(self.tmpDir, f))
        dumpH1(os.path.join(self.tmpDir, "latt_param.dat"), Ham.H1["cd"])
        dumpH2(os.path.join(self.tmpDir, "model_param.dat"), Ham.H2["ccdd"])
        dumpOptions(os.path.join(self.tmpDir, "method_param.dat"), settings)
        outputfile = self.callAFQMC()
        # read energy
        E, eE = self.extractE(grep("energy:", outputfile, A = 3))
        log.result("AFQMC energy uncertainty (1 sigma) = %20.12f", eE)
        return self.onepdm(), E + Ham.H0

    def onepdm(self):
        # read density matrix
        norbs = self.norbs
        rho, drho = read1pdm(os.path.join(self.tmpDir, "cicj.dat"))
        rho = rho.reshape((2, norbs, norbs))
        erho = np.max(abs(drho))
        log.result("AFQMC density matrix uncertainty (max) = %20.12f", erho)
        return rho

    def spin_corr(self):
        # read spin correlation matrix
        norbs = self.norbs
        scorr, dscorr = read1pdm(os.path.join(self.tmpDir, "sisj.dat"))
        scorr = scorr.reshape((norbs, norbs))
        escorr = np.max(abs(dscorr))
        log.result("AFQMC spin correlation matrix uncertainty (max)" \
                " = %20.12f", escorr)
        return scorr

    def extractE(self, text):
        lines = text.split('\n')
        energy = float(lines[1].split()[-1].split(",")[0][1:])
        d_energy = float(lines[-1].split()[-1])
        return energy, d_energy

    def callAFQMC(self):
        outputfile = os.path.join(self.tmpDir, "afqmc.out.%03d" % self.count)

        log.info("AFQMC call No. %d", self.count)
        log.debug(0, "Written to file %s", outputfile)
        cwd = os.getcwd()
        os.chdir(self.tmpDir)
        with open(outputfile, "w", buffering = 1) as f:
            if AFQMC.env_slurm:
                sub.check_call(" ".join(["srun", \
                        os.path.join(AFQMC.execPath, "afqmc")]), \
                        stdout = f, shell = True)
            else:
                sub.check_call(["mpirun", "-np", "%d" % (AFQMC.nproc * AFQMC.nnode), \
                        os.path.join(AFQMC.execPath, "afqmc")], stdout = f)
        self.count += 1
        os.chdir(cwd)
        return outputfile

    def cleanup(self):
        import shutil  
        shutil.rmtree(self.tmpDir)
