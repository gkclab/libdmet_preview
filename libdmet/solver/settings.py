import os

# Stackblock folder
#BLOCKPATH = '/home/shengg/opt/stackblocklatest/stackblock/' # ZHC add
#BLOCKPATH = '/home/zhcui/program/libdmet_ZHC/stackblock_bx/' # ZHC add
#BLOCKPATH = '/home/zcui/program/stackblock_bx/' # ZHC add
BLOCKPATH = '/home/zhcui/program/stackblock_hc/'
use_prebuild = ("_bx" in BLOCKPATH)

#BLOCK2PATH = "/home/zhcui/program/block2/pyblock2/driver/"
#BLOCK2PATH = "/home/zcui/program/block2/pyblock2/driver/"
BLOCK2PATH = "/global/homes/z/zhcui/program/block2/pyblock2/driver/"

# Genetic algorithm reorder for dmrgci
#GAOPTEXE = "/home/zhcui/program/libdmet_ZHC/stackblock_bx/genetic/gaopt"
#GAOPTEXE = "/home/zcui/program/stackblock_bx/genetic/gaopt"
GAOPTEXE = "/home/zhcui/program/stackblock_hc/genetic/gaopt"

GAOPT2 = BLOCK2PATH + "/gaopt"

MPI_GCCSD_PATH = os.path.dirname(os.path.realpath(__file__)) +  "/mpicc_main.py"

DQMC_PATH = "/home/zhcui/program/VMC_ghf/bin/DQMC"
DQMC_BLOCKING = "/home/zhcui/program/VMC_ghf/scripts/blocking.py"

DICE_PATH = "/home/zhcui/program/dice/ZDice2"
