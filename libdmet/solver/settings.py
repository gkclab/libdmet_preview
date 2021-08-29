import os

# Stackblock folder
#BLOCKPATH = '/home/shengg/opt/stackblocklatest/stackblock/' # ZHC add
#BLOCKPATH = '/home/zhcui/program/libdmet_ZHC/stackblock_bx/' # ZHC add
#BLOCKPATH = '/home/zcui/program/stackblock_bx/' # ZHC add
BLOCKPATH = '/home/zhcui/program/stackblock_hc/'
use_prebuild = ("_bx" in BLOCKPATH)

#BLOCK2PATH = "/home/zhcui/program/libdmet/libdmet/solver/block2_solver/"
#BLOCK2PATH = "/home/zcui/program/block2/pyblock2/driver/"
BLOCK2PATH = "/home/zhcui/program/block2/pyblock2/driver/"

# Genetic algorithm reorder for dmrgci
#GAOPTEXE = "/home/zhcui/program/libdmet_ZHC/stackblock_bx/genetic/gaopt"
#GAOPTEXE = "/home/zcui/program/stackblock_bx/genetic/gaopt"
GAOPTEXE = "/home/zhcui/program/stackblock_hc/genetic/gaopt"

GAOPT2 = BLOCK2PATH + "/gaopt"
