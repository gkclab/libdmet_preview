libDMET
===============================================
![Build Status](https://github.com/zhcui/libdmet_solid/workflows/CI/badge.svg)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A library of density matrix embedding theory (DMET) for lattice models and realistic solids.

Installation
------------

* Prerequisites
    - [PySCF](https://github.com/pyscf/pyscf) 2.0 or higher.

* Add libdmet top-level directory to your `PYTHONPATH` and you are all set!
  e.g. if libdmet_preview is installed in `/opt`, your `PYTHONPATH` should be

        export PYTHONPATH=/opt/libdmet_preview:$PYTHONPATH
	
* Extensions
    - [Wannier90](https://github.com/wannier-developers/wannier90): optional, for wannier functions as local orbitals.
	- [Block2](https://github.com/block-hczhai/block2-preview.git): optional, for DMRG solver.
	- [Stackblock](https://github.com/sanshar/StackBlock): optional, for DMRG solver.
	- [Arrow](https://github.com/QMC-Cornell/shci/tree/master): optional, for SHCI solver.

Reference
------------

The following papers should be cited in publications utilizing the libDMET program package:

Zhi-Hao Cui, Tianyu Zhu, Garnet Kin-Lic Chan, Efficient Implementation of Ab Initio Quantum Embedding in Periodic Systems: 
Density Matrix Embedding Theory, [J. Chem. Theory Comput. 2020, 16, 1, 119-129.](https://pubs.acs.org/doi/10.1021/acs.jctc.9b00933)

Tianyu Zhu, Zhi-Hao Cui, Garnet Kin-Lic Chan, Efficient Formulation of Ab Initio Quantum Embedding in Periodic Systems: 
Dynamical Mean-Field Theory, [J. Chem. Theory Comput. 2020, 16, 1, 141-153.](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.9b00934)


If you use the quantum chemistry formulation for superconductivity, please cite:

Zhi-Hao Cui, Junjie Yang, Johannes Tölle, Hong-Zhou Ye, Huanchen Zhai, Raehyun Kim, Xing Zhang, Lin Lin, Timothy C. Berkelbach, Garnet Kin-Lic Chan, Ab initio quantum many-body description of superconducting trends in the cuprates, [arXiv preprint arXiv:2306.16561](https://arxiv.org/abs/2306.16561)


Bug reports and feature requests
--------------------------------
Please submit tickets on the issues page.

