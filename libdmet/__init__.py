__version__ = "0.5"

__doc__ = \
"""
libDMET   version %s
A periodic DMET library for lattice model and realistic solid.
""" % (__version__)

import libdmet.settings
import libdmet.basis_transform
import libdmet.dmet
import libdmet.integral
import libdmet.lo
import libdmet.routine
import libdmet.solver
import libdmet.system
import libdmet.utils

__all__ = ["settings", "basis_transform", "dmet", "integral", "lo", "routine", "solver", "system", "utils"]
