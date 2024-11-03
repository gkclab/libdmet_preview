#! /usr/bin/env python

def test_dca():
    import numpy as np
    from libdmet.utils.dca_transform import transformHam
    lattice = (33, 31)
    cell = (8, 8)
    H = np.zeros(lattice + cell)
    # because of translational invariance
    # indices are H(x2-x1, y2-y1, z2-z1, X2-X1, Y2-Y1, Z2-Z1)
    # where ai is supercell index and Ai is site index
    H[0, 0, 0, 1] = -1
    H[0, 0, 1, 0] = -1
    H[-1, 0, -1, 0] = -1
    H[0, -1, 0, -1] = -1
    #t1 = 0.2
    #H[0, 0, 1, 1] = t1
    #H[0, -1, 1, -1] = t1
    #H[-1, 0, -1, 1] = t1
    #H[-1, -1, -1, -1] = t1

    Hnew = transformHam(lattice, cell, H)
    for idx, v in Hnew:
        print (idx, v)

if __name__ == "__main__":
    test_dca()
