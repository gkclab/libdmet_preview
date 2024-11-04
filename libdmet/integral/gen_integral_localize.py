from libdmet.utils import logger as log
from libdmet.bitgen import integral_transform as trans
from libdmet.bitgen import basic

log.section("The most general integral transformation code")

Ham = basic.h0term() + basic.H1(False, True) + basic.H2(False, True)

log.result("H = %s", Ham)

sub = trans.Substitution([
    trans.FermionSub(basic.Fermion(True, 'A','i'), basic.OpSum(basic.Coeff("v_A",'ip') * \
            basic.C('A','p'))),
    trans.FermionSub(basic.Fermion(True, 'B','i'), basic.OpSum(basic.Coeff("v_B",'ip') * \
            basic.C('B','p'))),
])

log.result("Substitute:\n%s", sub)

H = trans.transform(sub, Ham)

with open("integral_localize1.py", "w") as f:
    f.write("import numpy as np\n\n")
    f.write("def transform(v_A, v_B, h_0, h_A, h_B, D, w_A, w_B, w_AB, y_A, y_B, x):\n    ")
    f.write(trans.generate_code(H, indices = "pqrs").replace("\n", "\n    ") + "\n")
    f.write("""
    return H0_H0, np.asarray([H1A_H1, H1B_H1]), H1D_H1[np.newaxis], \\
            np.asarray([H2wA_H2, H2wB_H2, H2wAB_H2]), \\
            np.asarray([H2yA_H2, H2yB_H2]), H2x_H2[np.newaxis]""")
for term in trans.registered:
    if term.final:
        log.result("term %s has symmetry:\n%s", term._get_name(), term.symm)
