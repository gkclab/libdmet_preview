from libdmet.utils import logger as log
from libdmet.bitgen import integral_transform as trans
from libdmet.bitgen import basic

log.section("The most general integral transformation code")

Ham = basic.H2(True, False)

log.result("H = %s", Ham)

sub = trans.Substitution([
    trans.FermionSub(basic.Fermion(True, 'A','i'), basic.OpSum(basic.Coeff("v_A",'ip') * \
            basic.C('A','p')) + basic.OpSum(basic.Coeff("u_A",'ip') * basic.D('B','p'))),
    trans.FermionSub(basic.Fermion(True, 'B','i'), basic.OpSum(basic.Coeff("v_B",'ip') * \
            basic.C('B','p')) + basic.OpSum(basic.Coeff("u_B",'ip') * basic.D('A','p'))),
])

log.result("Substitute:\n%s", sub)

H = trans.transform(sub, Ham)

# now further simplify the terms by classify according to integrals

with open("integral_nonlocal_emb1.py", "w") as f:
    f.write("import numpy as np\n\n")
    f.write("def transform(v_A, v_B, u_A, u_B, w):\n    ")
    f.write(trans.generate_code(H, indices = "pqrs").replace("\n", "\n    ") + "\n")
    f.write("""
    return H0_H2, np.asarray([H1A_H2, H1B_H2]), H1D_H2[np.newaxis], \\
            np.asarray([H2wA_H2, H2wB_H2, H2wAB_H2]), np.asarray([H2yA_H2, H2yB_H2]), \\
            H2x_H2[np.newaxis]""")
for term in trans.registered:
    if term.final:
        log.result("term %s has symmetry:\n%s", term._get_name(), term.symm)
