from libdmet.utils import logger as log
from libdmet.bitgen import integral_transform as trans
from libdmet.bitgen import basic

log.section("General integral transformation code with restricted two body integral")

# step 1: for 1 body terms
Ham = basic.h0term() + basic.H1(False, True)

log.result("H = %s", Ham)

sub = trans.Substitution([
    trans.FermionSub(basic.Fermion(True, 'A','i'), basic.OpSum(basic.Coeff("v_A",'ip') * \
            basic.C('A','p')) + basic.OpSum(basic.Coeff("u_A",'ip') * basic.D('B','p'))),
    trans.FermionSub(basic.Fermion(True, 'B','i'), basic.OpSum(basic.Coeff("v_B",'ip') * \
            basic.C('B','p')) + basic.OpSum(basic.Coeff("u_B",'ip') * basic.D('A','p'))),
])

log.result("Substitute:\n%s", sub)

H = trans.transform(sub, Ham)

with open("integral_emb_casci_save_mem1.py", "w") as f:
    f.write("import numpy as np\n\n")
    f.write("def __transform01(v_A, v_B, u_A, u_B, h_0, h_A, h_B, D):\n    ")
    f.write(trans.generate_code(H, indices = "pqrs").replace("\n", "\n    ") + "\n")
    f.write("""
    return H0_H0 + H0_H1, np.asarray([H1A_H1, H1B_H1]), H1D_H1[np.newaxis]""")
for term in trans.registered:
    if term.final:
        log.result("term %s has symmetry:\n%s", term._get_name(), term.symm)

trans.cleanup()

# step 2: for 2 body terms
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

with open("integral_emb_casci_save_mem1.py", "a") as f:
    f.write("\n\n")
    f.write("def __transform2(v_A, v_B, u_A, u_B, w):\n    ")
    f.write(trans.generate_code(H, indices = "pqrs").replace("\n", "\n    ") + "\n")
    f.write("""
    return H0_H2, np.asarray([H1A_H2, H1B_H2]), H1D_H2[np.newaxis], \\
            np.asarray([H2wA_H2, H2wB_H2, H2wAB_H2]), \\
            np.asarray([H2yA_H2, H2yB_H2]), H2x_H2[np.newaxis]""")
for term in trans.registered:
    if term.final:
        log.result("term %s has symmetry:\n%s", term._get_name(), term.symm)
