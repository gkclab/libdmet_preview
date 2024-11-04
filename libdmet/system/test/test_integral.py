#! /usr/bin/env python

def test_integral():
    import os
    import shutil
    import numpy as np
    from subprocess import call
    from tempfile import mkdtemp
    from libdmet.system import integral
    from libdmet.utils import logger as log

    log.result("Testing Bogoliubov unrestricted integrals ...")
    input = os.path.dirname(os.path.realpath(__file__)) + "/DMETDUMP"
    Ham = integral.read(input, 8, False, True, "FCIDUMP")
    output = "./DMETDUMPtest"
    integral.dump(output, Ham, "FCIDUMP")
    s = call(["diff", input, output])

    if s == 0:
        log.result("... Successful")
        call(["rm", output])
    else:
        log.result("... Failed")
        log.result("Manually check input vs. output file: %s %s", input, output)
    assert s == 0
    log.result("")

    log.result("Testing Hubbard restricted integrals ...")
    input = os.path.dirname(os.path.realpath(__file__)) + "/HUBDUMP"
    Ham = integral.read(input, 12, True, False, "FCIDUMP")
    output = "./HUBDUMPtest"
    integral.dump(output, Ham, "FCIDUMP")
    s = call(["diff", input, output])

    if s == 0:
        log.result("... Successful")
        call(["rm", output])
    else:
        log.result("... Failed")
        log.result("Manually check input vs. output file: %s %s", input, output)
    assert s == 0
    log.result("")

    log.result("Testing memory map integrals ...")
    output = mkdtemp(prefix="MMAP_INT", dir="./")
    log.result("Stored to %s", output)
    integral.dump(output, Ham, "MMAP")
    Ham1 = integral.read(output, 12, True, False, "MMAP", copy=False)
    log.result("H0: %f %f", Ham1.H0, Ham.H0)
    log.result("H1: %s", np.allclose(Ham1.H1["cd"], Ham.H1["cd"]))
    log.result("H2: %s", np.allclose(Ham1.H2["ccdd"], Ham.H2["ccdd"]))
    if Ham1.H0 == Ham.H0 and np.allclose(Ham1.H1["cd"], Ham.H1["cd"]) and np.allclose(Ham1.H2["ccdd"], Ham.H2["ccdd"]):
        log.result("...Successful")
        call(["rm", "-rf", output])
        #shutil.rmtree(output, ignore_errors=True)
    else:
        log.result("... Failed")
        assert False

    integral.dumpFCIDUMP_no_perm("FCIDUMP", Ham1, thr=1e-12)

def test_dump_FCIDUMP():
    import numpy as np
    from pyscf import ao2mo
    from libdmet.system import integral
    norb = 5
    H0 = 1.0
    H1 = np.random.random((norb, norb))
    H1 = H1 + H1.T

    H2 = np.random.random((norb, norb, norb, norb))
    H2 = H2 + H2.transpose(0, 1, 3, 2)
    H2 = H2 + H2.transpose(1, 0, 2, 3)
    H2 = H2 + H2.transpose(2, 3, 0, 1)

    output = "FCIDUMP"
    Ham = integral.Integral(norb, True, False, H0, {"cd": H1[None]}, \
            {"ccdd": H2[None]}, ovlp=None)
    integral.dump(output, Ham, "FCIDUMP")

    Ham = integral.Integral(norb, True, False, H0, {"cd": H1[None]}, \
            {"ccdd": ao2mo.restore(4, H2, norb)[None]}, ovlp=None)
    integral.dump(output, Ham, "FCIDUMP")

    Ham = integral.Integral(norb, True, False, H0, {"cd": H1[None]}, \
            {"ccdd": ao2mo.restore(8, H2, norb)[None]}, ovlp=None)
    integral.dump(output, Ham, "FCIDUMP")

    H2_a = H2
    H2_b = H2 + 1.0
    H2_ab = H2 - 0.4
    Ham = integral.Integral(norb, False, False, H0, {"cd": np.array((H1, H1))}, \
            {"ccdd": np.array((H2_a, H2_b, H2_ab))}, ovlp=None)
    integral.dump(output, Ham, "FCIDUMP")


    Ham = integral.Integral(norb, False, False, H0, {"cd": np.array((H1, H1))}, \
            {"ccdd": np.array((ao2mo.restore(4, H2_a, norb), \
                               ao2mo.restore(4, H2_b, norb),
                               ao2mo.restore(4, H2_ab, norb)))}, ovlp=None)
    integral.dump(output, Ham, "FCIDUMP")

def test_save_load_integral():
    import os
    import numpy as np
    from libdmet.system import integral
    from libdmet.utils import logger as log

    log.result("Testing Bogoliubov unrestricted integrals ...")
    input = os.path.dirname(os.path.realpath(__file__)) + "/DMETDUMP"
    Ham = integral.read(input, 8, False, True, "FCIDUMP")
    Ham.save(fname="int.h5")
    Ham_2 = integral.load("int.h5")
    for k, v1 in Ham.__dict__.items():
        v2 = Ham_2.__dict__[k]
        if isinstance(v1, dict):
            for i, x in v1.items():
                assert np.allclose(x, v2[i])
        else:
            assert np.allclose(v1, v2)

    log.result("Testing Hubbard restricted integrals ...")
    input = os.path.dirname(os.path.realpath(__file__)) + "/HUBDUMP"
    Ham_3 = integral.read(input, 12, True, False, "FCIDUMP")

    Ham_3.save(fname="int.h5")
    Ham_4 = integral.load("int.h5")
    for k, v1 in Ham_3.__dict__.items():
        v2 = Ham_4.__dict__[k]
        if isinstance(v1, dict):
            for i, x in v1.items():
                assert np.allclose(x, v2[i])
        else:
            assert np.allclose(v1, v2)

    # overwrite the original integrals
    Ham.load(fname="int.h5")
    for k, v1 in Ham_3.__dict__.items():
        v2 = Ham.__dict__[k]
        if isinstance(v1, dict):
            for i, x in v1.items():
                assert np.allclose(x, v2[i])
        else:
            assert np.allclose(v1, v2)

def test_get_eri_format():
    import numpy as np
    from pyscf import ao2mo
    from libdmet.system import integral
    nao = 5

    # spin = 0
    eri = np.random.random((nao, nao, nao, nao))
    eri_format, spin_dim = integral.get_eri_format(eri, nao)
    assert eri_format == 's1'
    assert spin_dim == 0

    eri_s4 = ao2mo.restore(4, eri, nao)
    eri_format, spin_dim = integral.get_eri_format(eri_s4, nao)
    assert eri_format == 's4'
    assert spin_dim == 0

    eri_s8 = ao2mo.restore(8, eri_s4, nao)
    eri_format, spin_dim = integral.get_eri_format(eri_s8, nao)
    assert eri_format == 's8'
    assert spin_dim == 0

    # spin = 1
    eri_format, spin_dim = integral.get_eri_format(eri[None], nao)
    assert eri_format == 's1'
    assert spin_dim == 1

    eri_format, spin_dim = integral.get_eri_format(eri_s4[None], nao)
    assert eri_format == 's4'
    assert spin_dim == 1

    eri_format, spin_dim = integral.get_eri_format(eri_s8[None], nao)
    assert eri_format == 's8'
    assert spin_dim == 1

    # spin = 3
    eri = np.random.random((3, nao, nao, nao, nao))
    eri_format, spin_dim = integral.get_eri_format(eri, nao)
    assert eri_format == 's1'
    assert spin_dim == 3

    eri_s4 = np.array((eri_s4, eri_s4, eri_s4))
    eri_format, spin_dim = integral.get_eri_format(eri_s4, nao)
    assert eri_format == 's4'
    assert spin_dim == 3

if __name__ == "__main__":
    test_integral()
    test_save_load_integral()
    test_dump_FCIDUMP()
    test_get_eri_format()
