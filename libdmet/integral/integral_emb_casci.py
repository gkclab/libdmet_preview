import numpy as np

def transform(v_A, v_B, u_A, u_B, h_0, h_A, h_B, D, w_A, w_B, w_AB):
    H0_H0 = h_0
    val001 = np.dot(u_A.T, h_A)
    val002 = np.dot(u_B.T, h_B)
    H0_H1 = 1.0*np.sum(val001.T*u_A) + \
            2.0*np.sum(np.dot(u_A.T, D).T*v_B) + \
            1.0*np.sum(val002.T*u_B)
    val003 = np.dot(u_A, u_A.T)
    val004 = np.tensordot(val003, w_A, axes=((0, 1),(0, 1)))
    val005 = np.tensordot(val003, w_A, axes=((0, 1),(0, 2)))
    val006 = np.dot(u_B, u_B.T)
    val007 = np.tensordot(val006, w_B, axes=((0, 1),(0, 1)))
    val008 = np.tensordot(val006, w_B, axes=((0, 1),(0, 2)))
    val009 = np.dot(u_A, v_B.T)
    val010 = np.tensordot(val009, w_AB, axes=((0, 1),(0, 2)))
    val011 = np.tensordot(val003, w_AB, axes=((0, 1),(0, 1)))
    H0_H2 = 0.5*np.sum(val004*val003) + \
            1.0*np.sum(val011*val006) + \
            1.0*np.sum(val010*val009) + \
            0.5*np.sum(val007*val006) + \
            -0.5*np.sum(val005*val003) + \
            -0.5*np.sum(val008*val006)
    val012 = np.dot(v_A.T, h_A)
    val013 = np.dot(v_A.T, D)
    H1D_H1 = 1.0*np.dot(val012, u_A) + \
            1.0*np.dot(np.dot(u_B.T, D.T), u_A) + \
            1.0*np.dot(val013, v_B) + \
            -1.0*np.dot(val002, v_B)
    val014 = np.dot(val013, u_B)
    H1A_H1 = 1.0*val014 + \
            1.0*np.dot(val012, v_A) + \
            -1.0*np.dot(val002, u_B) + \
            1.0*val014.T
    val015 = np.dot(np.dot(v_B.T, D.T), u_A)
    H1B_H1 = -1.0*np.dot(val001, u_A) + \
            -1.0*val015 + \
            -1.0*val015.T + \
            1.0*np.dot(np.dot(v_B.T, h_B), v_B)
    val016 = np.dot(v_A.T, val004)
    val017 = np.dot(v_A.T, val005)
    val018 = np.dot(u_B.T, val007)
    val019 = np.dot(u_B.T, val008)
    val020 = np.dot(v_A.T, val010)
    val021 = np.tensordot(val006, w_AB, axes=((0, 1),(2, 3)))
    val022 = np.dot(v_A.T, val021)
    val023 = np.dot(u_B.T, val011)
    H1D_H2 = 1.0*np.dot(val022, u_A) + \
            -1.0*np.dot(val023, v_B) + \
            1.0*np.dot(np.dot(u_B.T, val010.T), u_A) + \
            1.0*np.dot(val020, v_B) + \
            1.0*np.dot(val016, u_A) + \
            1.0*np.dot(val019, v_B) + \
            -1.0*np.dot(val017, u_A) + \
            -1.0*np.dot(val018, v_B)
    val024 = np.dot(val020, u_B)
    H1A_H2 = 1.0*np.dot(val016, v_A) + \
            1.0*val024.T + \
            1.0*np.dot(val022, v_A) + \
            1.0*np.dot(val019, u_B) + \
            -1.0*np.dot(val023, u_B) + \
            -1.0*np.dot(val017, v_A) + \
            -1.0*np.dot(val018, u_B) + \
            1.0*val024
    val025 = np.dot(np.dot(u_A.T, val010), v_B)
    H1B_H2 = -1.0*np.dot(np.dot(u_A.T, val021), u_A) + \
            -1.0*np.dot(np.dot(u_A.T, val004), u_A) + \
            -1.0*np.dot(np.dot(v_B.T, val008), v_B) + \
            -1.0*val025 + \
            1.0*np.dot(np.dot(v_B.T, val011), v_B) + \
            -1.0*val025.T + \
            1.0*np.dot(np.dot(v_B.T, val007), v_B) + \
            1.0*np.dot(np.dot(u_A.T, val005), u_A)
    val026 = np.tensordot(u_A, w_A, axes=(0,0))
    val027 = np.tensordot(np.tensordot(np.tensordot(u_A, val026, axes=(0,2)), v_A, axes=(3,0)), u_A, axes=(2,0))
    val028 = np.tensordot(v_B, w_B, axes=(0,0))
    val029 = np.tensordot(np.tensordot(np.tensordot(v_B, val028, axes=(0,2)), u_B, axes=(3,0)), v_B, axes=(2,0))
    val030 = np.tensordot(v_B, w_AB, axes=(0,2))
    val031 = np.tensordot(u_A, val030, axes=(0,1))
    val032 = -1.0*np.transpose(np.tensordot(np.tensordot(val031, v_A, axes=(2,0)), v_B, axes=(2,0)), (1, 0, 2, 3)) + \
            -0.5*val027 + \
            1.0*np.transpose(np.tensordot(np.tensordot(val031, u_B, axes=(3,0)), u_A, axes=(2,0)), (1, 0, 2, 3)) + \
            -0.5*val029 + \
            0.5*np.transpose(val029, (1, 0, 2, 3)) + \
            0.5*np.transpose(val027, (1, 0, 2, 3))
    H2yB_H2 = val032 - np.transpose(val032, (1,0,2,3))
    val033 = np.tensordot(v_A, w_A, axes=(0,0))
    val034 = np.tensordot(np.tensordot(v_A, val033, axes=(0,2)), u_A, axes=(3,0))
    val035 = np.tensordot(val034, v_A, axes=(2,0))
    val036 = np.tensordot(u_B, w_B, axes=(0,0))
    val037 = np.tensordot(np.tensordot(u_B, val036, axes=(0,2)), v_B, axes=(3,0))
    val038 = np.tensordot(val037, u_B, axes=(2,0))
    val039 = np.tensordot(v_A, w_AB, axes=(0,0))
    val040 = np.tensordot(u_B, val039, axes=(0,2))
    val041 = np.tensordot(val040, v_B, axes=(3,0))
    val042 = 0.5*np.transpose(val038, (1, 0, 2, 3)) + \
            0.5*np.transpose(val035, (1, 0, 2, 3)) + \
            1.0*np.transpose(np.tensordot(np.tensordot(val040, u_A, axes=(2,0)), u_B, axes=(2,0)), (1, 0, 2, 3)) + \
            -1.0*np.transpose(np.tensordot(val041, v_A, axes=(2,0)), (1, 0, 2, 3)) + \
            -0.5*val035 + \
            -0.5*val038
    H2yA_H2 = val042 - np.transpose(val042, (1,0,2,3))
    val043 = 0.5*np.tensordot(np.tensordot(np.tensordot(v_B, val028, axes=(0,1)), v_B, axes=(3,0)), v_B, axes=(2,0)) + \
            -1.0*np.tensordot(np.tensordot(np.tensordot(v_B, val030, axes=(0,3)), u_A, axes=(3,0)), u_A, axes=(2,0)) + \
            0.5*np.tensordot(np.tensordot(np.tensordot(u_A, val026, axes=(0,1)), u_A, axes=(3,0)), u_A, axes=(2,0))
    H2wB_H2 = val043 + np.transpose(val043, (2,3,0,1))
    val044 = np.tensordot(v_A, val033, axes=(0,1))
    val045 = np.tensordot(u_B, val036, axes=(0,1))
    val046 = np.tensordot(v_A, val039, axes=(0,1))
    val047 = -1.0*np.tensordot(np.tensordot(val046, u_B, axes=(3,0)), u_B, axes=(2,0)) + \
            0.5*np.tensordot(np.tensordot(val044, v_A, axes=(3,0)), v_A, axes=(2,0)) + \
            0.5*np.tensordot(np.tensordot(val045, u_B, axes=(3,0)), u_B, axes=(2,0))
    H2wA_H2 = val047 + np.transpose(val047, (2,3,0,1))
    val048 = np.tensordot(val034, u_A, axes=(2,0))
    val049 = np.tensordot(val037, v_B, axes=(2,0))
    val050 = np.tensordot(val041, u_A, axes=(2,0))
    val051 = np.transpose(val050, (1, 0, 2, 3))
    val052 = 0.5*np.transpose(val049, (0, 1, 3, 2)) + \
            -1.0*val051 + \
            0.5*np.transpose(val048, (0, 1, 3, 2))
    val053 = val052 - np.transpose(val052, (1,0,2,3))
    H2x_H2 = val053 - np.transpose(val053, (0,1,3,2))
    H2wAB_H2 = -1.0*np.transpose(val051, (0, 1, 3, 2)) + \
            1.0*val049 + \
            -1.0*np.tensordot(np.tensordot(val044, u_A, axes=(3,0)), u_A, axes=(2,0)) + \
            1.0*np.tensordot(np.tensordot(np.tensordot(u_B, np.tensordot(u_B, w_AB, axes=(0,2)), axes=(0,3)), u_A, axes=(3,0)), u_A, axes=(2,0)) + \
            1.0*np.tensordot(np.tensordot(val046, v_B, axes=(3,0)), v_B, axes=(2,0)) + \
            -1.0*val050 + \
            -1.0*np.tensordot(np.tensordot(val045, v_B, axes=(3,0)), v_B, axes=(2,0)) + \
            1.0*val048

    return H0_H0 + H0_H1 + H0_H2, np.asarray([H1A_H1 + H1A_H2, H1B_H1 + H1B_H2]), \
            (H1D_H1 + H1D_H2)[np.newaxis], np.asarray([H2wA_H2, H2wB_H2, H2wAB_H2]), \
            np.asarray([H2yA_H2, H2yB_H2]), H2x_H2[np.newaxis]