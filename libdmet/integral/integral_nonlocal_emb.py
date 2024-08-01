import numpy as np

def transform(v_A, v_B, u_A, u_B, w):
    val001 = np.dot(u_A, u_A.T)
    val002 = np.tensordot(val001, w, axes=((0, 1),(0, 1)))
    val003 = np.tensordot(val001, w, axes=((0, 1),(0, 2)))
    val004 = np.dot(u_B, u_B.T)
    val005 = np.tensordot(val004, w, axes=((0, 1),(0, 1)))
    val006 = np.tensordot(val004, w, axes=((0, 1),(0, 2)))
    val007 = np.dot(u_A, v_B.T)
    val008 = np.tensordot(val007, w, axes=((0, 1),(0, 2)))
    H0_H2 = -0.5*np.sum(val003*val001) + \
            1.0*np.sum(val002*val004) + \
            0.5*np.sum(val005*val004) + \
            1.0*np.sum(val008*val007) + \
            0.5*np.sum(val002*val001) + \
            -0.5*np.sum(val006*val004)
    val009 = np.dot(v_A.T, val002)
    val010 = np.dot(v_A.T, val003)
    val011 = np.dot(u_B.T, val005)
    val012 = np.dot(u_B.T, val006)
    val013 = np.dot(v_A.T, val008)
    val014 = np.dot(v_A.T, val005)
    val015 = np.dot(u_B.T, val002)
    H1D_H2 = -1.0*np.dot(val010, u_A) + \
            1.0*np.dot(val013, v_B) + \
            1.0*np.dot(val009, u_A) + \
            -1.0*np.dot(val011, v_B) + \
            -1.0*np.dot(val015, v_B) + \
            1.0*np.dot(val014, u_A) + \
            1.0*np.dot(np.dot(u_B.T, val008.T), u_A) + \
            1.0*np.dot(val012, v_B)
    val016 = np.dot(val013, u_B)
    H1A_H2 = -1.0*np.dot(val015, u_B) + \
            1.0*np.dot(val012, u_B) + \
            1.0*val016 + \
            1.0*np.dot(val014, v_A) + \
            1.0*val016.T + \
            -1.0*np.dot(val011, u_B) + \
            1.0*np.dot(val009, v_A) + \
            -1.0*np.dot(val010, v_A)
    val017 = np.dot(np.dot(u_A.T, val008), v_B)
    H1B_H2 = -1.0*val017.T + \
            -1.0*np.dot(np.dot(u_A.T, val005), u_A) + \
            -1.0*np.dot(np.dot(v_B.T, val006), v_B) + \
            1.0*np.dot(np.dot(v_B.T, val005), v_B) + \
            -1.0*np.dot(np.dot(u_A.T, val002), u_A) + \
            1.0*np.dot(np.dot(v_B.T, val002), v_B) + \
            -1.0*val017 + \
            1.0*np.dot(np.dot(u_A.T, val003), u_A)
    val018 = np.tensordot(u_A, w, axes=(0,0))
    val019 = np.tensordot(np.tensordot(np.tensordot(u_A, val018, axes=(0,2)), v_A, axes=(3,0)), u_A, axes=(2,0))
    val020 = np.tensordot(v_B, w, axes=(0,0))
    val021 = np.tensordot(np.tensordot(np.tensordot(v_B, val020, axes=(0,2)), u_B, axes=(3,0)), v_B, axes=(2,0))
    val022 = np.tensordot(u_A, val020, axes=(0,2))
    val023 = 0.5*np.transpose(val019, (1, 0, 2, 3)) + \
            -0.5*val021 + \
            -0.5*val019 + \
            0.5*np.transpose(val021, (1, 0, 2, 3)) + \
            -1.0*np.transpose(np.tensordot(np.tensordot(val022, v_A, axes=(3,0)), v_B, axes=(2,0)), (1, 0, 2, 3)) + \
            1.0*np.transpose(np.tensordot(np.tensordot(val022, u_B, axes=(2,0)), u_A, axes=(2,0)), (1, 0, 2, 3))
    H2yB_H2 = val023 - np.transpose(val023, (1,0,2,3))
    val024 = np.tensordot(v_A, w, axes=(0,0))
    val025 = np.tensordot(v_A, val024, axes=(0,1))
    val026 = np.tensordot(u_B, w, axes=(0,0))
    val027 = np.tensordot(u_B, val026, axes=(0,1))
    val028 = -1.0*np.tensordot(np.tensordot(val025, u_B, axes=(3,0)), u_B, axes=(2,0)) + \
            0.5*np.tensordot(np.tensordot(val025, v_A, axes=(3,0)), v_A, axes=(2,0)) + \
            0.5*np.tensordot(np.tensordot(val027, u_B, axes=(3,0)), u_B, axes=(2,0))
    H2wA_H2 = val028 + np.transpose(val028, (2,3,0,1))
    val029 = np.tensordot(v_B, val020, axes=(0,1))
    val030 = 0.5*np.tensordot(np.tensordot(np.tensordot(u_A, val018, axes=(0,1)), u_A, axes=(3,0)), u_A, axes=(2,0)) + \
            -1.0*np.tensordot(np.tensordot(val029, u_A, axes=(3,0)), u_A, axes=(2,0)) + \
            0.5*np.tensordot(np.tensordot(val029, v_B, axes=(3,0)), v_B, axes=(2,0))
    H2wB_H2 = val030 + np.transpose(val030, (2,3,0,1))
    val031 = np.tensordot(np.tensordot(v_A, val024, axes=(0,2)), u_A, axes=(3,0))
    val032 = np.tensordot(val031, v_A, axes=(2,0))
    val033 = np.tensordot(np.tensordot(u_B, val026, axes=(0,2)), v_B, axes=(3,0))
    val034 = np.tensordot(val033, u_B, axes=(2,0))
    val035 = np.tensordot(u_B, val024, axes=(0,2))
    val036 = np.tensordot(val035, v_B, axes=(3,0))
    val037 = -0.5*val032 + \
            0.5*np.transpose(val032, (1, 0, 2, 3)) + \
            0.5*np.transpose(val034, (1, 0, 2, 3)) + \
            -1.0*np.transpose(np.tensordot(val036, v_A, axes=(2,0)), (1, 0, 2, 3)) + \
            -0.5*val034 + \
            1.0*np.transpose(np.tensordot(np.tensordot(val035, u_A, axes=(2,0)), u_B, axes=(2,0)), (1, 0, 2, 3))
    H2yA_H2 = val037 - np.transpose(val037, (1,0,2,3))
    val038 = np.tensordot(val031, u_A, axes=(2,0))
    val039 = np.tensordot(val033, v_B, axes=(2,0))
    val040 = np.tensordot(val036, u_A, axes=(2,0))
    val041 = np.transpose(val040, (1, 0, 2, 3))
    val042 = 0.5*np.transpose(val039, (0, 1, 3, 2)) + \
            0.5*np.transpose(val038, (0, 1, 3, 2)) + \
            -1.0*val041
    val043 = val042 - np.transpose(val042, (1,0,2,3))
    H2x_H2 = val043 - np.transpose(val043, (0,1,3,2))
    H2wAB_H2 = 1.0*val038 + \
            -1.0*np.transpose(val041, (0, 1, 3, 2)) + \
            1.0*val039 + \
            -1.0*np.tensordot(np.tensordot(val027, v_B, axes=(3,0)), v_B, axes=(2,0)) + \
            -1.0*val040 + \
            1.0*np.tensordot(np.tensordot(val025, v_B, axes=(3,0)), v_B, axes=(2,0)) + \
            1.0*np.tensordot(np.tensordot(val027, u_A, axes=(3,0)), u_A, axes=(2,0)) + \
            -1.0*np.tensordot(np.tensordot(val025, u_A, axes=(3,0)), u_A, axes=(2,0))

    return H0_H2, np.asarray([H1A_H2, H1B_H2]), H1D_H2[np.newaxis], \
            np.asarray([H2wA_H2, H2wB_H2, H2wAB_H2]), np.asarray([H2yA_H2, H2yB_H2]), \
            H2x_H2[np.newaxis]
