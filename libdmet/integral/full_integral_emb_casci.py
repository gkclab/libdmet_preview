import numpy as np

def transform(v_A, v_B, u_A, u_B, h_0, h_A, h_B, D, w_A, w_B, w_AB, y_A, y_B, x):
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
    val012 = np.tensordot(val003, y_A, axes=((0, 1),(0, 3)))
    val013 = np.dot(v_A, u_B.T)
    val014 = np.tensordot(val006, y_B, axes=((0, 1),(0, 3)))
    val015 = np.tensordot(val009, x, axes=((0, 1),(0, 2)))
    val016 = (-val015)
    H0_H2 = 1.0*np.sum(val016*val009) + \
            0.5*np.sum(val004*val003) + \
            1.0*np.sum(val011*val006) + \
            0.5*np.sum(val007*val006) + \
            1.0*np.sum(val010*val009) + \
            -0.5*np.sum(val005*val003) + \
            2.0*np.sum(val012*val009) + \
            2.0*np.sum(val014*val013.T) + \
            -0.5*np.sum(val008*val006)
    val017 = np.dot(v_A.T, h_A)
    val018 = np.dot(v_A.T, D)
    H1D_H1 = 1.0*np.dot(val017, u_A) + \
            1.0*np.dot(np.dot(u_B.T, D.T), u_A) + \
            1.0*np.dot(val018, v_B) + \
            -1.0*np.dot(val002, v_B)
    val019 = np.dot(val018, u_B)
    H1A_H1 = 1.0*val019.T + \
            -1.0*np.dot(val002, u_B) + \
            1.0*np.dot(val017, v_A) + \
            1.0*val019
    val020 = np.dot(np.dot(v_B.T, D.T), u_A)
    H1B_H1 = -1.0*np.dot(val001, u_A) + \
            -1.0*val020 + \
            -1.0*val020.T + \
            1.0*np.dot(np.dot(v_B.T, h_B), v_B)
    val021 = np.dot(v_A.T, val004)
    val022 = np.dot(v_A.T, val005)
    val023 = np.dot(u_B.T, val007)
    val024 = np.dot(u_B.T, val008)
    val025 = np.dot(v_A.T, val010)
    val026 = np.tensordot(val006, w_AB, axes=((0, 1),(2, 3)))
    val027 = np.dot(v_A.T, val026)
    val028 = np.dot(u_B.T, val011)
    val029 = (-np.tensordot(val009, y_A, axes=((0, 1),(0, 2))))
    val030 = np.dot(v_A.T, val029)
    val031 = (-val012)
    val032 = np.dot(v_A.T, val031)
    val033 = (-np.tensordot(val013, y_B, axes=((1, 0),(0, 2))))
    val034 = np.dot(u_B.T, val033.T)
    val035 = (-val014)
    val036 = np.dot(v_A.T, val035.T)
    val037 = np.dot(v_A.T, val016)
    H1D_H2 = 1.0*np.dot(val025, v_B) + \
            1.0*np.dot(val036, v_B) + \
            -1.0*np.dot(val022, u_A) + \
            -1.0*np.dot(val028, v_B) + \
            1.0*np.dot(val030, u_A) + \
            -1.0*np.dot(np.dot(u_B.T, val031.T), u_A) + \
            1.0*np.dot(np.dot(u_B.T, val035), u_A) + \
            1.0*np.dot(np.dot(u_B.T, val010.T), u_A) + \
            1.0*np.dot(val037, v_B) + \
            1.0*np.dot(np.dot(v_A.T, val029.T), u_A) + \
            -1.0*np.dot(np.dot(u_B.T, val015.T), u_A) + \
            -1.0*np.dot(val034, v_B) + \
            1.0*np.dot(val024, v_B) + \
            1.0*np.dot(val027, u_A) + \
            -1.0*np.dot(val032, v_B) + \
            1.0*np.dot(val021, u_A) + \
            -1.0*np.dot(val023, v_B) + \
            -1.0*np.dot(np.dot(u_B.T, val033), v_B)
    val038 = np.dot(val025, u_B)
    val039 = np.dot(val030, v_A)
    val040 = np.dot(val032, u_B)
    val041 = np.dot(val034, u_B)
    val042 = np.dot(val036, u_B)
    val043 = np.dot(val037, u_B)
    H1A_H2 = 1.0*val042.T + \
            -1.0*val041.T + \
            1.0*val039 + \
            -1.0*val040 + \
            1.0*np.dot(val027, v_A) + \
            1.0*val039.T + \
            1.0*val038 + \
            -1.0*np.dot(val028, u_B) + \
            1.0*np.dot(val024, u_B) + \
            1.0*val042 + \
            -1.0*np.dot(val023, u_B) + \
            -1.0*val040.T + \
            1.0*val043.T + \
            1.0*np.dot(val021, v_A) + \
            1.0*val043 + \
            -1.0*val041 + \
            -1.0*np.dot(val022, v_A) + \
            1.0*val038.T
    val044 = np.dot(np.dot(u_A.T, val010), v_B)
    val045 = np.dot(np.dot(u_A.T, val029.T), u_A)
    val046 = np.dot(np.dot(v_B.T, val031.T), u_A)
    val047 = np.dot(np.dot(v_B.T, val033), v_B)
    val048 = np.dot(np.dot(v_B.T, val035), u_A)
    val049 = np.dot(np.dot(v_B.T, val016.T), u_A)
    H1B_H2 = 1.0*val047.T + \
            -1.0*val044.T + \
            -1.0*val045.T + \
            -1.0*val044 + \
            -1.0*np.dot(np.dot(u_A.T, val026), u_A) + \
            -1.0*np.dot(np.dot(u_A.T, val004), u_A) + \
            -1.0*np.dot(np.dot(v_B.T, val008), v_B) + \
            -1.0*val049.T + \
            1.0*val046 + \
            1.0*val047 + \
            -1.0*val048 + \
            -1.0*val045 + \
            1.0*np.dot(np.dot(v_B.T, val011), v_B) + \
            -1.0*val049 + \
            1.0*val046.T + \
            -1.0*val048.T + \
            1.0*np.dot(np.dot(v_B.T, val007), v_B) + \
            1.0*np.dot(np.dot(u_A.T, val005), u_A)
    val050 = np.tensordot(u_A, w_A, axes=(0,0))
    val051 = np.tensordot(np.tensordot(np.tensordot(u_A, val050, axes=(0,2)), v_A, axes=(3,0)), u_A, axes=(2,0))
    val052 = np.tensordot(v_B, w_B, axes=(0,0))
    val053 = np.tensordot(np.tensordot(np.tensordot(v_B, val052, axes=(0,2)), u_B, axes=(3,0)), v_B, axes=(2,0))
    val054 = np.tensordot(v_B, w_AB, axes=(0,2))
    val055 = np.tensordot(u_A, val054, axes=(0,2))
    val056 = np.tensordot(v_B, y_A, axes=(0,2))
    val057 = np.tensordot(v_B, y_B, axes=(0,0))
    val058 = np.tensordot(v_B, val057, axes=(0,1))
    val059 = np.tensordot(u_A, (-np.tensordot(u_A, y_A, axes=(0,0))), axes=(0,1))
    val060 = np.tensordot(v_B, x, axes=(0,2))
    val061 = 0.5*np.tensordot(np.tensordot(val058, u_B, axes=(3,0)), u_A, axes=(2,0)) + \
            0.5*np.transpose(val053, (1, 0, 2, 3)) + \
            0.5*np.tensordot(np.tensordot(val059, u_B, axes=(2,0)), u_A, axes=(2,0)) + \
            -1.0*np.transpose(np.tensordot((-np.tensordot(np.tensordot(u_A, np.tensordot(v_B, y_B, axes=(0,3)), axes=(0,3)), u_B, axes=(3,0))), v_B, axes=(2,0)), (1, 0, 2, 3)) + \
            0.5*np.transpose(val051, (1, 0, 2, 3)) + \
            1.0*np.transpose(np.tensordot(np.tensordot(val055, u_B, axes=(3,0)), u_A, axes=(2,0)), (1, 0, 2, 3)) + \
            -1.0*np.transpose(np.tensordot(np.tensordot(val055, v_A, axes=(2,0)), v_B, axes=(2,0)), (1, 0, 2, 3)) + \
            0.5*np.tensordot(np.tensordot(np.tensordot(v_B, val060, axes=(0,3)), v_A, axes=(3,0)), u_A, axes=(2,0)) + \
            -0.5*np.tensordot(np.tensordot(val058, v_A, axes=(2,0)), v_B, axes=(2,0)) + \
            -0.5*val053 + \
            0.5*np.tensordot(np.tensordot(np.tensordot(u_A, np.tensordot(u_A, x, axes=(0,0)), axes=(0,1)), u_B, axes=(3,0)), v_B, axes=(2,0)) + \
            1.0*np.transpose(np.tensordot((-np.tensordot(np.tensordot(u_A, val056, axes=(0,3)), v_A, axes=(3,0))), u_A, axes=(2,0)), (1, 0, 2, 3)) + \
            -0.5*np.tensordot(np.tensordot(val059, v_A, axes=(3,0)), v_B, axes=(2,0)) + \
            -0.5*val051
    H2yB_H2 = val061 - np.transpose(val061, (1,0,2,3))
    val062 = np.tensordot(v_A, w_A, axes=(0,0))
    val063 = np.tensordot(np.tensordot(v_A, val062, axes=(0,2)), u_A, axes=(3,0))
    val064 = np.tensordot(val063, v_A, axes=(2,0))
    val065 = np.tensordot(u_B, w_B, axes=(0,0))
    val066 = np.tensordot(np.tensordot(u_B, val065, axes=(0,2)), v_B, axes=(3,0))
    val067 = np.tensordot(val066, u_B, axes=(2,0))
    val068 = np.tensordot(v_A, w_AB, axes=(0,0))
    val069 = np.tensordot(u_B, val068, axes=(0,3))
    val070 = np.tensordot(val069, v_B, axes=(3,0))
    val071 = np.tensordot(v_A, y_A, axes=(0,0))
    val072 = np.tensordot(v_A, val071, axes=(0,1))
    val073 = np.tensordot(val072, v_B, axes=(2,0))
    val074 = np.tensordot(v_A, y_B, axes=(0,2))
    val075 = (-np.tensordot(np.tensordot(u_B, val074, axes=(0,3)), v_B, axes=(3,0)))
    val076 = np.tensordot(np.tensordot(u_B, np.tensordot(v_A, y_A, axes=(0,3)), axes=(0,3)), u_A, axes=(3,0))
    val077 = np.tensordot(u_B, (-np.tensordot(u_B, y_B, axes=(0,0))), axes=(0,1))
    val078 = np.tensordot(val077, v_B, axes=(3,0))
    val079 = np.tensordot(v_A, x, axes=(0,0))
    val080 = np.tensordot(np.tensordot(v_A, val079, axes=(0,1)), v_B, axes=(3,0))
    val081 = np.tensordot(np.tensordot(u_B, np.tensordot(u_B, x, axes=(0,2)), axes=(0,3)), u_A, axes=(3,0))
    val082 = -0.5*np.tensordot(val073, v_A, axes=(2,0)) + \
            1.0*np.transpose(np.tensordot(np.tensordot(val069, u_A, axes=(2,0)), u_B, axes=(2,0)), (1, 0, 2, 3)) + \
            1.0*np.transpose(np.tensordot(val075, u_B, axes=(2,0)), (1, 0, 2, 3)) + \
            0.5*np.transpose(val067, (1, 0, 2, 3)) + \
            -0.5*np.tensordot(val078, v_A, axes=(2,0)) + \
            0.5*np.tensordot(np.tensordot(val077, u_A, axes=(2,0)), u_B, axes=(2,0)) + \
            0.5*np.tensordot(np.tensordot(val072, u_A, axes=(3,0)), u_B, axes=(2,0)) + \
            0.5*np.transpose(val064, (1, 0, 2, 3)) + \
            -1.0*np.transpose(np.tensordot(val070, v_A, axes=(2,0)), (1, 0, 2, 3)) + \
            -0.5*val064 + \
            -0.5*val067 + \
            0.5*np.tensordot(val080, u_B, axes=(2,0)) + \
            0.5*np.tensordot(val081, v_A, axes=(2,0)) + \
            -1.0*np.transpose(np.tensordot((-val076), v_A, axes=(2,0)), (1, 0, 2, 3))
    H2yA_H2 = val082 - np.transpose(val082, (1,0,2,3))
    val083 = np.tensordot(np.tensordot((-np.tensordot(u_A, val056, axes=(0,1))), u_A, axes=(3,0)), u_A, axes=(2,0))
    val084 = np.tensordot(np.tensordot(np.tensordot(v_B, val057, axes=(0,3)), v_B, axes=(2,0)), u_A, axes=(2,0))
    val085 = np.tensordot(np.tensordot((-np.tensordot(u_A, val060, axes=(0,1))), v_B, axes=(3,0)), u_A, axes=(2,0))
    val086 = 0.25*np.transpose(val085, (1, 0, 2, 3)) + \
            0.5*np.tensordot(np.tensordot(np.tensordot(v_B, val052, axes=(0,1)), v_B, axes=(3,0)), v_B, axes=(2,0)) + \
            0.5*np.transpose(val083, (1, 0, 2, 3)) + \
            0.5*np.transpose(val084, (1, 0, 2, 3)) + \
            -1.0*np.transpose(np.tensordot(np.tensordot(np.tensordot(v_B, val054, axes=(0,3)), u_A, axes=(3,0)), u_A, axes=(2,0)), (0, 1, 3, 2)) + \
            0.5*np.transpose(val084, (0, 1, 3, 2)) + \
            0.5*np.transpose(val083, (0, 1, 3, 2)) + \
            0.25*np.transpose(val085, (0, 1, 3, 2)) + \
            0.5*np.tensordot(np.tensordot(np.tensordot(u_A, val050, axes=(0,1)), u_A, axes=(3,0)), u_A, axes=(2,0))
    H2wB_H2 = val086 + np.transpose(val086, (2,3,0,1))
    val087 = np.tensordot(v_A, val062, axes=(0,1))
    val088 = np.tensordot(u_B, val065, axes=(0,1))
    val089 = np.tensordot(v_A, val068, axes=(0,1))
    val090 = np.tensordot(v_A, val071, axes=(0,3))
    val091 = np.tensordot(np.tensordot(val090, v_A, axes=(2,0)), u_B, axes=(2,0))
    val092 = (-np.tensordot(u_B, val074, axes=(0,1)))
    val093 = np.tensordot(np.tensordot(val092, u_B, axes=(3,0)), u_B, axes=(2,0))
    val094 = (-np.tensordot(u_B, val079, axes=(0,2)))
    val095 = np.tensordot(np.tensordot(val094, v_A, axes=(2,0)), u_B, axes=(2,0))
    val096 = 0.5*np.tensordot(np.tensordot(val087, v_A, axes=(3,0)), v_A, axes=(2,0)) + \
            0.5*np.transpose(val093, (0, 1, 3, 2)) + \
            0.5*np.transpose(val091, (0, 1, 3, 2)) + \
            0.5*np.transpose(val091, (1, 0, 2, 3)) + \
            0.25*np.transpose(val095, (1, 0, 2, 3)) + \
            0.5*np.transpose(val093, (1, 0, 2, 3)) + \
            0.25*np.transpose(val095, (0, 1, 3, 2)) + \
            0.5*np.tensordot(np.tensordot(val088, u_B, axes=(3,0)), u_B, axes=(2,0)) + \
            -1.0*np.transpose(np.tensordot(np.tensordot(val089, u_B, axes=(3,0)), u_B, axes=(2,0)), (0, 1, 3, 2))
    H2wA_H2 = val096 + np.transpose(val096, (2,3,0,1))
    val097 = np.tensordot(val063, u_A, axes=(2,0))
    val098 = np.tensordot(val066, v_B, axes=(2,0))
    val099 = -1.0*np.transpose(np.tensordot(val070, u_A, axes=(2,0)), (1, 0, 2, 3)) + \
            0.5*np.transpose(np.tensordot(val075, v_B, axes=(2,0)), (1, 0, 2, 3)) + \
            -0.5*np.tensordot(val078, u_A, axes=(2,0)) + \
            0.5*np.transpose(val098, (0, 1, 3, 2)) + \
            0.5*np.transpose(val097, (0, 1, 3, 2)) + \
            -0.5*np.tensordot(val073, u_A, axes=(2,0)) + \
            0.5*np.transpose(np.tensordot(val076, u_A, axes=(2,0)), (1, 0, 2, 3)) + \
            0.25*np.tensordot(val081, u_A, axes=(2,0)) + \
            0.25*np.tensordot(val080, v_B, axes=(2,0))
    val100 = val099 - np.transpose(val099, (1,0,2,3))
    H2x_H2 = val100 - np.transpose(val100, (0,1,3,2))
    val101 = np.tensordot(np.tensordot(np.tensordot(u_B, val068, axes=(0,2)), u_A, axes=(2,0)), v_B, axes=(2,0))
    val102 = np.tensordot(np.tensordot(val090, v_B, axes=(3,0)), u_A, axes=(2,0))
    val103 = np.tensordot(np.tensordot(np.tensordot(u_B, val071, axes=(0,2)), u_A, axes=(3,0)), u_A, axes=(2,0))
    val104 = np.tensordot(np.tensordot(val092, v_B, axes=(2,0)), v_B, axes=(2,0))
    val105 = np.tensordot(np.tensordot((-np.tensordot(u_B, np.tensordot(u_B, y_B, axes=(0,3)), axes=(0,1))), v_B, axes=(2,0)), u_A, axes=(2,0))
    val106 = np.tensordot(np.tensordot(val094, v_B, axes=(3,0)), u_A, axes=(2,0))
    H2wAB_H2 = 1.0*np.transpose(val105, (0, 1, 3, 2)) + \
            1.0*val098 + \
            1.0*np.transpose(val103, (1, 0, 2, 3)) + \
            1.0*np.transpose(val103, (0, 1, 3, 2)) + \
            1.0*np.tensordot(np.tensordot(val089, v_B, axes=(3,0)), v_B, axes=(2,0)) + \
            -1.0*np.transpose(val106, (1, 0, 2, 3)) + \
            -1.0*np.tensordot(np.tensordot(val087, u_A, axes=(3,0)), u_A, axes=(2,0)) + \
            -1.0*np.transpose(val102, (0, 1, 3, 2)) + \
            -1.0*np.tensordot(np.tensordot(val088, v_B, axes=(3,0)), v_B, axes=(2,0)) + \
            -1.0*np.transpose(val101, (0, 1, 3, 2)) + \
            -1.0*np.transpose(val104, (1, 0, 2, 3)) + \
            1.0*np.tensordot(np.tensordot(np.tensordot(u_B, np.tensordot(u_B, w_AB, axes=(0,2)), axes=(0,3)), u_A, axes=(3,0)), u_A, axes=(2,0)) + \
            1.0*np.transpose(val105, (1, 0, 2, 3)) + \
            1.0*val097 + \
            -1.0*np.transpose(val106, (0, 1, 3, 2)) + \
            -1.0*np.transpose(val101, (1, 0, 2, 3)) + \
            -1.0*np.transpose(val102, (1, 0, 2, 3)) + \
            -1.0*np.transpose(val104, (0, 1, 3, 2))

    return H0_H0 + H0_H1 + H0_H2, np.asarray([H1A_H1 + H1A_H2, H1B_H1 + H1B_H2]), \
            (H1D_H1 + H1D_H2)[np.newaxis], np.asarray([H2wA_H2, H2wB_H2, H2wAB_H2]), \
            np.asarray([H2yA_H2, H2yB_H2]), H2x_H2[np.newaxis]
