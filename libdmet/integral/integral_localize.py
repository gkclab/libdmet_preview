import numpy as np

def transform(v_A, v_B, h_0, h_A, h_B, D, w_A, w_B, w_AB, y_A, y_B, x):
    H0_H0 = h_0
    H1D_H1 = np.dot(np.dot(v_A.T, D), v_B)
    H1A_H1 = np.dot(np.dot(v_A.T, h_A), v_A)
    H1B_H1 = np.dot(np.dot(v_B.T, h_B), v_B)
    H2yB_H2 = (-np.tensordot(np.tensordot(np.tensordot(v_B, np.tensordot(v_B, y_B, axes=(0,0)), axes=(0,1)), v_A, axes=(2,0)), v_B, axes=(2,0)))
    H2x_H2 = np.tensordot(np.tensordot(np.tensordot(v_A, np.tensordot(v_A, x, axes=(0,0)), axes=(0,1)), v_B, axes=(3,0)), v_B, axes=(2,0))
    H2wB_H2 = np.tensordot(np.tensordot(np.tensordot(v_B, np.tensordot(v_B, w_B, axes=(0,0)), axes=(0,1)), v_B, axes=(3,0)), v_B, axes=(2,0))
    H2wA_H2 = np.tensordot(np.tensordot(np.tensordot(v_A, np.tensordot(v_A, w_A, axes=(0,0)), axes=(0,1)), v_A, axes=(3,0)), v_A, axes=(2,0))
    H2yA_H2 = (-np.tensordot(np.tensordot(np.tensordot(v_A, np.tensordot(v_A, y_A, axes=(0,0)), axes=(0,1)), v_B, axes=(2,0)), v_A, axes=(2,0)))
    H2wAB_H2 = np.tensordot(np.tensordot(np.tensordot(v_A, np.tensordot(v_A, w_AB, axes=(0,0)), axes=(0,1)), v_B, axes=(3,0)), v_B, axes=(2,0))

    return H0_H0, np.asarray([H1A_H1, H1B_H1]), H1D_H1[np.newaxis], \
            np.asarray([H2wA_H2, H2wB_H2, H2wAB_H2]), \
            np.asarray([H2yA_H2, H2yB_H2]), H2x_H2[np.newaxis]