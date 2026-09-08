import numpy as np
def jacobi(b, h=1.0, iter=100):
    p = np.zeros_like(b, dtype=float)
   
    for _ in range(iter):
        p_left = np.column_stack((p[:, -1], p))
        p_right = np.column_stack((p, p[:, 0]))
        p_top = np.vstack((p[-1, :], p))
        p_bottom = np.vstack((p, p[0, :]))

        left = p_left[:, :-1]
        right = p_right[:, 1:]
        top = p_top[:-1, :]
        bottom = p_bottom[1:, :]

        p = (left + right + top + bottom -h**2 * b) / 4.0

    return p

        
