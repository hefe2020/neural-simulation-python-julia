def laplacian(p, h):
    p_left = np.column_stack((p[:, -1], p))
    p_right = np.column_stack((p, p[:, 0]))
    p_top = np.vstack((p[-1, :], p))
    p_bottom = np.vstack((p, p[0, :]))

    left = np.zeros_like(p)
    right = np.zeros_like(p)
    top = np.zeros_like(p)
    bottom = np.zeros_like(p)

    left[:, :] = p_left[:, :-1]
    right[:, :] = p_right[:, 1:]
    top[:, :] = p_top[:-1, :]
    bottom[:, :] = p_bottom[1:, :]

    p_laplace = (left + right + top + bottom -4.0 * p) / h**2
    return p_laplace 