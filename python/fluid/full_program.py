import numpy as np

def divergence(u, v, h):
    """
    We first compute the divergence machinery.
    U has shape (Ny, Nx+1)
    V has shape (Ny+1, Nx)

    This will return (Ny, Nx) shape, divergence at pressure centers
    """
    du_dx = (u[:, 1:] - u[:, :-1]) / h
    dv_dy = (v[1:, :] - v[:-1, :]) / h

    return du_dx + dv_dy

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

def pressure_gradient_staggered(p, h=1.0): 
    p_field_x = np.column_stack([p[:, -1], p, p[:, 0]]) 
    p_field_y = np.vstack([p[-1, :], p, p[0, :]]) 
    p_grid_x = (p_field_x[:, 1:] - p_field_x[:, :-1]) / h 
    p_grid_y = (p_field_y[1:, :] - p_field_y[:-1, :]) / h 
    return p_grid_x, p_grid_y

def project_velocity(u_star, v_star, dt=1.0, rho=1.0, h=1.0, iter=100):
    velocity_star_div = divergence(u_star, v_star, h)
    b = (rho / dt) * velocity_star_div
    jacobi_pressure = jacobi(b, h, iter=iter)
    pressure_div_x, pressure_div_y = pressure_gradient_staggered(jacobi_pressure, h=h)
    u = u_star - (dt / rho) * pressure_div_x
    v = v_star - (dt / rho) * pressure_div_y

    return u, v, jacobi_pressure

if __name__ == "__main__":
    nx = 3
    ny = 3

    h = 1.0
    dt = 1.0
    rho = 1.0

    u_star = np.zeros((ny, nx + 1))
    v_star = np.zeros((ny + 1, nx))

    u_star[1, 2] = 1.0


    print("Tentative u:")
    print(u_star)


    div_before = divergence(
        u_star,
        v_star,
        h,
    )

    print("\nDivergence before projection:")
    print(div_before)


    u, v, p = project_velocity(
        u_star,
        v_star,
        dt=dt,
        rho=rho,
        h=h,
        iter=100,
    )


    print("\nPressure:")
    print(p)


    div_after = divergence(
        u,
        v,
        h,
    )

    print("\nDivergence after projection:")
    print(div_after)


    print(
        "\nL2 divergence before:",
        np.linalg.norm(div_before),
    )

    print(
        "L2 divergence after:",
        np.linalg.norm(div_after),
    )

