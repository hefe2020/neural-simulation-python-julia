import numpy as np
from python.fluid.divergence import divergence
from python.fluid.jacobi import jacobi
from python.fluid.laplacian import laplacian

# u = np.zeros((3, 4))
# v = np.zeros((4, 3))

# u[1, 2] = 1.0

# u_div = divergence(u, v, h=1.0)

# jaco = jacobi(u_div, h=1.0, iter=100)
# print(jaco)

random = np.random.randn(2, 3)
print(random)
stack = np.vstack([random[-1, :], random, random[0, :]])
print(stack)
