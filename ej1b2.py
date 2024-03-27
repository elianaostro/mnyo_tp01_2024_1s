# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.interpolate import interp2d, CubicSpline

# def f(x1, x2):
#     return (
#                 0.75 * np.exp ( - ( (10 * x1 - 2 ) ** 2 ) / 4 - ( ( 9 * x2 - 2 ) ** 2 ) / 4 ) + 
#                 0.65 * np.exp ( - ( ( 9 * x1 + 1 ) ** 2 ) / 9 - ( (10 * x2 + 1 ) ** 2 ) / 2 ) + 
#                 0.55 * np.exp ( - ( ( 9 * x1 - 6 ) ** 2 ) / 4 - ( ( 9 * x2 - 3 ) ** 2 ) / 4 ) - 
#                 0.01 * np.exp ( - ( ( 9 * x1 - 7 ) ** 2 ) / 4 - ( ( 9 * x2 - 3 ) ** 2 ) / 4 )
#             )

# # Generate grid points
# x1 = np.linspace(-1, 1, 50)
# x2 = np.linspace(-1, 1, 50)
# X1, X2 = np.meshgrid(x1, x2)
# Z = f(X1, X2)

# # Generate Chebyshev nodes
# def chebyshev_nodes(n, a=-1, b=1):
#     """Generate Chebyshev nodes in the interval [a, b]."""
#     k = np.arange(1, n + 1)
#     nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) * np.pi / (2 * n))
#     return np.sort(nodes)

# # Number of Chebyshev nodes
# n_cheb = 20
# cheb_nodes_x1 = chebyshev_nodes(n_cheb)
# cheb_nodes_x2 = chebyshev_nodes(n_cheb)
# cheb_nodes_X1, cheb_nodes_X2 = np.meshgrid(cheb_nodes_x1, cheb_nodes_x2)

# # Interpolate using Chebyshev nodes
# interp_func_cheb = interp2d(cheb_nodes_x1, cheb_nodes_x2, f(cheb_nodes_X1, cheb_nodes_X2), kind='cubic')
# Z_interp_cheb = interp_func_cheb(x1, x2)

# # Interpolate using linear method
# interp_func_linear = interp2d(x1, x2, Z, kind='linear')
# Z_interp_linear = interp_func_linear(x1, x2)

# # Compute absolute errors
# error_cheb = np.abs(f(X1, X2) - Z_interp_cheb)
# error_linear = np.abs(f(X1, X2) - Z_interp_linear)

# # Plot
# fig = plt.figure(figsize=(16, 8))
# gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])

# # Plot Chebyshev interpolation
# ax1 = fig.add_subplot(gs[0, 0], projection='3d')
# ax1.plot_surface(X1, X2, Z_interp_cheb, cmap='plasma')
# ax1.set_title('Chebyshev Interpolation')

# # Plot linear interpolation
# ax2 = fig.add_subplot(gs[0, 1], projection='3d')
# ax2.plot_surface(X1, X2, Z_interp_linear, cmap='plasma')
# ax2.set_title('Linear Interpolation')

# plt.tight_layout()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d

def f(x1, x2):
    return (
                0.75 * np.exp ( - ( (10 * x1 - 2 ) ** 2 ) / 4 - ( ( 9 * x2 - 2 ) ** 2 ) / 4 ) + 
                0.65 * np.exp ( - ( ( 9 * x1 + 1 ) ** 2 ) / 9 - ( (10 * x2 + 1 ) ** 2 ) / 2 ) + 
                0.55 * np.exp ( - ( ( 9 * x1 - 6 ) ** 2 ) / 4 - ( ( 9 * x2 - 3 ) ** 2 ) / 4 ) - 
                0.01 * np.exp ( - ( ( 9 * x1 - 7 ) ** 2 ) / 4 - ( ( 9 * x2 - 3 ) ** 2 ) / 4 )
            )

# Generate grid points
x1 = np.linspace(-1, 1, 50)
x2 = np.linspace(-1, 1, 50)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

# Generate Chebyshev nodes
def chebyshev_nodes(n, a=-1, b=1):
    """Generate Chebyshev nodes in the interval [a, b]."""
    k = np.arange(1, n + 1)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) * np.pi / (2 * n))
    return np.sort(nodes)

# Number of Chebyshev nodes
n_cheb = 20
cheb_nodes_x1 = chebyshev_nodes(n_cheb)
cheb_nodes_x2 = chebyshev_nodes(n_cheb)
cheb_nodes_X1, cheb_nodes_X2 = np.meshgrid(cheb_nodes_x1, cheb_nodes_x2)

# Interpolate using Chebyshev nodes
interp_func_cheb = interp2d(cheb_nodes_x1, cheb_nodes_x2, f(cheb_nodes_X1, cheb_nodes_X2), kind='cubic')
Z_interp_cheb = interp_func_cheb(x1, x2)

# Interpolate using linear method
interp_func_linear = interp2d(x1, x2, Z, kind='linear')
Z_interp_linear = interp_func_linear(x1, x2)

# Compute absolute errors
error_cheb = np.abs(f(X1, X2) - Z_interp_cheb)
error_linear = np.abs(f(X1, X2) - Z_interp_linear)

# Plot
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 4)

# Plot Chebyshev interpolation
ax1 = fig.add_subplot(gs[0, 0], projection='3d')
ax1.plot_surface(X1, X2, Z_interp_cheb, cmap='plasma')
ax1.set_title('Chebyshev Interpolation')

# Plot linear interpolation
ax2 = fig.add_subplot(gs[0, 1], projection='3d')
ax2.plot_surface(X1, X2, Z_interp_linear, cmap='plasma')
ax2.set_title('Linear Interpolation')

# Plot Chebyshev error
ax3 = fig.add_subplot(gs[1, 0])
cheb_error_plot = ax3.imshow(error_cheb, extent=(-1, 1, -1, 1), cmap='plasma')
ax3.set_title('Chebyshev Error')
plt.colorbar(cheb_error_plot, ax=ax3)

plt.tight_layout()
plt.show()
