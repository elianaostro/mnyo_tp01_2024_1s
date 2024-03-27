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

# Interpolate the function
interp_func = interp2d(x1, x2, Z, kind='cubic')

# New grid for interpolated values
x_new = np.linspace(-1, 1, 15)
X1_new, X2_new = np.meshgrid(x_new, x_new)
Z_new = interp_func(x_new, x_new)

# Interpolate the function using a different method
interp_func2 = interp2d(x1, x2, Z, kind='linear')
# New grid for interpolated values
x_new2 = np.linspace(-1, 1, 15)
X1_new2, X2_new2 = np.meshgrid(x_new2, x_new2)
Z_new2 = interp_func2(x_new2, x_new2)

# Compute the absolute difference (error) between the original and interpolated values
error_cubic = np.abs(f(X1_new, X2_new) - Z_new)

# Interpolate the function using 'linear' method
interp_func_linear = interp2d(x1, x2, Z, kind='linear')
Z_interp_linear = interp_func_linear(x_new, x_new)

# Compute the absolute difference (error) between the original and interpolated values
error_linear = np.abs(f(X1_new, X2_new) - Z_interp_linear)

# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.imshow(error_cubic, extent=(-1, 1, -1, 1), cmap='plasma')
# plt.colorbar(label='Absolute Error')
# plt.title('Absolute Error (cubic interpolation)')
# plt.xlabel('X1')
# plt.ylabel('X2')

# plt.subplot(1, 2, 2)
# plt.imshow(error_linear, extent=(-1, 1, -1, 1), cmap='plasma')
# plt.colorbar(label='Absolute Error')
# plt.title('Absolute Error (linear interpolation)')
# plt.xlabel('X1')
# plt.ylabel('X2')

# plt.tight_layout()
# plt.show()


# # Plot original function
# fig = plt.figure(figsize=(12, 6))

# ax1 = fig.add_subplot(1, 3, 1, projection='3d')
# ax1.plot_surface(X1, X2, Z, cmap='viridis')
# ax1.set_title('Original Function')


# # Plot interpolated function using cubic interpolation
# ax2 = fig.add_subplot(1, 3, 2, projection='3d')
# ax2.plot_surface(X1_new, X2_new, Z_new, cmap='plasma')
# ax2.plot_surface(X1, X2, Z, color=(0, 0, 1, 0.2))
# ax2.set_title('Cubic Interpolation')


# # Plot interpolated function using linear interpolation
# ax3 = fig.add_subplot(1, 3, 3, projection='3d')
# ax3.plot_surface(X1_new2, X2_new2, Z_new2, cmap='plasma')
# ax3.plot_surface(X1, X2, Z, color=(0, 0, 1, 0.2))
# ax3.set_title('Linear Interpolation')

# plt.show()

import matplotlib.gridspec as gridspec

# Create a 1x2 grid
fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 2])

# Original function on the left
ax_original = fig.add_subplot(gs[0, 0], projection='3d')
ax_original.plot_surface(X1, X2, Z, cmap='viridis')
ax_original.set_title('Original Function')

# Subgrid for interpolation and error plots on the right
gs_right = gs[0, 1].subgridspec(2, 2, width_ratios=[2, 1])

# Plot interpolated function using cubic interpolation
ax_interp_cubic = fig.add_subplot(gs_right[0, 0], projection='3d')
ax_interp_cubic.plot_surface(X1_new, X2_new, Z_new, cmap='plasma')
ax_interp_cubic.set_title('Cubic Interpolation')

# Plot error for cubic interpolation
ax_error_cubic = fig.add_subplot(gs_right[0, 1])
cubic_error_plot = ax_error_cubic.imshow(error_cubic, extent=(-1, 1, -1, 1), cmap='plasma')
cubic_colorbar = plt.colorbar(cubic_error_plot, ax=ax_error_cubic, shrink=0.5, aspect=10)
ax_error_cubic.set_title('Absolute Error (Cubic Interpolation)')

# Set scientific notation for the colorbar ticks
cubic_colorbar.formatter.set_powerlimits((-3, 3))
cubic_colorbar.update_ticks()

# Plot interpolated function using linear interpolation
ax_interp_linear = fig.add_subplot(gs_right[1, 0], projection='3d')
ax_interp_linear.plot_surface(X1_new2, X2_new2, Z_new2, cmap='plasma')
ax_interp_linear.set_title('Linear Interpolation')

# Plot error for linear interpolation
ax_error_linear = fig.add_subplot(gs_right[1, 1])
linear_error_plot = ax_error_linear.imshow(error_linear, extent=(-1, 1, -1, 1), cmap='plasma')
linear_colorbar = plt.colorbar(linear_error_plot, ax=ax_error_linear, shrink=0.5, aspect=10)
ax_error_linear.set_title('Absolute Error (Linear Interpolation)')

# Set scientific notation for the colorbar ticks
linear_colorbar.formatter.set_powerlimits((-3, 3))
linear_colorbar.update_ticks()

plt.tight_layout()
plt.show()

# Generate n non-equidistant Chebyshev nodes
def chebyshev_nodes(n, a=-4, b=4):
    """Generate n non-equidistant Chebyshev nodes in the interval [a, b]."""
    k = np.arange(1, n + 1)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) * np.pi / (2 * n))
    return np.sort(nodes)

# Generate nodes in R^2
def generate_chebyshev_nodes(n, a=-1, b=1):
    """Generate n non-equidistant Chebyshev nodes in the interval [a, b] in R^2."""
    x1_nodes = chebyshev_nodes(n, a, b)
    x2_nodes = chebyshev_nodes(n, a, b)
    nodes = np.column_stack((x1_nodes, x2_nodes))
    return nodes
n = 20  # Number of nodes
nodes = np.zeros((n, 2))  # Initialize array to store nodes
x1_nodes = chebyshev_nodes(n) 

# Plot original function
fig = plt.figure(figsize=(18, 6))

# Plot original function
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
ax1.plot_surface(X1, X2, Z, cmap='viridis')
ax1.set_title('Original Function')

# Plot interpolated function using cubic interpolation
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.plot_surface(X1_new, X2_new, Z_new, cmap='viridis')
ax2.set_title('Cubic Interpolation')

# Plot interpolated function using linear interpolation
ax3 = fig.add_subplot(1, 3, 3, projection='3d')
ax3.plot_surface(X1_new2, X2_new2, Z_new2, cmap='viridis')
ax3.set_title('Linear Interpolation')

plt.show()
