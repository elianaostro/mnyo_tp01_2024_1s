import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d
import matplotlib.gridspec as gridspec

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

# Original function plot
fig1 = plt.figure(figsize=(12, 9))
ax_original = fig1.add_subplot(111, projection='3d')
ax_original.plot_surface(X1, X2, Z, cmap='viridis')
ax_original.set_title('Original Function')

plt.show()

# Linear interpolation plot
fig2 = plt.figure(figsize=(12, 9))
ax_interp_linear = fig2.add_subplot(111, projection='3d')
ax_interp_linear.plot_surface(X1_new2, X2_new2, Z_new2, cmap='plasma')
ax_interp_linear.set_title('Linear Interpolation')

plt.show()

# Linear error plot
fig3 = plt.figure(figsize=(8, 6))
ax_error_linear = fig3.add_subplot(111)
linear_error_plot = ax_error_linear.imshow(error_linear, extent=(-1, 1, -1, 1), cmap='plasma')
linear_colorbar = plt.colorbar(linear_error_plot, ax=ax_error_linear)
ax_error_linear.set_title('Absolute Error (Linear Interpolation)')

# Set scientific notation for the colorbar ticks
linear_colorbar.formatter.set_powerlimits((-3, 3))
linear_colorbar.update_ticks()

plt.show()

# Cubic interpolation plot
fig4 = plt.figure(figsize=(12, 9))
ax_interp_cubic = fig4.add_subplot(111, projection='3d')
ax_interp_cubic.plot_surface(X1_new, X2_new, Z_new, cmap='plasma')
ax_interp_cubic.set_title('Cubic Interpolation')

plt.show()

# Cubic error plot
fig5 = plt.figure(figsize=(8, 6))
ax_error_cubic = fig5.add_subplot(111)
cubic_error_plot = ax_error_cubic.imshow(error_cubic, extent=(-1, 1, -1, 1), cmap='plasma')
cubic_colorbar = plt.colorbar(cubic_error_plot, ax=ax_error_cubic)
ax_error_cubic.set_title('Absolute Error (Cubic Interpolation)')

# Set scientific notation for the colorbar ticks
cubic_colorbar.formatter.set_powerlimits((-3, 3))
cubic_colorbar.update_ticks()

plt.show()

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


# Generate grid points
x1_cheb = np.linspace(-1, 1, 50)
x2_cheb = np.linspace(-1, 1, 50)
X1_cheb, X2_cheb = np.meshgrid(x1_cheb, x2_cheb)
Z_cheb = f(X1_cheb, X2_cheb)

interp_func_cheb = interp2d(x1_cheb, x2_cheb, Z_cheb, kind='cubic')

# New grid for interpolated values
x_new_cheb = np.linspace(-1, 1, 15)
X1_new_cheb, X2_new_cheb = np.meshgrid(x_new_cheb, x_new_cheb)
Z_new_cheb = interp_func_cheb(x_new_cheb, x_new_cheb)

# Interpolate the function using a different method
interp_func2_cheb = interp2d(x1_cheb, x2_cheb, Z_cheb, kind='linear')
# New grid for interpolated values
x_new2_cheb = np.linspace(-1, 1, 15)
X1_new2_cheb, X2_new2_cheb = np.meshgrid(x_new2_cheb, x_new2_cheb)
Z_new2_cheb = interp_func2_cheb(x_new2_cheb, x_new2_cheb)

# Compute the absolute difference (error) between the original and interpolated values
error_cubic_cheb = np.abs(f(X1_new_cheb, X2_new_cheb) - Z_new_cheb)

# Interpolate the function using 'linear' method
interp_func_linear_cheb = interp2d(x1_cheb, x2_cheb, Z_cheb, kind='linear')
Z_interp_linear_cheb = interp_func_linear_cheb(x_new_cheb, x_new_cheb)

# Compute the absolute difference (error) between the original and interpolated values
error_linear_cheb = np.abs(f(X1_new_cheb, X2_new_cheb) - Z_interp_linear_cheb)



# Original function plot
fig1 = plt.figure(figsize=(12, 9))
ax_original = fig1.add_subplot(111, projection='3d')
ax_original.plot_surface(X1_cheb, X2_cheb, Z_cheb, cmap='viridis')
ax_original.set_title('Original Function Chebysheb')
plt.show()

# Linear interpolation plot
fig2 = plt.figure(figsize=(12, 9))
ax_interp_linear_cheb = fig2.add_subplot(111, projection='3d')
ax_interp_linear_cheb.plot_surface(X1_new2_cheb, X2_new2_cheb, Z_new2_cheb, cmap='plasma')
ax_interp_linear_cheb.set_title('Linear Interpolation Chebysheb')

plt.show()

# Linear error plot
fig3 = plt.figure(figsize=(8, 6))
ax_error_linear = fig3.add_subplot(111)
linear_error_plot = ax_error_linear.imshow(error_linear_cheb, extent=(-1, 1, -1, 1), cmap='plasma')
linear_colorbar = plt.colorbar(linear_error_plot, ax=ax_error_linear)
ax_error_linear.set_title('Absolute Error (Linear Interpolation) Chebysheb')

# Set scientific notation for the colorbar ticks
linear_colorbar.formatter.set_powerlimits((-3, 3))
linear_colorbar.update_ticks()

plt.show()

# Cubic interpolation plot
fig4 = plt.figure(figsize=(12, 9))
ax_interp_cubic = fig4.add_subplot(111, projection='3d')
ax_interp_cubic.plot_surface(X1_new_cheb, X2_new_cheb, Z_new_cheb, cmap='plasma')
ax_interp_cubic.set_title('Cubic Interpolation')

plt.show()

# Cubic error plot
fig5 = plt.figure(figsize=(8, 6))
ax_error_cubic = fig5.add_subplot(111)
cubic_error_plot = ax_error_cubic.imshow(error_cubic_cheb, extent=(-1, 1, -1, 1), cmap='plasma')
cubic_colorbar = plt.colorbar(cubic_error_plot, ax=ax_error_cubic)
ax_error_cubic.set_title('Absolute Error (Cubic Interpolation)')

# Set scientific notation for the colorbar ticks
cubic_colorbar.formatter.set_powerlimits((-3, 3))
cubic_colorbar.update_ticks()

plt.show()