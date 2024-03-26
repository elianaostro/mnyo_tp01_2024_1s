import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define the function to interpolate
def f(x):
    return 0.3**abs(x) * np.sin(4*x) - np.tanh(2*x) + 2

# Choose x-values and calculate corresponding y-values
num_points = 10  # Change this to the desired number of points
x_values = np.linspace(-4, 4, num_points)
y_values = f(x_values)


# Interpolate the function using Lagrange interpolation
def lagrange_basis(x_values, k, x):
    """Calculate the k-th Lagrange basis polynomial."""
    basis = 1
    for i, xi in enumerate(x_values):
        if i != k:
            basis *= (x - xi) / (x_values[k] - xi)
    return basis

def lagrange_interpolation(x_values, y_values, x):
    """Perform Lagrange interpolation to find y-value at x."""
    interpolated_y = 0
    for k, yk in enumerate(y_values):
        interpolated_y += yk * lagrange_basis(x_values, k, x)
    return interpolated_y

x_interpolated_lagrange = np.linspace(-4, 4, 1000)
y_interpolated_lagrange = [lagrange_interpolation(x_values, y_values, x) for x in x_interpolated_lagrange]

# Interpolate the function using cubic spline interpolation
cubic_spline_interpolation = interp1d(x_values, y_values, kind='cubic')
y_interpolated_cubic = cubic_spline_interpolation(x_interpolated_lagrange)


plt.figure(figsize=(10, 6))

# First subplot
plt.subplot(2, 1, 1)
# Plot the original function and the interpolated curves
plt.plot(x_interpolated_lagrange, f(x_interpolated_lagrange), label='Ground Truth', linestyle='--', color='blue')
plt.scatter(x_values, y_values, label='Interpolation Points', color='red')
plt.plot(x_interpolated_lagrange, y_interpolated_lagrange, label='Lagrange Interpolation', color='green')
plt.plot(x_interpolated_lagrange, y_interpolated_cubic, label='Cubic Spline Interpolation', color='orange')

# plt.xlabel('x')
# plt.ylabel('y')
plt.title('Comparison of Interpolation Methods With Equidistant Points')
plt.legend()
plt.grid(True)


plt.subplot(2, 1, 2)
def chebyshev_nodes(n, a=-1, b=1):
    """Generate n non-equidistant Chebyshev nodes in the interval [a, b]."""
    k = np.arange(1, n + 1)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) * np.pi / (2 * n))
    return nodes

# Example usage:
a = -4  # Interval start
b = 4   # Interval end

# Generate Chebyshev nodes
x_values_chebyshev = chebyshev_nodes(num_points, a, b)
y_values_chebyshev = f(x_values_chebyshev)

# Interpolate the function using Lagrange interpolation
x_interpolated_chebyshev = np.linspace(a, b, 1000)
y_interpolated_lagrange_chebyshev = [lagrange_interpolation(x_values_chebyshev, y_values_chebyshev, x) for x in x_interpolated_chebyshev]

# Interpolate the function using cubic spline interpolation
cubic_spline_interpolation_chebyshev = interp1d(x_values_chebyshev, y_values_chebyshev, kind='cubic', bounds_error=False, fill_value="extrapolate")
y_interpolated_cubic = cubic_spline_interpolation_chebyshev(x_interpolated_lagrange)

# Plot the original function and the interpolated curves
plt.plot(x_interpolated_chebyshev, f(x_interpolated_chebyshev), label='Ground Truth', linestyle='--', color='blue')
plt.scatter(x_values_chebyshev, y_values_chebyshev, label='Interpolation Points', color='red')
plt.plot(x_interpolated_chebyshev, y_interpolated_lagrange_chebyshev, label='Lagrange Interpolation', color='green')
plt.plot(x_interpolated_chebyshev, y_interpolated_cubic, label='Cubic Spline Interpolation', color='orange')

# plt.xlabel('x')
# plt.ylabel('y')
plt.title('Comparison of Interpolation Methods With Chebyshev Nodes')
plt.legend()
plt.grid(True)

plt.show()


x_error = np.linspace(-4, 4, 1000)
f_interpolated = cubic_spline_interpolation(x_error)
f_exact = f(x_error)
error_splines = np.abs(f_interpolated - f_exact)

f_lagrange_interpolated = [lagrange_interpolation(x_values, y_values, x) for x in x_error]
error_lagrange = np.abs(f_lagrange_interpolated - f_exact)

# Plot error
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(x_error, error_splines, label='Error Using Splines', color='red')
plt.plot(x_error, error_lagrange, label='Error Using Lagrange', color='blue')
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.title('Error of Cubic Spline Interpolation')
plt.legend()
plt.grid(True)
plt.show()