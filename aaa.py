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

# Plot the original function and the interpolated curves
plt.plot(x_interpolated_lagrange, f(x_interpolated_lagrange), label='Ground Truth', linestyle='--', color='blue')
plt.scatter(x_values, y_values, label='Interpolation Points', color='red')
plt.plot(x_interpolated_lagrange, y_interpolated_lagrange, label='Lagrange Interpolation', color='green')
plt.plot(x_interpolated_lagrange, y_interpolated_cubic, label='Cubic Spline Interpolation', color='orange')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Interpolation Methods')
plt.legend()
plt.grid(True)
plt.show()
