import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.interpolate import lagrange
from scipy.interpolate import CubicSpline

# Define the function to interpolate
def f(x):
    return 0.3**abs(x) * np.sin(4*x) - np.tanh(2*x) + 2

def lagrange_interpolation(x_values, y_values, x):
    # """Perform Lagrange interpolation to find y-value at x."""
    poly = lagrange(x_values, y_values)
    return poly(x)

def cubic_spline_interpolation(x_values, y_values, x):
    # """Perform Lagrange interpolation to find y-value at x."""
    poly = CubicSpline(x_values, y_values)
    return poly(x)

def lagrange_error(x_values, y_values, x):
    poly = lagrange(x_values, y_values)
    return abs(poly(x) - f(x))

def cubic_spline_error(x_values, y_values, x):
    poly = CubicSpline(x_values, y_values)
    return abs(poly(x) - f(x))

def chebyshev_nodes(n, a=-4, b=4):
    """Generate n non-equidistant Chebyshev nodes in the interval [a, b]."""
    k = np.arange(1, n + 1)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) * np.pi / (2 * n))
    return np.sort(nodes)

def plot_interpolation_comparison(x_ground_truth, y_ground_truth, x_values, y_values, x_interpolated_lagrange, y_interpolated_lagrange, lagrange_error_values,
                                  x_interpolated_cubic, y_interpolated_cubic, cubic_spline_error_values, title):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)

    # Plot Lagrange interpolation with regular nodes
    axs[0, 0].plot(x_ground_truth, y_ground_truth, label='Ground Truth', linestyle='--', color='blue')
    axs[0, 0].scatter(x_values, y_values, label='Puntos de Interpolacion', color='red', s=8)
    axs[0, 0].plot(x_interpolated_lagrange, y_interpolated_lagrange, label='Interpolacion Lagrange', color='green')
    axs[0, 0].set_title('Interpolacion Lagrange')
    axs[0, 0].legend()

    # Plot error for Lagrange interpolation
    axs[0, 1].plot(x_interpolated_lagrange, lagrange_error_values, label='Error Lagrange', color='red')
    axs[0, 1].set_title('Error Lagrange')

    # Plot Cubic Spline interpolation with regular nodes
    axs[1, 0].plot(x_ground_truth, y_ground_truth, label='Ground Truth', linestyle='--', color='blue')
    axs[1, 0].scatter(x_values, y_values, label='Puntos de Interpolacion', color='red', s=8)
    axs[1, 0].plot(x_interpolated_cubic, y_interpolated_cubic, label='Interpolacion Splines Cubicos', color='orange')
    axs[1, 0].set_title('Interpolacion Splines Cubicos')
    axs[1, 0].legend()

    # Plot error for Cubic Spline interpolation
    axs[1, 1].plot(x_interpolated_cubic, cubic_spline_error_values, label='Error Splines Cubicos', color='red')
    axs[1, 1].set_title('Error Splines Cubicos')

    plt.tight_layout()
    plt.show()

def main():
    # Choose x-values and calculate corresponding y-values
    num_points = 10
    x_values = np.linspace(-4, 4, num_points)
    y_values = f(x_values)

    x_values_chebyshev = chebyshev_nodes(num_points)
    y_values_chebyshev = f(x_values_chebyshev)

    x_ground_truth = np.linspace(-4, 4, 100)
    y_ground_truth = f(x_ground_truth)

    x_interpolated_lagrange = np.linspace(-4, 4, 100)
    y_interpolated_lagrange = lagrange_interpolation(x_values, y_values, x_interpolated_lagrange)

    x_interpolated_lagrange_chebyshev = np.linspace(-4, 4, 100)
    y_interpolated_lagrange_chebyshev = lagrange_interpolation(x_values_chebyshev, y_values_chebyshev, x_interpolated_lagrange_chebyshev)

    lagrange_error_values = lagrange_error(x_values, y_values, x_interpolated_lagrange)
    lagrange_error_values_chebyshev = lagrange_error(x_values_chebyshev, y_values_chebyshev, x_interpolated_lagrange_chebyshev)

    x_interpolated_cubic = np.linspace(-4, 4, 100)
    y_interpolated_cubic = cubic_spline_interpolation(x_values, y_values, x_interpolated_cubic)

    x_interpolated_cubic_chebyshev = np.linspace(-4, 4, 100)
    y_interpolated_cubic_chebyshev = cubic_spline_interpolation(x_values_chebyshev, y_values_chebyshev, x_interpolated_cubic_chebyshev)

    cubic_spline_error_values = cubic_spline_error(x_values, y_values, x_interpolated_cubic)
    cubic_spline_error_values_chebyshev = cubic_spline_error(x_values_chebyshev, y_values_chebyshev, x_interpolated_cubic_chebyshev)

    # Plot with equispaced nodes
    plot_interpolation_comparison(x_ground_truth, y_ground_truth, x_values, y_values,
                                  x_interpolated_lagrange, y_interpolated_lagrange, lagrange_error_values,
                                  x_interpolated_cubic, y_interpolated_cubic, cubic_spline_error_values,
                                  "Nodos Equiespaciados")

    # Plot with non-equispaced nodes
    plot_interpolation_comparison(x_ground_truth, y_ground_truth, x_values_chebyshev, y_values_chebyshev,
                                  x_interpolated_lagrange_chebyshev, y_interpolated_lagrange_chebyshev, lagrange_error_values_chebyshev,
                                  x_interpolated_cubic_chebyshev, y_interpolated_cubic_chebyshev, cubic_spline_error_values_chebyshev,
                                  "Nodos No Equiespaciados Por Chebyshev")

if(__name__ == "__main__"):
    main()