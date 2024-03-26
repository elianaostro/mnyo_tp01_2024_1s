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


def main():
    # Choose x-values and calculate corresponding y-values
    num_points = 10
    x_values = np.linspace(-4, 4, num_points)
    y_values = f(x_values)

    x_values_chebyshev = chebyshev_nodes(100)
    y_values_chebyshev = f(x_values_chebyshev)

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

    fig, axs = plt.subplots(2, 2)

    axs[0,0].plot(x_ground_truth, y_ground_truth, label='Ground Truth', linestyle='--', color='blue')
    axs[0,0].scatter(x_values, y_values, label='Interpolation Points', color='red', s=8)
    axs[0,0].plot(x_interpolated_lagrange, y_interpolated_lagrange, label='Lagrange Interpolation', color='green')
    axs[0,0].set_title('Lagrange Interpolation')

    axs[0,1].plot(x_interpolated_lagrange, lagrange_error_values, label='Lagrange Error', color='red')
    axs[0,1].set_title('Lagrange Error')

    axs[1,0].plot(x_ground_truth, y_ground_truth, label='Ground Truth', linestyle='--', color='blue')
    axs[1,0].scatter(x_values, y_values, label='Interpolation Points', color='red', s=8)
    axs[1,0].plot(x_interpolated_cubic, y_interpolated_cubic, label='Cubic Spline Interpolation', color='orange')
    axs[1,0].set_title('Cubic Spline Interpolation')

    axs[1,1].plot(x_interpolated_cubic, cubic_spline_error_values, label='Cubic Spline Error', color='red')
    axs[1,1].set_title('Cubic Spline Error')

    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(2, 2)

    axs[0,0].plot(x_ground_truth, y_ground_truth, label='Ground Truth', linestyle='--', color='blue')
    axs[0,0].scatter(x_values_chebyshev, y_values_chebyshev, label='Interpolation Points', color='red', s=8)
    axs[0,0].plot(x_interpolated_lagrange_chebyshev, y_interpolated_lagrange_chebyshev, label='Lagrange Interpolation', color='green')
    axs[0,0].set_title('Lagrange Interpolation (Chebyshev Nodes)')

    axs[0,1].plot(x_interpolated_lagrange_chebyshev, lagrange_error_values_chebyshev, label='Lagrange Error', color='red')
    axs[0,1].set_title('Lagrange Error (Chebyshev Nodes)')

    axs[1,0].plot(x_ground_truth, y_ground_truth, label='Ground Truth', linestyle='--', color='blue')
    axs[1,0].scatter(x_values_chebyshev, y_values_chebyshev, label='Interpolation Points', color='red', s=8)
    axs[1,0].plot(x_interpolated_cubic_chebyshev, y_interpolated_cubic_chebyshev, label='Cubic Spline Interpolation', color='orange')
    axs[1,0].set_title('Cubic Spline Interpolation (Chebyshev Nodes)')

    axs[1,1].plot(x_interpolated_cubic_chebyshev, cubic_spline_error_values_chebyshev, label='Cubic Spline Error', color='red')
    axs[1,1].set_title('Cubic Spline Error (Chebyshev Nodes)')

    plt.tight_layout()
    plt.show()
    return

if(__name__ == "__main__"):
    main()