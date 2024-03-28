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

def generate_equiespaced_points(n, a=-1, b=1):
    """Generate n equiespaced points in the interval [a, b]."""
    x1 = np.linspace(a, b, n)
    x2 = np.linspace(a, b, n)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f(X1, X2)
    return X1, X2, Z

def chebyshev_nodes(n, a=-1, b=1):
    """Generate Chebyshev nodes in the interval [a, b]."""
    k = np.arange(1, n + 1)
    nodes = 0.5 * (a + b) + 0.5 * (b - a) * np.cos((2 * k - 1) * np.pi / (2 * n))
    return np.sort(nodes)

def generate_chebyshev_points(n):
    x1 = chebyshev_nodes(n)
    x2 = chebyshev_nodes(n)
    X1, X2 = np.meshgrid(x1, x2)
    Z = f(X1, X2)
    return X1, X2, Z

def spline_interpolation(X1, X2, Z, kind='cubic'):
    interp_func = interp2d(X1[0], X2[:, 0], Z, kind=kind)  # Use 1-D arrays for X1 and X2
    x_new = np.linspace(min(X1[0]), max(X1[0]), 15)
    X1_new, X2_new = np.meshgrid(x_new, x_new)
    Z_new = interp_func(x_new, x_new)
    return X1_new, X2_new, Z_new

def error_interpolation(X1, X2, Z_new):
    return np.abs(f(X1, X2) - Z_new)

def plot_original_function(X1, X2, Z):
    fig1 = plt.figure(figsize=(12, 9))
    ax_original = fig1.add_subplot(111, projection='3d')
    ax_original.plot_surface(X1, X2, Z, cmap='viridis')
    ax_original.set_title('Original Function')
    plt.show()

def plot_interpolation(X1_new, X2_new, Z_new, kind):
    fig2 = plt.figure(figsize=(12, 9))
    ax_interp = fig2.add_subplot(111, projection='3d')
    ax_interp.plot_surface(X1_new, X2_new, Z_new, cmap='plasma')
    ax_interp.set_title(f'{kind} Interpolation')
    plt.show()

def plot_error(X1, X2, error, kind):
    plt.figure(figsize=(8, 6))
    plt.imshow(error, extent=(min(X1[0]), max(X1[0]), min(X2[:, 0]), max(X2[:, 0])), cmap='plasma', origin='lower')
    plt.colorbar(label='Absolute Error')
    plt.title(f'{kind} Interpolation Error')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()


def main():
    x1, x2, z = generate_equiespaced_points(50)
    x1_chebyshev, x2_chebyshev, z_chebyshev = generate_chebyshev_points(50)

    x1_cubic, x2_cubic, z_new_cubic = spline_interpolation(x1, x2, z, 'cubic')
    error_cubic = error_interpolation(x1_cubic, x2_cubic, z_new_cubic)

    x1_chebyshev_cubic, x2_chebyshev_cubic, z_chebyshev_cubic = spline_interpolation(x1_chebyshev, x2_chebyshev, z_chebyshev, 'cubic')
    error_chebyshev = error_interpolation(x1_chebyshev_cubic, x2_chebyshev_cubic, z_chebyshev_cubic)

    x1_linear, x2_linear, z_new_linear = spline_interpolation(x1, x2, z, 'linear')
    error_linear = error_interpolation(x1_linear, x2_linear, z_new_linear)
    
    x1_chebyshev_linear, x2_chebyshev_linear, z_chebyshev_linear = spline_interpolation(x1_chebyshev, x2_chebyshev, z_chebyshev, 'linear')
    error_chebyshev_linear = error_interpolation(x1_chebyshev_linear, x2_chebyshev_linear, z_chebyshev_linear)    

    plot_original_function(x1, x2, z)
    plot_interpolation(x1_cubic, x2_cubic, z_new_cubic, 'Cubic Spline')
    plot_error(x1_cubic, x2_cubic, error_cubic, 'Cubic Spline')

    plot_interpolation(x1_chebyshev_cubic, x2_chebyshev_cubic, z_chebyshev_cubic, 'Cubic Spline Chebyshev')
    plot_error(x1_chebyshev_cubic, x2_chebyshev_cubic, error_chebyshev, 'Cubic Spline Chebyshev')

    plot_interpolation(x1_linear, x2_linear, z_new_linear, 'Linear Spline')
    plot_error(x1_linear, x2_linear, error_linear, 'Linear Spline')

    plot_interpolation(x1_chebyshev_linear, x2_chebyshev_linear, z_chebyshev_linear, 'Linear Spline Chebyshev')
    plot_error(x1_chebyshev_linear, x2_chebyshev_linear, error_chebyshev_linear, 'Linear Spline Chebyshev')


if __name__ == '__main__':
    main()