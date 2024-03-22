import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return 0.3**abs(x) * np.sin(4*x) - np.tanh(2*x) +2

def cubic_spline_interpolation(x, y):

    interpolator = interp1d(x, y, kind='cubic')

    return interpolator

def lagranje(x):
    n = len(x_splines)
    result = 0.0
    for i in range(n):
        term = f(x_splines[i])
        for j in range(n):
            if j != i:
                term *= (x - x_splines[j]) / (x_splines[i] - x_splines[j])
        result += term
    return result

# Generate x values
n = 400
x_values = np.linspace(-4, 4, n)

m = 15
x_splines = np.linspace(-4, 4, m)

# Generate y values
y_values = f(x_values)

# Plot the graph
plt.plot(x_values, y_values, label='Ground Truth')

spl = interpolate.UnivariateSpline(x_values, y_values)

plt.plot(x_splines, spl(x_splines), label='Splines')

y_lagrange = lagranje(x_splines)
plt.plot(x_splines, y_lagrange,label='Lagrange')

# for i in [100, 75, 50, 25, 15]:
#     x_splines = np.linspace(-4, 4, i)
#     plt.plot(x_splines, spl(x_splines), 'g')

#     # Generate y values for Lagrange interpolation
#     y_lagrange = lagranje(x_splines)
#     # Plot the Lagrange interpolation
#     plt.plot(x_splines, y_lagrange)


# Add labels and title
plt.xlabel('x')
plt.ylabel('f(x)')
#plt.title('Graph of $f(x) = x^2$')

# Add grid
plt.grid(True)

# Add legend
plt.legend()

# Show the plot
plt.show()