import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return 0.3**abs(x) * np.sin(4*x) - np.tanh(2*x) + 2

def lineal_spline_interpolation(x, y):
    interpolator = interpolate.interp1d(x, y)
    return interpolator

def quadratic_spline_interpolation(x, y):
    interpolator = interpolate.interp1d(x, y, kind='quadratic')
    return interpolator

def cubic_spline_interpolation(x, y):
    interpolator = interpolate.interp1d(x, y, kind='cubic')
    return interpolator

def lagranje15(x):
    n = len(x_splines_15)
    result = 0.0
    for i in range(n):
        term = f(x_splines_15[i])
        for j in range(n):
            if j != i:
                term *= (x - x_splines_15[j]) / (x_splines_15[i] - x_splines_15[j])
        result += term
    return result

# Generate x values
x_values = np.linspace(-4, 4, 100)

x_splines_15 = np.linspace(-4, 4, 15)
x_splines_25 = np.linspace(-4, 4, 25)
x_splines_50 = np.linspace(-4, 4, 50)
x_splines_75 = np.linspace(-4, 4, 75)
x_splines_100 = np.linspace(-4, 4, 100)

# Generate y values
y_values = f(x_values)
y_splines_15 = f(x_splines_15)

# Plot the graph
plt.plot(x_values, y_values, label='Ground Truth', linestyle='--')

# spl = interpolate.UnivariateSpline(x_values, y_values)
# plt.plot(x_splines_15, spl(x_splines_15), label='Splines')

y_lagrange = lagranje15(x_splines_15)
plt.plot(x_splines_15, y_lagrange,label='Lagrange')

# y_lineal_spline = lineal_spline_interpolation(x_splines_15, y_splines_15)
# y_splines_lineal=[]
# for x in x_values:
#     y_splines_lineal.append(y_lineal_spline(x))
# plt.plot(x_values, y_splines_lineal, label='Lineal Spline')

# y_quadratic_spline = quadratic_spline_interpolation(x_splines_15, y_splines_15)
# y_splines_quadratic = []
# for x in x_values:
#     y_splines_quadratic.append(y_quadratic_spline(x))
# plt.plot(x_values, y_splines_quadratic, label='Quadratic Spline')

y_cubic_spline = cubic_spline_interpolation(x_splines_15, y_splines_15)
y_splines_cubic = []
for x in x_values:
    y_splines_cubic.append(y_cubic_spline(x))
plt.plot(x_values, y_splines_cubic, label='Cubic Spline')


# Add labels and title
# plt.xlabel('x')
# plt.ylabel('f(x)')
#plt.title('Graph of $f(x) = x^2$')

# Add grid
plt.grid(False)

# Add legend
plt.legend()

plt.scatter(x_splines_15, y_splines_15, color='red', s=10)

# Show the plot
plt.show()