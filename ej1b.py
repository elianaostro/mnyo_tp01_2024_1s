import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp2d

def f(x1, x2):
    return (
                0.75*np.exp(-((10*x1 - 2)**2)/4 - ((9*x2 - 2)**2)/4) + 
                0.65*np.exp(-((9*x1+1)**2)/9 - ((10*x2 + 1)**2)/2) + 
                0.55*np.exp(-((9*x1 - 6)**2)/4 - ((9*x2 - 3)**2)/4) - 
                0.01*np.exp( -((9*x1 - 7)**2)/4 - ((9*x2 - 3)**2)/4)
            )

# Generate grid points
x1 = np.linspace(-1, 1, 50)
x2 = np.linspace(-1, 1, 50)
X1, X2 = np.meshgrid(x1, x2)
Z = f(X1, X2)

# Interpolate the function
interp_func = interp2d(x1, x2, Z, kind='cubic')

# New grid for interpolated values
x_new = np.linspace(-1, 1, 100)
X1_new, X2_new = np.meshgrid(x_new, x_new)
Z_new = interp_func(x_new, x_new)

# Plot original function
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.plot_surface(X1, X2, Z, cmap='viridis')
ax1.set_title('Original Function')

# Plot interpolated function
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.plot_surface(X1_new, X2_new, Z_new, cmap='viridis')
ax2.set_title('Interpolated Function')

plt.show()
