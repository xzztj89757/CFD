import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos
matplotlib.use('Agg')

# Define the independent variable
k_values = np.linspace(0, 6, 400)
# Define the range of parameters
alpha_values = [0, 0.033, 0.01, 0.02, 0.05, 0.1]
beta_values = [0, 0.033, 0.01, 0.02, 0.05,0.1]


# Define the functions
def Re(k, alpha):
    return alpha * sin(3 * k) - (4 * alpha + 1 / 6) * sin(2 * k) + (5 * alpha + 4 / 3) * sin(k)


def Im(k, beta):
    return beta * cos(3 * k) - 6 * beta * cos(2 * k) + 15 * beta * cos(k) - 10 * beta


# Create the plots
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# Plot for function1 and y = x
for alpha in alpha_values:
    axs[0].plot(k_values, Re(k_values, alpha), label=f'alpha = {alpha}')
axs[0].plot(k_values, k_values, label='y = x', linestyle='--', color='black')
axs[0].legend()
axs[0].set_title('Re(dispersion)')
axs[0].set_xlabel('k')
axs[0].set_ylabel('Re(k)')
axs[0].set_xlim(0., 4.)
axs[0].set_ylim(0., 4.)
# Plot for function2
for beta in beta_values:
    axs[1].plot(k_values, Im(k_values, beta), label=f'beta = {beta} ')
axs[1].legend()
axs[1].set_title('Im(dissipation)')
axs[1].set_xlabel('k')
axs[1].set_ylabel('Im(k)')
#axs[1].set_xlim(0., 4.)
#axs[1].set_ylim(0., -4.)
# Save the figure
plt.savefig('multiple_functions_plot1.png',dpi=600)
