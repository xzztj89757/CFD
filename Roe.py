import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use('Agg')
from numpy import *
from matplotlib import rc

CFL = 0.50
gamma = 1.4
ncells = 200
x_ini = 0
x_fin = 1.  # computational domain
dx = (x_fin - x_ini) / ncells
nx = ncells + 1  # Number of points
x = np.linspace(x_ini + dx / 2., x_fin, nx)  # Mesh

# Initial conditions
rho0 = np.where(x <= 0.5, 1.0, 0.125)  # density
u0 = np.where(x <= 0.5, 0.0, 0.0)  # velocity
p0 = np.where(x <= 0.5, 1.0, 0.1)  # pressure

E0 = p0 / ((gamma - 1.) * rho0) + 0.5 * u0 ** 2  # Total Energy density
a0 = sqrt(gamma * p0 / rho0)  # Speed of sound
U = np.array([rho0, rho0 * u0, rho0 * E0])  # Vector of conserved variables

# Solver loop
t = 0
tEnd = 0.2
it = 0
a = a0
dt = CFL * dx / max(abs(u0) + a0)
nt = int((tEnd - t) / dt)


def flux(U):
    r = U[0]
    u = U[1] / r
    E = U[2] / r
    p = (gamma - 1.) * r * (E - 0.5 * u ** 2)

    # Flux vector of conserved properties
    F0 = np.array(r * u)
    F1 = np.array(r * u ** 2 + p)
    F2 = np.array(u * (r * E + p))
    flux = np.array([F0, F1, F2])
    return flux


def roe_average(U):
    r = U[0]
    u = U[1] / r
    E = U[2] / r
    p = (gamma - 1.) * r * (E - 0.5 * u ** 2)
    H = gamma / (gamma - 1.) * p / r + 0.5 * u ** 2

    FHalf = np.zeros((3, ncells))

    for j in range(0, nx - 1):
        R = sqrt(r[j + 1] / r[j])
        rhat = R * r[j]
        uhat = (u[j] + R * u[j + 1]) / (1. + R)
        Hhat = (H[j] + R * H[j + 1]) / (1 + R)
        ahat = sqrt((gamma - 1) * (Hhat - 0.5 * uhat ** 2))

        alpha = (gamma - 1.) / (ahat ** 2)
        w = U[:, j + 1] - U[:, j]
        L = 0.5 * np.array([[0.5 * alpha * uhat ** 2 + uhat / ahat, -(alpha * uhat + 1 / ahat), alpha],
                            [(2 - alpha * uhat ** 2), 2 * alpha * uhat, -2 * alpha],
                            [(0.5 * alpha * uhat ** 2 - uhat / ahat), -(alpha * uhat - 1 / ahat), alpha]])
        R = np.array([[1, 1, 1],
                      [uhat - ahat, uhat, uhat + ahat],
                      [Hhat - uhat, 0.5 * uhat ** 2, Hhat + uhat * ahat]])
        lamb = np.array([[abs(uhat - ahat), 0, 0],
                         [0, abs(uhat), 0],
                         [0, 0, abs(uhat + ahat)]])

        A = np.dot(R, lamb)
        A = np.dot(A, L)
        FHalf[:, j] = np.dot(A, w)
    F = flux(U)
    FHalf = 0.5 * (F[:, 0:nx - 1] + F[:, 1:nx]) - 0.5 * FHalf
    dF = (FHalf[:, 1:-1] - FHalf[:, 0:-2])
    return (dF)


while t < tEnd:

    U0 = U.copy()

    # Primary variables
    dF=roe_average(U0)

    U[:,1:-2] = U0[:,1:-2] - dt / dx * dF
    U[:, 0] = U0[:, 0]
    U[:, -1] = U0[:, -1]
    U[:, -2] = U0[:,-2]
    print(U.shape,dF.shape)
    # Compute primary variables
    rho = U[0]
    u = U[1] / rho
    E = U[2] / rho
    p = (gamma - 1.) * rho * (E - 0.5 * u ** 2)
    a = sqrt(gamma * p / rho)
    if min(p) < 0: print('negative pressure found!')

    # Update/correct time step
    dt = CFL * dx / max(abs(u) + a)

    # Update time and iteration counter
    t = t + dt
    # print(t)
    it = it + 1
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10, 12))

# Plot density
axes[0].plot(x, rho, 'k-')
axes[0].set_ylabel('$\\rho$', fontsize=16)
axes[0].tick_params(axis='x', bottom=False, labelbottom=False)  # Disable x-axis labels for this subplot
axes[0].grid(True)

# Plot velocity
axes[1].plot(x, u, 'r-')
axes[1].set_ylabel('$U$', fontsize=16)
axes[1].tick_params(axis='x', bottom=False, labelbottom=False)  # Disable x-axis labels for this subplot
axes[1].grid(True)

# Plot pressure
axes[2].plot(x, p, 'b-')
axes[2].set_ylabel('$p$', fontsize=16)
axes[2].tick_params(axis='x', bottom=False, labelbottom=False)  # Disable x-axis labels for this subplot
axes[2].grid(True)

# Plot energy
axes[3].plot(x, E, 'g-')
axes[3].set_ylabel('$E$', fontsize=16)
axes[3].grid(True)
axes[3].set_xlim(x_ini, x_fin)  # Assuming x_ini and x_fin are defined
axes[3].set_xlabel('x', fontsize=16)

# Adjust subplots to give some spacing and padding
fig.subplots_adjust(left=0.2, bottom=0.15, top=0.95)
plt.savefig('Roe' + str(ncells))
plt.close()
