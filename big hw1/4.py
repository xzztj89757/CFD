import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

matplotlib.use('Agg')
from tqdm import tqdm


# Set the parameters for the simulation
def analytical_solution(x, m, t):
    u = np.zeros_like(x)
    for l in range(1, m + 1):
        u += np.sin(2 * np.pi * l * (x - t))
    u /= m
    return u


def Boundary_exact_solution(x, t):
    u00 = np.zeros_like(x)
    u = np.zeros_like(x)
    for k in range(1, 65):
        E_k = (k / k0) ** 4 * np.exp(-2 * (k / k0) ** 2)
        psi_k = np.random.uniform(0, 1)
        u00 += np.sqrt(E_k) * np.sin(2 * np.pi * k * (x + psi_k))
        u += np.sqrt(E_k) * np.sin(2 * np.pi * k * (x - t + psi_k))

    return 1 + epsilon1 * u00, 1 + epsilon1 * u


# Set the initial condition


# u0 = bc1(x)
# u1 = analytical_solution(x, m, tend)


# 7-order Dispersion-Relation-Preserving scheme
def DRP_M7(u):
    a0 = 0.
    a1 = 0.770882380518
    a2 = -0.166705904415
    a3 = 0.020843142770
    ux = np.zeros_like(u)
    for i in range(Nx):
        ip1 = i + 1 if i + 1 < Nx else i + 1 - Nx
        ip2 = i + 2 if i + 2 < Nx else i + 2 - Nx
        ip3 = i + 3 if i + 3 < Nx else i + 3 - Nx

        ux[i] = (a3 * u[ip3] + a2 * u[ip2] + a1 * u[ip1] + a0 * u[i] - a3 * u[i - 3] - a2 * u[i - 2] - a1 * u[
            i - 1]) / dx

    '''ux = a3 * np.roll(u, -3) + a2 * np.roll(u, -2) + a1 * np.roll(u, -1) - a3 * np.roll(u, 3) - a2 * np.roll(u,
                                                                                                           2) - a1 * np.roll(
        u, 1)'''
    return ux


def DRP7(u):
    a0 = 0.
    a1 = 0.79926643
    a2 = -0.18941314
    a3 = 0.02651995
    ux = np.zeros_like(u)
    for i in range(Nx):
        ip1 = i + 1 if i + 1 < Nx else i + 1 - Nx
        ip2 = i + 2 if i + 2 < Nx else i + 2 - Nx
        ip3 = i + 3 if i + 3 < Nx else i + 3 - Nx

        ux[i] = (a3 * u[ip3] + a2 * u[ip2] + a1 * u[ip1] + a0 * u[i] - a3 * u[i - 3] - a2 * u[i - 2] - a1 * u[
            i - 1]) / dx
    return ux


def MDCD7(u):
    alpha = 0.04389997
    beta = 0.
    gp = alpha + beta
    gs = alpha - beta
    ux = np.zeros_like(u)
    for i in range(Nx):
        ip1 = i + 1 if i + 1 < Nx else i + 1 - Nx
        ip2 = i + 2 if i + 2 < Nx else i + 2 - Nx
        ip3 = i + 3 if i + 3 < Nx else i + 3 - Nx

        ux[i] = ((0.5 * gp - 0.5 * gs) * u[ip3] + (-2 * gp + 3 * gs - 1 / 12) * u[ip2] + (2.5 * gp - 7.5 * gs + 2 / 3) *
                 u[ip1] + 10 * gs * u[i] + (
                         -0.5 * gp - 0.5 * gs) * u[i - 3] + (2 * gp + 3 * gs + 1 / 12) * u[i - 2] + (
                         -2.5 * gp - 7.5 * gs - 2 / 3) * u[
                     i - 1]) / dx
    return ux


def compute_dispersion_param(k_ESW):
    if 0. <= k_ESW < 0.01:
        gamma_disp = 1 / 30
    elif 0.01 <= k_ESW < 2.5:
        gamma_disp = (
                             k_ESW + (1 / 6) * np.sin(2 * k_ESW)
                             - (4 / 3) * np.sin(k_ESW)) / (np.sin(3 * k_ESW)
                                                           - 4 * np.sin(2 * k_ESW) + 5 * np.sin(k_ESW)
                                                           )
    else:
        gamma_disp = 0.1985842
    return gamma_disp


def compute_dissipation_param(k_ESW):
    if 0 <= k_ESW <= 1.0:
        gamma_diss = np.sign(a) * 0.001
    else:
        gamma_diss = np.sign(a) * np.minimum(
            0.001 + 0.011 * np.sqrt((k_ESW - 1.0) / (np.pi - 1.0)), 0.012
        )
    return gamma_diss


def scale_sensor(u, i):
    if (u.max() - u.min()) < 1e-3:
        k_ESW = 0
        gamma_disp = 1 / 30
        gamma_diss = 0.001
    else:
        epsilon = 1e-8
        ip1 = i + 1 if i + 1 < Nx else i + 1 - Nx
        ip2 = i + 2 if i + 2 < Nx else i + 2 - Nx
        ip3 = i + 3 if i + 3 < Nx else i + 3 - Nx

        S1 = u[ip1] - 2 * u[i] + u[i - 1]
        S2 = (u[ip2] - 2 * u[i] + u[i - 2]) / 4
        S3 = u[ip2] - 2 * u[ip1] + u[i]
        S4 = (u[ip3] - 2 * u[ip1] + u[i - 1]) / 4
        C1 = u[ip1] - u[i]
        C2 = (u[ip2] - u[i - 1]) / 3

        numerator = (
                abs(abs(S1 + S2) - abs(S1 - S2))
                + abs(abs(S3 + S4) - abs(S3 - S4))
                + abs(abs(C1 + C2) - abs(C1 - C2) / 2)
                + 2 * epsilon
        )
        denominator = (
                abs(S1 + S2) + abs(S1 - S2)
                + abs(S3 + S4) + abs(S3 - S4)
                + abs(C1 + C2) + abs(C1 - C2) + epsilon
        )

        k_ESW = np.arccos(2 * np.minimum(numerator / denominator, 1) - 1)

        gamma_disp = compute_dispersion_param(k_ESW)
        gamma_diss = compute_dissipation_param(k_ESW)
    # print(k_ESW)
    return gamma_disp, gamma_diss


def SA_DRP(u):
    ux = np.zeros_like(u)
    # fr = np.zeros_like(u)
    # fl = np.zeros_like(u)
    for i in range(Nx):
        gpr, gsr = scale_sensor(u, i)
        gpl, gsl = scale_sensor(u, i - 1)
        # print(gpr,gsr)
        ip1 = i + 1 if i + 1 < Nx else i + 1 - Nx
        ip2 = i + 2 if i + 2 < Nx else i + 2 - Nx
        ip3 = i + 3 if i + 3 < Nx else i + 3 - Nx

        fr = (
                (1 / 2) * (gpr + gsr) * u[i - 2]
                + (-(3 / 2) * gpr - (5 / 2) * gsr - 1 / 12) * u[i - 1]
                + (gpr + 5 * gsr + 7 / 12) * u[i]
                + (gpr - 5 * gsr + 7 / 12) * u[ip1]
                + (-3 / 2 * gpr + 5 / 2 * gsr - 1 / 12) * u[ip2]
                + 1 / 2 * (gpr - gsr) * u[ip3]
        )
        fl = (
                (1 / 2) * (gpl + gsl) * u[i - 3]
                + (-(3 / 2) * gpl - (5 / 2) * gsl - 1 / 12) * u[i - 2]
                + (gpl + 5 * gsl + 7 / 12) * u[i - 1]
                + (gpl - 5 * gsl + 7 / 12) * u[i]
                + (-3 / 2 * gpl + 5 / 2 * gsl - 1 / 12) * u[ip1]
                + 1 / 2 * (gpl - gsl) * u[ip2]
        )
        # print(fr,fl)
        ux[i] = (1 / dx) * (fr - fl)
    # ux[i] = (fr - fl)
    # print(ux)
    return ux


# 3-order Runge-Kutta
'''def Runge_Kutta_DRP_M7(u, dt):
    u0 = u
    u1 = u0 - dt * a * DRP_M7(u)
    u2 = 3 / 4 * u0 + 1 / 4 * (u1 - dt * a * DRP_M7(u1))
    u3 = 1 / 3 * u0 + 2 / 3 * (u2 - dt * a * DRP_M7(u2))
    unew = u3
    return unew


def Runge_Kutta_DRP7(u, dt):
    u0 = u
    u1 = u0 - dt * a * DRP7(u)
    u2 = 3 / 4 * u0 + 1 / 4 * (u1 - dt * a * DRP7(u1))
    u3 = 1 / 3 * u0 + 2 / 3 * (u2 - dt * a * DRP7(u2))
    unew = u3
    return unew'''


def Runge_Kutta3(u, dt, method):
    u00 = u
    u11 = u00 - dt * a * method(u)
    u2 = 3 / 4 * u00 + 1 / 4 * (u11 - dt * a * method(u11))
    u3 = 1 / 3 * u00 + 2 / 3 * (u2 - dt * a * method(u2))
    unew = u3
    # print('1',u,'2',u11,'3',u2,'4',unew)
    return unew


def Runge_Kutta4(u, dt, method):
    u00 = u
    u11 = u00 - dt / 4 * a * method(u)
    u2 = u00 - dt / 3 * a * method(u11)
    u3 = u00 - dt / 2 * a * method(u2)
    unew = u00 - dt / 1 * a * method(u3)
    # print('1',u,'2',u11,'3',u2,'4',unew)
    return unew


def F1(u, dt, method):
    unew = u - dt * method(u)
    return unew


def calculate_1_norm_error(u, u_exact):
    nx = u.shape
    errors = 0.
    errors = np.sum(np.abs(u - u_exact)) / nx
    return errors


def run_simulation(u00, method_t, method_x):
    u = u00.copy()
    # print(u)
    t = 0
    it = 0
    # while it<10:
    while t < tend:
        u = method_t(u, dt, method_x)
        t += dt
        it += 1
        # t=it*dt
        # print(it, t, str(method_x))
    errors = calculate_1_norm_error(u, u1)
    return u, errors


colors = {
    'DRP_M': '#E24A33',  # Orange-Red
    'DRP': '#348ABD',  # Blue
    'MDCD': '#988ED5',  # Lavender
    'SA_DRP': '#FBC15E',  # Golden Yellow
}
markers = {
    'DRP_M': 's',
    'DRP': 'o',
    'MDCD': '^',
    'SA_DRP': 'x'
}
errors = {
    'DRP_M': [],
    'DRP': [],
    'MDCD': [],
    'SA_DRP': []
}
grid_sizes = [64, 128, 256, 512, 1024]
#grid_sizes = [160, 170, 180, 190, 200, 210, 220, 230, 240, 250]
max_error = 0.005
min_grid_size = {}

for N in grid_sizes:
    xmin, xmax = 0., 1
    CFL = 0.2
    ncells = N
    Nx = ncells
    # dx = (xmax - xmin) / ncells
    dx = (xmax - xmin) / Nx
    a = 1  # Advection speed
    dt = CFL * dx / a  # Time step size with CFL condition
    tend = 1.
    m = 20
    # Initialize the spatial domain
    x = np.linspace(xmin, xmax, Nx, endpoint=False)
    k0 = 12
    epsilon1 = 0.1
    np.random.seed(42)

    u0, u1 = Boundary_exact_solution(x, tend)

    u_MCDC, err_MCDC = run_simulation(u0, Runge_Kutta4, MDCD7)
    u_SA_DRP, err_SA_DRP = run_simulation(u0, Runge_Kutta4, SA_DRP)
    # u_SA_DRP = run_simulation(u0, F1, SA_DRP)
    # u_MCDC = run_simulation(u0, Runge_Kutta, MDCD7)
    u_DRP_M, err_DRP_M = run_simulation(u0, Runge_Kutta4, DRP_M7)
    u_DRP, err_DRP = run_simulation(u0, Runge_Kutta4, DRP7)
    errors['DRP_M'].append(err_DRP_M)
    errors['DRP'].append(err_DRP)
    errors['MDCD'].append(err_MCDC)
    errors['SA_DRP'].append(err_SA_DRP)

    # Create a figure for plotting

    fig, ax = plt.subplots()
    ax.scatter(x, u_DRP_M, color=colors['DRP_M'], label='DRP_M', s=12, marker=markers['DRP_M'])
    ax.scatter(x, u_DRP, color=colors['DRP'], label='DRP', s=12, marker=markers['DRP'])
    ax.scatter(x, u_MCDC, color=colors['MDCD'], label='MDCD', s=12, marker=markers['MDCD'])
    ax.scatter(x, u_SA_DRP, color=colors['SA_DRP'], label='SA_DRP', s=12, marker=markers['SA_DRP'])

    line1, = ax.plot(x, u1, 'k', label='Exact')
    # line2, = ax.plot(x, u0, label='0')
    ax.legend(loc='upper right')
    ax.set_xlim(xmin, xmax)
    # ax.set_ylim(-1., 1.)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("u", fontsize=14)
    # ax.set_title('Schemes Comparison', fontsize=16)

    fig.tight_layout()
    # plt.savefig('45')
    plt.savefig('result' + '_N=' + str(N) + '.png', dpi=600)
    plt.close()

    for scheme, err_list in errors.items():
        if min(err_list) <= max_error and scheme not in min_grid_size:
            min_grid_size[scheme] = N
    print("N", N, "mgs", min_grid_size,"loss",errors)
for scheme, N in min_grid_size.items():
    print(f"Minimum number of grids for {scheme}: {N}")

# for scheme in errors.keys():
# np.save(f'{scheme}_errors.npy', np.array(errors[scheme]))

for scheme, color in colors.items():
    plt.loglog(grid_sizes, errors[scheme], label=scheme, marker=markers.get(scheme), color=color, linestyle='-')

plt.xlabel('Number of Grids (N)', fontsize=14)
plt.ylabel('1-norm Error', fontsize=14)
plt.title('1-norm Error Convergence', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save the plot as an image
plt.savefig('1norm_errors-N.png', dpi=600)
plt.close()
'''def generate_individual_plot(u, label, color, marker):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, u, color=color, label=label, s=20, marker=marker)
    ax.plot(x, u1, color=colors['Analytical'], label='Analytical Solution', linewidth=2)
    ax.legend(loc='upper right', fontsize=12)
    ax.set_xlim(xmin, xmax)
    # ax.set_ylim(-1., 1.)
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("u", fontsize=14)
    ax.set_title(f'{label} Scheme Comparison', fontsize=18)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    plt.savefig(f'result_{label}.png', dpi=300)


    # Generate individual plots for each scheme
    generate_individual_plot(u_DRP_M, 'DRP_M', colors['DRP_M'], markers['DRP_M'])
    generate_individual_plot(u_DRP, 'DRP', colors['DRP'], markers['DRP'])
    generate_individual_plot(u_MCDC, 'MDCD', colors['MDCD'], markers['MDCD'])
    generate_individual_plot(u_SA_DRP, 'SA_DRP', colors['SA_DRP'], markers['SA_DRP'])
    # generate_individual_plot(u_SA_DRP, '42', colors['SA_DRP'], markers['SA_DRP'])
    # Number of categories

    positions = [0.5, 1.5, 2.5, 3.5]

    plt.figure(figsize=(10, 6))

    # Creating bars
    plt.bar(positions[0], err_DRP_M, width=0.8, color=colors['DRP_M'], label='DRP_M', linewidth=2, edgecolor='black')
    plt.bar(positions[1], err_DRP, width=0.8, color=colors['DRP'], label='DRP', linewidth=2, edgecolor='black')
    plt.bar(positions[2], err_MCDC, width=0.8, color=colors['MDCD'], label='MCDC', linewidth=2, edgecolor='black')
    plt.bar(positions[3], err_SA_DRP, width=0.8, color=colors['SA_DRP'], label='SA_DRP', linewidth=2, edgecolor='black')

    # Customizing the plot
    plt.xlabel('Schemes', fontsize=14)
    plt.ylabel('1-Norm Error', fontsize=14)
    plt.title('Error Comparison Between Different Schemes', fontsize=16)
    plt.xticks(positions, ['DRP_M', 'DRP', 'MCDC', 'SA_DRP'], fontsize=12)  # Setting custom x-tick labels
    plt.yticks(fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)  # Adding a grid for easier reading

    # Enhance layout to ensure everything fits without clipping
    plt.tight_layout()

    # Save the plot to an image file
    plt.savefig('1norm_errors_comparison' + '_N=' + str(N) + '.png', dpi=600)
    plt.close()'''
