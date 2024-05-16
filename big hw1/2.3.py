import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from numpy import sin, exp, pi
from scipy.optimize import newton

v = 8


# Define the Re function
def Re(k, alpha):
    return alpha * sin(3 * k) - (4 * alpha + 1 / 6) * sin(2 * k) + (5 * alpha + 4 / 3) * sin(k)


def Re_d(k):
    return sin(3 * k) - 4 * sin(2 * k) + 5 * sin(k)


# Define the integral function E
def E1(alpha):
    def integrand(k):
        return 2 * exp(v * (pi - k)) * (Re(k, alpha) - k) * Re_d(k)

    integral_value, _ = quad(integrand, 0, pi, epsabs=1e-10, epsrel=1e-10)
    return integral_value / (exp(pi * v))

def E2(alpha):
    def integrand(k):
        return exp(v * - k) * (Re(k, alpha) - k) * Re_d(k)

    integral_value, _ = quad(integrand, 0, pi, epsabs=1e-10, epsrel=1e-10)
    return integral_value

def dE(alpha):
    def integrand_de(k):
        return exp(v * - k) * Re_d(k)**2

    derivative_value, _ = quad(integrand_de, 0, pi, epsabs=1e-10, epsrel=1e-10)
    return derivative_value


# Find the approximate value of alpha such that E(alpha) = 0
alpha_solution = brentq(E2, 0, 0.05,xtol=1e-12)
#alpha_solution = newton(E2, 0.04, fprime=dE, tol=1e-12)

print(f'Approximate solution for alpha such that E(alpha) = 0: {alpha_solution}')
