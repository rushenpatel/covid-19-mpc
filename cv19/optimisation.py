# https://www.linkedin.com/pulse/coronavirus-policy-design-stable-population-recovery-greg-stewart/?trackingId=oU9sqJxvTm6SWOwdg0d5Lw%3D%3D
# https://github.com/jckantor/covid-19

import numpy as np
# import sympy as sym
import pickle as pkl
# from opty.direct_collocation import Problem
# from opty.utils import building_docs
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.animation as animation
from scipy.integrate import odeint


def step(t):
    return 1 if t >= 7 * t_social_distancing else 0


# SEIR model differential equations.
def deriv(x, t, u, alpha, beta, gamma):
    s, e, i, r = x
    dsdt = -(1 - u * step(t) / 100) * beta * s * i
    dedt = (1 - u * step(t) / 100) * beta * s * i - alpha * e
    didt = alpha * e - gamma * i
    drdt = gamma * i
    return [dsdt, dedt, didt, drdt]


if __name__ == "__main__":

    R0 = 2.4  # @param {type:"slider", min:0.9, max:5, step:0.1}
    t_incubation = 5.1  # @param {type:"slider", min:1, max:14, step:0.1}
    t_infective = 3.3  # @param {type:"slider", min:1, max:14, step:0.1}
    N = 14000  # @param {type:"slider", min:1000, max:350000, step: 1000}
    n = 10  # @param {type:"slider", min:0, max:100, step:1}
    t_social_distancing = 2  # @param {type:"slider", min:0, max:30, step:0.1}
    u_social_distancing = 40  # @param {type:"slider", min:0, max:100, step:1}

    # initial number of infected and recovered individuals
    e_initial = n / N
    i_initial = 0.00
    r_initial = 0.00
    s_initial = 1 - e_initial - i_initial - r_initial

    alpha = 1 / t_incubation
    gamma = 1 / t_infective
    beta = R0 * gamma

    t = np.linspace(0, 210, 210)
    x_initial = s_initial, e_initial, i_initial, r_initial
    s, e, i, r = odeint(deriv, x_initial, t, args=(u_social_distancing, alpha, beta, gamma)).T
    s0, e0, i0, r0 = odeint(deriv, x_initial, t, args=(0, alpha, beta, gamma)).T

    a=1