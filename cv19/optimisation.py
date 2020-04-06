# https://www.linkedin.com/pulse/coronavirus-policy-design-stable-population-recovery-greg-stewart/?trackingId=oU9sqJxvTm6SWOwdg0d5Lw%3D%3D
# https://github.com/jckantor/covid-19

import numpy as np
import pickle as pkl
from casadi import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class SeirModel:

    def __init__(self, u_social_distancing=40, timeperiod=210):
        R0 = 2.4  # @param {type:"slider", min:0.9, max:5, step:0.1}
        t_incubation = 5.1  # @param {type:"slider", min:1, max:14, step:0.1}
        t_infective = 3.3  # @param {type:"slider", min:1, max:14, step:0.1}
        self.N = 50000  # @param {type:"slider", min:1000, max:350000, step: 1000}
        n = 10  # @param {type:"slider", min:0, max:100, step:1}
        # t_social_distancing = 2  # @param {type:"slider", min:0, max:30, step:0.1}
        self.u_social_distancing = u_social_distancing  # @param {type:"slider", min:0, max:100, step:1}

        hospitalisation = 4.4 / 100.    # 4.4% of those infected require hospitalisation
        critical_care = 30. / 100.      # 30% of those hospitalised require critical care   [IC paper]
        self.icu_load = hospitalisation * critical_care

        self.time = np.linspace(0, timeperiod, timeperiod)

        self.alpha = 1 / t_incubation
        self.gamma = 1 / t_infective
        self.beta = R0 * self.gamma

        # initial number of infected and recovered individuals
        e_initial = n / self.N
        i_initial = 0.00
        r_initial = 0.00
        s_initial = 1 - e_initial - i_initial - r_initial
        h_initial = self.icu_load * i_initial
        self.x_initial = s_initial, e_initial, i_initial, r_initial, h_initial

    # SEIR model differential equations.
    def state_space(self, x, t):
        s, e, i, r, h = x
        dsdt = -(1 - self.u_social_distancing / 100) * self.beta * s * i
        dedt = (1 - self.u_social_distancing / 100) * self.beta * s * i - self.alpha * e
        didt = self.alpha * e - self.gamma * i
        drdt = self.gamma * i
        dhdt = self.icu_load * didt
        return [dsdt, dedt, didt, drdt, dhdt]

    def integrate(self):
        s, e, i, r, h = odeint(self.state_space, self.x_initial, self.time).T
        return s, e, i, r, h


if __name__ == "__main__":

    sm = SeirModel(u_social_distancing=40, timeperiod=210)
    s, e, i, r, h = sm.integrate()
    sm.u_social_distancing = 0
    s0, e0, i0, r0, h0 = sm.integrate()

    plt.figure()
    plt.plot(sm.time, e0, 'g')
    plt.plot(sm.time, i0, 'b')
    plt.plot(sm.time, h0, 'r')
    plt.show()