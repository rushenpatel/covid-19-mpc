# https://www.linkedin.com/pulse/coronavirus-policy-design-stable-population-recovery-greg-stewart/?trackingId=oU9sqJxvTm6SWOwdg0d5Lw%3D%3D
# https://github.com/jckantor/covid-19

import numpy as np
import pickle as pkl
from casadi import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt


class SeirModel:

    def __init__(self, horizon=210):
        R0= 2.4                 # R0 for covid-19 based on china data
        t_incubation = 5.1      # incubation period
        t_infective = 3.3       # infectious period
        self.N = 100000         # population
        n = 100                  # number of population initially exposed
        self.u_social_distancing_baseline = 40  # effectiveness of social distancing %

        hospitalisation = 4.4 / 100.    # 4.4% of those exposed require hospitalisation
        critical_care = 30. / 100.      # 30% of those hospitalised require critical care [IC paper]
        self.icu_load = hospitalisation * critical_care
        self.max_bed_capacity = 20 / self.N    # number of ICU beds per N of population
        self.target = 0.9         # target proportion of peak bed capacity

        self.horizon = horizon
        self.time = np.linspace(0, horizon, horizon)

        self.alpha = 1 / t_incubation
        self.gamma = 1 / t_infective
        self.beta = R0 * self.gamma

        self.u_min = 0.0
        self.u_max = 50.0

        # initial number of infected and recovered individuals
        e_initial = n / self.N
        i_initial = 0.00
        r_initial = 0.00
        s_initial = 1 - e_initial - i_initial - r_initial
        h_initial = self.icu_load * e_initial
        self.x_initial = s_initial, e_initial, i_initial, r_initial, h_initial

    # SEIR model differential equations (from Kantor).
    def state_space(self, x, t):
        s, e, i, r, h = x
        dsdt = -(1 - self.u_social_distancing_baseline ) * self.beta * s * i
        dedt = (1 - self.u_social_distancing_baseline ) * self.beta * s * i - self.alpha * e
        didt = self.alpha * e - self.gamma * i
        drdt = self.gamma * i
        dhdt = self.icu_load * dedt
        return [dsdt, dedt, didt, drdt, dhdt]

    def integrate(self):
        s, e, i, r, h = odeint(self.state_space, self.x_initial, self.time).T
        return s, e, i, r, h

    def casadi_model(self, u_fixed=True, u_profile=None, w_initial=None, bed_constraint=False, u_start_week=0):
        assert len(u_sd) == self.horizon / 7, 'input length vector is incorrect'

        # model variables
        s = MX.sym('s')
        e = MX.sym('e')
        i = MX.sym('i')
        r = MX.sym('r')
        h = MX.sym('h')
        u = MX.sym('u')

        x = vertcat(s, e, i, r, h)
        x_size = x.shape[0]

        dsdt = -(1 - u) * self.beta * s * i
        dedt = (1 - u) * self.beta * s * i - self.alpha * e
        didt = self.alpha * e - self.gamma * i
        drdt = self.gamma * i
        dhdt = self.icu_load * dedt
        x_dot = vertcat(dsdt, dedt, didt, drdt, dhdt)

        integral_objective = u**2 + 10000 * v**2

        # no_control_intervals = self.horizon
        no_control_intervals = int(self.horizon / 7)
        integrator = self.integrator_func(self.horizon, no_control_intervals,
                                          x, x_dot, u, integral_objective)

        # Empty NLP
        w = []  # soln vector
        w0 = []  # initial guess
        lbw = []  # state / input lower bound
        ubw = []  # state / input upper bound
        objective = 0  # objective
        g = []  # constraint vector
        lbg = []  # constraint lower bound
        ubg = []  # constraint upper bound

        # Formulate the NLP
        Xk = MX.sym('X0', x_size)
        w.append(Xk)
        lbw += list(self.x_initial)
        ubw += list(self.x_initial)
        w0 += list(self.x_initial)

        for k in range(no_control_intervals):
            # New NLP variable for the control
            Uk = MX.sym('U' + str(k))
            w.append(Uk)
            if u_fixed:
                lbw += [u_profile[k]]
                ubw += [u_profile[k]]
                w0 += [u_profile[k]]
            else:
                if k >= u_start_week:
                    lbw += [0]
                    ubw += [1]
                    w0 += [u_profile[k]]
                else:
                    lbw += [0]
                    ubw += [0]
                    w0 += [0]

            # Integrate until the end of the interval
            Fk = integrator(x0=Xk, p=Uk)
            Xk_end = Fk['xf']
            objective = objective + Fk['qf']

            # New NLP variable for state at end of interval
            Xk = MX.sym('X' + str(k + 1), x_size)
            w.append(Xk)
            lbw += [0, 0, 0, 0, 0]
            if bed_constraint:
                # ubw += [1, 1, 1, 1, self.target * self.max_bed_capacity]
                ubw += [1, 1, 1, 1, 1]
            else:
                ubw += [1, 1, 1, 1, 1]
            w0 += list(self.x_initial)

            # Add equality constraints (state continuity)
            g.append(Xk_end - Xk)
            lbg += [0] * x_size
            ubg += [0] * x_size

        # if provided with initial guess, use it
        if w_initial is not None:
            w0 = w_initial

        # Create an NLP solver
        opts = {}
        opts['ipopt.acceptable_tol'] = 1e-3
        prob = {'f': objective, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)

        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        time = [
            self.horizon / no_control_intervals * k
            for k in range(no_control_intervals + 1)
        ]
        s_opt = w_opt[0::x_size + 1]
        e_opt = w_opt[1::x_size + 1]
        i_opt = w_opt[2::x_size + 1]
        r_opt = w_opt[3::x_size + 1]
        h_opt = w_opt[4::x_size + 1]
        u_opt = w_opt[5::x_size + 1]

        a=1
        result = {
                  'time': time,
                  's_opt': s_opt,
                  'e_opt': e_opt,
                  'i_opt': i_opt,
                  'r_opt': r_opt,
                  'h_opt': h_opt,
                  'u_opt': u_opt,
                  'w_opt': w_opt
                }

        return result

    def integrator_func(self, horizon, no_control_points, x,
                        x_dot, u, integral_objective):
        # Fixed step Runge-Kutta 4 integrator
        M = 4  # RK4 steps per interval
        DT = horizon / no_control_points / M
        f = Function('f', [x, u], [x_dot, integral_objective])
        X0 = MX.sym('X0', x.shape[0])
        U = MX.sym('U')
        X = X0
        Q = 0
        for j in range(M):
            k1, k1_q = f(X, U)
            k2, k2_q = f(X + DT / 2 * k1, U)
            k3, k3_q = f(X + DT / 2 * k2, U)
            k4, k4_q = f(X + DT * k3, U)
            X = X + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            Q = Q + DT / 6 * (k1_q + 2 * k2_q + 2 * k3_q + k4_q)
        return Function('integrator', [X0, U], [X, Q], ['x0', 'p'], ['xf', 'qf'])

    def plot_results(self, result):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(result['time'], result['s_opt'], label='susceptible', linewidth=2.0)
        ax1.plot(result['time'], result['e_opt'], label='exposed / no symptoms', linewidth=2.0)
        ax1.plot(result['time'], result['i_opt'], label='infectious / symptomatic', linewidth=2.0)
        ax1.plot(result['time'], result['r_opt'], label='recovered', linewidth=2.0)
        ax2.plot(result['time'], self.N * result['h_opt'], label='ICU beds required', linewidth=2.0)
        ax2.plot(result['time'], [self.N * self.max_bed_capacity] * len(result['time']), label='peak bed capacity', color='black', linewidth=2.0)
        ax3.step(result['time'][:-1], result['u_opt']*100, label='u_opt', linewidth=2.0)
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax1.legend()
        ax2.legend()
        ax3.legend()
        ax3.set_xlabel('days')
        ax1.set_ylabel('proportion of total popn')
        ax2.set_ylabel('# per 100,000 popn')
        ax3.set_ylabel('social distancing, % effectiveness')
        return fig








if __name__ == "__main__":

    horizon = 210
    sm = SeirModel(horizon=horizon)

    # Open Loop
    # fixed social distancing profiles

    # constant 40% social distancing
    u_sd = [0.4] * int(horizon / 7)
    result = sm.casadi_model(u_fixed=True, u_profile=u_sd, w_initial=None)
    fig = sm.plot_results(result)
    fig.suptitle('Constant social distancing of 40%')

    # constant 0% social distancing
    w_initial = result['w_opt']
    u_sd = [0.0] * int(horizon / 7)
    result = sm.casadi_model(u_fixed=True, u_profile=u_sd, w_initial=w_initial)
    fig = sm.plot_results(result)
    fig.suptitle('Constant social distancing of 0%')

    # # 40% social distancing applied for two months only
    # u_sd = [0] * int(horizon / 7)
    # w_initial = result['w_opt']
    #
    # for i in range(5, 9):
    #     u_sd[i] = 40
    # result = sm.casadi_model(u_fixed=True, u_profile=u_sd, w_initial=w_initial)
    # fig = sm.plot_results(result)
    # fig.suptitle('Social distancing applied after one month for two months duration')

    # constant 0% social distancing
    w_initial = result['w_opt']
    u_sd = [0.0] * int(horizon / 7)
    result = sm.casadi_model(u_fixed=False, u_profile=u_sd, w_initial=w_initial, bed_constraint=True, u_start_week=4)
    fig = sm.plot_results(result)
    fig.suptitle('Open loop optimal social distancing effectiveness (u)')

    plt.show()

    # horizon = 210
    # sm = SeirModel(horizon=horizon)
    # s, e, i, r, h = sm.integrate()
    # sm.u_social_distancing = 0
    # s0, e0, i0, r0, h0 = sm.integrate()
    #
    # plt.figure()
    # plt.plot(sm.time, e0, 'g')
    # plt.plot(sm.time, i0, 'b')
    # plt.plot(sm.time, h0, 'r')
    # plt.show()