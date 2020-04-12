# References:
# https://www.linkedin.com/pulse/coronavirus-policy-design-stable-population-recovery-greg-stewart/?trackingId=oU9sqJxvTm6SWOwdg0d5Lw%3D%3D
# https://github.com/jckantor/covid-19

import numpy as np
import pickle
from casadi import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import random


class SeirModel:
    def __init__(self, horizon=210, n=100):
        self.R0 = 2.4  # R0 for covid-19 based on china data [1]
        t_incubation = 5.1  # incubation period
        t_infective = 3.3  # infectious period
        self.N = 100000  # population
        self.n = n  # number of population initially exposed
        self.u_social_distancing_baseline = 40  # effectiveness of social distancing %

        hospitalisation = 4.4 / 100.  # 4.4% of those exposed require hospitalisation
        critical_care = 30. / 100.  # 30% of those hospitalised require critical care [IC paper]
        self.icu_load = hospitalisation * critical_care
        self.max_bed_capacity = 20 / self.N  # number of ICU beds per N of population
        self.target = 0.9  # target proportion of peak bed capacity

        self.horizon = horizon
        self.time = np.linspace(0, horizon, horizon)

        self.alpha = 1 / t_incubation
        self.gamma = 1 / t_infective
        # self.beta = self.R0 * self.gamma

        self.u_max = 0.4

        # initial number of infected and recovered individuals
        e_initial = self.n / self.N
        i_initial = 0.00
        r_initial = 0.00
        s_initial = 1 - e_initial - i_initial - r_initial
        h_initial = self.icu_load * e_initial
        v1_initial = 1
        self.x_initial = s_initial, e_initial, i_initial, r_initial, h_initial, v1_initial

    # SEIR model differential equations with noise applied to u and R0 (from Kantor).
    def state_space(self, x, t, u):
        s, e, i, r, h = x
        dsdt = -(1 - u) * self.R0 * self.gamma * s * i
        dedt = (1 - u) * self.R0 * self.gamma * s * i - self.alpha * e
        didt = self.alpha * e - self.gamma * i
        drdt = self.gamma * i
        dhdt = self.icu_load * dedt
        return [dsdt, dedt, didt, drdt, dhdt]

    def integrate(self, t, u, x_initial):
        s, e, i, r, h = odeint(self.state_space, x_initial, t, args=(u, )).T
        return s, e, i, r, h

    def casadi_model(self,
                     u_fixed=True,
                     u_profile=None,
                     w_initial=None,
                     x_initial=None,
                     objective_selector=0,
                     u_start_day=0,
                     horizon=None,
                     increasing_capacity=None,
                     step=None):
        if u_profile:
            assert len(
                u_profile) == self.horizon, 'input length vector is incorrect'

        # model variables
        s = MX.sym('s')
        e = MX.sym('e')
        i = MX.sym('i')
        r = MX.sym('r')
        h = MX.sym('h')
        u = MX.sym('u')
        v1 = MX.sym('v1')  # slack variable

        # slack variable constraints
        # h - ht <= v1
        # v1 >= 0
        # where ht = max_bed_capacity * target

        x = vertcat(s, e, i, r, h, v1)
        x_size = x.shape[0]

        dsdt = -(1 - u) * self.R0 * self.gamma * s * i
        dedt = (1 - u) * self.R0 * self.gamma * s * i - self.alpha * e
        didt = self.alpha * e - self.gamma * i
        drdt = self.gamma * i
        dhdt = self.icu_load * dedt
        dv1dt = 0
        x_dot = vertcat(dsdt, dedt, didt, drdt, dhdt, dv1dt)

        if objective_selector == 0:
            integral_objective = 0
        elif objective_selector == 1:
            integral_objective = (10 * u)**2
        elif objective_selector == 2:
            integral_objective = (10 * u)**2 + 1e9 * v1
        else:
            assert False, 'objective not found'

        no_control_intervals = horizon if horizon else self.horizon
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

        x_init = x_initial if x_initial else self.x_initial

        # Formulate the NLP
        Xk = MX.sym('X0', x_size)
        w.append(Xk)
        lbw += list(x_init[:-1]) + [0]
        ubw += list(x_init[:-1]) + [inf]
        w0 += list(x_init)

        capacity_count = 0
        if increasing_capacity:
            capacity_count = step if step else 0
        ht_vec = []
        for k in range(no_control_intervals):
            # New NLP variable for the control
            Uk = MX.sym('U' + str(k))
            w.append(Uk)
            if u_fixed:
                lbw += [u_profile[k]]
                ubw += [u_profile[k]]
                w0 += [u_profile[k]]
            else:
                if k >= u_start_day:
                    lbw += [0.0]
                    ubw += [self.u_max]
                    w0 += [self.u_max]
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
            ubw += [1, 1, 1, 1, 1]
            # slack variables
            lbw += [0]
            ubw += [inf]
            w0 += list(self.x_initial)

            # Add equality constraints (state continuity)
            g.append(Xk_end[:-1] - Xk[:-1])
            lbg += [0] * (x_size - 1)
            ubg += [0] * (x_size - 1)

            # Add slack variable constraints
            _h = Xk_end[4]
            # increasing capacity constraint
            if increasing_capacity and k >= u_start_day:
                _ht = self.target * self.max_bed_capacity + self.target * (
                    (increasing_capacity[0] / self.N) -
                    self.max_bed_capacity) * (capacity_count / increasing_capacity[1])
                if _ht > self.target * increasing_capacity[0] / self.N:
                    _ht = self.target * increasing_capacity[0] / self.N
                capacity_count += 1
            else:
                _ht = self.target * self.max_bed_capacity
            _v1 = Xk_end[5]
            g.append((_h - _ht) - _v1)
            lbg += [-inf]
            ubg += [0]

            ht_vec.append(_ht)

        # if provided with initial guess, use it
        if w_initial is not None:
            w0 = w_initial

        # Create an NLP solver
        opts = {}
        # opts['ipopt.acceptable_tol'] = 1e-3
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
        v1_opt = w_opt[5::x_size + 1]
        u_opt = w_opt[6::x_size + 1]

        result = {
            'time': time,
            's_opt': s_opt,
            'e_opt': e_opt,
            'i_opt': i_opt,
            'r_opt': r_opt,
            'h_opt': h_opt,
            'u_opt': u_opt,
            'v1_opt': v1_opt,
            'w_opt': w_opt,
            'ht_vec': ht_vec
        }
        return result

    def mpc(self, u_start_day=0, increasing_capacity=None):
        result = None

        s_vec = [self.x_initial[0]]
        e_vec = [self.x_initial[1]]
        i_vec = [self.x_initial[2]]
        r_vec = [self.x_initial[3]]
        h_vec = [self.x_initial[4]]
        u_vec = []
        u_actual_vec = []
        ht_vec = []
        time = [0]

        for k in range(self.horizon):
            time.append(k + 1)
            x_initial = (s_vec[-1], e_vec[-1], i_vec[-1], r_vec[-1], h_vec[-1])
            horizon = self.horizon - k
            horizon = 30 if horizon <= 30 else horizon  # minimum 30 day look ahead

            if k < u_start_day:
                ht_vec.append(self.target * self.max_bed_capacity)
                u_vec.append(0.0)
                u_actual = 0
                u_actual_vec.append(u_actual)
                s, e, i, r, h = self.integrate(t=(0, 1),
                                               u=u_actual,
                                               x_initial=x_initial)
            else:
                if result:
                    result = self.casadi_model(
                        u_fixed=False,
                        u_profile=None,
                        w_initial=None,  # warm start optimisation
                        # w_initial=result['w_opt'],  # warm start optimisation
                        objective_selector=2,
                        u_start_day=0,
                        horizon=horizon,
                        increasing_capacity=increasing_capacity,
                        step=k-u_start_day)
                else:
                    result = self.casadi_model(u_fixed=False,
                                               u_profile=None,
                                               w_initial=None,
                                               objective_selector=2,
                                               u_start_day=0,
                                               horizon=horizon,
                                               increasing_capacity=increasing_capacity,
                                               step=k-u_start_day)
                u_vec.append(result['u_opt'][0])
                u_actual = result['u_opt'][0] - random.uniform(
                    0, 0.1) * self.u_max  # apply noise to u
                u_actual_vec.append(u_actual)
                s, e, i, r, h = self.integrate(t=(0, 1),
                                               u=u_actual,
                                               x_initial=x_initial)
                ht_vec.append(result['ht_vec'][0])

            print('----------------------')
            print('STEP NUMBER:', k)
            print('SEIR:', s, e, i, r, h)
            print('X_INIT:', x_initial)
            print('U:', u_vec[-1])
            print('U_ACTUAL:', u_actual)
            print('HORIZON:', horizon)
            print('----------------------')
            s_vec.append(s[1])
            e_vec.append(e[1])
            i_vec.append(i[1])
            r_vec.append(r[1])
            h_vec.append(max(0, h[1]))

            data = {
                'time': np.array(time),
                's_opt': np.array(s_vec),
                'e_opt': np.array(e_vec),
                'i_opt': np.array(i_vec),
                'r_opt': np.array(r_vec),
                'h_opt': np.array(h_vec),
                'u_opt': np.array(u_vec),
                'u_actual': np.array(u_actual_vec),
                'ht_vec': np.array(ht_vec)
            }
        return data

    def integrator_func(self, horizon, no_control_points, x, x_dot, u,
                        integral_objective):
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
        return Function('integrator', [X0, U], [X, Q], ['x0', 'p'],
                        ['xf', 'qf'])

    def plot_results(self, result):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
        ax1.plot(result['time'],
                 result['s_opt'],
                 label='susceptible',
                 linewidth=2.0)
        ax1.plot(result['time'],
                 result['e_opt'],
                 label='exposed / no symptoms',
                 linewidth=2.0)
        ax1.plot(result['time'],
                 result['i_opt'],
                 label='infectious / symptomatic',
                 linewidth=2.0)
        ax1.plot(result['time'],
                 result['r_opt'],
                 label='recovered',
                 linewidth=2.0)
        ax2.plot(result['time'],
                 self.N * result['h_opt'],
                 label='ICU beds required',
                 linewidth=2.0)
        ax2.plot(result['time'][:-1],
                 self.N * 1/0.9 * np.array(result['ht_vec']),
                 # [self.N * self.max_bed_capacity] * len(result['time']),
                 label='peak bed capacity',
                 color='black',
                 linewidth=2.0)
        ax3.step(result['time'][:-1],
                 result['u_opt'] * 100,
                 label='u_opt',
                 linewidth=2.0)
        if 'u_actual' in result.keys():
            ax3.step(result['time'][:-1],
                     result['u_actual'] * 100,
                     label='u_actual',
                     linewidth=2.0)
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

    # # OPEN LOOP (run with n=100)
    sm = SeirModel(horizon=horizon)

    # constant 40% social distancing
    # u_sd = [0.4] * int(horizon)
    # result = sm.casadi_model(u_fixed=True, u_profile=u_sd, w_initial=None, objective_selector=0)
    # fig = sm.plot_results(result)
    # fig.suptitle('Constant social distancing of 40%')
    #
    # # constant 0% social distancing
    # w_initial = result['w_opt']
    # u_sd = [0.0] * int(horizon)
    # result = sm.casadi_model(u_fixed=True,
    #                          u_profile=u_sd,
    #                          w_initial=w_initial,
    #                          objective_selector=0)
    # fig = sm.plot_results(result)
    # fig.suptitle('Constant social distancing of 0%')
    #
    # social distancing for 2 months from week 4
    # w_initial = result['w_opt']
    # u_sd = [0.0] * int(horizon)
    # for i in range(21, 81):
    #     u_sd[i] = 0.4
    # result = sm.casadi_model(u_fixed=True,
    #                          u_profile=u_sd,
    #                          w_initial=w_initial,
    #                          objective_selector=0)
    # fig = sm.plot_results(result)
    # fig.suptitle('Social distancing for 2 months starting from week 4')
    # plt.show()

    # optimal social distancing effectiveness
    # sm.u_max = 0.5
    # result = sm.casadi_model(u_fixed=False,
    #                          u_profile=None,
    #                          w_initial=None,
    #                          objective_selector=2,
    #                          u_start_day=14)
    # fig = sm.plot_results(result)
    # fig.suptitle('Open loop optimal social distancing effectiveness (u)')

    # optimal social distancing effectiveness
    # sm.u_max = 0.4
    # result = sm.casadi_model(u_fixed=False,
    #                          u_profile=None,
    #                          w_initial=None,
    #                          objective_selector=2,
    #                          u_start_day=14,
    #                          increasing_capacity=(50, 60))
    # fig = sm.plot_results(result)
    # fig.suptitle(
    #     'Open loop optimal social distancing effectiveness (u), with delayed start'
    # )
    # plt.show()

    # closed loop MPC
    # sm = SeirModel(horizon=horizon, n=100)
    # sim_result = sm.mpc(u_start_day=14)
    # fig = sm.plot_results(sim_result)
    # fig.suptitle(
    #     'MPC feedback control with fixed ICU peak bed capacity constraint'
    # )

    sm = SeirModel(horizon=210, n=100)
    sim_result = sm.mpc(u_start_day=14, increasing_capacity=(50, 60))
    fig = sm.plot_results(sim_result)
    fig.suptitle(
        'MPC feedback control with increasing ICU peak bed capacity constraint'
    )
    plt.show()
