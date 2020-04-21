from cv19.mpc import SeirModel
import matplotlib.pyplot as plt

horizon = 210

######### OPEN LOOP (run with n=100)

# constant 40% social distancing
sm = SeirModel(horizon=horizon)
u_sd = [0.4] * int(horizon)
result = sm.casadi_model(u_fixed=True, u_profile=u_sd, w_initial=None, objective_selector=0)
fig = sm.plot_results(result)
fig.suptitle('Constant social distancing of 40%')

# constant 0% social distancing
sm = SeirModel(horizon=horizon)
w_initial = result['w_opt']
u_sd = [0.0] * int(horizon)
result = sm.casadi_model(u_fixed=True,
                         u_profile=u_sd,
                         w_initial=w_initial,
                         objective_selector=0)
fig = sm.plot_results(result)
fig.suptitle('Constant social distancing of 0%')

# social distancing for 2 months from week 4
sm = SeirModel(horizon=horizon)
w_initial = result['w_opt']
u_sd = [0.0] * int(horizon)
for i in range(21, 81):
    u_sd[i] = 0.4
result = sm.casadi_model(u_fixed=True,
                         u_profile=u_sd,
                         w_initial=w_initial,
                         objective_selector=0)
fig = sm.plot_results(result)
fig.suptitle('Social distancing for 2 months starting from week 4')

# optimal social distancing effectiveness
sm = SeirModel(horizon=horizon)
sm.u_max = 0.5
result = sm.casadi_model(u_fixed=False,
                         u_profile=None,
                         w_initial=None,
                         objective_selector=2,
                         u_start_day=14)
fig = sm.plot_results(result)
fig.suptitle('Open loop optimal social distancing effectiveness (u)')

# optimal social distancing effectiveness with increasing bed capacity
sm = SeirModel(horizon=horizon)
sm.u_max = 0.4
result = sm.casadi_model(u_fixed=False,
                         u_profile=None,
                         w_initial=None,
                         objective_selector=2,
                         u_start_day=14,
                         increasing_capacity=(50, 60))
fig = sm.plot_results(result)
fig.suptitle(
    'Open loop optimal social distancing effectiveness (u), with increasing bed capacity'
)
plt.show()


##### CLOSED LOOP MPC

# sm = SeirModel(horizon=horizon, n=100)
# sm.u_max = 0.4
# sim_result = sm.mpc(u_start_day=14)
# fig = sm.plot_results(sim_result)
# fig.suptitle(
#     'MPC feedback control with fixed ICU peak bed capacity constraint'
# )
#
#
# sm = SeirModel(horizon=horizon, n=100)
# sim_result = sm.mpc(u_start_day=14, increasing_capacity=(50, 60))
# fig = sm.plot_results(sim_result)
# fig.suptitle(
#     'MPC feedback control with increasing ICU peak bed capacity constraint'
# )
# plt.show()