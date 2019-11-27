# Demonstrate simulate on synthetic gene expression model
# different implementations are used and compared

# import stuff
import matplotlib.pyplot as plt
import numpy as np

# custom files
from pyssa.models.special_models import TASEP
import pyssa.ssa as ssa

# activate or deactivate plotting
plotting = True

# set up the model
num_sites = 48
num_stems = 14
rates = np.array([0.05, 0.5, 0.05])
model = TASEP(num_sites, rates)
stems = np.concatenate([np.array([(i+1) for i in range(14)]), num_stems*np.ones(num_sites-num_stems)])

# prepare initial conditions
initial = np.zeros(num_sites)
tspan = np.array([0.0, 120*10])
# delta_t = 300.0
# obs_times = np.arange(tspan[0]+0.5*delta_t, tspan[1], delta_t)

# set up simulator
simulator = ssa.Simulator(model, initial)

# get trajectory 
trajectory = simulator.simulate(initial, tspan)
simulator.events2states(trajectory)

# get a subsampling for plotting
t_plot = np.linspace(tspan[0], tspan[1], 200)
states_discrete = ssa.discretize_trajectory(trajectory, t_plot)
pol_tr = np.sum(states_discrete, axis=1)
stem_tr = states_discrete@stems

# get mean
states_avg = ssa.sample(model, initial, t_plot, num_samples=100, output='avg')
print(states_avg.shape)

# plot result 
if plotting:
    plt.plot(t_plot, pol_tr, '-k')
    #plt.plot(obs_times.numpy(), obs.numpy(), 'xk')
    plt.plot(t_plot, stem_tr/14, '-r')
    plt.show()

#     plt.plot(t_plot, 100*states_avg[:, 1], '-k')
#     plt.plot(t_plot, states_avg[:, 2], '-b')
#     plt.plot(t_plot, states_avg[:, 3], '-r')
#     #plt.plot(obs_times.numpy(), obs.numpy(), 'xk')
#     plt.show()