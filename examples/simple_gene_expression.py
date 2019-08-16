# Demonstrate simulate on synthetic gene expression model
# different implementations are used and compared

# import stuff
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import profile

# custom files
from pyssa.models.kinetic_model import KineticModel 
from pyssa.models.kinetic_model import PhysicalKineticModel
import pyssa.models.standard_models as sm
import pyssa.ssa as ssa

# activate or deactivate plotting
plotting = False

# set up the model
pre, post, rates = sm.get_standard_model("simple_gene_expression")
model = KineticModel(np.array(pre), np.array(post), np.array(rates))

# prepare initial conditions
initial = np.array([0.0, 1.0, 0.0, 0.0])
tspan = np.array([0.0, 3e3])
delta_t = 300.0
obs_times = np.arange(tspan[0]+0.5*delta_t, tspan[1], delta_t)

# set up simulator
simulator = ssa.Simulator(model, initial)

# get trajectory 
trajectory = simulator.simulate(initial, tspan)
simulator.events2states(trajectory)

# get a subsampling for plotting
t_plot = np.linspace(tspan[0], tspan[1], 200)
states_plot = ssa.discretize_trajectory(trajectory, t_plot)

# get mean
states_avg = ssa.sample(model, initial, t_plot, num_samples=10, output='avg')
print(states_avg.shape)

# plot result 
if plotting:
    plt.plot(t_plot, 100*states_plot[:, 1], '-k')
    plt.plot(t_plot, states_plot[:, 2], '-b')
    plt.plot(t_plot, states_plot[:, 3], '-r')
    #plt.plot(obs_times.numpy(), obs.numpy(), 'xk')
    plt.show()

    plt.plot(t_plot, 100*states_avg[:, 1], '-k')
    plt.plot(t_plot, states_avg[:, 2], '-b')
    plt.plot(t_plot, states_avg[:, 3], '-r')
    #plt.plot(obs_times.numpy(), obs.numpy(), 'xk')
    plt.show()