"""
Perform a simulation from the gene expression model
Optimize conrols via smoothing.
Use the optimized control as to check the time-dependent tasep simulation.
This file requires acces to the pymbvi package. 
"""

# import stuff
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
#import scipy.optimize as opt
import scipy.integrate as quad

# custom files
#from pyssa.models.special_models import TASEP
import pyssa.models.standard_models as sm
import pyssa.ssa as ssa
from pyssa.models.kinetic_model import KineticModel 
import pyssa.ssa_compiled.gillespie_timedep as gillespie

# include pymbvi
pymbvi_path = '/Users/christian/Documents/Code/pymbvi'
sys.path.append(pymbvi_path)
from pymbvi.models.observation.kinetic_obs_model import LognormObs
from pymbvi.models.mjp.autograd_partition_specific_models import SimpleGeneExpression
from pymbvi.variational_engine import VariationalEngine
from pymbvi.optimize import robust_gradient_descent

# fix seed
#np.random.seed(2002281414)

# load data
data_file = '/Users/christian/Documents/Code/pyssa/data/gene_expression_controls.npz'
data = np.load(data_file)

# parse data
t_sim = data['t_sim']
states_sim = data['states_sim']
t_prior = data['t_prior']
states_prior = data['states_prior']
control = data['control']
time = data['time']
t_obs = data['t_obs']
observations = data['observations']
t_plot = data['t_plot']
states_plot = data['states_plot']
t_post = data['t_post']
states_post = data['states_post']

# set up the model
pre, post, rates = sm.get_standard_model("simple_gene_expression")
pre = np.array(pre)
post = np.array(post)
rates = np.array(rates)
model = KineticModel(np.array(pre), np.array(post), np.array(rates))

# prepare initial conditions
initial = np.array([0.0, 1.0, 0.0, 0.0])
tspan = np.array([0.0, 3e3])
delta_t = 300.0
obs_times = np.arange(tspan[0]+0.5*delta_t, tspan[1], delta_t)
time_grid = np.append(time[:, :, 0].flatten(), tspan[1])
control = np.exp(control.reshape((-1, control.shape[-1]))) * np.expand_dims(rates, axis=0)
seed = np.random.randint(2**16)

# simulate 
trajectory = gillespie.simulate(pre, post, control, time_grid, initial, tspan, seed)

# set up an observation model
sigma = np.array([0.15])
num_species = 4
obs_species = 3
obs_model = LognormObs(sigma, num_species, obs_species, num_species-1, obs_species-1)

# # get trajectory 
# trajectory = simulator.simulate(initial, tspan)
# simulator.events2states(trajectory)

# get a subsampling for plotting
t_sample = np.linspace(tspan[0], tspan[1], 200)
states_sample = ssa.discretize_trajectory(trajectory, t_plot)

# # produce observations 
# delta_t = 300.0
# t_obs = np.arange(tspan[0]+0.5*delta_t, tspan[1], delta_t)
# observations = ssa.discretize_trajectory(trajectory, t_obs, obs_model=obs_model)

# # set up gene expression model
# moment_initial = np.zeros(9)
# moment_initial[0] = 1
# model = SimpleGeneExpression(moment_initial, np.log(np.array(rates)), tspan)

# # get forward  mean and stuff
# def ode_fun(time, state):#, control_time, control):
#     return(model.forward(time, state, np.zeros(len(rates)), np.log(np.array(rates))))

# # solve 
# sol = solve_ivp(ode_fun, tspan, moment_initial, t_eval=t_plot)
# t_prior = sol['t']
# states_prior = sol['y'].T

# # set up variational engine
# vi_engine = VariationalEngine(moment_initial, model, obs_model, t_obs, observations, subsample=30, tspan=tspan)
# vi_engine.initialize()

# # project to a given domain
# def bound_projector(arg, arg_old):
#     """
#     Cut of values above or below boundary
#     """
#     # set bounds
#     bounds = np.array([-13, 13])
#     # cut off larger values
#     arg[arg > bounds[1]] = bounds[1]
#     # cut off lower values
#     arg[arg < bounds[0]] = bounds[0]

# # test optimization
# initial_control = vi_engine.control.copy()
# optimal_control = robust_gradient_descent(vi_engine.objective_function, initial_control, projector=bound_projector, iter=50, h=1e-3)[0]
# vi_engine.objective_function(optimal_control)

# # get stuff for plotting
# t_post = vi_engine.get_time()
# states_post = vi_engine.get_forward()

# plot result 
plt.subplot(2, 2, 1)
plt.plot(t_plot, 100*states_plot[:, 1], '-k')
plt.plot(t_plot, states_plot[:, 2], '-b')
plt.plot(t_plot, states_plot[:, 3], '-r')
plt.plot(t_obs, observations, 'xk')

plt.subplot(2, 2, 2)
plt.plot(t_sample, 100*states_sample[:, 1], '-k')
plt.plot(t_sample, states_sample[:, 2], '-b')
plt.plot(t_sample, states_sample[:, 3], '-r')
plt.plot(t_obs, observations, 'xk')

plt.subplot(2, 2, 3)
plt.plot(t_prior, 100*states_prior[:, 0], '-k')
plt.plot(t_prior, states_prior[:, 1], '-b')
plt.plot(t_prior, states_prior[:, 2], '-r')
plt.plot(t_obs, observations, 'xk')

plt.subplot(2, 2, 4)
plt.plot(t_post, 100*states_post[:, 0], '-k')
plt.plot(t_post, states_post[:, 1], '-b')
plt.plot(t_post, states_post[:, 2], '-r')
plt.plot(t_obs, observations, 'xk')


plt.show()


# # get stuff for plotting
# t_posterior = vi_engine.get_time()
# states_posterior = vi_engine.get_forward()
# control_posterior = np.exp(vi_engine.get_control())
# t_control = vi_engine.time[:, :, 0].flatten()

# # plotting
# plt.subplot(3, 1, 1)
# plt.plot(t_plot, states_plot@alpha, '-r')
# plt.plot(t_prior, states_prior@alpha, '-b')
# plt.plot(t_posterior, states_posterior@alpha, '-k')

# plt.subplot(3, 1, 2)
# intensity = obs_model.intensity(states_plot, t_plot)
# prior_intensity = obs_model.intensity(states_prior, t_prior)
# posterior_intensity = obs_model.intensity(states_posterior, t_posterior)
# plt.plot(t_plot, intensity, '-r')
# plt.plot(t_prior, prior_intensity, '-b')
# plt.plot(t_posterior, posterior_intensity, '-k')

# plt.subplot(3, 1, 3)
# plt.plot(t_obs, observations, '-k')

# plt.show()

# plt.subplot(3, 1, 1)
# plt.plot(t_control, control_posterior[:, 0])

# plt.subplot(3, 1, 2)
# plt.plot(t_control, control_posterior[:, 1])

# plt.subplot(3, 1, 3)
# plt.plot(t_control, control_posterior[:, 2])

# plt.show()

# outfile = '/Users/christian/Documents/Code/pyssa/data/gene_expression_controls.npz'
# np.savez(outfile, t_sim=trajectory['times'], states_sim=trajectory['states'], t_prior=t_prior, states_prior=states_prior, control=vi_engine.control, time=vi_engine.time, forward=vi_engine.forward, 
#     backward=vi_engine.backward, obs_times=vi_engine.obs_times, obs_data=vi_engine.obs_data,
#     t_plot=t_plot, states_plot=states_plot)