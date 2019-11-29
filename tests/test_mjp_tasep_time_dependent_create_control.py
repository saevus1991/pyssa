"""
Perform a simulation fromt the tasep model. 
Optimize conrols via smoothing.
Use the optimized control as to check the time-dependent tasep simulation.
This file requires acces to the pymbvi package. 
"""

# import stuff
import sys
import matplotlib.pyplot as plt
import numpy as np
#import scipy.optimize as opt
import scipy.integrate as quad

# custom files
from pyssa.models.special_models import TASEP
import pyssa.models.standard_models as sm
import pyssa.ssa as ssa

# include pymbvi
pymbvi_path = '/Users/christian/Documents/Code/pymbvi'
sys.path.append(pymbvi_path)
from pymbvi.models.observation.tasep_obs_model import LognormGauss
from pymbvi.models.initial.specific_models import ProductBernoulli
from pymbvi.models.mjp.specific_models import BernoulliTasep1
from pymbvi.variational_engine import VariationalEngine
from pymbvi.optimize import robust_gradient_descent

# fix seed
np.random.seed(1910230930)
np.random.seed(1911191616)

# activate plotting
plotting = True

# prepare  model for simulation
num_stems = 14
alpha, rates, obs_param = sm.get_standard_model("tasep")
model = TASEP(len(alpha), np.array(rates))

# prepare initial conditions
initial = np.zeros(48)
tspan = np.array([0.0, 120*10])

# set up simulator
simulator = ssa.Simulator(model, initial)

# set up an observation model
obs_param[4] = 0.1
obs_model = LognormGauss(np.array(obs_param), np.array(alpha), var_reg=0.2)
#obs_model = Gauss(np.array(obs_param), np.array(alpha))

# get trajectory 
trajectory = simulator.simulate(initial, tspan)
simulator.events2states(trajectory)

# get a subsampling for plotting
t_plot = np.linspace(tspan[0], tspan[1], 200)
states_plot = ssa.discretize_trajectory(trajectory, t_plot)

# produce observations 
delta_t = 10.0
t_obs = np.arange(delta_t, tspan[1], delta_t)
observations = ssa.discretize_trajectory(trajectory, t_obs, obs_model=obs_model)

# set up variational tasep model
moment_initial = initial
#rates = [rates[0]] + [rates[1] for i in range(len(alpha)-1)] + [rates[2]]
model = BernoulliTasep1(moment_initial, np.log(np.array(rates)), tspan)

# get forward  mean and stuff
def ode_fun(time, state):#, control_time, control):
    return(model.forward(time, state, np.zeros(model.num_controls()), np.log(np.array(rates))))

# solve 
test = quad.solve_ivp(ode_fun, tspan, moment_initial, t_eval=t_plot)
t_prior = test['t']
states_prior = test['y'].T

# set up variational engine
vi_engine = VariationalEngine(moment_initial, model, obs_model, t_obs, observations, num_controls=2, subsample=10, tspan=tspan)
vi_engine.initialize()

# project to a given domain
def bound_projector(arg, arg_old):
    """
    Cut of values above or below boundary
    """
    # set bounds
    bounds = np.array([-10, 10])
    # cut off larger values
    arg[arg > bounds[1]] = bounds[1]
    # cut off lower values
    arg[arg < bounds[0]] = bounds[0]

# perform optimization
initial_control = vi_engine.control.copy()
optimal_control = robust_gradient_descent(vi_engine.objective_function, initial_control, projector=bound_projector, iter=50, h=1e-3)[0]
vi_engine.objective_function(optimal_control)

# get stuff for plotting
t_posterior = vi_engine.get_time()
states_posterior = vi_engine.get_forward()
control_posterior = np.exp(vi_engine.get_control())
t_control = vi_engine.time[:, :, 0].flatten()

# plotting
plt.subplot(3, 1, 1)
plt.plot(t_plot, states_plot@alpha, '-r')
plt.plot(t_prior, states_prior@alpha, '-b')
plt.plot(t_posterior, states_posterior@alpha, '-k')

plt.subplot(3, 1, 2)
intensity = obs_model.intensity(states_plot, t_plot)
prior_intensity = obs_model.intensity(states_prior, t_prior)
posterior_intensity = obs_model.intensity(states_posterior, t_posterior)
plt.plot(t_plot, intensity, '-r')
plt.plot(t_prior, prior_intensity, '-b')
plt.plot(t_posterior, posterior_intensity, '-k')

plt.subplot(3, 1, 3)
plt.plot(t_obs, observations, '-k')

plt.show()

plt.subplot(3, 1, 1)
plt.plot(t_control, control_posterior[:, 0])

plt.subplot(3, 1, 2)
plt.plot(t_control, control_posterior[:, 1])

plt.subplot(3, 1, 3)
plt.plot(t_control, control_posterior[:, 2])

plt.show()

outfile = 'data/tasep_controls.npz'
np.savez(outfile, control=vi_engine.control, time=vi_engine.time, forward=vi_engine.forward, 
    backward=vi_engine.backward, obs_times=vi_engine.obs_times, obs_data=vi_engine.obs_data,
    t_plot=t_plot, states_plot=states_plot)