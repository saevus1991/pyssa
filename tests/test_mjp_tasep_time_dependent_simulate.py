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
from scipy.interpolate import interp1d

# custom files
from pyssa.models.special_models import TASEP_Timedep
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

# fix rng
np.random.seed(1911291507)

# load stored data
infile = 'data/tasep_controls.npz'
data = np.load(infile)

# get stuff for plotting
t_plot = data['t_plot']
states_plot = data['states_plot']
t_posterior = data['time'].flatten()
states_posterior = data['forward'].reshape((-1, 48))
control_posterior = np.exp(data['control']).reshape((-1, 3))
t_control = data['time'][:, :, 0].flatten()
t_obs = data['obs_times']
observations = data['obs_data']

# prepare  model for simulation
num_stems = 14
alpha, rates, obs_param = sm.get_standard_model("tasep")
model = TASEP_Timedep(len(alpha), np.array(rates), t_control, control_posterior)

# prepare initial conditions
L = 48
initial = np.zeros(L)
tspan = np.array([0.0, 120*10])

# set up simulator
simulator = ssa.Simulator(model, initial, mode='non-homogenous')

# set up an observation model
obs_param[4] = 0.1
obs_model = LognormGauss(np.array(obs_param), np.array(alpha), var_reg=0.2)

# get trajectory 
trajectory = simulator.simulate(initial, tspan)
simulator.events2states(trajectory)


def posterior_probability(trajectory, model, obs_model, obs_times, obs_data):
    """
    Evaluate the posterior likelihood under the model
    """
    # preparatioin
    llh = 0.0
    state = trajectory['initial']
    time = trajectory['tspan'][0]
    # iterate over events
    for ind in range(len(trajectory['times'])):
        if trajectory['times'][ind] < trajectory['tspan'][1]:
            # evaluate propensity of the current state
            rate, prob = model.exit_stats(state)
            event = trajectory['events'][ind]
            # evaluate reaction contribution
            llh += np.log(rate) + np.log(prob[event])
            # evaluate waiting time contribution
            llh += -rate*(trajectory['times'][ind]-time)
            # update state
            time = trajectory['times'][ind]
            state = trajectory['states'][ind]
    # terminal waiting time contribution waiting time contribution
    rate, prob = model.exit_stats(state)
    llh += -rate*(trajectory['tspan'][1]-time)
    # observation contribution
    states_obs = ssa.discretize_trajectory(trajectory, obs_times)
    for (t_obs, obs, state) in zip(obs_times, obs_data, states_obs):
        x = 1
        #llh += obs_model.llh(state, t_obs, obs)
    return(llh)

# simulate a number of posterior trajectories
N = 1000
trajectories = []
msg = "Sampling trajectory {0} of {1}"
for i in range(N):
    if i % 10 == 0:
        print(msg.format(i, N))
    trajectory = simulator.simulate(initial, tspan)
    simulator.events2states(trajectory)
    llh = posterior_probability(trajectory, model, obs_model, t_obs, observations)
    trajectory['llh'] = llh
    trajectories.append(trajectory)

# sort by likelihood
trajectories = sorted(trajectories, key=lambda trajectory:trajectory['llh'], reverse=True)
for i in range(100):
    print(trajectories[i]['llh'])

    trajectory = trajectories[i]

    # get a subsampling for plotting
    #t_plot = np.linspace(tspan[0], tspan[1], 200)
    states_sim = ssa.discretize_trajectory(trajectory, t_plot)

    # # produce observations 
    # delta_t = 10.0
    # t_obs = np.arange(delta_t, tspan[1], delta_t)
    # observations = ssa.discretize_trajectory(trajectory, t_obs, obs_model=obs_model)


    # plotting
    plt.subplot(3, 1, 1)
    plt.plot(t_plot, states_plot@alpha, '-k', label='true')
    plt.plot(t_posterior, states_posterior@alpha, '-r', label='posterior mean')
    plt.plot(t_plot, states_sim@alpha, '-b', label='VMAP')
    plt.legend()

    plt.subplot(3, 1, 2)
    intensity = obs_model.intensity(states_plot, t_plot)
    posterior_intensity = obs_model.intensity(states_posterior, t_posterior)
    sim_intensity = obs_model.intensity(states_sim, t_plot)
    plt.plot(t_plot, intensity, '-k')
    plt.plot(t_posterior, posterior_intensity, '-r')
    plt.plot(t_plot, sim_intensity, '-b')

    plt.subplot(3, 1, 3)
    plt.plot(t_obs, observations, '-k')
    plt.plot(t_posterior, posterior_intensity, '-r')
    plt.plot(t_plot, sim_intensity, '-b')

    plt.show()

    plt.subplot(3, 1, 1)
    plt.plot(t_control, control_posterior[:, 0])

    plt.subplot(3, 1, 2)
    plt.plot(t_control, control_posterior[:, 1])

    plt.subplot(3, 1, 3)
    plt.plot(t_control, control_posterior[:, 2])

    plt.show()
