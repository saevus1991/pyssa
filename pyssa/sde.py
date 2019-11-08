# class for sde simulation

import numpy as np
import pyssa.util as ut
from scipy.interpolate import interp1d
import time


class Simulator:
    """
    This class implements a Euler-Maruyama discretization for SDEs
    """

    def __init__(self, model, timestep, projector=None):
        self.model = model
        self.timestep = timestep
        self.projector = self.set_projector(projector)
        self.diff_dim = model.get_dimension()

    def simulate(self, initial, tspan):
        # initialize
        times = np.arange(tspan[0], tspan[1], self.timestep)
        states = np.zeros((len(times), len(initial)))
        states[0] = initial
        noise_std = np.sqrt(self.timestep)
        # iterate
        for i in range(len(times)-1):
            drift, diffusion = self.model.eval(states[i], times[i])
            noise = np.random.standard_normal(self.diff_dim[1])*noise_std
            states[i+1] = states[i] + drift*self.timestep + diffusion@noise
            states[i+1] = self.projector(states[i+1])
        # produce output dictionary
        trajectory = {'initial': initial, 'tspan': tspan, 'times': times, 'states': states}
        return(trajectory)

    def set_projector(self, projector):
        if projector is None:
            def id(x): return(x)
            return(id)
        else:
            return(projector)


def discretize_trajectory(trajectory, sample_times, obs_model=None):
    """ 
    Discretize a trajectory of a jump process by linear interpolation 
    at the support points given in sample times
    Input
        trajectory: a dict with keys 'initial', 'tspan', 'times', 'states'
        sample_times: np.array containin the sample times
    """
    times = np.concatenate([trajectory['tspan'][0:1], trajectory['times']])
    states = np.concatenate([trajectory['initial'].reshape(1, -1), trajectory['states']])
    sample_states = interp1d(times, states, kind='zero', axis=0)(sample_times)
    if obs_model is not None:
        test = obs_model.sample(states[0], sample_times[0])
        obs_dim = (obs_model.sample(states[0], sample_times[0])).size
        obs_states = np.zeros((sample_states.shape[0], obs_dim))
        for i in range(len(sample_times)):
            obs_states[i] = obs_model.sample(sample_states[i], sample_times[i])
        sample_states = obs_states
    return(sample_states)

def sample(model, initial, sample_times, time_step=None, projector=None, num_samples=1, output='full'):
    """
    Draw num_samples samples from the model for a given initial
    Discretize the trajectories over the grid sample_times
    Store in an num_samples x times x num_speies array
    """
    # set time step
    if time_step is None:
        time_step = 0.1*np.mean(sample_times[1:]-sample_times[:-1])
    # set up output
    samples = np.zeros((num_samples, len(sample_times), len(initial)))
    # get tspan
    tspan = np.array([sample_times[0], sample_times[-1]+10*time_step])
    # set up model
    sim = Simulator(model, time_step, projector)
    start = time.time()
    for i in range(num_samples):
        # run simulation 
        trajectory = sim.simulate(initial, tspan)
        # discretize
        sample_states = discretize_trajectory(trajectory, sample_times)
        # store in output array
        samples[i, :, :] = sample_states
    end = time.time()
    print('Generated {0} samples in {1} seconds.'.format(num_samples, end-start))
    if output == 'full':
        return(samples)
    elif output == 'avg':
        return(np.mean(samples, axis=0).squeeze())
    else:
        raise ValueError('Unknown value '+output+' for option output.')