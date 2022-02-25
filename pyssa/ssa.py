# ssa class

#import
import sys
import numpy as np
import pyssa.util as ut
from scipy.interpolate import interp1d
import time


class Simulator:
    """
    This class implements the stochastic simulation algorithm for
    a markov jump process. It requires an instance of th MJP or one 
    of the sublcasses. These are used to compute exit rates and transition
    probabilities of the embeded chain.
    """

    def __init__(self, model, initial, time=0.0, mode='homogenous'):
        self.model = model
        self.state = model.state2label(initial)
        self.time = time
        self.mode = mode
        if mode == 'homogenous':
            self.next_event = self.next_event_homogenous
        elif mode == 'non-homogenous':
            self.next_event = self.next_event_nonhomogenous
        else:
            raise ValueError('Unrecognized option ' + mode)

    def next_event_homogenous(self):
        """ 
        simulte next event
        """
        # compute exit rate and target state probabilities
        rate, prob = self.model.exit_stats(self.state)
        # draw time and event index
        if rate == 0.0:
            tau = np.inf
            event = None
        else:
            tau = -np.log(np.random.rand())/rate
            event_ind = ut.sample_discrete(prob)
            event = self.model.get_event(event_ind)
        # update time and state
        self.time += tau
        self.state = self.model.update(self.state, event)
        return(event)

    def next_event_nonhomogenous(self):
        """ 
        simulte next event
        """
        # compute waiting time and new event
        tau, event = self.model.next_event(self.state, self.time)
        # update time and state
        self.time += tau
        self.state = self.model.update(self.state, event)
        return(event)

    def simulate(self, initial, tspan, get_states=False):
        # prepare state for simulation
        self.time = tspan[0]
        self.state = self.model.state2label(initial)
        # initialize output
        times = []
        event_hist = []
        # run until time is above upper limit
        while (self.time < tspan[1]):
            event = self.next_event()
            # store stuff
            times.append(self.time)
            event_hist.append(event)
        # construct output dictionary
        trajectory = {'initial': initial, 'tspan': tspan, 'times': np.array(times), 'events': np.array(event_hist)}
        if get_states:
            self.events2states(trajectory)
        return(trajectory)

    def events2states(self, trajectory):
        """
        Extend a trajecoty dict produced by simulate to contain the states
        """
        # construct output
        if np.isscalar(trajectory['initial']):
            dim = 1
        else:
            dim = len(trajectory['initial'])
        num_steps = len(trajectory['times'])
        trajectory['states'] = []
        # fill states
        state = self.model.state2label(trajectory['initial'])
        for i in range(num_steps):
            state = self.model.update(state, trajectory['events'][i])
            trajectory['states'].append(self.model.label2state(state))
        try:
            trajectory['states'] = np.stack(trajectory['states'], axis=0)
        except:
            pass
        return

            
def simulate(model, initial, tspan, get_states=False):
    """
    Wrapper that allows simulation without explicitly constructing the simulator object
    """
    # set up model
    sim = Simulator(model, initial, tspan[0])
    # run simulation 
    trajectory = sim.simulate(initial, tspan, get_states=get_states)
    return(trajectory)


def discretize_trajectory(trajectory, sample_times, obs_model=None, converter=None):
    """ 
    Discretize a trajectory of a jump process by linear interpolation 
    at the support points given in sample times
    Input
        trajectory: a dict with keys 'initial', 'tspan', 'times', 'states'
        sample_times: np.array containin the sample times
        obs_model: an observation model from which to sample the state
        converter: a converter function to translate the state representation to something that the obs_model understands
    """
    if isinstance(trajectory['states'], np.ndarray) and converter is None:
        initial = np.array(trajectory['initial'])
        if (len(trajectory['times']) == 0):
            times = trajectory['tspan']
            states = np.stack([initial, initial])
        elif (trajectory['times'][-1] < trajectory['tspan'][1]):
            delta = (trajectory['tspan'][1]-trajectory['tspan'][0])/1e-3
            times = np.concatenate([trajectory['tspan'][0:1], trajectory['times'], trajectory['tspan'][1:]+delta])
            states = np.concatenate([initial.reshape(1, -1), trajectory['states'], trajectory['states'][-1:, :]])
        else:
            times = np.concatenate([trajectory['tspan'][0:1], trajectory['times']])
            states = np.concatenate([initial.reshape(1, -1), trajectory['states']])
        sample_states = interp1d(times, states, kind='zero', axis=0)(sample_times)
    elif isinstance(trajectory['states'], list) or converter is not None:
        if converter is None:
            converter = lambda x: x
        state = trajectory['initial'].copy()
        time = trajectory['tspan'][0]
        cnt = 0
        sample_states = []
        for i, t_i in enumerate(trajectory['times']):
            while cnt < len(sample_times) and time <= sample_times[cnt] and t_i > sample_times[cnt]:
                sample_states.append(converter(state))
                cnt += 1
            time = t_i
            state = trajectory['states'][i]
            if cnt == len(sample_times):
                break
        # convert to np array if possible
        # try:
        #     sample_states = np.concatenate(sample_states, axis=0)
        # except ValueError:
        #     pass
    else:
        msg = "Unsupported type {type(trajectory['states'])} for trajectory states"
        raise ValueError(msg)
    # sample observations
    if obs_model is not None:
        obs_dim = (obs_model.sample(sample_states[0], sample_times[0])).size
        obs_states = np.zeros((len(sample_states), obs_dim))
        for i in range(len(sample_times)):
            obs_states[i] = obs_model.sample(sample_states[i], sample_times[i])
        sample_states = obs_states
    return(sample_states)


def sample(model, initial, sample_times, num_samples=1, output='full'):
    """
    Draw num_samples samples from the model for a given initial
    Discretize the trajectories over the grid sample_times
    Store in an num_samples x times x num_speies array
    """
    # set up output
    samples = np.zeros((num_samples, len(sample_times), len(initial)))
    # get tspan
    tspan = np.array([sample_times[0], sample_times[-1]])
    # set up model
    sim = Simulator(model, initial, tspan[0])
    start = time.time()
    for i in range(num_samples):
        # run simulation 
        trajectory = sim.simulate(initial, tspan, get_states=True)
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

def sample_compiled(simulate, sample_times, num_samples=1, output='full'):
    """
    Draw num_samples samples from the model for a given initial
    Discretize the trajectories over the grid sample_times
    Store in an num_samples x times x num_speies array
    """
    start = time.time()
    # get tspan
    tspan = np.array([sample_times[0], sample_times[-1]])
    # run simulation 
    trajectory = simulate(tspan)
    # set up output
    samples = np.zeros((num_samples, len(sample_times), len(trajectory['initial'])))
    # set up model
    for i in range(num_samples):
        # run simulation 
        trajectory = simulate(tspan)
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
    states_obs = discretize_trajectory(trajectory, obs_times)
    for (t_obs, obs, state) in zip(obs_times, obs_data, states_obs):
        llh += obs_model.llh(state, t_obs, obs)
    return(llh)