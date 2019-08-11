# ssa class

#import
import numpy as np
import pyssa.util as ut
from scipy.interpolate import interp1d


class Simulator:
    """
    This class implements the stochastic simulation algorithm for
    a markov jump process. It requires an instance of th MJP or one 
    of the sublcasses. These are used to compute exit rates and transition
    probabilities of the embeded chain.
    """

    def __init__(self, model, initial, time=0.0):
        self.model = model
        self.state = initial
        self.time = time

    def next_event(self):
        """ 
        simulte next event
        """
        # compute exit rate and target state probabilities
        rate, prob = self.model.exit_stats(self.state)
        # draw time and event index
        tau = -np.log(np.random.rand())/rate
        event = ut.sample_discrete(prob)
        # update time and state
        self.time += tau
        self.state = self.model.update(self.state, event)
        return(event)

    def simulate(self, initial, tspan, get_states=False):
        # prepare state for simulation
        self.time = tspan[0]
        self.initial = initial
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
        dim = len(trajectory['initial'])
        num_steps = len(trajectory['times'])
        trajectory['states'] = np.zeros([num_steps, dim])
        # fill states
        state = trajectory['initial']
        for i in range(num_steps):
            state = self.model.update(state, trajectory['events'][i])
            trajectory['states'][i, :] = state


def discretize_trajectory(trajectory, sample_times):
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
    return(sample_states)