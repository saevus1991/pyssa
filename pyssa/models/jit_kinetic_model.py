# kinetic model class storing the pre, post and adjecency matrix of a kinetic model

#imports
import numpy as np
from pyssa.models.mjp import MJP
import pyssa.util as ut
from scipy import sparse
from numba import jitclass
from numba import int32, int64, float64

spec = [
    ('num_reactions', int32),
    ('num_species', int32),
    ('pre', float64[:, :]),
    ('post', float64[:, :]),
    ('stoichiometry', float64[:, :]),
    ('rates', float64[:]),          # an array field
]

@jitclass(spec)
class KineticModel:
    """
    Implementation of a basic mass-action gillespie model 
    """

    # implemented abstract methods

    def __init__(self, pre, post, rates):
        """
        The class requires 2 matrices of shape (num_reactions x num_species)
        specifying the populations before and after each reaction as well
        as an array of length num_reactions specifying the time scale of each reaction
        """
        self.num_reactions, self.num_species = pre.shape
        self.pre = pre
        self.post = post
        self.rates = rates
        self.stoichiometry = post-pre

    def simulate(self, initial, tspan):
        # prepare state for simulation
        time = tspan[0]
        state = initial.copy()
        # initialize output
        times = []
        event_hist = []
        # run until time is above upper limit
        while (time < tspan[1]):
            # perform update
            tau, event = self.next_event(state)
            time += tau
            state += self.stoichiometry[event, :]
            # store stuff
            times.append(time)
            event_hist.append(event)
        # construct output dictionary
        trajectory = {'initial': initial.copy(), 'tspan': tspan.copy(), 'times': np.array(times), 'events': np.array(event_hist)}
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
        state = trajectory['initial'].copy()
        for i in range(num_steps):
            event = trajectory['events'][i]
            state += self.stoichiometry[event]
            trajectory['states'][i, :] = state

    def next_event(self, state):
        """ 
        simulte next event
        """
        # compute exit rate and target state probabilities
        rate, prob = self.exit_stats(state)
        # draw time and event index
        if rate == 0.0:
            tau = np.inf
            event = None
        else:
            tau = -np.log(np.random.rand())/rate
            event = self.sample_discrete(prob)
        return(tau, event)

    def exit_stats(self, state):
        """
        Returns the exit rate corresponding to the current state
        and an array containing a probability distribution over target states
        """
        # compute raw mass action propensity
        prop = self.mass_action(state)*self.rates
        # compute rate and
        rate = prop.sum()
        # catch for absorbing states
        if rate == 0.0:
            transition = np.zeros(prop.shape)
        else:
            transition = prop/rate
        return(rate, transition)

    def mass_action(self, state):
        """
        Compute the mass-action propensity
        """
        # initialize with ones
        prop = np.ones(self.num_reactions)
        # iterate over reactions
        for i in range(self.num_reactions):
            for j in range(self.num_species):
                prop[i] *= self.falling_factorial(state[j], self.pre[i, j])
        return(prop)

    def sample_discrete(self, prob):
        # cumulative dist
        csum = np.cumsum(prob)
        # sample uniform
        u = np.random.random()
        # find entry
        sample = np.searchsorted(csum, u)
        return(sample.astype('int32'))
    
    def falling_factorial(self, n, k):
        if (n < k):
            return(0)
        else:
            res = 1.0
            for i in range(k):
                res *= n-i
            return(res)    
