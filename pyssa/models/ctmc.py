# finite state ctmc class

#imports
import numpy as np
from pyssa.models.mjp import MJP
import pyssa.util as ut
import scipy.sparse

class CTMC(MJP):
    """
    Implementation of a ctmc model defined by a generator matrix
    """

    # implemented abstract methods

    def __init__(self, generator, keymap=None, form='full'):
        """
        The class requires a valid (n x n) generator matrix.
        However, the generator is stored as a n-tensor of waiting times
        and an (n x n) matrix of the embedded chain
        """
        self.exit_rates, self.embedded = self.get_transition(generator, form)
        self.num_states = len(self.exit_rates)
        self.keymap = keymap
        self.reversemap = self.get_reversemap()

    def label2state(self, label):
        """
        Map the integer label to the underlying system state
        """
        if self.keymap is None:
            return(label)
        else:
            return(self.keymap[label])

    def state2label(self, state):
        """
        Map system state to the underlying integer label
        """
        if self.keymap is None:
            return(state)
        else:
            return(self.reversemap[state.astype('int64').tobytes()])

    def exit_stats(self, state):
        """
        Returns the exit rate corresponding to the current state
        and an array containing a probability distribution over target states
        """
        # compute raw mass action propensity
        prop = self.embedded[state]
        # compute rate and
        rate = self.exit_rates[state]
        return(rate, prop)

    def update(self, state, event):
        """
        For a ctmc the event index corresponds to the new state
        """
        return(event)

    # additional functions

    def get_transition(self, generator, form):
        """ Construct an exit rate vector and the embedded transition matrix 
        from generator Generator can be 
        - a (num_states x num_states) transition matrix (form = 'full')
        - a (num_states x num_states) rate matrix with diagonals zero
        - a tuple of size 2 containing an exit rate vector and the embedded matrix
        """
        if form == 'full':
            """
            assumes generator is a csr_matrix
            """
            # check that generator matches expected form
            ut.assert_stochmat(generator)
            # get exit rates
            exit_rates = -np.diagonal(generator)
            # set up matrix
            transition = np.copy(generator)
            np.fill_diagonal(transition, 0.0)
            # normalize by rates
            ind = exit_rates > 0.0
            transition[ind, :] /= exit_rates[ind].reshape(-1, 1)
        elif form == 'rates':
            # check generator
            ut.assert_ratemat(generator)
            # get exit rates
            exit_rates = np.sum(generator, axis=1)
            # normalize by rates
            transition = np.copy(generator)/exit_rates.reshape(-1, 1)
        elif form == 'embedded':
            # check stuff
            assert(type(generator) == tuple)
            assert(len(generator) == 2)
            assert(np.all(generator[0] >= 0.0))
            ut.assert_transmat(generator[1])
            # copy rates and embeded transition matrix
            exit_rates = generator[0].copy()
            transition = generator[1].copy()
        else:
            ValueError('Unkown form argument '+form)
        return(exit_rates, transition)

    def get_reversemap(self):
        if self.keymap is None:
            return None
        else:
            reversemap = {value.tobytes(): key for (key, value) in self.keymap.items()}
            return(reversemap)


class SparseCTMC(CTMC):
    """
    Sparse version of ctmc class where the embedded chain transition
    matrix is stored as a sparse csr_matrix
    """

    def exit_stats(self, state):
        """
        Returns the exit rate corresponding to the current state
        and an array containing a probability distribution over target states
        """
        # compute raw mass action propensity
        prop = self.embedded[state][1]
        # compute rate and
        rate = self.exit_rates[state]
        return(rate, prop)

    def update(self, state, event):
        """
        For a sparse ctmc the event index corresponds to the corresponding state with non-zero transition
        """
        return(self.embedded[state][0][event])

    def get_transition(self, generator, form):
        """ Construct an exit rate vector and the embedded transition matrix 
        from generator Generator can be 
        - a (num_states x num_states) rate matrix with diagonals zero
        - a tuple of size 2 containing an exit rate vector and the embedded matrix
        """
        if form == 'full':
            # check that generator matches expected form
            #ut.assert_stochmat(generator)
            # get exit rates
            exit_rates = generator.diagonal()
            # set up matrix
            transition = []
            for row in generator:
                ind = row.nonzero()[1]
                val = row.data/row.data.sum()
                transition.append([ind, val])
        elif form == 'rates':
            # check generator
            #ut.assert_ratemat(generator)
            # get exit rates
            exit_rates = generator.sum(axis=1).A1
            # set up matrix
            transition = []
            for row in generator:
                ind = row.nonzero()[1]
                val = row.data/row.data.sum()
                transition.append([ind, val])
        elif form == 'embedded':
            # check stuff
            #assert(type(generator) == tuple)
            #assert(len(generator) == 2)
            #assert(np.all(generator[0] >= 0.0))
            #ut.assert_transmat(generator[1])
            # copy rates and embeded transition matrix
            exit_rates = generator[0].copy()
            transition = generator[1].copy()
            transition = []
            for row in generator[1]:
                ind = row.nonzero()[1]
                val = row.data
                transition.append([ind, val])
        else:
            ValueError('Unkown form argument '+form)
        return(exit_rates, transition)
