# contains a collection of specific models that do not fit into the other frameworks

#imports
import numpy as np
from pyssa.models.mjp import MJP


class TASEP(MJP):
    """
    Implementation of the Tasep model
    change vector stored in the stoichiometry matrix and an arbitrary propensity functor
    """

    # implementation of abstract methods

    def __init__(self, num_sites, rates):
        """
        Class for a basic tasep process. Requires the number of sites and an array of rate constants
        """
        self.num_sites = num_sites
        self.rates = rates
        self.stoichiometry = self.get_stoichiometry()

    def label2state(self, label):
        """
        Map a reaction index to a state change
        """
        return(label)

    def state2label(self, state):
        """
        For a kinetic model, this works on the level of reactions,
        i.e a change vector is mapped to an index
        """
        return(state)

    def exit_stats(self, state):
        """
        Returns the exit rate corresponding to the current state
        and an array containing a probability distribution over target states
        """
        # compute raw mass action propensity
        prop = self.propfun(state)
        prop[0] *= self.rates[0]
        prop[1:-1] *= self.rates[1]
        prop[-1] *= self.rates[2]
        # compute rate and
        rate = prop.sum()
        return(rate, prop/rate)

    def update(self, state, event):
        """
        Update the state using the current reaction index
        """
        new_state = state+self.stoichiometry[event]
        return(new_state)

    # additional functions

    def get_stoichiometry(self):
        """
        Compute stoichiometry matrix for basic tasep model
        """
        # construct matrix
        stoichiometry = np.zeros((self.num_sites+1, self.num_sites))
        # set increase
        ind = [i for i in range(self.num_sites)]
        stoichiometry[ind, ind] = 1
        # set decrease
        row = [i for i in range(1, self.num_sites+1)]
        stoichiometry[row, ind] = -1
        return(stoichiometry)

    def propfun(self, state):
        """ 
        reaction i is possible whenever site i is empty and site i-1 is occupied
        """
        prop = (1-np.concatenate([state, np.array([0])]))*np.concatenate([np.array([1]), state])
        return(prop)


# class SparseTASEP(MJP):

#     def __init__(self, num_sites, rates):
#         """
#         Class for a basic tasep process. Requires the number of sites and an array of rate constants
#         This version expects the state to be a list of occupied indices
#         """
#         self.num_sites = num_sites
#         self.rates = rates

#     def label2state(self, label):
#         """
#         Map sparse representation to vector
#         """
#         state = np.zeros(self.num_sites)
#         state[label] = 1
#         return(label)

#     def state2label(self, state):
#         """
#         For a kinetic model, this works on the level of reactions,
#         i.e a change vector is mapped to an index
#         """
#         label = state.nonzero()[0]
#         return(state)

#     def exit_stats(self, state):
#         """
#         Returns the exit rate corresponding to the current state
#         and an array containing a probability distribution over target states
#         """
#         # get the possilbe events
#         events = np.concatenate([np.array([0]), state+1])
#         rate = prop.sum()
#         return(rate, prop/rate)

#     def update(self, state, event):
#         """
#         Update the state using the current reaction index
#         """
#         new_state = state+self.stoichiometry[event]
#         return(new_state)
    