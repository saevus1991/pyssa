# kinetic model class storing the pre, post and adjecency matrix of a kinetic model

#imports
import numpy as np
from pyssa.models.mjp import MJP


def falling_factorial(n, k):
    if (n < k):
        return(0)
    else:
        res = 1.0
        for i in range(k):
            res *= n-i
        return(res)


class KineticModel(MJP):
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

    def label2state(self, label):
        """
        For a kinetic model, this works on the level of reactions,
        i.e a reaction index is mapped to a state change
        """
        return(self.stoichiometry[label, :])

    def state2label(self, state):
        """
        For a kinetic model, this works on the level of reactions,
        i.e a change vector is mapped to an index
        """
        raise NotImplementedError

    def exit_stats(self, state):
        """
        Returns the exit rate corresponding to the current state
        and an array containing a probability distribution over target states
        """
        # compute raw mass action propensity
        prop = self.mass_action(state)*self.rates
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

    def mass_action(self, state):
        """
        Compute the mass-action propensity
        """
        # initialize with ones
        prop = np.ones(self.num_reactions)
        # iterate over reactions
        for i in range(self.num_reactions):
            for j in range(self.num_species):
                prop[i] *= falling_factorial(state[j], self.pre[i, j])
        return(prop)


