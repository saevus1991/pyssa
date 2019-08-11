# kinetic model class storing the pre, post and adjecency matrix of a kinetic model

#imports
import numpy as np
from pyssa.models.mjp import MJP


class KineticModel(MJP):
    """
    Implementation of a reaction model defined by a number of
    change vector stored in the stoichiometry matrix and an arbitrary propensity functor
    """

    # implemented abstract methods

    def __init__(self, stoichiometry, propensity, rates)
        """
        The class requires 2 matrices of shape (num_reactions x num_species)
        specifying the populations before and after each reaction as well
        as an array of length num_reactions specifying the time scale of each reaction
        """
        self.num_reactions, self.num_species = stoichiometry.shape
        self.rates = rates
        self.stoichiometry = stoichiometry
        self.propensity = propensity

    def label2state(self, label):
        """
        Map a reaction index to a state change
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
        prop = self.propensity.eval(state, self.rates)
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

