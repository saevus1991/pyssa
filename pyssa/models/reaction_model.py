# kinetic model class storing the pre, post and adjecency matrix of a kinetic model

#imports
import numpy as np
from pyssa.models.mjp import MJP


class ReactionModel(MJP):
    """
    Implementation of a reaction model defined by a number of
    change vector stored in the stoichiometry matrix and an arbitrary propensity functor
    """

    # implementation of abstract methods

    def __init__(self, stoichiometry, propfun, rates):
        """
        The class requires a stoichometry matrix of dimension (num_reactions x num_species)
        a propensity function R^num_species -> R^num_reactions and an an array of length 
        num_reactions specifying the time scale of each reaction
        """
        self.num_reactions, self.num_species = stoichiometry.shape
        self.rates = rates
        self.stoichiometry = stoichiometry
        self.propfun = propfun

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
        prop = self.propfun(state)*self.rates
        # compute rate and
        rate = prop.sum()
        return(rate, prop/rate)

    def update(self, state, event):
        """
        Update the state using the current reaction index
        """
        new_state = state+self.stoichiometry[event]
        return(new_state)


class BaseReactionModel(MJP):
    """
    Implementation of a reaction model defined by a number of
    change vector stored in the stoichiometry matrix and an arbitrary propensity functor
    """

    # implementation of abstract methods

    def __init__(self, stoichiometry, rates):
        """
        The class requires a stoichometry matrix of dimension (num_reactions x num_species)
        a propensity function R^num_species -> R^num_reactions and an an array of length 
        num_reactions specifying the time scale of each reaction
        """
        self.num_reactions, self.num_species = stoichiometry.shape
        self.rates = rates
        self.stoichiometry = stoichiometry

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
        prop = self.propfun(state)*self.rates
        # compute rate and
        rate = prop.sum()
        return(rate, prop/rate)

    def update(self, state, event):
        """
        Update the state using the current reaction index
        """
        new_state = state+self.stoichiometry[event]
        return(new_state)

    def propfun(self, state):
        raise NotImplementedError