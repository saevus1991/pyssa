# kinetic model class storing the pre, post and adjecency matrix of a kinetic model

#imports
import numpy as np
from pyssa.models.mjp import MJP
import pyssa.util as ut
from scipy import sparse


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
        return(label)

    def state2label(self, state):
        """
        For a kinetic model, state and label
        """
        return(state)

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

    def update(self, state, event):
        """
        Update the state using the current reaction index
        """
        new_state = state+self.stoichiometry[event]
        return(new_state)

    # additional functions

    # def event2change(self, label):
    #     """
    #     For a kinetic model, this works on the level of reactions,
    #     i.e a reaction index is mapped to a state change
    #     """
    #     return(self.stoichiometry[label, :])

    def mass_action(self, state):
        """
        Compute the mass-action propensity
        """
        # initialize with ones
        prop = np.ones(self.num_reactions)
        # iterate over reactions
        for i in range(self.num_reactions):
            for j in range(self.num_species):
                prop[i] *= ut.falling_factorial(state[j], self.pre[i, j])
        return(prop)


class PhysicalKineticModel(KineticModel):
    """
    Implementation of mass acton model with reactions up to order two.
    Uses tensor math to compute propensity
    """

    # constructor
    def __init__(self, pre, post, rates):
        super().__init__(pre, post, rates)
        self.second_order = None
        self.zeroth, self.first, self.second = self.get_reaction_tensor(pre)


    # reimplemented parent functions

    def exit_stats(self, state):
        """
        Returns the exit rate corresponding to the current state
        and an array containing a probability distribution over target states
        """
        # zeroth oder contribution
        prop = self.zeroth.copy()
        # first oder contribution
        tmp = np.dot(self.first, state.reshape(-1, 1))
        prop += tmp.squeeze()
        # second order contribution
        if self.second_order:
            tmp = np.einsum('i,j,aij->a', state, state, self.second)
            prop += tmp
        # multiply with rates
        prop *= self.rates
        # normalize
        rate = np.sum(prop)
        return(rate, prop/rate)

    # helper functions

    def get_reaction_tensor(self, pre):
        # get order of the individual reactions 
        reaction_order = np.sum(pre, axis=1)
        if np.any(reaction_order == 2):
            self.second_order = True
        # reject if higher order reaction is included
        if (np.max(reaction_order) > 2):
            raise Exception('System should not contain reactions of order larger than two.')
        # set up the outputs
        zeroth = np.zeros(self.num_reactions)
        first = np.zeros([self.num_reactions, self.num_species])
        second = np.zeros([self.num_reactions, self.num_species, self.num_species])
        # iterate over reactions 
        for i in range(self.num_reactions):
            if reaction_order[i] == 0:
                zeroth[i] == 1
            elif reaction_order[i] == 1:
                first[i, :] = pre[i, :]
            elif reaction_order[i] == 2:
                if np.max(pre[i, :]) == 2:
                    ind = pre[i, :].nonzero()
                    second[i, ind[0][0], ind[0][0]] = 1
                    first[i, ind[0][0]] = -1
                else:
                    ind = pre[i, :].nonzero()
                    second[i, ind[0][0], ind[0][1]] = 0.5
                    second[i, ind[0][1], ind[0][0]] = 0.5
            else:
                raise Exception('Check system matrices.')
        return zeroth, first, second


class SparseKineticModel(PhysicalKineticModel):
    """
    Implementation of mass acton model with reactions up to order two.
    Similar to PysicalKinetic model but uses a sparse tensor implementation 
    to compute the probensities
    """

    # constructor

    def __init__(self, pre, post, rates):
        super().__init__(pre, post, rates)
        self.sparsify_tensors()


    # reimplemented parent functions

    def exit_stats(self, state):
        """
        Returns the exit rate corresponding to the current state
        and an array containing a probability distribution over target states
        """
        # zeroth oder contribution
        prop = self.zeroth.copy()
        # first oder contribution
        tmp = self.first.dot(state)
        prop += tmp.flatten()
        # second order contribution
        if self.second_order:
            extended_state = np.kron(state, state)
            tmp = self.second.dot(extended_state)
            prop += tmp.flatten()
        # multiply with rates
        prop *= self.rates
        # normalize
        rate = np.sum(prop)
        return(rate, prop/rate)

    
    # helper functions

    def sparsify_tensors(self):
        """
        This functions converts the implementation of a kinetic model by transforming
        the matrices used for computing the propensity to sparse arrays
        """
        # convert first order matrix to sparse
        self.first = sparse.csr_matrix(self.first)
        # convert second order tensor to matrix and then to sparse
        tmp = self.second.reshape(self.num_reactions, -1)
        self.second = sparse.csr_matrix(tmp)


def kinetic_to_generator(kinetic_model, bounds):
    """
    Convert a kinetic model to a rate matrix using state space truncation
    """
    # construct label to state dictionary
    keymap = {}
    num_states = np.prod(bounds)
    for i in range(num_states):
        state = np.array(np.unravel_index(i, bounds))
        keymap[i] = state
    # construct the reverse dict 
    reversemap = {value.tobytes(): key for (key, value) in keymap.items()}
    # find the nonzero elements of generator
    row_ind = []
    col_ind = []
    val = []
    rates = np.zeros(num_states)
    for i in range(num_states):
        # get current state
        state = keymap[i]
        # evalute propensity
        rate, props = kinetic_model.exit_stats(state)
        raw_props = rate*props
        props = raw_props.copy()
        # get target states and store transition element
        for ind, prop in enumerate(raw_props):
            if prop > 0.0:
                target = kinetic_model.update(state, ind)
                try:
                    label = reversemap[target.tobytes()]
                    row_ind.append(i)
                    col_ind.append(label)
                except KeyError:
                    props[ind] = 0.0
                    pass
                    #print('Transition rate '+str(state)+'->'+str(target)+' set to zero.')
        rates[i] = props.sum()
        eff_props = props[props > 0.0]
        if len(eff_props) > 0:
            val += list(props[props > 0.0]/props.sum())
        else:
            row_ind.append(i)
            col_ind.append(i)
            val.append(1.0)
    # construct sparse generator
    transition = sparse.csr_matrix((val, (row_ind, col_ind)), shape=(num_states, num_states))
    return(rates, transition, keymap)

