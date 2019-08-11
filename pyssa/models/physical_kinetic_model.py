# kinetic model subclass that allows only reactions up to second order (pair interactions)

# import kinetic model
import numpy as np
from pyssa.models.KineticModel import KineticModel


class PhysicalKineticModel(KineticModel):

    # constructor
    def __init__(self, pre, post, rates):
        super().__init__(pre, post, rates)
        self.zeroth, self.first, self.second = self.get_reaction_tensor(pre)


    # construct reaction tensors for simplifying propensity computation
    def get_reaction_tensor(self, pre):
        # get order of the individual reactions and number of reactions
        num_reactions, num_species = pre.size()
        reaction_order = np.sum(pre, dim=1)
        # reject if higher order reaction is included
        if (np.max(reaction_order) > 2):
            raise Exception('System should not contain reactions of order larger than two.')
        # set up the outputs
        zeroth = np.zeros(num_reactions, dtype=torch.float64)
        first = np.zeros([num_reactions, num_species], dtype=torch.float64)
        second = np.zeros([num_reactions, num_species, num_species], dtype=torch.float64)
        # iterate over reactions 
        for i in range(num_reactions):
            if reaction_order[i] == 0:
                zeroth[i] == 1
            elif reaction_order[i] == 1:
                first[i, :] = pre[i, :]
            elif reaction_order[i] == 2:
                if torch.max(pre[i, :]) == 2:
                    ind = pre[i, :].nonzero()
                    second[i, ind, ind] = 1
                    first[i, ind] = -1
                else:
                    ind = pre[i, :].nonzero()
                    second[i, ind[0], ind[1]] = 0.5
                    second[i, ind[1], ind[0]] = 0.5
            else:
                raise Exception('Check system matrices.')
        return zeroth, first, second
