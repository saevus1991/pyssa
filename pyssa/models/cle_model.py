# chemical langevin approximation for general kinetic model
# as the kinetic_model, the pre, post and adjecency matrix are used to compute the required quantities

#imports
import numpy as np
from pyssa.models.kinetic_model import KineticModel
from pyssa.models.diffusion import Diffusion
#import pyssa.util as ut
#from scipy import sparse


class CLEModel(KineticModel, Diffusion):
    """
    general chemical Langevin model
    """

    def eval(self, state, time):
        # compute the propensity
        prop = self.mass_action(state)*self.rates
        # evaluate drift
        drift = self.stoichiometry.T @ prop
        # evaluate diffusion
        diffusion = self.stoichiometry.T @ np.diag(prop) @ self.stoichiometry
        try:
            diffusion = np.linalg.cholesky(diffusion)
        except:
            tmp = np.linalg.svd(diffusion)
            diffusion = tmp[0] @ np.diag(np.sqrt(tmp[1]))
        # return drift and diffusion
        return(drift, diffusion)

    def get_dimension(self):
        dim = (self.num_species, self.num_species)
        return(dim)


class CLEModelExtended(KineticModel, Diffusion):
    """
    general chemical Langevin model
    uses a higher order driving wiender process. Thus, we do not have to compute a root of the diffusion tensor
    """

    def eval(self, state, time):
        # compute the propensity
        prop = self.mass_action(state)*self.rates
        # evaluate drift
        drift = self.stoichiometry.T @ prop
        # evaluate diffusion
        diffusion = self.stoichiometry.T @ np.diag(np.sqrt(prop)) 
        # return drift and diffusion
        return(drift, diffusion)

    def get_dimension(self):
        dim = (self.num_species, self.num_reactions)
        return(dim)


class CLEModelLV(KineticModel, Diffusion):
    """
    chemical Langevin model
    uses a higher order driving wiender process. Thus, we do not have to compute a root of the diffusion tensor
    uses a large volume assumption for evaluating the propensities
    """

    def eval(self, state, time):
        # compute the propensity
        prop = np.prod(state**self.pre, axis=1)*self.rates
        # evaluate drift
        drift = self.stoichiometry.T @ prop
        # evaluate diffusion
        diffusion = self.stoichiometry.T @ np.diag(np.sqrt(prop)) 
        # return drift and diffusion
        return(drift, diffusion)

    def get_dimension(self):
        dim = (self.num_species, self.num_reactions)
        return(dim)