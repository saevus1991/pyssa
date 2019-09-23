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
            diffusion = tmp[0] @ np.diag(tmp[1])
        # return drift and diffusion
        return(drift, diffusion)