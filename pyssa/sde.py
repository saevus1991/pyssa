# class for sde simulation

import numpy as np


class Simulator:
    """
    This class implements a Euler-Maruyama discretization for SDEs
    """

    def __init__(self, model, timestep, projector=None):
        self.model = model
        self.timestep = timestep
        self.projector = self.set_projector(projector)

    def simulate(self, initial, tspan):
        # initialize
        times = np.arange(tspan[0], tspan[1], self.timestep)
        states = np.zeros((len(times), len(initial)))
        states[0] = initial
        noise_std = np.sqrt(self.timestep)
        # iterate
        for i in range(len(times)-1):
            drift, diffusion = self.model.eval(states[i], times[i])
            noise = np.random.standard_normal(initial.shape)*noise_std
            states[i+1] = states[i] + drift*self.timestep + diffusion@noise
            states[i+1] = self.projector(states[i+1])
        # produce output dictionary
        trajectory = {'initial': initial, 'tspan': tspan, 'times': times, 'states': states}
        return(trajectory)

    def set_projector(self, projector):
        if projector is None:
            def id(x): return(x)
            return(id)
        else:
            return(projector)