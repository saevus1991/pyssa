# test the sde simulator with a few models

import numpy as np
import matplotlib.pyplot as plt
from pyssa.sde import Simulator
from pyssa.models.diffusion import Diffusion

# create a custom model as a descendant of Diffusion
class GBM(Diffusion):

    def __init__(self, gamma, sigma):
        self.gamma = gamma 
        self.sigma = sigma 

    def eval(self, state, time):
        return(self.gamma*state, self.sigma*state)


# initialize model
gamma = np.array([0.2])
sigma = np.array([0.1])
model = GBM(gamma, sigma)

# construct sde engine
timestep = 1e-2
simulator = Simulator(model, timestep)

# perform simulation
initial = np.array([5.0])
tspan = np.array([0.0, 10.0])
trajectory = simulator.simulate(initial, tspan)

# create observations
obs_times = np.arange(1.0, 10.0, 2.0)

# plot
t_plot = trajectory['times']
states_plot = trajectory['states']


plt.plot(t_plot, states_plot, '-r')
plt.show()