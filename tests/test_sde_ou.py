# test the sde simulator with a few models

import numpy as np
import matplotlib.pyplot as plt
from pyssa.sde import Simulator
from pyssa.models.diffusion import Diffusion

# create a custom model as a descendant of Diffusion
class OU_Process(Diffusion):

    def __init__(self, gamma, sigma, mu=0.0):
        self.gamma = gamma 
        self.sigma = sigma 
        self.mu = mu

    def eval(self, state, time):
        return(-self.gamma @ (state-mu), self.sigma)


# initialize model
gamma = np.array([0.5])
sigma = np.array([0.3])
mu = np.array([2.0])
model = OU_Process(gamma, sigma)

# construct sde engine
timestep = 1e-2
simulator = Simulator(model, timestep)

# perform simulation
initial = np.array([5.0])
tspan = np.array([0.0, 10.0])
trajectory = simulator.simulate(initial, tspan)

# plot
t_plot = trajectory['times']
states_plot = trajectory['states']


plt.plot(t_plot, states_plot, '-r')
plt.show()