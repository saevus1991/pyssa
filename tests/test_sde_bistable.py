# test the sde simulator with a few models

import numpy as np
import matplotlib.pyplot as plt
from pyssa.sde import Simulator
from pyssa.models.diffusion import Diffusion

# create a custom model as a descendant of Diffusion
class Bistable(Diffusion):

    def __init__(self, theta, sigma):
        self.theta = theta 
        self.sigma = sigma

    def eval(self, state, time):
        drift = 4.0*state*(self.theta-state**2)
        return(drift, self.sigma)


# initialize model
theta = np.array([1.0])
sigma = np.array([0.5])
model = Bistable(theta, sigma)

# construct sde engine
timestep = 1e-2
simulator = Simulator(model, timestep)

# perform simulation
initial = np.array([1.0])
tspan = np.array([0.0, 1000.0])
trajectory = simulator.simulate(initial, tspan)

# plot
t_plot = trajectory['times']
states_plot = trajectory['states']


plt.plot(t_plot, states_plot, '-r')
plt.show()