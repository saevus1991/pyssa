# test the sde simulator with chemical langevin approximation of the lotka volterra model

import numpy as np
import matplotlib.pyplot as plt
from pyssa.sde import Simulator
from pyssa.models.cle_model import CLEModel
import pyssa.models.standard_models as sm

# set up the model
pre, post, rates = sm.get_standard_model("predator_prey")
rates = np.array([0.5, 0.0025, 0.0025, 0.3]) 
model = CLEModel(np.array(pre), np.array(post), np.array(rates))

# construct projector
def reflect(state):
    new_state = np.abs(state)
    return(new_state)

def zero_out(state):
    new_state = state.copy()
    new_state[new_state < 0] = 0.0
    return(new_state)

# construct sde engine
timestep = 1e-1
simulator = Simulator(model, timestep, zero_out)   

# perform simulation
initial = np.array([71.0, 79.0])
tspan = np.array([0.0, 100])
trajectory = simulator.simulate(initial, tspan)

# # create observations
# obs_times = np.arange(1.0, 10.0, 2.0)

# plot
t_plot = trajectory['times']
states_plot = trajectory['states']


plt.plot(t_plot, states_plot[:, 0], '-r')
plt.plot(t_plot, states_plot[:, 1], '-b')
plt.show()