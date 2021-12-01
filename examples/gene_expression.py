import numpy as np
import torch
import matplotlib.pyplot as plt
# from simulation.utils import discretize_trajectory
from pyssa.models.kinetic_model import KineticModel
from pyssa.models.cle_model import RREModel
from pyssa import ssa
import pyssa.util as ut
from scipy.integrate import solve_ivp


# fix random seed
np.random.seed(2009181718)

# create model
model = RREModel(name='Two stage gene expression')
model.add_species('G0')
model.add_species('G1')
model.add_species('mRNA')
model.add_species('Protein')
model.add_reaction('Activation', '1 G0 -> 1 G1', 0.001)
model.add_reaction('Deactivation', '1 G1 -> 1 G0', 0.001)
model.add_reaction('Transcription', '1 G1 -> 1 G1 + 1 mRNA', 0.15)
model.add_reaction('mRNA Decay', '1 mRNA -> 0 mRNA', 0.001)
model.add_reaction('Translation', '1 mRNA -> 1 mRNA + 1 Protein', 0.04)
model.add_reaction('Protein Decay', '1 Protein -> 0 Protein', 0.008)
model.build()


# input for the function
initial = np.array([1.0, 0.0, 0.0, 0.0])
tspan = np.array([0.0, 5000.0])

# simulate a trajectory (including all jumps)
seed = np.random.randint(2**16)
simulator = ssa.Simulator(model, initial, tspan)
trajectory = simulator.simulate(initial, tspan, get_states=True)

# get a subsampling for plotting
t_plot = np.linspace(tspan[0], tspan[1], 200)
states_plot = ut.discretize_trajectory(trajectory, t_plot)

# get ode simulation
states_rre = solve_ivp(lambda t, x: model.eval(x, t), tspan, initial, t_eval=t_plot)['y'].T

# plot simulated trajetory
plt.plot(t_plot, 100 * states_plot[:, 1], '--k')
plt.plot(t_plot, 100*states_rre[:, 1], '-k')
plt.plot(t_plot, states_plot[:, 2], '--b')
plt.plot(t_plot, states_rre[:, 2], '-b')
plt.plot(t_plot, states_plot[:, 3], '--r')
plt.plot(t_plot, states_rre[:, 3], '-r')
plt.ylim(0, 800)
plt.show()