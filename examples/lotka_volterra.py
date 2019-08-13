# Demonstrate simulate on synthetic gene expression model
# different implementations are used and compared

# import stuff
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import time

# custom files
from pyssa.models.kinetic_model import KineticModel 
from pyssa.models.kinetic_model import PhysicalKineticModel
from pyssa.models.kinetic_model import SparseKineticModel
from pyssa.models.kinetic_model import kinetic_to_generator
from pyssa.models.ctmc import CTMC
from pyssa.models.ctmc import SparseCTMC
import pyssa.models.standard_models as sm
import pyssa.ssa as ssa

# (de)activate plotting
plotting = True

# set up the model
pre, post, rates = sm.get_standard_model("predator_prey")
model_kinetic = KineticModel(np.array(pre), np.array(post), np.array(rates))
model_physical = PhysicalKineticModel(np.array(pre), np.array(post), np.array(rates))
model_sparse = SparseKineticModel(np.array(pre), np.array(post), np.array(rates))

# construct an additional ctmc approximation via state space truncation
bounds = np.array([50, 50])
exit_rates, embedded, keymap = kinetic_to_generator(model_kinetic, bounds)

#print(trunc_generator)
#print(type(trunc_generator[0].shape))
# get generator matrix
rate_mat = embedded.toarray()*exit_rates.reshape(-1, 1)
generator = rate_mat -exit_rates
#print(np.sum(generator, axis=1))
model_ctmc = CTMC((exit_rates, embedded.toarray()), keymap, form='embedded')
model_sparse_ctmc = SparseCTMC((exit_rates, embedded), keymap, form='embedded')

# prepare initial conditions
initial = np.array([25.0, 5.0])
tspan = np.array([0.0, 3e3])
delta_t = 300.0
obs_times = np.arange(tspan[0]+0.5*delta_t, tspan[1], delta_t)

# set up simulator
iter = 1000
simulator = ssa.Simulator(model_kinetic, initial)
start = time.time()
for i in range(iter):
    simulator.simulate(initial, tspan)
end = time.time()
print('Kinetic model required '+str(end-start)+' seconds.')
simulator = ssa.Simulator(model_physical, initial)
start = time.time()
for i in range(iter):
    simulator.simulate(initial, tspan)
end = time.time()
print('Physical kinetic model required '+str(end-start)+' seconds.')
simulator = ssa.Simulator(model_sparse, initial)
start = time.time()
for i in range(iter):
    simulator.simulate(initial, tspan)
end = time.time()
print('Sparse kinetic model required '+str(end-start)+' seconds.')
simulator = ssa.Simulator(model_ctmc, initial)
start = time.time()
for i in range(iter):
    simulator.simulate(initial, tspan)
end = time.time()
print('CTMC approximation required '+str(end-start)+' seconds.')
simulator = ssa.Simulator(model_sparse_ctmc, initial)
start = time.time()
for i in range(iter):
    simulator.simulate(initial, tspan)
end = time.time()
print('Sparse CTMC approximation required '+str(end-start)+' seconds.')

# get trajectory 
simulator = ssa.Simulator(model_sparse_ctmc, initial)
trajectory = simulator.simulate(initial, tspan)
# simulator = ssa.Simulator(model_sparse, initial)
# trajectory = simulator.simulate(initial, tspan)
simulator.events2states(trajectory)

#print(trajectory)

# get a subsampling for plotting
t_plot = np.linspace(tspan[0], tspan[1], 200)
states_plot = ssa.discretize_trajectory(trajectory, t_plot)

# plot result 
if plotting:
    plt.plot(t_plot, states_plot[:, 0], '-b')
    plt.plot(t_plot, states_plot[:, 1], '-r')
    #plt.plot(obs_times.numpy(), obs.numpy(), 'xk')
    plt.show()
