# Demonstrate simulate on synthetic gene expression model
# different implementations are used and compared

# import stuff
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# custom files
from pyssa.models.reaction_model import ReactionModel 
from pyssa.models.reaction_model import BaseReactionModel 
from pyssa.models.jit_kinetic_model import KineticModel as jitKinetic
import pyssa.models.standard_models as sm
import pyssa.ssa as ssa
from numba import jit, jitclass, int32, float64
import time

# activate or deactivate plotting
plotting = True

# propensity function
#@jit(nopython=True)
def propfun(state):
    prop = np.zeros(6)
    # gene on
    prop[0] = state[0]
    # gene off
    prop[1] = state[1]
    # transcription
    prop[2] = state[1]
    # mrna degration
    prop[3] = state[2]
    # translation
    prop[4] = state[2]
    # protein degradation
    prop[5] = state[3]
    return(prop)

# state = np.array([0, 1, 53, 436])
# propfun(state)
# start = time.time()
# for i in range(100000):
#     test = propfun(state)
#     state[3] += 1
# end = time.time()
# print(end-start)

# drived class with jit support
spec = [
    ('num_reactions', int32),               # a simple scalar field
    ('num_species', int32), 
    ('rates', float64[:]),          # an array field
    ('stoichiometry', float64[:, :])
]

@jitclass(spec)
class GeneExpression(BaseReactionModel):

    def propfun(self, state):
        prop = np.zeros(6)
        # gene on
        prop[0] = state[0]
        # gene off
        prop[1] = state[1]
        # transcription
        prop[2] = state[1]
        # mrna degration
        prop[3] = state[2]
        # translation
        prop[4] = state[2]
        # protein degradation
        prop[5] = state[3]
        return(prop)


# set up the model
pre, post, rates = sm.get_standard_model("simple_gene_expression")
model = ReactionModel(np.array(post)-np.array(pre), propfun, np.array(rates))
stoichiometry = (np.array(post)-np.array(pre)).astype('float64')
model2 = GeneExpression(stoichiometry, np.array(rates))

model3 = jitKinetic(np.array(pre, dtype='float64'), np.array(post, dtype='float64'), np.array(rates, dtype='float64'))

# prepare initial conditions
initial = np.array([0.0, 1.0, 0.0, 0.0])
tspan = np.array([0.0, 3e3])
delta_t = 300.0
obs_times = np.arange(tspan[0]+0.5*delta_t, tspan[1], delta_t)

# set up simulator
simulator = ssa.Simulator(model, initial)

# get trajectory 
#trajectory = simulator.simulate(initial, tspan)
#simulator.events2states(trajectory)
start = time.time()
N = 10
for i in range(N):
    trajectory = model3.simulate(initial.astype('float64'), tspan.astype('float64'))
stop = time.time()
print('Generated {0} samples in {1} seconds.'.format(N, stop-start))

# get a subsampling for plotting
t_plot = np.linspace(tspan[0], tspan[1], 200)
states_plot = ssa.discretize_trajectory(trajectory, t_plot)

# get mean
states_avg = ssa.sample(model, initial, t_plot, num_samples=1, output='avg')
print(states_avg.shape)

# plot result 
if plotting:
    plt.plot(t_plot, 100*states_plot[:, 1], '-k')
    plt.plot(t_plot, states_plot[:, 2], '-b')
    plt.plot(t_plot, states_plot[:, 3], '-r')
    #plt.plot(obs_times.numpy(), obs.numpy(), 'xk')
    plt.show()

    # plt.plot(t_plot, 100*states_avg[:, 1], '-k')
    # plt.plot(t_plot, states_avg[:, 2], '-b')
    # plt.plot(t_plot, states_avg[:, 3], '-r')
    # #plt.plot(obs_times.numpy(), obs.numpy(), 'xk')
    # plt.show()