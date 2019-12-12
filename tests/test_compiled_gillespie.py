"""
test compiled gillespie for mass action models with the standard gene expression model
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import pyssa.gillespie as gillespie
from pyssa.models.standard_models import get_standard_model
import pyssa.ssa as ssa

# activate or deactivate plotting
#np.random.seed(1912121445)
plotting = True

# import gene expression model
pre, post, rates = get_standard_model('simple_gene_expression')

# input for the function
pre = np.array(pre, dtype=np.float64)
post = np.array(post, dtype=np.float64)
rates = np.array(rates)
initial = np.array([1.0, 0.0, 0.0, 0.0])
tspan = np.array([0.0, 5000.0])
seed = np.random.randint(2**16)

# get trajectory 
trajectory = gillespie.simulate(pre, post, rates, initial, tspan, seed)

# get a subsampling for plotting
t_plot = np.linspace(tspan[0], tspan[1], 200)
states_plot = ssa.discretize_trajectory(trajectory, t_plot)

# get mean 
def simulate(tspan):
    seed = np.random.randint(2**16)
    trajectory = gillespie.simulate(pre, post, rates, initial, tspan, seed)
    return(trajectory)
states_avg = ssa.sample_compiled(simulate, t_plot, num_samples=1000, output='avg')
#print(states_avg.shape)
# print(states_avg.shape)
# ind = states_avg > 10000
# print(ind.sum())

# plot result 
if plotting:
    plt.plot(t_plot, 100*states_plot[:, 1], '-k')
    plt.plot(t_plot, states_plot[:, 2], '-b')
    plt.plot(t_plot, states_plot[:, 3], '-r')
    #plt.plot(obs_times.numpy(), obs.numpy(), 'xk')
    plt.show()

    plt.plot(t_plot, 100*states_avg[:, 1], '-k')
    plt.plot(t_plot, states_avg[:, 2], '-b')
    plt.plot(t_plot, states_avg[:, 3], '-r')
    #plt.plot(obs_times.numpy(), obs.numpy(), 'xk')
    plt.show()


