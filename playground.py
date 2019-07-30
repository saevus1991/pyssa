## play around with pyro to check inference 

# import stuff
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

# custom files
import sys
sys.path.append('/Users/christian/Documents/code/pyssa/')
from models.PhysicalKineticModel import PhysicalKineticModel
import models.standard_models as sm

# set data type to double
torch.set_default_dtype(torch.float64)

def get_reaction_tensor(pre):
    # get order of the individual reactions and number of reactions
    num_reactions, num_species = pre.size()
    reaction_order = torch.sum(pre, dim=1)
    # reject if higher order reaction is included
    if (torch.max(reaction_order) > 2):
        raise Exception('System should not contain reactions of order larger than two.')
    # set up the outputs
    zeroth = torch.zeros(num_reactions, dtype=torch.float64)
    first = torch.zeros([num_reactions, num_species], dtype=torch.float64)
    second = torch.zeros([num_reactions, num_species**2], dtype=torch.float64)
    # iterate over reactions 
    for i in range(num_reactions):
        if reaction_order[i] == 0:
            zeroth[i] == 1
        elif reaction_order[i] == 1:
            first[i, :] = pre[i, :]
        elif reaction_order[i] == 2:
            if torch.max(pre[i, :]) == 2:
                ind = pre[i, :].nonzero()
                second[i, ind*(num_species+1)] = 1
                first[i, ind] = -1
            else:
                ind = pre[i, :].nonzero()
                second[i, ind[0]*num_species+ind[1]] = 0.5
                second[i, ind[1]*num_species+ind[0]] = 0.5
        else:
            raise Exception('Check system matrices.')
    return zeroth, first, second


def mass_action(state, model):
    """
    A function R^num_species -> R^num_reactions parametrized by the stoichiometry matrix 
    Inputs
        state: current state of the system, torch array of size [num_species]
        model: a model in form of PyhsicalMassAction

    """
    # zeroth oder contribution
    prop = model.zeroth
    # first oder contribution
    tmp = torch.mm(model.first, state.view(-1,1))
    prop += tmp.squeeze()
    # second order contribution
    tmp = torch.einsum('i,j,aij->a', state, state, model.second)
    prop += tmp
    # multiply with rates
    prop *= model.rates
    # normalize
    delta = torch.sum(prop)
    prop /= delta
    return(delta, prop)

def next_event(prop):
    """
    Compute reaction time and next sate
    """
    time = torch.sum(prop)

def pyro_ssa(initial, tspan, model, obs_times):
    sigma = 0.15
    # iterate over time
    time = tspan[0]
    state = initial.clone()
    cnt = 0
    cnt_obs = 0
    # create latent variables
    while (time < tspan[1]):
        # compute propensity
        delta, prop = mass_action(state, model)
        # sample updates
        tau = pyro.sample("tau_{}".format(cnt), dist.Exponential(delta))
        reaction = pyro.sample("reaction_{}".format(cnt), dist.Categorical(prop))
        # check if new state ill be larger than current obs time
        if (cnt_obs < len(obs_times)) and (time < obs_times[cnt_obs]).item() and (time+tau > obs_times[cnt_obs]).item():
            mu_loc = torch.log(state[-1]+1)
            obs = pyro.sample("obs_{}".format(cnt_obs), dist.LogNormal(mu_loc, sigma))
            cnt_obs += 1
        # update states
        time = time+tau
        state = state+model.stoichiometry[reaction, :]
        cnt += 1

def pyro_ssa2(initial, tspan, model, obs_times):
    sigma = 0.1
    # iterate over time
    time = tspan[0]
    state = initial.clone()
    cnt_obs = 0
    # create latent variables
    for cnt in pyro.markov(range(int(1e6))):
        # compute propensity
        delta, prop = mass_action(state, model)
        # sample updates
        tau = pyro.sample("tau_{}".format(cnt), dist.Exponential(delta))
        reaction = pyro.sample("reaction_{}".format(cnt), dist.Categorical(prop))
        # check if new state ill be larger than current obs time
        if (cnt_obs < len(obs_times)) and (time < obs_times[cnt_obs]).item() and (time+tau > obs_times[cnt_obs]).item():
            mu_loc = torch.log(state[-1]+1)
            obs = pyro.sample("obs_{}".format(cnt_obs), dist.LogNormal(mu_loc, sigma))
            cnt_obs += 1
        # update states
        time = time+tau
        state = state+model.stoichiometry[reaction, :]
        # terminating condtion
        if time > tspan[1]:
            break


def get_event_history(nodes):
    # initialize
    tau = []
    reaction = []
    obs = []
    for name, props in nodes.items():
        if props['type'] == 'sample':
            if 'obs' in name:
                obs.append(props['value'].item())
            elif 'tau' in name:
                tau.append(props['value'].item())
            elif 'reaction' in name:
                reaction.append(props['value'].item())
    # list of tensors -> tensor
    tau = torch.tensor(tau)
    reaction = torch.tensor(reaction)
    obs = torch.tensor(obs)
    return tau, reaction, obs


def get_latent_states(t0, initial, tau, reaction, model):
    # set up outputs
    num_steps = len(tau)+1
    latent_dim = len(initial)
    states = torch.zeros([num_steps, latent_dim])
    times = torch.zeros(num_steps)
    # initialize 
    times[0] = t0
    states[0, :] = initial
    # fill iteratively
    for i in range(num_steps-1):
        times[i+1] = times[i] + tau[i]
        states[i+1, :] = states[i, :] + model.stoichiometry[reaction[i], :]
    return times, states

#def sub_sample()

# set up the model
pre, post, rates = sm.get_standard_model("simple_gene_expression")
model = PhysicalKineticModel(torch.tensor(pre, dtype=torch.float64), torch.tensor(post, dtype=torch.float64), torch.tensor(rates, dtype=torch.float64))

# prepare initial conditions
initial = torch.tensor([0.0, 1.0, 0.0, 0.0])
tspan = torch.tensor([0.0, 3e3])
delta_t = 300.0
obs_times = torch.arange(tspan[0]+0.5*delta_t, tspan[1], delta_t)

# poutine trace wrappper 
trace_syn = poutine.trace(pyro_ssa).get_trace(initial, tspan, model, obs_times)

# reconstruct data
tau, reaction, obs = get_event_history(trace_syn.nodes)
times, states = get_latent_states(tspan[0], initial, tau, reaction, model)

# get a subsampling for plotting
t_plot = np.linspace(tspan[0], tspan[1], 200)
states_plot = interp1d(times.numpy(), states.numpy(), kind='zero', axis=0)(t_plot)

# plot result 
plt.plot(t_plot, 100*states_plot[:, 1], '-k')
plt.plot(t_plot, states_plot[:, 2], '-b')
plt.plot(t_plot, states_plot[:, 3], '-r')
plt.plot(obs_times.numpy(), obs.numpy(), 'xk')
plt.show()

# define condtional models 
# def conditioned_model(data):
#     return poutine.condition(blocked_model, data={"obs": data})(len(data))
conditoned = poutine.condition(pyro_ssa, data={'obs': obs})