## play around with pyro to check inference 

# import stuff
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine

from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate, infer_discrete

# custom files
import sys
sys.path.append('/Users/christian/Documents/code/pyssa/')

# define a simple model

K = 2  # Fixed number of components.

def print_event_history(nodes):
    # initialize
    for name, props in nodes.items():
        print(name)
        print(props)

# set up data
data = torch.tensor([0., 1., 10., 11., 12.])

# define the model
#@config_enumerate
def model(data):
    # Global variables.
    weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))
    scale = pyro.sample('scale', dist.LogNormal(0., 2.))
    with pyro.plate('components', K):
        locs = pyro.sample('locs', dist.Normal(0., 10.))

    with pyro.plate('data', len(data)):
        # Local variables.
        assignment = pyro.sample('assignment', dist.Categorical(weights))
        pyro.sample('obs', dist.Normal(locs[assignment], scale), obs=data)

# run to obtain synthetic data
#trace_syn = poutine.trace(model).get_trace(data)

#print_event_history(trace_syn.nodes)

# reconstructed model
def prior():
    weights = pyro.sample('weights', dist.Dirichlet(0.5 * torch.ones(K)))
    scale = pyro.sample('scale', dist.LogNormal(0., 2.))
    return(weights, scale)

def model1(data):
    # draw from prior
    weights, scale = prior()
    with pyro.plate('components', K):
        locs = pyro.sample('locs', dist.Normal(0., 10.))

    with pyro.plate('data', len(data)):
        # Local variables.
        assignment = pyro.sample('assignment', dist.Categorical(weights))
        pyro.sample('obs', dist.Normal(locs[assignment], scale), obs=data)

trace_syn = poutine.trace(model1).get_trace(data)

print_event_history(trace_syn.nodes)