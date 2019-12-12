import numpy as np
import sys
import pyssa.gillespie as gillespie
from pyssa.models.standard_models import get_standard_model

# import gene expression model
pre, post, rates = get_standard_model('simple_gene_expression')

# input for the function
pre = np.array(pre, dtype=np.float64)
post = np.array(post, dtype=np.float64)
rates = np.array(rates)
initial = np.array([1.0, 0.0, 0.0, 0.0])
tspan = np.array([0.0, 5000.0])
seed = np.random.randint(2**16)


# compiled tasep forward
control = np.zeros(3)
sample = gillespie.simulate(pre, post, rates, initial, tspan, seed)

print(sys.getrefcount(sample['initial']))
print(sys.getrefcount(sample['tspan']))
print(sys.getrefcount(sample['times']))
print(sys.getrefcount(sample['events']))
print(sys.getrefcount(sample['states']))




