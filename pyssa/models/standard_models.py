# kinetic model subclass that allows only reactions up to second order (pair interactions)

# This file contains a number of standard kinetic models


# simple gene expression model 
def simple_gene_expression(*args):

    pre = [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]

    post = [[0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 1], 
        [0, 0, 0, 0]]

    rates = [0.001, 0.001, 0.15, 0.001, 0.04, 0.008]

    return (pre, post, rates)


# standard predator prey  model 
def predator_prey(*args):

    pre = [[1, 0],
        [1, 1],
        [1, 1],
        [0, 1]]

    post = [[2, 0],
        [0, 1],
        [1, 2],
        [0, 0]]

    rates = [5e-4, 1e-4, 1e-4, 5e-4]

    return(pre, post, rates)


def tasep(*args):

    if len(args) == 2:
        L = args[0]
        num_stems = args[1]
    else:
        L = 48
        num_stems = 14

    alpha1 = [i+1 for i in range(num_stems)]
    alpha2 = [num_stems for i in range(L-num_stems)]
    alpha = alpha1+alpha2

    rates = [0.005, 0.5, 0.05]

    obs_param = [80.0, 100.0, 0.001, 10.0, 0.2]

    return(alpha, rates, obs_param)

## getter function


def get_standard_model(ident, *args):
    models = {
        "simple_gene_expression": simple_gene_expression,
        "predator_prey": predator_prey,
        "tasep": tasep
    }
    return(models.get(ident)(args))