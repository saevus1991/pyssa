# kinetic model subclass that allows only reactions up to second order (pair interactions)

# This file contains a number of standard kinetic models


# simple gene expression model 
def simple_gene_expression():

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
def predator_prey():

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


## getter function

def get_standard_model(ident):
    models = {
        "simple_gene_expression": simple_gene_expression, 
        "predator_prey": predator_prey
    }
    return(models.get(ident)())