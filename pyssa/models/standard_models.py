# This file contains a number of standard kinetic models

import numpy as np

def from_excel(file_name):
    # get file path
    from pathlib import Path
    dir_path = Path(__file__).absolute().parents[0]
    #print(dir_path)
    file_path = str(dir_path) + '/collection/' + file_name
    # load file to pandas data frame
    import pandas as pd
    df = pd.ExcelFile(file_path).parse()
    # get indices for extraction 
    pre_begin = -1
    for i in range(df.shape[0]):
        if df.iat[i, 0] == 'Pre':
            pre_begin = i
        try:
            float(df.iat[i, 0])
            if np.isnan(df.iat[i, 0]) and pre_begin >= 0:
                pre_end = i
                break
        except ValueError:
            None
    post_begin = -1
    for i in range(pre_end, df.shape[0]):
        if df.iat[i, 0] == 'Post':
            post_begin = i
        try:
            float(df.iat[i, 0])
            if np.isnan(df.iat[i, 0]) and post_begin >= 0:
                post_end = i
                break
        except ValueError:
            None
    rates_begin = -1
    rates_end = df.shape[0]
    for i in range(post_end, df.shape[0]):
        if df.iat[i, 0] == 'Rates':
            rates_begin = i
        try:
            float(df.iat[i, 0])
            if np.isnan(df.iat[i, 0]) and rates_begin >= 0:
                rates_end = i
                break
        except ValueError:
            None
    # extract data 
    pre = df.iloc[pre_begin+1:pre_end, 1:df.shape[1]].fillna(0).to_numpy().astype(np.float64)
    post = df.iloc[post_begin+1:post_end, 1:df.shape[1]].fillna(0).to_numpy().astype(np.float64)
    rates = df.iloc[rates_begin+1:rates_end, 1].to_numpy().astype(np.float64)
    return(pre, post, rates)


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

# one component oscillator by negative feedback
def single_gene_oscillator(*args):

    pre = [[1, 0, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]]

    post = [[0, 1, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 1, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 1], 
        [0, 0, 0, 0]]

    rates = [0.01, 0.001, 0.15, 0.01, 0.04, 0.08]

    return (pre, post, rates)

# oscillator based on induced degradation
def degradation_oscillator(*args):

    pre = [[0, 0, 0, 0],    # mrna production
        [1, 0, 0, 0],       # mrna decay
        [1, 0, 0, 0],       # protein production
        [0, 1, 0, 0],       # protein decay
        [0, 1, 0, 0],       # E0 production
        [0, 0, 1, 0],       # E0 E1 conversion
        [0, 0, 0, 1],       # E1 decay 
        [0, 1, 0, 1]]       # catalyzed protein decay

    post = [[1, 0, 0, 0],   # mrna production
        [0, 0, 0, 0],       # mrna decay
        [1, 1, 0, 0],       # protein production
        [0, 0, 0, 0],       # protein decay
        [0, 1, 1, 0],       # E0 production
        [0, 0, 0, 1],       # E0 E1 conversion
        [0, 0, 0, 0],       # E1 decay 
        [0, 0, 0, 1]]       # catalyzed protein decay

    rates = [0.0007, 0.0008, 0.11, 0.00003, 0.0003, 0.0004, 0.0001, 0.001]

    return (pre, post, rates)


def stochastic_repressilator(*args):

    file_name = 'stochastic_repressilator.xlsx'
    pre, post, rates = from_excel(file_name)

    return(pre, post, rates)


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

# totatlly asymmetric exclusion process
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
    
# totatlly asymmetric exclusion process
def bursting_tasep(*args):

    if len(args) == 2:
        L = args[0]
        num_stems = args[1]
    else:
        L = 48
        num_stems = 14

    alpha1 = [i for i in range(num_stems)]
    alpha2 = [num_stems for i in range(L-num_stems)]
    alpha = alpha1+alpha2

    rates = [0.21, 0.51, 0.28, 0.5, 0.5]

    obs_param = [80.0, 100.0, 0.001, 10.0, 0.2]

    return(alpha, rates, obs_param)


## getter function


def get_standard_model(ident, *args):
    models = {
        "simple_gene_expression": simple_gene_expression,
        "single_gene_oscillator": single_gene_oscillator,
        "stochastic_repressilator": stochastic_repressilator,
        "predator_prey": predator_prey,
        "tasep": tasep,
        "bursting_tasep": bursting_tasep,
        "degradation_oscillator": degradation_oscillator
    }
    return(models.get(ident)(args))