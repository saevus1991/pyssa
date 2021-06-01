import numpy as np
from scipy.interpolate import interp1d
# import xarray as xr
# import matplotlib.pyplot as plt
# import inspect
# import pyemd
# from pathos.multiprocessing import ProcessingPool
# from numpy_groupies import aggregate
# from itertools import repeat, count
# from collections import namedtuple


def falling_factorial(n, k):
    if (n < k):
        return(0)
    else:
        res = 1.0
        for i in range(int(k)):
            res *= n-i
        return(res)

def sample_discrete(weights, size=1, axis=0, keepdims=False, binsearch=True):
    """
    Generates samples from a set of discrete distributions.

    :param weights: Array of positive numbers representing the (unnormalized) weights of the distributions
    :param size: Integer indicating the number of samples to be generated per distribution
    :param axis: Axis along which the distributions are oriented in the array
    :param binsearch: If true, the distributions are processed sequentially but for each distribution the samples are
        drawn in parallel via binary search (fast for many categories and large numbers of samples). Otherwise, the
        distributions are processed in parallel but samples are drawn sequentially (fast for large number of
        distributions).
    :return: Array containing the samples. The shape coincides with that of the weight array, except that the length of
        the specified axis is now given by the size parameter.
    """
    # cast to numpy array and assert non-negativity
    weights = np.array(weights, dtype=float)
    try:
        assert np.all(weights >= 0)
    except AssertionError:
        raise ValueError('negative probability weights')

    # always orient distributions along the last axis
    weights = np.swapaxes(weights, -1, axis)

    # normalize weights and compute cumulative sum
    #weights /= np.sum(weights, axis=-1, keepdims=True)
    csum = np.cumsum(weights, axis=-1)

    # get shape of output array and sample uniform numbers
    shape = (*weights.shape[0:-1], size)
    x = np.zeros(shape, dtype=int)
    p = np.random.random(shape)

    # generate samples
    if binsearch:
        # total number of distributions
        nDists = int(np.prod(weights.shape[0:-1]))

        # orient all distributions along a single axis --> shape: (nDists, size)
        csum = csum.reshape(nDists, -1)
        x = x.reshape(nDists, -1)
        p = p.reshape(nDists, -1)

        # generate all samples per distribution in parallel, one distribution after another
        for ind in range(nDists):
            x[ind, :] = np.searchsorted(csum[ind, :], p[ind, :])

        # undo reshaping
        x = x.reshape(shape)
    else:
        # generate samples in parallel for all distributions, sample after sample
        for s in range(size):
            x[..., s] = np.argmax(p[..., s] <= csum, axis=-1)

    # undo axis swapping
    x = np.swapaxes(x, -1, axis)

    # remove unnecessary axis
    if size == 1 and not keepdims:
        x = np.squeeze(x, axis=axis)
        if x.size == 1:
            x = int(x)

    return x


def assert_squaremat(mat):
    """
    Check that mat is a square matrix
    """
    # confirm numpy array
    assert(type(mat) is np.ndarray)
    # confirm 2 dimensions
    assert(mat.ndim == 2)
    # confirm quardatic
    assert(mat.shape[0] == mat.shape[1])


def assert_stochmat(mat, tol=1e-6):
    """
    Check that mat is a stochastic matrix
    """
    # check square matrix
    assert_squaremat(mat)
    # check row sum zero property
    row_sum = np.sum(mat, axis=1)
    assert(np.max(np.abs(row_sum)) < tol)
    # check non-negative off-diagonals 
    q_mat = np.copy(mat)
    np.fill_diagonal(q_mat, 0.0)
    assert(np.all(q_mat >= 0.0))


def assert_ratemat(mat):
    # check square matrix
    assert_squaremat(mat)
    # check diagonal is zero
    assert(np.all(mat.diagonal() == 0.0))
    # check non-negative elements
    assert(np.all(mat >= 0.0))


def assert_transmat(mat, tol=1e-10):
    """
    Check that mat is a stochastic matrix
    """
    # check square matrix
    assert_squaremat(mat)
    # check row sum one property
    row_sum = np.sum(mat, axis=1)
    assert(np.max(np.abs(row_sum-1.0)) < tol)
    # check non-negative elements
    assert(np.all(mat >= 0.0))


def softmax_stepfun(time, time_grid, vals, inv_temp):
    """ 
    compute a softmax type interpolation of a function defined on a time_grid
    """
    centered_grid = time_grid[0:-1]
    arg = inv_temp*(time-centered_grid)**2
    weights = np.exp(-arg)
    weights = weights/np.sum(weights)
    return(np.sum(weights*vals))


def sigmoid(x):
    return(1/(1+np.exp(-x)))

def logistic_stepfun(time, time_grid, vals, inv_temp):
    # compute sigmoid weights
    arg = inv_temp*(time-time_grid[:-1])
    weights = sigmoid(arg)
    # convert to increments and multiply
    increments = vals-np.concatenate([np.array([0.0]), vals[:-1]])
    # return weighted sum of increments
    return(np.sum(weights*increments))


def get_stats(trajectories):
    """
    extract mean and covariance from a gillespie2 results file
    """
    # compute mean and cov
    mean = np.mean(trajectories, axis=0)
    tmp = trajectories - np.expand_dims(mean, 0)
    cov = np.mean(np.expand_dims(tmp, -1) @ np.expand_dims(tmp, -2), axis=0)
    return(mean, cov)


def discretize_trajectory(trajectory, sample_times, obs_model=None, kind='zero'):
    """ 
    Discretize a trajectory of a jump process by linear interpolation 
    at the support points given in sample times
    Input
        trajectory: a dict with keys 'initial', 'tspan', 'times', 'states'
        sample_times: np.array containin the sample times
    """
    initial = np.array(trajectory['initial'])
    if (len(trajectory['times']) == 0):
        times = trajectory['tspan']
        states = np.stack([initial, initial])
    elif (trajectory['times'][-1] < trajectory['tspan'][1]):
        delta = (trajectory['tspan'][1]-trajectory['tspan'][0])/1e-3
        times = np.concatenate([trajectory['tspan'][0:1], trajectory['times'], trajectory['tspan'][1:]+delta])
        states = np.concatenate([initial.reshape(1, -1), trajectory['states'], trajectory['states'][-1:, :]])
    else:
        times = np.concatenate([trajectory['tspan'][0:1], trajectory['times']])
        states = np.concatenate([initial.reshape(1, -1), trajectory['states']])
    sample_states = interp1d(times, states, kind=kind, axis=0)(sample_times)
    if obs_model is not None:
        test = obs_model.sample(states[0], sample_times[0])
        obs_dim = (obs_model.sample(states[0], sample_times[0])).size
        obs_states = np.zeros((sample_states.shape[0], obs_dim))
        for i in range(len(sample_times)):
            obs_states[i] = obs_model.sample(sample_states[i], sample_times[i])
        sample_states = obs_states
    return(sample_states)
