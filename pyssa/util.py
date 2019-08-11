import numpy as np
# import xarray as xr
# import matplotlib.pyplot as plt
# import inspect
# import pyemd
# from pathos.multiprocessing import ProcessingPool
# from numpy_groupies import aggregate
# from itertools import repeat, count
# from collections import namedtuple


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
