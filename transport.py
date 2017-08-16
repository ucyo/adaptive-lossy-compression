#!/usr/bin/env python
# coding: utf-8
"""Helper functions for transporting compressed values from one block to the other."""

import logging
import pickle

import xarray as xr
import numpy as np
import manualsarima as ms


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def cumcorr(timeseries, other):
    """Cumulative correlation of a time series."""
    return timeseries.index.map(lambda x: timeseries.corr(other.loc[:x]))


def cumcorr_close_to_one(timeseries, other, atol=1e-05):
    """Return if cumulative correlation is close to one."""
    return cumcorr(timeseries, other) >= 1-atol


def indices_need_correction(timeseries, other, atol=1e-05):
    """Return indices which need correction."""
    return np.where(cumcorr(timeseries, other) < 1-atol)[0]


def get_best_compressed(df, other, **kwargs):
    """Return no. of bits of best compressed time series."""
    for i in range(1, 33):
        if not indices_need_correction(df[i], other, **kwargs).any():
            return i
    return "uncompressed"


def indices(indiceslist, blocksize, from_series):
    """Takes list of startingpoints for blocks."""
    indiceslist = sorted(indiceslist)
    if not isinstance(indiceslist, np.ndarray):
        indiceslist = np.array(indiceslist)
    intersections = np.diff(indiceslist) > blocksize
    neg = indiceslist[~intersections]
    message = "{} not allowed: Next index within block range".format(neg)
    assert all(intersections), message
    logging.info("Starting points: %s", indiceslist)
    for start in indiceslist:
        yield from_series[start:start+blocksize]


def linear(quantity, blocksize, from_series):
    """Takes number of blocks and returns equidistant blocks."""
    blocklength = int((from_series.size - quantity*blocksize) / (quantity + 1))
    startingpoints = [x * (blocksize + blocklength) - (blocksize//2)
                      for x in range(1, quantity+1)]
    logging.info("Starting points: %s", startingpoints)
    for start in startingpoints:
        yield from_series[start:start+blocksize]


def percentage(percentage, blocksize, from_series):
    """Calculates linearly distributed number of blocks based on percentage."""
    quantity = int(from_series.size * percentage / blocksize)
    return linear(quantity=quantity,
                  blocksize=blocksize, from_series=from_series)


def replace(quantity, blocksize, from_series, to_series, mode='linear'):
    """Replace blocks from one series to another."""
    result = to_series.copy()
    if mode == 'linear':
        replacement_blocks = linear(quantity=quantity,
                                    blocksize=blocksize,
                                    from_series=from_series)
    elif mode == 'percentage':
        replacement_blocks = percentage(percentage=quantity,
                                        blocksize=blocksize,
                                        from_series=from_series)
    elif mode == 'indices':
        replacement_blocks = indices(indiceslist=quantity,
                                     blocksize=blocksize,
                                     from_series=from_series)
    elif mode == 'auto':
        if isinstance(quantity, (tuple, list)):
            replacement_blocks = indices(indiceslist=quantity,
                                         blocksize=blocksize,
                                         from_series=from_series)
        elif isinstance(quantity, float) and 0 <= quantity < 1:
            replacement_blocks = percentage(percentage=quantity,
                                            blocksize=blocksize,
                                            from_series=from_series)
        elif isinstance(quantity, (float, int)) and quantity >= 1:
            replacement_blocks = linear(quantity=int(quantity),
                                        blocksize=blocksize,
                                        from_series=from_series)
    else:
        m = "Expected mode=['linear', 'percentage',"
        essage = "'indices', 'auto'], got {}".format(mode)
        raise AttributeError(m+essage)
    replacement_blocks = list(replacement_blocks)
    logging.debug("Replacement blocks: %s", replacement_blocks)
    replaced = 0
    for x in replacement_blocks:
        for i, val in enumerate(x):
            valx_to_valy = '{:.3f} to {:.3f}'.format(result[x.index[i]], val)
            posx = "{}@{}".format(result.name, x.index[i])
            posy = "{}@{}".format(from_series.name, x.index[i])
            logging.debug('Replacing %s with %s > %s', posx, posy, valx_to_valy)
            result.set_value(x.index[i], val)
            replaced += 1
    print(replaced)
    return result


def _get_percentages(start=.0, end=.5, num=20):
    """Return percentages within borders."""
    return np.round(np.linspace(start, end, num), 3)


def _generate_random(timeseries, blocksize, factor=1, size=10):
    """Generate random index points"""
    while True:
        crange = range(int(timeseries.size/(blocksize*factor)))
        choices = sorted(np.random.choice(crange), replace=False, size=size)
        if all(np.diff(choices) > blocksize):
            break
    return choices


def _get_update_list(timeseries, other, blocksize, seed=1986):
    np.random.seed(seed)
    update_list = dict()
    indices = indices_need_correction(timeseries, other)
    try:
        while True:
            choosen = sorted(np.random.choice(indices, 10, replace=False))
            if all(np.diff(choosen) > blocksize):
                break
        return choosen
    except ValueError:
        logging.warning("Intended value error: %s", choosen)
    return update_list


def reconstruct(var, timeline, resid):
    """Reconstruction method for ARIMA time series."""
    assert resid.dtype == np.float32, "ERR: {}".format(np.float32)
    ds = xr.open_dataset('../database/{}_weather.nc'.format(timeline))
    modelname = './model/model_{}_{}.model'.format(var, timeline)
    model = pickle.load(open(modelname, 'rb'))
    data = getattr(ds, var).to_dataframe()[var]
    parameters = ms.getparams(model)
    r, _ = ms.differentiation(model.specification.get('k_diff', 0))
    fitted = ms.calc_fitted_seasonal(resid=resid, **parameters)
    return r(fitted, data)
