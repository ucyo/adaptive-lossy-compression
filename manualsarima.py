#!/usr/bin/env python
# coding: utf-8
"""Calculate residue and fitted_values manually of ARIMA model."""

import numpy as np
import pandas as pd


def calc_resid_no_diff(arparams, maparams, intercept, data, *args, **kwargs):
    max_param = max(arparams.size, maparams.size)
    errarr = np.array(data[:max_param], np.float32).copy()
    for i in range(max_param, data.size):
        new_err = 0
        new_err += data[i]
        new_err -= intercept.astype(np.float64)
        new_err -= np.sum(arparams[::-1].astype(np.float64) * data[i-len(arparams):i])
        new_err -= np.sum(maparams[::-1].astype(np.float64) * errarr[i-len(maparams):i].astype(np.float64))
        errarr = np.append(errarr, new_err).astype(np.float32)
    return errarr


def calc_fitted_no_diff(arparams, maparams, intercept, resid, *args, **kwargs):
    max_param = max(arparams.size, maparams.size)
    val = resid[:max_param].copy()
    for i in range(max_param, resid.size):
        new_val = 0
        new_val += resid[i]
        new_val += intercept.astype(np.float64)
        new_val += np.sum(arparams[::-1].astype(np.float64) * val[i-len(arparams):i].astype(np.float64))
        new_val += np.sum(maparams[::-1].astype(np.float64) * resid[i-len(maparams):i].astype(np.float64))
        val = np.append(val, new_val).astype(np.float32)
    return val


def differentiation(diff, *args, **kwargs):

    def rev(diffarr, orig, dif=2):
        assert not np.isnan(orig[:dif]).any(), "First <{}> values are not allowed to be np.nan".format(diff)
        part1 = orig[:dif].copy()
        part2 = diffarr.copy()
        fulldiff = np.append(part1, part2)
        tmps = [fulldiff[k::dif].cumsum() for k in range(dif)]
        res = pd.Series(np.nan, index=orig.index)
        for x in tmps:
            res = res.add(x, fill_value=0.0)
        return res

    def rev_one(diffarr, orig):
        return rev(diffarr, orig, 1)

    def rev_two(diffarr, orig):
        return rev_one(rev_one(diffarr, orig.diff().dropna()), orig)
    
    def rev_zero(diffarr, orig):
        return diffarr

    def diff_one(orig):
        return orig.diff().dropna()

    def diff_two(orig):
        return orig.diff().diff().dropna()
    
    def diff_zero(orig):
        return orig

    if diff == 1:
        return (rev_one, diff_one)
    elif diff == 2:
        return (rev_two,diff_two)
    elif diff == 0:
        return (rev_zero,diff_zero)


def calc_resid_seasonal(arparams, maparams, intercept, data, seasonal_arparams, seasonal_maparams, season, *args, **kwargs):
    max_param = max(arparams.size, maparams.size, seasonal_arparams.size, seasonal_maparams.size)
    errarr = np.array(data[:max_param], np.float32).copy()
    for i in range(max_param, data.size):
        new_err = 0.0
        new_err += data[i]
        new_err -= intercept.astype(np.float64)
        new_err -= np.sum(arparams[::-1].astype(np.float64) * data[i-len(arparams):i])
        new_err -= np.sum(maparams[::-1].astype(np.float64) * errarr[i-len(maparams):i].astype(np.float64))

        new_err -= np.sum(seasonal_arparams.astype(np.float64) * data[i-len(seasonal_arparams):i])
        new_err -= np.sum(seasonal_maparams.astype(np.float64) * errarr[i-len(seasonal_maparams):i].astype(np.float64))

        errarr = np.append(errarr, new_err).astype(np.float32)
    return errarr


def calc_fitted_seasonal(arparams, maparams, intercept, resid, seasonal_arparams, seasonal_maparams, season, *args, **kwargs):
    max_param = max(arparams.size, maparams.size, seasonal_arparams.size, seasonal_maparams.size)
    val = resid[:max_param].copy()

    for i in range(max_param, resid.size):
        new_val = 0
        new_val += resid[i]
        new_val += intercept.astype(np.float64)
        new_val += np.sum(arparams[::-1].astype(np.float64) * val[i-len(arparams):i].astype(np.float64))
        new_val += np.sum(maparams[::-1].astype(np.float64) * resid[i-len(maparams):i].astype(np.float64))

        new_val += np.sum(seasonal_arparams.astype(np.float64) * val[i-len(seasonal_arparams):i].astype(np.float64))
        new_val += np.sum(seasonal_maparams.astype(np.float64) * resid[i-len(seasonal_maparams):i].astype(np.float64))

        val = np.append(val, new_val).astype(np.float32)
    return val


def getparams(model):
    arparams = model.arparams.astype(np.float32).copy() if 'ar' in model.param_terms else np.array([0], np.float32)
    maparams = model.maparams.astype(np.float32).copy() if 'ma' in model.param_terms else np.array([0], np.float32)
    seasonal_arparams = model.polynomial_seasonal_ar[1:].astype(np.float32).copy() if 'seasonal_ar' in model.param_terms else np.array([0], np.float32)
    seasonal_maparams = model.polynomial_seasonal_ma[1:].astype(np.float32).copy() if 'seasonal_ma' in model.param_terms else np.array([0], np.float32)
    intercept = model.params[0].astype(np.float32).copy()
    season = getattr(model.specification, 'seasonal_periods', 0)
    return dict(arparams=arparams,
                maparams=maparams,
                seasonal_arparams=seasonal_arparams,
                seasonal_maparams=seasonal_maparams,
                intercept=intercept,
                season=season)
