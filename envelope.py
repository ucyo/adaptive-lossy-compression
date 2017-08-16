#!/usr/bin/env python
# coding: utf-8


import numpy as np
from glob import glob
import os
import pandas as pd
import pickle
import manualsarima as ms
from functools import lru_cache
import transport as ts
import concurrent

import logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class BaseEnvelope(object):

    def __init__(self, folder, index, timeline, *args, **kwargs):
        self.folder = folder
        self.timeline = timeline
        self.index = index
        model_basename = 'model_{}_{}.model'.format(self.index, self.timeline)
        truth_basename = 'origin_{}_{}.raw'.format(self.index, self.timeline)
        self._modelpath = os.path.join(os.path.dirname(__file__), 'model', model_basename)
        self._truthpath = os.path.join(os.path.dirname(__file__),'data','original',truth_basename)

    @property
    def truth(self):
        return pd.Series(np.fromfile(self._truthpath, np.float32))

    @property
    def model(self):
        return pickle.load(open(self._modelpath, 'rb'))

    @property
    def parameters(self):
        return ms.getparams(self.model)

    @property
    def _core(self):
        df = pd.DataFrame()
        files = [os.path.join(self.folder,
                              '*_{}_{}.{:02d}.raw'.format(self.index,
                                                          self.timeline, bits))
                 for bits in range(1, 33)]
        files = [glob(x)[0] for x in files]
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_np = {executor.submit(np.fromfile,
                                            fname, np.float32):
                            fname for fname in files}
            for ftre in concurrent.futures.as_completed(future_to_np):
                fname = future_to_np[ftre]
                bits = int(fname[-6:-4])
                try:
                    data = ftre.result()
                    df[self.index, bits] = data
                except Exception:
                    raise
                else:
                    logger.info('Loaded:', fname)
        df.columns = pd.MultiIndex.from_tuples(df.columns,
                                               names=['index', 'bits'])
        return df

    def uncompressed_bits(self, bits):
        return self._core.loc[:, (self.index, bits)]


class TimeSeriesEnvelope(BaseEnvelope):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def signal(self):
        return self._core

    def signal_bits(self, bits):
        return self.uncompressed_bits(bits)


class ARIMAEnvelope(BaseEnvelope):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rev, _ = ms.differentiation(self.model.specification.get('k_diff',
                                                                      0))

    @property
    @lru_cache(maxsize=32)
    def signal(self):
        df = pd.DataFrame()
        r, _ = ms.differentiation(self.model.specification.get('k_diff', 0))
        for column in self._core:
            resid = self._core.loc[:, column]
            df[column] = self._reverse(resid)
        df.columns = pd.MultiIndex.from_tuples(df.columns,
                                               names=['index', 'bits'])
        df.sort_index(axis=1, level=1, inplace=True)
        return df

    def signal_bits(self, bits):
        return self._reverse(bits)

    def _reverse(self, resid):
        if isinstance(resid, (int, float)):
            resid = self.uncompressed_bits(resid)
        if not isinstance(resid, pd.Series):
            raise ValueError(resid, 'must be a pd.Series')
        fitted = ms.calc_fitted_seasonal(resid=resid, **self.parameters)
        return pd.Series(self.rev(fitted, self.truth))


class Modified(ARIMAEnvelope):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @lru_cache(maxsize=32)
    def best_compressed(self, atol=1e-05):
        for i in range(1, 33):
            if not ts.indices_need_correction(self.signal_bits(i), self.truth,
                                              atol=atol).any():
                return self.uncompressed_bits(i)
        return self.truth

    @property
    def blocksize(self):
        arparams = self.parameters['arparams'].size
        maparams = self.parameters['maparams'].size
        return max(arparams, maparams)

    def replace_first(self, percentage, from_bits, to_bits):
        from_series = self.uncompressed_bits(from_bits)
        to_series = self.uncompressed_bits(to_bits)
        num = int(to_series.size*percentage)
        resid = from_series[:num].append(to_series[num:])

        return self._reverse(resid)

    def replace_evenly(self, percentage, from_bits, to_bits):
        from_series = self.uncompressed_bits(from_bits)
        to_series = self.uncompressed_bits(to_bits)
        resid = ts.replace(percentage, self.blocksize,
                           from_series, to_series, mode='percentage')

        return self._reverse(resid)

    def replace_special(self, percentage, from_bits, to_bits, naked=False):
        from_series = self.uncompressed_bits(from_bits)
        to_series = self.uncompressed_bits(to_bits)
        if not naked:
            cumseries = pd.Series(np.diff(ts.cumcorr(self._reverse(to_series),
                                                     self.truth)))
        else:
            cumseries = pd.Series(np.diff(ts.cumcorr(to_series, self.truth)))
        num = int(to_series.size*percentage)
        listing = sorted([x for x in cumseries.sort_values()[:num].index])

        resid = to_series.copy()
        print('Special replaces:', len(listing))
        for x in listing:
            addpos = self.blocksize
            try:
                resid.set_value(x+addpos, from_series.iloc[x+addpos])
            except IndexError:
                resid.set_value(x, from_series.iloc[x])

        return self._reverse(resid)

    def raw_replace_special(self, percentage, from_bits, to_bits):
        return self.replace_special(percentage, from_bits, to_bits, naked=True)

    def replace_singles(self, listing, from_bits, to_bits):
        from_series = self.uncompressed_bits(from_bits)
        to_series = self.uncompressed_bits(to_bits)
        resid = to_series.copy()

        for x in listing:
            msg = 'Replace: Pos {} from {} to {}'.format(x, to_series.iloc[x],
                                                         from_series.iloc[x])
            logging.info(msg)
            resid.set_value(x, from_series.iloc[x])
        return self._reverse(resid)

    def replace_rolling(self, percentage, from_bits, to_bits, naked=False):
        from_series = self.uncompressed_bits(from_bits)
        to_series = self.uncompressed_bits(to_bits)
        num = int(to_series.size*percentage)

        if naked:
            roll_coor = to_series.rolling(self.blocksize).corr(self.truth)
        else:
            roll_coor = self._reverse(to_series).rolling(self.blocksize).corr(self.truth)
        blockstarts = roll_coor[roll_coor.diff() < -1e-5].index.values
        logger.info(blockstarts)

        listing = set(sorted([x-y
                              for x in blockstarts
                              for y in range(self.blocksize)]))
        listing = list(listing)[:num]
        logging.info(listing)

        resid = to_series.copy()
        for x in listing[:num]:
            msg = 'Replace: Pos {} from {} to {}'.format(x, to_series.iloc[x],
                                                         from_series.iloc[x])
            logging.info(msg)
            resid.set_value(x, from_series.iloc[x])
        return self._reverse(resid)

    def raw_replace_rolling(self, percentage, from_bits, to_bits):
        return self.replace_rolling(percentage, from_bits, to_bits,
                                    naked=True)

    def replace_cumcorr(self, percentage, from_bits, to_bits, naked=False):
        from_series = self.uncompressed_bits(from_bits)
        to_series = self.uncompressed_bits(to_bits)
        num = int(to_series.size*percentage)
        injected = list()

        resid = to_series.copy()
        while len(injected) < num:
            signal = self._reverse(resid=resid) if not naked else resid
            ccorr = pd.Series(ts.cumcorr(timeseries=signal,
                                         other=self.truth))
            grouped = ccorr.diff().groupby((ccorr > ccorr.shift()).cumsum())
            downfall = [(d[1:].index.values[0], d[1:].sum())
                        for _, d in grouped if len(d[1:].index.values) > 0]
            sorted_downfall = sorted(downfall, key=lambda tup: tup[1])

            for idx, _ in sorted_downfall:
                minimum = idx
                while minimum - 1 in injected:
                    minimum -= 1
                if minimum < self.blocksize:
                    continue
                listing = sorted([minimum-y for y in range(1, self.blocksize+1)])
                if not all([x in injected for x in listing]):
                    break

            for x in listing:
                msg = 'Replace: Pos {} from {} to {}'.format(x, to_series.iloc[x],
                                                             from_series.iloc[x])
                logging.warning(msg)
                resid.set_value(x, from_series.iloc[x])
            injected += listing
        logger.info("Changed {} values.".format(injected))
        return self._reverse(resid)

    def raw_replace_cumcorr(self, percentage, from_bits, to_bits):
        return self.replace_cumcorr(percentage, from_bits, to_bits,
                                    naked=True)
