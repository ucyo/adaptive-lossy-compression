#!/usr/bin/env python
# coding: utf-8
"""Calculate replacements concurrently."""

import os
import multiprocessing as mp
from concurrent import futures

import pandas as pd
import transport as ts
from envelope import TimeSeriesEnvelope, Modified


def main():
    """Calculate replacement methods cocurrently."""
    percentages = [5, 10]  # percent of data points to be updated
    timeline = ['mm']  # time spectrum with 'mm' for months and 'dm' for days
    indices = ['nao', 'enso34', 'qbo30', 'qbo50']  # which environmental index to calculate
    w = mp.cpu_count()  # how many max. worker nodes to use
    for perc in percentages:
        for t in timeline:
            for x in indices:
                with futures.ProcessPoolExecutor(max_workers=w) as executor:
                    future_corrs = {executor.submit(get_corr, x, t, y, k, perc): '{}_{}_{:02d}_{:02d}'.format(x, t, y, k)
                                    for y in range(6, 7)
                                    for k in range(1, 4)}
                    for ftre in futures.as_completed(future_corrs):
                        try:
                            cname = future_corrs[ftre]
                            data = ftre.result()
                            data.to_csv('Reproduce_'+str(perc)+'percent_'+cname+'.csv')
                        except Exception:
                            print('Error', cname)
                            raise
                        else:
                            print('Success', cname)


def get_corr(index, timeline, to_bits, from_bits=1, perc=10):
    """Calculate cumulative correlation of each replacement method."""
    print('Doing timeline {}, compressed at {}, getting from +{}'.format(timeline, to_bits, from_bits))
    from_bits = to_bits + from_bits
    percentage = perc/100.
    arima_path = os.path.join(os.path.dirname(__file__), 'data', 'arima_uncompressed')
    zfp_path = os.path.join(os.path.dirname(__file__), 'data', 'direct_uncompressed')
    arima = Modified(arima_path, index=index, timeline=timeline)
    zfp = TimeSeriesEnvelope(zfp_path, index=index, timeline=timeline)

    df = pd.DataFrame(dict(
        arima=arima.signal_bits(to_bits),
        truth=arima.truth,
        zfp=zfp.signal_bits(to_bits),
        special=arima.replace_special(percentage=percentage,
                                      from_bits=from_bits, to_bits=to_bits),
        evenly=arima.replace_evenly(percentage=percentage,
                                    from_bits=from_bits, to_bits=to_bits),
        first=arima.replace_first(percentage=percentage,
                                  from_bits=from_bits, to_bits=to_bits),
        roll=arima.replace_rolling(percentage=percentage,
                                   from_bits=from_bits, to_bits=to_bits),
        cumcorr=arima.replace_cumcorr(percentage=percentage,
                                      from_bits=from_bits, to_bits=to_bits),
        raw_cumcorr=arima.raw_replace_cumcorr(percentage=percentage,
                                              from_bits=from_bits,
                                              to_bits=to_bits),
        raw_roll=arima.raw_replace_rolling(percentage=percentage,
                                           from_bits=from_bits,
                                           to_bits=to_bits),
        raw_special=arima.raw_replace_special(percentage=percentage,
                                              from_bits=from_bits,
                                              to_bits=to_bits)
    ))

    dfcumcorr = pd.DataFrame({x: ts.cumcorr(df.loc[:, x], df.truth)
                              for x in df})
    return dfcumcorr


if __name__ == '__main__':
    main()
