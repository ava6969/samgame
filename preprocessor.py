import numpy as np
import pandas as pd
import talib
import talib.abstract
from sklearn.preprocessing import MinMaxScaler

from common import Order


def conv_list_int(x):
    if isinstance(x, list):
        return [int(v) for v in x]
    else:
        return int(x)


def add_tech_ind(df, indicators, debug=True):
    if isinstance(indicators, str):
        indicators = indicators.split(' ')

    for ind in indicators:

        if ind == 'close_diff':
            close_diff = df['close'].diff()
            df['close_diff'] = np.tanh(close_diff)
        else:
            try:
                if debug:
                    print('adding', ind, 'to data')

                top = ind.split('!')
                assert len(top) <= 2, print('can only get one output or none')
                extra = '' if len(top) == 1 else top[1]
                splitted = top[0].split('(')
                assert len(splitted) != 0
                indicator = splitted[0]
                params = None if len(splitted) == 1 else splitted[1]
                ind_ = str(indicator).upper()
                fnc_a = getattr(talib.abstract, ind_)
                if params:
                    parameters = params[:-1].split('-')
                    flattened_params = [p.split(',') for p in parameters]
                    # convert to int
                    flattened_params = list(map(conv_list_int, flattened_params))
                    column_name = [ind_ + '_' + p + extra for p in parameters]
                    if extra != '':
                        temp_dict = {c: fnc_a(df, *f)[extra] for c, f in zip(column_name, flattened_params)}
                    else:
                        temp_dict = {c: fnc_a(df, *f) for c, f in zip(column_name, flattened_params)}
                else:
                    if extra != '':
                        temp = fnc_a(df)[extra]
                    else:
                        temp = fnc_a(df)
                    temp_dict = {ind_: temp}

                df = pd.concat([df, pd.DataFrame(temp_dict)], axis=1)

            except KeyError as e:
                print(e)

    df = df.bfill().ffill()

    if np.any(pd.isna(df)):
        df = df.fillna(0)

    return df


def normalize(df_dict, _min=0, _max=1):
    for t, df in df_dict.items():
        scalar = MinMaxScaler(feature_range=(_min, _max))
        df_dict[t] = scalar.fit_transform(df[1:])
    return df_dict


def make_orders(tickers, qty_idx, frame, qty_val):
    market_values = [frame[t].close.values[-1] for t in tickers]  # get all closing price
    return [Order(t, abs(qty_val[q]), 'buy' if qty_val[q] > 0 else 'sell' if qty_val[q] < 0 else 'hold', v)
            for t, q, v in zip(tickers, qty_idx, market_values) ]