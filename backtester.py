from typing import List

import pandas as pd
from ray.pickle5_files import pickle5
import numpy as np
import quantstats as qs
import enum


class BTMetrics(enum.Enum):
    SHARPE_RATIO = 'SR'
    MAX_DRAWDOWN = 'MDR'


qs.extend_pandas()


def weekly_backtest_with(account_balances,
                         metrics:List[BTMetrics]):
    """
    :param account_balances:
    :param paramaters: ex ['sharpe_ratio', '']
    :return:
    """
    periods = len(account_balances)
    returns = []
    result = dict()
    if BTMetrics.SHARPE_RATIO in metrics:
        result[BTMetrics.SHARPE_RATIO] = qs.stats.sharpe(returns, periods=periods, annualize=False)
    # returns only performance metrics given in list

    return


def weekly_backtest_all(account_balances):
    """
    returns.columns = [timedelta, account_balance]
    Ex
    index|timedelta|returns
    0|XXX-XXX-X|10000
    0|XXX-XXX-X|10010
    returns: |timedelta|returns
    timeframe: minute, day,
    """

    # return all performance metrics
    return


if __name__ == '__main__':
    period = 200
    date = pd.date_range(start=pd.datetime(2020, 1, 1, 9, 30, 0), periods=period, freq="min")
    returns = np.random.randint(1000, 2000, period)

    returns_pd = pd.DataFrame({'timedelta' : date, 'returns' : returns})
    print(returns_pd)
