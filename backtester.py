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


def weekly_backtest_with(account_balances, metrics:List[BTMetrics]):
    """
    :param account_balances:
    :param paramaters: ex ['sharpe_ratio', '']
    :return:
    """
    periods = len(account_balances)
    if periods < 2:
        return {'empty': 0}
    returns = account_balances.pct_change()
    result = dict()
    if BTMetrics.SHARPE_RATIO in metrics:
        result[BTMetrics.SHARPE_RATIO] = float(qs.stats.sharpe(returns, periods=periods, annualize=False))
    if BTMetrics.MAX_DRAWDOWN in metrics:
        result[BTMetrics.MAX_DRAWDOWN] = float(qs.stats.max_drawdown(account_balances))
    # returns only performance metrics given in list

    return result


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
