import enum
from typing import List
import pandas as pd
import numpy as np
import quantstats as qs
from quantstats import stats


class BTMetrics(enum.Enum):

    SHARPE_RATIO = 'SR'
    WIN_RATE = 'WR'
    VOLATILITY = 'VOL'
    SORTINO_RATIO = 'SOR'
    CALMAR_RATIO = 'CAL'
    RECOVERY_FACTOR = 'RF'
    RISK_RETURN_RATIO = 'RRR'
    MAX_DRAWDOWN = 'MDR'
    DRAWDOWN_DETAILS = 'DRD'
    GREEKS = 'AB'


qs.extend_pandas()


def calculate_returns(account_balances):
    shift = len(account_balances)

    returns = account_balances[['Portfolio Value']] - account_balances[['Portfolio Value']].shift(shift) /\
        account_balances.shift(shift)

    return returns


def backtest(account_balances, metrics: List[BTMetrics]):
    """
    :param account_balances:
    :param metrics: ex [BTMetrics.SHARPE_RATIO , BTMetrics.WIN_RATE ]
    :return:
    """
    periods = len(account_balances)
    portfolio_returns = calculate_returns(account_balances)

    sharpe_ratio = qs.stats.sharpe(portfolio_returns, periods=periods, annualize=False)
    win_rate = qs.stats.win_rate(portfolio_returns)
    volatility = qs.stats.volatility(portfolio_returns, periods=periods, annualize=False)
    sortino_ratio = qs.stats.sortino(portfolio_returns, periods=periods, annualize=False)
    calmar_ratio = qs.stats.calmar(portfolio_returns)
    recovery_factor = qs.stats.recovery_factor(portfolio_returns)
    risk_return_ratio = qs.stats.risk_return_ratio(portfolio_returns)
    max_drawdown = qs.stats.max_drawdown(account_balances)
    drawdown_details = qs.stats.drawdown_details(max_drawdown)
    # greeks = qs.stats.greeks(portfolio_returns, SPY / BTC? / cryptoETF?, periods=periods)

    result = dict()  # map of the metric - > calculated value

    if BTMetrics.SHARPE_RATIO in metrics:
        result[BTMetrics.SHARPE_RATIO] = sharpe_ratio(portfolio_returns, periods=periods, annualize=False)
    if BTMetrics.WIN_RATE in metrics:
        result[BTMetrics.WIN_RATE] = win_rate(portfolio_returns)
    if BTMetrics.VOLATILITY in metrics:
        result[BTMetrics.VOLATILITY] = volatility(portfolio_returns, periods=periods, annualize=False)
    if BTMetrics.SORTINO_RATIO in metrics:
        result[BTMetrics.SORTINO_RATIO] = sortino_ratio(portfolio_returns, periods=periods, annualize=False)
    if BTMetrics.CALMAR_RATIO in metrics:
        result[BTMetrics.CALMAR_RATIO] = calmar_ratio(portfolio_returns)
    if BTMetrics.RECOVERY_FACTOR in metrics:
        result[BTMetrics.RECOVERY_FACTOR] = recovery_factor(portfolio_returns)
    if BTMetrics.RISK_RETURN_RATIO in metrics:
        result[BTMetrics.RISK_RETURN_RATIO] = risk_return_ratio(portfolio_returns)
    if BTMetrics.MAX_DRAWDOWN in metrics:
        result[BTMetrics.MAX_DRAWDOWN] = max_drawdown(account_balances)
    if BTMetrics.DRAWDOWN_DETAILS in metrics:
        result[BTMetrics.DRAWDOWN_DETAILS] = drawdown_details(max_drawdown)
    # if BTMetrics.GREEKS in metrics:
    #    result[BTMetrics.GREEKS] = greeks(portfolio_returns, SPY / BTC? / cryptoETF?, periods=periods)

    return result
