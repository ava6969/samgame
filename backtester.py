import enum
from typing import List
import pandas as pd
import numpy as np
import quantstats as qs
from quantstats import stats


class BTMetrics(enum.Enum):

    D_SHARPE_RATIO = 'D-SR'
    D_WIN_RATE = 'D-WR'
    D_VOLATILITY = 'D-VOL'
    D_SORTINO_RATIO = 'D-SOR'
    D_CALMAR_RATIO = 'D-CAL'
    D_RECOVERY_FACTOR = 'D-RF'
    D_RISK_RETURN_RATIO = 'D-RRR'
    D_MAX_DRAWDOWN = 'D-MDR'
    D_DRAWDOWN_DETAILS = 'D-DD'
    D_ALPHA = 'D-A'
    D_BETA = 'D-B'

    W_SHARPE_RATIO = 'W-SR'
    W_WIN_RATE = 'W-WR'
    W_VOLATILITY = 'W-VOL'
    W_SORTINO_RATIO = 'W-SOR'
    W_CALMAR_RATIO = 'W-CAL'
    W_RECOVERY_FACTOR = 'W-RF'
    W_RISK_RETURN_RATIO = 'W-RRR'
    W_MAX_DRAWDOWN = 'W-MDR'
    W_DRAWDOWN_DETAILS = 'W-DD'
    W_ALPHA = 'W-A'
    W_BETA = 'W-B'

    M_SHARPE_RATIO = 'M-SR'
    M_WIN_RATE = 'M-WR'
    M_VOLATILITY = 'M-VOL'
    M_SORTINO_RATIO = 'M-SOR'
    M_CALMAR_RATIO = 'M-CAL'
    M_RECOVERY_FACTOR = 'M-RF'
    M_RISK_RETURN_RATIO = 'M-RRR'
    M_MAX_DRAWDOWN = 'M-MDR'
    M_DRAWDOWN_DETAILS = 'M-DD'
    M_ALPHA = 'M-A'
    M_BETA = 'M-B'

    A_SHARPE_RATIO = 'A-SR'
    A_WIN_RATE = 'A-WR'
    A_VOLATILITY = 'A-VOL'
    A_SORTINO_RATIO = 'A-SOR'
    A_CALMAR_RATIO = 'A-CAL'
    A_RECOVERY_FACTOR = 'A-RF'
    A_RISK_RETURN_RATIO = 'A-RRR'
    A_MAX_DRAWDOWN = 'A-MDR'
    A_DRAWDOWN_DETAILS = 'A-DD'
    A_ALPHA = 'A-A'
    A_BETA = 'A-B'

qs.extend_pandas()

def calculate_returns(account_balance):
    #Need method to get BOD and EOD values
    daily_returns = account_balance[['Portfolio Value']] - account_balance[['Portfolio Value']].shift(1)/\
                    account_balance.shift(1)
    #Use 7? Use a calculated period based on current time/date?
    weekly_returns = account_balance[['Portfolio Value']] - account_balance[['Portfolio Value']].shift(7)/\
                    account_balance.shift(7)
    # Use 30? Use a calculated period based on current time/date?
    monthly_returns = account_balance[['Portfolio Value']] - account_balance[['Portfolio Value']].shift(30)/\
                    account_balance.shift(30)
    # Use 365? Use a calculated period based on current time/date?
    annual_returns = account_balance[['Portfolio Value']] - account_balance[['Portfolio Value']].shift(365)/\
                    account_balance.shift(365)

    return daily_returns, weekly_returns, monthly_returns, annual_returns

def daily_backtest(account_balances, daily_returns, metrics):
    d_sharpe_ratio = qs.stats.sharpe(daily_returns, periods=1, annualize=False)
    d_win_rate = qs.stats.win_rate(daily_returns)
    d_volatility = qs.stats.volatility(daily_returns, periods=1, annualize=False)
    d_sortino_ratio = qs.stats.sortino(daily_returns, periods=1, annualize=False)
    d_calmar_ratio = qs.stats.calmar(daily_returns)
    d_recovery_factor = qs.stats.recovery_factor(daily_returns)
    d_risk_return_ratio = qs.stats.risk_return_ratio(daily_returns)
    d_max_drawdown = qs.stats.max_drawdown(account_balances)
    d_drawdown_details = qs.stats.drawdown_details(d_max_drawdown)
    d_alpha, d_beta = qs.stats.greeks(daily_returns, SPY / BTC? / cryptoETF?, periods = 1)

    return d_sharpe_ratio,d_win_rate,d_volatility,d_sortino_ratio,d_calmar_ratio,d_recovery_factor, \
           d_risk_return_ratio,d_max_drawdown,d_drawdown_details,d_alpha,d_beta

def weekly_backtest(account_balances, weekly_returns,
                         metrics:List[BTMetrics]):
    """
    :param account_balances:
    :param paramaters: ex ['sharpe_ratio', '']
    :return:
    """
    periods = len(account_balances)
    #returns = []
    #result = dict()
    if BTMetrics.SHARPE_RATIO in metrics:
        result[BTMetrics.SHARPE_RATIO] = qs.stats.sharpe(weekly_returns, periods=periods, annualize=False)

    w_sharpe_ratio = qs.stats.sharpe(weekly_returns, periods=7, annualize=False)
    w_win_rate = qs.stats.win_rate(weekly_returns)
    w_volatility = qs.stats.volatility(weekly_returns, periods=7, annualize=False)
    w_sortino_ratio = qs.stats.sortino(weekly_returns, periods=7, annualize=False)
    w_calmar_ratio = qs.stats.calmar(weekly_returns)
    w_recovery_factor = qs.stats.recovery_factor(weekly_returns)
    w_risk_return_ratio = qs.stats.risk_return_ratio(weekly_returns)
    w_max_drawdown = qs.stats.max_drawdown(account_balances)
    w_drawdown_details = qs.stats.drawdown_details(w_max_drawdown)
    w_alpha, w_beta = qs.stats.greeks(weekly_returns, SPY / BTC? / cryptoETF?, periods=7)

    return w_sharpe_ratio,w_win_rate,w_volatility,w_sortino_ratio,w_calmar_ratio,w_recovery_factor, \
           w_risk_return_ratio,w_max_drawdown,w_drawdown_details,w_alpha,w_beta

def monthly_backtest(account_balances, monthly_returns, metrics):

    m_sharpe_ratio = qs.stats.sharpe(monthly_returns, periods=30, annualize=False)
    m_win_rate = qs.stats.win_rate(monthly_returns)
    m_volatility = qs.stats.volatility(monthly_returns, periods=30, annualize=False)
    m_sortino_ratio = qs.stats.sortino(monthly_returns)
    m_calmar_ratio = qs.stats.calmar(monthly_returns)
    m_recovery_factor = qs.stats.recovery_factor(monthly_returns)
    m_risk_return_ratio = qs.stats.risk_return_ratio(monthly_returns)
    m_max_drawdown = qs.stats.max_drawdown(account_balances)
    m_drawdown_details = qs.stats.drawdown_details(m_max_drawdown)
    m_alpha, m_beta = qs.stats.greeks(monthly_returns, SPY / BTC? / cryptoETF?, periods = 30)

    return m_sharpe_ratio, m_win_rate, m_volatility, m_sortino_ratio, m_calmar_ratio, m_recovery_factor, \
           m_risk_return_ratio, m_max_drawdown, m_drawdown_details, m_alpha, m_beta

def annual_backtest(account_balances, annual_returns, metrics):

    a_sharpe_ratio = qs.stats.sharpe(annual_returns, periods=365, annualize=True)
    a_win_rate = qs.stats.win_rate(annual_returns)
    a_volatility = qs.stats.volatility(annual_returns, periods=365, annualize=True)
    a_sortino_ratio = qs.stats.sortino(annual_returns)
    a_calmar_ratio = qs.stats.calmar(annual_returns)
    a_recovery_factor = qs.stats.recovery_factor(annual_returns)
    a_risk_return_ratio = qs.stats.risk_return_ratio(annual_returns)
    a_max_drawdown = qs.stats.max_drawdown(account_balances)
    a_drawdown_details = qs.stats.drawdown_details(a_max_drawdown)
    a_alpha, a_beta = qs.stats.greeks(annual_returns, SPY / BTC? / cryptoETF?, periods=365)

    return a_sharpe_ratio, a_win_rate, a_volatility, a_sortino_ratio, a_calmar_ratio, a_recovery_factor, \
           a_risk_return_ratio, a_max_drawdown, a_drawdown_details, a_alpha, a_beta








#def weekly_backtest_all(weekly_returns):
#    """
#    returns.columns = [timedelta, account_balance]
#    Ex
#    index|timedelta|returns
#    0|XXX-XXX-X|10000
#    0|XXX-XXX-X|10010
#    returns: |timedelta|returns
#    timeframe: minute, day,
#    """
#    w_sharpe_ratio = qs.stats.sharpe(weekly_returns, periods=7, annualize=False)
#    w_win_rate = qs.stats.win_rate(weekly_returns, annualize=False)
#    w_volatility = qs.stats.volatility(weekly_returns, periods=7, annualize=False)
#    w_sortino_ratio = qs.stats.sortino()
#    w_calmar_ratio = qs.stats.calmar()
#    w_recovery_factor = qs.stats.recovery_factor()
#    w_risk_return_ratio = qs.stats.risk_return_ratio()
#    w_max_drawdown = qs.stats.max_drawdown()
#    w_drawdown_details = qs.stats.drawdown_details()
#    w_alpha, w_beta = qs.stats.greeks(weekly_returns, SPY/BTC?/cryptoETF?, periods=7)
#
#    # return all performance metrics
 #   return sharpe_ratio


#if __name__ == '__main__':
#    period = 200
#    date = pd.date_range(start=pd.datetime(2020, 1, 1, 9, 30, 0), periods=period, freq="min")
#    returns = np.random.randint(1000, 2000, period)
#
#    returns_pd = pd.DataFrame({'timedelta' : date, 'returns' : returns})
#    print(returns_pd)
