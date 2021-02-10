import pandas as pd
from ray.pickle5_files import pickle5
import numpy as np


class BackTester:
    def __init__(self, returns_pd:pd.DataFrame, timeframe):
        """
        returns: |timedelta|returns
        timeframe: minute, day,
        """
        self.returns = returns
        self.tf = timeframe

    def backtest(self) -> pd.DataFrame :
        pass

    def sharpe_ratio(self) -> float:
        returns = self.returns['returns']
        change = returns.pct_change(1)
        mu = change.mean()
        sigma = change.std()
        sr = mu/sigma
        return sr

    def max_drawndown(self) -> float:
        pass


if __name__ == '__main__':
    period = 200
    date = pd.date_range(start=pd.datetime(2020, 1, 1, 9, 30, 0), periods=period, freq="min")
    returns = np.random.randint(1000, 2000, period)

    returns_pd = pd.DataFrame({'timedelta' : date, 'returns' : returns})
    print(returns_pd)

    tester = BackTester(returns_pd, 'min')