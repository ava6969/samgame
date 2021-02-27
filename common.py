from collections import namedtuple, defaultdict
from typing import List
import dataclasses

"""
cash -> Cash balance
pattern_day_trader -> boolean: Whether or not the account has been flagged as a pattern day trader
equity -> float: Cash + long_market_value + short_market_value
last equity -> Equity as of previous trading day at 16:00:00 ET
date_trade_count -> The current number of daytrades that have been made in the last 5 trading days (inclusive of today)
"""


@dataclasses.dataclass
class Account:
    account_number: int
    cash: float
    daytrade_count: int
    equity: int
    last_equity: int
    stocks_owned: dict
    pattern_day_trader: bool

    def flat(self):
        pv = [self.cash,
              self.daytrade_count,
              self.equity,
              self.last_equity,
              self.pattern_day_trader,
              ]
        pv.extend(list(self.stocks_owned.values()))
        return pv


"""
constraints
asset_class = us_equity
qty always filled
only market orders
only day trades
no extended hours
https://alpaca.markets/docs/trading-on-alpaca/orders/
"""
Order = namedtuple('Order', 'symbol, qty, side, market_value')

if __name__ == '__main__':
    order = Order('AAPL', 20, 'buy', 'success')
    print(order.symbol)
