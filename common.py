from collections import namedtuple
from typing import List


Account = namedtuple('Account', 'account_number, buying_power, cash, daytrade_count, daytrading_buying_power, equity, '
                                'id, initial_margin, '
                                'last_equity, last_maintenance_margin, long_market_value, maintenance_margin,'
                                'multiplier, pattern_day_trader, portfolio_value, regt_buying_power,'
                                'short_market_value, shorting_enabled, sma, status, trade_suspended_by_user, '
                                'trading_blocked')

"""
constraints
asset_class = us_equity
qty always filled
only market orders
only day trades
no extended hours
"""
Position = namedtuple('Position', 'symbol, qty, side, market_value')

"""
constraints
asset_class = us_equity
qty always filled
only market orders
only day trades
no extended hours
https://alpaca.markets/docs/trading-on-alpaca/orders/
"""
Order = namedtuple('Order', 'symbol, qty, side, status')


if __name__ == '__main__':

    order = Order('AAPL', 20, 'buy', 'success')
    print(order.symbol)

    pos = Position('AAPL', 20, 'buy', 100)
    print(pos.side)

    # order = ('AAPL', 20, 'buy', 'success')
    # print(order[0])

