import os
import time
from collections import defaultdict
from typing import List
from common import Account, Order
from datagenerator import DataGenerator


class Broker:
    """
     https://alpaca.markets/docs/api-documentation/api-v2/account/
    """

    def __init__(self, account: Account, max_limit=500):
        # attributes
        self.account = account
        self.dataloader = DataGenerator()
        self.index = 0
        self.max_limit = max_limit
        self.tickers = []
        self.tech_indicators = []
        self.max_idx = None
        self.current_day = None

        self.minute_account_balances = []
        self.daily_account_balances = []
        self.all_tickers = self.dataloader.all_syms
        self.generator = None

    def start(self, tickers: List[str], use_image, tech_indicators=None, debug=True):
        """
        starts new session
        :param debug:
        :param tickers:
        :param end:
        :param start:
        :param tech_indicators:
        :return:
        """
        self.dataloader.load(tickers)
        self.dataloader.preprocess(tech_indicators, debug)
        self.generator = self.dataloader.generate_weekly(use_image)
        return list(iter(next(self.generator)))[0]

    def capture_equity(self, ts):
        captured = {'timestamp': ts, 'equity': self.account.equity}
        return captured

    @property
    def last_equity(self):
        return self.account.last_equity

    @property
    def equity(self):
        return self.account.equity

    def next(self, use_image):
        """
        step to next timestep[1 min, 1 week, ]
        :param step_amount how many steps to go into data
        :return:
        """
        res = list(iter(next(self.generator)))
        ts = res[-1]
        self.minute_account_balances.append(self.capture_equity(ts))

        if res[1]: # 1 for new day
            self.daily_account_balances.append(self.capture_equity(ts))

        return res

    def place_orders(self, orders: List[Order]):
        self.account.last_equity = self.account.equity
        statuses = []
        if len(orders) > 0:
            self.account.equity = 0
            statuses = [self._place_order(order) for order in orders]
        self.account.equity += self.account.cash
        return statuses

    def _place_order(self, order: Order):

        if order.side == 'buy':
            status = self.buy_stocks(order)
        elif order.side == 'sell':
            status = self.sell_stocks(order)
        else:
            status = True
        self.account.equity += self.account.stocks_owned[order.symbol] * order.market_value
        return status

    def done(self):
        return self.dataloader.done

    def buy_stocks(self, order):
        """
        update account
        :param order:
        :return:
        """
        value = order.qty * order.market_value
        if self.account.cash > value:
            status = True
            self.account.cash -= value
            self.account.equity += value
            self.account.stocks_owned[order.symbol] += 1
        else:
            status = False

        return status

    def sell_stocks(self, order):
        """
        update account
        :param order:
        :return:
        """
        value = order.qty * order.market_value
        if self.account.stocks_owned[order.symbol] >= order.qty:
            status = True
            self.account.cash += value
            self.account.equity -= value
            self.account.stocks_owned[order.symbol] -= 1
        else:
            status = False

        return status

    def __repr__(self):
        return self.account.__repr__()


if __name__ == '__main__':

    broker = Broker(Account(account_number=32425, cash=1000, daytrade_count=0, equity=1000, last_equity=0,
                            pattern_day_trader=False, stocks_owned=defaultdict(lambda: int)))
    print(broker)
    broker.start(['AAPL', 'GOOGL'], False,
                 'MA EMA ATR ROC')

    for i in range(10):
        frame = broker.next(False)
        print(frame, end='\r', flush=True)
        time.sleep(0.33)
        os.system('cls')
