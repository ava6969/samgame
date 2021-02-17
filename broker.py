import os
import time
from collections import defaultdict
from typing import List
from common import Account, Order
from dataloader import Dataloader
from preprocessor import add_tech_ind
import mplfinance as mpf
import matplotlib
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image


matplotlib.use('TKAgg')


class Broker:
    """
     https://alpaca.markets/docs/api-documentation/api-v2/account/
    """

    def __init__(self, account: Account, max_limit=500):
        # attributes
        self.account = account
        self.dataloader = Dataloader()
        self.live_data = dict()
        self.index = 0
        self.max_limit = max_limit
        self.tickers = []
        self.tech_indicators = []
        self.max_idx = None
        self.current_day = None

    def start(self, tickers: List[str], tech_indicators=None, debug=True):
        """
        starts new session
        :param debug:
        :param tickers:
        :param end:
        :param start:
        :param tech_indicators:
        :return:
        """
        data = self.dataloader.load(tickers)

        if tech_indicators:
            for t, df in zip(tickers, data):
                self.live_data[t] = add_tech_ind(df, tech_indicators, debug).reset_index()

        self.index = self.max_limit
        self.tickers = tickers
        if isinstance(tech_indicators, str):
            self.tech_indicators = tech_indicators.split()
        else:
            self.tech_indicators = tech_indicators
        self.max_idx = data[tickers[0]].index[-1]
        return self.peek()

    def peek(self, step_amount=1):
        """
        dont update time just check next
        :param step_amount:
        :return:
        """
        idx = self.index + step_amount
        frame_stack = {t: df[idx - self.max_limit - 1: idx] for t, df in self.live_data.items()}
        # self.current_day = frame[self.index]['start_day']
        return frame_stack

    def next(self, step_amount=1):
        """
        step to next timestep[1 min, 1 week, ]
        :param step_amount how many steps to go into data
        :return:
        """
        frame_stack = self.peek(step_amount)
        self.index += 1
        return frame_stack

    def place_orders(self, orders: List[Order]):
        self.account.equity = 0
        return [self._place_order(order) for order in orders]

    def _place_order(self, order: Order):

        if order.side == 'buy':
            status = self.buy_stocks(order)
        elif order.side == 'sell':
            status = self.sell_stocks(order)
        else:
            raise ValueError('position side can only be buy and sell')
        self.account.equity += self.account.stocks_owned[order.symbol] * order.market_value
        return status

    def create_data(self,  tickers: List[str], data, tech_indicators=None, debug=True ):
        if tech_indicators:
            for t, df in zip(tickers, data):
                self.live_data[t] = add_tech_ind(df, tech_indicators, debug).reset_index()

    def update_tech_indicators(self, tech_indicators):
        """
        updates all live data
        :param tech_indicators:
        :return:
        """
        data = { t: self.live_data[t].drop(self.tech_indicators, axis=1) for t in self.tickers}
        self.create_data(self.tickers, data, tech_indicators, False)

    def get_view(self, frame, sym_to_watch, view=True):
        extras = ['vwap']
        extras.extend(self.tech_indicators)
        df = frame[sym_to_watch].set_index('index')

        ap = [mpf.make_addplot(df[t], ylabel=t) for t in extras]
        buf = io.BytesIO()
        fig, _ = mpf.plot(df, type='candle', volume=True, addplot=ap,
                          axtitle=sym_to_watch, # style='yahoo',
                          figsize=(9, 6),
                          savefig=dict(fname=buf,  format='raw'), returnfig=True)

        buf.seek(0)
        np_img = np.frombuffer(buf.getvalue(), dtype=np.uint8).reshape(int(fig.bbox.bounds[3]),
                                                                       int(fig.bbox.bounds[2]), -1)
        buf.close()

        if view:
            plt.imshow(np_img)
            plt.show(block=False)
        return np_img

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

    def done(self):
        return self.index == self.max_idx

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
    broker.start(['AAPL', 'GOOGL'], 'MA EMA ATR ROC')

    for i in range(10):
        frame = broker.step()
        print(frame, end='\r', flush=True)
        time.sleep(0.33)
        os.system('cls')
