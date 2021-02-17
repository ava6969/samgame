import os
import time
from collections import defaultdict

from PIL import Image

from broker import Broker, Account
import numpy as np
from preprocessor import make_orders
from common import Order


def random_policy(obs):
    tickers, prices = obs
    # select random ticker
    sym_to_watch = np.random.choice(tickers)
    qty = np.random.choice([-2, -1, 0, 1, 2], size=len(tickers))
    return sym_to_watch, qty


if __name__ == '__main__':

    broker = Broker(Account(account_number=32425,  cash=1000, daytrade_count=0,
                            equity=1000, last_equity=0, pattern_day_trader=False,
                            stocks_owned=defaultdict(lambda: 0)))

    print(broker)
    tickers = ['AAPL', 'GOOGL']
    frame = broker.start(tickers, 'MA EMA ATR ROC')
    img = broker.get_view(frame, np.random.choice(tickers), True)
    # action_space

    for i in range(1000):
        sym_to_watch, qty = random_policy([tickers, frame])

        # create orders
        orders = make_orders(tickers, qty, frame)

        # place order / update account
        statuses = broker.place_orders(orders)

        # gets next frame
        frame = broker.step()
        print(broker.account, 'action:', qty, 'statuses:', statuses)
        img = broker.get_view(frame, sym_to_watch, True)  # currently really slow

        # print(broker.account, end='\r', flush=True)
        # time.sleep(2)
