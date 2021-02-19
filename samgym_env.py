import numpy as np
from collections import defaultdict
from preprocessor import make_orders
import gym
from broker import Broker, Account
import backtester as bt
import pandas as pd


class SAMGameGym(gym.Env):
    def __init__(self, env_config):
        account = Account(000000, env_config['initial_cash'], 0, 0, 0, defaultdict(lambda: int), False)

        self.broker = Broker(account, 400)
        self.all_tickers = env_config['all_tickers']
        self.n_symbols = env_config['n_symbols']
        self.tech_indicators = env_config['tech_indicators']

        length = 6 + self.tech_indicators
        self.observation_space = gym.spaces.Dict({'data': gym.spaces.Box(-np.inf, np.inf, (self.n_symbols, 400, length)),
                                                  'imgs': gym.spaces.Box(-np.inf, np.inf, (self.n_symbols, 84, 84, 3)),
                                                   })

        self.action_space = gym.spaces.MultiDiscrete([7] * self.n_symbols)

    def reset(self):
        t = np.random.choice(self.all_tickers, self.n_symbols)
        frame = np.array(list(self.broker.start(t, self.tech_indicators).values()))
        images = [self.broker.get_view(frame, tick) for tick in t]

        return {'data' : frame, 'images' : images}

    def end_of_week(self):
        return True

    def step(self, action):

        next_frame = self.broker.next()
        orders = make_orders(self.all_tickers, action, next_frame)
        self.broker.place_orders(orders)

        if self.end_of_week():
            weekly_perf = bt.weekly_backtest_with(pd.DataFrame(), [bt.BTParameters.SHARPE_RATIO,
                                                                   bt.BTParameters.MAX_DRAWDOWN])

        reward = sum(perf.values())

        done = self.broker.done()

        return next_frame, reward, done, {}
