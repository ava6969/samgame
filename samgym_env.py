import numpy as np
from collections import defaultdict
from preprocessor import make_orders
import gym
from broker import Broker, Account
import backtester as bt
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
ERASELINE = '\x1b[2K'
# matplotlib.use('TKAgg')

OHCLVV = 6


class SAMGameGym(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, env_config):
        account = Account(000000, env_config['initial_cash'], 0, env_config['initial_cash'], env_config['initial_cash'],
                          dict(), False)
        max_limit = env_config['window']
        self.resized_image_width = 100
        self.resized_image_height = 100
        image_channel = 4
        self.log_every = env_config['log_every']
        self.broker = Broker(account, max_limit=max_limit)
        self.all_tickers = self.broker.all_tickers
        self.n_symbols = env_config['n_symbols']
        self.tech_indicators = env_config['tech_indicators']
        length = OHCLVV + len(self.tech_indicators.split())
        self.use_image = env_config['use_image']

        if self.use_image:
            self.observation_space = gym.spaces.Dict(
                {'data': gym.spaces.Box(-np.inf, np.inf, (self.n_symbols, max_limit, length)),
                 'images': gym.spaces.Box(-np.inf, np.inf, (self.n_symbols,
                                                            self.resized_image_height,
                                                            self.resized_image_width, image_channel)),
                 'privates': gym.spaces.Box(-np.inf, np.inf, (5 + self.n_symbols * 2,))
                 })
        else:
            self.observation_space = gym.spaces.Dict(
                {'data': gym.spaces.Box(-np.inf, np.inf, (self.n_symbols, max_limit, length)),
                 'privates': gym.spaces.Box(-np.inf, np.inf, (5 + self.n_symbols * 2,))
                 })

        self.action_space = gym.spaces.MultiDiscrete([env_config['bins']] * self.n_symbols)
        self.current_tickers = None
        self.qty_val = np.linspace(-env_config['max_shares'], env_config['max_shares'], env_config['bins'])
        self.images = None
        self.refresh_data = True

    def extract_data(self, df_dict, t, statuses, images=None):
        columns = ['open', 'high', 'low', 'close', 'volume', 'vwap']
        columns.extend(self.tech_indicators.split())
        dfs = [df_dict[tick][columns] for tick in t]
        frame = np.array(dfs)
        privates = self.broker.account.flat()  # fix
        privates.extend(statuses)

        # if images is not None:
        #     images = np.array(list(
        #         self.broker.get_view(frame=df_dict,
        #                              resize=(self.resized_image_width, self.resized_image_height)).values()))
        #
        #     self.images = images[0, :, :, :3]
        #     return frame, images, privates
        # else:
        return frame, privates

    def render(self, mode='rgb_array'):
        if self.use_image:
            assert self.images is not None
            plt.imshow(self.images)
            # plt.pause(0.1)
            plt.show(block=False)

    def reset(self):
        if self.refresh_data:
            t = np.random.choice(self.all_tickers, self.n_symbols)
            self.current_tickers = t
            self.broker.account.stocks_owned = {tick: 0 for tick in t}
            df_dict = self.broker.start(t, self.use_image, self.tech_indicators)
            self.refresh_data = False
        else:
            df_dict = self.broker.next(use_image=False)[0]

        # if self.use_image:
        #     frame, images, privates = self.extract_data(df_dict, t, [1] * self.n_symbols)
        #     return {'data': frame, 'images': images, 'privates': np.array(privates, dtype=np.float)}
        # else:
        frame, privates = self.extract_data(df_dict, self.current_tickers, [1] * self.n_symbols)
        return {'data': frame, 'privates': np.array(privates, dtype=np.float)}

    def step(self, action):
        res = self.broker.next(self.use_image)
        orders = make_orders(self.current_tickers, action, res[0], self.qty_val)
        statuses = self.broker.place_orders(orders)

        reward = self.broker.equity - self.broker.last_equity
        done = self.broker.done()
        self.refresh_data = done

        # print(res[-1], flush=True, end='\r')
        if res[2]: # 2 - new week
            #todo: bug here
            print(ERASELINE , self.broker.account, '\norders:', orders, '\nstatuses:', statuses, flush=True)
            weekly_df = pd.DataFrame(self.broker.daily_account_balances).set_index('timestamp')
            weekly_perf = bt.weekly_backtest_with(weekly_df, [bt.BTMetrics.SHARPE_RATIO,
                                                              bt.BTMetrics.MAX_DRAWDOWN])
            reward += sum(weekly_perf.values())
            done = True

        if self.use_image:
            next_frame, next_images, next_privates = self.extract_data(res[0], self.current_tickers, statuses)
            # todo: add order statuses to observation private variables
            return dict(data=next_frame, images=next_images, privates=np.array(next_privates, dtype=np.float)), \
                   reward, done, {}
        else:
            next_frame, next_privates = self.extract_data(res[0], self.current_tickers, statuses)
            # todo: add order statuses to observation private variables

            return dict(data=next_frame, privates=np.array(next_privates, dtype=np.float)), \
                   reward, done, {}

