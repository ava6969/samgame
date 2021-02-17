import math
import os
import sys
from multiprocessing import Value, Process, Queue, Pipe
from threading import Thread
import mplfinance as mpf
import pandas as pd
import finplot as fplt
import requests
import time
from broker import Broker, Account


class LivePlotter:
    def __init__(self, pipe, init_data, init_sym, tech_indicators, fps=10):
        super().__init__()
        self.plots = []
        self.tech_indicators = tech_indicators
        self.data, self.watched_sym = init_data, init_sym
        self.ax, self.ax2 = fplt.create_plot('Live Stock Trader', init_zoom_periods=500, maximize=False, rows=2)
        self.live_data = None
        self.update_data()
        self.fps = fps
        self.plots = [fplt.candlestick_ochl(self.live_data[0], ax=self.ax), fplt.volume_ocv(self.live_data[1],
                                                                                            ax=self.ax.overlay()),
                      fplt.plot(self.live_data[2], ax=self.ax, legend='VWAP')]
        self.plots.extend([fplt.plot(ti_d, ax=self.ax2, legend=t)
                           for t, ti_d in zip(tech_indicators, self.live_data[3:])])
        fplt.timer_callback(self.step, 1/fps)  # update (using synchronous rest call) every N seconds
        self.pipe = pipe

    def update_data(self):
        # load data
        df = self.data[self.watched_sym]
        # pick columns for our three data sources: candlesticks and TD sequencial labels for up/down
        data = [df['index open close high low'.split()], df['index open close volume'.split()],
                df['index vwap'.split()]]
        data.extend([df[f'index {t}'.split()] for t in self.tech_indicators])
        self.live_data = data

    def watch(self, data, watched_sym):
        self.data = data
        self.watched_sym = watched_sym

    def update_plot(self):
        for i in range(len(self.plots)):
            self.plots[i].update_data(self.live_data[i])

    def step(self):
        # fplt.timer_callback(self.update, 1/self.fps)  # update (using synchronous rest call) every N seconds
        reader, writer = self.pipe
        writer.close()
        data, watched_sym = reader.recv()
        self.watch(data, watched_sym)
        self.update_data()
        self.update_plot()
        print('step plotter: ')


def update(pipe, broker):
    reader, writer = pipe
    reader.close()
    for i in range(1000):
        data = broker.step()
        writer.send([data, 'AAPL'])
        print('step update: ', i)
        time.sleep(1/10)


if __name__ == '__main__':
    fps = 2
    broker = Broker(Account(account_number=32425, buying_power=1000, cash=1000, daytrade_count=0,
                            daytrading_buying_power=0, equity=1000, initial_margin=1000, last_equity=1000,
                            multiplier=1, pattern_day_trader=False, portfolio_value=0, status='active'))
    ti_s = ['ROC', 'MA']
    data = broker.start(['AAPL', 'GOOGL'], ti_s)
    w_sym = 'AAPL'

    pipe = Pipe()
    p = Process(target=update, args=(pipe, broker))
    p.daemon = True
    p.start()

    plotter = LivePlotter(pipe, data, w_sym, ti_s, fps)
    fplt.show()



