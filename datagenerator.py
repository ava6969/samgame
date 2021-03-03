import glob
import math
import pickle
from typing import List
import pandas as pd
import numpy as np
import datetime
import pandas_market_calendars as mcal
from tqdm.notebook import tqdm
import mplfinance as mpf
from preprocessor import add_tech_ind
import io
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')
NY = 'America/New_York'
START = pd.Timestamp('2009-01-02 09:30', tz=NY)
END = pd.Timestamp('2020-12-31 15:00', tz=NY)
ERASE_LINE = '\x1b[2K'

fig = mpf.figure(figsize=(12, 9))


class DataGenerator:
    def __init__(self, start=None, end=None):
        all_minute_loc = glob.glob(f'/home/dewe/samgame/datasets/minute/*')
        self.sym_dict = {s.split('\\')[-1].split('_')[0]: s for s in all_minute_loc}
        if start and end:
            nyse = mcal.get_calendar('NYSE')
            early = nyse.schedule(start_date=start, end_date=end)
            full_date_range = mcal.date_range(early, frequency='1min').tz_convert(NY)
            self.full_date_range = full_date_range
            with open(f'/home/dewe/samgame/datasets/dates_{start.year}_{end.year}.pkl', 'wb') as pkl:
                pickle.dump(full_date_range, pkl)
        else:
            with open(f'/home/dewe/samgame/datasets/dates_2004_2020.pkl', 'rb') as pkl:
                self.full_date_range = pickle.load(pkl)

        self.all_syms = list(self.sym_dict.keys())
        self.live_data = {}
        self.tech_indicators = None
        self.done = False

    def load(self, stocks: List[str]):
        # make sure date is clipped based on smallest data
        data_set = {s: pd.read_pickle(self.sym_dict[s]) for s in stocks}
        least_start, least_end = self.full_date_range[0], self.full_date_range[-1]

        for t, df in tqdm(data_set.items()):
            df['day_start'] = df.index.date
            df.drop_duplicates(inplace=True)
            dates = self.full_date_range[self.full_date_range > df.index[0]]
            new_df = df.reindex(dates, method='nearest')
            data_set[t] = new_df
            if df.index[0] > least_start:
                least_start = df.index[0]
            if df.index[-1] < least_end:
                least_end = df.index[-1]

        for s in stocks:
            data_set[s] = data_set[s].loc[(data_set[s].index > least_start) & (data_set[s].index < least_end)]

        self.live_data = data_set
        return data_set

    def preprocess(self, tech_indicators, debug=True):
        if tech_indicators is None:
            return self.live_data

        for t, df in self.live_data.items():
            df = add_tech_ind(df, tech_indicators, debug).rename_axis('timestamp')
            self.live_data[t] = df.reset_index()
        self.tech_indicators = tech_indicators.split()

    @staticmethod
    def fig2data():
        """
        @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
        @param fig a matplotlib figure
        @return a numpy 3D array of RGBA values
        """
        # draw the renderer
        fig.canvas.draw()

        # Get the RGBA buffer from the figure
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        buf = np.roll(buf, 3, axis=2)
        return buf

    def _get_view(self):

        return []
        assert len(self.live_data.items()) <= 4
        extras = ['vwap']
        if self.tech_indicators:
            extras.extend(self.tech_indicators)  # tech_indicators

        s = mpf.make_mpf_style(base_mpf_style='classic', rc={'figure.facecolor': 'lightgray'})
        for i, t in enumerate(self.live_data.keys(), start=1):
            ax2 = fig.add_subplot(2, 2, i, style=s)
            df = self.live_data[t].set_index('timestamp')
            ap = [mpf.make_addplot(df[t], ylabel=t) for t in extras]
            mpf.plot(df, type='candle', # addplot=ap, ax=ax2,
                     # panel_ratios=(4, 1),
                     # volume_panel=1,
                     axtitle=t)
        np_img = self.fig2data(fig)

        return np_img

    def generate_weekly(self, use_image=False):
        assert len(self.live_data.items()) > 0
        df_list = list(self.live_data.values())
        max_frame = len(df_list[0])  # 0 or any works
        day_len = 200 # todo: unify with yaml
        index = day_len
        self.done = False

        while index < max_frame:
            index += 1
            curr_data = df_list[0]
            ts_prev = curr_data.iloc[index - 1].timestamp
            ts_curr = curr_data.iloc[index].timestamp
            new_day = ts_curr.day != ts_prev.day
            new_week = df_list[0].iloc[index].timestamp.hour == 9 and df_list[0].iloc[index].timestamp.minute == 31 \
                       and ts_curr.weekday() == 0
            new_month = ts_curr.day != ts_prev.day and ts_curr.date().day == 1
            if use_image:
                yield {t: df[index - day_len: index] for t, df in self.live_data.items()}, new_day, new_week, new_month, self._get_view()
            else:
                yield {t: df[index - day_len: index] for t, df in self.live_data.items()}, new_day, new_week, new_month, ts_curr
        self.done = True

    def generate_daily(self, df, date=None):
        if date is None:
            date = np.random.choice(df.day_start.unique())
        episode = df[df.day_start == date]
        return episode


if __name__ == '__main__':
    dl = DataGenerator()
    dl.load(['A', 'AAP', 'AAPL'])
    dl.preprocess('MA EMA ATR ROC')

    for week_df, newday, new_week, _, img in dl.generate_weekly():
        print(week_df['A'].tail(), end='\r', flush=True)
        if new_week:
            print(ERASE_LINE, week_df, flush=True)
