import glob
import pickle
from typing import List
import pandas as pd
import numpy as np
import datetime
import pandas_market_calendars as mcal
from tqdm.notebook import tqdm

NY = 'America/New_York'
START = pd.Timestamp('2004-01-02 09:30', tz=NY)
END = pd.Timestamp('2020-12-31 15:00', tz=NY)


class Dataloader:
    def __init__(self, start=None, end=None):
        all_minute_loc = glob.glob(f'C:\\Users\\Dewe\\samgame\\datasets\\minute\\*')
        self.sym_dict = {s.split('\\')[-1].split('_')[0]: s for s in all_minute_loc}
        if start and end:
            nyse = mcal.get_calendar('NYSE')
            early = nyse.schedule(start_date=start, end_date=end)
            full_date_range = mcal.date_range(early, frequency='1min').tz_convert(NY)
            self.full_date_range = full_date_range
            with open(f'C:\\Users\\Dewe\\samgame\\datasets\\dates_{start.year}_{end.year}.pkl', 'wb') as pkl:
                pickle.dump(full_date_range, pkl)
        else:
            with open(f'C:\\Users\\Dewe\\samgame\\datasets\\dates_2004_2020.pkl', 'rb') as pkl:
                self.full_date_range = pickle.load(pkl)

    def load(self, stocks: List[str]):
        data_set = [pd.read_pickle(self.sym_dict[s]) for s in stocks]
        for i, df in tqdm(enumerate(data_set)):
            df['day_start'] = df.index.date
            df.drop_duplicates(inplace=True)
            dates = self.full_date_range[self.full_date_range > df.index[0]]
            new_df = df.reindex(dates, method='nearest')
            data_set[i] = new_df
        return data_set

    def generate_week_data(self, df, start=None):
        if start is None:
            start = np.random.choice(df.day_start.unique())
        days = [start + datetime.timedelta(i) for i in range(5)]
        episode = df.loc[df.day_start.isin(days)]
        return episode

    def generate_day_data(self, df, date=None):
        if date is None:
            date = np.random.choice(df.day_start.unique())
        episode = df[df.day_start == date]
        return episode


if __name__ == '__main__':
    dl = Dataloader()
    data = dl.load(['A', 'AAP', 'AAPL'])

    week_episode = dl.generate_week_data(data[0])
    print(week_episode)
    day_episode = dl.generate_day_data(data[0])
    print(day_episode)

    start = data[0].day_start[0]

    week_episode = dl.generate_week_data(data[0], start)
    print(week_episode)
    day_episode = dl.generate_day_data(data[0], start)
    print(day_episode)
