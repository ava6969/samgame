#!/usr/bin/env python3

import finplot as fplt
import pandas as pd
import requests
from io import StringIO
from time import time

import matplotlib
import matplotlib

import numpy as np
# load data and convert date
from qtpy import QtGui, QtCore


end_t = int(time())
start_t = end_t - 12*30*24*60*60 # twelve months
symbol = 'SPY'
interval = '1d'
url = 'https://query1.finance.yahoo.com/v7/finance/download/%s?period1=%s&period2=%s&interval=%s&events=history' % (symbol, start_t, end_t, interval)
r = requests.get(url)
df = pd.read_csv(StringIO(r.text))
df['Date'] = pd.to_datetime(df['Date']).astype('int64') # use finplot's internal representation, which is ns

ax, ax2 = fplt.create_plot('S&P 500 MACD', rows=2, maximize=False, init_zoom_periods=100)

# plot macd with standard colors first
macd = df.Close.ewm(span=12).mean() - df.Close.ewm(span=26).mean()
signal = macd.ewm(span=9).mean()
df['macd_diff'] = macd - signal
fplt.volume_ocv(df[['Date','Open','Close','macd_diff']], ax=ax2, colorfunc=fplt.strength_colorfilter)
fplt.plot(macd, ax=ax2, legend='MACD')
fplt.plot(signal, ax=ax2, legend='Signal')

# change to b/w coloring templates for next plots
fplt.candle_bull_color = fplt.candle_bear_color = '#000'
fplt.volume_bull_color = fplt.volume_bear_color = '#333'
fplt.candle_bull_body_color = fplt.volume_bull_body_color = '#fff'

# plot price and volume
fplt.candlestick_ochl(df[['Date','Open','Close','High','Low']], ax=ax)
hover_label = fplt.add_legend('', ax=ax)
axo = ax.overlay()
fplt.volume_ocv(df[['Date','Open','Close','Volume']], ax=axo)
fplt.plot(df.Volume.ewm(span=24).mean(), ax=axo, color=1)


#######################################################
## update crosshair and legend when moving the mouse ##

def update_legend_text(x, y):
    row = df.loc[df.Date==x]
    # format html with the candle and set legend
    fmt = '<span style="color:#%s">%%.2f</span>' % ('0b0' if (row.Open<row.Close).all() else 'a00')
    rawtxt = '<span style="font-size:13px">%%s %%s</span> &nbsp; O%s C%s H%s L%s' % (fmt, fmt, fmt, fmt)
    hover_label.setText(rawtxt % (symbol, interval.upper(), row.Open, row.Close, row.High, row.Low))

i = 0
def update_crosshair_text(x, y, xtext, ytext):
    global i
    np_data = fplt.screenshot()
    ytext = '%s (Close%+.2f)' % (ytext, (y - df.iloc[x].Close))
    i += 1
    return xtext, ytext


def convert_numpy(ax):
    height = int(ax.size().height())
    width = int(ax.size().width())
    png = QtGui.QImage(width, height, QtGui.QImage.Format.Format_RGBA8888)
    painter = QtGui.QPainter(png)
    painter.setRenderHints(painter.Antialiasing | painter.TextAntialiasing)
    ax.scene().render(painter, QtCore.QRectF(), ax.mapRectToScene(ax.boundingRect()))
    painter.end()

    channels_count = 4

    ptr = png.constBits()
    ptr.setsize(height * width * channels_count)
    return np.frombuffer(ptr, dtype=np.uint8).reshape((height, width, channels_count))

fplt.set_time_inspector(update_legend_text, ax=ax, when='hover')
fplt.add_crosshair_info(update_crosshair_text, ax=ax)

ax0_np = convert_numpy(ax)
ax2_np = convert_numpy(ax2)





