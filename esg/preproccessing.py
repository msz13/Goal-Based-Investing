import requests
from io import StringIO
import pandas as pd
import yfinance as yf
import numpy as np

def loadStooqData(ticker: str,start, frequency='d'):
    url = f'https://stooq.pl/q/d/l/?s={ticker}&i={frequency}'
    response = requests.get(url)
    data = pd.read_csv(StringIO(response.text))
    data.set_index('Data', inplace=True)
    data = data[start:]
    return data['Zamkniecie']


def loadYahooData(ticker, start, frequency = '1d'):
    ticker  = yf.Ticker(ticker)
    hist = ticker.history(start=start,interval=frequency)
    price = hist['Close']
    return price 

def log_returns(series, freg='m'):
    shift = {
        'm': 1,
        'q': 4,
        'y': 12
    }
    return np.log(series/series.shift(shift[freg])).dropna()


def corr_to_cov(corr_matrix, std_devs):
    """
    Converts vector of standard deviations and correlation matrix into covariance matrix
    """
    D = np.diag(std_devs)
    return D @ corr_matrix @ D