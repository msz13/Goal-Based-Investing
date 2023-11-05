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

def calculate_returns(series, freg='m'):
    shift = {
        'm': 1,
        'q': 4,
        'y': 12
    }
    return series.pct_change(shift[freg]).dropna()

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

def annualised_mean(returns):
    return returns.mean() * 12

def annualised_sigma(returns):
    return returns.std() * 12**(1/2)

def foreign_asset_std(asset_std, currency_std, corr):
    variance = asset_std**2 + currency_std**2 + 2*corr*asset_std*currency_std
    return np.sqrt(variance)

def sharp_ratio(returns, risk_free_rate=0.03):
    return (annualised_mean(returns)-risk_free_rate)/returns.std()

def max_drawdown(returns):
    wealth_index = np.exp(returns).cumprod()
    rolling_max = wealth_index.cummax()
    drawdown = wealth_index/rolling_max -1
    return drawdown.min()
    
def assets_performance(returns: pd.DataFrame):
    #return returns.agg(['mean', 'std', 'median', 'skew', 'kurtosis', sharp_ratio, max_drawdown])
    return returns.agg([annualised_mean, annualised_sigma, 'skew', 'kurtosis', sharp_ratio, max_drawdown])

