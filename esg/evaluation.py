import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from preproccessing import assets_performance

def fanchart(hist,percentiles):
    length = 60
    x = np.linspace(0,length+1,length+1)
    fig,ax = plt.subplots(figsize=(12,6))
    ax.fill_between(x=x,y1=percentiles[0],y2=percentiles[6], color='blue', alpha=0.1)
    ax.fill_between(x=x,y1=percentiles[1],y2=percentiles[5], color='blue', alpha=0.2)
    ax.fill_between(x=x,y1=percentiles[2],y2=percentiles[4], color='blue', alpha=0.3)
    ax.plot(percentiles[2],color='blue')

def describe_scenarios_vertically(scenarios: pd.DataFrame):
    """"
    Returns mean, standard devation, percentiles of scenarios performence: 
    annualised mean, std, skew, kurtosis, sharp ratio, maxdrawdown
    """
    return assets_performance(scenarios).T.describe()
    