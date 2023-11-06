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

def percentile_summary(scenarios, data_freq=1):
    percentiles = [1,5,25,50,75,95,99]
    perc = np.percentile(scenarios,percentiles,axis=0)
    years = np.array([1,2,3,5,10,15,20,50]) *data_freq


    perc_summary = pd.DataFrame({})
   
    for year in years:
        perc_summary[year] = perc[:,year]
   
    perc_summary.index = percentiles
    return perc_summary


def describe_scenarios_vertically(scenarios: pd.DataFrame, data_freq):
    """"
    Returns mean, standard devation, percentiles of scenarios performence: 
    annualised mean, std, skew, kurtosis, sharp ratio, maxdrawdown
    """
    return assets_performance(scenarios, data_freq).T.describe()
    