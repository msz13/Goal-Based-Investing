import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from preproccessing import assets_performance
import seaborn as sns

def fanchart(hist,scenarios):
    percentiles = np.percentile(scenarios,[1,5,25,50,75,95,99],axis=0)
    hist_len = hist.shape[0]
    n_steps = scenarios.shape[1] 
    x1 = np.arange(-hist_len+1,1)
    x2 = np.arange(0,n_steps)
    
    fig,ax = plt.subplots(figsize=(12,6))
    ax.plot(x1, hist,color='red')
    ax.plot(x2, percentiles[3], color='blue')
      
    for i in range(1,4):
        ax.fill_between(x=x2,y1=percentiles[i-1],y2=percentiles[-i], color='blue', alpha=i/10)

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

def sample_paths(scenarios, number_of_paths=10):
    ax,fig = plt.subplots(figsize=(12,4))
    number_of_scenarios = scenarios.shape[0]
    for i in np.random.randint(0,number_of_scenarios,number_of_paths):
        sns.lineplot(data=scenarios[i])
    