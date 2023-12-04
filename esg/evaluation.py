import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from preproccessing import assets_performance
import seaborn as sns
import preproccessing as prep
from IPython.display import display

def fanchart(scenarios, hist=None):
    print('Fanchart')
    percentiles = np.percentile(scenarios, [10, 15, 25, 50, 75, 85, 90],axis=0)    
    n_steps = scenarios.shape[1]     
    x2 = np.arange(0,n_steps)
    fig,ax = plt.subplots(figsize=(12,6))

    if (hist is not None):
        hist_len = hist.shape[0]
        x1 = np.arange(-hist_len+1,1)
        ax.plot(x1, hist,color='red')
        
    
    ax.plot(x2, percentiles[3], color='blue')
      
    for i in range(1,4):
        ax.fill_between(x=x2,y1=percentiles[i-1],y2=percentiles[-i], color='blue', alpha=i/10)
    
    plt.show()

def percentile_summary(scenarios, data_freq=1, years=np.array([1,3,5,10])):
    print('Percentaile summary')
    percentiles =  [10, 15, 25, 50, 75, 85, 90]
    perc = np.percentile(scenarios,percentiles,axis=0)
    periods = years * data_freq

    perc_summary = pd.DataFrame({})
   
    for period in periods:
        perc_summary[period] = perc[:,period]
   
    perc_summary.index = percentiles
    return perc_summary


def describe_scenarios_vertically(scenarios: pd.DataFrame, data_freq):
    """"
    Returns mean, standard devation, percentiles of scenarios performence: 
    annualised mean, std, skew, kurtosis, sharp ratio, maxdrawdown
    """
    print("Scenarios summary stats")
    return assets_performance(scenarios, data_freq).describe()

def sample_paths(scenarios: pd.DataFrame, number_of_paths=7):
    print('Sample paths')
    fig, ax = plt.subplots(figsize=(12,4))
    number_of_scenarios = scenarios.shape[0]
    for i in np.random.randint(0,number_of_scenarios,number_of_paths):
        #sns.lineplot(data=scenarios.iloc[i], ax=ax,x=scenarios.columns)
        scenarios.iloc[i].plot(ax=ax)

    plt.show()

def histplot(scenarios, hist):
    fig, ax = plt.subplots()
    sns.histplot(data=scenarios.to_numpy().reshape(scenarios.shape[0]*scenarios.shape[1]),stat='probability', ax=ax, bins=64)
    sns.histplot(data=hist, ax=ax, stat='probability', color='orange', bins=64)

def plot_returns(scenarios, hist):
    fix, axs = plt.subplots(1,2, figsize=(12,6))
    scenario = np.random.randint(scenarios.shape[1])
    hist.plot(ax=axs[0],title='Historical returns')
    scenarios.iloc[:,scenario].plot(ax=axs[1], title='Scenario')
    plt.show()

def show_scenarios_evaluation(scenarios, hist):
    """
    hist - historical retursn to compere histogram with scenarios
    """
    print(sample_paths(scenarios))

    scenarios_cum_returns = scenarios/100
    print(fanchart(scenarios_cum_returns))
    
    display(percentile_summary(scenarios_cum_returns,data_freq=12,years = np.array([1,3,5,10,20,25])))

    scenarios_returns = prep.log_returns(scenarios.T)
    display(describe_scenarios_vertically(scenarios_returns,'m'))

    plot_returns(scenarios_returns, hist)
    histplot(scenarios_returns,hist)



    
    