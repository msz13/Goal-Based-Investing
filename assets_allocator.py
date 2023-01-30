import numpy as np
import pandas as pd

class AssetsAllocator:
    def __init__(self, portfolios: pd.DataFrame) -> None:
        self.portfolios = portfolios
        

    def expectedAllocation(self, goals):
        self.goals = goals
        assets = self.portfolios.columns
        goals = self.goals['Nazwa'].tolist()
        return pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=assets, columns=goals)
    

def generatePortfolio(returns, initial,payment):
    portfolioValue = initial
    
    for r in returns:
        portfolioValue = (portfolioValue + payment)* (1+r)

    return np.round(portfolioValue,2)

def probabilityOfGoal(goal, values: np.array):
    # return np.count_nonzero(values >= goal)/len(values)
    return [np.percentile(25,values), np.percentile(50, values), np.percentile(75,values)]


def allocateGoals(goalsdf,initialAllocation: np.array, initialValue, payments):
    cashflows = np.arange()
    for t in range(goalsdf['Czas_trwania'][0]):
       cashflows[t] = initialAllocation*payments
    return cashflows.T
    
