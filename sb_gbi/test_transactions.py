from dataclasses import dataclass
from unittest.mock import Mock
import numpy as np
import numpy.testing as npt
import pytest

@dataclass
class Transaction:
    delta_shares: np.array
    outflows: int = 0
 

def transactions_dep(inflow, shares_owned, assets_weights, prices, goal = (0,0)):
    
    goal_target = goal[0] 
    goal_max_outflow_percent = goal[1]
    cashflow = inflow - goal_target
    current_value = np.sum(shares_owned * prices, 1) 
    max_outflow_value = current_value * goal_max_outflow_percent
    delta_shares = np.zeros_like(shares_owned)

    if(cashflow >0 ):
        delta_shares = np.fix((cashflow*assets_weights) / prices)
    else:
        delta_shares = np.where(np.abs(cashflow) <= max_outflow_value, np.fix((cashflow*assets_weights) / prices), -np.fix(shares_owned * goal_max_outflow_percent))

    outflows = np.round(np.abs(np.sum(delta_shares * prices,axis=1)),2)
    
    return Transaction(delta_shares, outflows)

def transactions(inflow, shares_owned, assets_weights, prices, goal = (0,0)):
    
    goal_target = goal[0] 
    goal_max_outflow_percent = goal[1]
    current_assets_value = shares_owned * prices
    current_value = np.sum(current_assets_value, 1) 
    expected_value = current_value + inflow - goal_target
    #max_outflow_value = current_value * goal_max_outflow_percent
    
    delta_value = expected_value.reshape((2,1)) * assets_weights - current_assets_value
    
    delta_shares =  np.fix(delta_value / prices)

    outflows = np.round(np.abs(np.sum(delta_shares * prices,axis=1)),2)
    
    return Transaction(delta_shares, outflows)

def test_should_buy_assets():

    assets_weights = np.array([0.6,0.4])
    prices = np.array([[30, 50],[30, 50]])
    result = transactions(10000, np.array([[0, 0],[0, 0]]), assets_weights,prices)

    expected_result = Transaction(np.array([[200, 80],[200, 80]]))

    npt.assert_array_equal(result.delta_shares, expected_result.delta_shares)   

'''
TODO
- multiple goals, gaol reached or not 
'''

ids = ['max withrowal 100, goal reached',
       'max withrowal 100, goal not reached', 
       'max withrowal less than 100, goal reached',
       'max withrowal less than 100, goal not reached']

test_data = [((14000,1), [[-255, -88],[-285, -89]], [13963.16, 13928.02]),
             ((16500,1), [[-285, -107],[-292, -109]], [16120.99, 15265.56]),
             ((12000,0.8), [[-210, -80],[-229, -85]], [11947.60, 11944.98]),
             ((13000,0.8), [[-228, -85],[-233, -87]], [12861.05, 12182.42]),]


@pytest.mark.parametrize('goal,expected_delta_shares,expected_outflows', test_data, ids=ids)
def test_should_withrow_money_for_goal(goal, expected_delta_shares,expected_outflows):
    
    assets_weights = np.array([0.6,0.4])
    prices = np.array([[34.2, 59.57],[31.42, 55.88]])
    shares_owned = np.array([[288, 101],[291, 92]])
    inflow = 0
    #current value = [16120.99,15265.56]
       
    result = transactions(inflow, shares_owned,assets_weights,prices, goal)
    
    npt.assert_array_equal(result.delta_shares, expected_delta_shares) 
    npt.assert_array_equal(result.outflows, expected_outflows)



