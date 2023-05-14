from dataclasses import dataclass
from unittest.mock import Mock
import numpy as np
import numpy.testing as npt
import pytest
from portfolio_simulator import transactions, Transaction



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
       'max withrowal less than 100, goal not reached',
       'no outflows']

test_data = [((14000,1), [[-255, -88],[-285, -89]], [14000, 14000]),
             ((16000,1), [[-288, -101],[-291, -92]], [15866.17, 14284.18]),
             ((11000.00,0.8), [[-202, -68],[-228, -68]], [11000, 11000]),
             ((13000,0.8), [[-232, -79],[-236, -71]], [12692.94, 11427.34]),
             (None, [[-9,5],[-18,10]], [0,0])]


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



