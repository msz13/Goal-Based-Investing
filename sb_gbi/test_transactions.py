from dataclasses import dataclass
import numpy as np
import numpy.testing as npt
import pytest

@dataclass
class Transaction:
    delta_shares: np.array
    outflows: int = 0
 


def transactions(inflow, assets_weights, prices, goal = None):
    delta_shares = np.fix((inflow*assets_weights) / prices)    
    return Transaction(delta_shares, goal)


def test_should_buy_assets():
    assets_weights = np.array([0.6,0.4])

    prices = np.array([[30, 50],[30, 50]])
    result = transactions(10000,assets_weights,prices)

    expected_result = Transaction(np.array([[200, 80],[200, 80]]))

    npt.assert_array_equal(result.delta_shares, expected_result.delta_shares)   


test_data = [(15000, Transaction(
        delta_shares=np.array([[-263, -100],[-286, -107]]), 
        outflows=15000))]  

#@pytest.mark.parametrize('goal,expected', test_data)
def test_should_withrow_money_for_goal():
    assets_weights = np.array([0.6,0.4])

    prices = np.array([[34.2, 59.57],[31.42, 55.88]])
    
    expected = test_data[0][1]
    goal = test_data[0][0]

    result = transactions(-15000, assets_weights,prices, goal)
    
    npt.assert_array_equal(result.delta_shares, expected.delta_shares) 
    assert result.outflows == expected.outflows


