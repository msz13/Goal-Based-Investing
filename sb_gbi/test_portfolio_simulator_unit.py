import pytest
import numpy.testing as npt
import numpy as np
from portfolio_simulator import PortfoliosSimulator, transactions
from unittest.mock import Mock, call


@pytest.fixture
def prices():
    return np.array([[[30.00, 50.00],       
                        [32.7, 51.5],
                        [25.18, 56.14],
                        [23.67, 56.70],
                        [30.53, 57.83],
                        [34.2, 59.57]],
                        [[30.0,	50.0],
                         [28.2,	52.00],
                         [24.82, 47.84],
                         [24.07, 51.67],
                         [25.76, 54.25],
                         [31.42, 55.88]]])

def test_spike():

    mock = Mock()
    mock.side_effect = [1,2,3]
    mock(np.array([1,2]))
    mock(np.array([1,2,3]))
    
    arr = mock.call_args_list
    npt.assert_array_equal(arr[0].args[0],np.array([1,2]))

@pytest.mark.skip
def test_should_get_porfolios_last_value_(prices):  
       
    inflows = np.array([10000,0,0,0,0])
    expected = [11605.60, 10754.40]
    
    mock = mock.Mock()

    portfolios_simulator = PortfoliosSimulator(transactions)
    
    portfolios_simulator.set_params(
        assets_prices=prices,        
        assets_weights=np.array([0.6,0.4]),        
        inflows = inflows,
        outflows = np.zeros(5)
        )



    portfolios_simulator.run()
    
    wealth = portfolios_simulator.get_porfolio_final_value()

    npt.assert_array_equal(wealth, expected)



porfoliosimulator -> 
    oprations
porfolio sumilator - calculate balance



