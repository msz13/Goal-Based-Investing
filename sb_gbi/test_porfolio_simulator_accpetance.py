import numpy as np
import numpy.testing as npt
import pytest
from portfolio_simulator import PortfoliosSimulator
        

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

      


test_data= [(np.array([10000,1000,1000,1000,1000]),[16873.39,15398.00]),
       (np.array([10000,0,0,0,0]), [11605.60, 10754.40])]

@pytest.mark.skip("Bo tak chce")
@pytest.mark.parametrize('inflows,expected',test_data, ids=['inflows', 'no inflows'])
def test_should_get_porfolios_last_value_(prices, inflows, expected):  
       
        
    portfolios_simulator = PortfoliosSimulator()
    
    portfolios_simulator.set_params(
        assets_prices=prices,        
        assets_weights=np.array([0.6,0.4]),        
        inflows = inflows,
        goal = {}
        )


    portfolios_simulator.run()
    
    wealth = portfolios_simulator.get_porfolio_final_value()

    npt.assert_array_equal(wealth, expected)




def test_should_get_porfolios_outflows_single(prices):  
        
    inflows = np.array([10000,1000,1000,1000,1000,0])  

    goals = {
        5: (16000,1)
    }  

    expected_final_value = np.array([873.39, 0])
    expectet_outflows = np.array([[16000.00, 15398.00]])

    portfolios_simulator = PortfoliosSimulator()
    
    portfolios_simulator.set_params(
        assets_prices=prices,        
        assets_weights=np.array([0.6,0.4]),       
        inflows = inflows,
        goals = goals
        )


    portfolios_simulator.run()
    
    outflows = portfolios_simulator.get_outflows()    
    wealth = portfolios_simulator.get_porfolio_final_value()

    npt.assert_array_equal(outflows, expectet_outflows)
    #npt.assert_array_equal(wealth, expected_final_value)

''' outflows = {
    1: 10000,
    2: 100000,
    3: 50000
}

outflows = np.array([10000,100000,50000])
 '''