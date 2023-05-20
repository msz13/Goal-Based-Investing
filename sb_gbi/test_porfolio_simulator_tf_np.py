import numpy as np
import numpy.testing as npt
import pytest
from portfolio_simulator_tf_numpy import PortfoliosSimulator, calculate_prob_of_goal_achivement
        

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



def test_should_get_porfolios_outflows_single(prices):  
        
    inflows = np.array([10000,1000,1000,1000,1000,0])  

    goals = {5: (16000,1)}  

    assets_weights = np.array([[0.6,0.4],[0.6,0.4],[0.6,0.4],[0.6,0.4],[0.6,0.4],[0.6,0.4]])

    expected_final_value = np.array([580.28, 0])
    expectet_outflows = np.array([[16000.00, 15398.00]])

    portfolios_simulator = PortfoliosSimulator()
    
    portfolios_simulator.set_params(
        assets_prices=prices,        
        assets_weights=assets_weights,       
        inflows = inflows,
        goals = goals
        )


    portfolios_simulator.run()
    
    outflows = portfolios_simulator.get_outflows()    
    wealth = portfolios_simulator.get_porfolio_final_value()

    npt.assert_array_equal(outflows, expectet_outflows)
    #npt.assert_array_equal(portfolios_simulator.get_shares(),[[10,4],[0,0]])
    npt.assert_array_equal(wealth, expected_final_value)



def test_should_get_porfolios_outflows_multiple_goals(prices):  
        
    inflows = np.array([10000,1000,1000,1000,1000,0])  

    goals = {3: (8000,1), 5: (6000,1)}  
    assets_weights = np.array([[0.6,0.4],[0.6,0.4],[0.6,0.4],[0.6,0.4],[0.6,0.4],[0.6,0.4]])

    expected_final_value = np.array([383.91, 0])
    expectet_outflows = np.array([[8000, 8000], [6000, 5775.12]])

    portfolios_simulator = PortfoliosSimulator()
    
    portfolios_simulator.set_params(
        assets_prices=prices,        
        assets_weights=assets_weights,       
        inflows = inflows,
        goals = goals
        )


    portfolios_simulator.run()
    
    outflows = portfolios_simulator.get_outflows()    
    wealth = portfolios_simulator.get_porfolio_final_value()

    npt.assert_array_equal(outflows, expectet_outflows)
    #npt.assert_array_equal(portfolios_simulator.get_shares(),[[10,4],[0,0]])
    #npt.assert_array_equal(wealth, expected_final_value)

def test_test_should_get_goals_propabilities(prices):
    
    inflows = np.array([10000,1000,1000,1000,1000,0]) 
    goals = {3: (8000,1), 5: (6000,1)}  
    assets_weights = np.array([[0.6,0.4],[0.6,0.4],[0.6,0.4],[0.6,0.4],[0.6,0.4],[0.6,0.4]])

  
    expected_probabilties = [1., 0.98]

    portfolios_simulator = PortfoliosSimulator()
    
    portfolios_simulator.set_params(
        assets_prices=prices,        
        assets_weights=assets_weights,       
        inflows = inflows,
        goals = goals
        )

    portfolios_simulator.run()
    
    probabilities = portfolios_simulator.get_goals_achevement_probabilities()    
 
    npt.assert_array_equal(probabilities, expected_probabilties)



def test_test_should_get_outflows_different_weights(prices):
    inflows = np.array([10000,1000,1000,1000,1000,0])  

    goals = {5: (15000,1)}  

    assets_weights = [[0.8,0.2],[0.8,0.2],[0.8,0.2],[0.7,0.3],[0.6,0.4],[0.5,0.5]]
   
    expectet_outflows = np.array([[15000, 14797.44]])

    portfolios_simulator = PortfoliosSimulator()
    
    portfolios_simulator.set_params(
        assets_prices=prices,        
        assets_weights=assets_weights,       
        inflows = inflows,
        goals = goals
        )

    portfolios_simulator.run()
    
    outflows = portfolios_simulator.get_outflows()    

    npt.assert_array_equal(outflows, expectet_outflows)



def test_function_should_get_goals_propabilities(prices):  
    
    goal_outflows1 = np.array([[8000, 8000, 8000], [6000, 5775.12, 6000], [6000,6000,6000], [0,0,0]])
    goal_target = [8000,6000, 12000, 4000]

    expcted_goals_probabilities = [1., 0.99, 0.5, 0.]

    result = calculate_prob_of_goal_achivement(goal_target, goal_outflows1)

    npt.assert_array_equal(result, expcted_goals_probabilities)        
    

