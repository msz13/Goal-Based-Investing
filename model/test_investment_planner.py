import pytest
from model.investment_planner import InvestmentPlanner, calculateBelmanForT, calculateBelman, calculateValuesForLastPeriod, get_portfolios_strategies, get_goals_strategies
from model._utilities import Goals
import numpy.testing as npt
import numpy as np


def test_should_values_last_period():
    
    goals = np.array([[0,0,],[40, 80], [60,100], [100,180]])
    WT = np.array([39, 40, 59,60, 80, 100])
    expected_values = np.array([0, 80, 80, 100, 100, 180])
    expected_goal_strategies = np.array([0,1,1,2,2,3])

    values, strategies = calculateValuesForLastPeriod(WT, goals)

    npt.assert_array_equal(values, expected_values, verbose=True)
    npt.assert_array_equal(strategies, expected_goal_strategies)

def test_should_get_porfolios_strategies():
    VT1 = np.array([0,0,100,100])
    probabilities =  np.array([[[1.,    0.,    0.,    0.   ],
                                [0.926, 0.065, 0.008, 0.   ],
                                [0.017, 0.105, 0.244, 0.634],
                                [0.001, 0.017, 0.075, 0.908]],

                                [[0.65, 0.226, 0.108, 0.016],
                                [0.388, 0.284, 0.222, 0.106],
                                [0.177, 0.234, 0.268, 0.322],
                                [0.14, 0.211, 0.262, 0.387]],

                                [[0.368, 0.275, 0.225, 0.133],
                                [0.287, 0.264, 0.247, 0.202],
                                [0.227, 0.246, 0.256, 0.271],
                                [0.215, 0.241, 0.257, 0.288]]])
    
    expected_propabilities = np.array([[0.368, 0.275, 0.225, 0.133],
                                       [0.287, 0.264, 0.247, 0.202],
                                       [0.017, 0.105, 0.244, 0.634],
                                       [0.001, 0.017, 0.075, 0.908]
                                       ])
    
    portfolios_strategies, values, chosen_probabilities = get_portfolios_strategies(VT1,probabilities)
    
    npt.assert_array_equal(portfolios_strategies, np.array([2,2,0,0]))
    npt.assert_array_almost_equal(values, np.array([35.8, 44.9, 87.8, 98.3]),3)
    npt.assert_array_equal(chosen_probabilities, expected_propabilities)




''' 
get_goal_strategies tests
* none goal
* single goal
* multiple goals
* zeros vt1 
'''


@pytest.mark.parametrize('goals_utilities,VT1,expected_V,expected_goal_strategies', 
[([[100,150],[0,0,0,0],[0,100,150,150], [0,1,2,2]]),
 ([[100,150],[0,0,100,150],[20,120,185,185], [0,1,1,2]])
 ])
def test_should_return_goals_strategies_for_zeros_VT1(goals_utilities: list[list[int]], VT1: list[list[int]], expected_V: list[list[int]], expected_goal_strategies: list[list[int]]):      
     
    probabilities = np.array([[[0.4, 0.4, 0.2, 0],
                           [0.4, 0.3, 0.2, 0.1],
                           [0.2, 0.3, 0.3, 0.2],
                           [0.1, 0.3, 0.4, 0.2]
                           ],
                           [[np.nan, np.nan, np.nan, np.nan],                            
                            [0.4, 0.4, 0.2, 0],
                            [0.1, 0.2, 0.4, 0.3],
                            [0.1, 0.2, 0.5, 0.2]
                           ],[
                           [np.nan, np.nan, np.nan, np.nan],  
                           [np.nan, np.nan, np.nan, np.nan],                         
                            [0.4, 0.4, 0.2, 0],
                            [0.4, 0.3, 0.2, 0.1]                            
                           ]])      
        
    result_V, result_goal_strategies = get_goals_strategies(probabilities, np.array(goals_utilities), np.array(VT1))

    npt.assert_equal(result_V, expected_V)
    npt.assert_equal(result_goal_strategies, expected_goal_strategies)

def test_shoulf_calculateBelmanForT_for_last_t():

    goals = np.array([[30,100]])
    infusion = 0
    WT = np.array([0.68, 1.27, 2.37, 4.43, 8.26, 15.41, 28.74, 53.61, 100., 186.52])
    WT1= np.array([37.69, 48.11, 61.4 , 78.36, 100., 127.62, 162.88, 207.87, 265.29, 338.58])
    VTK1 = np.array([0,0,0,0,0,0,0,0,0,0])
    portfolios = np.array([[0.0526, 0.0374], [0.07059443, 0.103057  ], [0.0886, 0.1954]])
    goal_strategies, chosen_portfolios_strategies, values, chosen_goal_propabilities = calculateBelmanForT(goals,infusion,WT,WT1,VTK1,portfolios)

    npt.assert_array_equal(goal_strategies, np.array([0,0,0,0,0,0,0,0,0,100]))

    
def test_calculateBelman():
    
    grid = np.array([[100, 100, 100, 100, 100, 100],
                     [64., 80., 100, 126, 158, 200],
                     [56., 74., 100, 137, 187, 259],
                     [51., 70., 100, 144, 211, 311]                     
                     ])
        
    goals = goals = Goals([{        
        "time": 2,
        "cost": 107,
        "utility": 100                
    }])

    portfolios = np.array([[0.0526, 0.0374], [0.07059443, 0.103057  ], [0.0886, 0.1954]])

    expected_goal_strategies = np.zeros((4,6))
    expected_portfolios_strategies = np.zeros((4,6))
    expected_probabilities = np.zeros((4,6,6))

    values3 = np.array([0,0,0,0,00,00])
    expected_goal_strategies[2], expected_portfolios_strategies[2], values2, expected_probabilities[2] = calculateBelmanForT(goals.get_k_array(2),0,grid[2], grid[3], values3, portfolios)
    expected_goal_strategies[1], expected_portfolios_strategies[1], values1, expected_probabilities[1] = calculateBelmanForT(None,0,grid[1], grid[2], values2, portfolios)
    expected_goal_strategies[0], expected_portfolios_strategies[0], values0, expected_probabilities[0] = calculateBelmanForT(None,0,grid[0], grid[1], values1, portfolios)
        
    goal_strategies, portfolios_strategies, propabilities  = calculateBelman(grid, goals,portfolios)
  
    npt.assert_array_equal(goal_strategies, expected_goal_strategies)
    npt.assert_array_equal(portfolios_strategies, expected_portfolios_strategies)
    npt.assert_array_almost_equal(propabilities, expected_probabilities, 3)

            





