import pytest
from model.investment_planner import InvestmentPlanner, calculateValuesForLastPeriod, get_portfolios_strategies, convert_goals_to_k, get_goals_strategies
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



def test_goals_transformer():

    goals =  [{
        "time": 5,
        "cost": 100,
        "utility": 1000                
    },
    {
        "time": 10,
        "cost": 60,
        "utility": 500               
    },
    ]

    result = convert_goals_to_k(goals)

    expected = {
        5: np.array([[100, 1000]]),
        10: np.array([[60,500]])
    }
    npt.assert_equal(result, expected)

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
def test_should_return_goals_strategies_for_zeros_VT1(goals_utilities, VT1, expected_V, expected_goal_strategies):      
     
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

    
def test_calculateBelman():

    grid = np.array([[100, 100, 100, 100, 100, 100],
                     [64., 80, 100, 126, 158, 200],
                     [56., 74, 100, 137, 187, 259],
                     [51., 70, 100, 144, 211, 311]])
    assert 1==1

            





