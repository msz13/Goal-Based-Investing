import numpy as np
import numpy.testing as npt
from model._utilities import Goals

def test_should_return_k_matrix():
    
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

    result = Goals(goals).get_k()

    expected = {
        5: np.array([[100, 1000]]),
        10: np.array([[60,500]])
    }
    npt.assert_equal(result, expected)


def test_should_return_investment_period_length():

    goals_input =  [{
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
    goals  = Goals(goals_input)

    result = goals.get_investment_period()

    assert result == 12


def test_should_get_highest_cost_for_time():

    goals_input =  [{
        "time": 5,
        "cost": 100,
        "utility": 1000                
    }  
    ]
    goals  = Goals(goals_input)

    result = goals.get_highest_cost_for_time(4)
    result2 = goals.get_highest_cost_for_time(5)

    assert result == 0
    assert result2 == 100



def test_should_get_k_array_for_time():
    
    goals_input =  [{
        "time": 5,
        "cost": 100,
        "utility": 1000                
    }  
    ]
    goals  = Goals(goals_input)

    result = goals.get_k_array(4)
    result2 = goals.get_k_array(5)

    assert result == None
    npt.assert_array_equal(result2, np.array([[100,1000]]))
