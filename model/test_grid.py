from model.grid import generateGrid, WMax, WMin
from model._utilities import Goals
import numpy as np
import numpy.testing as npt
import pytest


portfolios = np.array([[0.0526, 0.037 ],
       [0.0552, 0.0396],
       [0.0577, 0.0463],
       [0.0603, 0.0556],
       [0.0629, 0.0664],
       [0.0655, 0.0781],
       [0.068 , 0.0904],
       [0.0706, 0.1031],
       [0.0732, 0.1159],
       [0.0757, 0.129 ],
       [0.0783, 0.1421],
       [0.0809, 0.1554],
       [0.0834, 0.1687],
       [0.086 , 0.1821],
       [0.0886, 0.1955]])


test_data = [
    (1,[0],196),
    (5,[0,0,0,0,0],576),
    (1,[10],206 ),
    (2,[10,10],303),
    (5,[10,10,11,11,11],721)]

@pytest.mark.parametrize('t,infusions,expected_result',test_data)
def test_should_calculate_grid_max_value(t,infusions,expected_result):
    
    W0 = 100
    result = WMax(t, W0, infusions,portfolios[-1,0], portfolios[-1,1], portfolios[0,1])

    assert result == expected_result

test_data = [
    (1,[0,],[0], 58),
    (5,[0,0,0,0,0],[0,0,0,0,0],32),
    (1,[10],[0],68),
    (5,[10,10,11,11,11],[0,0,0,0,0],62),
    (1,[10],[20],48),
    (5,[10,10,11,11,11],[0,0,20,0,50],3)
    ]

@pytest.mark.parametrize('t,infusions,max_goal_cost,expected_result',test_data)
def test_should_calculate_grid_min_value(t,infusions,max_goal_cost,expected_result):
    
    W0 = 100
    result = WMin(t, W0, infusions, max_goal_cost, portfolios[0,0], portfolios[0,1], portfolios[-1,1])

    assert result == expected_result

def test_should_generate_grid_without_infusions():
    
    W0 = 100
    imax = 10  
    
    goals = [0,0,20,0,50]
    infusions = [5,5,5,5,5]
 
    grid = generateGrid(W0, imax, infusions, goals, portfolios[0,0], portfolios[0,1], portfolios[-1,0], portfolios[-1,1])

    expectedGrid =     [[100.00,    100.00,    100.00,   100.00,   100.00,   100.00,   100.00,   100.00,   100.00,   100.00],
     [50.56,      57.95,      66.42,      76.12,      87.25,     100.00,     114.61,     131.36,    150.56,    172.56],
     [45.58,      55.47,      67.51,      82.17,     100.00,     121.71,     148.12,     180.28,    219.41,    267.03],
     [37.69,      48.11,      61.40,      78.36,     100.00,     127.62,     162.88,     207.87,    265.29,    338.58],
     [32.00,      42.54,      56.56,      75.21,     100.00,     132.96,     176.79,     235.06,    312.54,    415.56],
     [27.61,      38.09,      52.55,      72.49,     100.00,     137.95,     190.30,     262.51,    362.13,    499.56],
     [24.12,      34.41,      49.11,      70.08,     100.00,     142.70,     203.63,     290.59,    414.67,    591.73]]

    assert grid.shape == (7,10)
    npt.assert_almost_equal(grid,expectedGrid, 2)
    