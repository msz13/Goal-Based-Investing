import pytest
from investment_planner import InvestmentPlanner
import numpy.testing as npt
import numpy as np


def test_should_simulateporfolio_values_without_cashflows():

    investent_planner = InvestmentPlanner()
    portfolios = np.array([[0.0526, 0.0374], [0.07059443, 0.103057  ], [0.0886, 0.1954]])
    W0 = 100
    T = 3
    infusion = 0
    infusionInterval = 0
    infusionGrowth = 0
    goals = [0,0,0,0]

    investent_planner.set_params(T,W0, infusion, infusionInterval, goals, portfolios)


    grid = investent_planner.grid

    expected_grid = [[ 49.60772875,  59.11014269,  70.43275428,  83.92422432, 100., 119.15510785, 141.97939727, 169.17570394],
                     [ 36.43633678,  46.89764455,  60.36251882,  77.69331942, 100., 128.71119518, 165.66571766, 213.23032521],
                     [ 39.05447018,  53.42942499,  73.09543419, 100., 136.80745057, 187.16278532, 256.05263502, 350.2990821 ]]

    assert grid.shape == (3,8)
    npt.assert_array_almost_equal(expected_grid, grid)

def test_should_simulateporfolio_values_with_infusions():
    
    investent_planner = InvestmentPlanner()
    portfolios = np.array([[0.0526, 0.0374], [0.07059443, 0.103057], [0.0886, 0.1954]])
    W0 = 100
    T = 2
    infusion = 10
    infustionInteval = 1
    infusionGrowth = 0
    goals = [0, 0, 65]

    investent_planner.set_params(T,W0, infusion, infustionInteval, goals, portfolios)

    grid = investent_planner.grid

    expected_grid = [[63.3121159, 73.73231763,  85.86752449, 100., 116.45846388, 135.6257381, 157.94765122, 183.94340836],
                     [0.7689, 1.7306, 3.8955, 8.7684, 19.7370, 44.4264, 100., 225.0916]]
        
    assert grid.shape == (2,8)
    ''' assert pytest.approx(0.53,2) == grid[0,0]
    assert pytest.approx(193.68,2) == grid[0,-1] '''
    npt.assert_array_almost_equal(expected_grid, grid, 4)
    


