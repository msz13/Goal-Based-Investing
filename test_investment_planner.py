import pytest
from investment_planner import InvestmentPlanner, calculateTransitionPropabilitiesForAllPorfolios, calculateValuesForLastPeriod
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
    
def test_should_generate_glide_path():
    
    investent_planner = InvestmentPlanner()
    portfolios = np.array([[0.0526, 0.0374], [0.07059443, 0.103057], [0.0886, 0.1954]])
    W0 = 100
    T = 2
    infusion = 10
    infustionInteval = 1
    infusionGrowth = 0
    goals = [0, 0, 65]
    iMax = 8

    investent_planner.set_params(T,W0, infusion, infustionInteval, goals, portfolios)

    glide_paths = investent_planner.glide_paths

    expected_glide_path = np.ones((iMax,T))

    assert glide_paths.shape == (iMax,T)
    assert npt.assert_almost_equal(glide_paths, expected_glide_path)

def test_generate_propabilities():
    portfolios = portfolios = np.array([[0.0526, 0.0374], [0.07059443, 0.103057], [0.0886, 0.1954]])
    WT = np.array([90,95,100,110])
    WT1 = np.array([90,100,110,120])
    i = len(WT1)
    propabilities = calculateTransitionPropabilitiesForAllPorfolios(portfolios,WT,WT1)
    expected_propabilities = np.array([[[0.514, 0.485, 0.,    0.   ],
                                        [0.361, 0.409, 0.186, 0.043],
                                        [0.288, 0.302, 0.245, 0.165]],

                                        [[0.017, 0.945, 0.038, 0.   ],
                                        [0.205, 0.397, 0.293, 0.105],
                                        [0.238, 0.289, 0.269, 0.204]],

                                        [[0., 0.427, 0.57, 0.003],
                                        [0.099, 0.32, 0.375, 0.205],
                                        [0.194, 0.272, 0.288, 0.246]],

                                        [[0., 0., 0.372, 0.627],
                                        [0.017, 0.139, 0.384, 0.459],
                                        [0.127, 0.232, 0.311, 0.33 ]]])                                       

    npt.assert_almost_equal(propabilities,expected_propabilities,3)

def test_should_calclulate_values_for_last_period():
    k = [[105,100]]
    wealthInT = [90,95,100,105,110]
    expectedValues = [0,0,0,100,100]

    V = calculateValuesForLastPeriod(wealthInT,k)    

    npt.assert_array_equal(V,expectedValues)





