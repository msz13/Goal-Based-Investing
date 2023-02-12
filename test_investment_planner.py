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

    investent_planner.set_params(T,W0, infusion, infusionInterval, portfolios)

    grid = investent_planner.grid

    expected_grid = [[ 54.15089958, 73.58729481, 100., 135.89302373, 184.66913898],
                     [41.33737257, 64.29414637, 100., 155.53515467, 241.91184338],
                     [33.38995822, 57.7840447, 100., 173.05815216, 299.4912403 ]]

    assert grid.shape == (3,5)
    npt.assert_array_almost_equal(expected_grid, grid)

def test_should_simulateporfolio_values_with_infusions():
    
    investent_planner = InvestmentPlanner()
    portfolios = np.array([[0.0526, 0.0374], [0.07059443, 0.103057  ], [0.0886, 0.1954]])
    W0 = 100
    T = 1
    iMax = 5
    infusion = 10
    infustionInteval = 1
    infusionGrowth = 0
    investent_planner.set_params(T,W0, infusion, infustionInteval, portfolios)

    grid = investent_planner.grid

    expected_grid = [[ 51.63085692,  71.85461497, 100. , 139.16990585, 193.68262695]]
    
    assert grid.shape == (1,5)
    npt.assert_array_almost_equal(expected_grid, grid, 4)
    


