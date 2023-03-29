import pytest
from investment_planner import calculateTransitionPropabilitiesForAllPorfolios
import numpy.testing as npt
import numpy as np
from scipy.stats import norm

def __prob(W0, W1, mean, std, Infusion, Cost, h):
    return norm.pdf((np.log(W1/(W0+Infusion+Cost))-(mean-0.5*std**2)*h)/(std*np.sqrt(h)))

def test_should_calculate_TransitionPropabilitiesForAllPorfolios():
    portfolios = np.array([[0.0526, 0.0374], [0.07059443, 0.103057  ], [0.0886, 0.1954]])
    W0 = np.array([49,50,60,80,100,105])
    W1 = np.array([90,95,100,103,105,110])
    infusions = 5
    h = 1

    probabilities = calculateTransitionPropabilitiesForAllPorfolios(portfolios, W0,W1,infusions,h) 
    expected_prob1= __prob(W0[3],W1,portfolios[0,0],portfolios[0,1],infusions, 0, h)
    expected_prob1 = expected_prob1/expected_prob1.sum()
    expected_prob2= __prob(W0[3],W1,portfolios[1,0],portfolios[1,1],infusions, 0, h)
    expected_prob2 = expected_prob2/expected_prob2.sum()
    expected_prob3 = __prob(W0[3],W1,portfolios[2,0],portfolios[2,1],infusions, 0, h)
    expected_prob3 = expected_prob3/expected_prob3.sum()

    assert probabilities.shape == (3, 6, 6)    
    npt.assert_array_almost_equal(probabilities[0,3], expected_prob1,3)
    npt.assert_array_almost_equal(probabilities[1,3], expected_prob2,3)
    npt.assert_array_almost_equal(probabilities[2,3], expected_prob3,3)
    
 





