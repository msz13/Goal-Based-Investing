import numpy.testing as npt
import numpy as np
from scipy.stats import norm
from model.propabilities import calculateTransitionPropabilitiesForAllPorfolios, calculateTransitionPropabilitiesForGoals

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


def test_should_calculate_transition_propabilities_for_goals():
    
    Wtc = np.array([[-6, -5,  0, 25, 45, 50],
                    [ 0,  5, 16, 36, 56, 61]]) 
    
    WT1 = np.array([90,95,100,105,110, 200])

    portfolios_for_goals = np.array([[[0.0886, 0.1954],                                  
                                  [0.0886, 0.1954],
                                  [0.0886, 0.1954],
                                  [0.0886, 0.1954],
                                  [0.0886, 0.1954],                                  
                                  [0.0526, 0.0374]],                        
                                 [[0.0886, 0.1954],                                                                    
                                  [0.0886, 0.1954],
                                  [0.0886, 0.1954],
                                  [0.0886, 0.1954],
                                  [0.0526, 0.0374],
                                  [0.0526, 0.0374]]
                                  ])

    
    result = calculateTransitionPropabilitiesForGoals(Wtc,WT1,portfolios_for_goals)
    expected1 = __prob(Wtc[0,5],WT1,portfolios_for_goals[0,5,0], portfolios_for_goals[0,5,1],0,0,1)
    expected1 = expected1/expected1.sum()
    expected2 = __prob(Wtc[0,3],WT1,portfolios_for_goals[0,3,0], portfolios_for_goals[0,3,1],0,0,1)
    expected2 = expected2/expected2.sum()
    expected3 = __prob(Wtc[1,1],WT1,portfolios_for_goals[1,1,0], portfolios_for_goals[1,1,1],0,0,1)
    expected3 = expected3/expected3.sum()
    

    assert result.shape == (2,6,6)
    npt.assert_array_equal(result[0,0],np.nan)
    npt.assert_array_equal(result[0,1],np.nan)
    npt.assert_array_equal(result[0,2],np.nan)    
    npt.assert_array_equal(result[1,0], np.nan)
    npt.assert_almost_equal(result[0,3], expected2, 3)
    npt.assert_almost_equal(result[0,5], expected1, 3)
    npt.assert_almost_equal(result[1,1], expected3, 3)

 