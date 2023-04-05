import pytest
from investment_planner import InvestmentPlanner, calculateTransitionPropabilitiesForAllPorfolios, calculateValuesForLastPeriod, get_portfolios_strategies
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

def test_should_calculate_for_single_goal():

    portfolios = np.array([[0.05258386, 0.03704926],
       [0.05515672, 0.03960988],
       [0.05772681, 0.04625568],
       [0.06029967, 0.05555016],
       [0.06286483, 0.06637403],
       [0.06545148, 0.07813246],
       [0.06801664, 0.09041086],
       [0.07059443, 0.103057  ],
       [0.07315959, 0.11592848],
       [0.07573245, 0.12898073],
       [0.07830254, 0.14213997],
       [0.0808754 , 0.15540374],
       [0.08344549, 0.16872224],
       [0.08601835, 0.18210899],
       [0.08858351, 0.19552512]])

    empty_goal = np.array([[0,0]])
    goals = np.array([empty_goal,empty_goal, np.array([[0,0],[150,50]])])

    planner = InvestmentPlanner()
    planner.set_params(2,100,5,1,goals, portfolios)


def convert_goals_to_k(goals):
    result = {}

    for goal in goals:
        result[goal['time']] = np.array([goal['cost'], goal['utility']]) 

    return result

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
        5: np.array([100, 1000]),
        10: np.array([60,500])
    }

    npt.assert_equal(result, expected)

    

            





