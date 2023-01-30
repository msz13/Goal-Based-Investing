import pytest
import numpy as np
import numpy.testing as npt
from assets_allocator import generatePortfolio, probabilityOfGoal, allocateGoals
import pandas as pd

class TestAssetsAllocator:
    def test_should_return_initial_value_for_zero_returns(self):
        returns = [0,0,0,0,0,0,0,0,0,0]
        assert generatePortfolio(returns,10000,0) == 10000

    def test_should_return_calculated_value_for_initial_value(self):
        returns = [0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07]
        assert generatePortfolio(returns,10000,0) == 19671.51 

    def test_should_return_calculated_value_with_payments(self):
        returns = [0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07]
        assert generatePortfolio(returns,10000,1500) == 41846.91   

    def test_shoult_return_propability_of_goal_value(self):
        values = np.array([10, 5, 11, 22, 8])
        assert probabilityOfGoal(10,values) == 0.6  


def test_should_allocate_goals():
    initialAllocation = [0.3,0.5,0.6]
    initialValue = 0
    payments = 10000
    paymentsGrowth = 0
    initialAllocation = np.array([0.4,0.6])

    goals = {'Nazwa': ["Dom", 'Emerytura'], 'Czas_trwania': [5,5], 'ExpectedValue': [120000, 400000]}
    goalsdf = pd.DataFrame(goals)

    cashFlows = allocateGoals(goalsdf,initialAllocation, initialValue, payments)
    print(goalsdf['Czas_trwania'][0])
    npt.assert_array_equal(cashFlows, np.array([[4000,4000,4000,4000,4000],[6000,6000,6000,6000,6000]]))










