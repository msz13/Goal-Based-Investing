from typing import Any
from unittest import mock

from pyesg import stochastic_process
from esg.regime_switching_brownian_motion import RegimeSwitching
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, Mock, call, patch
import unittest

class NumpyArg:

    def __init__(self, arg: Any) -> None:
        self.arg = arg

    def __eq__(self, __value: object) -> bool:
        return np.array_equiv(self.arg, __value)
    
    def __repr__(self) -> str:
        return f'numpy arg: {self.arg}'

class ReturnValueWithArgs:

    def __init__(self, return_value, *args, **kwargs) -> None:
        self.args = args
        self.return_value = return_value

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if(self.args == args):
            return self.return_value
        else:
            raise ValueError(f'No return value for args: {args}, but for {self.args}')


#transition matrix for two regimes
k2_tm = [[0.65, 0.35], 
         [0.06, 0.94]]

#transition matrix for three regimes
k3_tm = [[0.65, 0.25, 0.1], 
         [0.06, 0.74, 0.2], 
         [0.11, 0.24, 0.65]]


@pytest.fixture()
def transition_matrix():
    def create_transition_matrix(n_regimes):
        if (n_regimes == 2):
            return k2_tm
        if (n_regimes == 3):
            return k3_tm
        if(n_regimes != 2 or n_regimes != 3):
            raise ValueError(f'Returns transition matrix for 2 and 3 regimes, you passed: {n_regimes}')
        
    return create_transition_matrix
        



class TestRegimeSwithing():
   
    def setup_method(cls):
      cls.model = RegimeSwitching.example()
      
    
    @pytest.mark.parametrize('number_of_regimes,expected', [
        (2, [0,1]),
        (3, [0,1,2])
        ])
    def test_should_return_regimes_ids(self, number_of_regimes, expected, transition_matrix):

        self.model.probs = transition_matrix(number_of_regimes)

        actual = self.model.regimes_idx

        np.testing.assert_array_equal(actual, expected)

    
    @pytest.mark.parametrize('number_of_scenarios,current_regimes,expected_calls',[
        (2, 1, [call(n_scenarios=1,probs=k2_tm[1])]),
        (2, [0,0,0], [call(n_scenarios=3,probs=k2_tm[0])]),
        (2,[0,1,0], [call(n_scenarios=2,probs=k2_tm[0]),call(n_scenarios=1,probs=k2_tm[1])]),
        (3,[0,1,0,2], [call(n_scenarios=2,probs=k3_tm[0]),call(n_scenarios=1,probs=k3_tm[1]),call(n_scenarios=1,probs=k3_tm[2])])        
    ])
    def test_when_next_regime_called_should_call_random_regime(self, number_of_scenarios, current_regimes, expected_calls,transition_matrix):

        self.model.probs = transition_matrix(number_of_scenarios)            
               
        self.model.random_regimes = Mock(return_value=[0])
                           
        self.model.next_regime(current_regimes)
        
        self.model.random_regimes.assert_has_calls(expected_calls)


    @pytest.mark.parametrize('number_of_regimes,current_regimes,random_regimes,expected_regimes',[
        (2, 1, [np.array([1])], [1]),
        (2, [0,0], [np.array([0,1])], [0,1]),
        (2, [0,1,0], [np.array([0,1]), np.array([1])], [0,1,1]),  
        (3, [0,1,0,2], [np.array([1,2]), np.array([0]), np.array([1])], [1,0,2,1])      
    ])
    def test_when_next_regime_called_should_return_next_regimes(self, number_of_regimes, current_regimes, random_regimes, expected_regimes,transition_matrix):

        self.model.probs = transition_matrix(number_of_regimes)        
              
        self.model.random_regimes = Mock(side_effect=random_regimes)
                           
        result = self.model.next_regime(current_regimes)

        np.testing.assert_equal(result,expected_regimes)           


    @pytest.mark.parametrize('n_regimes, n_scenarios,expected_shape',[
        (2, 1,(1,)),
        (2, 5,(5,)),
        (3, 5,(5,))
    ])
    def test_random_regimes_should_return_shape(self, n_regimes, n_scenarios,expected_shape, transition_matrix):

        self.model.probs = transition_matrix(n_regimes)
        result = self.model.random_regimes(n_scenarios=n_scenarios,probs=self.model.probs[0])

        assert result.shape == expected_shape
        assert all(item in self.model.regimes_idx for item in result)


    
    @pytest.mark.parametrize('regime,expected_value',[
        ([1], 11.3),
        ([0], 10.9),
    ]) 
    @patch('pyesg.stochastic_process.StochasticProcess')
    @patch('pyesg.stochastic_process.StochasticProcess')   
    def test_call_step_should_return_value_for_single_initial_value(self,  stochastic_model1, stochastic_model2, regime,expected_value):
        
        model0 = stochastic_model2()
        model0.step.return_value = 10.9
        model1 = stochastic_model1()
        model1.step.return_value = 11.3
                        
        regime_switching_model = RegimeSwitching(models= [model0, model1],probs=k2_tm)
        regime_switching_model.next_regime = Mock(side_effect = ReturnValueWithArgs(regime,current_regimes=0))

        result = regime_switching_model.step(x0=10,current_regimes=0)
            
        assert result == expected_value

    @pytest.mark.parametrize('next_regimes,initial_values0,initial_values1,model0_return, model1_return,expected_value',[
        ([0,0,0], [10,15,20], None, [10.9, 15.6, 20.8], None, [10.9, 15.6,20.8]),
        ([1,0,1], [15], [10,20], [11.6], [15.3,20.3], [15.3, 11.6, 20.3]),
        ([0,1,0], [10,20], [15], [10.9,20.8], [15.6], [10.9, 15.6, 20.8]),
    ]) 
    @patch('pyesg.stochastic_process.StochasticProcess')
    @patch('pyesg.stochastic_process.StochasticProcess')   
    def test_call_step_should_return_value_for_multiple_initial_value(self,  
                                                                      stochastic_model1, 
                                                                      stochastic_model2, 
                                                                      next_regimes,
                                                                      initial_values0,
                                                                      initial_values1, 
                                                                      model0_return, 
                                                                      model1_return, 
                                                                      expected_value):
                
        model0 = stochastic_model2()
        model0.step.side_effect = ReturnValueWithArgs(model0_return,NumpyArg(initial_values0),1) 
        model1 = stochastic_model1()
        model1.step.side_effect = ReturnValueWithArgs(model1_return,NumpyArg([initial_values1]),1)

        initial_values= [10,15,20]
                
        regime_switching_model = RegimeSwitching(models= [model0, model1],probs=k2_tm)
        regime_switching_model.next_regime = Mock(side_effect = ReturnValueWithArgs(next_regimes,current_regimes=[0,0,1]))

        result = regime_switching_model.step(x0=initial_values,current_regimes=[0,0,1],dt=1)
            
        np.testing.assert_array_equal(result, expected_value)

    def test_scenarios_regimes_multiple_initial_values(self):

        n_scenarios = 3
        n_steps = 5
        current_regime = [0, 1, 0]

        result = self.model.scenarios_regimes(current_regime,n_steps)
        assert result.shape == (n_steps+1, n_scenarios)
        
    def test_scenarios_with_multiple_initial_value(self):

        n_steps = 3
        n_scenarios = 2
        initial_values = [10, 10]
        current_regimes = [0, 0]
        dt = 1/4
        
        result = self.model.scenarios(initial_values, current_regimes, dt, n_steps, n_scenarios)

        assert result.shape == (n_scenarios,n_steps+1)
        
    def test_scenarios_with_single_initial_value(self):

        n_steps = 3
        n_scenarios = 2
        initial_values = 10
        current_regimes = 0
        dt = 1/4
        
        result = self.model.scenarios(initial_values, current_regimes, dt, n_steps, n_scenarios)

        assert result.shape == (n_scenarios,n_steps+1)

    

    
    