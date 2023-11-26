import unittest
from inspect import getfullargspec
import numpy as np
import pandas as pd
from independent_lognormal_model import IndependentLogNormal


class BaseProcessMixin:
    """
    Holds common tests for all processes. Each model should subclass this mixin to
    inherit tests for scenario generation, expectation, dimension-checking, etc. Each
    model subclass will need to create a `self.model` instance that can be tested, plus
    `self.single_x0` and `self.multiple_x0` attributes that define reasonable starting
    value arrays for single start values and multiple start values, respectively.
    """

    def test_coefficient_dictionary(self):
        """Ensure the coefficient dictionary returns the starting params"""
        # inspect the __init__ method of the model to extract the coefficients required
        # to create the model, then confirm they all exist in the "coefs" dictionary.
        expected = getfullargspec(self.model.__init__).args[1:]
        actual = list(self.model.coefs().keys())
        self.assertListEqual(expected, actual)

    def test_single_initial_value_drift_shape(self):
        """Ensure the drift has the correct shape for a single start value"""
        drift = self.model.drift(x0=self.single_x0)
        self.assertEqual(drift.shape, self.single_x0.shape)

    def test_multiple_initial_value_drift_shape(self):
        """Ensure the drift has the correct shape for multiple start values"""
        drift = self.model.drift(x0=self.multiple_x0)
        self.assertEqual(drift.shape, self.multiple_x0.shape)

    def test_single_initial_value_expectation_shape(self):
        """Ensure the expectation has the correct shape for a single start value"""
        exp = self.model.expectation(x0=self.single_x0, dt=1.0)
        self.assertEqual(exp.shape, self.single_x0.shape)

    def test_multiple_initial_value_expectation_shape(self):
        """Ensure the expectation has the correct shape for multiple start values"""
        exp = self.model.expectation(x0=self.multiple_x0, dt=1.0)
        self.assertEqual(exp.shape, self.multiple_x0.shape)

    def test_single_initial_value_standard_deviation_shape(self):
        """Ensure the std deviation has the correct shape for a single start value"""
        if self.model.dim == 1:
            expected_shape = self.single_x0.shape
        else:
            expected_shape = (self.model.dim, self.model.dim)
        std = self.model.standard_deviation(x0=self.single_x0, dt=1.0)
        self.assertEqual(std.shape, expected_shape)

    def test_multiple_initial_value_standard_deviation_shape(self):
        """Ensure the std deviation has the correct shape for multiple start value"""
        if self.model.dim == 1:
            expected_shape = self.multiple_x0.shape
        else:
            expected_shape = (self.multiple_x0.shape[0], self.model.dim, self.model.dim)
        std = self.model.standard_deviation(x0=self.multiple_x0, dt=1.0)
        self.assertEqual(std.shape, expected_shape)

    def test_single_initial_value_step_shape(self):
        """Ensure the step function returns an array matching the initial array"""
        step = self.model.step(x0=self.single_x0, dt=1.0)
        self.assertEqual(step.shape, self.single_x0.shape)

    def test_multiple_initial_value_step_shape(self):
        """Ensure the step function returns an array matching the initial array"""
        step = self.model.step(x0=self.multiple_x0, dt=1.0)
        self.assertEqual(step.shape, self.multiple_x0.shape)

    def test_single_initial_value_scenario_shape(self):
        """Ensure scenarios from a single start value have the right shape"""
        if self.model.dim == 1:
            expected_shape = (50, 31)
        else:
            expected_shape = (50, 31, self.model.dim)
        scenarios = self.model.scenarios(
            x0=self.single_x0, dt=1.0, n_scenarios=50, n_steps=30
        )
        self.assertEqual(scenarios.shape, expected_shape)

    def test_multiple_initial_value_scenario_shape(self):
        """Ensure scenarios from a single start value have the right shape"""
        if self.model.dim == 1:
            expected_shape = (self.multiple_x0.shape[0], 31)
        else:
            expected_shape = (self.multiple_x0.shape[0], 31, self.model.dim)
        scenarios = self.model.scenarios(
            x0=self.multiple_x0,
            dt=1.0,
            n_scenarios=self.multiple_x0.shape[0],
            n_steps=30,
        )
        self.assertEqual(scenarios.shape, expected_shape)

    def test_single_initial_value_step_float_dtype(self):
        """Ensure we can pass single initial values as a float"""
        if self.model.dim == 1:  # can only run this for models with a single parameter
            step = self.model.step(x0=float(self.single_x0), dt=1.0)
            self.assertEqual(step.shape, self.single_x0.shape)

    def test_single_initial_value_step_list_dtype(self):
        """Ensure we can pass single initial values as a list of floats"""
        step = self.model.step(x0=list(self.single_x0), dt=1.0)
        self.assertEqual(step.shape, self.single_x0.shape)

    def test_single_initial_value_step_series_dtype(self):
        """Ensure we can pass single initial values as a pd.Series"""
        step = self.model.step(x0=pd.Series(self.single_x0), dt=1.0)
        self.assertEqual(step.shape, self.single_x0.shape)

    def test_single_initial_value_unique_scenarios(self):
        """Ensure when we generate lots of scenarios, they are all unique"""
        scenarios = self.model.scenarios(
            x0=self.single_x0, dt=1.0, n_scenarios=50, n_steps=30
        )
        if self.model.dim == 1:
            # check the only model ouptut variable
            self.assertEqual(50, len(set(scenarios[:, -1])))
        else:
            # check all model output variables
            for dimension in range(self.model.dim):
                self.assertEqual(50, len(set(scenarios[:, -1, dimension])))

    def test_repeat_initial_value_drift(self):
        """Ensure a repeated array of inputs has a repeated array of outputs"""
        # create a new array of five identical start arrays stacked vertically. Then
        # make sure the output array is five identical arrays also stacked vertically.
        if self.model.dim == 1:
            x0 = np.repeat(self.single_x0, 5)
            actual = self.model.drift(x0=x0)
            expected = np.repeat(actual[0], 5)
        else:
            x0 = np.repeat(self.single_x0[None, :], 5, axis=0)
            actual = self.model.drift(x0=x0)
            expected = np.repeat(actual[0][None, :], 5, axis=0)
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_repeat_initial_value_expectation(self):
        """Ensure a repeated array of inputs has a repeated array of outputs"""
        # create a new array of five identical start arrays stacked vertically. Then
        # make sure the output array is five identical arrays also stacked vertically.
        if self.model.dim == 1:
            x0 = np.repeat(self.single_x0, 5)
            actual = self.model.expectation(x0=x0, dt=1.0)
            expected = np.repeat(actual[0], 5)
        else:
            x0 = np.repeat(self.single_x0[None, :], 5, axis=0)
            actual = self.model.expectation(x0=x0, dt=1.0)
            expected = np.repeat(actual[0][None, :], 5, axis=0)
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))

    def test_scenario_failed_broadcasting_raises(self):
        """Ensure we raise a ValueError if the x0 array can't be broadcast"""
        self.assertRaises(
            ValueError,
            self.model.scenarios,
            self.multiple_x0,  # x0 array
            1.0,  # dt
            41,  # number of scenarios, can't be broadcast
            30,  # n_steps
        )

    def test_rvs(self):
        """Ensure the array of random variates matches what we expect"""
        # scenarios are generated sequentially by time step, so the number of scenarios
        # impacts where the array of random variates goes. For example, an array of rvs
        # for 100 scenarios and 30 timesteps will be generated 30 at a time, populating
        # each time step of each scenario in a row. To test this, we generate an array
        # of random variables, then reshape it to check that it matches what we expect.
        scen, step, state, dim = 1000, 120, 123, self.model.dim
        if dim == 1:
            expected = self.model.dW.rvs(size=scen * step, random_state=state)
            expected = expected.reshape(scen, step, order="F")
        else:
            expected = self.model.dW.rvs(size=scen * step * dim, random_state=state)
            expected = expected.reshape(scen * step, dim)
            expected = expected.reshape(scen, step, dim, order="F")
        actual = self.model.rvs(scen, step, state)
        self.assertIsNone(np.testing.assert_array_equal(actual, expected))


class TestIndependentLogNormal(BaseProcessMixin, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = IndependentLogNormal.example()
        cls.single_x0 = np.array([100.0])
        cls.multiple_x0 = np.full(50, 100.0)

  