import pytest
from efficientfrontier import optimize
import numpy.testing as npt

def test_should_return_optimased_weights():
    means = [0.01030459, 0.00255585]
    cov_table = [[ 1.88223655e-03, -7.41578939e-05],
                 [-7.41578939e-05,  1.61690180e-04]]
    expectedMean = 0.07900492

    result = optimize(expectedMean, means, cov_table)
    npt.assert_array_almost_equal([0.5198125, 0.4801875],result) 