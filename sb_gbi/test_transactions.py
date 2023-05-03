import numpy as np
import numpy.testing as npt


def transactions(inflow, assets_weights, prices):
    return np.fix((inflow*assets_weights) / prices)


def test_should_make_transactions():
    assets_weights = np.array([0.6,0.4])

    prices = np.array([[32.7, 51.5],[28.2, 52.00]])
    shares = transactions(1000,assets_weights,prices)

    npt.assert_array_equal(shares, np.array([[18, 7],[21, 7]]))   



def test_should_make_transactions_with_outflows():
    assets_weights = np.array([0.6,0.4])

    prices = np.array([[32.7, 51.5],[28.2, 52.00]])
    shares = transactions(-9000, assets_weights,prices)

    npt.assert_array_equal(shares, np.array([[-165, -69],[-191,-69]])) 