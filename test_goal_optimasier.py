from goal_optimasier import generateGrid
import pytest


def test_should_generateGrid_without_cashflows():
    W0 = 100
    T = 11
    iMax = 476
    minMean = 0.0526
    minStd = 0.0374
    maxMean = 0.0886
    maxStd = 0.1954
    grid = generateGrid(W0, T, iMax, minMean, minStd, maxMean, maxStd)

    assert grid.shape == (T,iMax)
    assert pytest.approx(1834,abs=0.5)  == grid[T-1, iMax-1] 

