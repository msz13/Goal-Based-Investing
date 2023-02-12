import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def pstd(weights, cov_matrix):
    #variance = np.transpose(weights)@cov_matrix@weights
    variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    return np.sqrt(variance)*np.sqrt(12)

def pmean(weightsDf, means):  
    weights = weightsDf
    return weights.dot(means) *12



def optimize(expectedMean, means: np.array, cov_table: np.array):

    constraint1 = NonlinearConstraint(lambda x : x.sum(),1,1)
    constraint2 = NonlinearConstraint(lambda x: x.dot(means)*12, expectedMean, expectedMean)

    obj = lambda x: x.T@cov_table@x
    start = [1/len(means) for n in range(len(means))]
    result = minimize(obj,start,constraints=[constraint1, constraint2],bounds=[(0,1) for n in range(len(means))])
    #result = np.append(result.x, [expectedMean], [result.fun])    

    return result.x, result