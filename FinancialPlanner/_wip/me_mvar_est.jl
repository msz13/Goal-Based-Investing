using Revise
using .ESGModels

using StatsBase


X = [1, .12, .06]

transition_matrix = [.85 .15;.2 .8]

states_zero = [.67, .33]


Β1 = [.09 .3 .2; .03 .1 .6]
Σ1 = cor2cov([1 .3; .3 1], [.08, .03])
Β2 = [-.02 .35 .25; .035 .15 .63]
Σ2 = cor2cov([1 .35; .35 1], [.18, .035])

regime_probs = [.95 .05; .9 .1; .8 .2; .7 .3 ; .65 .35]

