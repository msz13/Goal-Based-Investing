
params = []

y = π*β + ψ - β*ψ + ε

state = [π, ψ, ψ, β]

F = []

lagmat([1 2 3; 4 5 6],1)

using Distributions

cdf(Normal(0,1), [1.5,1.9 ,2.5, 3])'

.46/1.9


vasiceck_step(theta, kappa, sigma, previous, n_samples) = theta .+ kappa *(previous - theta) .+ sigma * randn(n_samples) 

k1 = 0.2
k2 = 0.6
theta = .02
sigma = .03
init = 0.06

infl1 = vasiceck_step(theta, k1, sigma, init, 1000)
infl2 = vasiceck_step(theta, k2, sigma, init, 1000)
quantile(infl1, [.03, .25, .5, .75, .97])'
quantile(infl2, [.03, .25, .5, .75, .97])'


-log(2)/k1
-log(2)/k2
-log(2)/log(.99)