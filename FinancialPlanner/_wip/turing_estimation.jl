using Turing, Distributions, Random, LinearAlgebra, StatsPlots
using Revise

includet("turing_model.jl")


T = 200  # Number of time steps
data, true_states = generate_data(T, 2)

#normalwishart bvar
model = var_model(data)
chain = sample(model, NUTS(.65), 5000);

summarize(MCMCChains.group(chain, :mus))

summarize(MCMCChains.group(chain, :betas))

summarize(MCMCChains.group(chain, :sigmas))



#ms var

model = switching_model(data)

# Sampling
g = Gibbs(NUTS(1000, 0.65, :mus, :betas, :sigmas, :P), PG(120, :s))
chain = sample(model, g, 1000); 

mus = summarize(MCMCChains.group(chain, :mus))

summarize(MCMCChains.group(chain, :betas))

summarize(MCMCChains.group(chain, :sigmas))


using StatsBase
Σ1 = cor2cov([1 .3; .3 1], [.08, .03])
cor2cov([1 .35; .35 1], [.18, .035])

xt = [1, .15, .06]

i = Matrix(I(2))
ξ = [.8, .2]
B = [0.3, 0.2, 0.2, 0.6] 
kron(xt', ones(2))

