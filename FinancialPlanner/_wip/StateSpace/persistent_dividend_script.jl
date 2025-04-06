using Revise
using Distributions, LinearAlgebra
using StableRNGs

includet("persistent_dividend_model.jl")



Θ1 = [.333, .343, .003, .528, .234]
Θ2 = [.338, .286, -.130, -.012, .625]

R = [1.45*10^-1, 1.59*10^-14, 3.01*10^-13, 7.43*10^-10, 3.78*10^-12 ]

Q = diagm([4.66 * 10^-2, 2.34 * 10^-2, 5.51 * 10^-2, 2.2, 5.0 * 10^-1, 2.53 * 10^-1, 62.11, 2.09])
#Q = [4.66 * 10^-2, 2.34 * 10^-2, 5.51 * 10^-2, 2.2, 5.0 * 10^-1, 2.53 * 10^-1, 62.11, 2.09]

sqrt.(R) 
sqrt.(Q) ./ 100 * 2

model = persistent_dividend_SSM(NamedTuple{(:Θ1, :Θ2, :R, :Q)}([Θ1, Θ2, R, Q]))

rng = StableRNG(1234)
x, y = generate_data(rng, model, 100)




prior = product_distribution(product_distribution(Normal(1,0), Normal(1,0), Normal(1,0), Normal(), Normal(),Normal(), Normal(),Normal()), 
                             product_distribution(Normal(1,0), Normal(1,0), Normal(1,0), Normal(), Normal(),Normal(), Normal(),Normal()),
                             product_distribution(truncated(Cauchy(.01, .03); lower=0), truncated(Cauchy(.01, .03); lower=0), truncated(Cauchy(.01, .03); lower=0), truncated(Cauchy(.01, .03); lower=0), truncated(Cauchy(.01, .03); lower=0), truncated(Cauchy(.01, .03); lower=0), truncated(Cauchy(.01, .03); lower=0), truncated(Cauchy(.01, .03); lower=0)),
                             product_distribution(truncated(Cauchy(.01, .03); lower=0), truncated(Cauchy(.01, .03); lower=0), truncated(Cauchy(.01, .03); lower=0), truncated(Cauchy(.01, .03); lower=0), truncated(Cauchy(.01, .03); lower=0), truncated(Cauchy(.01, .03); lower=0), truncated(Cauchy(.01, .03); lower=0), truncated(Cauchy(.01, .03); lower=0))
                            )

ll_pmmh = PMMH(
                5_000,
                θ -> MvNormal(θ, Matrix((0.1)*I(32))),
                θ -> model,
                prior
                )


kf_pmmh = sample(rng, ll_pmmh, y, KF(); burn_in=1000)

kf_pmmh[1]

a = mean(getproperty.(kf_pmmh, :params))



g = rand(InverseWishart(100,Matrix(I(3))),1000)



g = rand(InverseGamma(80, 2.5), 1000)

mean(g)
quantile(g, [0, .05, .25, .50, .75, .95, 1.])

g = rand(truncated(Cauchy(.01, .03); lower=0), 1000)

mean(g)
quantile(g, [0, .05, .25, .50, .75, .95, 1.])

g = rand(truncated(Normal(); lower=0), 1000)

mean(g)
quantile(g, [0, .05, .25, .50, .75, .95, 1.])

