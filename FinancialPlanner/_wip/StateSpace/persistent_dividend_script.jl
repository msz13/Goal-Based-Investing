using Revise
using Distributions, LinearAlgebra
using StableRNGs

includet("persistent_dividend_model.jl")



Θ1 = [.333, .343, .003, .528, .234]
Θ2 = [.338, .286, -.130, -.012, .625]

R = [1.45*10^-1, 1.59*10^-14, 3.01*10^-13, 7.43*10^-10, 3.78*10^-12 ]

#Q = diagm([4.66 * 10^-2, 2.34 * 10^-2, 5.51 * 10^-2, 2.2, 5.0 * 10^-1, 2.53 * 10^-1, 62.11, 2.09])
Q = [4.66 * 10^-2, 2.34 * 10^-2, 5.51 * 10^-2, 2.2, 5.0 * 10^-1, 2.53 * 10^-1, 62.11, 2.09]

sqrt.(R) 
sqrt.(Q) ./ 100 * 2

#model = PersistentDividendSSM(NamedTuple{(:Θ1, :Θ2, :R, :Q)}([Θ1, Θ2, R, Q]))

params_vec = [Θ1; Θ1; R; Q]
model = persistent_dividend_ssm(params_vec)


rng = StableRNG(1234)
x, y = generate_data(rng, model, 100)



prior = Product([fill(Normal(), 10); fill(truncated(Cauchy(.01, .03); lower=0), 13)])                 

rand(prior)



pdf(prior, params_vec)



ll_pmmh = PMMH(
                5_000,
                θ -> MvNormal(θ, Matrix((0.1)*I(23))),
                θ -> persistent_dividend_ssm(θ),
                prior
                )


kf_pmmh = sample(rng, ll_pmmh, y, KF(); burn_in=3000)

a = mean(getproperty.(kf_pmmh, :params))


pf_pmmh = sample(rng, ll_pmmh, y, PF(256, 1.0); burn_in=1000)

b = mean(getproperty.(pf_pmmh, :params))


params_vec'
a'
b'

println(round.(params_vec, digits=4))
println(round.(a, digits=4))
println(round.(b, digits=4))

#true params
[0.333, 0.343, 0.003, 0.528, 0.234, 0.333, 0.343, 0.003, 0.528, 0.234, 0.145, 0.0, 0.0, 0.0, 0.0, 0.0466, 0.0234, 0.0551, 2.2, 0.5, 0.253, 62.11, 2.09]
#a
[0.505, 0.2374, 0.0428, -0.4, 0.3867, -2.6465, -1.7975, -0.899, 3.3745, 1.487, 3.2911, 0.0791, 0.2928, 0.1183, 0.2123, 0.4743, 0.2578, 0.2549, 2.375, 1.0027, 0.1415, 1.0267, 0.3902]

#b
[1.756, 0.3019, -0.0992, 0.5201, -1.0967, 0.1373, 0.2291, -0.7503, 0.2902, 0.6158, 0.6958, 0.0252, 0.0928, 0.0341, 0.3044, 0.0072, 0.0533, 0.0607, 0.0145, 0.0404, 0.0153, 0.0113, 0.0383]