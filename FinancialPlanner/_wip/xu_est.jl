using Distributions, StatsBase

using Revise

includet("xu_model.jl")


Θ_mean = -1.01E-04
ρθθ = 0.1395 
ρθθu = 1.31E-05
ρθθd = -2.18E-04
Θt = -0.0132
Θu_mean = 500 
pΘu = 0.9993
Θd_mean = 12.9194
pΘd = 0.9040
Θut = 420
Θdt = 65
σΘΘu = 1.12E-04
σΘΘd = -0.0019
σΘu = 0.7860
σΘd = 2.0579


output_step(Θ_mean, ρθθ, ρθθu, ρθθd, Θt, Θu_mean, pΘu, Θut, Θd_mean, pΘd, Θdt, σΘΘu, σΘΘd, σΘu, σΘd)

n_samples = 1000

Θ, Θu, Θd = zeros(n_samples), zeros(n_samples), zeros(n_samples) 


for s in 1:n_samples
    Θ[s], Θu[s], Θd[s] =  output_step(Θ_mean, ρθθ, ρθθu, ρθθd, Θt, Θu_mean, pΘu, Θut, Θd_mean, pΘd, Θdt, σΘΘu, σΘΘd, σΘu, σΘd)
end

quantile(Θ, [.05, .25, .5, .75, .95]) #.* 12


quantile(Θu, [.05, .25, .5, .75, .95])
quantile(Θd, [.05, .25, .5, .75, .95]) 


params = π_params(
    0.0017,
    0, 
    0,
    0,
    .5,
    0,
    0,
    -8.49E-06, 
    8.58E-06,
    4.22E-04,
    -3.56E-04  
) 



inflation_step(params, πu_params(3.9091), πd_params(100), -1.01E-04, 500, 12.9194, 0.005, 500, 10, 0.001, 2, 100)


