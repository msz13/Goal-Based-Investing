using Distributions, StatsBase

using Revise

includet("xu_model.jl")
include("VARs/utils.jl")

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

Θ_p =  Output_Params(
    -1.01E-04,
    0.1395, 
    1.31E-05,
    -2.18E-04,
    1.12E-04,
    -0.0019)


Θstate_p = OutputState_Params(
    500, 
    0.9993,
    12.9194,
    0.9040,
    0.7860,
    2.0579
    )

output_step(Θ_p, Θstate_p, -1.01E-04, 500., 12.92)

n_samples = 1000
T =  120

Θ, Θu, Θd = zeros(n_samples, T), zeros(n_samples, T), zeros(n_samples, T) 


for s in 1:n_samples
    Θ[s, 1], Θu[s, 1], Θd[s, 1] =  output_step(Θ_p, Θstate_p, .005, 500., 12.91)
end

for t in 2:T
    for s in 1:n_samples
        Θ[s, t], Θu[s, t], Θd[s, t] =  output_step(Θ_p, Θstate_p, Θ[s, t-1], Θu[s, t-1], Θd[s, t-1])
    end
end

quantile(Θ[:,1], [.03, .25, .5, .75, .97]) * 100 #* 12

quantile(Θu[:,1], [.03, .25, .5, .75, .97])
quantile(Θd[:,1], [.03, .25, .5, .75, .97]) 

theta = reshape(Θ', (1, T, n_samples))

cum_theta = cum_returns_in_periods(theta, [1, 5], 12, true)

quantile(cum_theta[1, 1, :], [.03, .25, .5, .75, .97])'

params = π_params(
    0.0017,
    0.0002, 
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


π_state = πstate_params(3.9091, 100.)

inflation_step(params, π_state ,-1.01E-04, 500., 12.9194, 0.002, 2., 120., 0.005, 510., 10., 3.79, -5.78)

π = zeros(n_samples)

for s in 1:n_samples
    π[s] =  inflation_step(params, π_state ,-1.01E-04, 500., 12.9194, 0.0075, 30.5, 90., 0.005, 510., 10., 3.79, -5.78)
end

quantile(π, [.03, .25, .5, .75, .97]) * 12


bl = rand(Geometric(1/120), 1000)
quantile(bl, [.03, .25, .5, .75, .97])'
