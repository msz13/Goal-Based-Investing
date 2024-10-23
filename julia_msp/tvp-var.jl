using Distributions

struct SVparams
    μ 
    ρ 

end


function step(x, Βtm1, p, ν0,  ν1, htm1, SVparams)
    n_variables = 1

    s = rand(Bernoulli(p))
    Θ = s*ν1 + (1-s)*ν0
    Β = Βtm1 + rand(Normal(0, Θ))

    h = SVparams.μ + SVparams.ρ * (htm1 - SVparams.μ)

    σ = 0.1

    y = rand(Normal(x * Β, σ))

    return y
end