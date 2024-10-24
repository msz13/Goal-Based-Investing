using Distributions
using LinearAlgebra


struct SVparams
    μ 
    ρ 

end

"""
generate_coeffs(Β0, p, ν0, ν1, h)
samples  coeeficents for h periods

Β0: initial Cooeficeint
sp: probability of break in time varing cooefs
ν0: coeeficients volatility for steady state. Close to zero.
ν1: coeeficients volatility for time varing state. 
h: number of perios

"""
function generate_coeffs(Β0, sp, ν0, ν1, h)
    j = length(Β0)

    result = zeros(j,h+1)
    result[:,1] = Β0

    for t in 2:(h+1)
        s = [rand(Bernoulli(p)) for p in sp] 
        Θ = [s[i]*ν1[i] + (1-s[i])*ν0[i] for i in 1:j] 
    
        V = diagm(Θ) * Matrix(I, j, j)
       
        result[:,t] = result[:, t-1] .+ rand(MvNormal(zeros(j), V),1)

    end
    
    return result[:,2:end]

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