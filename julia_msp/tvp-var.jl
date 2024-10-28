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


"""
μ: mean of Volatility
ρ: autoregresive coeff of volatility
"""
function drawStochVolatility(h0, μ, ρ, ξ, h)
    i = length(μ)
    result = zeros(i, h+1)
    result[:,1] = h0

    for t in 2:(h+1)
        result[:,t] = μ .+ ρ .* (result[:,t-1] - μ) + rand(MvNormal(zeros(i), diagm(ξ)))
    end

    return result
end



function sample(h)

    result = zeros(i, h)
   
    #coeff = generate_coeffs()
    σ2 = 0.09




    
end

#= 
function posterior()


    Β0 = drawBeta0()

    Β = drawΒ() #FFBS algorythm

    ν = drawν()

    κ = drawκ() #hyperparameter

    λ = drawλ() #hyperparameter

    d = drawd() #threshold parameter

end
 =#

 residuals(Y, X, Β) = Y .- Β .* X 

 k_gain(P_predicted, X, Σ) =  P_predicted * X' * inv(X * P_predicted * X' .+ Σ)
 

 function kalmanFilter(X, Β0, P0, Σ, ν)
    T = length(X)
    

 end 
