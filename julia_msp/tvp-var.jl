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
    i, j = size(Β0)
   

    result = zeros(h+1, i, j)
    result[1, :, :] = Β0
    
    
    for t in 2:(h+1)

        s = map(p -> rand(Bernoulli(p)), sp) #draw if time varing
        Θ = map(x -> x[2]*ν1[x[1]] + (1-x[2])*ν0[x[1]], enumerate(s)) #coefficient volatility 

        vol = map(x -> rand(Normal(0, x)), Θ) #draw coeficeint innovation

        result[t, :, : ] = result[t-1, :, :] + vol       

    end 
    
    return result[2:end, :, :]
    
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

struct TTVAR_Result

    sp
    ν0
    ν1
    Σ

end


function sample(X1:: Matrix, coef::Array{Float64,3}, σ, h)

    i = length(X1)
    result = zeros(i, h+1)
    result[:,1] = X1 

    Β = coef
    
    for t in 2:h+1
        drift = Β[t] * vec(result[:,t-1]) 
        result[:,t] .=  rand(MvNormal(drift, σ))
    end
  

    return result
    
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

 


residuals(Y, X, Β) = Y .- X * Β 
measurement_cov(X, P1, Σ) = X * P1 * X' .+ Σ
k_gain(X, P1, S) = P1 * X' ./ S
update_state(S1, K, res) = S1 .+ K * res
update_cov(P1, K, X) = P1 - K * X * P1 



function kalman_step2(Sm1:: Vector{Float64}, Pm1:: Matrix{Float64}, Y:: Float64, X:: Vector{Float64}, Σ:: Float64, Q:: Matrix{Float64})
    
    X = X'
    #predict state
    S1 = Sm1
    P1 = Pm1 +  Q

    S = measurement_cov(X, P1, Σ)
    K = k_gain(X, P1, S)
    res = residuals(Y,X,S1)
    Β_f = update_state(S1, K, res)
    P_f = update_cov(P1, K, X)

    return Β_f, P_f

end


 function kalmanFilter2(X, Y, Β0, P0, Σ, ν)
    T,i = size(X)
    
    S_filtered = zeros(T+1, i)
    P_filtered = zeros(T+1, i, i)

    S_filtered[1,:] = Β0
    P_filtered[1,:,:] = P0

   for t in 2:T+1
     S = S_filtered[t-1,:]
     P = P_filtered[t-1, :, :]
     Yt = Y[t-1]
     Xt =  X[t-1,:]  
     S_filtered[t,:], P_filtered[t,:, :] = kalman_step2(S, P, Yt, Xt, Σ, ν)

    end  
     
    return S_filtered, P_filtered 

 end 

 smooth_state(Sp1:: Vector{Float64}, S_filtered:: Vector{Float64}, P_filtered:: Matrix{Float64}, ν:: Matrix{Float64}) = S_filtered +  P_filtered * inv((P_filtered + ν)) * (Sp1 - S_filtered)
 smooth_cov(P_filtred:: Matrix{Float64}, ν:: Matrix{Float64}) = P_filtred - P_filtred * inv(P_filtred + ν) * P_filtred

function simulation_smoother(S_filtered, P_filtered, ν)

    T, i = size(S_filtered)
    S = zeros(T, i)
    
    S[end, :] .= rand(MvNormal(S_filtered[end, :], P_filtered[end, :, :])) 
    
    for t in T-1:-1:1
    
        S_smoothed = smooth_state(S[t+1, :], S_filtered[t, :], P_filtered[t, :, :], ν)
        P_smoothed = smooth_cov(P_filtered[t, :, :], ν)
        
        S[t, :] = S_smoothed + vec(randn(1,i) * cholesky(Hermitian(P_smoothed), check=false).L) #rand(MvNormal(S_smoothed, Hermitian(P_smoothed)))

    end
 
    return S

end

function carter_kohn(X, Y, B0, P0, σ, ν)

    S_filtered, P_filtered = kalmanFilter2(X, Y, B0, P0, σ, ν)
    draws = simulation_smoother(S_filtered, P_filtered, ν)

    return draws

end
    
