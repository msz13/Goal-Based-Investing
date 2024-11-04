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


function sample(X1:: Matrix, Β0:: Matrix, params:: TTVAR_Result, h)

    i = length(X1)
    result = zeros(i, h+1)
    result[:,1] = X1 

    Β = generate_coeffs(Β0, params.sp, params.ν0, params.ν1, h)

    for t in 2:h+1
        drift = Β[t] * vec(result[:,t-1]) 
        result[:,t] .=  rand(MvNormal(drift, params.Σ))
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

 residuals(Y, X, Β) = Y .- Β * X 

 k_gain(P_predicted, X, Σ) =  P_predicted * X' * inv(X * P_predicted * X' .+ Σ)
 
function kalman_step(Sm1, Pm1, Y, X, Σ, Q)
    
    #predict state
    S = Sm1
    P1 = Pm1 +  Q

    F = X * P1 * X' + Σ  #state transision
    K = P1 * X' * inv(F) # kalman gain

    res = Y .- S * X #residuals

    Β_f = S + K * res #Beta measurement
    P_f = P1 - K * X * P1 #covariance measurmenet

    return Β_f, P_f

end


 function kalmanFilter(X, Y, Β0, P0, Σ, ν)
    T,i = size(X)
    
    S_filtered = zeros(T+1, i,i)
    P_filtered = zeros(T+1, i, i)

    S_filtered[1,:,:] .= Β0
    P_filtered[1,:,:] .= P0

    for t in 2:T+1

        S_filtered[t,:,:], P_filtered[t,:,:] = kalman_step(S_filtered[t-1,:,:], P_filtered[t-1,:,:], Y[t-1,:], X[t-1,:,:], Σ, ν)

    end  
     
    return S_filtered, P_filtered

 end 
