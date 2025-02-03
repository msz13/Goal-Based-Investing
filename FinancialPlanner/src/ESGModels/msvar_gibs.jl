
function simulate_regimes(Y, X, Β, Σ, transition_matrix, states_zero)

    P = transition_matrix
    k = size(transition_matrix, 1)

    regimes_probs = hamilton_filter(Y,X, Β, Σ, P, states_zero)

    T = size(regimes_probs)[1]
    result = zeros(Int64, T)
        
    result[end] = sample([1,2], ProbabilityWeights(regimes_probs[end,:]),1)[1] 

    for t in T-1:-1:1
        Stp1 = zeros(k)
        Stp1[result[t+1]] = 1.
        smoothed_prob = smooth_step(Stp1, regimes_probs[t, :], P)
        result[t] = sample([1, 2], ProbabilityWeights(smoothed_prob),1)[1] 

    end       
    
    return result    

end

function count_regime_transitions(regimes, k)

    result = zeros(k, k)

    for t in 1:T-1
        result[regimes[t], regimes[t+1]] += 1
    end

    return result

end

function sample_transition_matrix(regimes, k)

    result = zeros(k,k)

    transition_count = count_regime_transitions(regimes, k)

    for r in 1:k
        result[r,:] = rand(Dirichlet(transition_count[r,:])) 
    end      

    return result    

end

filter_X(X, regimes, regime) = X[regimes .== regime, :]

function sample_covariance(Y, X, Β, regimes, k)

    n_variables = size(Y,2)
    result = zeros(k, n_variables, n_variables)

    for r in 1:k
        Ym = filter_X(Y, regimes, r)
        Xm = filter_X(X, regimes, r)
        Tm = size(Ym, 1)
        U = calc_residuals(Ym,Xm, Β[r]')
        μ = 1/Tm * U' * U
        ν = Tm-k-1
        result[r,:,:] =  rand(InverseWishart(ν, μ))
        #result[r,:,:] = μ
    end   

    return result
    
end