
"""
Simulate Markov Switching VARS  
"""
predict(B::Matrix{Float64}, X::Vector{Float64}) = B * X 

function simulate_initial_regimes(initial_regimes_probs, transition_matrix, n_scenarios)
    next_regimes_probs = next_regime(initial_regimes_probs, transition_matrix)
    return sample([1,2], ProbabilityWeights(next_regimes_probs), n_scenarios)
end

function simulate_next_regimes_for_previous_regimes(previous_regimes, transition_matrix)
    return [sample([1,2], ProbabilityWeights(transition_matrix[r,:]), 1)[1] for r in previous_regimes]
end

function expected_regimes(initial_regimes_probs, transition_matrix, T)
    
    result = zeros(T, length(initial_regimes_probs))

    result[1,:] = next_regime(initial_regimes_probs, transition_matrix)

    for t in 2:T 
        result[t,:] = next_regime(result[t-1,:], transition_matrix)
    end

    return result

end

add_intercept(X, n_scenarios) = vcat(ones(n_scenarios)', X)


function  simulate_returns_step(regimes, Β, Σ, X0)
    n_variables = size(X0)[1] 
    n_scenarios =length(regimes)
    result = zeros(n_variables, n_scenarios)
    X = add_intercept(X0, n_scenarios) # add intercept
    for (i, r) in enumerate(regimes)
        result[:, i] = rand(MvNormal(predict(Β[r],X[:,i]), Σ[r]))
    end
    
    return result

end 

function simulate_regimes(initial_regimes_probs, transition_matrix, n_steps, n_scenarios) 

    result = zeros(Int64, n_steps, n_scenarios)

    
    result[1,:] = simulate_initial_regimes(initial_regimes_probs, transition_matrix, n_scenarios)

    for t in 2:n_steps

        result[t,:] =  [sample([1,2], ProbabilityWeights(transition_matrix[r,:])) for r in result[1,:]]

    end         

    return result

end 


function simulate_msvar_returns(regimes, Β, Σ, X0, n_steps, n_scenarios)
    
    n_variables = length(X0) 
    result = zeros(n_variables, n_steps, n_scenarios)
    
    result[:,1,:] = simulate_returns_step(regimes[1,:], Β, Σ, repeat(X0, 1, n_scenarios))

   for t in 2:n_steps
        next_regimes = regimes[t,:]
    
        result[:,t,:] = simulate_returns_step(next_regimes, Β, Σ, result[:,t-1,:])

    end       

    return result

end
