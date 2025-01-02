next_regime(previous_probs, transition_matrix) = transition_matrix' * previous_probs


function likehood_t(y, X, Β, Σ)
    
    k = length(Β)
    result = [pdf(MvNormal(vec(X' * Β[s]), Σ[s]), y) for s in 1:k]

    return result

end

function hamilton_step(y, X, Β, Σ, transition_matrix, states_zero)

    s = next_regime(states_zero, transition_matrix)
    η = likehood_t(y, X, Β, Σ)
    
    probs = η .* s 
   
    return return probs / sum(probs)
    
end


function hamilton_filter(Y, X, Β, Σ, transition_matrix, states_zero)

    T = size(Y)[1]
    n_states = length(states_zero)
    result = zeros(T, n_states)

    result[1,:] = hamilton_step(Y[1,:], X[1,:], Β, Σ, transition_matrix, states_zero)
    
    for t in 2:T
        result[t,:] = hamilton_step(Y[t,:], X[t,:], Β, Σ, transition_matrix, result[t-1, :])
    end
  
    return return result
    
end

smooth_step(St1T, St1, St, transition_matrix) = transition_matrix * (St1T ./ St1) .* St

"""
#p: lag 

result = MSBVAR(Y, p, n_regimes, n_burning, n_draw, intercept_switching=true, coef_switching=true, covariance_switching=true)


result.B[1][1].B0
result.B[1].

reult:
sample, regime, var struct
result, param, sample

"""

"""
model: fitted MSVAR model
X0: current data
S0: current regimes probilities
n_steps: number of steps
n_scenarios: number of n_scenarios

simulated = simulate(model,X0, S0, n_steps,n_scenarios)

"""



