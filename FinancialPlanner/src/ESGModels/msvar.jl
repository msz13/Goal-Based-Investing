next_regime(previous_probs, transition_matrix) = transition_matrix' * previous_probs


function likehood_t(y, X, Β, Σ)
    
    k = length(Β)
    result = [pdf(MvNormal(vec(Β[s] * X), Σ[s]), y) for s in 1:k]

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

smooth_step(Stp1T, St, transition_matrix) = (transition_matrix * (Stp1T ./ next_regime(St, transition_matrix))) .* St

function smoother(regime_probs, transition_matrix)

    result = zeros(size(regime_probs))
    T = size(regime_probs)[1]

    result[end,:] = regime_probs[end,:]

    for t in T-1:-1:1
        result[t,:] = smooth_step(result[t+1,:], regime_probs[t,:], transition_matrix)
    end

    return result

end



