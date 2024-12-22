module MSVAR

using Distributions

export hamilton_filter

function next_regime(previous_probs, transition_matrix)

    return transition_matrix' * previous_probs
end


function likehood_t(y, X, Β, Σ)
    
    k = length(Β)
    result = [pdf(MvNormal(vec(X' * Β[s]), Σ[s]), y) for s in 1:k]

    return result

end

function hamilton_filter(y, X, Β, Σ, transition_matrix, states_zero)

    s = next_regime(states_zero, transition_matrix)
    η = likehood_t(y,X, Β, Σ)
    probs = η .* s 

    return probs / sum(probs)
    
end

end