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


calc_xhat(X, regime_probs) = regime_probs.^.5 * X

"""
Y: y
Xhat: transformed variabes 
regime_probs: probabilieties of regimes
"""
calc_regime_coeficientents(Y, Xhat, regime_probs) = inv(Xhat'regime_probs*Xhat)*(Xhat'regime_probs)*Y

calc_residuals(Y, X, Β) = Y - X * Β

calc_cov_matrix(residuals, regime_probs) = inv(tr(regime_probs)) * residuals' * regime_probs * residuals 

function est_regimes_params(Y, X, regimes_probs)

    n_regimes = size(regimes_probs)[2]
    coef = []
    covm = []
    

    for r in 1:n_regimes
        regime_matrix = diagm(regimes_probs[:,r])
        Xhat = calc_xhat(X, regime_matrix) 

        Β = calc_regime_coeficientents(Y, Xhat, regime_matrix)

        U = calc_residuals(Y, Xhat, Β)

        Σ = calc_cov_matrix(U, regime_matrix)

        push!(coef, Β)
        push!(covm, Σ)
    end

    return coef, covm

end

function joined_regimes_probs(regime_probs, smoothed_probs, states_zero, transition_matrix)

    T, n = size(regime_probs)
    result = zeros(T, n^2)

    result[1,:] = vec(transition_matrix) .* (kron(smoothed_probs[1,:] ./ regime_probs[1,:], states_zero))

    for t in 1:T-1
        result[t+1,:] = vec(transition_matrix) .* (kron(smoothed_probs[t+1,:] ./ regime_probs[t+1,:], regime_probs[t, :]))
    end
    
    return result

end

function est_transition_matrix(joined_regimes_probs, regimes_probs, initial_regimes_probs)

    s = sum(regimes_probs[1:end-1,:], dims=1)[1,:] + initial_regimes_probs

    k = kron([1, 1], s)

    e_tm = sum(joined_regimes_probs, dims=1)[1,:] ./ k
    e_tm = reshape(e_tm, 2, 2)
 
    return e_tm
end

