function evaluate_all_regimes_sampled(regimes, n_regimes, n_variables)
    
    regime_count = [count(x -> x == r, regimes) for r in 1:n_regimes ]
    
    return all(regime_count .>= n_variables+3) 

end


function simulate_regimes(Y, X, Β, Σ, transition_matrix, states_zero)

    P = transition_matrix
    k = size(transition_matrix, 1) # n_regimes

    regimes_probs = hamilton_filter(Y,X, Β, Σ, P, states_zero)
    n_variables = size(X, 2)

    T = size(regimes_probs)[1]
    result = zeros(Int64, T)

    result[end] = sample(1:k, ProbabilityWeights(regimes_probs[end,:]),1)[1] 
    for t in T-1:-1:1
        Stp1 = zeros(k)
        Stp1[result[t+1]] = 1.
        smoothed_prob = smooth_step(Stp1, regimes_probs[t, :], P)
        result[t] = sample(1:k, ProbabilityWeights(smoothed_prob),1)[1] 
    end   
        
   

#=     while true
        result[end] = sample(1:k, ProbabilityWeights(regimes_probs[end,:]),1)[1] 
        for t in T-1:-1:1
            Stp1 = zeros(k)
            Stp1[result[t+1]] = 1.
            smoothed_prob = smooth_step(Stp1, regimes_probs[t, :], P)
            result[t] = sample(1:k, ProbabilityWeights(smoothed_prob),1)[1] 
        end   
        if evaluate_all_regimes_sampled(result, k, n_variables)
            break
        end
    end     =#
    
    return result    

end

function count_regime_transitions(regimes, k)

    T = size(regimes, 1)
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
        result[r,:] = rand(Dirichlet(transition_count[r,:] .+ .001)) 
    end      

    return result    

end

filter_X(X, regimes, regime) = X[regimes .== regime, :]

function sample_covariance(Y, X, Β, regimes, k)

    n_variables = size(Y,2)
    #result = zeros(k, n_variables, n_variables)
    result = []

    for r in 1:k
        Ym = filter_X(Y, regimes, r)
        Xm = filter_X(X, regimes, r)
        Tm = size(Ym, 1)
        U = calc_residuals(Ym,Xm, Β[r]')
        μ = U' * U + diagm(fill(1e-8,n_variables))
        ν = Tm - 1 #Tm-k-1 #n_variables + Tm
        
        if isposdef(μ) != true 
            throw("sigma not positive define")
        end
        push!(result, rand(InverseWishart(ν, μ)))
    end   

    return result
    
end

function sample_betas(Y,X,regimes, posterior_sigmas, k)

    n_variables = size(Y,2)
    #result = zeros(k, n_variables+1, n_variables)
    result = []

    for r in 1:k
        Ym = filter_X(Y, regimes, r)
        Xm = filter_X(X, regimes, r)
        Beta_mean = []

        try
            Beta_mean = inv(Xm' * Xm) * Xm' * Ym
        catch e
            isa(e, LinearAlgebra.SingularException)
            println("singular exeption, length Y $(size(Ym, 1))")
        end
        Beta_var = kron(posterior_sigmas[r], Hermitian(inv(Xm'* Xm)))
        Βm = rand(MvNormal(vec(Beta_mean), Beta_var))
        #result[r,:,:]  = reshape(Βm, n_variables+1, n_variables)
        push!(result, reshape(Βm, n_variables+1, n_variables)')

    end
    
    return result

end

function msvar(Y, X, transition_matrix0, Β0, Σ0, n_burn, n_samples)

    T = size(Y, 1) 
    n_variables = size(X, 2)
    n = n_burn + n_samples
    k = size(transition_matrix0, 1) # n_regimes
    states = zeros(Int64, n, T)
    t_m = zeros(n, k, k)

    states_zero = unconditional_regimes(transition_matrix0)
    states[1, :] = simulate_regimes(Y, X, Β0, Σ0, transition_matrix0, states_zero)
    t_m[1, :, :] = sample_transition_matrix(states[1, :], k)

    cov_sample = [sample_covariance(Y, X, Β0, states[1,:], k)]  

    Β_sample = [sample_betas(Y,X,states[1,:], cov_sample[1],k)]
    

   for s in 2:n
        regimes = simulate_regimes(Y, X, Β_sample[s-1], cov_sample[s-1], t_m[s-1, :, :], states_zero)
        states[s, :] = evaluate_all_regimes_sampled(regimes, k, n_variables) ? regimes : states[s-1, :]
        t_m[s, :, :] = sample_transition_matrix(states[s, :], k)
        #cov_sample[n, :, :, :] = sample_covariance(Y, X, Β0, states[s,:], k)
        push!(cov_sample, sample_covariance(Y, X, Β_sample[s-1], states[s,:], k))
        push!(Β_sample, sample_betas(Y,X,states[s,:], cov_sample[s],k))
        states_zero = unconditional_regimes(t_m[s, :, :])
    end 
     
    return states[n_burn+1:end, :], t_m[n_burn+1:end, :, :], Β_sample[n_burn+1:end], cov_sample[n_burn+1:end]

end