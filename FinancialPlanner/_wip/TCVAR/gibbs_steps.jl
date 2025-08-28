function covariance_posterior(data, scale_prior, d_posterior)
    
    res = diff(data, dims=1)
    posterior_mean = res' * res .+ scale_prior

    return InverseWishart(d_posterior, posterior_mean)    

end

function prepare_var_data(Y::Matrix{Float64}, p::Int, X::Union{Matrix{Float64},Vector{Float64}} = Matrix{Float64}(undef, 0, 0), add_intercept::Bool = false)
    T, n = size(Y)
    Y_lagged = zeros(T - p, n * p)
    for t in (p + 1):T
        Y_lagged[t - p, :] = Y[t-p:t-1, :]'
    end

    predictors = Y_lagged

    if !isempty(X)
        if size(X, 1) != T
            error("The number of rows in X must be equal to the number of rows in Y.")
        end
        X_subset = X[p+1:end, :]
        predictors = hcat(predictors, X_subset)
    end

    if add_intercept
        intercept = ones(T - p, 1)
        predictors = hcat(intercept, predictors)
    end

    return Y[p+1:end, :], predictors
end


posterior_beta_coefficient_mean(Y, X, beta_mean, beta_var_inv) = inv(X'X + beta_var_inv)*(X'Y + beta_var_inv*beta_mean)


posterior_beta_coefficient_var(X, Σ, Ω_inv) = kron(Σ, inv(X'X + Ω_inv))
