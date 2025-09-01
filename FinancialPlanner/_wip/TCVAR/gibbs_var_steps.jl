
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


posterior_beta_coefficient_mean(Y, X, beta_mean, Ω_inv) = inv(X'X + Ω_inv)*(X'Y + Ω_inv*beta_mean)


#posterior_beta_coefficient_var(X, Σ, Ω_inv) = kron(Σ, inv(X'X + Ω_inv)) do usuniecia

#Ω_inv prior of beta coefficient variance

function beta_posterior_dist(X, Σ, beta_posterior, Ω_inv)
      
    beta_var = kron(Σ, inv(X'X + Ω_inv))
    return MvNormal(vec(beta_posterior), beta_var)
  
end


function covariance_posterior_dist(Y, X, β_posterior_μ, posterior_df, variance_prior, β_prior_μ, Ω_inv)

    ε =  Y - X * β_posterior_μ'

    β_diff = β_posterior_μ - β_prior_μ

    S = ε' * ε + β_diff' * Ω_inv * β_diff  + variance_prior

    return InverseWishart(posterior_df, S)

end



"""
    sample_var_params(data,p, β_mean, Ω_inv)

    data: observations
    p: number of lags
    β_priormean: prior mean of beta coefficients     
    Ω_inv: inversion prior variance of beta coefficients 
    S: prior covariance scale
    df: posterior covariance distribution degrees of freedom    
"""
function sample_var_params(data, p, β_prior_μ, Ω_inv, S, df)

    Y, X = prepare_var_data(data, p)
    

    
    β_hat = posterior_beta_coefficient_mean(Y, X, β_prior_μ, Ω_inv) #mean of posterior distribution of coefficients

    Σ = rand(covariance_posterior_dist(Y, X, β_hat, df, S, β_prior_μ, Ω_inv))

    β = rand(beta_posterior_dist(X, Σ, β_hat, Ω_inv))

   

    return β, Σ

end
  
 