
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


@model function mv_model(Y::Matrix{Float64})
    T, n = size(Y)   

    # Priors for mean (Normal)
    μ ~ filldist(Normal(0,5.0), n)  
      
   
    # Prior for the covariance matrix (Inverse Wishart)
    Σ ~ InverseWishart(T-1, Matrix(I(n)))

    # Likelihood
    for t in 1:T
        Y[t, :] ~ MvNormal(μ, Symmetric(Σ))
    end

    return μ, Σ
end

#= # Define a custom MvNormal constructor using multiple dispatch
function MvNormal(μ::AbstractVector{T}, σ::AbstractVector{T}, L::Cholesky) where {T<:Real}
    # Validate inputs
 #=    d = length(μ)
    length(σ) == d || throw(ArgumentError("Length of μ and σ must match"))
    size(L.L, 1) == d || throw(ArgumentError("Cholesky factor dimension must match μ and σ"))
    all(σ .> 0) || throw(ArgumentError("Standard deviations must be positive")) =#

    # Construct covariance matrix: Σ = D * L * L' * D
    D = Diagonal(σ)
    Σ = PDMat(Cholesky(D * L.L + eps() * I))

    # Call the standard MvNormal constructor
    return MvNormal(μ, Σ)
end =#

# Bayesian multivariate normal model with LKJCholesky prior
@model function mvnormal_lkjcholesky_model(Y, K, T)
    # Priors
    # Mean vector
    μ ~ filldist(Normal(0, 5.), K)  # Normal means
    
    # Standard deviations (diagonal of D)
    σ ~ filldist(truncated(Normal(0, 5.), lower=0), K)  # Positive standard deviations
    
    # Cholesky factor of correlation matrix (LKJCholesky)
    η = 1.0  # Shape parameter (1.0 = uniform over correlations)
    L ~ LKJCholesky(K, η)
    
    # Construct covariance matrix: Σ = D L L^T D
    D = Diagonal(σ)
    Σ = D * L.L
    Σ_est = PDMat(Cholesky(Σ + eps() * I)) 

      
    # Likelihood
    for t in 1:T
        Y[t, :] ~ MvNormal(μ, Σ_est)
    end

    return (;Omega = Matrix(L))
end

# Bayesian multivariate normal model with LKJCholesky prior
@model function mvnormal_lkjcholesky_model2(Y, K, T)
    # Priors
    # Mean vector
    μ ~ filldist(Normal(0, 5.), K)  # Normal means
    
    # Standard deviations (diagonal of D)
    σ ~ filldist(truncated(Normal(0, 5.), lower=0), K)  # Positive standard deviations
    
    # Cholesky factor of correlation matrix (LKJCholesky)
    η = 1.0  # Shape parameter (1.0 = uniform over correlations)
    L ~ LKJCholesky(K, η)
    
      
    
    # Likelihood
    for t in 1:T
        Y[t, :] ~ MvNormal(μ, σ, L)
    end

    return (;Omega = Matrix(L))
end


@model function var_model(Y::Matrix{Float64}, predictors::Matrix{Float64}, p::Int, num_exo::Int = 0)
    T_eff, num_regressors = size(predictors)
    n = size(Y, 2)

    # Priors for the coefficients (Normal)
    μ_β ~ filldist(Normal(0, 5.), n*num_regressors)    
    β = reshape(μ_β, n, num_regressors) # Reshape to n x (pn + k) matrix

     # Standard deviations (diagonal of D)
    σ ~ filldist(truncated(Normal(0, .2), lower=0), n)  # Positive standard deviations
    
    # Cholesky factor of correlation matrix (LKJCholesky)
    η = 1.0  # Shape parameter (1.0 = uniform over correlations)
    L ~ LKJCholesky(n, η)
    
    # Construct covariance matrix: Σ = D L L^T D
    D = Diagonal(σ)
    Σ = D * L.L
    Σ_est = PDMat(Cholesky(Σ + eps() * I)) 
   

    # Likelihood
    for t in 1:T_eff
        #yt = Y[t, :]
        
        μ = β * predictors[t, :]
        Y[t, :] ~ MvNormal(μ, Σ_est)
    end

    return β, Σ
end