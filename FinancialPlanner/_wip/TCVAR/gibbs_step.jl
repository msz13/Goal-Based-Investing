
using LinearAlgebra
using Distributions

"""
    var_covariance_posterior(Y, X, prior_params)

Compute posterior distribution parameters for the covariance matrix of a VAR model 
with Minnesota priors. Returns parameters for Inverse Wishart distribution.

# Arguments
- `Y::Matrix{Float64}`: Response matrix (T × n) where T is time periods, n is variables
- `X::Matrix{Float64}`: Design matrix (T × k) where k is total number of regressors
- `prior_params::NamedTuple`: Prior parameters containing:
  - `ν₀`: Prior degrees of freedom (scalar)
  - `Ψ₀`: Prior scale matrix (n × n)
  - `B₀`: Prior mean for coefficients (k × n) 
  - `Ω₀`: Prior precision matrix for coefficients (k × k)

# Returns
- `NamedTuple` with:
  - `ν_post`: Posterior degrees of freedom
  - `Ψ_post`: Posterior scale matrix
  - `distribution`: InverseWishart distribution object
"""
function var_covariance_posterior(Y::Matrix{Float64}, X::Matrix{Float64}, prior_params)
    T, n = size(Y)  # T time periods, n variables
    k = size(X, 2)  # k regressors
    
    # Extract prior parameters
    ν₀ = prior_params.ν₀
    Ψ₀ = prior_params.Ψ₀
    B₀ = prior_params.B₀
    Ω₀ = prior_params.Ω₀
    
    # Compute posterior precision and mean for coefficients
    Ω_post = Ω₀ + X' * X
    B_post = inv(Ω_post) * (Ω₀ * B₀ + X' * Y)
    
    # Compute residual sum of squares components
    # RSS = (Y - X*B_post)'*(Y - X*B_post) + (B₀ - B_post)'*Ω₀*(B₀ - B_post)
    residuals = Y - X * B_post
    RSS_data = residuals' * residuals
    
    coeff_diff = B₀ - B_post
    RSS_prior = coeff_diff' * Ω₀ * coeff_diff
    
    # Posterior parameters for Inverse Wishart
    ν_post = ν₀ + T
    Ψ_post = Ψ₀ + RSS_data + RSS_prior
    
    # Ensure scale matrix is positive definite
    Ψ_post = (Ψ_post + Ψ_post') / 2  # Make symmetric
    
    # Create Inverse Wishart distribution
    posterior_dist = InverseWishart(ν_post, Ψ_post)
    
    return (
        ν_post = ν_post,
        Ψ_post = Ψ_post,
        distribution = posterior_dist,
        B_post = B_post,
        Ω_post = Ω_post
    )
end

"""
    minnesota_priors(n_vars, n_lags; λ₁=0.2, λ₂=0.5, λ₃=1.0, ν₀=nothing)

Generate standard Minnesota prior parameters for VAR model.

# Arguments
- `n_vars::Int`: Number of variables
- `n_lags::Int`: Number of lags
- `λ₁::Float64`: Overall tightness (default 0.2)
- `λ₂::Float64`: Cross-variable shrinkage (default 0.5) 
- `λ₃::Float64`: Lag decay rate (default 1.0)
- `ν₀::Union{Int,Nothing}`: Prior degrees of freedom (default n_vars + 2)

# Returns
- `NamedTuple` with Minnesota prior parameters
"""
function minnesota_priors(n_vars::Int, n_lags::Int; λ₁=0.2, λ₂=0.5, λ₃=1.0, ν₀=nothing)
    # Total number of coefficients (including intercept)
    k = n_vars * n_lags + 1  # +1 for intercept
    
    # Prior mean: random walk assumption (first own lag = 1, others = 0)
    B₀ = zeros(k, n_vars)
    for i in 1:n_vars
        B₀[i, i] = 1.0  # First own lag coefficient = 1
    end
    
    # Prior precision matrix (diagonal)
    Ω₀ = zeros(k, k)
    
    # Intercept has loose prior (small precision)
    Ω₀[end, end] = 1e-4
    
    # Minnesota prior for VAR coefficients
    idx = 1
    for lag in 1:n_lags
        for j in 1:n_vars  # equation j
            for i in 1:n_vars  # variable i
                if i == j
                    # Own lags: tighter prior
                    Ω₀[idx, idx] = 1 / (λ₁^2 * lag^(2*λ₃))
                else
                    # Cross-variable lags: looser prior
                    Ω₀[idx, idx] = 1 / (λ₁^2 * λ₂^2 * lag^(2*λ₃))
                end
                idx += 1
            end
        end
    end
    
    # Prior scale matrix for covariance (identity scaled by residual variance proxy)
    Ψ₀ = Matrix{Float64}(I, n_vars, n_vars) * 0.1
    
    # Prior degrees of freedom
    ν₀ = isnothing(ν₀) ? n_vars + 2 : ν₀
    
    return (
        ν₀ = ν₀,
        Ψ₀ = Ψ₀,
        B₀ = B₀,
        Ω₀ = Ω₀,
        λ₁ = λ₁,
        λ₂ = λ₂,
        λ₃ = λ₃
    )
end



# Example usage:
"""
# Generate some sample data
n_vars = 3
n_lags = 2
T = 100

# Simulated data
data = randn(T, n_vars)

# Construct VAR matrices
Y, X = construct_var_matrices(data, n_lags)

# Set up Minnesota priors
priors = minnesota_priors(n_vars, n_lags)

# Compute posterior
posterior = var_covariance_posterior(Y, X, priors)

# Sample from posterior
Σ_samples = rand(posterior.distribution, 1000)

println("Posterior degrees of freedom: ", posterior.ν_post)
println("Posterior scale matrix shape: ", size(posterior.Ψ_post))
"""


function covariance_posterior(data, scale_prior, d_posterior)
    
    res = diff(data, dims=1)
    posterior_mean = res' * res .+ scale_prior

    return InverseWishart(d_posterior, posterior_mean)    

end

#coeffs and variance with minnesota_priors secon version

using LinearAlgebra, Distributions, Statistics

# Structure to hold VAR model specifications
struct VARModel
    Y::Matrix{Float64}      # Data matrix (T x n)
    X::Matrix{Float64}      # Lagged variables matrix (T x k)
    n::Int                  # Number of variables
    p::Int                  # Number of lags
    k::Int                  # Number of regressors (n*p + constant)
    T::Int                  # Number of observations
end

# Structure for Minnesota prior hyperparameters
struct MinnesotaPriors
    λ₁::Float64    # Overall tightness
    λ₂::Float64    # Cross-variable shrinkage
    λ₃::Float64    # Lag decay
    λ₄::Float64    # Constant term prior variance
    δ::Float64     # Prior mean for own first lag (typically 1.0 for unit root)
end

# Structure for posterior distributions
struct PosteriorDistributions
    β_mean::Matrix{Float64}     # Posterior mean of coefficients
    β_var::Matrix{Float64}      # Posterior variance of coefficients
    Σ_shape::Float64           # Inverse-Wishart shape parameter
    Σ_scale::Matrix{Float64}    # Inverse-Wishart scale matrix
end

"""
    construct_var_data(Y::Matrix{Float64}, p::Int; include_constant::Bool=true)

Construct VAR data matrices from time series data.

# Arguments
- `Y`: Time series data matrix (T x n)
- `p`: Number of lags
- `include_constant`: Whether to include constant term

# Returns
- `VARModel`: Structured VAR model data
"""
function construct_var_data(Y::Matrix{Float64}, p::Int; include_constant::Bool=true)
    T_orig, n = size(Y)
    T = T_orig - p
    
    # Construct lagged variables matrix
    X = zeros(T, n * p + (include_constant ? 1 : 0))
    
    for t in 1:T
        col_idx = 1
        # Add lagged variables
        for lag in 1:p
            for var in 1:n
                X[t, col_idx] = Y[t + p - lag, var]
                col_idx += 1
            end
        end
        # Add constant term if requested
        if include_constant
            X[t, end] = 1.0
        end
    end
    
    Y_dep = Y[(p+1):end, :]
    k = size(X, 2)
    
    return VARModel(Y_dep, X, n, p, k, T)
end

"""
    minnesota_prior_moments(var_model::VARModel, priors::MinnesotaPriors; 
                           include_constant::Bool=true)

Calculate Minnesota prior means and variances for VAR coefficients.

# Arguments
- `var_model`: VARModel structure
- `priors`: MinnesotaPriors hyperparameters
- `include_constant`: Whether constant term is included

# Returns
- `β₀`: Prior mean vector
- `Ω₀`: Prior variance matrix (diagonal)
"""
function minnesota_prior_moments(var_model::VARModel, priors::MinnesotaPriors; 
                                include_constant::Bool=true)
    n, p, k = var_model.n, var_model.p, var_model.k
    
    β₀ = zeros(k)
    Ω₀_diag = zeros(k)
    
    # Calculate sample variances for scaling (use first p+1 observations)
    σ² = var(var_model.Y[1:(min(size(var_model.Y, 1), p+10)), :], dims=1)
    σ² = max.(σ²[:], 1e-8)  # Avoid division by zero
    
    col_idx = 1
    
    # Prior for lagged coefficients
    for lag in 1:p
        for j in 1:n  # equation j
            for i in 1:n  # variable i
                if i == j && lag == 1
                    # Own first lag: prior mean = δ (usually 1 for unit root)
                    β₀[col_idx] = priors.δ
                    # Own first lag variance
                    Ω₀_diag[col_idx] = (priors.λ₁)^2
                elseif i == j
                    # Own higher lags: prior mean = 0
                    β₀[col_idx] = 0.0
                    # Own higher lag variance with decay
                    Ω₀_diag[col_idx] = (priors.λ₁ / lag^priors.λ₃)^2
                else
                    # Cross-variable lags: prior mean = 0
                    β₀[col_idx] = 0.0
                    # Cross-variable variance with additional shrinkage
                    Ω₀_diag[col_idx] = (priors.λ₁ * priors.λ₂ * σ²[i] / σ²[j] / lag^priors.λ₃)^2
                end
                col_idx += 1
            end
        end
    end
    
    # Prior for constant term
    if include_constant
        β₀[end] = 0.0
        Ω₀_diag[end] = priors.λ₄^2
    end
    
    Ω₀ = diagm(Ω₀_diag)
    
    return β₀, Ω₀
end

"""
    posterior_coefficients(var_model::VARModel, priors::MinnesotaPriors; 
                          include_constant::Bool=true)

Calculate posterior distribution for VAR coefficients with Minnesota priors.

# Arguments
- `var_model`: VARModel structure
- `priors`: MinnesotaPriors hyperparameters
- `include_constant`: Whether constant term is included

# Returns
- Named tuple with posterior mean and variance of coefficients
"""
function posterior_coefficients(var_model::VARModel, priors::MinnesotaPriors; 
                               include_constant::Bool=true)
    Y, X = var_model.Y, var_model.X
    T, k, n = var_model.T, var_model.k, var_model.n
    
    # Get prior moments
    β₀, Ω₀ = minnesota_prior_moments(var_model, priors; include_constant=include_constant)
    Ω₀_inv = inv(Ω₀)
    
    # Calculate posterior moments equation by equation
    β_post = zeros(k, n)
    V_post = zeros(k, k, n)
    
    for j in 1:n  # For each equation
        y_j = Y[:, j]
        
        # Posterior precision matrix
        V_post_inv = Ω₀_inv + X' * X
        V_post[:, :, j] = inv(V_post_inv)
        
        # Posterior mean
        β_post[:, j] = V_post[:, :, j] * (Ω₀_inv * β₀ + X' * y_j)
    end
    
    return (β_mean = β_post, β_var = V_post)
end

"""
    posterior_covariance(var_model::VARModel, priors::MinnesotaPriors; 
                        include_constant::Bool=true, ν₀::Float64=n+2.0)

Calculate posterior distribution for error covariance matrix.

# Arguments
- `var_model`: VARModel structure  
- `priors`: MinnesotaPriors hyperparameters
- `include_constant`: Whether constant term is included
- `ν₀`: Prior degrees of freedom for inverse-Wishart

# Returns
- Named tuple with Inverse-Wishart shape and scale parameters
"""
function posterior_covariance(var_model::VARModel, priors::MinnesotaPriors; 
                             include_constant::Bool=true, ν₀::Float64=0.0)
    Y, X = var_model.Y, var_model.X
    T, n = var_model.T, var_model.n
    
    # Set default prior degrees of freedom if not specified
    if ν₀ <= 0.0
        ν₀ = Float64(n + 2)
    end
    
    # Get posterior coefficients
    post_coef = posterior_coefficients(var_model, priors; include_constant=include_constant)
    β_mean = post_coef.β_mean
    
    # Calculate residuals using posterior mean
    residuals = Y - X * β_mean
    S = residuals' * residuals
    
    # Prior scale matrix (could be made more sophisticated)
    S₀ = Matrix{Float64}(I, n, n) * 0.1  # Weak prior
    
    # Posterior parameters for inverse-Wishart distribution
    ν_post = ν₀ + T
    S_post = S₀ + S
    
    return (shape = ν_post, scale = S_post)
end

"""
    sample_var_posterior(var_model::VARModel, priors::MinnesotaPriors, n_draws::Int=1000;
                        include_constant::Bool=true)

Sample from the joint posterior distribution of VAR parameters.

# Arguments
- `var_model`: VARModel structure
- `priors`: MinnesotaPriors hyperparameters  
- `n_draws`: Number of posterior draws
- `include_constant`: Whether constant term is included

# Returns
- Named tuple with posterior samples of coefficients and covariance matrix
"""
function sample_var_posterior(var_model::VARModel, priors::MinnesotaPriors, n_draws::Int=1000;
                             include_constant::Bool=true)
    n, k = var_model.n, var_model.k
    
    # Get posterior distributions
    post_coef = posterior_coefficients(var_model, priors; include_constant=include_constant)
    post_cov = posterior_covariance(var_model, priors; include_constant=include_constant)
    
    # Storage for samples
    β_draws = zeros(k, n, n_draws)
    Σ_draws = zeros(n, n, n_draws)
    
    for i in 1:n_draws
        # Sample covariance matrix from inverse-Wishart
        Σ_draws[:, :, i] = rand(InverseWishart(post_cov.shape, post_cov.scale))
        
        # Sample coefficients equation by equation
        for j in 1:n
            β_draws[:, j, i] = rand(MvNormal(post_coef.β_mean[:, j], 
                                           Hermitian(post_coef.β_var[:, :, j])))
        end
    end
    
    return (β_samples = β_draws, Σ_samples = Σ_draws)
end

# Example usage:
# Create example data
# T, n, p = 100, 3, 2
# Y = randn(T, n)  # Replace with your data
# 
# # Set up Minnesota priors
# priors = MinnesotaPriors(0.1, 0.5, 1.0, 100.0, 1.0)
# 
# # Construct VAR model
# var_model = construct_var_data(Y, p)
# 
# # Get posterior moments
# β₀, Ω₀ = minnesota_prior_moments(var_model, priors)
# post_coef = posterior_coefficients(var_model, priors)
# post_cov = posterior_covariance(var_model, priors)
# 
# # Sample from posterior
# samples = sample_var_posterior(var_model, priors, 1000)