
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
    
    X = data[1:end-1, :]
    Y = data[2:end, :]
    res = Y -X
    posterior_mean = res' * res .+ scale_prior

    return InverseWishart(d_posterior, posterior_mean)    

end