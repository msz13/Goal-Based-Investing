using Optim
using LinearAlgebra
using Statistics
using Distributions
using Random
using Printf

"""
GARCH-in-Mean Model with Exogenous Variables

Mean equation: y_t = μ + λ*h_t + X_t'β + ε_t
Variance equation: h_t = ω + α*ε_{t-1}^2 + γ*h_{t-1} + Z_t'δ

where:
- y_t: dependent variable
- h_t: conditional variance (enters the mean equation)
- X_t: exogenous variables for mean equation
- Z_t: exogenous variables for variance equation
- ε_t ~ N(0, h_t)
"""

struct GARCHMResults
    params::Vector{Float64}
    stderr::Vector{Float64}  
    tstat::Vector{Float64}  
    pvalues::Vector{Float64} 
    loglik::Float64
    aic::Float64
    bic::Float64
    n_obs::Int
    param_names::Vector{String}
    fitted_mean::Vector{Float64}
    fitted_variance::Vector{Float64}
    residuals::Vector{Float64}
    standardized_residuals::Vector{Float64}
end

function garch_in_mean_likelihood(params, y, X, Z)
    """
    Compute negative log-likelihood for GARCH-in-Mean model
    
    Parameters:
    -----------
    params: [μ, λ, β..., ω, α, γ, δ...]
    y: dependent variable (T x 1)
    X: exogenous variables for mean equation (T x k_x)
    Z: exogenous variables for variance equation (T x k_z)
    """
    T = length(y)
    k_x = size(X, 2)
    k_z = size(Z, 2)
    
    # Extract parameters
    μ = params[1]
    λ = params[2]  # GARCH-in-Mean coefficient
    β = params[3:(2+k_x)]
    ω = params[3+k_x]
    α = params[4+k_x]
    γ = params[5+k_x]
    δ = params[(6+k_x):end]
    
    # Parameter constraints
    if ω <= 0 || α < 0 || γ < 0 || (α + γ) >= 1
        return Inf
    end
    
    # Initialize
    h = zeros(T)
    ε = zeros(T)
    loglik = 0.0
    
    # Initial variance (unconditional variance estimate)
    h[1] = var(y)
    
    for t in 1:T
        # Mean equation with GARCH-in-Mean term
        mean_t = μ + λ * h[t] + dot(X[t, :], β)
        ε[t] = y[t] - mean_t
        
        # Variance equation with exogenous variables
        if t < T
            h[t+1] = ω + α * ε[t]^2 + γ * h[t] + dot(Z[t, :], δ)
            
            # Ensure positive variance
            if h[t+1] <= 0
                return Inf
            end
        end
        
        # Log-likelihood contribution
        loglik += -0.5 * (log(2π) + log(h[t]) + ε[t]^2 / h[t])
    end
    
    return -loglik  # Return negative for minimization
end

function estimate_garch_in_mean(y::Vector{Float64}, 
                                X::Matrix{Float64}, 
                                Z::Matrix{Float64};
                                verbose::Bool=true)
    """
    Estimate GARCH-in-Mean model with exogenous variables
    
    Parameters:
    -----------
    y: dependent variable (T x 1)
    X: exogenous variables for mean equation (T x k_x)
    Z: exogenous variables for variance equation (T x k_z)
    verbose: print optimization progress
    
    Returns:
    --------
    GARCHMResults object with estimation results
    """
    T = length(y)
    k_x = size(X, 2)
    k_z = size(Z, 2)
    n_params = 3 + k_x + k_z + 2  # μ, λ, β, ω, α, γ, δ
    
    # Initial parameter values
    μ_init = mean(y)
    λ_init = 0.0  # GARCH-in-Mean coefficient
    β_init = zeros(k_x)
    ω_init = var(y) * 0.1
    α_init = 0.05
    γ_init = 0.90
    δ_init = zeros(k_z)
    
    initial_params = vcat(μ_init, λ_init, β_init, ω_init, α_init, γ_init, δ_init)
    
    # Optimization
    if verbose
        println("Starting GARCH-in-Mean estimation...")
        println("Number of observations: $T")
        println("Mean equation exogenous variables: $k_x")
        println("Variance equation exogenous variables: $k_z")
    end
    
    objective(p) = garch_in_mean_likelihood(p, y, X, Z)
    
    result = optimize(objective, initial_params, BFGS(), 
                     Optim.Options(iterations=10000, show_trace=verbose))
    
    params = Optim.minimizer(result)
    loglik = -Optim.minimum(result)
    
    # Compute standard errors using numerical Hessian
   #=  hessian_result = Optim.hessian!(objective, params)
    
    # Fisher Information Matrix and standard errors
    try
        inv_hessian = inv(hessian_result)
        stderr = sqrt.(diag(inv_hessian))
        tstat = params ./ stderr
        pvalues = 2 .* (1 .- cdf.(Normal(), abs.(tstat)))
    catch
        println("Warning: Could not compute standard errors (Hessian not positive definite)")
        stderr = fill(NaN, n_params)
        tstat = fill(NaN, n_params)
        pvalues = fill(NaN, n_params)
    end
     =#
    stderr = zeros(n_params)
    tstat =  zeros(n_params)
    pvalues = zeros(n_params)

    # Information criteria
    aic = -2 * loglik + 2 * n_params
    bic = -2 * loglik + log(T) * n_params
    
    # Fitted values and residuals
    μ_hat = params[1]
    λ_hat = params[2]
    β_hat = params[3:(2+k_x)]
    ω_hat = params[3+k_x]
    α_hat = params[4+k_x]
    γ_hat = params[5+k_x]
    δ_hat = params[(6+k_x):end]
    
    h_fitted = zeros(T)
    ε_fitted = zeros(T)
    mean_fitted = zeros(T)
    
    h_fitted[1] = var(y)
    
    for t in 1:T
        mean_fitted[t] = μ_hat + λ_hat * h_fitted[t] + dot(X[t, :], β_hat)
        ε_fitted[t] = y[t] - mean_fitted[t]
        
        if t < T
            h_fitted[t+1] = ω_hat + α_hat * ε_fitted[t]^2 + γ_hat * h_fitted[t] + dot(Z[t, :], δ_hat)
        end
    end
    
    std_resid = ε_fitted ./ sqrt.(h_fitted)
    
    # Parameter names
    param_names = ["μ", "λ (GARCH-M)"]
    for i in 1:k_x
        push!(param_names, "β_$i")
    end
    push!(param_names, "ω")
    push!(param_names, "α")
    push!(param_names, "γ")
    for i in 1:k_z
        push!(param_names, "δ_$i")
    end
    
    return GARCHMResults(params, stderr, tstat, pvalues, loglik, aic, bic, 
                         T, param_names, mean_fitted, h_fitted, 
                         ε_fitted, std_resid)
end

function print_results(results::GARCHMResults)
    """Print estimation results in a formatted table"""
    println("\n" * "="^70)
    println("GARCH-in-Mean Model Estimation Results")
    println("="^70)
    println("Number of observations: $(results.n_obs)")
    println("Log-likelihood: $(round(results.loglik, digits=4))")
    println("AIC: $(round(results.aic, digits=4))")
    println("BIC: $(round(results.bic, digits=4))")
    println("\n" * "-"^70)
    #println(@sprintf("%-15s %12s %12s %12s %12s", "Parameter", "Estimate", "Std. Error", "t-stat", "p-value"))
    println("-"^70)
    
    for i in 1:length(results.params)
        println(@sprintf("%-15s %12.6f %12.6f %12.4f %12.4f", 
                        results.param_names[i], 
                        results.params[i],
                        results.stderr[i],
                        results.tstat[i],
                        results.pvalues[i]))
    end
    println("="^70)
end

# Example usage
function example_estimation()
    """Generate synthetic data and estimate the model"""
    Random.seed!(123)
    T = 1000
    
    # Generate exogenous variables
    X = randn(T, 2)  # 2 exogenous variables for mean equation
    Z = randn(T, 1)  # 1 exogenous variable for variance equation
    
    # True parameters
    μ_true = 0.1
    λ_true = 0.5  # GARCH-in-Mean effect
    β_true = [0.3, -0.2]
    ω_true = 0.08
    α_true = 0.1
    γ_true = 0.85
    δ_true = [0.05]
    
    # Simulate GARCH-in-Mean process
    y = zeros(T)
    h = zeros(T)
    ε = zeros(T)
    
    h[1] = ω_true / (1 - α_true - γ_true)
    
    for t in 1:T
        mean_t = μ_true + λ_true * h[t] + dot(X[t, :], β_true)
        ε[t] = sqrt(h[t]) * randn()
        y[t] = mean_t + ε[t]
        
        if t < T
            h[t+1] = ω_true + α_true * ε[t]^2 + γ_true * h[t] + dot(Z[t, :], δ_true)
        end
    end
    
    println("Simulated data with true parameters:")
    println("μ = $μ_true, λ = $λ_true, β = $β_true")
    println("ω = $ω_true, α = $α_true, γ = $γ_true, δ = $δ_true")
    
    # Estimate the model
    results = estimate_garch_in_mean(y, X, Z, verbose=false)
    print_results(results)
    
    return results
end

# Run the example
println("GARCH-in-Mean Model with Exogenous Variables")
println("This implementation includes:")
println("- GARCH-in-Mean: conditional variance enters the mean equation")
println("- Exogenous variables in both mean and variance equations")
println("- Maximum likelihood estimation using BFGS optimization")
println("\nRunning example with simulated data...\n")

results = example_estimation()
