using LinearAlgebra
using Distributions
using Random

"""
Compute VAR residuals (y_star) from data and coefficients
y: data matrix (T x k), where T is time periods and k is number of variables
X: design matrix (T x (k*p + 1)), includes lags and intercept
β: coefficient matrix (k*p + 1) x k, VAR coefficients
p: lag order

Returns: y_star (T x k) residuals
"""
function compute_residuals(y, X, β)
    # y_star = y - X*β
    y_star = y - X * β
    return y_star
end

"""
Construct design matrix for VAR(p) model
y: data matrix (T x k)
p: lag order
include_intercept: whether to include intercept (default true)

Returns: 
- X: design matrix ((T-p) x (k*p + include_intercept))
- y_eff: effective y after removing initial p observations ((T-p) x k)
"""
function construct_var_design_matrix(y, p; include_intercept=true)
    T, k = size(y)
    T_eff = T - p
    
    # Number of regressors per equation
    n_regs = k * p + (include_intercept ? 1 : 0)
    
    X = zeros(T_eff, n_regs)
    
    for t in (p+1):T
        row_idx = t - p
        col_idx = 1
        
        # Add lagged values
        for lag in 1:p
            for j in 1:k
                X[row_idx, col_idx] = y[t - lag, j]
                col_idx += 1
            end
        end
        
        # Add intercept
        if include_intercept
            X[row_idx, col_idx] = 1.0
        end
    end
    
    y_eff = y[(p+1):end, :]
    
    return X, y_eff
end

"""
Compute residuals directly from data, coefficients, and lag order
y: raw data matrix (T x k)
β: coefficient matrix ((k*p + 1) x k) for VAR(p)
p: lag order

Returns: y_star (T-p x k) residuals
"""
function compute_var_residuals(y, β, p)
    X, y_eff = construct_var_design_matrix(y, p)
    y_star = compute_residuals(y_eff, X, β)
    return y_star
end

"""
Sample correlation parameters (off-diagonal elements of Σ) in VAR-SV model
Uses KSC approximation for log(y²) ≈ mixture of normals
"""
function sample_correlation_params(y_star, h, prior_R0_inv, prior_r0, n, k)
    # y_star: demeaned residuals (T x k)
    # h: log-volatilities (T x k)
    # prior_R0_inv: prior precision matrix for correlations
    # prior_r0: prior mean for correlations
    # n: dimension of VAR
    # k: number of variables
    
    T = size(y_star, 1)
    
    # Standardize residuals by exp(h/2)
    y_std = similar(y_star)
    for t in 1:T
        for i in 1:k
            y_std[t, i] = y_star[t, i] / exp(h[t, i] / 2)
        end
    end
    
    # For each pair of variables, sample correlation coefficient
    # Store in vector form (lower triangular elements)
    n_corr = div(k * (k - 1), 2)
    ρ = zeros(n_corr)
    idx = 1
    
    for i in 1:(k-1)
        for j in (i+1):k
            # Extract standardized residuals for pair (i,j)
            y_i = y_std[:, i]
            y_j = y_std[:, j]
            
            # Posterior precision and mean for ρ_ij
            # Likelihood: y_i = ρ_ij * y_j + ε, with ε ~ N(0, 1-ρ²)
            # Using auxiliary variable approach
            
            # Precision (approximate, treating as linear regression)
            prec_y_j = sum(y_j .^ 2)
            post_prec = prior_R0_inv[idx, idx] + prec_y_j
            
            # Mean
            cross_prod = sum(y_i .* y_j)
            post_mean = (prior_R0_inv[idx, idx] * prior_r0[idx] + cross_prod) / post_prec
            
            # Sample from truncated normal in (-1, 1)
            ρ_prop = rand(Normal(post_mean, 1/sqrt(post_prec)))
            
            # Truncate to ensure valid correlation
            ρ[idx] = clamp(ρ_prop, -0.99, 0.99)
            
            idx += 1
        end
    end
    
    # Construct correlation matrix from vector
    R = Matrix{Float64}(I, k, k)
    idx = 1
    for i in 1:(k-1)
        for j in (i+1):k
            R[i, j] = ρ[idx]
            R[j, i] = ρ[idx]
            idx += 1
        end
    end
    
    # Ensure positive definiteness (nearest correlation matrix if needed)
    if !isposdef(R)
        R = nearest_correlation_matrix(R)
    end
    
    return R, ρ
end

"""
Find nearest positive definite correlation matrix
"""
function nearest_correlation_matrix(A)
    k = size(A, 1)
    X = copy(A)
    
    for iter in 1:100
        # Eigenvalue decomposition
        F = eigen(Symmetric(X))
        λ = F.values
        V = F.vectors
        
        # Replace negative eigenvalues with small positive values
        λ_pos = max.(λ, 1e-8)
        
        # Reconstruct matrix
        X = V * Diagonal(λ_pos) * V'
        
        # Project to correlation matrix (unit diagonal)
        D = sqrt.(diag(X))
        X = X ./ (D * D')
        for i in 1:k
            X[i, i] = 1.0
        end
        
        # Check convergence
        if isposdef(X)
            break
        end
    end
    
    return X
end

"""
Sample variance parameters in VAR-SV model using KSC approximation
h: log-volatility states (T x k)
μ_h: mean of log-volatility
φ_h: persistence parameter
σ_η: volatility of volatility
"""
function sample_variance_params(y_star, h, prior_params)
    # y_star: residuals (T x k)
    # h: current log-volatilities (T x k)
    # prior_params: Dict with μ_h_mean, μ_h_var, φ_h_mean, φ_h_var, σ_η_shape, σ_η_scale
    
    T, k = size(y_star)
    
    μ_h = zeros(k)
    φ_h = zeros(k)
    σ_η = zeros(k)
    
    for i in 1:k
        # Sample μ_h (mean of log-volatility)
        # Prior: N(μ_h_mean, μ_h_var)
        
        # Conditional on φ_h, posterior for μ_h
        φ = prior_params[:φ_h_mean]  # Use current value
        h_i = h[:, i]
        
        # First observation
        post_var_μ = 1.0 / (1.0 / prior_params[:μ_h_var] + (1 - φ^2) / prior_params[:σ_η_scale]^2)
        post_mean_μ = post_var_μ * (prior_params[:μ_h_mean] / prior_params[:μ_h_var] + 
                                     (1 - φ) * h_i[1] / prior_params[:σ_η_scale]^2)
        
        μ_h[i] = rand(Normal(post_mean_μ, sqrt(post_var_μ)))
        
        # Sample φ_h (persistence)
        # Prior: N(φ_h_mean, φ_h_var) truncated to (-1, 1)
        
        # Demean log-volatilities
        h_dm = h_i .- μ_h[i]
        
        # Posterior for AR(1) coefficient
        sum_h_lag = sum(h_dm[1:(end-1)] .^ 2)
        sum_cross = sum(h_dm[1:(end-1)] .* h_dm[2:end])
        
        post_var_φ = 1.0 / (1.0 / prior_params[:φ_h_var] + sum_h_lag / prior_params[:σ_η_scale]^2)
        post_mean_φ = post_var_φ * (prior_params[:φ_h_mean] / prior_params[:φ_h_var] + 
                                     sum_cross / prior_params[:σ_η_scale]^2)
        
        # Sample from truncated normal
        φ_prop = rand(Normal(post_mean_φ, sqrt(post_var_φ)))
        φ_h[i] = clamp(φ_prop, -0.99, 0.99)
        
        # Sample σ_η² (innovation variance)
        # Prior: InverseGamma(shape, scale)
        
        # Sum of squared innovations
        h_dm = h_i .- μ_h[i]
        innovations = h_dm[2:end] .- φ_h[i] .* h_dm[1:(end-1)]
        ss_innov = sum(innovations .^ 2)
        
        # Add contribution from first observation
        ss_innov += (1 - φ_h[i]^2) * (h_i[1] - μ_h[i])^2
        
        # Posterior parameters for InverseGamma
        post_shape = prior_params[:σ_η_shape] + T / 2
        post_scale = prior_params[:σ_η_scale] + ss_innov / 2
        
        σ_η²_inv = rand(Gamma(post_shape, 1/post_scale))
        σ_η[i] = sqrt(1.0 / σ_η²_inv)
    end
    
    return μ_h, φ_h, σ_η
end

"""
Kim-Shephard-Chib mixture approximation for log(y²)
Returns mixture component probabilities, means, and variances
"""
function ksc_mixture_params()
    # 7-component mixture approximation
    q = [0.00609, 0.04775, 0.13057, 0.20674, 0.22715, 0.18842, 0.19423]
    m = [1.92677, 1.34744, 0.73504, 0.02266, -0.85173, -1.97278, -3.46788]
    v² = [0.11265, 0.17788, 0.26768, 0.40611, 0.62699, 0.98583, 1.57469]
    
    return q, m, v²
end

"""
Sample mixture indicators for KSC approximation
y²: squared residuals
h: log-volatilities
Returns: s (T vector of mixture component indices 1:7)
"""
function sample_mixture_indicators(y², h)
    T = length(y²)
    s = zeros(Int, T)
    
    q, m, v² = ksc_mixture_params()
    K = length(q)
    
    for t in 1:T
        # Log of squared residual
        y_star_t = log(y²[t] + 1e-10)  # Add small constant for numerical stability
        
        # Compute weights for each mixture component
        log_weights = zeros(K)
        for k in 1:K
            # p(s_t = k | y_t², h_t) ∝ q_k * N(y_star_t | h_t + m_k, v²_k)
            log_weights[k] = log(q[k]) - 0.5 * log(2π * v²[k]) - 
                             0.5 * (y_star_t - h[t] - m[k])^2 / v²[k]
        end
        
        # Normalize weights
        max_log_weight = maximum(log_weights)
        weights = exp.(log_weights .- max_log_weight)
        weights ./= sum(weights)
        
        # Sample component
        s[t] = rand(Categorical(weights))
    end
    
    return s
end

"""
Kalman Filter for log-volatility estimation
y_star: log of squared residuals adjusted by mixture component
μ_h: mean of log-volatility
φ_h: persistence parameter
σ_η: volatility of volatility
s: mixture indicators
Returns: filtered means, filtered variances, prediction means, prediction variances
"""
function kalman_filter_sv(y_star, μ_h, φ_h, σ_η, s)
    T = length(y_star)
    q, m, v² = ksc_mixture_params()
    
    # Storage for filtered and predicted quantities
    h_filt = zeros(T)      # E[h_t | Y_{1:t}]
    P_filt = zeros(T)      # Var[h_t | Y_{1:t}]
    h_pred = zeros(T)      # E[h_t | Y_{1:t-1}]
    P_pred = zeros(T)      # Var[h_t | Y_{1:t-1}]
    
    # Initial conditions (stationary distribution)
    h_pred[1] = μ_h
    P_pred[1] = σ_η^2 / (1 - φ_h^2)
    
    for t in 1:T
        # Observation equation: y_star_t = h_t + m_{s_t} + ε_t
        # where ε_t ~ N(0, v²_{s_t})
        
        # Innovation variance for observation equation
        v_t = v²[s[t]]
        
        # Kalman gain
        K_t = P_pred[t] / (P_pred[t] + v_t)
        
        # Filtered estimate
        innovation = y_star[t] - m[s[t]] - h_pred[t]
        h_filt[t] = h_pred[t] + K_t * innovation
        P_filt[t] = (1 - K_t) * P_pred[t]
        
        # Prediction for next period (if not last)
        if t < T
            h_pred[t+1] = μ_h + φ_h * (h_filt[t] - μ_h)
            P_pred[t+1] = φ_h^2 * P_filt[t] + σ_η^2
        end
    end
    
    return h_filt, P_filt, h_pred, P_pred
end

"""
Carter-Kohn backward smoother for log-volatilities
Samples from p(h_{1:T} | Y, θ, s) using forward-backward algorithm
h_filt: filtered means from Kalman filter
P_filt: filtered variances from Kalman filter
μ_h: mean of log-volatility
φ_h: persistence parameter
σ_η: volatility of volatility
Returns: smoothed draw of h (T vector)
"""
function carter_kohn_smoother(h_filt, P_filt, μ_h, φ_h, σ_η)
    T = length(h_filt)
    h_smooth = zeros(T)
    
    # Sample h_T from filtered distribution
    h_smooth[T] = rand(Normal(h_filt[T], sqrt(P_filt[T])))
    
    # Backward recursion
    for t in (T-1):-1:1
        # Prediction from time t to t+1
        h_pred_tp1 = μ_h + φ_h * (h_filt[t] - μ_h)
        P_pred_tp1 = φ_h^2 * P_filt[t] + σ_η^2
        
        # Smoothing gain
        J_t = φ_h * P_filt[t] / P_pred_tp1
        
        # Smoothed mean and variance
        h_smooth_mean = h_filt[t] + J_t * (h_smooth[t+1] - h_pred_tp1)
        P_smooth = P_filt[t] - J_t^2 * (P_pred_tp1 - P_filt[t] - σ_η^2)
        
        # Ensure positive variance
        P_smooth = max(P_smooth, 1e-8)
        
        # Sample from smoothed distribution
        h_smooth[t] = rand(Normal(h_smooth_mean, sqrt(P_smooth)))
    end
    
    return h_smooth
end

"""
Complete forward-filter backward-sample for log-volatilities
Combines Kalman filter and Carter-Kohn smoother
y_star_i: residuals for variable i (T vector)
μ_h: mean of log-volatility
φ_h: persistence parameter
σ_η: volatility of volatility
s: mixture indicators (T vector)
Returns: sampled log-volatilities (T vector)
"""
function sample_log_volatilities(y_star_i, μ_h, φ_h, σ_η, s)
    # Transform to log of squared residuals
    y_sq = y_star_i.^2
    y_log_sq = log.(y_sq .+ 1e-10)
    
    # Forward pass: Kalman filter
    h_filt, P_filt, h_pred, P_pred = kalman_filter_sv(y_log_sq, μ_h, φ_h, σ_η, s)
    
    # Backward pass: Carter-Kohn smoother (sample)
    h_sample = carter_kohn_smoother(h_filt, P_filt, μ_h, φ_h, σ_η)
    
    return h_sample
end

"""
Sample all log-volatilities for multivariate case
y_star: residuals (T x k)
μ_h: vector of means (k)
φ_h: vector of persistence parameters (k)
σ_η: vector of volatility of volatilities (k)
Returns: h (T x k) matrix of log-volatilities
"""
function sample_all_log_volatilities(y_star, μ_h, φ_h, σ_η)
    T, k = size(y_star)
    h = zeros(T, k)
    
    for i in 1:k
        # Sample mixture indicators for variable i
        y_sq_i = y_star[:, i].^2
        
        # Initialize with previous h if available, otherwise use mean
        h_init = fill(μ_h[i], T)
        s_i = sample_mixture_indicators(y_sq_i, h_init)
        
        # Sample log-volatilities for variable i
        h[:, i] = sample_log_volatilities(y_star[:, i], μ_h[i], φ_h[i], σ_η[i], s_i)
    end
    
    return h
end

# Example usage
function example_usage()
    Random.seed!(123)
    
    # Simulate VAR data
    T = 200  # time periods
    k = 3    # number of variables
    p = 2    # lag order
    
    # Simulate raw data
    y = cumsum(randn(T, k), dims=1) * 0.1
    
    # True VAR coefficients (k*p + 1) x k
    n_coef = k * p + 1
    β_true = [0.3 * Matrix{Float64}(I, k, k); 
              0.2 * Matrix{Float64}(I, k, k); 
              zeros(1, k)]  # Two lags + intercept
    
    # Compute residuals from data and coefficients
    y_star = compute_var_residuals(y, β_true, p)
    println("Residual dimensions: ", size(y_star))
    println("Mean of residuals: ", mean(y_star, dims=1))
    
    # Also get design matrix for other uses
    X, y_eff = construct_var_design_matrix(y, p)
    println("\nDesign matrix dimensions: ", size(X))
    
    # Simulated log-volatilities
    T_eff = size(y_star, 1)
    h = randn(T_eff, k)
    
    println("\n" * "="^60)
    println("TESTING LOG-VOLATILITY SAMPLING")
    println("="^60)
    
    # Initialize variance parameters
    μ_h_vec = zeros(k)
    φ_h_vec = fill(0.95, k)
    σ_η_vec = fill(0.2, k)
    
    # Sample log-volatilities using Kalman filter + Carter-Kohn smoother
    h_sampled = sample_all_log_volatilities(y_star, μ_h_vec, φ_h_vec, σ_η_vec)
    println("\nSampled log-volatility dimensions: ", size(h_sampled))
    println("Mean of sampled h: ", mean(h_sampled, dims=1))
    println("Std of sampled h: ", std(h_sampled, dims=1))
    
    # Test single variable case
    println("\nTesting single variable:")
    y_sq_1 = y_star[:, 1].^2
    h_init = fill(μ_h_vec[1], T_eff)
    s_1 = sample_mixture_indicators(y_sq_1, h_init)
    h_1 = sample_log_volatilities(y_star[:, 1], μ_h_vec[1], φ_h_vec[1], σ_η_vec[1], s_1)
    println("Mixture indicator counts: ", [sum(s_1 .== i) for i in 1:7])
    println("Mean of h_1: ", mean(h_1))
    println("Autocorrelation of h_1 (lag 1): ", cor(h_1[1:end-1], h_1[2:end]))
    
    println("\n" * "="^60)
    println("TESTING CORRELATION PARAMETERS")
    println("="^60)
    
    # Prior parameters for correlations
    n_corr = div(k * (k - 1), 2)
    prior_R0_inv = Matrix{Float64}(I, n_corr, n_corr) * 0.1
    prior_r0 = zeros(n_corr)
    
    # Sample correlation parameters
    R, ρ = sample_correlation_params(y_star, h, prior_R0_inv, prior_r0, k, k)
    println("Sampled correlation matrix:")
    println(R)
    
    # Prior parameters for variance parameters
    prior_params = Dict(
        :μ_h_mean => 0.0,
        :μ_h_var => 10.0,
        :φ_h_mean => 0.9,
        :φ_h_var => 0.1,
        :σ_η_shape => 2.5,
        :σ_η_scale => 0.1
    )
    
    # Sample variance parameters
    μ_h, φ_h, σ_η = sample_variance_params(y_star, h, prior_params)s
    println("\nSampled variance parameters:")
    println("μ_h: ", μ_h)
    println("φ_h: ", φ_h)
    println("σ_η: ", σ_η)
    
    # KSC mixture approximation example
    y² = y_star[:, 1].^2
    s = sample_mixture_indicators(y², h[:, 1])
    println("\nMixture indicator counts: ", [sum(s .== i) for i in 1:7])
end

# Run example
example_usage()s