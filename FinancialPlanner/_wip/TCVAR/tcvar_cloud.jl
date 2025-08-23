
using LinearAlgebra, Random, Distributions, Statistics, Plots

"""
    StateSpaceTrendCycleGibbs

Gibbs sampler for state space model with trend-cycle decomposition

Observation equation: y_t = Z * α_t + ε_t
State equation: α_t = T * α_{t-1} + R * η_t

State vector: α_t = [trend_interest, trend_inflation, cycle_interest, cycle_inflation, cycle_interest_{t-1}, cycle_inflation_{t-1}]
"""
mutable struct StateSpaceTrendCycleGibbs
    # Data
    y_obs::Matrix{Float64}
    T::Int
    n_vars::Int
    n_lags::Int
    n_cycle_vars::Int
    n_states::Int
    
    # System matrices
    Z::Matrix{Float64}  # observation matrix
    T_mat::Matrix{Float64}  # state transition matrix
    R::Matrix{Float64}  # selection matrix
    Q::Matrix{Float64}  # state innovation covariance
    
    # Parameters
    var_coeffs::Matrix{Float64}  # VAR coefficients
    σ_trend::Matrix{Float64}     # trend innovation covariance
    σ_cycle::Matrix{Float64}     # cycle innovation covariance
    σ_obs::Matrix{Float64}       # observation error covariance
    
    # Initial conditions
    α_1::Vector{Float64}
    P_1::Matrix{Float64}
    
    # Minnesota prior parameters
    λ_1::Float64  # overall tightness
    λ_2::Float64  # cross-variable shrinkage
    λ_3::Float64  # lag decay
    
    function StateSpaceTrendCycleGibbs(observations::Matrix{Float64}, n_lags::Int = 2)
        T, n_vars = size(observations)
        n_cycle_vars = n_vars * n_lags
        n_states = 2 + n_cycle_vars  # trend(2) + cycle(4 for 2 lags)
        
        # Initialize system matrices
        Z, T_mat, R = setup_system_matrices(n_vars, n_states, n_lags)
        
        # Initialize parameters
        var_coeffs = randn(n_vars, n_vars * n_lags) * 0.1
        σ_trend = Matrix{Float64}(I, n_vars, n_vars) * 0.01
        σ_cycle = Matrix{Float64}(I, n_vars, n_vars) * 0.1
        σ_obs = Matrix{Float64}(I, n_vars, n_vars) * 0.05
        
        Q = zeros(2 + n_vars, 2 + n_vars)
        Q[1:2, 1:2] = σ_trend
        Q[3:2+n_vars, 3:2+n_vars] = σ_cycle
        
        # Initial conditions
        α_1 = zeros(n_states)
        P_1 = Matrix{Float64}(I, n_states, n_states) * 10.0
        
        # Minnesota prior parameters
        λ_1, λ_2, λ_3 = 0.2, 1.0, 1.0
        
        new(observations, T, n_vars, n_lags, n_cycle_vars, n_states,
            Z, T_mat, R, Q, var_coeffs, σ_trend, σ_cycle, σ_obs,
            α_1, P_1, λ_1, λ_2, λ_3)
    end
end

function setup_system_matrices(n_vars::Int, n_states::Int, n_lags::Int)
    """Setup time-invariant system matrices"""
    
    # Observation matrix Z: maps states to observations
    # y_t = trend + cycle (first component of cycle)
    Z = zeros(n_vars, n_states)
    Z[1, 1] = 1.0  # interest trend
    Z[1, 2] = 1.0 # interest rate = inflation trend + interesrt rate trend
    Z[2, 2] = 1.0  # inflation trend
    Z[1, 3] = 1.0  # interest cycle
    Z[2, 4] = 1.0  # inflation cycle
    
    # State transition matrix T
    T_mat = zeros(n_states, n_states)
    
    # Trend follows random walk: trend_t = trend_{t-1} + η_trend_t
    T_mat[1, 1] = 1.0  # interest trend
    T_mat[2, 2] = 1.0  # inflation trend
    
    # Selection matrix R: maps innovations to states
    R = zeros(n_states, 2 + n_vars)  # trend innovations + cycle innovations
    R[1, 1] = 1.0  # trend interest innovation
    R[2, 2] = 1.0  # trend inflation innovation
    R[3, 3] = 1.0  # cycle interest innovation
    R[4, 4] = 1.0  # cycle inflation innovation
    
    return Z, T_mat, R
end

function update_transition_matrix!(model::StateSpaceTrendCycleGibbs)
    """Update state transition matrix with current VAR coefficients"""
    
    # Reset cycle part of transition matrix
    cycle_start = 3
    cycle_end = cycle_start + model.n_vars - 1
    
    # First lag coefficients
    model.T_mat[cycle_start:cycle_end, cycle_start:cycle_start+model.n_vars-1] = 
        model.var_coeffs[:, 1:model.n_vars]
    
    # Second lag coefficients (if n_lags > 1)
    if model.n_lags > 1
        model.T_mat[cycle_start:cycle_end, cycle_end+1:cycle_end+model.n_vars] = 
            model.var_coeffs[:, model.n_vars+1:2*model.n_vars]
    end
    
    # Companion form: shift lagged variables
    if model.n_lags > 1
        for i in 1:(model.n_lags-1)
            start_row = cycle_end + 1 + (i-1) * model.n_vars
            end_row = start_row + model.n_vars - 1
            start_col = cycle_start + (i-1) * model.n_vars
            end_col = start_col + model.n_vars - 1
            model.T_mat[start_row:end_row, start_col:end_col] = Matrix{Float64}(I, model.n_vars, model.n_vars)
        end
    end
end

function kalman_filter(model::StateSpaceTrendCycleGibbs)
    """
    Run Kalman filter to get filtered states and likelihood
    
    Returns:
    - state_filtered: filtered state estimates (T × n_states)
    - P_filtered: filtered state covariances (T × n_states × n_states)
    - state_predicted: predicted state estimates (T × n_states)
    - P_predicted: predicted state covariances (T × n_states × n_states)
    - log_likelihood: log likelihood of observations
    """
    
    state_filtered = zeros(model.T, model.n_states)
    P_filtered = zeros(model.T, model.n_states, model.n_states)
    state_predicted = zeros(model.T, model.n_states)
    P_predicted = zeros(model.T, model.n_states, model.n_states)
    
    log_likelihood = 0.0
    
    # Initial conditions
    state_pred = copy(model.α_1)
    P_pred = copy(model.P_1)
    
    for t in 1:model.T
        # Prediction step
        if t > 1
            state_pred = model.T_mat * state_filtered[t-1, :]
            P_pred = model.T_mat * P_filtered[t-1, :, :] * model.T_mat' + 
                     model.R * model.Q * model.R'
        end
        
        state_predicted[t, :] = state_pred
        P_predicted[t, :, :] = P_pred
        
        # Update step
        y_pred = model.Z * state_pred
        forecast_error = model.y_obs[t, :] - y_pred
        
        S = model.Z * P_pred * model.Z' + model.σ_obs  # forecast error covariance
        
        try
            K = P_pred * model.Z' * inv(S)  # Kalman gain
            
            state_filtered[t, :] = state_pred + K * forecast_error
            P_filtered[t, :, :] = P_pred - K * model.Z * P_pred
            
            # Update log likelihood
            log_likelihood += -0.5 * (logdet(S) + forecast_error' * inv(S) * forecast_error)
        catch
            # Penalty for numerical issues
            log_likelihood += -1e6
            state_filtered[t, :] = state_pred
            P_filtered[t, :, :] = P_pred
        end
    end
    
    return state_filtered, P_filtered, state_predicted, P_predicted, log_likelihood
end

function carter_kohn_smoother(model::StateSpaceTrendCycleGibbs, 
                             state_filtered::Matrix{Float64}, 
                             P_filtered::Array{Float64,3}, 
                             state_predicted::Matrix{Float64}, 
                             P_predicted::Array{Float64,3})
    """
    Carter-Kohn backward sampling smoother
    
    Returns:
    - state_smoothed: smoothed state draws (T × n_states)
    """
    
    state_smoothed = zeros(model.T, model.n_states)
    
    # Initialize with last filtered state
    state_smoothed[end, :] = rand(MvNormal(state_filtered[end, :], 
                                          P_filtered[end, :, :] + 1e-8 * I))
    
    # Backward pass
    for t in (model.T-1):-1:1
        # Smoothing equations
        try
            A = P_filtered[t, :, :] * model.T_mat' * inv(P_predicted[t+1, :, :])
            
            mean_smooth = state_filtered[t, :] + A * (state_smoothed[t+1, :] - state_predicted[t+1, :])
            var_smooth = P_filtered[t, :, :] - A * P_predicted[t+1, :, :] * A'
            
            # Ensure positive definite covariance
            var_smooth = var_smooth + 1e-8 * I
            
            state_smoothed[t, :] = rand(MvNormal(mean_smooth, Hermitian(var_smooth)))
        catch
            # Fallback to filtered state if numerical issues
            state_smoothed[t, :] = state_filtered[t, :]
        end
    end
    
    return state_smoothed
end

function sample_states(model::StateSpaceTrendCycleGibbs)
    """Sample states using Kalman filter and Carter-Kohn smoother"""
    
    # Update transition matrix with current VAR coefficients
    update_transition_matrix!(model)
    
    # Run Kalman filter
    state_filtered, P_filtered, state_predicted, P_predicted, log_likelihood = kalman_filter(model)
    
    # Sample smoothed states
    state_smoothed = carter_kohn_smoother(model, state_filtered, P_filtered, 
                                        state_predicted, P_predicted)
    
    return state_smoothed, log_likelihood
end

function sample_trend_covariance!(model::StateSpaceTrendCycleGibbs, state_smoothed::Matrix{Float64})
    """Sample trend innovation covariance matrix"""
    
    # Extract trend components
    trend_states = state_smoothed[:, 1:2]  # First two states are trends
    
    # Compute trend innovations
    trend_innovations = diff(trend_states, dims=1)
    
    # Posterior parameters for inverse Wishart
    ν_post = 2 + model.T - 1  # degrees of freedom
    S_post = Matrix{Float64}(I, 2, 2) * 0.01 + trend_innovations' * trend_innovations
    
    # Sample from inverse Wishart
    model.σ_trend = rand(InverseWishart(ν_post, S_post))
    
    # Update Q matrix
    model.Q[1:2, 1:2] = model.σ_trend
end

function sample_cycle_covariance!(model::StateSpaceTrendCycleGibbs, state_smoothed::Matrix{Float64})
    """Sample cycle innovation covariance matrix"""
    
    # Extract cycle components
    cycle_states = state_smoothed[:, 3:4]  # First two cycle states
    
    # Compute cycle innovations (residuals from VAR)
    cycle_innovations = Matrix{Float64}(undef, 0, 2)
    
    for t in 2:model.T
        # Construct lagged cycle variables
        lags = Float64[]
        for lag in 1:model.n_lags
            if t - lag >= 1
                append!(lags, state_smoothed[t - lag, 3:4])
            else
                append!(lags, zeros(2))
            end
        end
        
        X = reshape(lags, :)
        innovation = cycle_states[t, :] - model.var_coeffs * X
        cycle_innovations = vcat(cycle_innovations, innovation')
    end
    
    # Posterior parameters
    ν_post = 2 + size(cycle_innovations, 1)
    S_post = Matrix{Float64}(I, 2, 2) * 0.1 + cycle_innovations' * cycle_innovations
    
    # Sample from inverse Wishart
    model.σ_cycle = rand(InverseWishart(ν_post, S_post))
    
    # Update Q matrix
    model.Q[3:4, 3:4] = model.σ_cycle
end

function minnesota_prior_covariance(model::StateSpaceTrendCycleGibbs)
    """Construct Minnesota prior covariance matrix for VAR coefficients"""
    
    n_coeff = model.n_vars * model.n_vars * model.n_lags
    V_prior = zeros(n_coeff, n_coeff)
    
    idx = 1
    for i in 1:model.n_vars  # equation
        for j in 1:model.n_vars  # variable
            for lag in 1:model.n_lags  # lag
                if i == j  # own lag
                    V_prior[idx, idx] = (model.λ_1 / lag^model.λ_3)^2
                else  # cross variable
                    V_prior[idx, idx] = (model.λ_1 * model.λ_2 / lag^model.λ_3)^2
                end
                idx += 1
            end
        end
    end
    
    return V_prior
end

function sample_var_coefficients!(model::StateSpaceTrendCycleGibbs, state_smoothed::Matrix{Float64})
    """Sample VAR coefficients with Minnesota priors"""
    
    # Extract cycle components
    cycle_states = state_smoothed[:, 3:4]
    
    # Construct VAR regression matrices
    Y = Matrix{Float64}(undef, 0, 2)  # dependent variables
    X = Matrix{Float64}(undef, 0, model.n_vars * model.n_lags)  # lagged variables
    
    for t in (model.n_lags+1):model.T
        Y = vcat(Y, cycle_states[t, :]')
        
        # Construct lags
        x_t = Float64[]
        for lag in 1:model.n_lags
            append!(x_t, cycle_states[t - lag, :])
        end
        X = vcat(X, x_t')
    end
    
    # Minnesota prior
    V_prior = minnesota_prior_covariance(model)
    μ_prior = zeros(model.n_vars * model.n_vars * model.n_lags)
    
    # Posterior parameters
    V_post_inv = inv(V_prior) + kron(inv(model.σ_cycle), X' * X)
    V_post = inv(V_post_inv)
    
    μ_post = V_post * (inv(V_prior) * μ_prior + 
                      kron(inv(model.σ_cycle), X') * vec(Y))
    
    # Sample coefficients
    coeff_vec = rand(MvNormal(μ_post, Hermitian(V_post)))
    model.var_coeffs = reshape(coeff_vec, model.n_vars, model.n_vars * model.n_lags)
end

function gibbs_sample(model::StateSpaceTrendCycleGibbs; 
                     n_draws::Int = 1000, 
                     burn_in::Int = 200, 
                     thin::Int = 1)
    """
    Run Gibbs sampler
    
    Parameters:
    - n_draws: number of MCMC draws
    - burn_in: number of burn-in draws
    - thin: thinning interval
    
    Returns:
    - results: NamedTuple containing sampled parameters and states
    """
    
    total_draws = burn_in + n_draws * thin
    
    # Storage for results
    states_draws = Vector{Matrix{Float64}}()
    var_coeff_draws = Vector{Matrix{Float64}}()
    σ_trend_draws = Vector{Matrix{Float64}}()
    σ_cycle_draws = Vector{Matrix{Float64}}()
    log_likelihood_draws = Vector{Float64}()
    
    println("Running Gibbs sampler: $total_draws total draws, $burn_in burn-in, thin=$thin")
    
    for draw in 1:total_draws
        if draw % 100 == 0
            println("Draw $draw/$total_draws")
        end
        
        # Step 1: Sample states
        state_smoothed, log_likelihood = sample_states(model)
        
        # Step 2: Sample trend covariance
        sample_trend_covariance!(model, state_smoothed)
        
        # Step 3: Sample cycle covariance
        sample_cycle_covariance!(model, state_smoothed)
        
        # Step 4: Sample VAR coefficients
        sample_var_coefficients!(model, state_smoothed)
        
        # Store results after burn-in and according to thinning
        if draw > burn_in && (draw - burn_in) % thin == 0
            push!(states_draws, copy(state_smoothed))
            push!(var_coeff_draws, copy(model.var_coeffs))
            push!(σ_trend_draws, copy(model.σ_trend))
            push!(σ_cycle_draws, copy(model.σ_cycle))
            push!(log_likelihood_draws, log_likelihood)
        end
    end
    
    return (states_draws = states_draws,
            var_coeff_draws = var_coeff_draws,
            σ_trend_draws = σ_trend_draws,
            σ_cycle_draws = σ_cycle_draws,
            log_likelihood_draws = log_likelihood_draws,
            observations = model.y_obs)
end

function plot_results(model::StateSpaceTrendCycleGibbs, results; save_path::String = "")
    """Plot estimation results"""
    
    states_draws = cat(results.states_draws..., dims=3)
    
    # Compute posterior means
    state_mean = mean(states_draws, dims=3)[:, :, 1]
    state_lower = [quantile(states_draws[t, s, :], 0.05) for t in 1:model.T, s in 1:model.n_states]
    state_upper = [quantile(states_draws[t, s, :], 0.95) for t in 1:model.T, s in 1:model.n_states]
    
    time_axis = 1:model.T
    
    # Create plots
    p1 = plot(time_axis, state_mean[:, 1], 
             ribbon=(state_mean[:, 1] - state_lower[:, 1], state_upper[:, 1] - state_mean[:, 1]),
             label="Trend", linewidth=2, fillalpha=0.3, color=:blue,
             title="Interest Rate Trend")
    scatter!(p1, time_axis, model.y_obs[:, 1], label="Observed", alpha=0.5, color=:red, ms=3)
    
    p2 = plot(time_axis, state_mean[:, 2], 
             ribbon=(state_mean[:, 2] - state_lower[:, 2], state_upper[:, 2] - state_mean[:, 2]),
             label="Trend", linewidth=2, fillalpha=0.3, color=:green,
             title="Inflation Trend")
    scatter!(p2, time_axis, model.y_obs[:, 2], label="Observed", alpha=0.5, color=:red, ms=3)
    
    p3 = plot(time_axis, state_mean[:, 3], 
             ribbon=(state_mean[:, 3] - state_lower[:, 3], state_upper[:, 3] - state_mean[:, 3]),
             label="Cycle", linewidth=2, fillalpha=0.3, color=:blue,
             title="Interest Rate Cycle")
    hline!(p3, [0], color=:black, linestyle=:dash, alpha=0.5, label="")
    
    p4 = plot(time_axis, state_mean[:, 4], 
             ribbon=(state_mean[:, 4] - state_lower[:, 4], state_upper[:, 4] - state_mean[:, 4]),
             label="Cycle", linewidth=2, fillalpha=0.3, color=:green,
             title="Inflation Cycle")
    hline!(p4, [0], color=:black, linestyle=:dash, alpha=0.5, label="")
    
    final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))
    
    if !isempty(save_path)
        savefig(final_plot, save_path)
    end
    
    return final_plot
end

# Example usage
function run_example()
    # Generate synthetic data
    Random.seed!(42)
    T = 100
    
    # True trend (random walks)
    trend_interest = cumsum(randn(T) * 0.1)
    trend_inflation = cumsum(randn(T) * 0.1)
    
    # True cycle (AR(2) process)
    cycle_interest = zeros(T)
    cycle_inflation = zeros(T)
    for t in 3:T
        cycle_interest[t] = 0.7*cycle_interest[t-1] - 0.2*cycle_interest[t-2] + 
                           0.1*cycle_inflation[t-1] + randn() * 0.2
        cycle_inflation[t] = 0.1*cycle_interest[t-1] + 0.6*cycle_inflation[t-1] - 
                            0.1*cycle_inflation[t-2] + randn() * 0.2
    end
    
    # Observed data = trend + cycle + noise
    observations = hcat(
        trend_interest + cycle_interest + randn(T) * 0.05,
        trend_inflation + cycle_inflation + randn(T) * 0.05
    )
    
    # Run Gibbs sampler
    model = StateSpaceTrendCycleGibbs(observations, 2)
    results = gibbs_sample(model, n_draws=500, burn_in=100, thin=2)
    
    # Print convergence diagnostics
    println("\nConvergence Diagnostics:")
    println("Mean log-likelihood: $(mean(results.log_likelihood_draws))")
    println("VAR coefficient means:")
    display(mean(results.var_coeff_draws))
    
    # Plot results
    plot_results(model, results)
end

# Run example if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_example()
end