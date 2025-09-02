using LinearAlgebra
using Distributions
using Random

using LinearAlgebra
using Distributions
using Random



"""
Kalman Filter implementation
Returns filtered states, covariances, predicted states, and predicted covariances
"""
function kalman_filter(model::StateSpaceModel, observations::Matrix{Float64})
    n_time_steps, n_obs = size(observations)
    n_states = size(model.T, 1)
    
    # Storage for results
    state_filtered = zeros(n_time_steps, n_states)
    covariance_filtered = zeros(n_time_steps, n_states, n_states)
    state_predicted = zeros(n_time_steps, n_states)
    covariance_predicted = zeros(n_time_steps, n_states, n_states)
    log_likelihood = 0.0
    
    # Initialize
    state_current = model.initial_state_mean
    covariance_current = model.initial_state_covariance
    
    for t in 1:n_time_steps
        # Prediction step
        if t == 1
            state_predicted_t = model.T * state_current
            covariance_predicted_t = model.T * covariance_current * model.T' + model.R * model.Q * model.R'
        else
            state_predicted_t = model.T * state_filtered[t-1, :]
            covariance_predicted_t = model.T * reshape(covariance_filtered[t-1, :, :], n_states, n_states) * model.T' + model.R * model.Q * model.R'
        end
        
        state_predicted[t, :] = state_predicted_t
        covariance_predicted[t, :, :] = covariance_predicted_t
        
        # Update step (if observation is available)
        y_t = observations[t, :]
        if !any(isnan.(y_t))
            # Innovation
            innovation = y_t - model.Z * state_predicted_t
            innovation_covariance = model.Z * covariance_predicted_t * model.Z' + model.H
            
            # Kalman gain
            kalman_gain = covariance_predicted_t * model.Z' * inv(model.Z * covariance_predicted_t * model.Z' + model.H)
            
            # Filtered state and covariance
            state_filtered[t, :] = state_predicted_t + kalman_gain * innovation
            covariance_filtered[t, :, :] = covariance_predicted_t - kalman_gain * model.Z * covariance_predicted_t
            
            # Log-likelihood contribution
            log_likelihood += -0.5 * (log(det(innovation_covariance)) + innovation' * inv(innovation_covariance) * innovation)
        else
            # No observation available
            state_filtered[t, :] = state_predicted_t
            covariance_filtered[t, :, :] = covariance_predicted_t
        end
    end
    
    return state_filtered, covariance_filtered, state_predicted, covariance_predicted, log_likelihood
end

"""
Carter-Kohn Algorithm for sampling smoothed states
This algorithm samples from the joint posterior distribution of all states
given all observations using backward simulation
"""
function carter_kohn_sampler2(model::StateSpaceModel, observations::Matrix{Float64}; n_samples::Int=1000)
    n_time_steps, n_obs = size(observations)
    n_states = size(model.T, 1)
    
    # Run Kalman filter forward pass
    state_filtered, covariance_filtered, state_predicted, covariance_predicted, _ = 
        kalman_filter(model, observations)
    
    # Storage for sampled states
    state_smoothed_samples = zeros(n_samples, n_time_steps, n_states)
    
    for sample_idx in 1:n_samples
        state_smoothed_current = zeros(n_time_steps, n_states)
        
        # Sample final state from filtered distribution at T
        final_state_mean = state_filtered[end, :]
        final_state_covariance = covariance_filtered[end, :, :]
        state_smoothed_current[end, :] = rand(MvNormal(final_state_mean, Hermitian(final_state_covariance)))
        
        # Backward pass: sample states from T-1 down to 1
        for t in (n_time_steps-1):-1:1
            # Get filtered estimates at time t
            state_filtered_t = state_filtered[t, :]
            covariance_filtered_t = covariance_filtered[t, :, :]
            
            # Get predicted estimates at time t+1
            state_predicted_t_plus_1 = state_predicted[t+1, :]
            covariance_predicted_t_plus_1 = covariance_predicted[t+1, :, :]
            
            # Compute smoothing gain matrix
            smoothing_gain = covariance_filtered_t * model.T' * inv(covariance_predicted_t_plus_1)
            
            # Conditional mean and covariance for state at time t given state at t+1
            state_smoothed_mean = state_filtered_t + 
                smoothing_gain * (state_smoothed_current[t+1, :] - state_predicted_t_plus_1)
            
            covariance_smoothed = covariance_filtered_t - 
                smoothing_gain * model.T*covariance_filtered_t
            
            # Ensure covariance is positive definite
            covariance_smoothed = (covariance_smoothed + covariance_smoothed') / 2
            covariance_smoothed += 1e-10 * I
            
            # Sample state at time t
            state_smoothed_current[t, :] = rand(MvNormal(state_smoothed_mean, covariance_smoothed))
        end
        
        state_smoothed_samples[sample_idx, :, :] = state_smoothed_current
    end
    
    return state_smoothed_samples
end

"""
Carter-Kohn Algorithm for sampling smoothed states
This algorithm samples from the joint posterior distribution of all states
given all observations using backward simulation
"""
function carter_kohn_sampler(model::StateSpaceModel, observations::Matrix{Float64})
    n_time_steps, n_obs = size(observations)
    n_states = size(model.T, 1)
    
    # Run Kalman filter forward pass
    state_filtered, covariance_filtered, state_predicted, covariance_predicted, _ = 
        kalman_filter(model, observations)
                
    state_smoothed_current = zeros(n_time_steps, n_states)
        
    # Sample final state from filtered distribution at T
    final_state_mean = state_filtered[end, :]
    final_state_covariance = covariance_filtered[end, :, :] + I(4) * eps()
    state_smoothed_current[end, :] = rand(MvNormal(final_state_mean, Hermitian(final_state_covariance)))
        
    # Backward pass: sample states from T-1 down to 1
    for t in (n_time_steps-1):-1:1
        # Get filtered estimates at time t
        state_filtered_t = state_filtered[t, :]
        covariance_filtered_t = covariance_filtered[t, :, :]
           
        # Get predicted estimates at time t+1
        state_predicted_t_plus_1 = state_predicted[t+1, :]
        covariance_predicted_t_plus_1 = covariance_predicted[t+1, :, :]
            
        # Compute smoothing gain matrix
        smoothing_gain = covariance_filtered_t * model.T' * inv(covariance_predicted_t_plus_1)
            
        # Conditional mean and covariance for state at time t given state at t+1
        state_smoothed_mean = state_filtered_t + 
            smoothing_gain * (state_smoothed_current[t+1, :] - state_predicted_t_plus_1)
            
        covariance_smoothed = covariance_filtered_t - 
            smoothing_gain * model.T*covariance_filtered_t
            
        # Ensure covariance is positive definite
        covariance_smoothed = (covariance_smoothed + covariance_smoothed') / 2
        covariance_smoothed += 1e-10 * I
            
        # Sample state at time t
        state_smoothed_current[t, :] = rand(MvNormal(state_smoothed_mean, covariance_smoothed))
    end        
     
    return state_smoothed_current

end

"""
Compute posterior mean and credible intervals from Carter-Kohn samples
"""
function compute_posterior_statistics(state_smoothed_samples::Array{Float64,3}; credible_level::Float64=0.95)
    n_samples, n_time_steps, n_states = size(state_smoothed_samples)
    
    # Posterior means
    state_smoothed_mean = mean(state_smoothed_samples, dims=1)[1, :, :]
    
    # Credible intervals
    alpha = (1 - credible_level) / 2
    lower_quantile = alpha
    upper_quantile = 1 - alpha
    
    state_smoothed_lower = zeros(n_time_steps, n_states)
    state_smoothed_upper = zeros(n_time_steps, n_states)
    
    for t in 1:n_time_steps
        for s in 1:n_states
            state_samples_ts = state_smoothed_samples[:, t, s]
            state_smoothed_lower[t, s] = quantile(state_samples_ts, lower_quantile)
            state_smoothed_upper[t, s] = quantile(state_samples_ts, upper_quantile)
        end
    end
    
    return state_smoothed_mean, state_smoothed_lower, state_smoothed_upper
end

# Example usage
function example_usage()
    # Define a simple AR(1) model with observation noise
    # x_t = 0.8 * x_{t-1} + u_t,  u_t ~ N(0, 1)
    # y_t = x_t + v_t,             v_t ~ N(0, 0.5)
    
    n_time_steps = 100
    n_states = 1
    n_obs = 1
    
    # Model parameters
    F = [0.8]  # AR coefficient
    G = [1.0]  # State noise coefficient
    H = [1.0]  # Observation matrix
    Q = [1.0]  # State noise variance
    R = [0.5]  # Observation noise variance
    initial_state_mean = [0.0]
    initial_state_covariance = [2.0]
    
    model = StateSpaceModel(F, G, H, Q, R, initial_state_mean, initial_state_covariance)
    
    # Simulate some data
    Random.seed!(123)
    true_states = zeros(n_time_steps)
    observations_data = zeros(n_time_steps, n_obs)
    
    true_states[1] = rand(Normal(initial_state_mean[1], sqrt(initial_state_covariance[1, 1])))
    observations_data[1, 1] = true_states[1] + rand(Normal(0, sqrt(R[1])))
    
    for t in 2:n_time_steps
        true_states[t] = F[1] * true_states[t-1] + rand(Normal(0, sqrt(Q[1])))
        observations_data[t, 1] = true_states[t] + rand(Normal(0, sqrt(R[1])))
    end
    
    # Run Carter-Kohn sampler
    println("Running Carter-Kohn sampler...")
    state_smoothed_samples = carter_kohn_sampler(model, observations_data, n_samples=1000)
    
    # Compute posterior statistics
    state_smoothed_mean, state_smoothed_lower, state_smoothed_upper = 
        compute_posterior_statistics(state_smoothed_samples)
    
    println("Carter-Kohn sampling completed!")
    println("Posterior mean for first 5 time points: ", state_smoothed_mean[1:5, 1])
    println("True states for first 5 time points: ", true_states[1:5])
    
    return state_smoothed_samples, state_smoothed_mean, true_states, observations_data
end